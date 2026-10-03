//! The durable event journal (A4; RUST_RUNTIME_ARCHITECTURE.md §5–6).
//!
//! An ordered record of evidence and decisions, separate from presentation
//! frames: one line per committed tick, appended and flushed to stable storage
//! before the run advances, so a crash loses at most the tick in flight. Each
//! line carries its own checksum and the identity of the run that wrote it.
//!
//! Line format: `<16 hex FNV-1a of the JSON> <JSON record>\n`. A last line with
//! no newline, or whose checksum fails, is a **torn tail** -- a write the crash
//! interrupted -- and is discarded on recovery. A bad line *followed* by good
//! ones is **corruption**, which recovery refuses rather than skips.

use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, BufReader, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::config::fnv1a;

/// One committed tick's evidence and decisions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JournalRecord {
    /// Position in the journal: 0, 1, 2, ... with no gaps.
    pub seq: u64,
    /// The run's identity (its immutable configuration's hash): a record
    /// from another run is not this run's, whatever its position.
    pub run: u64,
    pub epoch: u64,
    pub tick: u64,
    pub timestamp_s: f64,
    pub body: serde_json::Value,
}

impl JournalRecord {
    fn line(&self) -> String {
        let json = serde_json::to_string(self).expect("a journal record serialises");
        format!("{:016x} {json}\n", fnv1a(json.as_bytes()))
    }

    fn parse(line: &str) -> Option<Self> {
        let (check, json) = line.split_once(' ')?;
        let check = u64::from_str_radix(check, 16).ok()?;
        (check == fnv1a(json.as_bytes()))
            .then(|| serde_json::from_str(json).ok())
            .flatten()
    }
}

/// What a scan of a journal file found.
#[derive(Debug, Clone, PartialEq)]
pub struct JournalScan {
    /// The intact records, in order.
    pub records: Vec<JournalRecord>,
    /// Byte length of the intact prefix: where an append resumes.
    pub intact_bytes: u64,
    /// A final line was incomplete or failed its checksum: an interrupted
    /// write, discarded.
    pub torn_tail: bool,
}

/// Why a journal cannot be used.
#[derive(Debug)]
pub enum JournalError {
    Io(io::Error),
    /// A damaged line with intact lines after it: not an interrupted write.
    Corrupt {
        line: u64,
    },
    /// A record out of sequence, or from another run.
    Foreign {
        line: u64,
        detail: String,
    },
}

impl std::fmt::Display for JournalError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JournalError::Io(error) => write!(formatter, "journal I/O: {error}"),
            JournalError::Corrupt { line } => write!(
                formatter,
                "journal line {line} is damaged and intact lines follow it: corrupt, not torn"
            ),
            JournalError::Foreign { line, detail } => {
                write!(formatter, "journal line {line} is not this run's: {detail}")
            }
        }
    }
}

impl std::error::Error for JournalError {}

impl From<io::Error> for JournalError {
    fn from(error: io::Error) -> Self {
        JournalError::Io(error)
    }
}

/// Read a journal written by run `run`.
pub fn scan(path: &Path, run: u64) -> Result<JournalScan, JournalError> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut records = Vec::new();
    let mut intact_bytes = 0u64;
    let mut damaged_at: Option<u64> = None;
    let mut line = String::new();
    let mut number = 0u64;
    loop {
        line.clear();
        let read = reader.read_line(&mut line)?;
        if read == 0 {
            break;
        }
        number += 1;
        let complete = line.ends_with('\n');
        let parsed = complete
            .then(|| JournalRecord::parse(line.trim_end_matches('\n')))
            .flatten();
        match (parsed, damaged_at) {
            (Some(_), Some(bad)) => return Err(JournalError::Corrupt { line: bad }),
            (Some(record), None) => {
                if record.run != run {
                    return Err(JournalError::Foreign {
                        line: number,
                        detail: format!("written by run {:#x}, this is {run:#x}", record.run),
                    });
                }
                if record.seq != records.len() as u64 {
                    return Err(JournalError::Foreign {
                        line: number,
                        detail: format!("sequence {} where {} belongs", record.seq, records.len()),
                    });
                }
                intact_bytes += read as u64;
                records.push(record);
            }
            (None, _) => {
                damaged_at.get_or_insert(number);
            }
        }
    }
    Ok(JournalScan {
        records,
        intact_bytes,
        torn_tail: damaged_at.is_some(),
    })
}

/// Appends records durably: each is on stable storage before
/// [`JournalWriter::append`] returns.
#[derive(Debug)]
pub struct JournalWriter {
    file: File,
    run: u64,
    next_seq: u64,
}

impl JournalWriter {
    /// A new, empty journal. Refuses to replace an existing one.
    pub fn create(path: &Path, run: u64) -> io::Result<Self> {
        let file = OpenOptions::new().write(true).create_new(true).open(path)?;
        Ok(Self {
            file,
            run,
            next_seq: 0,
        })
    }

    /// Continue a journal after its first `keep` records, dropping anything
    /// after them (a torn tail, or ticks after the checkpoint a recovery
    /// resumes from).
    pub fn resume(path: &Path, run: u64, keep: u64) -> Result<Self, JournalError> {
        let found = scan(path, run)?;
        if keep > found.records.len() as u64 {
            return Err(JournalError::Foreign {
                line: found.records.len() as u64 + 1,
                detail: format!(
                    "the checkpoint needs {keep} records, the journal has {}",
                    found.records.len()
                ),
            });
        }
        let bytes: u64 = found.records[..keep as usize]
            .iter()
            .map(|record| record.line().len() as u64)
            .sum();
        let file = OpenOptions::new().write(true).open(path)?;
        file.set_len(bytes)?;
        file.sync_all()?;
        let mut file = OpenOptions::new().append(true).open(path)?;
        file.flush()?;
        Ok(Self {
            file,
            run,
            next_seq: keep,
        })
    }

    /// Records written so far: the durable cursor.
    pub fn cursor(&self) -> u64 {
        self.next_seq
    }

    /// Write one record and wait for it to reach stable storage. An error
    /// leaves the cursor where it was: the record is not durable, and the
    /// caller must not advance past it.
    pub fn append(
        &mut self,
        epoch: u64,
        tick: u64,
        timestamp_s: f64,
        body: serde_json::Value,
    ) -> io::Result<u64> {
        let record = JournalRecord {
            seq: self.next_seq,
            run: self.run,
            epoch,
            tick,
            timestamp_s,
            body,
        };
        self.file.write_all(record.line().as_bytes())?;
        self.file.sync_data()?;
        self.next_seq += 1;
        Ok(record.seq)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("argusnet-journal-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(name);
        let _ = std::fs::remove_file(&path);
        path
    }

    fn write(path: &Path, run: u64, ticks: u64) {
        let mut writer = JournalWriter::create(path, run).unwrap();
        for tick in 0..ticks {
            writer
                .append(0, tick, tick as f64, serde_json::json!({ "n": tick }))
                .unwrap();
        }
    }

    #[test]
    fn a_journal_reads_back_what_was_written() {
        let path = scratch("roundtrip.jsonl");
        write(&path, 7, 5);
        let found = scan(&path, 7).unwrap();
        assert_eq!(found.records.len(), 5);
        assert!(!found.torn_tail);
        assert_eq!(found.records[3].body["n"], 3);
        assert!(
            JournalWriter::create(&path, 7).is_err(),
            "never overwritten"
        );
    }

    #[test]
    fn an_interrupted_last_write_is_a_torn_tail() {
        let path = scratch("torn.jsonl");
        write(&path, 7, 4);
        let intact = std::fs::metadata(&path).unwrap().len();
        // Half a line, as a crash mid-write leaves it.
        let mut file = OpenOptions::new().append(true).open(&path).unwrap();
        file.write_all(b"0123456789abcdef {\"seq\":4,\"ru").unwrap();
        let found = scan(&path, 7).unwrap();
        assert_eq!(found.records.len(), 4);
        assert!(found.torn_tail);
        assert_eq!(found.intact_bytes, intact);
        // Resuming drops it and continues the sequence.
        let mut writer = JournalWriter::resume(&path, 7, 4).unwrap();
        writer
            .append(0, 4, 4.0, serde_json::json!({ "n": 4 }))
            .unwrap();
        let found = scan(&path, 7).unwrap();
        assert_eq!(found.records.len(), 5);
        assert!(!found.torn_tail);
    }

    #[test]
    fn a_damaged_line_with_good_lines_after_it_is_corruption() {
        let path = scratch("corrupt.jsonl");
        write(&path, 7, 4);
        let text = std::fs::read_to_string(&path).unwrap();
        let damaged = text.replacen("\"n\":1", "\"n\":9", 1);
        assert_ne!(damaged, text);
        std::fs::write(&path, damaged).unwrap();
        assert!(matches!(
            scan(&path, 7),
            Err(JournalError::Corrupt { line: 2 })
        ));
    }

    #[test]
    fn another_runs_journal_is_refused() {
        let path = scratch("foreign.jsonl");
        write(&path, 7, 2);
        assert!(matches!(
            scan(&path, 8),
            Err(JournalError::Foreign { line: 1, .. })
        ));
    }

    #[test]
    fn resuming_past_the_journal_is_refused() {
        let path = scratch("short.jsonl");
        write(&path, 7, 2);
        assert!(JournalWriter::resume(&path, 7, 3).is_err());
    }
}
