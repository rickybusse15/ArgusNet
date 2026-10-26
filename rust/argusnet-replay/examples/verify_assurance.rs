//! Verify a client package with the same reader the viewer uses.
use std::error::Error;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn Error>> {
    let path = PathBuf::from(
        std::env::args_os()
            .nth(1)
            .ok_or("expected replay.json path")?,
    );
    let value: serde_json::Value =
        serde_json::from_str(&argusnet_replay::read_text_limited(&path)?)?;
    let document = argusnet_replay::ReplayDocument::try_from(value)?;
    let report = document
        .mission_assurance
        .ok_or("replay carries no assurance report")?
        .map_err(|reason| format!("invalid assurance report: {reason}"))?;
    println!(
        "Verified report for {} frames: {}",
        document.frames.len(),
        report.status
    );
    Ok(())
}
