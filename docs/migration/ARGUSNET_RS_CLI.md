# P4.9a: `argusnet-rs`, the Rust simulation from a command line

**Status:** delivered 2026-09-23, locally verified, then merged 2026-10-01 in #122 (`badb1e1`). Package ID
`P4.9a`. See [STATUS.md](STATUS.md).

Before this slice, the Rust tick ran only inside tests. `argusnet-rs` is the
first entry point a user runs. It takes a scenario bundle, runs it through the
same `assemble` the parity tests use, and writes a replay that the Python
evaluation and the viewer read. No Python runs at simulation time.

```text
argusnet-rs sim --config config.json --replay replay.json     # Rust only (P4.9b.2b)
argusnet-rs build --config config.json --bundle scenario.json # Rust only (P4.9b.2b)
argusnet sim --duration-s 60 --export-config config.json      # flags -> config (P4.9c)
argusnet-rs sim --config config.json --timing-json t.json     # timing record (P4.9c)
argusnet sim --duration-s 60 --export-bundle scenario.json    # Python: describe
argusnet-rs sim --bundle scenario.json --replay replay.json   # Rust: run
argusnet-rs sim --bundle scenario.json --checkpoint-at 120 --checkpoint ck.json
argusnet-rs sim --bundle scenario.json --resume ck.json --replay rest.json
argusnet-rs version
```

## The binary

The crate is `rust/argusnet-cli` and the binary is `argusnet-rs`. It is not
called `argusnet`: pip installs a Python `argusnet` console script, and a Cargo
binary of the same name would overwrite it or be shadowed, depending on PATH
order (DELIVERY_PLAN P4.9).

| Exit | Meaning |
| --- | --- |
| 0 | ran to completion; replay written if asked for |
| 1 | the run faulted (a stage error, a hole in truth, an I/O failure writing output); **no replay is written** |
| 2 | the bundle or checkpoint could not be read, is invalid, or does not belong to this bundle |
| 3 | the bundle is valid, but uses something this build does not assemble |

Unless `--quiet` is given, the binary prints provenance: its own version, the
bundle version, the scenario name, the seed, the tick count and `dt`, and the
tick it starts from. When the run ends, it prints the startup time, tick-time
p50/p95/max and the mission's final phase. `version` prints the bundle format
and version, plus the checkpoint format, runtime-algorithm and seed versions.

Every replay also carries the same provenance in `meta.provenance`: runtime,
version, the bundle and checkpoint formats, the runtime-algorithm version, the
input (`config` or `bundle`) and, for a config, the resolved config with every
default filled in. It records no clock and no host, so a replay's bytes depend
only on what was run. `--timing-json PATH` writes `startup_ms`, the tick count,
the tick-time `mean`/`p50`/`p95`/`p99`/`max`, `total_ms` and the build profile
(P4.9c). A hidden `fuse --capture C --output O` replays recorded
`IngestFrameRequest`s through the in-process engine, for the direct fusion
parity test ([PHASE_ACCEPTANCE.md](PHASE_ACCEPTANCE.md)).

A resumed run's replay holds the ticks from the resume point on. The
checkpoint is `SimulationCheckpoint` format 3, and restore checks every stage's
state against the run before changing anything (P4.8e.4). A checkpoint of a
different scenario is refused with exit 2, and the run is left untouched.

## Timestamped delivery (B4)

`--fusion-lag-s L` models delivery between capture and fusion: fusion ingests
the evidence acquired `L` seconds ago, at its acquisition time. The delays come
from `--delivery-delay-s` (every source), `--source-delay NODE_ID=SECONDS`
(repeatable) and a seeded `--delivery-jitter-s`, and must not exceed `L`
(otherwise exit 3, before the run). `--delivery-queue-capacity` bounds each
source's queue. Every dropped envelope is a rejected observation with a
`delivery_*` reason, and `meta.delivery_totals` accounts for them all. Without
`--fusion-lag-s` the run is the legacy run, byte for byte. Design and
evidence: [DELIVERY_TIMING.md](DELIVERY_TIMING.md).

## Assignment history (P0)

Every replay's `meta.assignment_history` lists each drone's task changes:
the task it flies on a tick (`search`, `follow:<target>`, `explore`,
`poi:<id>`, `egress`, `frontier`), when it changed and why, and the POI
handoffs and team joins stage 9 applied. It is additive and records nothing
the run reads back; `scripts/replay_hashes.py --ignore-meta
assignment_history` proves a change to it leaves everything else
byte-identical. See PLANNING_EFFICIENCY_PLAN.md P0.

## A lane at its own rate (B2.4, `scheduled_v1`)

`--executor lanes --mission-rate NUM/DEN` runs the mission lane (stage 9) at
`NUM/DEN` Hz instead of every tick: at the first tick at or after each of its
deadlines, sitting out the rest, with its next run given the time since it last
ran. Inspect dwell is still credited at every boundary, where the drone was.
The replay records `meta.execution_model: "scheduled_v1"`, `meta.lane_rates`
and `provenance.checkpoint_format` 5. Without the flag, `--executor lanes` is the legacy run, byte
for byte. Refused without `--executor lanes`. A `--run-dir` records the rate
in `run.json`, so `--recover` continues at it (a different `--mission-rate`
is refused). Its checkpoints carry the rate (format 5) and `--resume` them
only with the same `--mission-rate`. The viewer's `--run` takes the same flag
and runs the same scheduled run. `--mapping-rate NUM/DEN` does the same for
the mapping lane (B2.5): the ticks it sits out are kept and integrated when
it next runs, so coverage and the self-built map catch up exactly.
Design and evidence:
[SCHEDULER_LEGACY_MAPPING.md](SCHEDULER_LEGACY_MAPPING.md) (B2.4).

## Scheduled faults (F2)

`--faults MANIFEST` applies an `argusnet.fault-manifest`: sensor outages,
link delays and source restarts at given simulated times. The manifest is
validated before the run (exit 3 if a node has no sensor, an interval is
empty, or a link fault lacks `--fusion-lag-s` or exceeds it) and recorded in
`meta.fault_manifest`. See [FAILURE_SCENARIOS.md](FAILURE_SCENARIOS.md) and
`tests/fixtures/fault_manifest_example.json`.

## Crash-recoverable runs (A4)

`argusnet-rs sim --bundle B --run-dir D [--checkpoint-every N]` records the
run in a new directory `D`:

| File | What |
| --- | --- |
| `run.json` | The immutable run configuration: the bundle exactly as run, its FNV-1a, the runtime and checkpoint versions. Written once, atomically |
| `journal.jsonl` | One line per committed tick, flushed to stable storage before the run advances: the tick's deconfliction, safety and inspection events and the scan mission's state. Each line carries a checksum and the run's identity (`argusnet_sim::journal`) |
| `checkpoint.json` | The latest checkpoint, replaced atomically every `N` ticks; its header's `journal_cursor` is how much of the journal it covers |
| `complete.json` | Written last. Without it the run is incomplete |

`argusnet-rs sim --recover D` finishes a crashed run: it restores the latest
checkpoint, cuts the journal back to that checkpoint's cursor (dropping a
torn last write and any later ticks) and runs to completion. The journal is
then byte-identical to an uninterrupted run's. The replay holds only the ticks
after the checkpoint and records `meta.recovery.recovered_from_tick`, so a
crash never turns an incomplete artifact into a complete one. Recovery refuses,
with exit 2, a damaged `run.json` ("incomplete run artifact"), a journal damaged
inside what the checkpoint covers (corrupt, not torn), and a checkpoint from
another scenario. A journal write that fails stops the run at the last durable
tick (exit 1); nothing is dropped. A run directory is never reused. Recording
does not change the replay's bytes.

## The adapter: `argusnet sim --export-bundle PATH`

`--export-bundle` is a flag on the existing Python command, not a new command.
Every scenario flag the user already knows means the same thing, because the
scenario is built by the same code path the Python run uses. The flag writes
the bundle and returns without simulating. The bundle carries the tracker
configuration **this command** runs with (`DEFAULT_SIMULATION_TRACKER_CONFIG`),
not the dataclass default. A test compares the two, and they differ in
`max_stale_steps`.

Options the Rust runtime cannot honour are refused by name with
`UnsupportedScenario`. None are dropped silently:

* Scenario options bundle v1 cannot describe: `enforce_flight_envelope`,
  `use_vio_position`, follow controllers with route planners, non-waypoint
  search trajectories, and mission modes other than `target_tracking` and
  `scan_map_inspect` (P4.8e.1).
* **Run-time options** (new here): `--continuous`, `--realtime`, `--streaming`,
  `--compact`, and `--replay-subsample` greater than 1. `argusnet-rs` writes a
  full, unpaced replay. `--continuous` would otherwise have exported a bundle
  of `sys.maxsize` ticks.

`SimulationConstants` from `--config-file` reach the run only through the
scenario (`build_default_scenario`), so the bundle carries them.

## Verification

| Test | Shows |
| --- | --- |
| `cli::sim_writes_the_runtimes_replay` | the file the binary writes is equal to `assemble(..).run_to_replay()` |
| `cli::a_checkpoint_resumed_from_the_command_line_continues_the_same_run` | the scan mission, checkpointed at tick 150 by one process and resumed by another, matches the uninterrupted run frame for frame |
| `cli::bad_input_and_unsupported_bundles_exit_distinctly` | missing file 2, malformed JSON 2, unknown mission mode 3 |
| `cli::a_run_that_faults_exits_1_and_writes_no_replay` | a hole in the target's truth at t=3 (the truth table faults) and at t=10 (the pursuit oracle, which looks ahead, faults first in stage 1): exit 1, the fault named with its tick exactly once, no replay file |
| `cli::version_names_every_format` | the version output names every format |
| `test_export_bundle_writes_exactly_what_the_command_would_run` | the CLI export equals `export_bundle` of the same options, including the command's own tracker config |
| `test_export_bundle_refuses_what_the_runtime_cannot_run` | a scenario option is refused by name |
| `test_export_bundle_refuses_run_time_options_it_would_drop` | each of the five run-time flags is refused, and no file is written |
| `test_an_exported_bundle_runs_in_argusnet_rs_to_a_valid_replay` | Python exports, `argusnet-rs` runs it, and Python's `load_replay_document` accepts the replay (skipped if the binary is not built) |

Exported with default flags, a 60 s scenario (`argusnet sim --duration-s 60
--export-bundle`) takes 1.2 s. `argusnet-rs` runs its 241 ticks in 0.44 s on a
debug build. Timing distributions, measured properly, belong to P4.9c.

### Mutation round

10 mutants. **9 caught, 1 equivalent.**

| Mutant | Result |
| --- | --- |
| checkpoint written one tick late | caught |
| `--resume` ignored | caught |
| unsupported reported as invalid (3 → 2) | caught |
| replay never written | caught |
| a non-tick fault treated as completion (`Err(_) => break`) | caught (it survived before the fault test existed) |
| a tick fault's message prefixed with the tick a second time | caught |
| a tick fault treated as completion | **equivalent**: `run_to_replay` then retries the failed tick. A faulted tick has not advanced (validate-before-mutate), so the run faults again with the same message and the same exit code |
| export with the dataclass-default tracker config | caught |
| `--export-bundle` falls through and runs the simulation | caught |
| run-time options not refused | caught |

### Gate

Pinned toolchain 1.98.0-aarch64, exit codes read directly:

| Check | Result |
| --- | --- |
| `cargo test --workspace` | 1046 passed, 0 failed |
| `cargo clippy --workspace --all-targets -- -D warnings` | clean |
| `cargo fmt --all -- --check` | clean |
| `ruff check src/argusnet scripts tests`, `ruff format --check` | clean |
| `pytest tests/` | 1018 passed, 9 skipped, 0 failed |
| CI | **not run**: nothing is pushed |

## Findings

* **A tick error already names its tick and stage.** Prefixing the CLI's own
  `tick N:` printed the tick twice. Only truth and invariant faults, which do
  not name the tick, get the prefix.
* **The pursuit oracle faults before the truth table does** when a follower is
  engaged, because it reads ahead of the tick being prescribed. The fault is
  correct and names the target. Both paths are covered.

## Not in this slice

* **Replay enrichment.** Each frame of the Rust replay carries nodes, truths,
  observations, rejections, tracks and metrics. It does not yet carry the
  mapping state, localization, the scan mission's state (phase, POIs) or
  deconfliction events, which the viewer's panels read. That is P4.9a.2.
* **A Rust-native scenario builder.** A bundle still has to come from Python.
  That is P4.9b.
* **Canonical-scenario and multi-seed acceptance**, and release-build timing.
  That is P4.9c.

## Next

**P4.9a.2**: additive replay enrichment. The runtime's replay gains the fields
the viewer's mapping, localization and mission panels read, so that a replay
from `argusnet-rs` shows everything a Python replay of the same bundle shows.
