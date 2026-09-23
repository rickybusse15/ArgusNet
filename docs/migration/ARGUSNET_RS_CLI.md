# P4.9a: `argusnet-rs`, the Rust simulation from a command line

**Status:** delivered 2026-09-23, locally verified only (not pushed). Package ID
`P4.9a`. See [STATUS.md](STATUS.md).

Before this slice, the Rust tick ran only inside tests. `argusnet-rs` is the
first entry point a user runs. It takes a scenario bundle, runs it through the
same `assemble` the parity tests use, and writes a replay that the Python
evaluation and the viewer read. No Python runs at simulation time.

```text
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

A resumed run's replay holds the ticks from the resume point on. The
checkpoint is `SimulationCheckpoint` format 3, and restore checks every stage's
state against the run before changing anything (P4.8e.4). A checkpoint of a
different scenario is refused with exit 2, and the run is left untouched.

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
