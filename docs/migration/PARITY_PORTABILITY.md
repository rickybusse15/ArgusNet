# Parity fixtures off their capture platform

**Status:** 2026-10-03. Test-only; no runtime or fixture value changes.
See [STATUS.md](STATUS.md). Part of A3.3 (cross-platform determinism) in the
[agent execution plan](AGENT_EXECUTION_PLAN.md).

## What was wrong

The parity suites compare the Rust port with captures of the Python
reference, bit for bit. Python's `math.sin`, `math.exp`, `np.linalg.norm` and
friends call the platform's maths library, and so does Rust's `f64::sin`. The
fixtures were captured on macOS (aarch64), and CI ran on a self-hosted macOS
runner, so Apple's libm sat on both sides of every comparison.

On Linux x86_64 (glibc 2.39), the first full local run of `main` (`0b1032c`)
failed nine tests in six suites:

| Suite | Test | First difference |
| --- | --- | --- |
| `behaviors_parity` | `every_preset_matches_the_reference_bit_for_bit` | evasive velocity z, 1 ULP |
| `builder_world` | `box_and_wall_footprints_match_the_references_arithmetic` | box 29 footprint |
| `controller_parity` | `the_look_direction_matches_the_reference` | look at 30° tilt |
| `follow_path_parity` | `follow_path_matches_the_reference` | a terrain oracle lookup missed its recorded key by 1 ULP |
| `weather_parity` | `every_preset_matches_the_reference_bit_for_bit` | `visibility_at_range(9000)` |
| `scenario_bundle` | four whole runs: `a_python_scenario_runs_in_the_rust_tick_and_matches_where_it_must`, `an_all_follow_scenario_matches_on_every_tick`, `a_scan_map_inspect_run_with_replayed_localization_matches_throughout`, `a_blocking_safety_run_matches_on_every_tick` | a velocity component, 1 ULP (from tick 19 in `tt_follow`) |

Recapturing the fixtures on Linux would not fix it either: on Linux, the
Python reference and the committed fixture disagree on 2,463 of 85,680
behaviour values, and Rust still differs from a Linux capture in `mixed`,
where numpy's `norm` goes through a BLAS path Rust does not mirror.

`cargo test` also hid the scale of the failure: it stops at the first
failing test binary, so the first run reported only `behaviors_parity`.

## The rule now

Every parity fixture records where it was captured: `"_captured_on":
"<os>-<arch>"`, named as Rust's `std::env::consts` names it
(`scripts/fixture_platform.py`; every capture script writes it). The Rust
helper `rust/argusnet-sim/tests/common/parity.rs` then compares:

- **On the capture platform:** bit for bit, exactly as before. Nothing is
  loosened where the reference was produced.
- **Elsewhere:** within `MAX_ULPS = 2` units in the last place, or within
  `ABS_FLOOR = 1e-12` near zero. Recorded oracles (terrain, obstacle and
  line-of-sight answers in `follow_path_parity`) look up exact bits first and
  fall back to the nearest recorded argument within the same bound.

A fixture without the tag fails loudly rather than choosing a mode.

## Evidence

| Check | Command | Result (linux-x86_64, locally verified) |
| --- | --- | --- |
| The six suites at the bound | `cargo test -p argusnet-sim --test behaviors_parity --test weather_parity --test follow_path_parity --test scenario_bundle --test controller_parity --test builder_world` | all pass |
| The bound is not slack | the same suites with the bound at 1 ULP | all pass: the measured gap is exactly 1 ULP, whole runs included |
| The helper | `common::parity` unit tests | exact mode rejects 1 ULP; bounded mode rejects bound + 1, NaN, and a miss in another key component; an untagged fixture panics |
| Python consumers of the fixtures | `pytest tests/test_scenario_bundle.py` | 22 passed, 6 skipped (as before) |
| Lint | `cargo fmt --check`, `cargo clippy -p argusnet-sim --all-targets -D warnings`, `ruff` on the scripts | clean |

Not verifiable here: that the suites are still bit-exact on macOS aarch64.
The tag makes them so by construction (`Parity::Exact` compares bits),
but the run has to happen on that platform.

## What this does not change

- ADR-006's rule stands: no platform libm in a draw path. These are
  kinematics and world-geometry comparisons against a Python oracle, not
  random draws.
- Whole-run tests still stop being comparable where they did before
  (engagement or mission-phase divergence). A 1-ULP difference that changed
  a decision would surface as a divergence there, and fail.
- `HEIGHT_TOLERANCE_M` in `builder_world` (numpy `exp`) is unchanged.
