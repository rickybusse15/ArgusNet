# Real-site inspection baseline — 2026-10-26

This is an integration and reproducibility baseline, not hardware qualification
or a timing benchmark. Source/binary/workflow identities and full outcomes are
in [the machine-readable record](real-site-inspection-2026-10-26.json).

```bash
cargo build --release --locked -p argusnet-cli -p argusnet-native
python scripts/real_site_inspection.py --binary target/release/argusnet-rs \
  --output runs/real-site-release --seeds 7 42 --duration-s 120
```

The checked-in West Pleasant View DEM/OSM sources were imported with their
canonical shared origin. Each seed ran twice; replay bytes were identical.
Both replay schemas and package artifact hashes passed. Both annotated replays
passed `argusnet-replay`'s `verify_assurance` example, the viewer's own report
reader. Review scenes contain 41,487 triangles each.

| Measurement | Seed 7 | Seed 42 |
| --- | --- | --- |
| Frames | 481 | 481 |
| Known map fraction | 83.04% | 77.05% |
| Ground RMSE | 0.16875 m | 0.16687 m |
| Structure precision | 96.94% | 96.85% |
| Structure recall | 92.42% | 89.32% |
| Inspection points completed | 3/3 | 3/3 |
| Minimum recorded battery fraction | 90.57% | 91.28% |
| Final phase at 120 s | egress | egress |
| Report status | incomplete | incomplete |

Both outcomes record `mission_not_complete` and `localization_timed_out`.
Completing the inspection points alone does not make the mission complete.
These results must not be relabeled as passes by lengthening the time window
or weakening localization requirements without separate evidence.

## Next acceptance work

1. Explain and qualify localization timeout behavior on this site: record
   confidence/pose error and determine whether completion after timeout is an
   intended degraded mission, with a declared operating limit.
2. Measure egress to a declared terminal state. A completed mission needs its
   actual terminal evidence; proximity to home alone does not certify landing.
3. Declare map/inspection quality thresholds, then use held-out seeds and
   sensor/link/calibration faults. The two seeds here establish repeatability,
   not statistical qualification.
4. Remove remaining truth dependencies in scenario route/inspection setup
   before claiming a fully belief-driven mission. Continue scheduler and hardware
   qualification as their separate tracks.

## Render and packaging evidence

The combined viewer passes its test suite and the standard software-render smoke
fixture: a 650×480 image with 17,025 distinct colors and 231 green track pixels.
That checks the rendering pipeline; it does not establish a hardware frame budget.
The wheel was built, installed into a clean environment and checked outside the
source tree. Bandit passed. After updating outdated development-environment tools,
pip-audit reported no known dependency vulnerabilities; the local project itself
is not a PyPI-auditable distribution. Cargo-audit was not installed locally.

| Check | Result |
| --- | --- |
| Rust workspace excluding viewer | 1,539 passed; 1 ignored; 0 failed |
| Viewer | 396 passed; 0 failed |
| Python | 1,247 passed; 9 skipped; 78 subtests passed |
| Rust lint | Workspace/all targets, warnings denied: passed |
| Source lint/format/types | Passed |
| Protobuf regeneration and field disposition | Passed; all 172 fields covered |
| Installed-wheel contract | Passed outside the checkout |
| Windowed viewer | 20 s without panic; operator header visible (292 shades) |
