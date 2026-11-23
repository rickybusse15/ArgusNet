# Python inventory (P6.0) and analysis without the native library (P6.1, first slice)

**Status:** P6.0 inventory at 2026-11-23, with the empty stubs removed. No module with
code in it was deleted: P6.3 removes those as their last runtime callers go, after
the gates in [DELIVERY_PLAN.md](DELIVERY_PLAN.md#8-phase-6--reduce-python-without-breaking-analysis).

Made with `scripts/module_reachability.py` from the installed entry points
(`argusnet`, `argusnet-sim`, `argusnet-export`, `argusnet-build-scene`). RUNTIME is
reachable from one; test-only only from `tests/` (or `scripts/`); ORPHAN from neither.

| Package | RUNTIME | Test-only | ORPHAN | What it becomes |
| --- | ---: | ---: | ---: | --- |
| `simulation` | 9,160 | 0 | 0 | Retired after scheduled qualification and one compatibility release; the reference until then |
| `evaluation` | 5,288 | 964 | 0 | Stays: the analysis package |
| `world` | about 7,200 | about 1,500 | 0 | Offline import stays; the rest goes with `simulation` |
| `planning` | 3,253 | 615 | 7 | Goes with `simulation` (Rust owns it); `inspection` data types used by exporters move first |
| `core` | 1,767 | 0 | 133 | Stays: replay and configuration types; `ids`, `time` are unused |
| `cli` | 1,303 | 0 | 0 | Becomes `argusnet-analysis` (P6.2) |
| `adapters` | 957 | 0 | 0 | Stays while the daemon's Python client does |
| `localization` | 857 | 764 | 671 | Goes with `simulation`; five modules are unused now |
| `mission` | 786 | 61 | 0 | Goes with `simulation` |
| `mapping` | 579 | 960 | 155 | Goes with `simulation`; `change_detection` is superseded by site memory |
| `security` | 425 | 0 | 5 | Stays with the client |
| `sensing` | 223 | 192 | 535 | Goes with `simulation`; four modules are unused now |
| `safety` | 178 | 0 | 0 | Goes with `simulation` |
| `indexing` | 0 | 0 | 780 | Unused: three modules of foundations nothing calls |

Counts are lines, as the script reports them.

## Removed here

Eleven five-line files and one empty package, each a docstring and an empty `__all__`,
imported by nothing in `src`, `tests`, `scripts` or the packaging: `ui` (all of it),
`indexing/{descriptors,landmarks,observation_index,retrieval}`,
`planning/{next_best_view,revisit}`.

## Deletion map for what is unused but has code

| Modules | Lines | Before it goes |
| --- | ---: | --- |
| `indexing/{keyframes,mission_index,tile_index}` | 780 | Site memory (F6) replaces the idea; confirm no outside user of the API |
| `localization/{gnss_fusion,loop_closure,odometry,pose_graph,relocalization}` | 671 | The navigation filter covers GNSS fusion; the others have no Rust counterpart and are design notes in code |
| `sensing/{camera,depth,sensor_base,thermal}` | 535 | Superseded by the rig and imaging models |
| `mapping/{change_detection,fusion}` | 155 | Superseded by site memory |
| `core/{ids,time}` | 133 | Confirm no outside user |

`mypy` ignore lists in `pyproject.toml` name three of the `indexing` modules.

## Analysis without the native library

`tests/test_import_without_native.py` imports every module of `cli`, `core`,
`evaluation`, `mapping`, `mission`, `planning`, `security` and `world`, each first and on
its own, with the native library refused. All do: it is loaded by the first call into
it. Three did not import first, native or not (`world.obstacles`, `world.visibility`,
`world.environment_io`): they and `world.environment` import each other, and only an
import that happened to start at `environment` closed the circle. The package starts
there now.

Not done: an install that skips building Rust (P6.1's extra), `argusnet-analysis` and
the command table (P6.2), and everything in P6.3 and P6.4.
