# G0: versioned payloads and what their readers do

Guardrail **G0** ([DELIVERY_PLAN.md](DELIVERY_PLAN.md)) asks that every payload
be preserved, normalized explicitly or rejected, with no silent missing-field
success. [FIELD_PARITY.md](FIELD_PARITY.md) covers the proto's fields; this
file covers **versions**: every persisted or wire payload that declares a
format or version, who writes it, who reads it, and what each reader does with
a version it was not written for. VERIFICATION_AND_ROLLOUT.md §6 asks for this
table to be checked in.

Reader behaviours: **rejects** (an error naming the version), **warns** (reads
it, and says so where the operator sees it), **none** (reads it without
looking). A row with a reader and **none** is a gap.

| Payload | Identifier (writer) | Readers → behaviour on another version | Test feeding another version |
| --- | --- | --- | --- |
| Replay JSON | `meta.schema_version` "2.0": Python `REPLAY_SCHEMA_VERSION` (`evaluation/replay.py`), Rust `argusnet-sim` `replay::SCHEMA_VERSION` | Python `validate_replay_document` → **rejects** another major (undeclared accepted, as older replays carry none); viewer → **warns**: a Critical "Unsupported replay schema" condition over the view (C7.1); `argusnet-replay` crate → none (the viewer judges for it); Python files reach it through `load_replay_document`, which validates (scenario evaluation, export, scene packaging); the scorecard, outcome and metrics functions take an in-memory document and judge no version | `tests/test_contract_versions.py`; viewer `ui/conditions.rs` `replay_schema_versions_are_judged_by_major` |
| Scene package | `format_version` "smartscene-v1" (`scene_loader.py`, viewer `schema.rs`) | Python `validate_scene_manifest` → **rejects**; viewer `ScenePackage::load` → **rejects** | `tests/test_contract_versions.py`; viewer `schema.rs` `a_scene_or_style_of_another_version_is_refused_by_name` |
| Scene style | `style_version` "smartstyle-v1" (`_scene_style.py`, viewer `schema.rs`) | viewer → **rejects**; Python has no reader | viewer `schema.rs` (same test) |
| Land cover | `format`/`version` "argusnet.land-cover"/1 (`scene_loader.py`) | viewer `LandCoverRaster::from_document` → **rejects** either | viewer `land_cover_raster.rs` `documents_it_cannot_honour_are_refused_with_the_reason` (format and version) |
| Environment bundle | `format_version` "smartmap-v1": `ENVIRONMENT_BUNDLE_FORMAT` (`environment_io.py`) | `load_environment_bundle` → **rejects** (was none until E1) | `tests/test_contract_versions.py` |
| Scenario bundle | `format`/`version` "argusnet.scenario-bundle"/1 (`scenario_bundle.py`, `argusnet-sim` `scenario.rs`) | `ScenarioBundle::from_json` → **rejects** (`BundleError::Format`/`Version`) | `argusnet-sim/tests/scenario_bundle.rs` (version) |
| Scenario config | `format`/`version` "argusnet.scenario-config"/1 (`sim.py --dump-config`, `builder/mod.rs`) | `ScenarioConfig::from_json` → **rejects** either; unknown keys refused | `builder/mod.rs` `a_config_of_another_format_or_version_is_refused_by_name` |
| Checkpoint | header `format_version` 4, `runtime_algorithm_version` 2, `seed_version` 1 (`snapshot.rs`) | `SimulationCheckpoint::verify` → **rejects** each | `snapshot.rs` `every_compatibility_field_is_actually_checked`; `tests/tick_transaction.rs` |
| Command envelope | `protocol_version` 2 (`command.rs`) | admission → **rejects** with `RejectReason::ProtocolVersion`; a checkpoint holding one fails restore | `tests/tick_transaction.rs` |
| Run directory `run.json` | `format` "argusnet-run/1", or "argusnet-run/2" for a run under a non-reference `planning_policy` (P1, recorded beside it, so a pre-P1 build refuses the run rather than recover it under the reference planner), `checkpoint_format`, `runtime_version` (`rundir.rs`) | recovery → **rejects** another `format`, a format that disagrees with the recorded policy, or another `checkpoint_format` (the latter was none until E1); `runtime_version` is provenance, the checkpoint header carries the algorithm version that matters | `rundir.rs` tests |
| `journal.jsonl`, `complete.json` | not versioned: journal lines carry an FNV checksum, run id and sequence | journal scan → **rejects** a wrong run, sequence or checksum | `argusnet-sim` journal tests |
| Native C ABI | `version` op returns 1 (`argusnet-native`) | **none** at runtime; only `tests/test_world_authority.py` asserts it. Gap: a Python caller of a newer ABI would not notice. Becomes a check when the first ABI change lands (WS-D) | — |
| Proto package | `argusnet.v1` | gRPC method paths: a client of another package gets UNIMPLEMENTED from gRPC itself | — |
| Written, not read back | mission outcome "argusnet.mission-outcome/1", performance summary "argusnet-performance-v1", performance suite "argusnet-performance-suite-v1", benchmark summary 1, replay provenance block | no reader in the repository; external tools must check them | — |

## Changing a version

Bump the identifier in its writer, keep or extend the reader's acceptance
deliberately, and update this row and its test in the same change. A reader
never accepts a newer version by default: it is refused with the version named,
or warned about where the operator sees it when refusing would lose the last
coherent view (the viewer's replay case).
