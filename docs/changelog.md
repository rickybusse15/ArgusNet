# Changelog

## 2026-04-30 — Executor Authority and Blocking Safety Gate

- Promoted `MissionExecutor` from a passive observer to the choke-point for
  scan_map_inspect motion intent. Per-drone POI inspection redirects and
  return-to-home commands now flow through `MissionExecutor.dispatch()` →
  `validate_command → execute_command`.
- Replaced the placeholder `ExecutableCommand(description=str)` with a typed
  command (drone_id, target_xy_m, target_z_m, task_type, reason) and added
  `DroneRuntimeState` for safety-gate input.
- Added severity tagging (`hard` / `soft`) to `ConstraintViolation`. The
  blocking gate rejects `min_agl`, `max_agl`, and `min_drone_separation`;
  soft violations log only. `RETURN_HOME` is exempt from AGL gating.
- Rejected commands record a `SafetyEvent` on `MissionState.safety_events`,
  insert a single-tick `HOLD` task, and surface to the replay payload as
  `scan_mission_state.safety_events`. The replay schema is updated additively.
- Phase A determinism preserved: identical seed produces frame-byte-identical
  replays after wiring (only the wallclock timestamp in `meta` differs).

## 2026-04-29 — Documentation Reframing

- Reframed documentation around ArgusNet as a map, localize, inspect, revisit, and coordination
  system.
- Updated architecture and state ownership docs to use `src/argusnet` and `argusnet-*` crate names.
- Clarified current `scan_map_inspect` behavior: mapping coverage, localization confidence,
  inspection POI completion, deconfliction, egress, and replay state.
- Replaced outdated scenario and planning vocabulary with mapping, localization, inspection,
  revisit, POI assignment, and multi-drone coordination terminology.

## 2026-04-27 — Scan, Localize, Inspect, Egress

- Added scan → localize → inspect → egress mission state.
- Added scan coverage thresholds, POI localization ellipses, and inspection event logging.
- Added egress/RTH with per-drone home progress and arrival detection.
- Hardened multi-drone deconfliction.
- Improved viewer panels, selection details, event display, and scan overlay color.

## March 2026 — Viewer, Scene, Terrain, And Ingestion Foundations

- Consolidated documentation into `README.md` and `docs/`.
- Added native Bevy viewer and `smartscene-v1` package loading.
- Split scene compilation into terrain, geometry, GIS, and styling helpers.
- Added terrain presets, terrain-aware mission zones, rejection diagnostics, and export formats.
- Added MQTT and file replay ingestion adapters.
- Added Rust fusion service, replay export, and Python simulation workflow.
