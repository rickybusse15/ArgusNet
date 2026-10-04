# G0: closed string vocabularies

[CONTRACT_VERSIONS.md](CONTRACT_VERSIONS.md) covers versions; this file covers
the string fields whose values come from a fixed set. G0 asks that none be
accepted silently when it means nothing to the reader. Two checks keep the
producers honest against `docs/replay-schema.json`, which constrains the
replay's vocabularies:

- `rust/argusnet-sim/tests/replay_vocabulary.rs`: every value a Rust producer
  writes is in the schema's enum. Each enum's variants are listed through an
  exhaustive `match`, so adding a variant fails to compile until it is checked.
- `tests/test_replay_vocabulary.py`: the same for the Python producers' constants
  and enums.

Reader behaviours: **rejects**, **defaults** (an unknown value becomes a stated
default, and the default is a drawn state, not a guess), **displays** (shown as
written), **ignores** (not read).

| Field | Vocabulary (defined in) | Readers → unknown value |
| --- | --- | --- |
| `sensor_type` (node state, bundle, config) | optical, radar, thermal, lwir, mwir, ir: `argusnet.core.types.SENSOR_TYPES`, `argusnet_types::SENSOR_TYPES` (D-14) | Python `SensorConfig` → **rejects**; Rust `ScenarioBundle::from_json` → **rejects**; schema → **rejects**; `SensorBand::from_sensor_type` maps the four thermal names to the thermal band (everything that reaches it is now in the vocabulary); proto → empty is the wire's unset, read as optical |
| `TrackerConfig.data_association_mode` | labeled, gnn, jpda (`AssociationMode`) | Python adapter → **rejects**; `argusnet_proto::parse_association_mode` (gRPC) and the server's YAML/CLI overrides → **rejects** (was: silently labeled until E1.2); empty on the wire → labeled |
| Observation rejection `reason` | runtime cascade (`RejectReason`), fusion (`REJECT_*` in `argusnet-core` and the Python adapter), Python simulation (`REJECT_*` in `sim.py`) | schema → **rejects**; viewer → **defaults** to the generic rejection style; proto and Python adapter → pass through |
| Track `lifecycle_state` | tentative, confirmed, coasting, deleted (`LifecycleState`, `TRACK_STATE_*`) | schema → **rejects**; viewer → **ignores** (not in the replay crate) |
| Scan mission `phase` | scanning, localizing, inspecting, egress, complete (`MissionPhase`) | schema → **rejects**; viewer → **displays** the raw value in grey, "UNKNOWN" on the timeline |
| Inspection `event_type` | entered, coverage_updated, exited, violation (`argusnet_sim::mapping::INSPECTION_EVENT_TYPES`, `sim.py`) | schema → **rejects**; viewer → **displays** |
| Deconfliction `resolution` | lateral_offset, vertical_offset, corridor_hold (`Resolution`) | schema → **rejects**; viewer → **displays** |
| Belief cell `status` | unknown, known_safe, known_obstacle, uncertain, outside_geofence (`BeliefCellStatus`, both sides) | schema → **rejects**; viewer → **ignores** (belief reaches it as rasters) |
| Pose estimate `status`, mission `event_type` | `LocalizationStatus`, `MissionEventType` (Python) | schema → **rejects**; viewer → **ignores** |
| POI `status` | pending, active, complete | schema → **rejects**; viewer → **defaults** to the neutral colour and label |
| `MissionZone.zone_type` | `ZONE_TYPES` (Python) | Python `MissionZone` → **rejects**; Rust builder → **rejects** an unknown profile zone; runtime → anything but `exclusion` is non-exclusion; viewer → **defaults** to grey |
| Safety `state` / `reason` | nominal, caution, warning, abort and the monitor's reasons | checkpoint restore → **rejects**; viewer and Python adapter → **display** / pass through |
| Land-cover class codes | `LandCoverClass` 0–8 (both sides) | Rust `from_u8` → `None`; viewer → **defaults** to no tint (an unnamed code is drawn as untinted ground, never as a class) |
| Free text by design | `TargetMetadata.status`/`behavior`/`target_type`, `HealthResponse.status`, `mission_type`, `blocker_type` (an unknown blocker counts as building occlusion, as the reference does) | pass through; no reader branches on them |

## Adding a value

Add it to the producer, the schema's enum, and this row in one change; the two
vocabulary tests fail until the schema accepts what a producer writes.
