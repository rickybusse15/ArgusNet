# S1 contracts

Versioned JSON Schemas (draft-07, like `../replay-schema.json`) for the records
a client package carries. AUTONOMOUS_SYSTEM_PLAN S1 and AGENT_EXECUTION_PLAN
S1 call for them before the Plan and Review workspaces are built on them.

| Kind | Contract | From a replay today |
| --- | --- | --- |
| site | `argusnet.site/v1` | origin, CRS, bounds, terrain source, imported layers and their hashes |
| mission | `argusnet.mission/v1` | scenario, mission type, duration, objectives (POIs), the F1 outcome |
| platform | `argusnet.platform/v1` | each node: mobile, sensor, range, field of view. Calibration and firmware are unavailable |
| localization | `argusnet.localization/v1` | each platform's last estimate. Physical integrity is unavailable |
| spatial_revision | `argusnet.spatial_revision/v1` | prior world, or the self-built map's revisions and error |
| trajectory | `argusnet.trajectory/v1` | each platform's flown path. The planned trajectory is unavailable |
| safety | `argusnet.safety/v1` | each safety-monitor validation |
| evidence | `argusnet.evidence/v1` | each completed objective's dwell. Sufficiency is unavailable (F7) |
| reconstruction | `argusnet.reconstruction/v1` | not recorded: the index says why |
| finding | `argusnet.finding/v1` | each F1 reason, with the frames and events it rests on |
| change | `argusnet.change/v1` | not recorded: a single run has no earlier world (F6) |
| report | `argusnet.report/v1` | the assurance report's status, outcome and unavailable fields |

## The envelope

Every record carries:

- `contract`: its kind and version;
- `record_id`;
- `source`: a JSON pointer into the source replay;
- `ids`: the identity and revision keys of VIEWER_UI 10.2:
  `session_id / run_id / epoch / tick / world_revision / belief_revision /
  sequence`.

`sequence` is unique across a package, from 0. `tick` is the frame index for a
record about one moment. A run-level record has no tick, and says so.

## Unavailable, not empty

A value the source cannot support is `{"status": "unavailable", "reason":
"..."}`. It is never a guess, a zero or a success. Two cases look alike and
are kept apart:

- A kind no producer records is `"status": "unavailable"` in
  `records/index.json`, with its reason and a count of 0. Reconstruction and
  change are like this today.
- A recorded kind with no records is `"status": "recorded"` with a count of
  0. For example, no safety event happened.

## Producing them

`argusnet client-package --replay R --output DIR` writes `records/index.json`
(`argusnet.record-index/v1`) and one `records/<kind>.json` per recorded kind,
each listed and hashed in the manifest
(`argusnet.evaluation.records.records_from_replay`). `tests/test_contracts.py`
validates every record against these schemas, for synthetic replays and for
real `argusnet-rs` runs, both with a prior world and building their own.
