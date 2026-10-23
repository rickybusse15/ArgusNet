"""S1 records: the canonical contracts, derived from one replay.

Each record follows its schema in `docs/contracts/` (`argusnet.<kind>/v1`)
and carries the same envelope: `contract`, `record_id`, `source` (a JSON
pointer into the replay) and `ids`, the identity and revision keys
`session_id / run_id / epoch / tick / world_revision / belief_revision /
sequence` (VIEWER_UI 10.2). What the replay cannot support is the explicit
`{"status": "unavailable", "reason": ...}`, never a guess or a zero.

`records_from_replay(replay, assurance)` returns the record index and the
records by kind. A kind the producer does not record at all (reconstruction,
change) is `unavailable` in the index with its reason, which is not the
same as a recorded kind with no records (no safety event happened).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

KINDS = (
    "site",
    "mission",
    "platform",
    "localization",
    "spatial_revision",
    "trajectory",
    "safety",
    "evidence",
    "reconstruction",
    "finding",
    "change",
    "report",
)
INDEX_CONTRACT = "argusnet.record-index/v1"

# Kinds no producer records yet, and why.
NOT_RECORDED = {
    "reconstruction": "replays carry no reconstruction product (surface, volume or point cloud)",
    "change": "a single run has no earlier world to compare with; spatial memory (F6) is not built",
}


def unavailable(reason: str) -> dict[str, str]:
    return {"status": "unavailable", "reason": reason}


def _value(value: Any, reason: str) -> Any:
    """`value`, or unavailable for `reason` when it is missing or not finite."""
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return unavailable(reason)
    return value


class _Ids:
    """The run's identity keys, and a sequence over every record made."""

    def __init__(self, meta: Mapping[str, Any]) -> None:
        self.session_id = _value(meta.get("session_id"), "replays do not record a session")
        self.run_id = _value(meta.get("run_id"), "the replay records no run id")
        self.epoch = _value(meta.get("epoch"), "the replay records no epoch number")
        provenance = meta.get("provenance") or {}
        world = meta.get("world_revision") or provenance.get("world_content_hash")
        self.world_revision = _value(world, "the replay records no world revision")
        self.sequence = 0

    def __call__(self, tick: int | None, belief_revision: int | None = None) -> dict[str, Any]:
        ids = {
            "session_id": self.session_id,
            "run_id": self.run_id,
            "epoch": self.epoch,
            "tick": tick if tick is not None else unavailable("a run-level record"),
            "world_revision": self.world_revision,
            "belief_revision": belief_revision
            if belief_revision is not None
            else unavailable("no belief revision applies to this record"),
            "sequence": self.sequence,
        }
        self.sequence += 1
        return ids


def _record(
    kind: str,
    record_id: str,
    ids: dict[str, Any],
    source: str | None,
    **fields: Any,
) -> dict[str, Any]:
    return {
        "contract": f"argusnet.{kind}/v1",
        "record_id": record_id,
        "ids": ids,
        "source": source if source is not None else unavailable("no single source in the replay"),
        **fields,
    }


def _nodes(frames: list[dict[str, Any]]) -> dict[str, list[tuple[int, dict[str, Any]]]]:
    by_id: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for index, frame in enumerate(frames):
        for node in frame.get("nodes") or []:
            node_id = node.get("node_id")
            if node_id:
                by_id.setdefault(str(node_id), []).append((index, node))
    return by_id


def _path_length(points: Iterable[list[float]]) -> float:
    total = 0.0
    previous = None
    for point in points:
        if previous is not None:
            total += math.dist(previous[:3], point[:3])
        previous = point
    return total


def records_from_replay(
    replay: Mapping[str, Any], assurance: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    """The record index and the records by kind for `replay`, whose assurance
    report (`argusnet.mission-assurance/v1`, with its F1 `outcome`) is
    `assurance`."""
    meta: Mapping[str, Any] = replay.get("meta") or {}
    frames: list[dict[str, Any]] = list(replay.get("frames") or [])
    ids = _Ids(meta)
    records: dict[str, list[dict[str, Any]]] = {kind: [] for kind in KINDS}

    # site
    origin = meta.get("enu_origin")
    imports = []
    for layer in ("terrain_import", "buildings", "features"):
        record = meta.get(layer)
        if isinstance(record, Mapping) and record.get("format"):
            imports.append(
                {
                    "layer": layer,
                    "format": str(record["format"]),
                    "content_hash": _value(record.get("content_fnv1a"), "not recorded"),
                    "source_sha256": _value(record.get("source_sha256"), "not recorded"),
                }
            )
    terrain_source = ((meta.get("terrain_summary") or {}).get("source") or {}).get("source")
    records["site"].append(
        _record(
            "site",
            "site",
            ids(None),
            "/meta",
            origin=dict(origin) if isinstance(origin, Mapping) else unavailable("no enu_origin"),
            crs_id=_value(meta.get("crs_id"), "no crs_id"),
            bounds_m=dict(meta["environment_bounds_m"])
            if isinstance(meta.get("environment_bounds_m"), Mapping)
            else unavailable("no environment bounds"),
            terrain_source=_value(terrain_source, "no terrain summary"),
            imports=imports,
        )
    )

    # mission
    outcome = assurance.get("outcome") or {}
    objectives = []
    states = [
        (i, f["scan_mission_state"]) for i, f in enumerate(frames) if f.get("scan_mission_state")
    ]
    if states:
        index, state = states[-1]
        for j, poi in enumerate(state.get("poi_statuses") or []):
            objectives.append(
                {
                    "objective_id": str(poi.get("poi_id")),
                    "status": str(poi.get("status", "unknown")),
                    "platform_id": _value(poi.get("assigned_drone_id"), "unassigned"),
                    "completed_at_s": _value(poi.get("completion_time_s"), "not completed"),
                    "source": f"/frames/{index}/scan_mission_state/poi_statuses/{j}",
                }
            )
    records["mission"].append(
        _record(
            "mission",
            "mission",
            ids(None),
            "/meta",
            scenario_name=_value(meta.get("scenario_name"), "no scenario name"),
            mission_type=_value(
                (meta.get("scenario_options") or {}).get("mission_mode")
                or meta.get("mission_mode"),
                "no mission mode recorded",
            ),
            duration_s=_value(meta.get("actual_duration_s"), "no recorded duration"),
            outcome=_value(outcome.get("outcome"), "no F1 outcome"),
            objectives=objectives,
        )
    )

    # platform, localization, trajectory
    nodes = _nodes(frames)
    estimates: dict[str, tuple[int, int, dict[str, Any]]] = {}
    for index, state in states:
        for j, estimate in enumerate(state.get("localization_estimates") or []):
            drone = estimate.get("drone_id")
            if drone:
                estimates[str(drone)] = (index, j, estimate)
    for node_id in sorted(nodes):
        first_index, first = nodes[node_id][0]
        records["platform"].append(
            _record(
                "platform",
                f"platform/{node_id}",
                ids(first_index),
                f"/frames/{first_index}/nodes",
                platform_id=node_id,
                mobile=bool(first.get("is_mobile", False)),
                sensor_type=_value(first.get("sensor_type"), "no sensor type"),
                max_range_m=_value(first.get("max_range_m"), "no sensor range"),
                fov_half_angle_deg=_value(first.get("fov_half_angle_deg"), "no field of view"),
                calibration_version=unavailable("calibration is not versioned in replays"),
                firmware_version=unavailable("no flight software; a simulated platform"),
            )
        )
        if not first.get("is_mobile", False):
            continue
        samples: list[list[float]] = [
            list(node["position"]) for _, node in nodes[node_id] if node.get("position")
        ]
        times = [
            node.get("timestamp_s", frames[i].get("timestamp_s")) for i, node in nodes[node_id]
        ]
        records["trajectory"].append(
            _record(
                "trajectory",
                f"trajectory/{node_id}",
                ids(None),
                f"/frames/*/nodes[node_id={node_id}]",
                platform_id=node_id,
                samples=len(samples),
                start_s=_value(times[0] if times else None, "no samples"),
                end_s=_value(times[-1] if times else None, "no samples"),
                path_length_m=_value(
                    round(_path_length(samples), 3) if samples else None, "no samples"
                ),
                planned=unavailable("planned trajectories are not recorded in replays"),
            )
        )
        located = estimates.get(node_id)
        estimate = located[2] if located else {}
        records["localization"].append(
            _record(
                "localization",
                f"localization/{node_id}",
                ids(located[0] if located else None),
                f"/frames/{located[0]}/scan_mission_state/localization_estimates/{located[1]}"
                if located
                else None,
                platform_id=node_id,
                estimated_at_s=_value(estimate.get("timestamp_s"), "no estimate recorded"),
                position_m=_value(estimate.get("position_estimate"), "no estimate recorded"),
                position_std_m=_value(estimate.get("position_std_m"), "no estimate recorded"),
                confidence=_value(estimate.get("confidence"), "no estimate recorded"),
                integrity=unavailable("physical localization integrity is not modelled"),
            )
        )

    # spatial revision
    knowledge = (meta.get("world_knowledge") or {}).get("knowledge")
    belief: dict[str, Any] = {}
    for frame in reversed(frames):
        belief = (frame.get("mapping_state") or {}).get("belief") or {}
        if belief:
            break
    revisions = sum(1 for frame in frames if frame.get("belief_updates"))
    self_built = knowledge == "none"
    prior = "the run had the true world as a prior; it built no map"
    records["spatial_revision"].append(
        _record(
            "spatial_revision",
            "spatial_revision",
            ids(len(frames) - 1 if frames else None, revisions if self_built else None),
            "/meta/world_knowledge" if self_built else "/meta",
            world_knowledge="none" if self_built else "prior",
            belief_revisions=revisions if self_built else unavailable(prior),
            known_fraction=_value(belief.get("known_fraction"), prior),
            ground_rmse_m=_value(belief.get("ground_rmse_m"), prior),
            structure_precision=_value(belief.get("structure_precision"), prior),
            structure_recall=_value(belief.get("structure_recall"), prior),
        )
    )

    # safety
    for index, frame in enumerate(frames):
        for j, event in enumerate(frame.get("safety_events") or []):
            records["safety"].append(
                _record(
                    "safety",
                    str(
                        event.get("validation_id") or event.get("event_id") or f"safety/{index}/{j}"
                    ),
                    ids(index),
                    f"/frames/{index}/safety_events/{j}",
                    validation_id=_value(event.get("validation_id"), "no validation id"),
                    platform_id=_value(event.get("subject_id"), "no subject"),
                    accepted=_value(event.get("accepted"), "acceptance not recorded"),
                    reason=_value(event.get("reason"), "no reason recorded"),
                    violations=[str(v) for v in event.get("violations") or []],
                    timestamp_s=_value(
                        event.get("timestamp_s", frame.get("timestamp_s")), "no timestamp"
                    ),
                )
            )

    # evidence: each completed objective's dwell
    if states:
        index, state = states[-1]
        for j, poi in enumerate(state.get("poi_statuses") or []):
            if poi.get("status") != "complete":
                continue
            records["evidence"].append(
                _record(
                    "evidence",
                    f"evidence/{poi.get('poi_id')}",
                    ids(index),
                    f"/frames/{index}/scan_mission_state/poi_statuses/{j}",
                    objective_id=str(poi.get("poi_id")),
                    platform_id=_value(poi.get("assigned_drone_id"), "unassigned"),
                    captured_at_s=_value(poi.get("completion_time_s"), "no completion time"),
                    dwell_s=_value(poi.get("dwell_accumulated_s"), "no dwell recorded"),
                    sufficiency=unavailable("evidence quality is not evaluated (F7)"),
                )
            )

    # findings: the F1 outcome's reasons
    for k, reason in enumerate(outcome.get("reasons") or []):
        records["finding"].append(
            _record(
                "finding",
                f"finding/{k}/{reason.get('code')}",
                ids(None),
                f"/summary/mission_assurance/outcome/reasons/{k}",
                code=str(reason.get("code")),
                detail=str(reason.get("detail")),
                severity=unavailable("findings carry no severity yet"),
                evidence_count=int(reason.get("evidence_count", 0)),
                evidence=list(reason.get("evidence") or []),
            )
        )

    # report
    records["report"].append(
        _record(
            "report",
            "report",
            ids(None),
            "/summary/mission_assurance",
            status=str(assurance.get("status", "unavailable")),
            outcome=_value(outcome.get("outcome"), "no F1 outcome"),
            completion_rate=_value(assurance.get("completion_rate"), "no completion rate"),
            energy_reserve_min=_value(assurance.get("energy_reserve_min"), "no battery recorded"),
            unavailable_fields=list(assurance.get("unavailable_fields") or []),
            source_replay_sha256=str(assurance.get("source_replay_sha256", "")),
        )
    )

    record_index: dict[str, Any] = {
        "contract": INDEX_CONTRACT,
        "kinds": {
            kind: (
                {"status": "unavailable", "count": 0, "reason": NOT_RECORDED[kind]}
                if kind in NOT_RECORDED
                else {
                    "status": "recorded",
                    "count": len(records[kind]),
                    "path": f"records/{kind}.json",
                }
            )
            for kind in KINDS
        },
    }
    return record_index, records
