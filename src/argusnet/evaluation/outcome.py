"""Mission outcome scorecards with attributable reasons (F1).

``mission_outcome(replay)`` answers *how did this run end, and why*, from a
replay alone (Python or ``argusnet-rs``):

- ``outcome``: ``completed``, ``aborted``, ``incomplete`` or
  ``objectives_unmet``;
- ``reasons``: every reason the outcome is what it is, each with a stable
  ``code``, a human ``detail`` and the ``evidence`` it rests on -- frame ticks
  and event ids (safety ``validation_id``, POI ids) -- so a claim can be traced
  to the frames that support it;
- sections for objectives, evidence sufficiency, energy, violations and
  timeouts/degradation.

A metric the replay cannot support is ``{"status": "unavailable", "reason":
...}``: never success, never zero error.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from argusnet.evaluation.metrics import evaluate_replay

OUTCOME_FORMAT = "argusnet.mission-outcome/1"
# Cap on evidence references listed per reason; the count is always exact.
MAX_EVIDENCE = 20


def unavailable(reason: str) -> dict[str, Any]:
    return {"status": "unavailable", "reason": reason}


def _reason(code: str, detail: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "code": code,
        "detail": detail,
        "evidence_count": len(evidence),
        "evidence": evidence[:MAX_EVIDENCE],
    }


def _tick(frames: list[dict[str, Any]], index: int) -> dict[str, Any]:
    return {"kind": "frame", "tick": index, "timestamp_s": frames[index].get("timestamp_s")}


def _artifact(meta: dict[str, Any], frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reasons the replay itself is not a whole run."""
    reasons = []
    recovery = meta.get("recovery") or {}
    if "recovered_from_tick" in recovery:
        tick = recovery["recovered_from_tick"]
        reasons.append(
            _reason(
                "recovered_after_crash",
                f"the run crashed; this replay holds only the ticks from {tick} on",
                [{"kind": "meta", "key": "recovery.recovered_from_tick", "value": tick}],
            )
        )
    dt_s = meta.get("dt_s")
    requested = meta.get("requested_duration_s")
    if dt_s and requested is not None:
        expected = int(round(float(requested) / float(dt_s))) + 1
        if len(frames) < expected:
            reasons.append(
                _reason(
                    "truncated",
                    f"{len(frames)} frames of the {expected} the requested duration needs",
                    [{"kind": "meta", "key": "requested_duration_s", "value": requested}],
                )
            )
    if not frames:
        reasons.append(_reason("no_frames", "the replay has no frames", []))
    return reasons


def _violations(frames: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Safety violations by code, and the abort reasons they raise."""
    by_code: Counter[str] = Counter()
    aborts: dict[str, list[dict[str, Any]]] = {}
    rejected = 0
    for index, frame in enumerate(frames):
        for event in frame.get("safety_events") or []:
            if event.get("accepted", True):
                continue
            rejected += 1
            for code in event.get("violations") or []:
                by_code[code] += 1
            if event.get("reason") == "abort":
                aborts.setdefault(event.get("subject_id", "?"), []).append(
                    {
                        "kind": "safety_event",
                        "id": event.get("validation_id"),
                        "tick": index,
                        "violations": list(event.get("violations") or []),
                    }
                )
    section = {
        "rejected_validations": rejected,
        "by_violation": dict(sorted(by_code.items())),
        "aborted_platforms": sorted(aborts),
    }
    reasons = [
        _reason(
            "safety_abort",
            f"{subject} was aborted by the safety monitor {len(events)} time(s), first at "
            f"tick {events[0]['tick']} for {', '.join(events[0]['violations']) or 'unspecified'}",
            events,
        )
        for subject, events in sorted(aborts.items())
    ]
    return section, reasons


def _objectives(
    frames: list[dict[str, Any]], report: Any
) -> tuple[dict[str, Any], list[dict[str, Any]], bool | None]:
    """Objective outcomes. Returns the section, its reasons, and whether the
    objectives were met (None when the replay cannot say)."""
    scan = next(
        (f["scan_mission_state"] for f in reversed(frames) if f.get("scan_mission_state")), None
    )
    if scan is not None:
        statuses = scan.get("poi_statuses") or []
        unmet = [s for s in statuses if s.get("status") != "complete"]
        phase = scan.get("phase")
        section = {
            "kind": "scan_map_inspect",
            "final_phase": phase,
            "pois_completed": scan.get("completed_poi_count"),
            "pois_total": scan.get("total_poi_count"),
            "unmet_pois": [s.get("poi_id") for s in unmet],
            "coverage_fraction": scan.get("scan_coverage_fraction"),
            "coverage_threshold": scan.get("scan_coverage_threshold"),
        }
        reasons = []
        last = len(frames) - 1
        if unmet:
            reasons.append(
                _reason(
                    "pois_not_inspected",
                    f"{len(unmet)} of {len(statuses)} POIs not inspected",
                    [
                        {
                            "kind": "poi",
                            "id": s.get("poi_id"),
                            "status": s.get("status"),
                            "tick": last,
                        }
                        for s in unmet
                    ],
                )
            )
        if phase != "complete":
            reasons.append(
                _reason(
                    "mission_not_complete",
                    f"the mission ended in phase {phase!r}, not 'complete'",
                    [_tick(frames, last)],
                )
            )
        return section, reasons, not reasons
    total = report.required_objectives_total
    if total == 0:
        return (
            {"kind": "none", "status": "unavailable", "reason": "no required objectives"},
            [],
            None,
        )
    met = report.required_objectives_met
    section = {"kind": "evaluation", "met": met, "total": total}
    reasons = (
        []
        if met >= total
        else [
            _reason(
                "objectives_unmet",
                f"{met} of {total} required objectives met",
                [{"kind": "report", "key": "required_objectives_met", "value": met}],
            )
        ]
    )
    return section, reasons, met >= total


def _energy(frames: list[dict[str, Any]]) -> dict[str, Any]:
    lowest: dict[str, tuple[float, int]] = {}
    for index, frame in enumerate(frames):
        for node in frame.get("nodes") or []:
            fraction = node.get("battery_fraction")
            # -1.0 is the replay's "no battery" sentinel: a ground station,
            # not a platform at zero.
            if fraction is None or fraction < 0.0:
                continue
            node_id = node.get("node_id", "?")
            if node_id not in lowest or fraction < lowest[node_id][0]:
                lowest[node_id] = (float(fraction), index)
    if not lowest:
        return unavailable("no platform reports a battery fraction")
    worst = min(lowest, key=lambda node_id: lowest[node_id][0])
    return {
        "min_fraction": lowest[worst][0],
        "min_platform": worst,
        "min_tick": lowest[worst][1],
        "per_platform": {
            k: {"min_fraction": v[0], "tick": v[1]} for k, v in sorted(lowest.items())
        },
    }


def _degradation(frames: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    first_timeout = next(
        (
            index
            for index, frame in enumerate(frames)
            if (frame.get("scan_mission_state") or {}).get("localization_timed_out")
        ),
        None,
    )
    section = {"localization_timed_out_at_tick": first_timeout}
    reasons = (
        []
        if first_timeout is None
        else [
            _reason(
                "localization_timed_out",
                f"localization timed out at tick {first_timeout}",
                [_tick(frames, first_timeout)],
            )
        ]
    )
    return section, reasons


def _evidence_sufficiency(report: Any) -> dict[str, Any]:
    return {
        "localisation_rmse_m": (
            report.localisation_rmse_m
            if report.localisation_rmse_m is not None
            else unavailable("no fused track was matched to a truth")
        ),
        "track_continuity_per_target": dict(report.track_continuity_per_target)
        or unavailable("no targets"),
        "covariance_reduction_mean": (
            report.covariance_reduction_mean
            if report.covariance_reduction_mean is not None
            else unavailable("no track qualified")
        ),
        "covariance_reduction_tracks": report.covariance_reduction_tracks,
    }


def mission_outcome(replay: dict[str, Any], *, mission_type: str = "unknown") -> dict[str, Any]:
    """The run's outcome and every reason for it, from its replay."""
    meta = replay.get("meta") or {}
    frames = replay.get("frames") or []
    report = evaluate_replay(
        replay,
        scenario_name=str(meta.get("scenario_name", "unknown")),
        mission_type=mission_type,
        difficulty=0.0,
        seed=int(meta.get("seed", 0)),
    )
    artifact = _artifact(meta, frames)
    violations, aborts = _violations(frames)
    objectives, unmet, met = _objectives(frames, report)
    degradation, degraded = _degradation(frames)

    if artifact:
        outcome, reasons = "incomplete", artifact + aborts + unmet + degraded
    elif aborts:
        outcome, reasons = "aborted", aborts + unmet + degraded
    elif met is False:
        outcome, reasons = "objectives_unmet", unmet + degraded
    elif met is None:
        outcome = "completed"
        reasons = degraded + [
            _reason(
                "no_objectives_to_judge",
                "the run ended without abort; it states no objectives to judge success by",
                [_tick(frames, len(frames) - 1)],
            )
        ]
    else:
        outcome = "completed"
        reasons = degraded + [
            _reason(
                "objectives_met",
                "every objective met, without abort",
                [_tick(frames, len(frames) - 1)],
            )
        ]
    return {
        "format": OUTCOME_FORMAT,
        "scenario_name": meta.get("scenario_name"),
        "seed": meta.get("seed"),
        "outcome": outcome,
        "reasons": reasons,
        "objectives": objectives,
        "evidence_sufficiency": _evidence_sufficiency(report),
        "energy": _energy(frames),
        "violations": violations,
        "degradation": degradation,
    }
