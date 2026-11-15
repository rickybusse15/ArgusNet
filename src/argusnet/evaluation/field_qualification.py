"""Score a replay against a run profile's frozen thresholds.

A profile (``argusnet-rs sim --profile field_v1``) is qualified by running it
over predetermined and held-out seeds and scoring every replay against
thresholds declared beforehand (``docs/contracts/field_v1.thresholds.json``).
This module is the scoring: it reads a replay document and the thresholds and
says, gate by gate, what was measured and whether it passed.

Scoring reads the truth a replay records for evaluation (the true power
lines, the belief block's accuracy figures). The run itself never does.

A gate whose value the replay does not hold fails, unless the gate does not
apply to the world at all (wire clearance in a world without power lines):
an unknown requirement cannot pass.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

THRESHOLD_FORMAT = "argusnet.profile-thresholds"
THRESHOLD_VERSION = 1
REPORT_FORMAT = "argusnet.profile-qualification/v1"

#: Points along each span a conductor is sampled at for the clearance measure.
_SPAN_SAMPLES = 40


def load_thresholds(path: Path) -> dict[str, Any]:
    """Read and check a thresholds document."""
    document: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("format") != THRESHOLD_FORMAT:
        raise ValueError(f"{path}: format {document.get('format')!r}, expected {THRESHOLD_FORMAT}")
    if document.get("version") != THRESHOLD_VERSION:
        raise ValueError(
            f"{path}: version {document.get('version')!r}, this build reads {THRESHOLD_VERSION}"
        )
    for name, gate in document.get("per_run", {}).items():
        if not ({"min", "max", "equals"} & set(gate)):
            raise ValueError(f"{path}: gate {name} declares no min, max or equals")
    return document


def _conductor_polylines(
    power_lines: list[dict[str, Any]],
) -> list[list[tuple[float, float, float]]]:
    """Each conductor of each span, sampled as a polyline of 3-D points.

    A span's conductors hang between the tops of its two supports, offset
    sideways, with a parabolic sag of ``sag_ratio`` times the span's length
    at mid-span.
    """
    polylines = []
    for line in power_lines:
        supports = line["supports"]
        offsets = line.get("conductor_offsets_m") or [0.0]
        sag_ratio = float(line.get("sag_ratio", 0.0))
        for a, b in zip(supports, supports[1:], strict=False):
            ax, ay = a["xy_m"]
            bx, by = b["xy_m"]
            az = a["base_z_m"] + a["height_m"]
            bz = b["base_z_m"] + b["height_m"]
            length = math.hypot(bx - ax, by - ay)
            if length <= 0.0:
                continue
            left = (-(by - ay) / length, (bx - ax) / length)
            sag = sag_ratio * length
            for offset in offsets:
                points = []
                for k in range(_SPAN_SAMPLES + 1):
                    t = k / _SPAN_SAMPLES
                    points.append(
                        (
                            ax + (bx - ax) * t + left[0] * offset,
                            ay + (by - ay) * t + left[1] * offset,
                            az + (bz - az) * t - 4.0 * sag * t * (1.0 - t),
                        )
                    )
                polylines.append(points)
    return polylines


def _point_segment_distance(
    p: tuple[float, ...], a: tuple[float, ...], b: tuple[float, ...]
) -> float:
    ab = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    ap = (p[0] - a[0], p[1] - a[1], p[2] - a[2])
    dot = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2]
    t = (
        0.0
        if dot <= 0.0
        else max(0.0, min(1.0, (ap[0] * ab[0] + ap[1] * ab[1] + ap[2] * ab[2]) / dot))
    )
    return math.dist(p, (a[0] + ab[0] * t, a[1] + ab[1] * t, a[2] + ab[2] * t))


def _wire_clearance_min_m(
    frames: list[dict[str, Any]], power_lines: list[dict[str, Any]]
) -> float | None:
    """The closest any mobile platform came to a true conductor, or ``None``
    in a world without power lines."""
    polylines = _conductor_polylines(power_lines)
    if not polylines:
        return None
    # A bounding box per polyline, so most positions are dismissed cheaply.
    boxes = [
        (
            min(p[0] for p in points),
            max(p[0] for p in points),
            min(p[1] for p in points),
            max(p[1] for p in points),
        )
        for points in polylines
    ]
    best = math.inf
    for frame in frames:
        for node in frame.get("nodes", []):
            if not node.get("is_mobile"):
                continue
            p = tuple(node["position"])
            for points, (x0, x1, y0, y1) in zip(polylines, boxes, strict=True):
                gap = max(x0 - p[0], p[0] - x1, y0 - p[1], p[1] - y1, 0.0)
                if gap >= best:
                    continue
                for a, b in zip(points, points[1:], strict=False):
                    best = min(best, _point_segment_distance(p, a, b))
    return best


def measure(replay: dict[str, Any]) -> dict[str, Any]:
    """Every quantity the thresholds name, as the replay holds it.

    A quantity the replay does not hold is ``None``.
    """
    frames = replay.get("frames") or []
    meta = replay.get("meta") or {}
    if not frames:
        return {}
    last = frames[-1]
    mission = last.get("scan_mission_state") or {}
    belief = (last.get("mapping_state") or {}).get("belief") or {}
    quality = belief.get("quality") or {}

    phases = [(frame.get("scan_mission_state") or {}).get("phase") for frame in frames]
    complete_at = next((i for i, phase in enumerate(phases) if phase == "complete"), None)

    pois = mission.get("poi_statuses") or []
    pois_complete = (
        sum(1 for poi in pois if poi.get("status") == "complete") / len(pois) if pois else None
    )

    # Stations: where each drone stood on the first tick.
    homes = {
        node["node_id"]: node["position"]
        for node in frames[0].get("nodes", [])
        if node["is_mobile"]
    }
    home_radius = None
    if complete_at is not None and homes:
        home_radius = 0.0
        for frame in frames[complete_at:]:
            for node in frame.get("nodes", []):
                home = homes.get(node["node_id"])
                if home is not None:
                    away = math.hypot(node["position"][0] - home[0], node["position"][1] - home[1])
                    home_radius = max(home_radius, away)

    contacts = sum(len(frame.get("world_contacts") or []) for frame in frames)

    margin = None
    for frame in frames:
        for reserve in (frame.get("energy_reserves") or {}).values():
            value = reserve["remaining_wh"] - reserve["reserve_wh"]
            margin = value if margin is None else min(margin, value)

    separation = None
    for item in frames:
        mobile = [node["position"] for node in item.get("nodes", []) if node.get("is_mobile")]
        for index, first in enumerate(mobile):
            for second in mobile[index + 1 :]:
                apart = math.dist(first, second)
                separation = apart if separation is None else min(separation, apart)

    max_step = 0.0
    for before, after in zip(frames, frames[1:], strict=False):
        was = {node["node_id"]: node["position"] for node in before.get("nodes", [])}
        for node in after.get("nodes", []):
            if node.get("is_mobile") and node["node_id"] in was:
                max_step = max(max_step, math.dist(was[node["node_id"]], node["position"]))

    return {
        "mission_complete": complete_at is not None,
        "mission_complete_s": (
            frames[complete_at]["timestamp_s"] if complete_at is not None else None
        ),
        "final_phase": phases[-1],
        "pois_complete_fraction": pois_complete,
        "home_radius_m": home_radius,
        "world_contacts": contacts,
        "known_fraction": belief.get("known_fraction"),
        "quality_met_fraction": quality.get("met_fraction"),
        "structure_recall": belief.get("structure_recall"),
        "structure_precision": belief.get("structure_precision"),
        "ground_rmse_m": belief.get("ground_rmse_m"),
        "wire_clearance_min_m": _wire_clearance_min_m(frames, meta.get("power_lines") or []),
        "drone_separation_min_m": separation,
        "energy_reserve_margin_min_wh": margin,
        "max_step_m": max_step,
        "run_profile": (meta.get("run_profile") or {}).get("id"),
    }


def _judge(value: Any, gate: dict[str, Any]) -> bool:
    if value is None:
        return False
    if "equals" in gate:
        return bool(value == gate["equals"])
    if "min" in gate and not value >= gate["min"]:
        return False
    return not ("max" in gate and not value <= gate["max"])


def score_replay(
    replay: dict[str, Any],
    thresholds: dict[str, Any],
    *,
    repeat_identical: bool | None = None,
) -> dict[str, Any]:
    """Score one replay: measurements, a verdict per gate, and whether all passed.

    ``repeat_identical`` is whether a second run of the same seed and config
    gave the same replay bytes; the caller measures it, since one replay
    cannot. ``None`` (not measured) fails the gate.
    """
    measured = measure(replay)
    measured["repeat_identical"] = repeat_identical
    home_gate = thresholds["per_run"].get("home_radius_m", {})
    measured["drones_home"] = (
        None
        if measured.get("home_radius_m") is None
        else measured["home_radius_m"] <= home_gate.get("max", math.inf)
    )
    not_applicable = set()
    if not (replay.get("meta") or {}).get("power_lines"):
        not_applicable.add("wire_clearance_min_m")
    first = (replay.get("frames") or [{}])[0]
    if sum(1 for node in first.get("nodes", []) if node.get("is_mobile")) < 2:
        not_applicable.add("drone_separation_min_m")

    gates = {}
    for name, gate in thresholds["per_run"].items():
        if name in not_applicable:
            gates[name] = {"status": "not_applicable", "value": None}
            continue
        value = measured.get(name)
        bound = {key: gate[key] for key in ("min", "max", "equals") if key in gate}
        gates[name] = {
            "status": "pass" if _judge(value, gate) else "fail",
            "value": value,
            **bound,
        }
    failed = sorted(name for name, gate in gates.items() if gate["status"] == "fail")
    profile = thresholds.get("profile")
    if measured.get("run_profile") != profile:
        failed.append("run_profile")
        gates["run_profile"] = {
            "status": "fail",
            "value": measured.get("run_profile"),
            "equals": profile,
        }
    return {"measured": measured, "gates": gates, "failed": failed, "passed": not failed}


def summarize(
    runs: list[dict[str, Any]], thresholds: dict[str, Any], *, heldout_seeds: set[int]
) -> dict[str, Any]:
    """The campaign verdict over scored runs (each a ``score_replay`` result
    with ``seed`` and ``role`` added by the caller)."""
    campaign = thresholds["campaign"]
    scored = [run for run in runs if run.get("role") != "tuning"]
    qualification = [run for run in scored if run["seed"] not in heldout_seeds]
    heldout = [run for run in scored if run["seed"] in heldout_seeds]
    passed = [run for run in scored if run["passed"]]
    fraction = len(passed) / len(scored) if scored else 0.0
    reasons = []
    if len(qualification) < campaign["min_qualification_seeds"]:
        reasons.append(
            f"{len(qualification)} qualification seeds, "
            f"{campaign['min_qualification_seeds']} required"
        )
    if len(heldout) < campaign["min_heldout_seeds"]:
        reasons.append(f"{len(heldout)} held-out seeds, {campaign['min_heldout_seeds']} required")
    if fraction < campaign["min_pass_fraction"]:
        reasons.append(
            f"{len(passed)} of {len(scored)} runs passed every gate, "
            f"{campaign['min_pass_fraction']:.0%} required"
        )
    failures: dict[str, list[int]] = {}
    for run in scored:
        for name in run["failed"]:
            failures.setdefault(name, []).append(run["seed"])

    def spread(name: str) -> dict[str, float] | None:
        values = [
            run["measured"][name]
            for run in scored
            if isinstance(run["measured"].get(name), (int, float))
            and not isinstance(run["measured"].get(name), bool)
        ]
        if not values:
            return None
        ordered = sorted(values)
        return {
            "min": ordered[0],
            "median": ordered[len(ordered) // 2],
            "max": ordered[-1],
            "n": len(ordered),
        }

    return {
        "qualified": not reasons,
        "reasons": reasons,
        "runs": len(scored),
        "passed": len(passed),
        "pass_fraction": fraction,
        "failures_by_gate": {name: sorted(seeds) for name, seeds in sorted(failures.items())},
        "spread": {
            name: spread(name)
            for name in thresholds["per_run"]
            if name not in ("mission_complete", "drones_home", "repeat_identical")
        }
        | {"mission_complete_s": spread("mission_complete_s")},
    }


def markdown(report: dict[str, Any]) -> str:
    """The campaign report as a short Markdown page."""
    summary = report["summary"]
    lines = [
        f"# {report['profile']} qualification: {report['site']}",
        "",
        f"**{'Qualified' if summary['qualified'] else 'Not qualified'}**: "
        f"{summary['passed']} of {summary['runs']} runs passed every gate.",
        "",
    ]
    for reason in summary["reasons"]:
        lines.append(f"- {reason}")
    if summary["failures_by_gate"]:
        lines += ["", "| Gate failed | Seeds |", "| --- | --- |"]
        for name, seeds in summary["failures_by_gate"].items():
            lines.append(f"| `{name}` | {', '.join(str(seed) for seed in seeds)} |")
    lines += ["", "| Measure | Threshold | Min | Median | Max |", "| --- | --- | --- | --- | --- |"]
    gates = report["thresholds"]["per_run"]
    for name, spread in summary["spread"].items():
        if spread is None:
            continue
        gate = gates.get(name, {})
        bound = ", ".join(f"{key} {gate[key]}" for key in ("min", "max") if key in gate) or "—"
        lines.append(
            f"| `{name}` | {bound} | {spread['min']:.3f} | {spread['median']:.3f} "
            f"| {spread['max']:.3f} |"
        )
    lines += [
        "",
        f"Commit `{report['commit']}`, binary `{report['binary_sha256'][:16]}`, "
        f"world `{report['world_id']}`, {report['duration_s']:.0f} s runs.",
        "",
    ]
    for limit in report["limits"]:
        lines.append(f"- {limit}")
    return "\n".join(lines) + "\n"
