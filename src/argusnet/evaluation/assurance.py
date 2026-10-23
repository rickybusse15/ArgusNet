"""S1 replay-derived client package; recording is evidence, not qualification."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from argusnet.evaluation.metrics import evaluate_replay
from argusnet.evaluation.outcome import mission_outcome

ASSURANCE_VERSION = "argusnet.mission-assurance/v1"
PACKAGE_VERSION = "argusnet.client-package/v1"


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


@dataclass(frozen=True)
class ClientPackage:
    """Resolved file bytes. Hashes cover the exact bytes delivered to the client."""

    manifest: bytes
    assurance: bytes
    markdown: bytes
    replay: bytes

    def write(self, directory: Path) -> None:
        """Publish a new package atomically; never overwrite a reviewed package."""
        directory = Path(directory)
        if directory.exists():
            raise FileExistsError(f"Client package already exists: {directory}")
        directory.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".argusnet-package-", dir=directory.parent))
        try:
            for name, content in (
                ("manifest.json", self.manifest),
                ("assurance.json", self.assurance),
                ("assurance.md", self.markdown),
                ("replay.json", self.replay),
            ):
                (staging / name).write_bytes(content)
            os.rename(staging, directory)
        finally:
            if staging.exists():
                shutil.rmtree(staging)


def build_client_package(replay_doc: dict[str, Any]) -> ClientPackage:
    """Build a deterministic, unsigned assurance package from recorded fields.

    summary.mission_assurance is an additive replay extension consumed by Review.
    It is replaced on regeneration, so an older report never describes new frames.
    Missing contracts are explicit nulls; legacy timeout/POI dwell never certify
    localization integrity, evidence sufficiency, safe landing or qualification.
    """
    # The source hash is over the input as given, less any earlier report:
    # a `summary` that held only that report is dropped with it, so a package
    # of a package's replay names the same source as the first. The output's
    # summary is made only after hashing.
    source = copy.deepcopy(replay_doc)
    summary = source.get("summary")
    if isinstance(summary, dict) and "mission_assurance" in summary:
        summary.pop("mission_assurance")
        if not summary:
            del source["summary"]
    source_hash = _sha256(_json_bytes(source))
    meta = source.get("meta") or {}
    frames = source.get("frames") or []
    report = evaluate_replay(
        source,
        str(meta.get("scenario_name", "unknown")),
        str(meta.get("mission_mode", "unknown")),
        0.0,
        int(meta.get("seed", 0)),
    )
    provenance = {
        key: meta.get(key)
        for key in (
            "site_id",
            "mission_id",
            "run_id",
            "coordinate_frame",
            "altitude_datum",
            "time_basis",
            "world_revision",
            "belief_revision",
            "world_content_hash",
            "config_hash",
            "runtime_algorithm_version",
            "calibration_version",
            "firmware_version",
            "model_version",
        )
    }
    provenance.update(scenario_name=report.scenario_name, seed=report.seed)
    events: list[dict[str, Any]] = []
    for i, frame in enumerate(frames):
        for kind in (
            "mission_events",
            "inspection_events",
            "safety_events",
            "deconfliction_events",
        ):
            for j, event in enumerate(frame.get(kind) or []):
                pointer = f"/frames/{i}/{kind}/{j}"
                events.append(
                    {
                        "event_id": pointer,
                        "recorded_event_id": event.get("event_id"),
                        "kind": kind,
                        "timestamp_s": event.get("timestamp_s", frame.get("timestamp_s")),
                        "subject_id": event.get(
                            "subject_id",
                            event.get(
                                "platform_id", event.get("node_id", event.get("yielding_drone_id"))
                            ),
                        ),
                        "event_type": event.get("event_type"),
                        "record": event,
                        "reason": event.get("reason"),
                        "source_pointer": pointer,
                    }
                )
    states = [
        (i, f["scan_mission_state"]) for i, f in enumerate(frames) if f.get("scan_mission_state")
    ]
    phase = states[-1][1].get("phase") if states else None
    objectives: list[dict[str, Any]] = []
    if states:
        index, state = states[-1]
        for j, poi in enumerate(state.get("poi_statuses") or []):
            objectives.append(
                {
                    "objective_id": poi["poi_id"],
                    "status": poi.get("status", "unavailable"),
                    "source_pointer": f"/frames/{index}/scan_mission_state/poi_statuses/{j}",
                    "evidence_sufficiency": None,
                }
            )
    # A completed POI or all-met count does not prove terminal mission/landing.
    # The status never contradicts the embedded F1 outcome: an abort is the
    # safety monitor's (the mission phases never record one), and a run is
    # completed only when it reached its terminal phase AND F1 finds it
    # completed; a truncated or recovered run, or unmet objectives, is not.
    outcome = mission_outcome(source, mission_type=str(meta.get("mission_mode", "unknown")))
    status = (
        "unavailable"
        if not frames or phase is None
        else "aborted"
        if outcome["outcome"] == "aborted"
        else "completed"
        if phase == "complete" and outcome["outcome"] == "completed"
        else "incomplete"
    )
    unavailable = [f"provenance.{key}" for key, value in provenance.items() if value is None]
    unavailable += [
        "landing_outcome",
        "localization_integrity",
        "evidence_sufficiency",
        "qualification",
        "recording_completeness",
    ]
    if report.mission_completion_rate is None:
        unavailable.append("completion_rate")
    if report.energy_reserve_min is None:
        unavailable.append("energy_reserve_min")
    if report.localisation_rmse_m is None:
        unavailable.append("localisation_rmse_m")
    if report.track_continuity_mean is None:
        unavailable.append("track_continuity_mean")
    timeouts = [
        f"/frames/{i}/scan_mission_state/localization_timed_out"
        for i, state in states
        if state.get("localization_timed_out")
    ]
    assurance = {
        "schema_version": ASSURANCE_VERSION,
        "source_replay_sha256": source_hash,
        "provenance": provenance,
        "status": status,
        # The F1 record (`argusnet.mission-outcome/1`): how the run ended and
        # every reason why, each with the frames and events it rests on. One
        # outcome format: this report carries it rather than recomputing it.
        "outcome": outcome,
        "terminal_phase": phase,
        "terminal_source_pointer": f"/frames/{states[-1][0]}/scan_mission_state/phase"
        if states
        else None,
        "completion_rate": report.mission_completion_rate,
        "required_objectives_met": report.required_objectives_met,
        "required_objectives_total": report.required_objectives_total,
        "objectives": objectives,
        "energy_reserve_min": report.energy_reserve_min,
        "energy_reserve_per_platform": report.energy_reserve_per_drone,
        "localisation_rmse_m": report.localisation_rmse_m,
        "track_continuity_mean": report.track_continuity_mean,
        "localization_timeout_sources": timeouts,
        "landing_outcome": None,
        "localization_integrity": None,
        "recording_completeness": None,
        "metric_definitions": {
            "completion_rate": (
                "Recorded required-objective or POI completion fraction; not evidence acceptance"
            ),
            "energy_reserve_min": (
                "Minimum recorded mobile-platform battery fraction; not return feasibility"
            ),
            "localisation_rmse_m": (
                "Legacy fixed nearest-truth track pairing RMSE in meters; not pose integrity"
            ),
            "track_continuity_mean": (
                "Mean fraction of frames with target observations; not track survival"
            ),
        },
        "events": events,
        "unavailable_fields": unavailable,
    }
    assurance_bytes = _json_bytes(assurance)
    markdown = _assurance_markdown(assurance).encode()
    source.setdefault("summary", {})["mission_assurance"] = assurance
    replay = _json_bytes(source)
    artifacts = [
        ("assurance.json", assurance_bytes),
        ("assurance.md", markdown),
        ("replay.json", replay),
    ]
    manifest = _json_bytes(
        {
            "schema_version": PACKAGE_VERSION,
            "source_replay_sha256": source_hash,
            "provenance": provenance,
            "status": "incomplete" if unavailable else "complete",
            "signature": None,
            "available_formats": ["json", "markdown", "replay-json"],
            "unavailable_products": ["evidence", "reconstruction", "change_set", "qualification"],
            "artifacts": [
                {"path": name, "sha256": _sha256(content), "size_bytes": len(content)}
                for name, content in artifacts
            ],
        }
    )
    return ClientPackage(manifest, assurance_bytes, markdown, replay)


def _assurance_markdown(report: dict[str, Any]) -> str:
    def percent(value: float | None) -> str:
        return "Unavailable" if value is None else f"{value:.1%}"

    lines = [
        "# Mission assurance",
        "",
        f"Recorded outcome: **{report['status']}**.",
        "",
        f"- Terminal phase: {report['terminal_phase'] or 'Unavailable'}",
        f"- Objective completion: {percent(report['completion_rate'])}",
        f"- Minimum recorded energy reserve: {percent(report['energy_reserve_min'])}",
        "- Landing outcome: Unavailable",
        "- Localization integrity: Unavailable",
        f"- Recorded localization timeout frames: {len(report['localization_timeout_sources'])}",
        "",
        "## Objectives",
        "",
    ]
    if not report["objectives"]:
        lines.append("Per-objective results unavailable.")
    for objective in report["objectives"]:
        lines.append(
            f"- {objective['objective_id']}: {objective['status']} "
            f"(`{objective['source_pointer']}`); evidence sufficiency unavailable."
        )
    lines += ["", "## Recorded events", ""]
    if not report["events"]:
        lines.append("No recorded events; recording completeness unavailable.")
    for event in report["events"]:
        lines.append(
            f"- {event['kind']} at {event['timestamp_s']} s: "
            f"{event['reason'] or 'reason unavailable'} (`{event['source_pointer']}`)."
        )
    lines += [
        "",
        "## Unavailable fields",
        "",
        *[f"- {key}" for key in report["unavailable_fields"]],
        "",
        f"Source replay SHA-256: `{report['source_replay_sha256']}`.",
        "",
        "Unsigned replay-derived report. Recorded completion does not establish qualification.",
        "",
    ]
    return "\n".join(lines)
