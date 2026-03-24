"""evaluation.py — Evaluation harness for ArgusNet.

Implements the evaluation model defined in docs/SCENARIOS.md, Section 1.
The main entry point is ``evaluate_replay()``, which takes a replay document
(as produced by ``replay.build_replay_document()``) and returns an
``EvaluationReport`` containing all metrics.

Usage::

    from argusnet.evaluation import evaluate_replay, check_pass_fail

    report = evaluate_replay(replay_doc, "baseline_small_1t", "surveillance", 0.1, 42)
    report = check_pass_fail(report)
    print(report.passed, report.failure_reasons)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Default pass/fail thresholds (from SCENARIOS.md Section 1)
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLDS: Dict[str, object] = {
    # Track performance
    "time_to_reacquire_mean_s_max": 10.0,
    "time_to_reacquire_mean_s_fail": 30.0,
    "track_continuity_mean_min": 0.80,
    "localisation_rmse_m_max": 15.0,
    "localisation_rmse_m_fail": 50.0,
    "covariance_reduction_mean_min": 0.50,
    # Reliability
    "false_handoff_rate_max": 0.5,
    "safety_override_count_max": 0,
    "comms_dropout_duration_s_max": 5.0,
    # Mission outcome
    "mission_completion_rate_min": 1.0,
    "energy_reserve_min_min": 0.10,
}


# ---------------------------------------------------------------------------
# EvaluationReport dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvaluationReport:
    """Immutable evaluation report produced by ``evaluate_replay()``.

    All field names are canonical metric identifiers as defined in
    docs/SCENARIOS.md Section 1.4.
    """

    # Identity
    scenario_name: str
    mission_type: str
    difficulty: float
    seed: int
    duration_s: float

    # Track performance
    time_to_reacquire_mean_s: Optional[float] = None
    time_to_reacquire_p95_s: Optional[float] = None
    track_continuity_mean: float = 0.0
    track_continuity_per_target: Dict[str, float] = field(default_factory=dict)
    localisation_rmse_m: Optional[float] = None
    localisation_rmse_per_track: Dict[str, float] = field(default_factory=dict)
    covariance_reduction_mean: Optional[float] = None

    # Reliability
    false_handoff_rate: float = 0.0
    infeasible_path_rejection_count: int = 0
    safety_override_count: int = 0
    comms_dropout_count: int = 0
    comms_dropout_duration_s: float = 0.0

    # Mission outcome
    mission_completion_rate: float = 0.0
    required_objectives_met: int = 0
    required_objectives_total: int = 0
    energy_reserve_min: float = 1.0
    energy_reserve_per_drone: Dict[str, float] = field(default_factory=dict)

    # Summary
    passed: bool = False
    failure_reasons: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    generated_at_utc: str = ""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_tracks_from_frame(frame: dict) -> List[dict]:
    """Return the tracks list from a replay frame dict."""
    return frame.get("tracks") or []


def _extract_truths_from_frame(frame: dict) -> List[dict]:
    """Return the truths list from a replay frame dict."""
    return frame.get("truths") or []


def _extract_observations_from_frame(frame: dict) -> List[dict]:
    """Return the observations list from a replay frame dict."""
    return frame.get("observations") or []


def _track_position(track: dict) -> np.ndarray:
    """Return position as a numpy array from a track dict."""
    pos = track.get("position", [0.0, 0.0, 0.0])
    return np.asarray(pos, dtype=float)


def _truth_position(truth: dict) -> np.ndarray:
    """Return position as a numpy array from a truth dict."""
    pos = truth.get("position", [0.0, 0.0, 0.0])
    return np.asarray(pos, dtype=float)


# ---------------------------------------------------------------------------
# Metric computation functions
# ---------------------------------------------------------------------------

def compute_time_to_reacquire(frames: List[dict]) -> Tuple[Optional[float], Optional[float]]:
    """Compute mean and 95th-percentile time-to-reacquire across all track-loss events.

    A *track-loss event* begins when ``stale_steps > 0`` for a given track and
    ends when ``stale_steps`` returns to 0.  If reacquisition never occurs the
    event duration is capped at the mission duration (last frame timestamp).

    Args:
        frames: List of replay frame dicts, each containing a ``tracks`` list.

    Returns:
        Tuple of ``(mean_s, p95_s)`` or ``(None, None)`` when no loss events
        are found.
    """
    if not frames:
        return None, None

    # Build per-track stale_steps time series
    # Structure: {track_id: [(timestamp_s, stale_steps), ...]}
    track_stale: Dict[str, List[Tuple[float, int]]] = {}
    for frame in frames:
        ts = float(frame.get("timestamp_s", 0.0))
        for track in _extract_tracks_from_frame(frame):
            tid = track.get("track_id", "")
            ss = int(track.get("stale_steps", 0))
            track_stale.setdefault(tid, []).append((ts, ss))

    if not track_stale:
        return None, None

    mission_end_s = float(frames[-1].get("timestamp_s", 0.0))

    reacquire_durations: List[float] = []
    for tid, series in track_stale.items():
        loss_start: Optional[float] = None
        for ts, ss in series:
            if ss > 0 and loss_start is None:
                loss_start = ts
            elif ss == 0 and loss_start is not None:
                reacquire_durations.append(ts - loss_start)
                loss_start = None
        # If still stale at mission end, cap at remaining time
        if loss_start is not None:
            reacquire_durations.append(mission_end_s - loss_start)

    if not reacquire_durations:
        return None, None

    arr = np.array(reacquire_durations, dtype=float)
    mean_s = float(np.mean(arr))
    p95_s = float(np.percentile(arr, 95))
    return mean_s, p95_s


def compute_track_continuity(
    frames: List[dict],
    target_ids: List[str],
) -> Tuple[float, Dict[str, float]]:
    """Compute track-continuity fraction per target and mission-wide mean.

    Continuity for a target is the fraction of simulation steps in which that
    target has at least one *accepted* bearing observation.

    Args:
        frames: List of replay frame dicts with an ``observations`` list.
        target_ids: Canonical list of target IDs to evaluate.

    Returns:
        Tuple of ``(mean_continuity, per_target_dict)`` where ``per_target_dict``
        maps each target_id to its continuity fraction.
    """
    if not frames or not target_ids:
        return 0.0, {}

    total_steps = len(frames)
    steps_with_obs: Dict[str, int] = {tid: 0 for tid in target_ids}

    for frame in frames:
        observed_targets = set()
        for obs in _extract_observations_from_frame(frame):
            t = obs.get("target_id", "")
            if t:
                observed_targets.add(t)
        for tid in target_ids:
            if tid in observed_targets:
                steps_with_obs[tid] += 1

    per_target: Dict[str, float] = {}
    for tid in target_ids:
        per_target[tid] = steps_with_obs[tid] / total_steps if total_steps > 0 else 0.0

    mean_continuity = float(np.mean(list(per_target.values()))) if per_target else 0.0
    return mean_continuity, per_target


def compute_localisation_rmse(
    frames: List[dict],
) -> Tuple[Optional[float], Dict[str, float]]:
    """Compute localisation RMSE (metres) per track and mission-wide.

    Track-to-truth matching uses minimum Euclidean distance at the *first*
    frame that contains both a track and at least one truth.  Once matched,
    the same pairing is used for the entire replay.

    Args:
        frames: List of replay frame dicts with ``tracks`` and ``truths`` lists.

    Returns:
        Tuple of ``(mission_rmse, per_track_dict)`` where ``per_track_dict``
        maps track_id to that track's RMSE.  Returns ``(None, {})`` when no
        matching data exists.
    """
    if not frames:
        return None, {}

    # Determine track-to-truth mapping at first frame with both
    track_to_truth: Dict[str, str] = {}
    for frame in frames:
        tracks = _extract_tracks_from_frame(frame)
        truths = _extract_truths_from_frame(frame)
        if not tracks or not truths:
            continue
        for track in tracks:
            tid = track.get("track_id", "")
            if tid in track_to_truth:
                continue
            tp = _track_position(track)
            best_dist = float("inf")
            best_target = None
            for truth in truths:
                tp2 = _truth_position(truth)
                d = float(np.linalg.norm(tp - tp2))
                if d < best_dist:
                    best_dist = d
                    best_target = truth.get("target_id", "")
            if best_target is not None:
                track_to_truth[tid] = best_target
        # We only need the first frame that has data
        if track_to_truth:
            break

    if not track_to_truth:
        return None, {}

    # Accumulate squared errors per track
    sq_errors: Dict[str, List[float]] = {tid: [] for tid in track_to_truth}

    for frame in frames:
        tracks = _extract_tracks_from_frame(frame)
        truths = _extract_truths_from_frame(frame)
        truth_by_id: Dict[str, np.ndarray] = {
            t.get("target_id", ""): _truth_position(t) for t in truths
        }
        for track in tracks:
            tid = track.get("track_id", "")
            if tid not in track_to_truth:
                continue
            target_id = track_to_truth[tid]
            if target_id not in truth_by_id:
                continue
            tp = _track_position(track)
            err = float(np.linalg.norm(tp - truth_by_id[target_id]))
            sq_errors[tid].append(err ** 2)

    per_track: Dict[str, float] = {}
    all_sq: List[float] = []
    for tid, sq in sq_errors.items():
        if sq:
            per_track[tid] = float(np.sqrt(np.mean(sq)))
            all_sq.extend(sq)

    if not all_sq:
        return None, {}

    mission_rmse = float(np.sqrt(np.mean(all_sq)))
    return mission_rmse, per_track


def compute_covariance_reduction(frames: List[dict]) -> Optional[float]:
    """Compute mean covariance reduction fraction across tracks.

    Reduction is ``(trace(P_initial) - trace(P_final)) / trace(P_initial)``
    where ``P`` is the 3×3 position block (first 9 elements of the flattened
    covariance array).  Only tracks that received ``>= 5`` updates are included.

    Args:
        frames: List of replay frame dicts.

    Returns:
        Mean covariance reduction fraction, or ``None`` when no eligible tracks
        exist.
    """
    if not frames:
        return None

    # Collect first and last covariance + update_count per track
    track_first_cov: Dict[str, np.ndarray] = {}
    track_last_cov: Dict[str, np.ndarray] = {}
    track_update_count: Dict[str, int] = {}

    for frame in frames:
        for track in _extract_tracks_from_frame(frame):
            tid = track.get("track_id", "")
            cov_raw = track.get("covariance")
            if cov_raw is None:
                continue
            cov_arr = np.asarray(cov_raw, dtype=float).flatten()
            if cov_arr.size < 9:
                continue
            uc = int(track.get("update_count", 0))
            if tid not in track_first_cov:
                track_first_cov[tid] = cov_arr
            track_last_cov[tid] = cov_arr
            track_update_count[tid] = uc

    reductions: List[float] = []
    for tid in track_first_cov:
        if track_update_count.get(tid, 0) < 5:
            continue
        p_init = track_first_cov[tid][:9].reshape(3, 3)
        p_final = track_last_cov[tid][:9].reshape(3, 3)
        tr_init = float(np.trace(p_init))
        tr_final = float(np.trace(p_final))
        if tr_init <= 0.0:
            continue
        reductions.append((tr_init - tr_final) / tr_init)

    if not reductions:
        return None

    return float(np.mean(reductions))


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------

def evaluate_replay(
    replay_doc: dict,
    scenario_name: str,
    mission_type: str,
    difficulty: float,
    seed: int,
) -> EvaluationReport:
    """Evaluate a replay document and return an ``EvaluationReport``.

    This is the primary entry point for the evaluation harness.

    Args:
        replay_doc: Replay document dict as produced by
            ``replay.build_replay_document()``.
        scenario_name: Name of the scenario (e.g. ``"baseline_small_1t"``).
        mission_type: Mission type string (e.g. ``"surveillance"``).
        difficulty: Difficulty scalar in [0, 1].
        seed: RNG seed used for the simulation run.

    Returns:
        An :class:`EvaluationReport` with all computable metrics populated.
        Call :func:`check_pass_fail` on the result to populate ``passed`` and
        ``failure_reasons``.
    """
    frames: List[dict] = replay_doc.get("frames") or []
    meta: dict = replay_doc.get("meta") or {}

    # --- duration ---
    if frames:
        duration_s = float(frames[-1].get("timestamp_s", 0.0))
    else:
        duration_s = float(meta.get("duration_s", 0.0))

    # --- collect all target and track IDs ---
    target_ids: List[str] = sorted(
        {
            t.get("target_id", "")
            for frame in frames
            for t in _extract_truths_from_frame(frame)
            if t.get("target_id")
        }
    )

    # -----------------------------------------------------------------------
    # Track performance metrics
    # -----------------------------------------------------------------------
    ttr_mean, ttr_p95 = compute_time_to_reacquire(frames)
    cont_mean, cont_per_target = compute_track_continuity(frames, target_ids)
    loc_rmse, loc_per_track = compute_localisation_rmse(frames)
    cov_reduction = compute_covariance_reduction(frames)

    # -----------------------------------------------------------------------
    # Reliability metrics — sourced from replay metadata where available
    # -----------------------------------------------------------------------
    planner_events: List[dict] = replay_doc.get("planner_events") or []
    infeasible_count = sum(
        1 for ev in planner_events if ev.get("event_type") in ("plan_rejected",)
    )
    safety_override_count = sum(
        1 for ev in planner_events if ev.get("event_type") == "safety_override"
    )

    # false_handoff_rate: stored in replay metadata if available, else 0
    false_handoff_rate = float(
        (replay_doc.get("evaluation") or {}).get("false_handoff_rate", 0.0)
    )

    # comms dropout from metadata
    comms_dropout_count = int(
        (replay_doc.get("evaluation") or {}).get("comms_dropout_count", 0)
    )
    comms_dropout_duration_s = float(
        (replay_doc.get("evaluation") or {}).get("comms_dropout_duration_s", 0.0)
    )

    # -----------------------------------------------------------------------
    # Mission outcome metrics
    # -----------------------------------------------------------------------
    # Energy reserves — computed from mobile node trajectories when available
    energy_per_drone: Dict[str, float] = _compute_energy_reserves(frames)
    energy_min = float(min(energy_per_drone.values())) if energy_per_drone else 1.0

    # Mission completion — use metadata if present, else default to 1.0 if we
    # have no objectives info (unknown objectives are treated as all met).
    mission_meta = replay_doc.get("mission") or {}
    required_total = int(mission_meta.get("required_objectives_total", 0))
    required_met = int(mission_meta.get("required_objectives_met", required_total))
    mission_completion_rate: float
    if required_total > 0:
        mission_completion_rate = float(required_met) / float(required_total)
    else:
        mission_completion_rate = 1.0

    # -----------------------------------------------------------------------
    # Assemble report (pass/fail is determined separately)
    # -----------------------------------------------------------------------
    now_utc = datetime.now(timezone.utc).isoformat()

    return EvaluationReport(
        scenario_name=scenario_name,
        mission_type=mission_type,
        difficulty=difficulty,
        seed=seed,
        duration_s=duration_s,
        # Track performance
        time_to_reacquire_mean_s=ttr_mean,
        time_to_reacquire_p95_s=ttr_p95,
        track_continuity_mean=cont_mean,
        track_continuity_per_target=cont_per_target,
        localisation_rmse_m=loc_rmse,
        localisation_rmse_per_track=loc_per_track,
        covariance_reduction_mean=cov_reduction,
        # Reliability
        false_handoff_rate=false_handoff_rate,
        infeasible_path_rejection_count=infeasible_count,
        safety_override_count=safety_override_count,
        comms_dropout_count=comms_dropout_count,
        comms_dropout_duration_s=comms_dropout_duration_s,
        # Mission outcome
        mission_completion_rate=mission_completion_rate,
        required_objectives_met=required_met,
        required_objectives_total=required_total,
        energy_reserve_min=energy_min,
        energy_reserve_per_drone=energy_per_drone,
        # Summary (pass/fail computed separately)
        passed=False,
        failure_reasons=[],
        tags=[],
        generated_at_utc=now_utc,
    )


def _compute_energy_reserves(frames: List[dict]) -> Dict[str, float]:
    """Estimate energy reserves for each mobile drone from position time series.

    Uses the linear depletion model from SCENARIOS.md:
        energy_reserve = 1.0 - (total_distance_m / (speed_mps * max_endurance_s))

    Default ``max_endurance_s = 600 s``.  Speed is estimated from average
    velocity magnitude across frames.

    Returns an empty dict when no mobile nodes are found.
    """
    MAX_ENDURANCE_S = 600.0  # seconds (baseline platform parameter)

    # Collect position time series per mobile node
    node_positions: Dict[str, List[Tuple[float, np.ndarray]]] = {}
    for frame in frames:
        ts = float(frame.get("timestamp_s", 0.0))
        for node in (frame.get("nodes") or []):
            if not node.get("is_mobile", False):
                continue
            nid = node.get("node_id", "")
            pos = np.asarray(node.get("position", [0.0, 0.0, 0.0]), dtype=float)
            node_positions.setdefault(nid, []).append((ts, pos))

    if not node_positions:
        return {}

    energy_per_drone: Dict[str, float] = {}
    for nid, series in node_positions.items():
        if len(series) < 2:
            energy_per_drone[nid] = 1.0
            continue
        total_dist = 0.0
        speeds: List[float] = []
        for i in range(1, len(series)):
            dt = series[i][0] - series[i - 1][0]
            dp = np.linalg.norm(series[i][1] - series[i - 1][1])
            total_dist += float(dp)
            if dt > 0.0:
                speeds.append(float(dp) / float(dt))

        avg_speed = float(np.mean(speeds)) if speeds else 1.0
        if avg_speed <= 0.0:
            avg_speed = 1.0
        max_range_m = avg_speed * MAX_ENDURANCE_S
        reserve = 1.0 - (total_dist / max_range_m)
        energy_per_drone[nid] = max(0.0, min(1.0, reserve))

    return energy_per_drone


# ---------------------------------------------------------------------------
# Pass / fail evaluation
# ---------------------------------------------------------------------------

def check_pass_fail(
    report: EvaluationReport,
    thresholds: Optional[Dict[str, object]] = None,
) -> EvaluationReport:
    """Evaluate pass/fail criteria and return an updated ``EvaluationReport``.

    Uses default thresholds from SCENARIOS.md Section 1 unless ``thresholds``
    is provided (partial override is supported — missing keys fall back to
    defaults).

    Args:
        report: The report to evaluate.
        thresholds: Optional dict of threshold overrides.  Keys must match
            entries in :data:`DEFAULT_THRESHOLDS`.

    Returns:
        A new :class:`EvaluationReport` with ``passed`` and ``failure_reasons``
        populated.
    """
    t: Dict[str, object] = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        t.update(thresholds)

    failures: List[str] = []

    # --- time to reacquire ---
    ttr_mean = report.time_to_reacquire_mean_s
    ttr_fail = float(t["time_to_reacquire_mean_s_fail"])  # type: ignore[arg-type]
    if ttr_mean is not None and ttr_mean > ttr_fail:
        failures.append(
            f"time_to_reacquire_mean_s={ttr_mean:.1f} > {ttr_fail:.1f} (hard fail)"
        )

    # --- track continuity ---
    cont_min = float(t["track_continuity_mean_min"])  # type: ignore[arg-type]
    if report.track_continuity_mean < cont_min:
        failures.append(
            f"track_continuity_mean={report.track_continuity_mean:.3f} < {cont_min:.2f}"
        )

    # --- localisation RMSE ---
    rmse_fail = float(t["localisation_rmse_m_fail"])  # type: ignore[arg-type]
    if report.localisation_rmse_m is not None and report.localisation_rmse_m > rmse_fail:
        failures.append(
            f"localisation_rmse_m={report.localisation_rmse_m:.1f} > {rmse_fail:.1f} (hard fail)"
        )

    # --- false handoff rate ---
    fhr_max = float(t["false_handoff_rate_max"])  # type: ignore[arg-type]
    if report.false_handoff_rate > fhr_max:
        failures.append(
            f"false_handoff_rate={report.false_handoff_rate:.3f} > {fhr_max:.2f}"
        )

    # --- safety overrides ---
    so_max = int(t["safety_override_count_max"])  # type: ignore[arg-type]
    if report.safety_override_count > so_max:
        failures.append(
            f"safety_override_count={report.safety_override_count} > {so_max}"
        )

    # --- comms dropout duration ---
    cdd_max = float(t["comms_dropout_duration_s_max"])  # type: ignore[arg-type]
    if report.comms_dropout_duration_s > cdd_max:
        failures.append(
            f"comms_dropout_duration_s={report.comms_dropout_duration_s:.1f} > {cdd_max:.1f}"
        )

    # --- mission completion ---
    mcr_min = float(t["mission_completion_rate_min"])  # type: ignore[arg-type]
    if report.mission_completion_rate < mcr_min:
        failures.append(
            f"mission_completion_rate={report.mission_completion_rate:.3f} < {mcr_min:.1f}"
        )

    # --- energy reserve ---
    er_min = float(t["energy_reserve_min_min"])  # type: ignore[arg-type]
    if report.energy_reserve_min < er_min:
        failures.append(
            f"energy_reserve_min={report.energy_reserve_min:.3f} < {er_min:.2f}"
        )

    passed = len(failures) == 0

    # Use object.__setattr__ is not needed for frozen dataclasses when creating
    # a new instance via dataclass replace — but we create a new instance
    # manually so we can update passed and failure_reasons.
    return EvaluationReport(
        scenario_name=report.scenario_name,
        mission_type=report.mission_type,
        difficulty=report.difficulty,
        seed=report.seed,
        duration_s=report.duration_s,
        time_to_reacquire_mean_s=report.time_to_reacquire_mean_s,
        time_to_reacquire_p95_s=report.time_to_reacquire_p95_s,
        track_continuity_mean=report.track_continuity_mean,
        track_continuity_per_target=report.track_continuity_per_target,
        localisation_rmse_m=report.localisation_rmse_m,
        localisation_rmse_per_track=report.localisation_rmse_per_track,
        covariance_reduction_mean=report.covariance_reduction_mean,
        false_handoff_rate=report.false_handoff_rate,
        infeasible_path_rejection_count=report.infeasible_path_rejection_count,
        safety_override_count=report.safety_override_count,
        comms_dropout_count=report.comms_dropout_count,
        comms_dropout_duration_s=report.comms_dropout_duration_s,
        mission_completion_rate=report.mission_completion_rate,
        required_objectives_met=report.required_objectives_met,
        required_objectives_total=report.required_objectives_total,
        energy_reserve_min=report.energy_reserve_min,
        energy_reserve_per_drone=report.energy_reserve_per_drone,
        passed=passed,
        failure_reasons=failures,
        tags=report.tags,
        generated_at_utc=report.generated_at_utc,
    )


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def report_to_dict(report: EvaluationReport) -> dict:
    """Serialise an :class:`EvaluationReport` to a JSON-serialisable dict.

    All numpy scalars are converted to plain Python floats/ints.

    Args:
        report: The report to serialise.

    Returns:
        A plain dict suitable for ``json.dumps()``.
    """
    return {
        "scenario_name": report.scenario_name,
        "mission_type": report.mission_type,
        "difficulty": float(report.difficulty),
        "seed": int(report.seed),
        "duration_s": float(report.duration_s),
        "time_to_reacquire_mean_s": (
            float(report.time_to_reacquire_mean_s)
            if report.time_to_reacquire_mean_s is not None
            else None
        ),
        "time_to_reacquire_p95_s": (
            float(report.time_to_reacquire_p95_s)
            if report.time_to_reacquire_p95_s is not None
            else None
        ),
        "track_continuity_mean": float(report.track_continuity_mean),
        "track_continuity_per_target": {
            k: float(v) for k, v in report.track_continuity_per_target.items()
        },
        "localisation_rmse_m": (
            float(report.localisation_rmse_m)
            if report.localisation_rmse_m is not None
            else None
        ),
        "localisation_rmse_per_track": {
            k: float(v) for k, v in report.localisation_rmse_per_track.items()
        },
        "covariance_reduction_mean": (
            float(report.covariance_reduction_mean)
            if report.covariance_reduction_mean is not None
            else None
        ),
        "false_handoff_rate": float(report.false_handoff_rate),
        "infeasible_path_rejection_count": int(report.infeasible_path_rejection_count),
        "safety_override_count": int(report.safety_override_count),
        "comms_dropout_count": int(report.comms_dropout_count),
        "comms_dropout_duration_s": float(report.comms_dropout_duration_s),
        "mission_completion_rate": float(report.mission_completion_rate),
        "required_objectives_met": int(report.required_objectives_met),
        "required_objectives_total": int(report.required_objectives_total),
        "energy_reserve_min": float(report.energy_reserve_min),
        "energy_reserve_per_drone": {
            k: float(v) for k, v in report.energy_reserve_per_drone.items()
        },
        "passed": bool(report.passed),
        "failure_reasons": list(report.failure_reasons),
        "tags": list(report.tags),
        "generated_at_utc": report.generated_at_utc,
    }


def report_from_dict(d: dict) -> EvaluationReport:
    """Deserialise an :class:`EvaluationReport` from a plain dict.

    Missing optional fields are filled with their default values.

    Args:
        d: Dict as returned by :func:`report_to_dict` or loaded from JSON.

    Returns:
        An :class:`EvaluationReport` instance.
    """
    return EvaluationReport(
        scenario_name=str(d["scenario_name"]),
        mission_type=str(d["mission_type"]),
        difficulty=float(d["difficulty"]),
        seed=int(d["seed"]),
        duration_s=float(d["duration_s"]),
        time_to_reacquire_mean_s=(
            float(d["time_to_reacquire_mean_s"])
            if d.get("time_to_reacquire_mean_s") is not None
            else None
        ),
        time_to_reacquire_p95_s=(
            float(d["time_to_reacquire_p95_s"])
            if d.get("time_to_reacquire_p95_s") is not None
            else None
        ),
        track_continuity_mean=float(d.get("track_continuity_mean", 0.0)),
        track_continuity_per_target=dict(d.get("track_continuity_per_target") or {}),
        localisation_rmse_m=(
            float(d["localisation_rmse_m"])
            if d.get("localisation_rmse_m") is not None
            else None
        ),
        localisation_rmse_per_track=dict(d.get("localisation_rmse_per_track") or {}),
        covariance_reduction_mean=(
            float(d["covariance_reduction_mean"])
            if d.get("covariance_reduction_mean") is not None
            else None
        ),
        false_handoff_rate=float(d.get("false_handoff_rate", 0.0)),
        infeasible_path_rejection_count=int(d.get("infeasible_path_rejection_count", 0)),
        safety_override_count=int(d.get("safety_override_count", 0)),
        comms_dropout_count=int(d.get("comms_dropout_count", 0)),
        comms_dropout_duration_s=float(d.get("comms_dropout_duration_s", 0.0)),
        mission_completion_rate=float(d.get("mission_completion_rate", 0.0)),
        required_objectives_met=int(d.get("required_objectives_met", 0)),
        required_objectives_total=int(d.get("required_objectives_total", 0)),
        energy_reserve_min=float(d.get("energy_reserve_min", 1.0)),
        energy_reserve_per_drone=dict(d.get("energy_reserve_per_drone") or {}),
        passed=bool(d.get("passed", False)),
        failure_reasons=list(d.get("failure_reasons") or []),
        tags=list(d.get("tags") or []),
        generated_at_utc=str(d.get("generated_at_utc", "")),
    )
