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

import math
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone

import numpy as np

# ---------------------------------------------------------------------------
# Default pass/fail thresholds (from SCENARIOS.md Section 1)
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLDS: dict[str, object] = {
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
    time_to_reacquire_mean_s: float | None = None
    time_to_reacquire_p95_s: float | None = None
    track_continuity_mean: float = 0.0
    track_continuity_per_target: dict[str, float] = field(default_factory=dict)
    localisation_rmse_m: float | None = None
    localisation_rmse_per_track: dict[str, float] = field(default_factory=dict)
    # Each track against truth at the frame time, whatever its evidence time
    # (B4): what a consumer of the published track experiences. Equal to
    # `localisation_rmse_m` when tracks are fused at frame time.
    present_error_rmse_m: float | None = None
    # Mean of (frame time - track time) over every scored track sample: how
    # old the evidence behind the published tracks is. 0 without delivery.
    track_evidence_age_mean_s: float | None = None
    # Mean of (frame time - the track's last measurement time) over every
    # track that records one (`evidence_time_s`, P4.10b): the evidence age
    # above plus any coasting since. None when no track records it (a legacy
    # replay).
    track_measurement_age_mean_s: float | None = None
    covariance_reduction_mean: float | None = None
    # Tracks behind covariance_reduction_mean; 0 when it is None.
    covariance_reduction_tracks: int = 0

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
    energy_reserve_per_drone: dict[str, float] = field(default_factory=dict)
    # The battery ledger's own reading: the lowest `battery_fraction` any
    # drone reached in the replay. None when the replay carries no battery
    # (older replays, or a run without a ledger). `energy_reserve_min` above
    # is a distance-based estimate and does not see hover or the profile.
    battery_remaining_min: float | None = None

    # Assignment history (P0): from `meta.assignment_history`, which the Rust
    # writer records; None for a replay without one (the Python reference).
    # Every change of a drone's task after its first; POIs a drone handed to
    # another; and thrash, a change within THRASH_WINDOW_S of that drone's
    # previous change (its first task is not a change, and leaving a POI it
    # completed restarts the window: see `assignment_counts`).
    task_changes: int | None = None
    poi_handoffs: int | None = None
    assignment_thrash: int | None = None

    # Time and effort (P0), reporting only. Makespan: from the first frame
    # to the last POI completed, when every POI was; None otherwise or
    # without POIs. Scan time: the first frame at the scan's coverage
    # threshold. POI service time: from the first time a drone was on the
    # POI (assignment history) to its completion, averaged over completed
    # POIs. Distance: the path each mobile node flew. Battery used: its first
    # minus its last `battery_fraction`. Idle: time drones spent on `hold`.
    mission_makespan_s: float | None = None
    scan_coverage_time_s: float | None = None
    poi_service_time_mean_s: float | None = None
    fleet_distance_m: float = 0.0
    distance_per_drone_m: dict[str, float] = field(default_factory=dict)
    battery_used_per_drone: dict[str, float] = field(default_factory=dict)
    idle_time_s: float | None = None

    # Performance matrix
    wall_clock_s: float | None = None
    frame_time_mean_ms: float | None = None
    frame_time_p95_ms: float | None = None
    frame_time_p99_ms: float | None = None
    peak_rss_mb: float | None = None
    replay_size_mb: float | None = None
    terrain_queries_per_sec: float | None = None
    observations_per_sec: float | None = None

    # Summary
    passed: bool = False
    failure_reasons: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    generated_at_utc: str = ""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_tracks_from_frame(frame: dict) -> list[dict]:
    """Return the tracks list from a replay frame dict."""
    return frame.get("tracks") or []


def _extract_truths_from_frame(frame: dict) -> list[dict]:
    """Return the truths list from a replay frame dict."""
    return frame.get("truths") or []


def _extract_observations_from_frame(frame: dict) -> list[dict]:
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


def compute_time_to_reacquire(frames: list[dict]) -> tuple[float | None, float | None]:
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
    track_stale: dict[str, list[tuple[float, int]]] = {}
    for frame in frames:
        ts = float(frame.get("timestamp_s", 0.0))
        for track in _extract_tracks_from_frame(frame):
            tid = track.get("track_id", "")
            ss = int(track.get("stale_steps", 0))
            track_stale.setdefault(tid, []).append((ts, ss))

    if not track_stale:
        return None, None

    mission_end_s = float(frames[-1].get("timestamp_s", 0.0))

    reacquire_durations: list[float] = []
    for _tid, series in track_stale.items():
        loss_start: float | None = None
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
    frames: list[dict],
    target_ids: list[str],
) -> tuple[float, dict[str, float]]:
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
    steps_with_obs: dict[str, int] = {tid: 0 for tid in target_ids}

    for frame in frames:
        observed_targets = set()
        for obs in _extract_observations_from_frame(frame):
            t = obs.get("target_id", "")
            if t:
                observed_targets.add(t)
        for tid in target_ids:
            if tid in observed_targets:
                steps_with_obs[tid] += 1

    per_target: dict[str, float] = {}
    for tid in target_ids:
        per_target[tid] = steps_with_obs[tid] / total_steps if total_steps > 0 else 0.0

    mean_continuity = float(np.mean(list(per_target.values()))) if per_target else 0.0
    return mean_continuity, per_target


def _truth_at_track_time(frames: list[dict]):
    """A lookup of truth positions by time, for scoring a track at its own
    timestamp. A fixed-lag run (B4) publishes tracks fused at the evidence
    time, earlier than the frame; the estimate is judged against where the
    target was then. Times resolve to the nearest frame within half a frame
    period; anything else falls back to the frame's own truths."""
    times = np.asarray([float(f.get("timestamp_s", 0.0)) for f in frames])
    period = float(np.median(np.diff(times))) if len(times) > 1 else 0.0
    by_index: list[dict[str, np.ndarray] | None] = [None] * len(frames)

    def truths(index: int) -> dict[str, np.ndarray]:
        cached = by_index[index]
        if cached is None:
            cached = {
                t.get("target_id", ""): _truth_position(t)
                for t in _extract_truths_from_frame(frames[index])
            }
            by_index[index] = cached
        return cached

    def lookup(frame_index: int, track: dict) -> dict[str, np.ndarray]:
        stamp = track.get("timestamp_s")
        frame_time = times[frame_index]
        if stamp is None or period <= 0.0 or abs(float(stamp) - frame_time) <= 1e-9:
            return truths(frame_index)
        nearest = int(np.argmin(np.abs(times[: frame_index + 1] - float(stamp))))
        if abs(times[nearest] - float(stamp)) <= period / 2.0:
            return truths(nearest)
        return truths(frame_index)

    return times, truths, lookup


def compute_localisation_rmse(
    frames: list[dict],
    at_evidence_time: bool = True,
) -> tuple[float | None, dict[str, float]]:
    """Compute localisation RMSE (metres) per track and mission-wide.

    With ``at_evidence_time`` (the default), a track is scored against truth
    at its own ``timestamp_s``: a fixed-lag run publishes tracks fused at the
    evidence time, and this is the estimator's error. Without it, against
    the frame's truths: the error of the track taken as describing the
    present. The two agree for tracks fused at frame time.

    Each track is matched to a truth once, in the first frame where it
    appears alongside at least one truth, and keeps that pairing for the
    rest of the replay. A track whose ID names a truth target (the fusion
    service keys tracks by the observations' target ID) is paired with that
    target; any other track with the nearest truth at that moment.

    Args:
        frames: List of replay frame dicts with ``tracks`` and ``truths`` lists.

    Returns:
        Tuple of ``(mission_rmse, per_track_dict)`` where ``per_track_dict``
        maps track_id to that track's RMSE.  Returns ``(None, {})`` when no
        matching data exists.
    """
    if not frames:
        return None, {}

    # Until P4.9c every track was matched in the first frame holding any
    # track, and by distance alone: tracks confirmed later were never scored,
    # and a track that happened to pass another target at that instant was
    # scored against it for the whole run.
    track_to_truth: dict[str, str] = {}
    for frame in frames:
        tracks = _extract_tracks_from_frame(frame)
        truths = _extract_truths_from_frame(frame)
        if not tracks or not truths:
            continue
        truth_ids = {truth.get("target_id", "") for truth in truths}
        for track in tracks:
            tid = track.get("track_id", "")
            if tid in track_to_truth:
                continue
            if tid in truth_ids:
                track_to_truth[tid] = tid
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

    if not track_to_truth:
        return None, {}

    # Accumulate squared errors per track
    sq_errors: dict[str, list[float]] = {tid: [] for tid in track_to_truth}
    _, frame_truths, truths_for = _truth_at_track_time(frames)

    for index, frame in enumerate(frames):
        tracks = _extract_tracks_from_frame(frame)
        for track in tracks:
            tid = track.get("track_id", "")
            if tid not in track_to_truth:
                continue
            truth_by_id = truths_for(index, track) if at_evidence_time else frame_truths(index)
            target_id = track_to_truth[tid]
            if target_id not in truth_by_id:
                continue
            tp = _track_position(track)
            err = float(np.linalg.norm(tp - truth_by_id[target_id]))
            sq_errors[tid].append(err**2)

    per_track: dict[str, float] = {}
    all_sq: list[float] = []
    for tid, sq in sq_errors.items():
        if sq:
            per_track[tid] = float(np.sqrt(np.mean(sq)))
            all_sq.extend(sq)

    if not all_sq:
        return None, {}

    mission_rmse = float(np.sqrt(np.mean(all_sq)))
    return mission_rmse, per_track


def compute_track_measurement_age(frames: list[dict]) -> float | None:
    """Mean of (frame time - the track's ``evidence_time_s``) over every track
    that records it: how long ago the published tracks last took a
    measurement. At least the evidence age (frame time - track time); the
    difference is the time tracks spent coasting. None when no track records
    an evidence time (the field is written only by timed runs, P4.10b)."""
    ages = [
        float(frame.get("timestamp_s", 0.0)) - float(track["evidence_time_s"])
        for frame in frames
        for track in _extract_tracks_from_frame(frame)
        if track.get("evidence_time_s") is not None
    ]
    return float(np.mean(ages)) if ages else None


def compute_track_evidence_age(frames: list[dict]) -> float | None:
    """Mean of (frame time - track time) over every track in every frame:
    how old the evidence behind the published tracks is. None without
    tracks."""
    ages = [
        float(frame.get("timestamp_s", 0.0)) - float(track["timestamp_s"])
        for frame in frames
        for track in _extract_tracks_from_frame(frame)
        if track.get("timestamp_s") is not None
    ]
    return float(np.mean(ages)) if ages else None


COVARIANCE_REDUCTION_MIN_UPDATES = 5
"""A track contributes to covariance reduction only after this many updates."""


def compute_covariance_reductions(frames: list[dict]) -> list[float]:
    """Per-track covariance reduction fractions, one per qualifying track.

    Reduction is ``(trace(P_initial) - trace(P_final)) / trace(P_initial)``
    where ``P`` is the 3×3 position block (first 9 elements of the flattened
    covariance array). ``P_initial`` is the covariance at the track's first
    appearance. ``P_final`` is the covariance at the track's **last update**,
    the last frame in which its ``update_count`` rose, not its last
    appearance: a track coasting towards deletion keeps appearing with a
    growing covariance, and that coast measures the deletion policy rather
    than the filter. Only tracks that received at least
    ``COVARIANCE_REDUCTION_MIN_UPDATES`` updates are included.

    Args:
        frames: List of replay frame dicts.

    Returns:
        One reduction fraction per qualifying track, in first-appearance order.
    """
    track_first_cov: dict[str, np.ndarray] = {}
    track_last_update_cov: dict[str, np.ndarray] = {}
    track_update_count: dict[str, int] = {}

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
                track_last_update_cov[tid] = cov_arr
                track_update_count[tid] = uc
            elif uc > track_update_count[tid]:
                track_last_update_cov[tid] = cov_arr
                track_update_count[tid] = uc

    reductions: list[float] = []
    for tid in track_first_cov:
        if track_update_count.get(tid, 0) < COVARIANCE_REDUCTION_MIN_UPDATES:
            continue
        p_init = track_first_cov[tid][:9].reshape(3, 3)
        p_final = track_last_update_cov[tid][:9].reshape(3, 3)
        tr_init = float(np.trace(p_init))
        tr_final = float(np.trace(p_final))
        if tr_init <= 0.0:
            continue
        reductions.append((tr_init - tr_final) / tr_init)
    return reductions


def compute_covariance_reduction(frames: list[dict]) -> float | None:
    """Mean of :func:`compute_covariance_reductions`, or ``None`` with no
    qualifying track."""
    reductions = compute_covariance_reductions(frames)
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
    frames: list[dict] = replay_doc.get("frames") or []
    meta: dict = replay_doc.get("meta") or {}

    # --- duration ---
    if frames:
        duration_s = float(frames[-1].get("timestamp_s", 0.0))
    else:
        duration_s = float(meta.get("duration_s", 0.0))

    # --- collect all target and track IDs ---
    target_ids: list[str] = sorted(
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
    present_rmse, _ = compute_localisation_rmse(frames, at_evidence_time=False)
    cov_reductions = compute_covariance_reductions(frames)
    cov_reduction = float(np.mean(cov_reductions)) if cov_reductions else None

    # -----------------------------------------------------------------------
    # Reliability metrics — sourced from replay metadata where available
    # -----------------------------------------------------------------------
    planner_events: list[dict] = replay_doc.get("planner_events") or []
    infeasible_count = sum(1 for ev in planner_events if ev.get("event_type") in ("plan_rejected",))
    safety_override_count = sum(
        1 for ev in planner_events if ev.get("event_type") == "safety_override"
    )

    # false_handoff_rate: stored in replay metadata if available, else 0
    false_handoff_rate = float((replay_doc.get("evaluation") or {}).get("false_handoff_rate", 0.0))

    # comms dropout from metadata
    comms_dropout_count = int((replay_doc.get("evaluation") or {}).get("comms_dropout_count", 0))
    comms_dropout_duration_s = float(
        (replay_doc.get("evaluation") or {}).get("comms_dropout_duration_s", 0.0)
    )

    # -----------------------------------------------------------------------
    # Mission outcome metrics
    # -----------------------------------------------------------------------
    # Energy reserves — computed from mobile node trajectories when available
    energy_per_drone: dict[str, float] = _compute_energy_reserves(frames)
    energy_min = float(min(energy_per_drone.values())) if energy_per_drone else 1.0

    # Mission completion: the top-level `mission` summary if a producer writes
    # one; otherwise the POIs of the last frame's `scan_mission_state` (both
    # the Python and the Rust writers record them there). Only a run with
    # neither -- one with no objectives -- counts as all met.
    mission_meta = replay_doc.get("mission") or _poi_objectives(frames)
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
    performance_meta = meta.get("performance") if isinstance(meta, dict) else {}
    performance_meta = performance_meta if isinstance(performance_meta, dict) else {}
    frame_times = performance_meta.get("frame_times_ms")
    frame_times = frame_times if isinstance(frame_times, dict) else {}
    now_utc = datetime.now(timezone.utc).isoformat()
    assignments = assignment_counts(replay_doc)

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
        present_error_rmse_m=present_rmse,
        track_evidence_age_mean_s=compute_track_evidence_age(frames),
        track_measurement_age_mean_s=compute_track_measurement_age(frames),
        covariance_reduction_mean=cov_reduction,
        covariance_reduction_tracks=len(cov_reductions),
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
        battery_remaining_min=_battery_remaining_min(frames),
        task_changes=assignments[0] if assignments else None,
        poi_handoffs=assignments[1] if assignments else None,
        assignment_thrash=assignments[2] if assignments else None,
        **time_and_effort(replay_doc),
        # Performance matrix
        wall_clock_s=_optional_float(performance_meta.get("wall_clock_s")),
        frame_time_mean_ms=_optional_float(frame_times.get("mean_ms")),
        frame_time_p95_ms=_optional_float(frame_times.get("p95_ms")),
        frame_time_p99_ms=_optional_float(frame_times.get("p99_ms")),
        peak_rss_mb=_optional_float(performance_meta.get("peak_rss_mb")),
        replay_size_mb=_optional_float(performance_meta.get("replay_size_mb")),
        terrain_queries_per_sec=_optional_float(performance_meta.get("terrain_queries_per_sec")),
        observations_per_sec=_optional_float(performance_meta.get("observations_per_sec")),
        # Summary (pass/fail computed separately)
        passed=False,
        failure_reasons=[],
        tags=[],
        generated_at_utc=now_utc,
    )


THRASH_WINDOW_S = 30.0


def assignment_counts(replay_doc: dict) -> tuple[int, int, int] | None:
    """`(task_changes, poi_handoffs, assignment_thrash)` from the replay's
    `meta.assignment_history`, or None when it has none.

    The history lists each drone's events in time order: `assign` (its first
    task), `reassign`, `handoff` and `team_join`. A handoff leaves the giving
    drone's own task unchanged (the reference keeps it flying to the POI), so
    it is counted apart from task changes.

    Thrash is a change of mind, not a change of work: leaving a POI the
    replay reports completed by then is never thrash, and restarts the
    drone's window like its first task, so finishing one POI and taking the
    next is not counted either.
    """
    completions = _poi_completions(replay_doc.get("frames") or [])
    meta = replay_doc.get("meta")
    history = meta.get("assignment_history") if isinstance(meta, dict) else None
    if not isinstance(history, dict):
        return None
    changes = handoffs = thrash = 0
    for events in history.values():
        last_change: float | None = None
        for event in events or []:
            kind = event.get("kind")
            t = float(event.get("timestamp_s", 0.0))
            if kind == "handoff":
                handoffs += 1
            elif kind == "reassign":
                changes += 1
                previous = str(event.get("previous") or "")
                done = completions.get(previous.removeprefix("poi:"))
                if previous.startswith("poi:") and done is not None and done <= t:
                    last_change = None
                    continue
                if last_change is not None and t - last_change < THRASH_WINDOW_S:
                    thrash += 1
                last_change = t
    return changes, handoffs, thrash


def _poi_completions(frames: list) -> dict[str, float]:
    """Each completed POI's completion time, from the last scan mission state.

    A POI id that more than one POI shares is left out: the history names a
    task only by id, so it cannot tell which of them a drone left."""
    states = [
        f["scan_mission_state"] for f in frames if isinstance(f.get("scan_mission_state"), dict)
    ]
    statuses = states[-1].get("poi_statuses") or [] if states else []
    counts: dict[str, int] = {}
    for p in statuses:
        counts[str(p["poi_id"])] = counts.get(str(p["poi_id"]), 0) + 1
    return {
        str(p["poi_id"]): float(p["completion_time_s"])
        for p in statuses
        if p.get("completion_time_s") is not None and counts[str(p["poi_id"])] == 1
    }


def time_and_effort(replay_doc: dict) -> dict:
    """The P0 time and effort figures (see `EvaluationReport`), as a dict of
    the report's field names."""
    frames = replay_doc.get("frames") or []
    meta = replay_doc.get("meta")
    history = meta.get("assignment_history") if isinstance(meta, dict) else None
    history = history if isinstance(history, dict) else None
    start = float(frames[0].get("timestamp_s", 0.0)) if frames else 0.0
    end = float(frames[-1].get("timestamp_s", 0.0)) if frames else 0.0

    states = [
        f["scan_mission_state"] for f in frames if isinstance(f.get("scan_mission_state"), dict)
    ]
    statuses = states[-1].get("poi_statuses") or [] if states else []
    completions = _poi_completions(frames)
    # A replay resumed from a checkpoint starts mid-run: a POI completed, or
    # the scan's threshold reached, before its first frame happened outside
    # the window, so neither figure is available from it.
    makespan = (
        max(completions.values()) - start
        if statuses and len(completions) == len(statuses) and min(completions.values()) >= start
        else None
    )

    def covered(frame: dict) -> bool:
        state = frame.get("scan_mission_state")
        return isinstance(state, dict) and float(state.get("scan_coverage_fraction", 0.0)) >= float(
            state.get("scan_coverage_threshold", 1.0)
        )

    scan_time = (
        None
        if frames and covered(frames[0]) and start > 0.0
        else next((float(f["timestamp_s"]) - start for f in frames if covered(f)), None)
    )

    service: list[float] = []
    idle: float | None = None
    if history is not None:
        first_on: dict[str, float] = {}
        idle = 0.0
        for events in history.values():
            held_since: float | None = None
            for event in events or []:
                t = float(event.get("timestamp_s", 0.0))
                task = str(event.get("task", ""))
                # A drone already on a POI when a resumed replay begins was
                # assigned it earlier: its service time is unknown.
                observed = event.get("kind") == "reassign" or start <= 0.0
                if task.startswith("poi:") and observed and event.get("kind") != "handoff":
                    first_on[task[4:]] = min(first_on.get(task[4:], t), t)
                if event.get("kind") not in ("assign", "reassign"):
                    continue
                if held_since is not None:
                    idle += t - held_since
                    held_since = None
                if task == "hold":
                    held_since = t
            if held_since is not None:
                idle += end - held_since
        service = [done - first_on[poi] for poi, done in completions.items() if poi in first_on]

    previous: dict[str, list[float]] = {}
    distance: dict[str, float] = {}
    battery_first: dict[str, float] = {}
    battery_last: dict[str, float] = {}
    for frame in frames:
        for node in frame.get("nodes") or []:
            if not node.get("is_mobile"):
                continue
            node_id = str(node["node_id"])
            position = [float(v) for v in node.get("position", (0.0, 0.0, 0.0))]
            if node_id in previous:
                distance[node_id] = distance.get(node_id, 0.0) + math.dist(
                    previous[node_id], position
                )
            else:
                distance.setdefault(node_id, 0.0)
            previous[node_id] = position
            fraction = node.get("battery_fraction")
            if fraction is not None and float(fraction) >= 0.0:
                battery_first.setdefault(node_id, float(fraction))
                battery_last[node_id] = float(fraction)

    return {
        "mission_makespan_s": makespan,
        "scan_coverage_time_s": scan_time,
        "poi_service_time_mean_s": (sum(service) / len(service)) if service else None,
        "fleet_distance_m": float(sum(distance.values())),
        "distance_per_drone_m": distance,
        "battery_used_per_drone": {
            node: battery_first[node] - battery_last[node] for node in battery_first
        },
        "idle_time_s": idle,
    }


def _battery_remaining_min(frames: list[dict]) -> float | None:
    """The lowest battery fraction any mobile node reports in the replay.

    Reads the ledger's published `battery_fraction`; a negative value is the
    no-battery sentinel and is skipped.
    """
    lowest: float | None = None
    for frame in frames:
        for node in frame.get("nodes") or []:
            if not node.get("is_mobile", False):
                continue
            fraction = node.get("battery_fraction")
            if fraction is None or fraction < 0:
                continue
            lowest = float(fraction) if lowest is None else min(lowest, float(fraction))
    return lowest


def _poi_objectives(frames: list[dict]) -> dict[str, int]:
    """The inspection POIs as objectives, from the last frame that reports
    them: every POI is required, and a completed one is met. Empty when no
    frame carries a `scan_mission_state`."""
    for frame in reversed(frames):
        state = frame.get("scan_mission_state")
        if isinstance(state, dict) and "total_poi_count" in state:
            return {
                "required_objectives_total": int(state["total_poi_count"]),
                "required_objectives_met": int(state.get("completed_poi_count", 0)),
            }
    return {}


def _compute_energy_reserves(frames: list[dict]) -> dict[str, float]:
    """Estimate energy reserves for each mobile drone from position time series.

    Uses the linear depletion model from SCENARIOS.md:
        energy_reserve = 1.0 - (total_distance_m / (speed_mps * max_endurance_s))

    Default ``max_endurance_s = 600 s``.  Speed is estimated from average
    velocity magnitude across frames.

    Returns an empty dict when no mobile nodes are found.
    """
    MAX_ENDURANCE_S = 600.0  # seconds (baseline platform parameter)

    # Collect position time series per mobile node
    node_positions: dict[str, list[tuple[float, np.ndarray]]] = {}
    for frame in frames:
        ts = float(frame.get("timestamp_s", 0.0))
        for node in frame.get("nodes") or []:
            if not node.get("is_mobile", False):
                continue
            nid = node.get("node_id", "")
            pos = np.asarray(node.get("position", [0.0, 0.0, 0.0]), dtype=float)
            node_positions.setdefault(nid, []).append((ts, pos))

    if not node_positions:
        return {}

    energy_per_drone: dict[str, float] = {}
    for nid, series in node_positions.items():
        if len(series) < 2:
            energy_per_drone[nid] = 1.0
            continue
        total_dist = 0.0
        speeds: list[float] = []
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


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)  # type: ignore[call-overload]


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


# ---------------------------------------------------------------------------
# Pass / fail evaluation
# ---------------------------------------------------------------------------


def check_pass_fail(
    report: EvaluationReport,
    thresholds: dict[str, object] | None = None,
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
    t: dict[str, object] = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        t.update(thresholds)

    failures: list[str] = []

    # --- time to reacquire ---
    ttr_mean = report.time_to_reacquire_mean_s
    ttr_fail = float(t["time_to_reacquire_mean_s_fail"])  # type: ignore[arg-type]
    if ttr_mean is not None and ttr_mean > ttr_fail:
        failures.append(f"time_to_reacquire_mean_s={ttr_mean:.1f} > {ttr_fail:.1f} (hard fail)")

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
        failures.append(f"false_handoff_rate={report.false_handoff_rate:.3f} > {fhr_max:.2f}")

    # --- safety overrides ---
    so_max = int(t["safety_override_count_max"])  # type: ignore[arg-type]
    if report.safety_override_count > so_max:
        failures.append(f"safety_override_count={report.safety_override_count} > {so_max}")

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
        failures.append(f"energy_reserve_min={report.energy_reserve_min:.3f} < {er_min:.2f}")

    passed = len(failures) == 0

    return replace(report, passed=passed, failure_reasons=failures)


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _float_or_none(value: float | None) -> float | None:
    return None if value is None else float(value)


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
            float(report.localisation_rmse_m) if report.localisation_rmse_m is not None else None
        ),
        "present_error_rmse_m": _float_or_none(report.present_error_rmse_m),
        "track_evidence_age_mean_s": _float_or_none(report.track_evidence_age_mean_s),
        "track_measurement_age_mean_s": _float_or_none(report.track_measurement_age_mean_s),
        "localisation_rmse_per_track": {
            k: float(v) for k, v in report.localisation_rmse_per_track.items()
        },
        "covariance_reduction_mean": (
            float(report.covariance_reduction_mean)
            if report.covariance_reduction_mean is not None
            else None
        ),
        "covariance_reduction_tracks": int(report.covariance_reduction_tracks),
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
        "wall_clock_s": _float_or_none(report.wall_clock_s),
        "frame_time_mean_ms": _float_or_none(report.frame_time_mean_ms),
        "frame_time_p95_ms": _float_or_none(report.frame_time_p95_ms),
        "frame_time_p99_ms": _float_or_none(report.frame_time_p99_ms),
        "peak_rss_mb": _float_or_none(report.peak_rss_mb),
        "mission_makespan_s": _float_or_none(report.mission_makespan_s),
        "scan_coverage_time_s": _float_or_none(report.scan_coverage_time_s),
        "poi_service_time_mean_s": _float_or_none(report.poi_service_time_mean_s),
        "fleet_distance_m": float(report.fleet_distance_m),
        "distance_per_drone_m": {k: float(v) for k, v in report.distance_per_drone_m.items()},
        "battery_used_per_drone": {k: float(v) for k, v in report.battery_used_per_drone.items()},
        "idle_time_s": _float_or_none(report.idle_time_s),
        "task_changes": report.task_changes,
        "poi_handoffs": report.poi_handoffs,
        "assignment_thrash": report.assignment_thrash,
        "replay_size_mb": _float_or_none(report.replay_size_mb),
        "terrain_queries_per_sec": _float_or_none(report.terrain_queries_per_sec),
        "observations_per_sec": _float_or_none(report.observations_per_sec),
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
            float(d["localisation_rmse_m"]) if d.get("localisation_rmse_m") is not None else None
        ),
        localisation_rmse_per_track=dict(d.get("localisation_rmse_per_track") or {}),
        present_error_rmse_m=_optional_float(d.get("present_error_rmse_m")),
        track_evidence_age_mean_s=_optional_float(d.get("track_evidence_age_mean_s")),
        track_measurement_age_mean_s=_optional_float(d.get("track_measurement_age_mean_s")),
        covariance_reduction_mean=(
            float(d["covariance_reduction_mean"])
            if d.get("covariance_reduction_mean") is not None
            else None
        ),
        covariance_reduction_tracks=int(d.get("covariance_reduction_tracks", 0)),
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
        battery_remaining_min=(
            None if d.get("battery_remaining_min") is None else float(d["battery_remaining_min"])
        ),
        mission_makespan_s=_optional_float(d.get("mission_makespan_s")),
        scan_coverage_time_s=_optional_float(d.get("scan_coverage_time_s")),
        poi_service_time_mean_s=_optional_float(d.get("poi_service_time_mean_s")),
        fleet_distance_m=float(d.get("fleet_distance_m", 0.0)),
        distance_per_drone_m=dict(d.get("distance_per_drone_m") or {}),
        battery_used_per_drone=dict(d.get("battery_used_per_drone") or {}),
        idle_time_s=_optional_float(d.get("idle_time_s")),
        task_changes=_optional_int(d.get("task_changes")),
        poi_handoffs=_optional_int(d.get("poi_handoffs")),
        assignment_thrash=_optional_int(d.get("assignment_thrash")),
        wall_clock_s=_optional_float(d.get("wall_clock_s")),
        frame_time_mean_ms=_optional_float(d.get("frame_time_mean_ms")),
        frame_time_p95_ms=_optional_float(d.get("frame_time_p95_ms")),
        frame_time_p99_ms=_optional_float(d.get("frame_time_p99_ms")),
        peak_rss_mb=_optional_float(d.get("peak_rss_mb")),
        replay_size_mb=_optional_float(d.get("replay_size_mb")),
        terrain_queries_per_sec=_optional_float(d.get("terrain_queries_per_sec")),
        observations_per_sec=_optional_float(d.get("observations_per_sec")),
        passed=bool(d.get("passed", False)),
        failure_reasons=list(d.get("failure_reasons") or []),
        tags=list(d.get("tags") or []),
        generated_at_utc=str(d.get("generated_at_utc", "")),
    )
