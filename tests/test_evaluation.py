"""Tests for the evaluation harness (argusnet.evaluation.metrics).

Covers metric computation functions, the full evaluate_replay pipeline,
pass/fail checking, serialisation round-trips, and edge cases.
"""

from __future__ import annotations

import numpy as np
import pytest

from argusnet.evaluation.metrics import (
    EvaluationReport,
    check_pass_fail,
    compute_covariance_reduction,
    compute_localisation_rmse,
    compute_time_to_reacquire,
    compute_track_continuity,
    evaluate_replay,
    report_from_dict,
    report_to_dict,
)

# ---------------------------------------------------------------------------
# Helpers — synthetic frame / replay builders
# ---------------------------------------------------------------------------


def _make_frame(
    ts: float,
    tracks: list[dict] | None = None,
    truths: list[dict] | None = None,
    observations: list[dict] | None = None,
    nodes: list[dict] | None = None,
) -> dict:
    """Build a single replay frame dict."""
    f: dict = {"timestamp_s": ts}
    if tracks is not None:
        f["tracks"] = tracks
    if truths is not None:
        f["truths"] = truths
    if observations is not None:
        f["observations"] = observations
    if nodes is not None:
        f["nodes"] = nodes
    return f


def _make_track(
    track_id: str,
    position: list,
    stale_steps: int = 0,
    covariance: list | None = None,
    update_count: int = 0,
) -> dict:
    d: dict = {
        "track_id": track_id,
        "position": position,
        "stale_steps": stale_steps,
        "update_count": update_count,
    }
    if covariance is not None:
        d["covariance"] = covariance
    return d


def _identity_cov(scale: float = 1.0) -> list:
    """Return a flat 9-element list representing scale * I_{3x3}."""
    m = np.eye(3) * scale
    return m.flatten().tolist()


def _build_good_replay() -> dict:
    """Return a synthetic replay document that should pass all default thresholds."""
    frames: list[dict] = []
    n_frames = 20
    for i in range(n_frames):
        ts = float(i)
        frames.append(
            _make_frame(
                ts=ts,
                tracks=[
                    _make_track(
                        "t1",
                        [10.0 + i * 0.1, 20.0, 0.0],
                        stale_steps=0,
                        covariance=_identity_cov(max(10.0 - i * 0.4, 1.0)),
                        update_count=i + 1,
                    ),
                ],
                truths=[{"target_id": "tgt1", "position": [10.0 + i * 0.1, 20.0, 0.0]}],
                observations=[{"target_id": "tgt1"}],
                nodes=[{"node_id": "d0", "is_mobile": True, "position": [0.0, float(i), 0.0]}],
            )
        )

    return {
        "meta": {"duration_s": float(n_frames - 1)},
        "frames": frames,
        "planner_events": [],
        "evaluation": {
            "false_handoff_rate": 0.0,
            "comms_dropout_count": 0,
            "comms_dropout_duration_s": 0.0,
        },
        "mission": {"required_objectives_total": 1, "required_objectives_met": 1},
    }


# ===================================================================
# compute_time_to_reacquire
# ===================================================================


class TestTimeToReacquire:
    def test_empty_frames_returns_none(self):
        mean, p95 = compute_time_to_reacquire([])
        assert mean is None
        assert p95 is None

    def test_no_tracks_returns_none(self):
        frames = [_make_frame(0.0, tracks=[]), _make_frame(1.0, tracks=[])]
        mean, p95 = compute_time_to_reacquire(frames)
        assert mean is None
        assert p95 is None

    def test_no_stale_events_returns_none(self):
        frames = [
            _make_frame(0.0, tracks=[_make_track("t1", [0, 0, 0], stale_steps=0)]),
            _make_frame(1.0, tracks=[_make_track("t1", [0, 0, 0], stale_steps=0)]),
        ]
        mean, p95 = compute_time_to_reacquire(frames)
        assert mean is None
        assert p95 is None

    def test_single_loss_reacquired(self):
        """Track goes stale at t=1, reacquires at t=4 => gap of 3 s."""
        frames = [
            _make_frame(0.0, tracks=[_make_track("t1", [0, 0, 0], stale_steps=0)]),
            _make_frame(1.0, tracks=[_make_track("t1", [0, 0, 0], stale_steps=1)]),
            _make_frame(2.0, tracks=[_make_track("t1", [0, 0, 0], stale_steps=2)]),
            _make_frame(3.0, tracks=[_make_track("t1", [0, 0, 0], stale_steps=3)]),
            _make_frame(4.0, tracks=[_make_track("t1", [0, 0, 0], stale_steps=0)]),
        ]
        mean, p95 = compute_time_to_reacquire(frames)
        assert mean == pytest.approx(3.0)
        assert p95 == pytest.approx(3.0)

    def test_unresolved_loss_capped_at_mission_end(self):
        """Track goes stale at t=2 and never reacquires; mission ends at t=5."""
        frames = [
            _make_frame(0.0, tracks=[_make_track("t1", [0, 0, 0], stale_steps=0)]),
            _make_frame(2.0, tracks=[_make_track("t1", [0, 0, 0], stale_steps=1)]),
            _make_frame(5.0, tracks=[_make_track("t1", [0, 0, 0], stale_steps=4)]),
        ]
        mean, _ = compute_time_to_reacquire(frames)
        assert mean == pytest.approx(3.0)  # 5.0 - 2.0

    def test_multiple_loss_events(self):
        """Two loss events: gap 2 s and gap 1 s."""
        frames = [
            _make_frame(0.0, tracks=[_make_track("t1", [0, 0, 0], stale_steps=0)]),
            _make_frame(1.0, tracks=[_make_track("t1", [0, 0, 0], stale_steps=1)]),
            _make_frame(3.0, tracks=[_make_track("t1", [0, 0, 0], stale_steps=0)]),  # gap = 2
            _make_frame(5.0, tracks=[_make_track("t1", [0, 0, 0], stale_steps=1)]),
            _make_frame(6.0, tracks=[_make_track("t1", [0, 0, 0], stale_steps=0)]),  # gap = 1
        ]
        mean, p95 = compute_time_to_reacquire(frames)
        assert mean == pytest.approx(1.5)
        assert p95 >= mean  # p95 >= mean for any distribution


# ===================================================================
# compute_track_continuity
# ===================================================================


class TestTrackContinuity:
    def test_empty_frames(self):
        mean, per = compute_track_continuity([], ["tgt1"])
        assert mean == 0.0
        assert per == {}

    def test_empty_target_ids(self):
        frames = [_make_frame(0.0, observations=[{"target_id": "tgt1"}])]
        mean, per = compute_track_continuity(frames, [])
        assert mean == 0.0
        assert per == {}

    def test_full_continuity(self):
        frames = [
            _make_frame(0.0, observations=[{"target_id": "tgt1"}]),
            _make_frame(1.0, observations=[{"target_id": "tgt1"}]),
            _make_frame(2.0, observations=[{"target_id": "tgt1"}]),
        ]
        mean, per = compute_track_continuity(frames, ["tgt1"])
        assert mean == pytest.approx(1.0)
        assert per["tgt1"] == pytest.approx(1.0)

    def test_partial_continuity(self):
        """tgt1 observed in 2 of 4 frames => 0.5."""
        frames = [
            _make_frame(0.0, observations=[{"target_id": "tgt1"}]),
            _make_frame(1.0, observations=[]),
            _make_frame(2.0, observations=[{"target_id": "tgt1"}]),
            _make_frame(3.0, observations=[]),
        ]
        mean, per = compute_track_continuity(frames, ["tgt1"])
        assert mean == pytest.approx(0.5)
        assert per["tgt1"] == pytest.approx(0.5)

    def test_multiple_targets(self):
        """tgt1: 3/3, tgt2: 1/3 => mean = 2/3."""
        frames = [
            _make_frame(0.0, observations=[{"target_id": "tgt1"}, {"target_id": "tgt2"}]),
            _make_frame(1.0, observations=[{"target_id": "tgt1"}]),
            _make_frame(2.0, observations=[{"target_id": "tgt1"}]),
        ]
        mean, per = compute_track_continuity(frames, ["tgt1", "tgt2"])
        assert per["tgt1"] == pytest.approx(1.0)
        assert per["tgt2"] == pytest.approx(1.0 / 3.0)
        assert mean == pytest.approx((1.0 + 1.0 / 3.0) / 2.0)


# ===================================================================
# compute_localisation_rmse
# ===================================================================


class TestLocalisationRMSE:
    def test_empty_frames(self):
        rmse, per = compute_localisation_rmse([])
        assert rmse is None
        assert per == {}

    def test_no_truths(self):
        frames = [
            _make_frame(0.0, tracks=[_make_track("t1", [0, 0, 0])], truths=[]),
        ]
        rmse, per = compute_localisation_rmse(frames)
        assert rmse is None
        assert per == {}

    def test_perfect_tracking(self):
        """Track exactly at truth position => RMSE = 0."""
        frames = [
            _make_frame(
                0.0,
                tracks=[_make_track("t1", [10.0, 20.0, 0.0])],
                truths=[{"target_id": "tgt1", "position": [10.0, 20.0, 0.0]}],
            ),
            _make_frame(
                1.0,
                tracks=[_make_track("t1", [11.0, 20.0, 0.0])],
                truths=[{"target_id": "tgt1", "position": [11.0, 20.0, 0.0]}],
            ),
        ]
        rmse, per = compute_localisation_rmse(frames)
        assert rmse == pytest.approx(0.0, abs=1e-9)
        assert per["t1"] == pytest.approx(0.0, abs=1e-9)

    def test_constant_offset(self):
        """Track offset by (3,4,0) from truth => error = 5.0 m each frame."""
        frames = [
            _make_frame(
                0.0,
                tracks=[_make_track("t1", [3.0, 4.0, 0.0])],
                truths=[{"target_id": "tgt1", "position": [0.0, 0.0, 0.0]}],
            ),
            _make_frame(
                1.0,
                tracks=[_make_track("t1", [13.0, 14.0, 0.0])],
                truths=[{"target_id": "tgt1", "position": [10.0, 10.0, 0.0]}],
            ),
        ]
        rmse, per = compute_localisation_rmse(frames)
        assert rmse == pytest.approx(5.0, abs=1e-6)

    def test_matching_uses_first_frame(self):
        """Two tracks, two truths — matching decided by nearest distance at first frame."""
        frames = [
            _make_frame(
                0.0,
                tracks=[
                    _make_track("t1", [0.0, 0.0, 0.0]),
                    _make_track("t2", [100.0, 0.0, 0.0]),
                ],
                truths=[
                    {"target_id": "tgt_near", "position": [1.0, 0.0, 0.0]},
                    {"target_id": "tgt_far", "position": [99.0, 0.0, 0.0]},
                ],
            ),
        ]
        rmse, per = compute_localisation_rmse(frames)
        assert "t1" in per
        assert "t2" in per
        # t1 should be matched to tgt_near (dist=1), t2 to tgt_far (dist=1)
        assert per["t1"] == pytest.approx(1.0, abs=1e-6)
        assert per["t2"] == pytest.approx(1.0, abs=1e-6)


# ===================================================================
# compute_covariance_reduction
# ===================================================================


class TestCovarianceReduction:
    def test_empty_frames(self):
        assert compute_covariance_reduction([]) is None

    def test_no_tracks_with_covariance(self):
        frames = [_make_frame(0.0, tracks=[_make_track("t1", [0, 0, 0])])]
        assert compute_covariance_reduction(frames) is None

    def test_below_update_threshold(self):
        """Tracks with < 5 updates are excluded."""
        frames = [
            _make_frame(
                0.0,
                tracks=[
                    _make_track("t1", [0, 0, 0], covariance=_identity_cov(10.0), update_count=3)
                ],
            ),
            _make_frame(
                1.0,
                tracks=[
                    _make_track("t1", [0, 0, 0], covariance=_identity_cov(2.0), update_count=4)
                ],
            ),
        ]
        assert compute_covariance_reduction(frames) is None

    def test_reduction_50_percent(self):
        """Initial trace=30 (10*I), final trace=15 (5*I) => reduction = 0.5."""
        frames = [
            _make_frame(
                0.0,
                tracks=[
                    _make_track("t1", [0, 0, 0], covariance=_identity_cov(10.0), update_count=1)
                ],
            ),
            _make_frame(
                1.0,
                tracks=[
                    _make_track("t1", [0, 0, 0], covariance=_identity_cov(5.0), update_count=10)
                ],
            ),
        ]
        result = compute_covariance_reduction(frames)
        assert result == pytest.approx(0.5)

    def test_no_reduction(self):
        """Covariance unchanged => reduction = 0."""
        cov = _identity_cov(10.0)
        frames = [
            _make_frame(0.0, tracks=[_make_track("t1", [0, 0, 0], covariance=cov, update_count=1)]),
            _make_frame(1.0, tracks=[_make_track("t1", [0, 0, 0], covariance=cov, update_count=6)]),
        ]
        result = compute_covariance_reduction(frames)
        assert result == pytest.approx(0.0)

    def test_covariance_too_small_ignored(self):
        """Covariance with fewer than 9 elements is skipped."""
        frames = [
            _make_frame(
                0.0, tracks=[_make_track("t1", [0, 0, 0], covariance=[1, 2, 3], update_count=10)]
            ),
        ]
        assert compute_covariance_reduction(frames) is None


# ===================================================================
# evaluate_replay
# ===================================================================


class TestEvaluateReplay:
    def test_basic_fields(self):
        replay = _build_good_replay()
        report = evaluate_replay(replay, "test_scn", "surveillance", 0.5, 42)
        assert report.scenario_name == "test_scn"
        assert report.mission_type == "surveillance"
        assert report.difficulty == pytest.approx(0.5)
        assert report.seed == 42
        assert report.duration_s == pytest.approx(19.0)

    def test_track_metrics_populated(self):
        replay = _build_good_replay()
        report = evaluate_replay(replay, "s", "m", 0.1, 1)
        # No stale events in the good replay => time_to_reacquire is None
        assert report.time_to_reacquire_mean_s is None
        # Full continuity
        assert report.track_continuity_mean == pytest.approx(1.0)
        # Perfect tracking (positions match exactly)
        assert report.localisation_rmse_m == pytest.approx(0.0, abs=1e-9)
        # Covariance reduction should be positive (scale decreases)
        assert report.covariance_reduction_mean is not None
        assert report.covariance_reduction_mean > 0.0

    def test_reliability_metrics(self):
        replay = _build_good_replay()
        replay["planner_events"] = [
            {"event_type": "plan_rejected"},
            {"event_type": "plan_rejected"},
            {"event_type": "safety_override"},
        ]
        replay["evaluation"]["false_handoff_rate"] = 0.1
        report = evaluate_replay(replay, "s", "m", 0.1, 1)
        assert report.infeasible_path_rejection_count == 2
        assert report.safety_override_count == 1
        assert report.false_handoff_rate == pytest.approx(0.1)

    def test_mission_completion(self):
        replay = _build_good_replay()
        replay["mission"] = {"required_objectives_total": 4, "required_objectives_met": 3}
        report = evaluate_replay(replay, "s", "m", 0.1, 1)
        assert report.mission_completion_rate == pytest.approx(0.75)
        assert report.required_objectives_met == 3
        assert report.required_objectives_total == 4

    def test_empty_replay(self):
        replay = {"meta": {"duration_s": 60.0}, "frames": []}
        report = evaluate_replay(replay, "empty", "surv", 0.0, 0)
        assert report.duration_s == pytest.approx(60.0)
        assert report.time_to_reacquire_mean_s is None
        assert report.track_continuity_mean == 0.0
        assert report.localisation_rmse_m is None
        assert report.covariance_reduction_mean is None

    def test_generated_at_utc_populated(self):
        replay = _build_good_replay()
        report = evaluate_replay(replay, "s", "m", 0.1, 1)
        assert len(report.generated_at_utc) > 0


# ===================================================================
# check_pass_fail
# ===================================================================


class TestCheckPassFail:
    def _good_report(self) -> EvaluationReport:
        """Return a report that satisfies all default thresholds."""
        return EvaluationReport(
            scenario_name="good",
            mission_type="surveillance",
            difficulty=0.1,
            seed=42,
            duration_s=60.0,
            time_to_reacquire_mean_s=5.0,
            track_continuity_mean=0.95,
            localisation_rmse_m=10.0,
            covariance_reduction_mean=0.7,
            false_handoff_rate=0.0,
            safety_override_count=0,
            comms_dropout_duration_s=0.0,
            mission_completion_rate=1.0,
            energy_reserve_min=0.5,
        )

    def test_all_pass(self):
        result = check_pass_fail(self._good_report())
        assert result.passed is True
        assert result.failure_reasons == []

    def test_fail_time_to_reacquire(self):
        report = EvaluationReport(
            scenario_name="x",
            mission_type="m",
            difficulty=0.0,
            seed=0,
            duration_s=60.0,
            time_to_reacquire_mean_s=35.0,
        )
        result = check_pass_fail(report)
        assert result.passed is False
        assert any("time_to_reacquire" in r for r in result.failure_reasons)

    def test_fail_track_continuity(self):
        report = EvaluationReport(
            scenario_name="x",
            mission_type="m",
            difficulty=0.0,
            seed=0,
            duration_s=60.0,
            track_continuity_mean=0.5,
            mission_completion_rate=1.0,
            energy_reserve_min=1.0,
        )
        result = check_pass_fail(report)
        assert result.passed is False
        assert any("track_continuity" in r for r in result.failure_reasons)

    def test_fail_localisation_rmse(self):
        report = EvaluationReport(
            scenario_name="x",
            mission_type="m",
            difficulty=0.0,
            seed=0,
            duration_s=60.0,
            localisation_rmse_m=55.0,
            track_continuity_mean=0.9,
            mission_completion_rate=1.0,
            energy_reserve_min=1.0,
        )
        result = check_pass_fail(report)
        assert result.passed is False
        assert any("localisation_rmse" in r for r in result.failure_reasons)

    def test_fail_safety_override(self):
        report = EvaluationReport(
            scenario_name="x",
            mission_type="m",
            difficulty=0.0,
            seed=0,
            duration_s=60.0,
            safety_override_count=1,
            track_continuity_mean=0.9,
            mission_completion_rate=1.0,
            energy_reserve_min=1.0,
        )
        result = check_pass_fail(report)
        assert result.passed is False
        assert any("safety_override" in r for r in result.failure_reasons)

    def test_fail_energy_reserve(self):
        report = EvaluationReport(
            scenario_name="x",
            mission_type="m",
            difficulty=0.0,
            seed=0,
            duration_s=60.0,
            energy_reserve_min=0.05,
            track_continuity_mean=0.9,
            mission_completion_rate=1.0,
        )
        result = check_pass_fail(report)
        assert result.passed is False
        assert any("energy_reserve" in r for r in result.failure_reasons)

    def test_custom_threshold_override(self):
        """Override one threshold — the rest still use defaults."""
        report = EvaluationReport(
            scenario_name="x",
            mission_type="m",
            difficulty=0.0,
            seed=0,
            duration_s=60.0,
            track_continuity_mean=0.7,
            mission_completion_rate=1.0,
            energy_reserve_min=1.0,
        )
        # Default threshold is 0.80 so 0.7 fails normally
        result_default = check_pass_fail(report)
        assert result_default.passed is False

        # Override continuity threshold to 0.60 so 0.7 passes
        result_custom = check_pass_fail(report, thresholds={"track_continuity_mean_min": 0.60})
        assert result_custom.passed is True

    def test_multiple_failures(self):
        """A report that fails on several metrics gets all reasons listed."""
        report = EvaluationReport(
            scenario_name="x",
            mission_type="m",
            difficulty=0.0,
            seed=0,
            duration_s=60.0,
            time_to_reacquire_mean_s=40.0,
            track_continuity_mean=0.5,
            localisation_rmse_m=60.0,
            false_handoff_rate=1.0,
            safety_override_count=5,
            comms_dropout_duration_s=10.0,
            mission_completion_rate=0.0,
            energy_reserve_min=0.01,
        )
        result = check_pass_fail(report)
        assert result.passed is False
        assert len(result.failure_reasons) >= 5


# ===================================================================
# Serialisation round-trip
# ===================================================================


class TestSerialisation:
    def test_roundtrip_all_fields(self):
        original = EvaluationReport(
            scenario_name="rt",
            mission_type="patrol",
            difficulty=0.3,
            seed=7,
            duration_s=120.0,
            time_to_reacquire_mean_s=4.5,
            time_to_reacquire_p95_s=8.2,
            track_continuity_mean=0.92,
            track_continuity_per_target={"tgt1": 0.95, "tgt2": 0.89},
            localisation_rmse_m=6.3,
            localisation_rmse_per_track={"t1": 5.0, "t2": 7.6},
            covariance_reduction_mean=0.65,
            false_handoff_rate=0.02,
            infeasible_path_rejection_count=1,
            safety_override_count=0,
            comms_dropout_count=2,
            comms_dropout_duration_s=1.5,
            mission_completion_rate=1.0,
            required_objectives_met=3,
            required_objectives_total=3,
            energy_reserve_min=0.42,
            energy_reserve_per_drone={"d0": 0.42, "d1": 0.88},
            passed=True,
            failure_reasons=[],
            tags=["nightly", "ci"],
            generated_at_utc="2025-01-01T00:00:00+00:00",
        )
        d = report_to_dict(original)
        restored = report_from_dict(d)

        assert restored.scenario_name == original.scenario_name
        assert restored.seed == original.seed
        assert restored.time_to_reacquire_mean_s == pytest.approx(original.time_to_reacquire_mean_s)
        assert restored.time_to_reacquire_p95_s == pytest.approx(original.time_to_reacquire_p95_s)
        assert restored.track_continuity_per_target == original.track_continuity_per_target
        assert restored.localisation_rmse_per_track == original.localisation_rmse_per_track
        assert restored.energy_reserve_per_drone == original.energy_reserve_per_drone
        assert restored.passed == original.passed
        assert restored.tags == original.tags
        assert restored.generated_at_utc == original.generated_at_utc

    def test_roundtrip_none_optionals(self):
        """Optional fields set to None survive the round-trip."""
        original = EvaluationReport(
            scenario_name="n",
            mission_type="m",
            difficulty=0.0,
            seed=0,
            duration_s=0.0,
            time_to_reacquire_mean_s=None,
            time_to_reacquire_p95_s=None,
            localisation_rmse_m=None,
            covariance_reduction_mean=None,
        )
        d = report_to_dict(original)
        restored = report_from_dict(d)
        assert restored.time_to_reacquire_mean_s is None
        assert restored.time_to_reacquire_p95_s is None
        assert restored.localisation_rmse_m is None
        assert restored.covariance_reduction_mean is None

    def test_dict_values_are_plain_python_types(self):
        """report_to_dict should not contain numpy scalars."""
        report = evaluate_replay(_build_good_replay(), "s", "m", 0.1, 1)
        d = report_to_dict(report)
        for key, val in d.items():
            if isinstance(val, float):
                assert type(val) is float, f"{key} is {type(val)}, expected float"
            elif isinstance(val, int) and not isinstance(val, bool):
                assert type(val) is int, f"{key} is {type(val)}, expected int"


# ===================================================================
# EvaluationReport dataclass
# ===================================================================


class TestEvaluationReportDataclass:
    def test_frozen(self):
        report = EvaluationReport(
            scenario_name="x",
            mission_type="m",
            difficulty=0.0,
            seed=0,
            duration_s=0.0,
        )
        with pytest.raises(AttributeError):
            report.passed = True  # type: ignore[misc]

    def test_default_values(self):
        report = EvaluationReport(
            scenario_name="x",
            mission_type="m",
            difficulty=0.0,
            seed=0,
            duration_s=0.0,
        )
        assert report.passed is False
        assert report.failure_reasons == []
        assert report.track_continuity_mean == 0.0
        assert report.energy_reserve_min == 1.0


# ===================================================================
# Integration: evaluate_replay -> check_pass_fail
# ===================================================================


class TestIntegration:
    def test_good_replay_passes(self):
        replay = _build_good_replay()
        report = evaluate_replay(replay, "integ", "surveillance", 0.1, 99)
        result = check_pass_fail(report)
        assert result.passed is True
        assert result.failure_reasons == []

    def test_bad_replay_fails(self):
        """Replay with large tracking errors and stale tracks should fail."""
        frames = []
        for i in range(10):
            ts = float(i)
            frames.append(
                _make_frame(
                    ts=ts,
                    tracks=[_make_track("t1", [1000.0, 1000.0, 0.0], stale_steps=5)],
                    truths=[{"target_id": "tgt1", "position": [0.0, 0.0, 0.0]}],
                    observations=[],  # no observations => continuity = 0
                )
            )
        replay = {
            "meta": {"duration_s": 9.0},
            "frames": frames,
            "planner_events": [{"event_type": "safety_override"}],
            "evaluation": {
                "false_handoff_rate": 0.0,
                "comms_dropout_count": 0,
                "comms_dropout_duration_s": 0.0,
            },
            "mission": {"required_objectives_total": 2, "required_objectives_met": 0},
        }
        report = evaluate_replay(replay, "bad", "search", 1.0, 0)
        result = check_pass_fail(report)
        assert result.passed is False
        assert len(result.failure_reasons) >= 2
