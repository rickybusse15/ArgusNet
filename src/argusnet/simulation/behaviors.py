"""Realistic target behavior models for the Smart Trajectory Tracker.

Provides pluggable trajectory generators (loiter, transit, evasive, search
pattern) that all satisfy the ``TrajectoryFn`` protocol used by the simulation
engine.  Each callable accepts a time in seconds and returns a
``(position, velocity)`` tuple of 3-D numpy arrays.

Typical usage::

    from argusnet.simulation.behaviors import build_target_trajectory, LoiterBehavior

    traj = build_target_trajectory("loiter", bounds, altitude_m=150.0,
                                    speed_mps=25.0, seed=42)
    pos, vel = traj(10.0)
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

TrajectoryFn = Callable[[float], tuple[np.ndarray, np.ndarray]]


# ---------------------------------------------------------------------------
# Configuration data-classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FlightEnvelope:
    """Physical constraints for an aerial vehicle."""

    max_speed_mps: float = 50.0
    min_speed_mps: float = 8.0
    max_acceleration_mps2: float = 5.0
    max_bank_angle_deg: float = 30.0
    max_climb_rate_mps: float = 8.0
    max_descent_rate_mps: float = 5.0
    min_altitude_agl_m: float = 30.0
    max_altitude_agl_m: float = 500.0

    @property
    def min_turn_radius_m(self) -> float:
        """Minimum turn radius derived from speed and bank angle.

        r = v^2 / (g * tan(bank_angle))
        Uses *min_speed_mps* for the tightest possible turn.
        """
        g = 9.80665
        bank_rad = math.radians(max(self.max_bank_angle_deg, 1.0))
        return self.min_speed_mps**2 / (g * math.tan(bank_rad))

    def min_turn_radius_at_speed(self, speed_mps: float) -> float:
        """Return the minimum turn radius at an arbitrary speed."""
        g = 9.80665
        bank_rad = math.radians(max(self.max_bank_angle_deg, 1.0))
        return speed_mps**2 / (g * math.tan(bank_rad))


@dataclass(frozen=True)
class TurbulenceModel:
    """Random perturbation layer applied on top of any trajectory."""

    intensity: float = 0.5
    spectral_scale_m: float = 200.0
    seed: int = 42

    def perturbation(self, time_s: float, position: np.ndarray) -> np.ndarray:
        """Return a deterministic 3-D velocity perturbation (m/s).

        The perturbation is a smooth, bounded function of time and position
        derived from seeded hashing so that repeated calls with the same
        arguments produce the same result.
        """
        rng = np.random.default_rng(self.seed ^ int(abs(time_s * 1000.0)) % (2**31))
        scale_inv = 1.0 / max(self.spectral_scale_m, 1.0)
        phase_x = math.sin(position[0] * scale_inv + time_s * 0.3) * 0.5
        phase_y = math.cos(position[1] * scale_inv + time_s * 0.25) * 0.5
        phase_z = math.sin((position[0] + position[1]) * scale_inv * 0.7 + time_s * 0.2) * 0.3

        noise = rng.standard_normal(3)
        base = np.array([phase_x, phase_y, phase_z], dtype=float)
        return (base + noise * 0.3) * self.intensity


# ---------------------------------------------------------------------------
# Behaviour classes
# ---------------------------------------------------------------------------


class LoiterBehavior:
    """Orbit around a point with configurable radius and altitude variation."""

    def __init__(
        self,
        center: np.ndarray,
        radius_m: float,
        speed_mps: float,
        altitude_variation_m: float = 20.0,
        clockwise: bool = True,
        altitude_period_s: float = 60.0,
    ) -> None:
        self._center = np.asarray(center, dtype=float).copy()
        self._radius_m = max(float(radius_m), 1.0)
        self._speed_mps = max(float(speed_mps), 0.5)
        self._alt_var_m = float(altitude_variation_m)
        self._direction = -1.0 if clockwise else 1.0
        self._alt_period_s = max(float(altitude_period_s), 1.0)
        self._omega = self._direction * self._speed_mps / self._radius_m

    def __call__(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        angle = self._omega * t
        cx, cy = float(self._center[0]), float(self._center[1])
        base_alt = float(self._center[2]) if self._center.size >= 3 else 150.0

        x = cx + self._radius_m * math.cos(angle)
        y = cy + self._radius_m * math.sin(angle)

        alt_phase = 2.0 * math.pi * t / self._alt_period_s
        z = base_alt + self._alt_var_m * math.sin(alt_phase)

        vx = -self._radius_m * self._omega * math.sin(angle)
        vy = self._radius_m * self._omega * math.cos(angle)
        vz = self._alt_var_m * (2.0 * math.pi / self._alt_period_s) * math.cos(alt_phase)

        position = np.array([x, y, z], dtype=float)
        velocity = np.array([vx, vy, vz], dtype=float)
        return position, velocity


class TransitBehavior:
    """Point-to-point flight with smooth acceleration/deceleration.

    Uses cubic Hermite interpolation between waypoints so that position
    and velocity are continuous.  After the final waypoint the vehicle
    loiters in a small circle.
    """

    def __init__(
        self,
        waypoints: list[np.ndarray],
        speed_mps: float,
        envelope: FlightEnvelope | None = None,
    ) -> None:
        if len(waypoints) < 2:
            raise ValueError("TransitBehavior requires at least 2 waypoints.")
        self._waypoints = [np.asarray(w, dtype=float).copy() for w in waypoints]
        self._speed = max(float(speed_mps), 0.5)
        self._envelope = envelope if envelope is not None else FlightEnvelope()

        # Pre-compute cumulative arc-length times for constant-speed flight
        self._segment_lengths: list[float] = []
        self._cumulative_times: list[float] = [0.0]
        for i in range(len(self._waypoints) - 1):
            seg = self._waypoints[i + 1] - self._waypoints[i]
            length = float(np.linalg.norm(seg))
            self._segment_lengths.append(max(length, 1e-6))
            self._cumulative_times.append(
                self._cumulative_times[-1] + max(length, 1e-6) / self._speed
            )
        self._total_time = self._cumulative_times[-1]

        # Build tangent vectors at each waypoint for Hermite interpolation.
        n = len(self._waypoints)
        self._tangents: list[np.ndarray] = []
        for i in range(n):
            if i == 0:
                d = self._waypoints[1] - self._waypoints[0]
            elif i == n - 1:
                d = self._waypoints[-1] - self._waypoints[-2]
            else:
                d = self._waypoints[i + 1] - self._waypoints[i - 1]
            norm = float(np.linalg.norm(d))
            d = d / norm * self._speed if norm > 1e-09 else np.zeros(3, dtype=float)
            self._tangents.append(d)

    # ---- Hermite helpers ----

    @staticmethod
    def _hermite(
        p0: np.ndarray, m0: np.ndarray, p1: np.ndarray, m1: np.ndarray, u: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate cubic Hermite at parameter *u* in [0, 1].

        Returns (position, tangent_du).
        """
        u2 = u * u
        u3 = u2 * u
        h00 = 2 * u3 - 3 * u2 + 1
        h10 = u3 - 2 * u2 + u
        h01 = -2 * u3 + 3 * u2
        h11 = u3 - u2
        pos = h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1
        dh00 = 6 * u2 - 6 * u
        dh10 = 3 * u2 - 4 * u + 1
        dh01 = -6 * u2 + 6 * u
        dh11 = 3 * u2 - 2 * u
        tangent = dh00 * p0 + dh10 * m0 + dh01 * p1 + dh11 * m1
        return pos, tangent

    def __call__(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        if t < 0.0:
            t = 0.0

        # After transit is done, loiter around the final waypoint.
        if t >= self._total_time:
            final = self._waypoints[-1]
            dt = t - self._total_time
            loiter_radius = 20.0
            omega = self._speed / loiter_radius
            angle = omega * dt
            x = float(final[0]) + loiter_radius * math.cos(angle)
            y = float(final[1]) + loiter_radius * math.sin(angle)
            z = float(final[2]) if final.size >= 3 else 150.0
            vx = -loiter_radius * omega * math.sin(angle)
            vy = loiter_radius * omega * math.cos(angle)
            return (
                np.array([x, y, z], dtype=float),
                np.array([vx, vy, 0.0], dtype=float),
            )

        # Find the active segment.
        seg_idx = 0
        for i in range(len(self._cumulative_times) - 1):
            if t < self._cumulative_times[i + 1]:
                seg_idx = i
                break
        else:
            seg_idx = len(self._segment_lengths) - 1

        seg_start_t = self._cumulative_times[seg_idx]
        seg_duration = self._cumulative_times[seg_idx + 1] - seg_start_t
        u = (t - seg_start_t) / max(seg_duration, 1e-9)
        u = max(0.0, min(1.0, u))

        # Scale tangents by segment duration for correct Hermite parameterisation.
        m0 = self._tangents[seg_idx] * seg_duration
        m1 = self._tangents[seg_idx + 1] * seg_duration
        p0 = self._waypoints[seg_idx]
        p1 = self._waypoints[seg_idx + 1]

        pos, tangent_du = self._hermite(p0, m0, p1, m1, u)
        # tangent_du is dp/du; velocity = dp/dt = dp/du * du/dt = tangent_du / seg_duration
        vel = tangent_du / max(seg_duration, 1e-9)
        return pos, vel


class EvasiveBehavior:
    """Terrain-masking evasive flight.

    Wraps an existing trajectory, intermittently dropping the altitude to
    nap-of-earth when random evasion windows trigger.
    """

    def __init__(
        self,
        base_trajectory: TrajectoryFn,
        terrain_height_fn: Callable[[float, float], float],
        nap_of_earth_agl_m: float = 40.0,
        evasion_probability: float = 0.3,
        evasion_duration_s: float = 15.0,
        seed: int = 42,
    ) -> None:
        self._base = base_trajectory
        self._terrain_fn = terrain_height_fn
        self._noe_agl = max(float(nap_of_earth_agl_m), 5.0)
        self._evasion_prob = max(0.0, min(1.0, float(evasion_probability)))
        self._evasion_dur = max(float(evasion_duration_s), 1.0)
        self._seed = seed

        # Pre-generate evasion windows for the first 600 seconds.
        self._windows: list[tuple[float, float]] = []
        self._max_precomputed_s = 600.0
        self._generate_windows(self._max_precomputed_s)

    def _generate_windows(self, max_time_s: float) -> None:
        rng = np.random.default_rng(self._seed)
        check_interval = 5.0
        t = 0.0
        windows: list[tuple[float, float]] = []
        while t < max_time_s:
            if rng.random() < self._evasion_prob * (check_interval / self._evasion_dur):
                end_t = t + self._evasion_dur
                windows.append((t, end_t))
                t = end_t
            else:
                t += check_interval
        self._windows = windows

    def _is_evading(self, t: float) -> tuple[bool, float]:
        """Return (is_evading, blend_factor) at time *t*.

        The blend factor ramps 0 -> 1 over the first 2 s of the window and
        1 -> 0 over the last 2 s for smooth altitude transitions.
        """
        ramp_s = 2.0
        for start, end in self._windows:
            if start <= t <= end:
                end - start
                elapsed = t - start
                remaining = end - t
                ramp_in = min(elapsed / ramp_s, 1.0) if ramp_s > 0 else 1.0
                ramp_out = min(remaining / ramp_s, 1.0) if ramp_s > 0 else 1.0
                blend = min(ramp_in, ramp_out)
                return True, blend
        return False, 0.0

    def __call__(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        pos, vel = self._base(t)
        evading, blend = self._is_evading(t)
        if not evading or blend <= 0.0:
            return pos, vel

        terrain_z = self._terrain_fn(float(pos[0]), float(pos[1]))
        noe_z = terrain_z + self._noe_agl
        target_z = pos[2] * (1.0 - blend) + noe_z * blend
        dz = target_z - pos[2]

        new_pos = pos.copy()
        new_pos[2] = target_z
        new_vel = vel.copy()
        new_vel[2] = vel[2] + dz * 0.5  # approximate descent rate
        return new_pos, new_vel


class SearchPatternBehavior:
    """Expanding square or sector search pattern."""

    def __init__(
        self,
        center: np.ndarray,
        pattern: str = "expanding_square",
        leg_length_m: float = 200.0,
        speed_mps: float = 30.0,
        altitude_m: float = 150.0,
    ) -> None:
        if pattern not in ("expanding_square", "sector"):
            raise ValueError(f"Unknown search pattern {pattern!r}.")
        self._center = np.asarray(center, dtype=float).copy()
        self._pattern = pattern
        self._leg = max(float(leg_length_m), 10.0)
        self._speed = max(float(speed_mps), 0.5)
        self._alt = float(altitude_m)

        # Pre-build waypoint list for the pattern.
        self._waypoints = self._build_waypoints()
        self._segment_lengths: list[float] = []
        self._cumulative_dists: list[float] = [0.0]
        for i in range(len(self._waypoints) - 1):
            d = float(np.linalg.norm(self._waypoints[i + 1] - self._waypoints[i]))
            self._segment_lengths.append(max(d, 1e-6))
            self._cumulative_dists.append(self._cumulative_dists[-1] + max(d, 1e-6))
        self._total_dist = self._cumulative_dists[-1] if self._cumulative_dists else 1.0

        # Pre-compute Catmull-Rom tangents at each waypoint for smooth corners.
        n = len(self._waypoints)
        self._tangents: list[np.ndarray] = []
        for i in range(n):
            if i == 0:
                d = self._waypoints[1] - self._waypoints[0]
            elif i == n - 1:
                d = self._waypoints[-1] - self._waypoints[-2]
            else:
                d = self._waypoints[i + 1] - self._waypoints[i - 1]
            norm = float(np.linalg.norm(d))
            self._tangents.append(
                d / norm * self._speed if norm > 1e-9 else np.zeros(3, dtype=float)
            )

    def _build_waypoints(self) -> list[np.ndarray]:
        cx = float(self._center[0])
        cy = float(self._center[1])

        if self._pattern == "expanding_square":
            # Expanding square: successive right turns with increasing leg length.
            pts: list[np.ndarray] = [np.array([cx, cy, self._alt], dtype=float)]
            directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
            leg_count = 20  # enough legs for long simulations
            for i in range(leg_count):
                multiplier = (i // 2 + 1) * self._leg
                dx, dy = directions[i % 4]
                last = pts[-1]
                pts.append(
                    np.array(
                        [
                            float(last[0]) + dx * multiplier,
                            float(last[1]) + dy * multiplier,
                            self._alt,
                        ],
                        dtype=float,
                    )
                )
            return pts
        else:
            # Sector search: radiating lines from center.
            pts = []
            sector_count = 8
            radius = self._leg
            for i in range(sector_count):
                angle = 2.0 * math.pi * i / sector_count
                pts.append(np.array([cx, cy, self._alt], dtype=float))
                pts.append(
                    np.array(
                        [
                            cx + radius * math.cos(angle),
                            cy + radius * math.sin(angle),
                            self._alt,
                        ],
                        dtype=float,
                    )
                )
                radius += self._leg * 0.25
            return pts

    def __call__(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        if len(self._waypoints) < 2:
            return self._center.copy(), np.zeros(3, dtype=float)

        dist = (self._speed * max(t, 0.0)) % self._total_dist
        seg_idx = 0
        for i in range(len(self._cumulative_dists) - 1):
            if dist < self._cumulative_dists[i + 1]:
                seg_idx = i
                break
        else:
            seg_idx = max(len(self._segment_lengths) - 1, 0)

        seg_start_d = self._cumulative_dists[seg_idx]
        seg_len = max(self._segment_lengths[seg_idx], 1e-9)
        seg_frac = (dist - seg_start_d) / seg_len
        seg_frac = max(0.0, min(1.0, seg_frac))

        p0 = self._waypoints[seg_idx]
        p1 = self._waypoints[seg_idx + 1]
        # Scale tangents by normalised segment length so Hermite is arc-length aware.
        scale = seg_len / self._speed
        m0 = self._tangents[seg_idx] * scale
        m1 = self._tangents[seg_idx + 1] * scale
        # Cubic Hermite basis.
        u = seg_frac
        u2 = u * u
        u3 = u2 * u
        h00 = 2 * u3 - 3 * u2 + 1
        h10 = u3 - 2 * u2 + u
        h01 = -2 * u3 + 3 * u2
        h11 = u3 - u2
        pos = h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1
        # Velocity = derivative of Hermite w.r.t. arc-length (chain-rule).
        dh00 = 6 * u2 - 6 * u
        dh10 = 3 * u2 - 4 * u + 1
        dh01 = -6 * u2 + 6 * u
        dh11 = 3 * u2 - 2 * u
        tangent_du = dh00 * p0 + dh10 * m0 + dh01 * p1 + dh11 * m1
        tnorm = float(np.linalg.norm(tangent_du))
        vel = tangent_du / tnorm * self._speed if tnorm > 1e-9 else np.zeros(3, dtype=float)

        return pos, vel


class CompositeTrajectory:
    """Chain multiple behaviours with smooth transitions.

    Each segment is a ``(start_time_s, trajectory_fn)`` pair.  Segments are
    sorted by start time.  Between segments there is a 2-second linear
    position/velocity blend so that the trajectory is continuous.
    """

    BLEND_DURATION_S: float = 2.0

    def __init__(self, segments: list[tuple[float, TrajectoryFn]]) -> None:
        if not segments:
            raise ValueError("CompositeTrajectory requires at least one segment.")
        self._segments = sorted(segments, key=lambda s: s[0])

    def _active_index(self, t: float) -> int:
        idx = 0
        for i, (start_t, _) in enumerate(self._segments):
            if t >= start_t:
                idx = i
        return idx

    def __call__(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        idx = self._active_index(t)
        seg_start, seg_fn = self._segments[idx]

        # If not the first segment, check whether we are in the blend zone.
        if idx > 0:
            blend_end = seg_start + self.BLEND_DURATION_S
            if t < blend_end:
                prev_start, prev_fn = self._segments[idx - 1]
                alpha = (t - seg_start) / self.BLEND_DURATION_S
                alpha = max(0.0, min(1.0, alpha))
                # Smooth-step for C1 continuity.
                alpha = 3 * alpha**2 - 2 * alpha**3
                p_old, v_old = prev_fn(t)
                p_new, v_new = seg_fn(t)
                pos = p_old * (1.0 - alpha) + p_new * alpha
                vel = v_old * (1.0 - alpha) + v_new * alpha
                return pos, vel

        return seg_fn(t)


# ---------------------------------------------------------------------------
# Target behavior policy families (architecture update Section 11)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TargetPolicyParams:
    """Parameters controlling a target behavior policy.

    These drive the difficulty and character of target motion in the simulation.
    All speeds in m/s, angles in radians, distances in metres.
    """

    speed_min_mps: float = 10.0
    speed_max_mps: float = 30.0
    heading_change_aggressiveness: float = 0.3  # 0 = straight, 1 = max zigzag
    terrain_preference: float = 0.0  # 0 = ignores terrain, 1 = hugs terrain
    exposure_penalty: float = 0.0  # 0 = ignores exposure, 1 = avoids open areas
    urgency: float = 0.5  # 0 = leisurely, 1 = max speed toward goal
    predictability: float = 0.8  # 0 = random, 1 = perfectly predictable


# Named policy families from architecture update Section 11.3
TARGET_POLICY_DIRECT_TRANSIT = TargetPolicyParams(
    speed_min_mps=15.0,
    speed_max_mps=35.0,
    heading_change_aggressiveness=0.05,
    terrain_preference=0.0,
    exposure_penalty=0.0,
    urgency=0.9,
    predictability=0.95,
)
TARGET_POLICY_CAUTIOUS_TRANSIT = TargetPolicyParams(
    speed_min_mps=10.0,
    speed_max_mps=25.0,
    heading_change_aggressiveness=0.15,
    terrain_preference=0.3,
    exposure_penalty=0.2,
    urgency=0.5,
    predictability=0.7,
)
TARGET_POLICY_COVER_SEEKING = TargetPolicyParams(
    speed_min_mps=8.0,
    speed_max_mps=20.0,
    heading_change_aggressiveness=0.3,
    terrain_preference=0.8,
    exposure_penalty=0.7,
    urgency=0.3,
    predictability=0.4,
)
TARGET_POLICY_DECEPTIVE_ZIGZAG = TargetPolicyParams(
    speed_min_mps=12.0,
    speed_max_mps=28.0,
    heading_change_aggressiveness=0.8,
    terrain_preference=0.2,
    exposure_penalty=0.1,
    urgency=0.4,
    predictability=0.15,
)
TARGET_POLICY_STOP_OBSERVE_MOVE = TargetPolicyParams(
    speed_min_mps=0.0,
    speed_max_mps=25.0,
    heading_change_aggressiveness=0.2,
    terrain_preference=0.5,
    exposure_penalty=0.5,
    urgency=0.2,
    predictability=0.3,
)
TARGET_POLICY_CORRIDOR_HUGGING = TargetPolicyParams(
    speed_min_mps=12.0,
    speed_max_mps=30.0,
    heading_change_aggressiveness=0.1,
    terrain_preference=0.6,
    exposure_penalty=0.3,
    urgency=0.7,
    predictability=0.6,
)
TARGET_POLICY_RANDOM_AMBIGUITY = TargetPolicyParams(
    speed_min_mps=5.0,
    speed_max_mps=35.0,
    heading_change_aggressiveness=0.5,
    terrain_preference=0.2,
    exposure_penalty=0.1,
    urgency=0.3,
    predictability=0.05,
)

NAMED_TARGET_POLICIES: dict = {
    "direct_transit": TARGET_POLICY_DIRECT_TRANSIT,
    "cautious_transit": TARGET_POLICY_CAUTIOUS_TRANSIT,
    "cover_seeking": TARGET_POLICY_COVER_SEEKING,
    "deceptive_zigzag": TARGET_POLICY_DECEPTIVE_ZIGZAG,
    "stop_observe_move": TARGET_POLICY_STOP_OBSERVE_MOVE,
    "corridor_hugging": TARGET_POLICY_CORRIDOR_HUGGING,
    "random_ambiguity": TARGET_POLICY_RANDOM_AMBIGUITY,
}


class StopObserveMoveBehavior:
    """Target alternates between moving and stopping to observe.

    During stop phases the target holds position. During move phases
    it executes a transit trajectory at the policy speed.
    """

    def __init__(
        self,
        waypoints: list[np.ndarray],
        speed_mps: float = 20.0,
        stop_duration_s: float = 10.0,
        move_duration_s: float = 20.0,
        seed: int = 42,
    ) -> None:
        self._transit = TransitBehavior(waypoints=waypoints, speed_mps=speed_mps)
        self._stop_dur = max(float(stop_duration_s), 1.0)
        self._move_dur = max(float(move_duration_s), 1.0)
        self._cycle = self._stop_dur + self._move_dur
        rng = np.random.default_rng(seed)
        self._phase_offset = rng.uniform(0, self._cycle)

    def __call__(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        phase = (t + self._phase_offset) % self._cycle
        if phase < self._move_dur:
            return self._transit(t)
        else:
            pos, _ = self._transit(t)
            return pos, np.zeros(3, dtype=float)


class DeceptiveZigzagBehavior:
    """Target transits toward a goal but with random heading perturbations.

    Produces an unpredictable zigzag path that stresses the tracker's
    turn-rate estimation.
    """

    def __init__(
        self,
        start_xy: np.ndarray,
        goal_xy: np.ndarray,
        altitude_m: float = 150.0,
        speed_mps: float = 22.0,
        zigzag_amplitude_m: float = 80.0,
        zigzag_period_s: float = 25.0,
        seed: int = 42,
    ) -> None:
        self._start = np.array(start_xy[:2], dtype=float)
        self._goal = np.array(goal_xy[:2], dtype=float)
        self._alt = float(altitude_m)
        self._speed = float(speed_mps)
        self._amp = float(zigzag_amplitude_m)
        self._period = max(float(zigzag_period_s), 1.0)
        rng = np.random.default_rng(seed)
        self._phase = rng.uniform(0, 2 * math.pi)
        diff = self._goal - self._start
        self._total_dist = float(np.linalg.norm(diff))
        if self._total_dist < 1.0:
            self._dir = np.array([1.0, 0.0], dtype=float)
            self._perp = np.array([0.0, 1.0], dtype=float)
        else:
            self._dir = diff / self._total_dist
            self._perp = np.array([-self._dir[1], self._dir[0]], dtype=float)

    def __call__(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        along = min(self._speed * t, self._total_dist)
        lateral = self._amp * math.sin(2 * math.pi * t / self._period + self._phase)
        xy = self._start + self._dir * along + self._perp * lateral

        d_along = self._speed if along < self._total_dist else 0.0
        d_lateral = (
            self._amp
            * 2
            * math.pi
            / self._period
            * math.cos(2 * math.pi * t / self._period + self._phase)
        )
        vxy = self._dir * d_along + self._perp * d_lateral

        pos = np.array([xy[0], xy[1], self._alt], dtype=float)
        vel = np.array([vxy[0], vxy[1], 0.0], dtype=float)
        return pos, vel


class SplitProbabilityBehavior:
    """Target randomly switches between N sub-trajectories.

    At each decision point (every ``decision_interval_s``), the target
    probabilistically selects a new trajectory from the library.
    This stresses multi-hypothesis tracking.
    """

    def __init__(
        self,
        trajectories: list[TrajectoryFn],
        decision_interval_s: float = 30.0,
        seed: int = 42,
    ) -> None:
        if not trajectories:
            raise ValueError("Need at least one trajectory.")
        self._trajectories = trajectories
        self._interval = max(float(decision_interval_s), 1.0)
        self._rng = np.random.default_rng(seed)
        # Pre-generate choices for 600 s
        n_decisions = int(600.0 / self._interval) + 1
        self._choices = self._rng.integers(0, len(trajectories), size=n_decisions)

    def __call__(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        decision_idx = int(t / self._interval) % len(self._choices)
        traj_idx = int(self._choices[decision_idx]) % len(self._trajectories)
        return self._trajectories[traj_idx](t)


def build_policy_trajectory(
    policy_name: str,
    bounds: Any,
    altitude_m: float,
    speed_mps: float,
    seed: int = 42,
    terrain_height_fn: Callable[[float, float], float] | None = None,
) -> TrajectoryFn:
    """Build a target trajectory from a named policy family.

    Parameters
    ----------
    policy_name:
        One of :data:`NAMED_TARGET_POLICIES` keys.
    bounds:
        Map bounds with ``x_min_m``, ``x_max_m``, ``y_min_m``, ``y_max_m``.
    altitude_m:
        Nominal altitude in metres.
    speed_mps:
        Base speed (scaled by policy urgency).
    seed:
        RNG seed for deterministic output.
    terrain_height_fn:
        Optional terrain height query for cover-seeking policies.
    """
    if policy_name not in NAMED_TARGET_POLICIES:
        raise ValueError(
            f"Unknown target policy {policy_name!r}. Choose from {sorted(NAMED_TARGET_POLICIES)}."
        )

    params = NAMED_TARGET_POLICIES[policy_name]
    rng = np.random.default_rng(seed)
    center, span = _bounds_center_and_span(bounds)
    effective_speed = speed_mps * params.urgency

    if policy_name == "direct_transit":
        start = center + rng.uniform(-0.3, 0.3, size=2) * span
        goal = center + rng.uniform(-0.3, 0.3, size=2) * span
        wps = [
            np.array([start[0], start[1], altitude_m], dtype=float),
            np.array([goal[0], goal[1], altitude_m], dtype=float),
        ]
        return TransitBehavior(waypoints=wps, speed_mps=effective_speed)

    if policy_name == "cautious_transit":
        n_wps = rng.integers(4, 8)
        wps = []
        for _ in range(n_wps):
            wp = np.array(
                [
                    center[0] + rng.uniform(-0.25, 0.25) * span[0],
                    center[1] + rng.uniform(-0.25, 0.25) * span[1],
                    altitude_m + rng.uniform(-10, 10),
                ],
                dtype=float,
            )
            wps.append(wp)
        return TransitBehavior(waypoints=wps, speed_mps=effective_speed)

    if policy_name == "cover_seeking":
        base = _sinusoid_trajectory(bounds, altitude_m, effective_speed, seed)
        if terrain_height_fn is None:
            terrain_height_fn = lambda x, y: 0.0  # noqa: E731
        return EvasiveBehavior(
            base_trajectory=base,
            terrain_height_fn=terrain_height_fn,
            nap_of_earth_agl_m=30.0,
            evasion_probability=0.6,
            evasion_duration_s=20.0,
            seed=seed,
        )

    if policy_name == "deceptive_zigzag":
        start = center + rng.uniform(-0.3, 0.3, size=2) * span
        goal = center + rng.uniform(-0.3, 0.3, size=2) * span
        return DeceptiveZigzagBehavior(
            start_xy=start,
            goal_xy=goal,
            altitude_m=altitude_m,
            speed_mps=effective_speed,
            zigzag_amplitude_m=float(span.min()) * 0.08,
            zigzag_period_s=25.0,
            seed=seed,
        )

    if policy_name == "stop_observe_move":
        n_wps = rng.integers(3, 6)
        wps = []
        for _ in range(n_wps):
            wp = np.array(
                [
                    center[0] + rng.uniform(-0.2, 0.2) * span[0],
                    center[1] + rng.uniform(-0.2, 0.2) * span[1],
                    altitude_m,
                ],
                dtype=float,
            )
            wps.append(wp)
        return StopObserveMoveBehavior(
            waypoints=wps,
            speed_mps=effective_speed,
            stop_duration_s=12.0,
            move_duration_s=18.0,
            seed=seed,
        )

    if policy_name == "corridor_hugging":
        # Linear corridor with slight meander
        start = np.array(
            [
                center[0] - span[0] * 0.35,
                center[1] + rng.uniform(-0.1, 0.1) * span[1],
            ],
            dtype=float,
        )
        end = np.array(
            [
                center[0] + span[0] * 0.35,
                center[1] + rng.uniform(-0.1, 0.1) * span[1],
            ],
            dtype=float,
        )
        n_intermediate = 4
        wps = [np.array([start[0], start[1], altitude_m], dtype=float)]
        for i in range(1, n_intermediate + 1):
            frac = i / (n_intermediate + 1)
            mid = start * (1 - frac) + end * frac
            mid += rng.uniform(-0.02, 0.02, size=2) * span
            wps.append(np.array([mid[0], mid[1], altitude_m], dtype=float))
        wps.append(np.array([end[0], end[1], altitude_m], dtype=float))
        return TransitBehavior(waypoints=wps, speed_mps=effective_speed)

    if policy_name == "random_ambiguity":
        # Multiple sub-trajectories, randomly switching
        sub_trajs: list[TrajectoryFn] = []
        for i in range(3):
            sub_seed = seed + i * 1000
            sub_trajs.append(_sinusoid_trajectory(bounds, altitude_m, effective_speed, sub_seed))
        return SplitProbabilityBehavior(
            trajectories=sub_trajs,
            decision_interval_s=20.0,
            seed=seed,
        )

    raise ValueError(f"Unhandled policy {policy_name!r}.")


def scale_policy_by_difficulty(
    policy_name: str,
    difficulty: float,
) -> str:
    """Select a target policy name based on difficulty level.

    At low difficulty, targets use direct/cautious transit.
    At high difficulty, targets use deceptive or random policies.

    Parameters
    ----------
    difficulty:
        Value in [0.0, 1.0].

    Returns
    -------
    A key from :data:`NAMED_TARGET_POLICIES`.
    """
    difficulty = max(0.0, min(1.0, float(difficulty)))
    if difficulty < 0.2:
        return "direct_transit"
    elif difficulty < 0.4:
        return "cautious_transit"
    elif difficulty < 0.6:
        return policy_name if policy_name in NAMED_TARGET_POLICIES else "cautious_transit"
    elif difficulty < 0.8:
        return "deceptive_zigzag"
    else:
        return "random_ambiguity"


# ---------------------------------------------------------------------------
# Factory / preset helpers
# ---------------------------------------------------------------------------

BEHAVIOR_PRESETS = frozenset(
    {
        "loiter",
        "transit",
        "evasive",
        "search_pattern",
        "sinusoid",
        "racetrack",
        "waypoint_patrol",
        "mixed",
    }
)


def _bounds_center_and_span(bounds: Any) -> tuple[np.ndarray, np.ndarray]:
    """Extract centre and span from a bounds mapping or dict-like object."""
    if hasattr(bounds, "__getitem__"):
        x_min = float(bounds["x_min_m"])
        x_max = float(bounds["x_max_m"])
        y_min = float(bounds["y_min_m"])
        y_max = float(bounds["y_max_m"])
    else:
        x_min, x_max, y_min, y_max = -500.0, 500.0, -500.0, 500.0
    center = np.array([(x_min + x_max) * 0.5, (y_min + y_max) * 0.5], dtype=float)
    span = np.array([x_max - x_min, y_max - y_min], dtype=float)
    return center, span


def _sinusoid_trajectory(
    bounds: Any,
    altitude_m: float,
    speed_mps: float,
    seed: int,
) -> TrajectoryFn:
    """Lightweight sinusoidal path for the factory (no terrain coupling)."""
    rng = np.random.default_rng(seed)
    center, span = _bounds_center_and_span(bounds)
    start_xy = center + rng.uniform(-0.2, 0.2, size=2) * span
    angle = rng.uniform(0, 2 * math.pi)
    vx = speed_mps * math.cos(angle) * 0.3
    vy = speed_mps * math.sin(angle) * 0.3
    lateral_amp = float(span.min()) * 0.08
    freq = speed_mps / max(float(span.min()), 100.0) * 2.0
    phase = rng.uniform(0, 2 * math.pi)

    def _traj(t: float) -> tuple[np.ndarray, np.ndarray]:
        x = start_xy[0] + vx * t + lateral_amp * math.sin(freq * t + phase)
        y = start_xy[1] + vy * t
        z = altitude_m + 15.0 * math.sin(0.08 * t + phase * 0.5)
        dx = vx + lateral_amp * freq * math.cos(freq * t + phase)
        dy = vy
        dz = 15.0 * 0.08 * math.cos(0.08 * t + phase * 0.5)
        return np.array([x, y, z], dtype=float), np.array([dx, dy, dz], dtype=float)

    return _traj


def _racetrack_trajectory(
    bounds: Any,
    altitude_m: float,
    speed_mps: float,
    seed: int,
) -> TrajectoryFn:
    """Simple racetrack oval for the factory."""
    rng = np.random.default_rng(seed)
    center, span = _bounds_center_and_span(bounds)
    cx = center[0] + rng.uniform(-0.15, 0.15) * span[0]
    cy = center[1] + rng.uniform(-0.15, 0.15) * span[1]
    straight = float(span.min()) * 0.2
    turn_r = float(span.min()) * 0.06
    track_len = 2.0 * straight + 2.0 * math.pi * turn_r
    phase_offset = rng.uniform(0, track_len)

    def _traj(t: float) -> tuple[np.ndarray, np.ndarray]:
        d = (speed_mps * t + phase_offset) % track_len
        if d < straight:
            x = cx - straight * 0.5 + d
            y = cy - turn_r
            vx, vy = speed_mps, 0.0
        elif d < straight + math.pi * turn_r:
            arc = d - straight
            ang = -math.pi / 2.0 + arc / turn_r
            x = cx + straight * 0.5 + turn_r * math.cos(ang)
            y = cy + turn_r * math.sin(ang)
            vx = -speed_mps * math.sin(ang)
            vy = speed_mps * math.cos(ang)
        elif d < 2 * straight + math.pi * turn_r:
            top_d = d - straight - math.pi * turn_r
            x = cx + straight * 0.5 - top_d
            y = cy + turn_r
            vx, vy = -speed_mps, 0.0
        else:
            arc = d - 2 * straight - math.pi * turn_r
            ang = math.pi / 2.0 + arc / turn_r
            x = cx - straight * 0.5 + turn_r * math.cos(ang)
            y = cy + turn_r * math.sin(ang)
            vx = -speed_mps * math.sin(ang)
            vy = speed_mps * math.cos(ang)
        z = altitude_m + 10.0 * math.sin(0.06 * t)
        dz = 10.0 * 0.06 * math.cos(0.06 * t)
        return np.array([x, y, z], dtype=float), np.array([vx, vy, dz], dtype=float)

    return _traj


def _waypoint_patrol_trajectory(
    bounds: Any,
    altitude_m: float,
    speed_mps: float,
    seed: int,
) -> TrajectoryFn:
    """Rectangular patrol path for the factory."""
    rng = np.random.default_rng(seed)
    center, span = _bounds_center_and_span(bounds)
    half_w = float(span[0]) * 0.15
    half_h = float(span[1]) * 0.15
    offset = rng.uniform(-0.1, 0.1, size=2) * span
    cx, cy = center[0] + offset[0], center[1] + offset[1]

    waypoints_2d = [
        np.array([cx - half_w, cy - half_h], dtype=float),
        np.array([cx + half_w, cy - half_h], dtype=float),
        np.array([cx + half_w, cy + half_h], dtype=float),
        np.array([cx - half_w, cy + half_h], dtype=float),
    ]
    # Close the loop
    waypoints_2d.append(waypoints_2d[0].copy())

    seg_lens: list[float] = []
    cum_dists: list[float] = [0.0]
    for i in range(len(waypoints_2d) - 1):
        d = float(np.linalg.norm(waypoints_2d[i + 1] - waypoints_2d[i]))
        seg_lens.append(max(d, 1e-6))
        cum_dists.append(cum_dists[-1] + max(d, 1e-6))
    total_dist = cum_dists[-1]

    def _traj(t: float) -> tuple[np.ndarray, np.ndarray]:
        dist = (speed_mps * max(t, 0.0)) % total_dist
        seg_idx = 0
        for i in range(len(cum_dists) - 1):
            if dist < cum_dists[i + 1]:
                seg_idx = i
                break
        else:
            seg_idx = len(seg_lens) - 1
        frac = (dist - cum_dists[seg_idx]) / max(seg_lens[seg_idx], 1e-9)
        frac = max(0.0, min(1.0, frac))
        p0 = waypoints_2d[seg_idx]
        p1 = waypoints_2d[seg_idx + 1]
        xy = p0 + (p1 - p0) * frac
        direction = p1 - p0
        norm = float(np.linalg.norm(direction))
        vel_xy = direction / max(norm, 1e-9) * speed_mps if norm > 1e-9 else np.zeros(2)
        z = altitude_m + 12.0 * math.sin(0.05 * t)
        dz = 12.0 * 0.05 * math.cos(0.05 * t)
        return (
            np.array([xy[0], xy[1], z], dtype=float),
            np.array([vel_xy[0], vel_xy[1], dz], dtype=float),
        )

    return _traj


def build_target_trajectory(
    preset: str,
    bounds: Any,
    altitude_m: float,
    speed_mps: float,
    seed: int = 42,
    terrain_height_fn: Callable[[float, float], float] | None = None,
) -> TrajectoryFn:
    """Build a trajectory function from a behaviour preset name.

    Parameters
    ----------
    preset:
        One of :data:`BEHAVIOR_PRESETS`.
    bounds:
        A mapping with keys ``x_min_m``, ``x_max_m``, ``y_min_m``, ``y_max_m``.
    altitude_m:
        Nominal flight altitude in metres.
    speed_mps:
        Nominal flight speed in m/s.
    seed:
        Random seed for deterministic output.
    terrain_height_fn:
        Optional ``(x, y) -> height`` callable used by evasive behaviours.
    """
    if preset not in BEHAVIOR_PRESETS:
        raise ValueError(
            f"Unknown behaviour preset {preset!r}. Choose from {sorted(BEHAVIOR_PRESETS)}."
        )

    rng = np.random.default_rng(seed)
    center, span = _bounds_center_and_span(bounds)

    if preset == "loiter":
        loiter_center = np.array(
            [
                center[0] + rng.uniform(-0.15, 0.15) * span[0],
                center[1] + rng.uniform(-0.15, 0.15) * span[1],
                altitude_m,
            ],
            dtype=float,
        )
        radius = float(span.min()) * 0.08
        return LoiterBehavior(
            center=loiter_center,
            radius_m=radius,
            speed_mps=speed_mps,
            clockwise=bool(rng.integers(0, 2)),
        )

    if preset == "transit":
        n_wps = rng.integers(3, 6)
        waypoints = []
        for _ in range(n_wps):
            wp = np.array(
                [
                    center[0] + rng.uniform(-0.3, 0.3) * span[0],
                    center[1] + rng.uniform(-0.3, 0.3) * span[1],
                    altitude_m + rng.uniform(-20.0, 20.0),
                ],
                dtype=float,
            )
            waypoints.append(wp)
        return TransitBehavior(waypoints=waypoints, speed_mps=speed_mps)

    if preset == "evasive":
        base = _sinusoid_trajectory(bounds, altitude_m, speed_mps, seed)
        if terrain_height_fn is None:
            terrain_height_fn = lambda x, y: 0.0  # noqa: E731  flat ground fallback
        return EvasiveBehavior(
            base_trajectory=base,
            terrain_height_fn=terrain_height_fn,
            nap_of_earth_agl_m=40.0,
            evasion_probability=0.3,
            evasion_duration_s=15.0,
            seed=seed,
        )

    if preset == "search_pattern":
        search_center = np.array(
            [
                center[0] + rng.uniform(-0.1, 0.1) * span[0],
                center[1] + rng.uniform(-0.1, 0.1) * span[1],
                altitude_m,
            ],
            dtype=float,
        )
        return SearchPatternBehavior(
            center=search_center,
            pattern="expanding_square",
            leg_length_m=float(span.min()) * 0.05,
            speed_mps=speed_mps,
            altitude_m=altitude_m,
        )

    if preset == "sinusoid":
        return _sinusoid_trajectory(bounds, altitude_m, speed_mps, seed)

    if preset == "racetrack":
        return _racetrack_trajectory(bounds, altitude_m, speed_mps, seed)

    if preset == "waypoint_patrol":
        return _waypoint_patrol_trajectory(bounds, altitude_m, speed_mps, seed)

    if preset == "mixed":
        # Composite: loiter for 30 s, then transit, then search.
        loiter_center = np.array(
            [
                center[0] + rng.uniform(-0.1, 0.1) * span[0],
                center[1] + rng.uniform(-0.1, 0.1) * span[1],
                altitude_m,
            ],
            dtype=float,
        )
        seg_loiter = LoiterBehavior(
            center=loiter_center,
            radius_m=float(span.min()) * 0.06,
            speed_mps=speed_mps,
            clockwise=True,
        )
        wp1 = loiter_center + np.array([float(span[0]) * 0.15, float(span[1]) * 0.1, 0.0])
        wp2 = loiter_center + np.array([float(span[0]) * 0.25, -float(span[1]) * 0.05, 0.0])
        seg_transit = TransitBehavior(
            waypoints=[loiter_center.copy(), wp1, wp2],
            speed_mps=speed_mps,
        )
        seg_search = SearchPatternBehavior(
            center=wp2,
            pattern="expanding_square",
            leg_length_m=float(span.min()) * 0.04,
            speed_mps=speed_mps,
            altitude_m=altitude_m,
        )
        return CompositeTrajectory(
            segments=[
                (0.0, seg_loiter),
                (30.0, seg_transit),
                (90.0, seg_search),
            ]
        )

    # Unreachable given the preset check above, but keep for safety.
    raise ValueError(f"Unhandled preset {preset!r}.")
