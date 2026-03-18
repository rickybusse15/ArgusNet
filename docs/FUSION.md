# Fused Track Schema and Lifecycle Contract

Stage 0 interface definition. Formalises the unified fused track schema,
lifecycle state machine, aging/staleness rules, and the relationship between
the Python IMM filter and the Rust tracker-core output.

---

## 1. Unified Fused Track Schema

A fused track is the authoritative, filtered estimate of a single target
produced by the tracker after associating and fusing raw sensor observations.

### 1.1 Canonical Rust type

```rust
use nalgebra::{Matrix6, Vector3};
use serde::{Deserialize, Serialize};

/// The unified output record for one tracked entity at one point in time.
///
/// This is the single schema used at every system boundary:
/// - gRPC response payload (`TrackState` proto field mapping below)
/// - Replay JSON `tracks[]` array entries
/// - Viewer display state
/// - Safety monitor input
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FusedTrack {
    // ------------------------------------------------------------------
    // Identity
    // ------------------------------------------------------------------

    /// Unique track identifier. Stable for the lifetime of the track.
    /// Format: opaque string, typically "t-{uuid4_short}".
    pub track_id: String,

    // ------------------------------------------------------------------
    // Kinematics  (local-ENU frame, metres / seconds)
    // ------------------------------------------------------------------

    /// Best-estimate position [x, y, z] in metres (local-ENU).
    pub position_m: Vector3<f64>,

    /// Best-estimate velocity [vx, vy, vz] in m/s.
    pub velocity_mps: Vector3<f64>,

    /// Best-estimate acceleration [ax, ay, az] in m/s².
    /// Derived from the IMM filter's CT model; zero if only CV is active.
    pub acceleration_mps2: Vector3<f64>,

    // ------------------------------------------------------------------
    // Uncertainty
    // ------------------------------------------------------------------

    /// 6×6 position+velocity covariance matrix (row-major).
    /// Upper-left 3×3 = position covariance (m²).
    /// Lower-right 3×3 = velocity covariance ((m/s)²).
    /// Off-diagonal 3×3 blocks = pos-vel cross-covariance.
    pub covariance: Matrix6<f64>,

    /// Scalar measurement noise standard deviation used in the most
    /// recent filter update (metres). Derived from sensor range × bearing_std.
    pub measurement_std_m: f64,

    // ------------------------------------------------------------------
    // Confidence and quality
    // ------------------------------------------------------------------

    /// Overall track confidence in [0.0, 1.0].
    ///
    /// Computed as:
    ///   confidence = quality_score * freshness_factor
    /// where freshness_factor decays linearly from 1.0 to 0.0 over
    /// max_coast_seconds when the track is coasting.
    pub confidence: f64,

    /// M-of-N quality score: fraction of recent confirmation_n frames
    /// that had at least one associated observation. Range [0.0, 1.0].
    pub quality_score: f64,

    /// IMM mode probability for the constant-velocity model. Range [0.0, 1.0].
    /// 1.0 - mode_probability_cv gives the coordinated-turn probability.
    pub mode_probability_cv: f64,

    // ------------------------------------------------------------------
    // Source history
    // ------------------------------------------------------------------

    /// Total number of filter update calls since track initialisation.
    pub update_count: u32,

    /// Number of consecutive frames since the last successful observation
    /// association (0 = updated this frame).
    pub stale_steps: u32,

    /// Wall-clock simulation time (seconds since epoch) of the most recent
    /// successful observation association.
    pub last_seen_s: f64,

    /// Set of sensor node IDs that contributed observations to this track
    /// within the last `source_history_window` frames.
    pub contributing_nodes: Vec<String>,

    // ------------------------------------------------------------------
    // Lifecycle
    // ------------------------------------------------------------------

    /// Current track lifecycle state. See Section 2 for the state machine.
    pub lifecycle_state: TrackLifecycleState,

    // ------------------------------------------------------------------
    // Timestamp
    // ------------------------------------------------------------------

    /// Simulation time at which this snapshot was produced (seconds).
    pub timestamp_s: f64,
}
```

### 1.2 Python equivalent (models.py)

The existing `TrackState` frozen dataclass in `models.py` maps to the schema
above as follows:

| `FusedTrack` field | `TrackState` field | Notes |
|--------------------|--------------------|-------|
| `track_id` | `track_id` | Same |
| `position_m` | `position` | Same |
| `velocity_mps` | `velocity` | Same |
| `acceleration_mps2` | (not present) | To be added |
| `covariance` | `covariance` (6×6 flattened to list in JSON) | Same |
| `measurement_std_m` | `measurement_std_m` | Same |
| `confidence` | (not present) | To be added; derived from `quality_score` + staleness |
| `quality_score` | `quality_score` | Optional field on current `TrackState` |
| `mode_probability_cv` | (not present) | To be added |
| `update_count` | `update_count` | Same |
| `stale_steps` | `stale_steps` | Same |
| `last_seen_s` | (not present) | To be added (currently on `ManagedTrack.last_update_time_s`) |
| `contributing_nodes` | (not present) | To be added |
| `lifecycle_state` | `lifecycle_state` (Optional[str]) | To be made non-optional and typed |
| `timestamp_s` | `timestamp_s` | Same |

**Migration path:** Add the missing fields to `TrackState` with sensible
defaults so existing replay JSON and gRPC callers are not broken (additive
change).

---

## 2. Track Lifecycle State Machine

### 2.1 States

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TrackLifecycleState {
    /// Track has been initialised from a first observation cluster but has
    /// not yet accumulated enough evidence to be trusted. Not reported to
    /// external consumers unless explicitly requested.
    Tentative,

    /// Track has passed the M-of-N confirmation gate and is being actively
    /// maintained. This is the normal operating state.
    Confirmed,

    /// Track was confirmed but has not received an associated observation
    /// for one or more frames. The filter continues to predict forward
    /// (coast) using the dynamic model. Reported to consumers with
    /// decreasing confidence.
    Coasting,

    /// Track has been deleted. Stale_steps exceeded the coast limit, or
    /// quality_score dropped below min_quality_score. The record is retained
    /// in memory for one additional frame so downstream consumers can detect
    /// the transition, then removed.
    Lost,
}
```

**Note on naming:** The existing Python code uses the string constant
`TRACK_STATE_DELETED = "deleted"`. The new canonical name for this state is
`Lost` to distinguish it from an intentional deletion (e.g. operator command).
The string serialisation for backward compatibility with replay JSON remains
`"deleted"` until the viewer and tests are updated together.

### 2.2 State transition diagram

```
                  [first observation cluster]
                           |
                           v
                      TENTATIVE
                     /         \
     M-of-N passed /             \ N frames elapsed,
                  /               \ < 2 updates
                 v                 v
           CONFIRMED             LOST
               |  ^
  missed frame |  | update received
               v  |
           COASTING
               |
               | stale_steps >= max_coast_frames
               | OR elapsed >= max_coast_seconds
               | OR quality_score < min_quality_score
               v
             LOST
```

### 2.3 Confirmation gate (M-of-N)

A tentative track transitions to `Confirmed` when at least `confirmation_m`
updates are recorded within a rolling window of the last `confirmation_n`
frames.

Default values (from `TrackLifecycleConfig` / `TrackerConfig`):
- `confirmation_m = 3`
- `confirmation_n = 5`

A tentative track transitions directly to `Lost` if `confirmation_n` frames
have elapsed and fewer than 2 updates have been recorded.

### 2.4 Coasting entry and exit

- `Confirmed` → `Coasting`: any frame with `stale_steps > 0`.
- `Coasting` → `Confirmed`: frame in which `stale_steps` returns to 0 (an
  observation was associated).
- `Coasting` → `Lost`: any of:
  - `stale_steps >= max_coast_frames` (default: 10)
  - `elapsed_since_last_update >= max_coast_seconds` (default: 5.0 s)
  - `quality_score < min_quality_score` (default: 0.1)

---

## 3. Aging and Staleness Rules

### 3.1 stale_steps counter

Incremented by 1 on every `mark_missed` call (a frame where no observation
was associated to this track). Reset to 0 on every `update` call.

### 3.2 Confidence decay during coasting

When `lifecycle_state == Coasting`, the reported `confidence` field decays
linearly:

```
freshness_factor = max(0.0, 1.0 - stale_steps / max_coast_frames)
confidence = quality_score * freshness_factor
```

When `lifecycle_state == Confirmed`, `freshness_factor = 1.0` and
`confidence = quality_score`.

### 3.3 Covariance growth during coasting

The Kalman predict step runs every frame regardless of whether an update
occurred. Process noise accumulates normally. After `max_coast_frames` without
an update, position uncertainty (standard deviation) grows by approximately:

```
sigma_pos_growth ~ process_accel_std * (dt * max_coast_frames)^2 / 2
```

At default values (`process_accel_std = 3.0 m/s²`, `dt = 0.25 s`,
`max_coast_frames = 10`): ~9.4 m additional position std. This is the
practical uncertainty bound a consumer should apply to coasting tracks.

### 3.4 Source history window

`contributing_nodes` is computed from the rolling update history. A node is
listed if it contributed at least one accepted observation within the last
`source_history_window` frames (default: same as `confirmation_n = 5`).

---

## 4. What Already Exists in tracker-core/lib.rs

The Rust `TrackerConfig` and the associated Rust track management implement
the same lifecycle as the Python `ManagedTrack` / `IMMTrack3D`. Key
correspondences:

| Rust field (TrackerConfig) | Python field | Default |
|---------------------------|--------------|---------|
| `max_stale_steps` | `TrackLifecycleConfig.max_coast_frames` | 10 |
| `max_coast_seconds` | `TrackLifecycleConfig.max_coast_seconds` | 5.0 |
| `min_quality_score` | `TrackLifecycleConfig.min_quality_score` | 0.1 |
| `confirmation_m` | `TrackLifecycleConfig.confirmation_m` | 3 |
| `confirmation_n` | `TrackLifecycleConfig.confirmation_n` | 5 |
| `cv_process_accel_std` | `AdaptiveFilterConfig.cv_accel_std` | 3.0 |
| `ct_process_accel_std` | `AdaptiveFilterConfig.ct_accel_std` | 8.0 |
| `ct_turn_rate_std` | `AdaptiveFilterConfig.ct_turn_rate_std` | 0.1 |

The Rust tracker is the **source of truth** for tracking output (per CLAUDE.md).
The Python `IMMTrack3D` / `ManagedTrack` are the simulation-side reference
implementation used for tests and offline analysis. The `FusedTrack` schema
defined here is derived from what Rust currently produces (via `TrackState`
proto) with the missing fields added.

---

## 5. Replay JSON Representation

Track entries in `replay.json` (`frames[].tracks[]`) currently serialise
`TrackState` fields. The additive changes required:

```json
{
  "track_id": "t-abc123",
  "timestamp_s": 42.0,
  "position": [100.0, 200.0, 180.0],
  "velocity": [5.0, 3.0, 0.2],
  "acceleration": [0.0, 0.0, 0.0],
  "covariance": [25.0, 0.0, 0.0, 0.0, 4.0, 0.0, ...],
  "measurement_std_m": 12.5,
  "confidence": 0.87,
  "quality_score": 0.87,
  "mode_probability_cv": 0.73,
  "update_count": 45,
  "stale_steps": 0,
  "last_seen_s": 42.0,
  "contributing_nodes": ["node-0", "node-3"],
  "lifecycle_state": "confirmed"
}
```

**Backward compatibility:** The viewer and `replay.py` treat missing new fields
as optional with defaults (`acceleration = [0,0,0]`, `confidence = quality_score`,
`mode_probability_cv = 1.0`, `last_seen_s = timestamp_s`,
`contributing_nodes = []`). This is enforced by the additive-change rule in
CLAUDE.md.

---

## 6. gRPC Proto Mapping

The current `tracker.proto` `TrackState` message carries a subset of the
`FusedTrack` fields. When proto is updated (requires updating both Python
bindings and Rust as per CLAUDE.md critical rules), the following new fields
are added:

```protobuf
message TrackState {
  // existing fields ...
  string track_id       = 1;
  double timestamp_s    = 2;
  repeated double position_m   = 3;  // [x, y, z]
  repeated double velocity_mps = 4;  // [vx, vy, vz]
  repeated double covariance   = 5;  // 36 elements, row-major
  double measurement_std_m     = 6;
  uint32 update_count          = 7;
  uint32 stale_steps           = 8;

  // new fields (additive) --
  repeated double acceleration_mps2 = 9;   // [ax, ay, az]
  double confidence                 = 10;
  double quality_score              = 11;
  double mode_probability_cv        = 12;
  double last_seen_s                = 13;
  repeated string contributing_nodes = 14;
  string lifecycle_state            = 15;  // "tentative"|"confirmed"|"coasting"|"deleted"
}
```
