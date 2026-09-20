# P4.4: three things the VIO port must not inherit

**Status:** measured 2026-09-10, before the estimator moves. Follows the
reachability inventory in `VIO_INVENTORY.md`.

P4.4 states two requirements that read as boilerplate until you look at
`EKFVIO`: *"No manufactured confidence on a failed estimator"* and *"Define
invalid or stale pose behavior explicitly."* Both are violated today, and a
faithful transcription would carry the violations into Rust.

Each finding below was checked by running the reference, not by reading it.

## 1. The estimator manufactures confidence when every feature fails the gate

`process_image` gates each matched feature on a 3σ Mahalanobis test. When
fewer than two survive, it falls back to a scalar update built from the
**feature count**:

```python
scale_factor = matched / (self._min_features * 2.0) - 0.5
y_meas = np.array([scale_factor * 0.1])
H[0, 8] = 1.0 / self._focal_length_px
```

`y_meas` is not a measurement. It is a number derived from how many features
matched, applied to the **position** state, with a noise covariance implying
real precision.

Driven with eight features displaced by 9×10⁴ px — outliers by any standard,
and all eight rejected by the gate:

| | value |
|---|---|
| `trace(P)` before | 3.366833 |
| `trace(P)` after | **3.066745** |
| position moved | **23.1 m** |

The estimator was handed nothing it believed, moved the position estimate 23
metres, and became **more** confident. That is precisely the failure mode P4.4
names.

Note also that `sim.py` reads `trace(covariance[6:9, 6:9])` as its confidence
signal — so this fabrication feeds the number the simulation actually uses.

**For the port:** a gate that rejects every feature is an estimator with no
information this frame. The correct behaviour is to propagate and let the
covariance **grow**, and to report the failure rather than hide it. That is a
behaviour change, and it is the one P4.4 asks for.

## 2. One branch of the update is unreachable

```python
if matched >= 2:
    ...
elif matched >= self._min_features:      # min_features default 6
```

The `elif` requires `matched < 2` **and** `matched >= min_features`. For any
`min_features >= 2` that is unsatisfiable:

| `min_features` | values of `matched` reaching the `elif` |
|---|---|
| 6 (default) | none — dead |
| 2 | none — dead |
| 1 | `matched == 1` |

Its comment reads *"Fewer than 2 features matched: original scalar update"*,
which describes an intent the condition does not implement. The duplicated
scalar block inside the first branch is what actually runs.

A neighbouring line, `matched = max(matched, 0)`, is also a no-op: `matched`
is a `len()`.

**For the port:** do not carry either across. Dead code ported is dead code
given a second life and a reviewer's attention.

## 3. A backwards timestamp drops the frame and leaves the feature store stale

```python
dt_total = timestamp_s - self._timestamp_s
if dt_total < 0:
    return self.get_state()
```

The early return happens **before** `_prev_features` is refreshed. Measured:

| | feature ids in the store |
|---|---|
| after a frame at t=1.0 | `{1}` |
| after a frame at t=0.5 | `{1}` — unchanged |

So the out-of-order frame is discarded silently, and the *next* forward frame
matches its features against a store two frames old while computing `dt_total`
from its own timestamp. The optical-flow residual then divides a two-frame
displacement by a one-frame interval.

Note the asymmetry: `dt_total == 0` is **not** caught by `< 0`, so a repeated
timestamp proceeds with `dt_total = 0`, making the predicted flow identically
zero and the residual the raw displacement.

**For the port:** P4.4 says define this explicitly. Out-of-order and duplicate
timestamps are distinct cases and need stated, tested answers — not an early
return whose side effect is invisible.

## Scope, and what this implies

Per `VIO_INVENTORY.md`, only `EKFVIO`, `VisualFeature`, `VIOState`,
`_rotvec_to_matrix` and `transforms.py` are reachable. `SimpleVIO` and the
`VIOBackend` protocol have no callers and no tests outside `vio.py`, so they
are not in the port either.

The consumer surface is narrow, which makes validation tractable: `sim.py`
reads only `position`, `velocity`, and `trace(covariance[6:9, 6:9])`.

`nalgebra 0.33` is already a workspace dependency (used by `argusnet-core`,
`argusnet-proto`, `argusnet-types`, `safety-engine`), so the 15-state filter
needs no new dependency.

## How the port should be validated

The prediction step is deterministic given buffered IMU, so it can be compared
value-for-value against recorded Python — the approach that worked for P4.2a
and P4.5a.

The update step cannot be compared that way where behaviour deliberately
diverges. Findings 1 and 3 are changes, so parity fixtures must record the
reference's behaviour as **evidence of what was changed**, and the Rust tests
assert the new contract. Mixing those two would let a fixture silently bless
the fabricated measurement.

Specifically, P4.4's own list — zero motion, turning, timestamp gaps, bias
drift, GNSS loss/recovery, singular or invalid covariance — should be
tested against the *stated* contract, and the three findings above should each
have a test that fails if the old behaviour returns.

One more to test rather than assume: `np.linalg.inv(S_batch)` has no
singularity guard, and the Mahalanobis gate's `np.linalg.solve` throws on a
singular per-feature innovation. P4.4 names singular covariance explicitly.

---

# P4.4 delivery: the estimator's surface (2026-09-16)

The internals were already ported and fixture-tested (`vio_prediction.json`,
`vio_update.json`). This slice delivers the part a caller touches, which is what
P4.4 actually names: *"no manufactured confidence on a failed estimator"* and
*"define invalid or stale pose behavior explicitly"*.

Fixture: `tests/fixtures/vio_surface.json` — 7 scenes, 40 KB.

## The port surface is smaller than the package, again

`VIO_INVENTORY.md` established that the surface is `vio.py` + `transforms.py`
rather than the 2,292 lines in the package. Re-running reachability for this
slice narrowed it further, and the numbers are worth recording:

| symbol | references outside its own file |
|---|---|
| `EKFVIO` | **5** — the live backend, constructed at `sim.py:5188` |
| `SimpleVIO` | **0** |
| `TransformTree` | **0** |
| `identity_se3`, `compose`, `invert`, `from_xyz_yaw` | **0** |
| `SE3` | 2 — `vio.py` (live) and `relocalization.py` (itself unreachable) |

So `transforms.py` needed no port at all. `EKFVIO` uses `SE3` purely as a
container for `R` and `t`, and `rotvec_to_matrix` — the only thing that builds
that `R` — was already ported in P4.2b. Porting `TransformTree` and the four
free functions would have been ~120 lines producing nothing any caller reaches,
which is precisely what the inventory exists to prevent.

## Four surface behaviours with nothing to do with the mathematics

Each has a test that fails if it regresses.

1. **The bias blocks are reversed between the vector and the state.** The state
   is `rotation(0:3) velocity(3:6) position(6:9) accel(9:12) gyro(12:15)` —
   accel first — while `VIOState(...)` lists `bias_gyro` before `bias_accel`. A
   port reading them positionally swaps two small 3-vectors near zero: nothing
   crashes, and the drift is quietly wrong.

2. **`tracked_feature_count` has two values for one field.** `process_image`
   sets the real matched count; `get_state` hardcodes **zero** even while the
   filter is tracking. Preserved rather than unified — a caller comparing the
   two paths would otherwise see a difference this port invented.

3. **An out-of-order frame returns a state stamped with the filter's clock, not
   the frame's.** The reference returns `get_state()` before touching anything,
   so the clock does not move. A caller stamping that pose with the frame's own
   time would date it wrongly. `ImageOutcome.update` is `None` on that path —
   distinct from `Some(NoInformation)`, which means the frame *was* processed
   and carried nothing.

4. **A repeated timestamp is not out of order.** The gate is `dt_total < 0`, not
   `<= 0`, so `dt == 0` runs the full predict-and-update. Off-by-one here
   silently drops frames.

## A finding the fixture surfaced: gyro bias is unobservable

The bias-swap guard passes for a **narrower reason than it first reads**, and
the capture script now asserts that reason rather than relying on it silently.

Gyro bias is **exactly zero in every scene**, including `both_biases`, whose
input gyro rate is 0.2 rad/s. The optical-flow measurement model observes
horizontal velocity only, so nothing in it makes gyro bias observable and that
block never leaves its initial zero. A swap is caught because accel bias reaches
~1e-3 while gyro stays at 0 — the **asymmetry** does the work, not two
independently driven biases.

That matters beyond the test: `VIOState.bias_gyro` is a constant zero on this
backend, so a consumer treating it as an estimate is reading a placeholder.
Driving it would need a measurement that sees rotation, which `EKFVIO` does not
have. The capture script fails if gyro bias ever moves, so the guard is
revisited rather than continuing to pass for a reason that no longer holds.

## Still open

`NoPose` has one variant today (`Uninitialized`). P4.4 also names GNSS
loss/recovery and singular covariance; `update_optical_flow` already reports
`NoInformation::SingularInnovation`, but there is no GNSS fusion on this
backend to lose — `gnss_fusion.py` is one of the unreachable modules. Adding a
variant for a path nothing reaches would be the stub-porting the inventory warns
against.
