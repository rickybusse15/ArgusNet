# P4.8d: the motion substrate

**Status:** delivered 2026-09-19. First slice of P4.8d, after P4.8c (the first
stage bodies). Package ID `P4.8d.1`.

P4.8c delivered nine of the eleven stage bodies. The gap it left is the one
that matters most: `Stage::SampleControllersAndBattery` has only its battery
half, so `Simulation` still cannot move its own platforms. Positions are
prescribed from outside through `Simulation::world_mut`, which P4.8c added for
exactly that reason, and P4.8's exit criterion — *"a full representative run is
owned by Rust"* — is unreachable until a stage computes them.

This slice is not the controllers either. It is the layer they stand on.

## Why the substrate goes first

The reference says why these helpers have never been extracted, at `sim.py`
line 136:

> Extraction is deferred because the controllers directly call module-level
> helpers (`compose_air_state`, `collision_aware_position`,
> `advance_along_polyline`, `clamp_to_bounds`, `within_bounds`,
> `_apply_weather_adjustments`) that are defined later in this file, which
> would create a circular import if `controllers.py` tried to import them back
> from `sim.py`.

That is an import-cycle problem, and it is a Python problem. Rust does not have
it. So the six helpers move now, as `argusnet-sim::motion`, and the three
controller classes — `FollowPathController`,
`ObservationTriggeredFollowController`, `LaunchableTrajectoryController` —
follow in the next slice with something to stand on.

`_apply_weather_adjustments` is on the reference's list and is **not** in this
slice. It is a wrapper over a trajectory rather than a kinematic helper, and it
belongs with the controllers that it wraps.

## Almost none of the arithmetic is new

Worth stating plainly, because it bounds what this slice can have broken. The
terrain sampling is `argusnet-world`'s already (P4.3), and
`CylinderObstacle.push_outside_xy` has been a
`call("obstacle", …, method="push")` into this workspace since Phase 3 — the
Python is a shim over Rust that already exists.

What is new is the **composition**: how altitude, vertical rate and the
collision loop are assembled from those reads. That is the same shape as
P4.8c's battery slice, where the arithmetic was `argusnet_planning::battery`
and the new code was the accounting loop around it.

## Capabilities, not a world dependency

`argusnet-sim` does not depend on `argusnet-world`, and this slice does not add
that dependency. The reference's `terrain: object` parameter is duck-typed to
exactly three methods, so `TerrainSampler` *is* its real contract written down;
`ObstacleSampler` is the same for the one obstacle read.

This follows P4.8b's truth capability — a stage gets the narrowest thing it can
do its job with. Wiring these traits to `argusnet-world` is the adapter the
controller slice needs and is deliberately not here: nothing calls the
substrate yet, so an adapter would be an untested seam.

## The oracle, and why the keys are half the test

Re-checking terrain and obstacle geometry here would test P4.3 and Phase 3 over
again. So `scripts/capture_motion_fixture.py` records every read the reference
made — keyed by `(x, y)` for heights, `(x, y, delta)` for gradients and
`(x, y, z, min_agl)` for clamps — and the test replays them through a stub that
**panics on a miss**.

That makes the keys part of the assertion, and it is stronger than driving a
real terrain, not weaker. There are three different gradient steps in this
codebase (`gradient_at` defaults to `base_resolution * 0.5`, `slope_rad_at`
uses `1.0`, `slope_rad_at_many` uses `max(base_resolution * 0.5, 0.25)`), and a
real terrain would answer the wrong one with a plausible number. The stub
fails. Two mutants confirm it: asking for an explicit `0.5` step, and calling
`clamp_altitude` on the path where the reference takes a plain `max`, are both
caught by a lookup miss rather than by a wrong value.

## Four behaviours worth naming

### The vertical rate is coupled to the terrain

`vz = dot(gradient, xy_velocity) + agl_rate`. Flying level over rising ground
*is* climbing. A port returning `agl_rate` alone is correct on flat ground and
wrong everywhere else — which is why the fixture's terrain is a tilted plane
plus a bump rather than a flat sheet, and why a coverage check asserts some
recorded gradient is non-trivial.

### The descent guard fires on a conjunction

```python
if z_m > nominal_z_m and vz_mps < 0.0:
    vz_mps = 0.0
```

Only when the clamp actually raised the altitude **and** the rate is negative.
Three cases pin it — `clamped_and_descending`, `clamped_and_ascending`,
`downhill_descent_unclamped` — and dropping either half of the conjunction is
caught by a different one of them.

### `compose_air_state`'s guard is an `or` over two arguments

`target_position is None or target_altitude_offset_m is None`, so *a target
position without an offset behaves exactly like no target at all*. The port
folds the four target parameters into one `TargetAltitudeHold` struct, which
makes that guard unrepresentable rather than merely tested — there is no state
in which half is supplied.

Two things travel with that branch and look like omissions until you check.
The delegating path forwards `nominal_agl_m` as `agl_m` and does **not**
forward `terrain_speed_adaptation`, so a target-following platform never gets
speed adaptation. And the target branch takes a plain `max` against
`ground + min_agl` rather than calling `clamp_altitude` — the same number by a
different route, which is the mutant the oracle catches.

### The collision loop re-clamps, and not converging is not an error

Probe, push, **re-clamp the altitude at the new xy**, probe again, up to eight
times. The re-clamp is what upholds CLAUDE.md's "physical collision must never
push entities below terrain": pushing sideways can move a platform over higher
ground.

After eight iterations the reference logs a warning and returns its best
effort. It does not raise and does not revert. That is preserved, with one
change: the fact is returned as `CollisionResolution::unresolved` instead of
existing only in a log line, so stage 4's safety monitor can see that a
position is *known* to be unresolved.

## A finding: `point_collides` is not a geometry test

`ObstacleLayer.point_collides` filters on
`blocker_type not in {"building", "wall"}` *before* testing containment. A
platform squarely inside a forest stand does not collide with it.

This is recorded rather than changed, and it is in the fixture twice — the same
point, the same cylinder, one `building` and one `vegetation` — so the filter,
not the geometry, is what separates them. It was found by the fixture's own
coverage check: the first draft used `blocker_type="structure"`, nothing ever
collided, and `the push loop did not iterate` failed at capture time. Without
that check the fixture would have recorded a tower nothing hits and blessed a
port with no collision handling at all.

## An equivalent mutant, proven rather than asserted

`max(0.7, 1 - min(|grad|, 1) * 0.3)` has two saturations, and they are the same
bound reached twice: with `min(|g|, 1) ∈ [0, 1]` the expression is already in
`[0.7, 1.0]`, so the `max` can never bind. The `0.7 → 0.0` mutant therefore
survives every fixture case, and no fixture case could catch it.

The floor is **dead code in the reference**. It is preserved rather than
deleted — this slice ports, it does not tidy, and the line becomes load-bearing
the moment the `0.3` coefficient moves. `the_steepness_floor_is_unreachable`
sweeps gradient magnitudes past the cap and asserts the floor never binds, so
if that ever changes something says so.

## Verification

| Gate | Result |
|---|---|
| `cargo test --workspace --exclude argusnet-viewer` | **842 passing**, exit 0 |
| `cargo test -p argusnet-viewer` | **99 passing**, exit 0 |
| `cargo clippy --workspace --all-targets -- -D warnings` | exit 0 |
| `cargo fmt --all -- --check` | exit 0 |
| `python3 -m pytest tests/ -q` | **981 passed**, 9 skipped, 65 subtests, exit 0 |

Every exit code was read directly from cargo and pytest, not from the tail of a
pipeline — `exit=$?` after `cmd | tail` reports `tail`'s status, which has
misreported a Rust result in this repo before.

11 new tests: 6 fixture-parity, 5 unit. 48 recorded cases.

**25 distinct mutants, 24 caught, 1 proven equivalent.** Three survived a first
pass and are the reason the fixture grew:

| survivor | what was too weak |
|---|---|
| the collision loop drops its re-clamp | every case flew high enough that the clamp was a no-op |
| `contains` ignores the margin | one case decided every rejection, on the x-min edge alone |
| the steepness floor is `0.0` | equivalent — see above |

The first two are now `push_over_higher_ground_reclamps` (a low pass over the
tower's foot, where the push does raise the altitude) and one rejection case
per edge. Both have coverage assertions in the capture script so they cannot
quietly stop discriminating.

## Not verified on CI

`vars.CI_RUNNER` points at a self-hosted runner registered with the **`X64`**
label on an arm64 Mac, so every Rust job fails at link time with
`xcrun: error: unable to load libxcrun … (have 'arm64,arm64e', need 'x86_64')`
before it compiles anything. Twelve runs are queued behind it.

Every result above is **locally verified** on `1.98.0-aarch64-apple-darwin`,
the pinned toolchain. Not pushed, not PR-checked, not merged — the distinction
DELIVERY_PLAN §2 asks for. Re-registering the runner with the `osx-arm64`
package is a prerequisite for trusting a green tick on this branch.

## Next

**`P4.8d.2` — the controller bodies.** `FollowPathController`,
`ObservationTriggeredFollowController` and `LaunchableTrajectoryController`
from `sim.py`, plus `_apply_weather_adjustments`, on top of this substrate.

Prerequisites, both met by this slice: the six motion helpers, and a stub-able
terrain/obstacle contract to test them against.

Deliberately left here for it:

* ~~**No adapter from `argusnet-world` to
  `TerrainSampler`/`ObstacleSampler`.**~~ **Delivered** as `P4.8d.1b`,
  `rust/argusnet-sim/src/motion_world.rs` — see the section below.
* **The stage handler is not wired.** `Stage::SampleControllersAndBattery`
  still runs only `BatteryStage`. Wiring a second handler into one stage is the
  controller slice's problem, not the substrate's.
* **`_apply_weather_adjustments`** — on the reference's own list of blocking
  helpers, but a trajectory wrapper rather than a kinematic one. It belongs
  with the controllers.

Then `P4.8e` (widening `NodeState` with `battery_fraction`, `sensor_type`,
`fov_half_angle_deg`, `max_range_m` — a proto change across 31 construction
sites, which CLAUDE.md requires be coordinated across the Python bindings and
three Rust crates, and which should land on a quiet tree), and `P4.8f`
(`CommitSnapshot`, checkpoints and the Rust replay writer).

---

# P4.8d.1b: the world adapter

**Status:** delivered 2026-09-19. Completes P4.8d.1 by supplying the
prerequisite it deferred.

`WorldTerrain` and `WorldObstacles` implement the substrate's two contracts
over `argusnet-world`. Deferred from the first slice because nothing called
the substrate yet and an adapter nobody exercises is an untested seam
pretending to be progress; delivered now, ahead of the controllers, so
P4.8d.2 starts against something that has been run.

## `motion.rs` still does not know the world exists

The adapter is a separate module, and the separation is load-bearing rather
than tidy. The substrate's tests drive it with a recorded oracle that panics
on an unrecorded read, and they can only do that because the contract is a
trait. Merging the two files would make the fixture tests quietly depend on
a real terrain and stop them testing what they claim to.

`argusnet-world` becomes a direct dependency of `argusnet-sim` for the first
time here, used by this module alone. It was already in the tree transitively
through `argusnet-planning`.

## The default gradient step is resolved here, on purpose

`TerrainSampler::gradient_at` takes `Option<f64>` precisely so the substrate
does not have to pick one of this codebase's three gradient steps.
`TerrainQuery::gradient_at` requires one, so the adapter resolves it:
`max(base_resolution * 0.5, 0.25)`, matching the reference's
`TerrainLayer.gradient_at`.

The resolution is a constructor argument rather than something read off the
terrain, because `TerrainQuery` does not expose one — a tiled field, an
analytic surface and an imported raster do not agree on what "the resolution"
is, and guessing on their behalf is how the wrong step gets picked.

The floor applies to the **default only**. A caller passing an explicit step
below it gets exactly what it asked for: `slope_rad_at` legitimately uses
`1.0`, and a caller may legitimately want something finer.

## The blocker-type filter is not re-implemented

`ObstacleLayer::point_collides` already tests
`matches!(blocker_type.as_str(), "building" | "wall")` before containment —
the same rule the Python `ObstacleLayer` applies, and the finding P4.8d.1
recorded. The adapter forwards to it and adds nothing. A test pushes the same
cylinder at the same point as `Building` and as `Vegetation` to pin that,
because a future reader "fixing" the adapter by adding its own containment
test would silently start colliding with forest.

## Verification

6 unit tests. **6 mutants, 6 caught.** One survived a first pass: replacing
the forwarded margin with `0.0` passed, because clearing to exactly the
radius lands a hair outside it in floating point and the test only asserted
"outside the radius". It now measures two different margins against each
other.

`the_substrate_runs_against_a_real_world` composes both halves through the
call a controller will actually make — the one thing neither file's own tests
can check alone.

| Gate | Result |
|---|---|
| `cargo test --workspace --exclude argusnet-viewer` | **848 passing**, exit 0 |
| `cargo test -p argusnet-viewer` | **99 passing**, exit 0 |
| `cargo clippy --workspace --all-targets -- -D warnings` | exit 0 |
| `cargo fmt --all -- --check` | exit 0 |

## Next

Unchanged: **`P4.8d.2`, the controller bodies**. Its two remaining
prerequisites from the list above still stand — `Stage::SampleControllersAndBattery`
runs only `BatteryStage`, and `_apply_weather_adjustments` is unported.

One requirement has been **added** since P4.8d.1 was written: P4.1d gave
`StageHandler` a `reset`/`checkpoint_state`/`restore_state` contract, and a
controller is stateful (`FollowPathController` carries a `_state` dict with a
cached timestamp and rotation phase). The controller stage must implement it,
or a resumed mission flies a different path than an uninterrupted one. See
[CHECKPOINT_COMPLETENESS.md](CHECKPOINT_COMPLETENESS.md).
