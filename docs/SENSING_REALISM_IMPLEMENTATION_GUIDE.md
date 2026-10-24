# Sensing realism implementation guide

**Date:** 2026-10-23. **Status:** proposed implementation contracts and validation
work. This document does not introduce executable configuration, select hardware,
or establish survey accuracy.

## 1. Scope and requirements

Implement sensing realism in the existing Rust runtime, from sensor acquisition
through mapping and quality reporting. Follow [simulation fidelity](SIMULATION_FIDELITY.md),
[runtime architecture](RUST_RUNTIME_ARCHITECTURE.md) and
[ADR-009](adr/009-independently-scheduled-runtime.md). The
[pod plan](SENSOR_POD_MAPPING.md) owns the preliminary installed mass allocation.

- Aircraft: lift-and-cruise VTOL, at most **55 lb / 24.94758035 kg total takeoff
  mass**. Battery-only versus fuel/electric hybrid remains unresolved.
- Scanning assembly: at most **10 lb / 4.5359237 kg**, excluding compute,
  storage, cooling and batteries. Excluded equipment still counts in aircraft mass.
- Moving pod: non-rotating LiDAR **below** two side-by-side visible cameras,
  thermal aperture above or nearby, mapping IMU and a rigid common sensor carrier.
  Each aperture has its own optical path/window compatible with its wavelength.
- Body-fixed forward navigation camera: independent of pod pointing, and the
  **primary flight-safety sensor for obstacle avoidance** (section 8.4). The pod
  is a mapping payload; avoidance must not depend on where it points.
- Enclosed gimbal: actual pose, joint limits, settling, vibration and housing
  occlusion must be represented. The concept illustration is not a validated mechanism.
- Fully solid-state versus internal moving scan optics remains a component
  decision. A rectangular aperture does not establish the scanner mechanism.

**Decided 2026-10-23 (maintainer):**

- **Gimbal:** 3-axis (yaw, pitch, roll). No joint limits are set: rotation is
  continuous on every axis. Slew rate, settling and housing occlusion are still
  modelled; their values are assumed until a mechanism is chosen.
- **Pod sensors:** all of them — LiDAR, two visible cameras, thermal and the
  mapping IMU on one carrier. The two cameras' roles (stereo pair or
  wide/detail) are not yet decided.
- **LiDAR:** fixed rectangular field of view. Its size is not yet known; the
  profile carries an assumed value.
- **Flight:** cruise is the operating mode. Hover is for take-off and landing
  only, so the planner never plans a hover station for sensing. Cruise dynamics
  (minimum speed, turn radius) are not yet known.

Altitude, speed, accuracy, optics/baseline, device models, power and endurance are
unresolved. Treat **60–120 m AGL and 250–300 m LiDAR range on 10%-reflective targets
in bright daylight as provisional sizing assumptions**, not selected operating
limits or instructions to change the legacy scanner defaults.

## 2. Current baseline and integration map

| Existing location | Baseline and proposed work |
| --- | --- |
| `rust/argusnet-sim/src/scenario.rs` | One observation sensor per node; add an optional versioned rig alongside legacy parsing. Include resolved calibration/model content in run identity. |
| `rust/argusnet-sim/src/range_scan.rs` | Velocity-heading ray fan, default 400 m cutoff and 0.3 m range noise. Preserve the legacy path; introduce an opt-in acquisition-time rig path. |
| `rust/argusnet-sim/src/observe.rs` | Bearing observations and thermal duplicates are not camera images or radiometry. Keep the products distinct. |
| `rust/argusnet-sim/src/scheduler/`, `delivery.rs` | Reuse scheduling/delivery foundations. Existing tick-aligned bearing capture does not establish off-boundary imaging or per-firing LiDAR capture. |
| `rust/argusnet-sim/src/deterministic.rs` | Reuse independent keyed random streams; retain correlated error state in checkpoints. |
| `rust/argusnet-sim/src/mapping.rs`, `belief_world.rs` | Integrate accepted evidence and quality; legacy circular coverage is not the pod's optical footprint. |
| `rust/argusnet-mapping/src/belief.rs` | Reuse domain belief ownership/revisions rather than creating a viewer-owned second map. |
| `rust/argusnet-planning/src/`, `rust/argusnet-sim/src/frontier.rs`, `mission_driver.rs` | Frontier, POI and routing read unmeasured share, hit counts and dwell; no commanded pointing. Add requirement-driven deficits, view planning and pointing arbitration (section 8). |
| `rust/argusnet-sim/src/replay.rs`, `snapshot.rs`, `rust/argusnet-types/src/lib.rs` | Add provenance, quality and actual sensor poses with consumer updates; decide shared type placement from actual consumers. |
| `rust/argusnet-viewer/src/` | Display backend footprints, age and uncertainty. The operator camera never produces sensor evidence. |

Read [state ownership](STATE_OWNERSHIP.md) before changing authority. Production
models belong in Rust; Python may support offline fitting/evaluation and reference
compatibility. Proto changes update both Python and Rust bindings/consumers.

## 3. Configuration and measurement contracts

Define typed contracts before wiring behavior. The following are required
concepts, not finalized API names or accepted configuration keys.

| Contract | Required contents |
| --- | --- |
| Rig profile | Version, unique sensor roles/IDs, body/pod attachments, model versions, supported effects/products, mass categories, power and gimbal envelope. |
| Calibration | Identity/epoch, transforms and covariance, intrinsics/distortion, camera baseline, optical transmission and calibration evidence. |
| Gimbal state | Commanded and measured pose/rates, limits, lag/settling, encoder health, covariance and source time. |
| Measurement envelope | Sensor/source identity and epoch, sequence, capture interval, raw clock, clock-conversion identity/uncertainty, availability and receipt times, calibration/model identity, validity and payload reference. |
| LiDAR evidence | Per-firing time offsets, beam directions, ranges, return index, intensity units/model, uncertainty and no-return/saturation status. Truth IDs belong in evaluation-only sidecars. |
| Camera evidence | Image reference/hash, intrinsics, exposure, gain, shutter/row timing, synchronization and validity. |
| Quality result | Metric, units, region/time, source evidence, estimated uncertainty, acceptance target and reason. Unavailable is distinct from zero or passing. |

Validate finite values, positive rates/ranges, valid rotations, units/frames,
unique identities, calibration references, supported capabilities and both mass
limits. Reject unsupported rigs/products before mutating runtime state; never
accept the profile while silently running the legacy model.

Store large images/clouds in bounded payload storage with hashes and explicit
retention/gap records; reference them from replay. Do not duplicate images in
each display frame. Missing payloads must be visible during reevaluation.

## 4. Moving geometry and acquisition time

Document frame axes, handedness and transform direction. Explicitly convert
device NED/FRD conventions; use metres and radians internally. Here `T_A_B`
maps coordinates from B into A:

```text
T_world_sensor(t) = T_world_body(t) * T_body_base * T_base_pod(t) * T_pod_sensor
T_world_nav(t)    = T_world_body(t) * T_body_nav
```

Stacked apertures have distinct lever arms; do not model them as coincident.
Generate synthetic observations from physical truth, but georeference them with
the estimated pose and imperfect calibration available to ArgusNet. Using truth
transforms in both paths hides localization and boresight errors. Airframe GNSS
antennas have moving offsets relative to the pod IMU. Define that relationship
and correlated attitude information rather than fusing shared evidence twice.

Model slew/acceleration limits, joint stops, settling, backlash and vibration as
declared capabilities. Respect airframe/housing occlusion for each aperture.
Commanded pointing is never substituted for measured pointing.

Evaluate pose history at firing/exposure time. Deskew moving LiDAR sweeps and
model camera exposure/rolling shutter. Unsupported off-boundary interpolation
or extrapolation must flag/reject the profile, not silently snap capture to the
latest tick. Preserve raw clocks, conversion versions, drift and reboot epochs.
Keep acquisition rate, delivery latency, processing time and rendering separate.
Latency alone does not create geometric error if capture time is correctly retained.

## 5. LiDAR range and return realism

Represent range as a conditional performance envelope: wavelength, reflectivity,
ambient light, incidence angle, temperature, detection probability and false-alarm
definition. Tag coefficients as vendor-supported, measured, assumed or unsupported.
Do not derive a universal reflectivity scaling law from one advertised maximum.

Over flat ground, `slant_range = AGL / cos(theta)`, where theta is the ray's angle
from downward vertical, including both pod pointing and scan angle:

| AGL | Nadir | 45 degrees off nadir |
| --- | ---: | ---: |
| 60 m | 60 m | 84.9 m |
| 100 m | 100 m | 141.4 m |
| 120 m | 120 m | 169.7 m |

Intersect actual terrain/structures in general scenes. The provisional 250–300 m
low-reflectivity daylight target provides sizing headroom for this example; it
does not guarantee returns through vegetation, fog, wet surfaces or grazing
angles. Evaluate the furthest required ray and ground-return density, not just
the boresight range.

Implement the opt-in LiDAR pipeline in order:

1. Generate the actual beam pattern with per-firing acquisition times. Match
   repetitive/non-repetitive scan behavior and dwell; non-rotating is not 360-degree.
2. Transform rays and trace geometry with housing, canopy and supported material
   effects. Finite beam footprint limits thin-object detection; no perfect wire returns.
3. Admit returns using calibrated or explicitly assumed probability curves.
   Enable intensity effects only when the model supports them.
4. Apply angular error, range noise, persistent bias, calibration error, dropout,
   false returns and quantization separately. Precision measured at short range
   is not automatically valid across the entire operating envelope.
5. Emit valid returns/misses with timing and uncertainty. Multiple returns require
   both device support and an actual multi-return model, not duplicated points.

For small angles, beam footprint is approximately range times beam divergence
in radians; transverse sample spacing is range times angular spacing. Neither
quantity is range precision. Under uniform straight-flight assumptions, estimate
ground density as `usable_ground_returns_per_second / (ground_speed * swath_width)`.
Then verify spatial density, overlap and holes from the actual accepted samples.

The existing return classes and world-derived wire identities are oracle
assistance. Label profiles retaining them accordingly. Operational segmentation
and wire fitting must consume measurements; truth is only for generation/scoring.

## 6. Visible and thermal realism

Visible cameras need intrinsics, resolution, distortion, baseline/extrinsics,
shutter/exposure, noise, illumination, saturation, blur and occlusion. For nadir
fronto-parallel geometry, `GSD = height * pixel_pitch / focal_length`. For
rectified stereo, `depth_std ≈ depth² * disparity_std / (focal_length_pixels * baseline)`.
State assumptions and units. Oblique surfaces, poor texture and occlusion require
projection/matching validity, not an always-valid depth sample. Two cameras do
not establish useful aerial stereo depth; a wide/detail pair is a separate capability.

Start with recorded-image ingestion. Synthetic image perception needs dedicated
calibrated offscreen capture, never viewer screenshots. Preserve inputs for
deterministic algorithm reevaluation; do not promise identical GPU pixels across
unrelated drivers.

Thermal profiles declare spectral response, resolution, NETD, gain, nonuniformity
correction, drift, emissivity, reflected apparent temperature, atmospheric effects
and window transmission as supported. NETD is not absolute temperature accuracy.
Separate relative anomaly detection from radiometric temperature products. A
recolored visible image is not a thermal model. Register thermal observations
using their own optics, pose and spatial/temperature uncertainty.

## 7. Mapping quality

Separate observed footprint, accepted geometry, reconstruction quality and mission
completion. Repeated rendering/reprocessing of the same evidence must not inflate
coverage or reduce persistent bias.

Admission checks epoch/calibration compatibility, capture-time pose, clock health,
measurement validity and supported lateness. Record rejection reasons. Late
evidence follows declared estimator/map revision policy, not pose at receipt time.

Propagate position, attitude, timing, boresight and measurement uncertainty. At
100 m, 0.1 degree pointing error contributes about 0.175 m laterally; at 25 m/s,
10 ms timestamp error corresponds to 0.25 m translation. These are separate error
contributions, not final accuracy predictions. Retain shared/correlated errors
in covariance propagation; unlimited revisits cannot erase calibration bias.

Publish quality-qualified coverage, density/distribution, horizontal/vertical
uncertainty, image sharpness/GSD, thermal validity, age and provenance. No-return
space is not automatically free space. Only supported clear portions of valid
rays may contribute free-space evidence, with uncertainty; weak/invalid returns
must not manufacture safe corridors.

## 8. Planner integration

Quality layers change nothing unless the planner acts on them. Today it cannot:
frontier selection in `rust/argusnet-sim/src/frontier.rs` ranks 50 m blocks by
unmeasured share, coverage is a hit count over a circular footprint
(`mapping.rs`, `rust/argusnet-mapping/src/footprint.rs`), the scanner fan follows
velocity heading with no commanded pointing, and POIs complete on dwell time within
a radius (`rust/argusnet-planning/src/poi.rs`). The mission driver can also be given
true power-line spans (`set_known_power_lines`), which is oracle assistance.
Implementing sections 3–7 without changing the planner leaves completion
count-based, which section 7 forbids.

The integration has four layers with one owner each. Names are concepts, not
finalized APIs. It extends the scheduled contract in [planning](PLANNING.md):
read immutable belief/pose revisions, emit a proposal with its source epoch and
decision reason, bound the work, and revalidate safety before adoption.

### 8.1 Quality requirements

The selected mission profile declares measurable targets per region or POI, frozen
before the run: ground-return density, horizontal/vertical uncertainty, GSD,
thermal validity, admissible view angles and maximum evidence age, each with
units and statistic. A product without a declared requirement cannot pass;
an unknown requirement is not met.

### 8.2 Quality-deficit map

`argusnet-mapping` derives, per cell and product, the gap between quality from
accepted evidence and the requirement, alongside the existing belief queries
(`belief.rs`). Each deficit carries a reason, because each calls for a
different action:

| Deficit reason | Planner response |
| --- | --- |
| Unseen | Frontier exploration, as today. |
| Sparse | Same geometry again. |
| Too oblique or too coarse | New geometry: lower, closer or nearer nadir. |
| Occluded (canopy, façade, edge) | A different look direction, not more passes. |
| Stale | Revisit once the age limit expires. |
| Bias-limited (shared calibration/boresight error dominates) | Not improvable by revisits. Plan an observability pattern (crossing strips, opposed headings) or report unreachable. |
| Rejected evidence (clock, pose, calibration epoch) | Re-acquire only after the cause clears. |

The bias-limited class stops the planner from revisiting cells it cannot
improve. Deficits carry the belief revision they were computed from.

### 8.3 View planner

A new pure module in `argusnet-planning`, testable like `frontier.rs`, proposes
views. A candidate view is an aircraft segment plus a time-tagged pod attitude
schedule: cruise sweeps for area density, or VTOL hover stations for detail and
POIs. Hover costs far more energy than cruise, so flight mode is a planning decision.
Preserve fixed-wing speed/turn constraints in cruise.

Inputs are the deficit map, belief geometry, estimated pose and calibration,
energy/reserve state and the rig profile's declared planning capabilities:
gimbal envelope, slew/acceleration and settling model, housing occlusion
masks, and footprint/range/return envelope. Section 3's rig profile must carry
these. Candidates outside joint limits, without a settle window before capture,
occluded by the housing or blocked in the belief are rejected with a reason.

Score feasible views by predicted quality gain per unit energy and time, using a
forward sensor model built only from declared capabilities and estimated
calibration. Never predict with the truth generation model: that would hide model
error the same way using truth transforms hides boresight error (section 4).
Propulsion affects vibration, heat, drag and power only through declared model
couplings; earlier endurance estimates are not calibrated aircraft models.

The output is a sensing plan: trajectory, pointing schedule with settle windows,
expected products and quality, belief revision, rig/calibration epoch, energy
estimate and decision reason. A plan expires when its revision, epoch or safety
validity no longer holds. The previous plan continues only while valid; fallback
is profile-specific.

### 8.4 Flight safety sensing, pointing arbitration and execution

**The forward navigation camera is the primary obstacle-avoidance sensor.** It
is body-fixed, so its coverage follows the flight path and never competes with
mapping for pointing. Its profile declares a detection envelope: field of view,
detection range per object class/size (including thin wires, at a stated
probability and false-alarm definition), range uncertainty, illumination limits,
frame rate and processing latency. Detections feed a safety obstacle layer owned
by the safety path, separate from mapping quality; truth is used only to
generate and score them.

The flight envelope is bounded by what that camera can see, not by the pod:

- **Speed:** the commanded speed allows the aircraft to stop or divert within
  the camera's declared detection range for the smallest required obstacle
  class, after latency and the declared braking/turn model.
- **Direction:** turns, climbs and descents head into space the camera has
  recently cleared or that belief marks free with supported evidence. A
  manoeuvre whose swept volume leaves the camera's field of view requires that
  evidence or a reduced speed.
- **Degraded camera:** low light, glare, fog, a stale frame or a fault shrinks
  the declared envelope. The aircraft slows, holds or returns per the profile.
  Pod returns may add obstacle evidence when the profile declares that support,
  but the pod is never commanded forward to restore a safety envelope unless the
  profile names that fallback explicitly.

The planner proposes pointing; it does not command the gimbal. A runtime
arbiter on the control lane owns the pod command with fixed priority:

1. **Declared safety fallback** (only if the profile names one), while the
   navigation camera is degraded.
2. **Mission view:** the active sensing plan's schedule.
3. **Stow or idle.**

The executor records measured against commanded pose. A capture that misses its
settle window or tracking tolerance is flagged or rejected at admission, and the
measured pose is what georeferencing uses (section 4). Planner latency
cannot stall control, safety or the arbiter
([ADR-009](adr/009-independently-scheduled-runtime.md)).

### 8.5 Feedback and replanning

```text
sensing plan -> acquisition (measured pose) -> admission (accept/reject + reason)
  -> quality layers -> deficit-map revision -> replan trigger
```

Triggers: a deficit revision in the active region, a burst of rejections, a
calibration epoch change, a gimbal health change, energy reserve, and a periodic
deadline. Record predicted against achieved quality per view. Persistent
underperformance raises a model-mismatch flag; it does not retune the predictive
model online in a qualified profile.

### 8.6 Rules

- Mission completion requires every requirement to be met on accepted
  evidence. Reprocessing or re-rendering the same evidence adds nothing. A region
  whose deficit cannot be improved ends **unreachable** with its reason, never
  complete.
- The planner reads belief only. Truth wire identities and return classes are
  oracle inputs; profiles that keep them are labelled as such.
- Route corridors use only supported clear portions of valid rays (section 7).
  No-return space stays unknown, and unknown is not free.
- With no rig declared, the view planner degrades to the legacy fixed
  heading-aligned fan and circular footprint through the existing coverage view,
  and legacy fixtures stay unchanged.
- Obstacle avoidance reads the navigation camera's safety layer first. Mapping
  evidence from the pod never widens the safety envelope beyond what the profile
  declares.
- Checkpoints include the active sensing plan, schedule position, arbiter state
  and the navigation camera's safety layer.

## 9. Delivery sequence

| Slice | Concrete output | Completion gate |
| --- | --- | --- |
| SR1: contracts | Opt-in rig/calibration types, validation, mass ledgers and run identity. | Legacy fixtures unchanged; malformed/unsupported rigs rejected before mutation; both weight limits enforced. |
| SR2: geometry/time | Pod truth/estimated pose paths, fixed camera branch, clocks and bounded history. | Known landmarks register through body/pod motion; injected errors remain observable; unsupported interpolation rejected. |
| SR3: LiDAR | Beam pattern, return envelope, separate errors and per-firing time. | Range/material/angle sweeps, density and beam footprint agree with declared evidence/assumptions. |
| SR4: imaging | Recorded visible/thermal inputs and camera models; synthetic capture only if needed. | Stereo/blur/occlusion and thermal-validity checks; no invented temperature or oracle depth. |
| SR5a: quality/deficit | Evidence admission, quality layers, uncertainty, frozen requirements and the reasoned deficit map (8.1–8.2). | Degraded input reduces quality; repeated/rejected evidence cannot shrink a deficit; bias-limited cells are classified. |
| SR5b: planner on belief quality | Frontier and POI logic consume the deficit map with fixed pod pointing; completion moves from counts to requirements (8.6). | Legacy runs unchanged without a rig; completion needs met requirements; planner reads no truth. |
| SR5c: safety camera and pointing | Navigation-camera detection envelope and safety obstacle layer, speed/direction limits from it, then the pointing schedule, arbiter and executor (8.4); pointing depends on SR2's gimbal state. | Avoidance holds with the pod pointed away; speed falls when the camera's envelope shrinks; settle misses are rejected at admission; measured pose is recorded and used. |
| SR5d: view optimization | Flight-mode choice, oblique and calibration-observability patterns, energy-weighted view scoring and replanning (8.3, 8.5). | Matches or beats legacy lanes on quality per energy; predicted and achieved quality agree within the declared model or flag mismatch. |
| SR6: replay/viewer | Additive provenance, payload references, actual frusta and health. | Old replays read; new fields have consumer coverage; missing evidence renders unavailable. |
| SR7: qualification | Held-out datasets and calibrated profile report, installed hardware evidence. | Claims stay within measured envelope and applicable SIL/HIL/field gates. |

Add meaningful behavior tests in each owning crate. Run `cargo test -p argusnet-sim`
for runtime changes, `cargo test -p argusnet-mapping` for mapping changes, and the
replay/viewer/Python contract checks when those consumers change. Follow
[local CI](LOCAL_CI.md) before a PR. Run `cargo test -p argusnet-planning` for
planner changes. This documentation update does not complete SR1.

## 10. Required validation matrix

| Experiment | Expected evidence |
| --- | --- |
| Surveyed plane, nadir/oblique rays | Slant range, footprints and height residuals agree with analytic geometry. |
| Dark/bright targets, edge of range, grazing incidence | Return fractions/error follow declared curves; include sample counts, confidence and missing calibration. |
| Thin wires/poles, canopy gaps, building edges | Beam sampling, occlusion and supported return modes matter; no perfect truth-ID perception. |
| Body translation/roll with independent pod slew | Time-varying transforms, fixed camera independence and deskew residuals. |
| Clock drift/reset, late/out-of-order/duplicate input | Explicit epochs, capture times and degraded/rejected outcomes. |
| Encoder lag, boresight bias, GNSS degradation, vibration | Map error/uncertainty increases; commands cannot hide actual pointing error. |
| Poor texture, exposure changes, thermal drift | Invalid depth/temperature becomes degraded or unavailable rather than precise. |
| Renderer off, varied pacing and permitted worker counts | Same causal results at fixed inputs/model; no render-driven sensing. |
| Checkpoint during scan/pending delivery | Resume matches uninterrupted results, preserving scan phase, pose history, clocks, bias/RNG state and payloads. |
| Queue overload, payload loss, device restart | Bounded work, explicit gaps/epochs and declared fallback; no silent decimation or reuse. |
| Occluded façade or canopy-gap target | Planner changes look direction or station; repeated identical passes do not count toward completion. |
| Injected boresight bias | Affected cells classify as bias-limited; planner schedules observability passes or reports unreachable, never endless revisits. |
| Fast cruise through obstacle field, pod pointed sideways | Navigation camera detects obstacles within its declared envelope; speed is capped to its stopping range; pod pointing does not change avoidance. |
| Navigation camera degraded (night, glare, fog, fault) | Envelope shrinks; aircraft slows, holds or returns per profile; no silent reliance on pod or truth. |
| Thin wire below camera detection threshold | Reported as outside the declared envelope; speed/clearance policy, not perfect detection, keeps margin. |
| Gimbal fault mid-plan | Plan expires; replan or declared fixed-pointing fallback, with the gap recorded. |
| Planner lane stalled or slow | Control, safety and arbiter keep their deadlines; the last plan expires when invalid. |
| Requirement absent from mission profile | Product reports unmet/unavailable; mission cannot complete on it. |

Separate tuning data from held-out validation. Field reports use independent
surveyed checkpoints excluded from calibration/adjustment. Report datum,
horizontal/vertical bias and RMSE, percentiles, sample counts, completeness and
uncertainty calibration. Freeze numeric acceptance targets in the selected mission
profile before scoring; unknown requirements cannot pass. Publish the qualified
range, speed, illumination, materials, angles and weather, alongside limitations.

## 11. Evidence and unresolved decisions

The [Livox Avia specification](https://www.livoxtech.com/avia/specs) illustrates
range conditioning: at 100 klx it lists 190 m for 10% reflectivity and 320 m for
80% reflectivity. These describe that device, not this pod, and do not establish
compatibility with a fully solid-state requirement. Freeze the selected device's
datasheet and firmware revision in its eventual profile.

Before selecting hardware or qualifying a model, resolve altitude/speed, maximum
ray angle, survey accuracy/statistic, usable ground density, camera arrangement,
thermal versus NIR, LiDAR field of view, cruise dynamics, power, interfaces and
required endurance. The gimbal axes, joint limits, pod sensor set, LiDAR scan
type and flight-mode use are decided (section 1).
Use [hardware integration](HARDWARE_INTEGRATION.md) for adapter/flight qualification
and [verification](migration/VERIFICATION_AND_ROLLOUT.md) for lifecycle/scheduler
acceptance. None of the proposed types, slices or experiments is claimed shipped.
