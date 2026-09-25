# Simulation fidelity, sensor models and real-time execution

**Updated:** 2026-09-23. **Status:** target architecture; profile names below are
design identifiers, not implemented CLI options or readiness claims.
This extends [the runtime design](RUST_RUNTIME_ARCHITECTURE.md) and separates
model fidelity from execution speed, presentation quality and
[hardware qualification](HARDWARE_INTEGRATION.md).

## 1. Independent configuration dimensions

Every resolved run manifest identifies four separate choices:

| Dimension | Meaning |
| --- | --- |
| Execution model | Legacy stage order or independently scheduled behavior |
| Time policy | Unrestricted virtual time, paced virtual time, or deadline-coupled external operation |
| Physical/sensor model | Dynamics, sensor sampling, noise/bias, environment coupling and supported operating envelope |
| Presentation quality | Mesh/texture/lighting/LOD settings; normally has no effect on physical observations |

A faster output rate is not a more accurate physical model. Photorealistic terrain
is not evidence of accurate localization, vehicle dynamics or sensor noise.
Changing a fidelity model is a different experiment, not a free performance gain.

## 2. Model profiles and evidence

| Profile | Intended role | Evidence required before claiming support |
| --- | --- | --- |
| Legacy kinematic | Compatibility with existing scenario motion and sensor assumptions | Existing port fixtures and documented known limitations |
| Timing-aware sensing | Independent sampling, capture/delivery latency, clocks and loss around a declared motion model | Timestamp/sampling fixtures, bounded queues, supported estimator lateness and repeatability |
| Qualified dynamics and sensing | Calibrated vehicle/controller/environment and sensor response over a stated envelope | Integration convergence, conservation/response checks where applicable, and comparison with recorded reference data |
| SIL / HIL | Actual selected flight software or hardware interacting with the environment | The separate S3/S4 identity, deadline and failure matrix; not just a simulation preset name |

Profiles compose only when their capabilities are compatible. Validation rejects
requests for accelerations, angular rates, contact response or image sampling
that the chosen physical/sensor model cannot produce honestly. Existing kinematic
motion must not be described as a calibrated flight-dynamics model.

Define each model's state, units/frames, integration method, validated step range,
initialization, saturation/failure behavior, version and checkpoint state. Test
step refinement on fixed continuous inputs before reducing the integration interval.
Keep contact/terrain protection separate from planner knowledge and log physical
corrections. No profile permits penetration below terrain.

## 3. Sensor generation and environmental realism

The simulation world owns physical motion. Synthetic sensors sample that state
at acquisition time through a declared model; sensor generation is independent
of whether the viewer is open, focused, rendering slowly or using a low quality
tier. Distinguish simulated physical platform state from its onboard estimate.

For image/depth sensors that require rendering, use a dedicated offscreen capture
path with its own camera calibration, scene revision, acquisition schedule and
resource budget. Never use the operator camera or an arbitrary latest screen
frame as a sensor input. Recorded reference frames provide a replayable input
when GPU image generation cannot satisfy the required determinism contract; do
not claim bit-identical sensor images across unrelated drivers without evidence.

Environmental models explicitly declare which effects are implemented:
illumination/occlusion, weather visibility, sensor bias/dropout, wind/dynamics,
communications delay/loss and energy consumption. A visual rain effect does not
imply an aerodynamic or sensing effect. A communications LOS query is not by
itself a calibrated radio propagation model. Unsupported couplings remain absent
and visible in the profile manifest.

Noise and fault streams are keyed independently from rendering and job order.
Separate nominal noise, correlated bias, calibration error and deliberate fault
injection; retain stateful bias in checkpoints. Scenario truth IDs may support
evaluation, but operational association and confidence cannot secretly use them
in profiles claiming unlabeled perception.

## 4. Three time policies

**Unrestricted virtual time:** advance through due events as fast as resources
permit. Wait for causal jobs and required recording without advancing source
time. This is the default mode for deterministic batch experiments.

**Paced virtual time:** map source time to a monotonic wall-clock origin and
requested speed. Sleep outside causal updates when ahead. When behind, perform
bounded catch-up work, report lag and reduce achieved speed; do not enlarge dt,
drop physical steps or silently change sensor rates. Pause/resume reanchors the
pacing origin so a pause is not interpreted as a missed interval to catch up.

**Deadline-coupled operation:** external hardware or a selected SIL integration
constrains time. It cannot silently slow the physical world. Declare maximum
allowed synchronization error, watchdog behavior and deadline-specific fallback
in its qualification profile. Exceeding those bounds fails the run qualification
and invokes the configured safe test response. Live control cannot wait for
recording, meshing or a new global plan. SIL may use virtual time only if the
flight-software integration explicitly supports it; HIL timing must be measured.

Replaying recorded sensor inputs through the estimator is different from playing
recorded output states in the viewer. Label both. Live hardware outcomes depend
on received inputs and timing; deterministic reevaluation requires preserving
those inputs, admission decisions and applied fallbacks.

## 5. Evaluation and delivery

P4.9 fixes the legacy reference. P4.10a/b qualify independent schedules and
timing-aware sensor profiles; P4.10f compares quality and cost at fixed fidelity.
Qualified dynamics, rendered sensor models and external timing promotion are
separate product/S3/S4 work, not prerequisites for early P5.0 viewer controls.

Test low/high sensor rates, off-boundary captures, correlated noise, clock reset,
fixed-input step refinement, bounded catch-up, pause reanchoring, renderer-off
equivalence and source loss. Compare estimation error/calibration, map fidelity,
mission results, energy and safety alongside CPU/latency. Reference datasets
record calibration, environment and collection conditions; hold out evaluation
cases from tuning. Publish limitations rather than one undifferentiated
"realism" score. See [verification](migration/VERIFICATION_AND_ROLLOUT.md).
