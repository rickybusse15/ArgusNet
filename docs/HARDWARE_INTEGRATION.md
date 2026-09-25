# Hardware integration and qualification

**Updated:** 2026-09-23. **Status:** target requirements, not an aircraft-ready
implementation. Current live ingestion/viewing does not imply flight authority.
This document operationalizes S3–S7 in the
[autonomous system plan](AUTONOMOUS_SYSTEM_PLAN.md) and the live-time boundary in
[the Rust runtime architecture](RUST_RUNTIME_ARCHITECTURE.md).

## Deployment and authority

The simulation runtime and future hardware runtime share domain algorithms and
measurement/decision contracts. A hardware adapter replaces simulated physical
and sensor endpoints; it does not let the ground viewer become the flight loop.
Existing autopilot firmware is not required to be Rust. The Rust-only goal refers
to ArgusNet's production application/runtime dependencies, not rewriting firmware.

| Component | Authority |
| --- | --- |
| Autopilot/flight controller | Stabilization and qualified low-level flight control |
| Onboard Rust autonomy | Estimated state, mission decisions, planning and qualified setpoint proposals |
| Independent onboard safety | Containment, energy/integrity limits, watchdog and contingency decisions |
| Onboard recorder/health | Local evidence and health even during ground-link loss |
| Ground viewer/service | Mission authoring, supervision and explicit authorized intent; no direct actuator access |

Loss of viewer, ground service, radio or cloud cannot disable stabilization or
the qualified contingency. Each aircraft retains its own safe response during
fleet partitions. A historical timeline or Replay source has no actuation authority.

## Adapter contract

Every adapter declares device/firmware identity, supported message versions,
frames/units, calibration identity, source clock/epoch, nominal and maximum data
rates, timestamp uncertainty, buffering limits, supported commands and health.
Keep capture, availability, host receipt and processing timestamps separate.
Preserve raw device time and the versioned host-time conversion used for it.

Handle sequence gaps, duplicates, reboot/time resets, delayed/out-of-order data,
calibration changes and reconnects explicitly. Missing synchronization produces
degraded/unavailable timing integrity, not an apparently exact timestamp.
Reject unsupported estimator timing profiles before enabling autonomy.

Sensor rate is independent of simulation/control/render rates. Sampling models
and real sensor bandwidth must be qualified; a 200 Hz output label does not
establish 200 Hz physical information. Clock conversion and adapter I/O do not
hold the runtime's state lock. Record source backpressure and drop policy.

Setpoint/control authority is a separate authenticated, capability-declared
interface with expiry, acknowledgement, freshness checks and safety validation.
Read-only streaming or successful telemetry ingestion never grants it. Prevent
stale commands from crossing source epochs or replay/live transitions.

## Qualification ladder

| Gate | Required evidence before promotion |
| --- | --- |
| Contract fixtures | Recorded device packets converted to stable frames/units/time; invalid messages and unsupported versions fail explicitly |
| S3 SIL | Actual selected flight software against the simulated environment; deterministic nominal/fault inputs where supported; actuator/control feedback recorded |
| S4 HIL | Selected production compute, flight controller, interfaces and radios; measured timing, resource use and independent contingencies |
| S5 contained aircraft | Operation-specific contained test plan, repeated bounded outcomes and evidence linking the hardware/config to HIL results |
| S6 field envelope | Qualified wind/payload/energy/localization limits, GNSS degradation and recovery, drift/containment and landing evidence |
| S7 fleet | Clock/frame alignment, coordinator loss, partial connectivity, stale reservations and independent per-aircraft safety |

No specific autopilot, board, transport, sensor rate or OS is selected by this
documentation update. Before implementing S3, freeze a hardware profile naming
those choices, their deadlines and acceptance bounds. Do not label a generic
simulated sensor adapter SIL/HIL without executing the corresponding software
or hardware. Operational approval is independent of technical test results.

## Simulation and external timing policy

Use [simulation fidelity](SIMULATION_FIDELITY.md) to declare whether a SIL adapter
supports virtual-time stepping or requires deadline-coupled pacing. HIL/live time
cannot silently slow down when the desktop falls behind. Freeze the model,
clock-conversion policy, maximum synchronization error and watchdog response in
the hardware profile. Qualification failure must be visible and invoke the safe
test response; replaying outputs afterward is not evidence the original deadline
was met. The selected sensor and flight model must support the quantities/rates
being tested, not merely emit a higher-frequency interpolated position.

## Timing, overload and failure matrix

On each declared hardware profile, measure acquisition-to-estimate,
estimate-to-control and command-to-ack latency distributions, deadline misses,
queue age, clock residuals, CPU/memory, thermal throttling, power and recording
backlog. Report worst observed duration and test length alongside percentiles;
measurements alone are not a proof of hard real-time schedulability.

Exercise sensor silence/bursts, timing drift/reboot, CPU saturation, storage-full,
radio loss/reordering, stale plans, localization loss and ground-app termination.
Each case identifies the detecting component, allowed response time, onboard
fallback, resulting state and supporting recorder events. A fallback must be
qualified for the vehicle/environment; do not assume indefinite hover or holding
the last command is safe.

Live wall time cannot wait for a mesh, route search, disk write or paused UI.
Reserve capacity for critical work; bound and cancel background work. Apply
profile-specific evidence retention under storage pressure and expose gaps.
Retain actual hardware/firmware/calibration/config versions in every report.

## Viewer and operator presentation

Operate shows Simulation/SIL/HIL/Live/Replay identity, current versus historical
time, onboard/ground connectivity, integrity, control authority and contingency.
Plan shows hardware-capability and rate compatibility before deployment.
Review correlates device clocks, delivered measurements, decisions, deadlines
and evidence gaps. Pause/Step/Seek apply only to supported simulation/history;
they never imply the physical aircraft has paused.

Use [viewer workflows](VIEWER_FEATURE_INTEGRATION.md) and the
[verification gates](migration/VERIFICATION_AND_ROLLOUT.md). Hardware acceptance
requires its own evidence even after every desktop simulation test passes.
