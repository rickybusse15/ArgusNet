# ADR-009: Independently scheduled Rust runtime

**Status:** accepted design; implementation in progress (P4.10a scheduler types and lane executors, P4.10b first delivery slice; see [migration status](../migration/STATUS.md))
**Date:** 2026-09-23
**Extends:** ADR-005, ADR-007 and ADR-008; retains ADR-006

## Context

The first Rust port deliberately preserves the Python eleven-stage causal order.
This verifies migration, but does not require all future sensors, estimation,
mapping, planning and presentation to execute at one frequency. A complete
Rust-only production path also needs explicit lifecycle and checkpoint ownership.

## Decision

Use one Rust session authority with independently scheduled subsystems and typed
local calls. Retain `legacy_stages_v1` for port acceptance; introduce opt-in
`scheduled_v1`. Physical integration remains fixed-step while event/sensor rates
use independent integer-time deadlines, including rational rates. They are not
required to be multiples of the physics interval.

Virtual-time simulation is deterministic regardless of rendering, pacing or
worker completion order. Live-time operation uses qualified deadline/fallback
policies and device-clock conversion; simulation determinism does not prove
hardware timing. Rust and async are implementation tools, not timing guarantees.

Keep commands, replaceable snapshots, sensor evidence and durable events on
paths with explicit different loss policies. A snapshot records component age
and revisions rather than pretending all estimates have one acquisition time.
Domain state has one writer; background results are proposals checked before use.

The detailed decision is [Rust runtime architecture](../RUST_RUNTIME_ARCHITECTURE.md).
It supersedes any earlier suggestion that all sensor periods must divide a global
tick. ADR-007's stage ordering remains binding for the compatibility port only.

The completion audit additionally fixes supporting contracts: independent
[fidelity/time policies](../SIMULATION_FIDELITY.md), reproducible world manifests,
track evidence age/identity, frame-correct presentation and verified journal
recovery. Their detailed requirements live in the linked domain documents;
qualification and implementation remain staged within the existing packages.

## Delivery and consequences

Complete P4.8/P4.9 first. Deliver P5.0 viewer controls before the redesign is
complete. P4.10a–f add and qualify the scheduler; P6 retires old execution after
acceptance. Existing domain crates and historical replay readers remain useful.

Checkpoints must include sensor deliveries, schedules, subsystem state and pending
causal jobs. Reset must rebuild them all. Sensor sampling between physical steps,
late observations and live overload need explicit capabilities; they cannot be
hidden behind changing dt or retimestamping inputs.

## Required evidence

Same causal state across pacing/worker counts, independent-rate scheduling,
acquisition-time handling, restore during pending work, bounded queues,
deadline/freshness reporting, responsive viewer controls, named-hardware
performance and separate SIL/HIL qualification. Preserve the determinism and
statistical rules in [verification](../migration/VERIFICATION_AND_ROLLOUT.md).
