# Parallel, multi-rate tick: performance plan for `scheduled_v1`

> **Status: proposed plan, not implemented, not accepted.** No scheduler or
> parallel executor exists in `rust/` today. This document refines P4.10a–f
> ([delivery plan](DELIVERY_PLAN.md)) and ADR-009; it does not renumber them.
> Rates, speed-ups and thresholds below are hypotheses to be replaced by
> measurements (§9). Hardware timing remains the separate S3–S7 track.

## 1. Goal and non-goals

**Goal.** Make the Rust tick cheaper per simulated second and keep it responsive
by (a) not doing work that has no new input, (b) running independent work
concurrently, and (c) letting each subsystem run at the rate its data
actually supports — while the virtual-time result stays **bit-identical for any
worker count and any completion order**.

**Non-goals.**

- No change to `legacy_stages_v1`. Its eleven-stage order stays the port
  acceptance contract ([`Stage::PIPELINE`](../../rust/argusnet-sim/src/stage.rs)).
- No claim that concurrency gives hard real-time guarantees (ADR-009).
- No load-adaptive rates inside deterministic simulation (§6).
- No multi-vehicle SLAM or new sensor fidelity; a faster output rate cannot
  manufacture physical accuracy ([RUST_RUNTIME_ARCHITECTURE §3](../RUST_RUNTIME_ARCHITECTURE.md)).

## 2. Order of levers

Existing project guidance is to benchmark dirty tiles and summed-area tables
before parallelism. This plan keeps that order, because removed work is free on
every core count and parallel work is not:

1. **Measure** per-stage cost in the Rust tick (none is recorded today).
2. **Reduce work**: dirty tiles, spatial indexing, buffer reuse, no-input skips (P4.10c).
3. **Multi-rate**: stop running subsystems at a rate their inputs do not justify (P4.10a/b).
4. **Data-parallel inside a phase**: independent platforms/pairs/tiles (§4.2).
5. **Task-graph concurrency across phases and background jobs** (§4.3, P4.10d).

Steps 4–5 ship only where step 1 shows the serial cost is large enough to
matter (Amdahl check, §9).

## 3. What the current tick looks like

Eleven stages, strictly serial, one rate (`dt_s`, legacy default 0.25 s):

| # | Stage | Reads | Writes | Parallel shape |
| --- | --- | --- | --- | --- |
| 0 | Apply commands | command queue | run state, targets | Serial (transaction boundary) |
| 1 | Controllers + battery | prior positions, mission | platform targets, battery | **Per platform** (independent) |
| 2 | Mission overrides | mission state | platform targets | Serial, precedence-ordered |
| 3 | Deconfliction | all platform targets | platform targets | **Cross-platform pairs**; cheap; serial first |
| 4 | Safety monitor/clamp | targets, terrain | targets | Per platform, but after 3 |
| 5 | VIO + pose select | truth (capability), IMU model | per-platform estimator | **Per platform** |
| 6 | Generate observations | truth, poses | observation batch | **Per (node, target)** |
| 7 | Fusion | observations | fused tracks | Single writer; association is serial |
| 8 | Mapping + localization | observations, poses | belief, localizer | Footprints parallel, belief single writer; localizer per platform |
| 9 | Mission transitions | belief, tracks | mission state, events | Serial |
| 10 | Commit snapshot | everything | frozen revision | Serial (one freeze) |

Two existing properties make parallelism safe to introduce:

- **Order-independent randomness.** ADR-006 keeps a per-(node, target, tick)
  stream keyed by a stable hash, so observation draws do not depend on
  iteration order. Per-platform estimator streams must be checked to hold the
  same property before they are parallelised (open item, §10).
- **Truth is a typed capability** (`TickContext::truth`, `Stage::may_read_truth`).
  Only VIO and observation generation receive it, which is exactly the set of
  tasks the sensing lane needs and no more.

Known caution: the 2026-09 baseline profile was measured on the **Python**
runtime (localizer 40 %, VIO 13 % of tick). It does not establish Rust costs
and the docs already warn against using it for priorities.

## 4. Target model

### 4.1 Lanes, not one tick

Each lane has its own trigger. Physics is the only fixed-step clock; everything
else is a schedule of integer-nanosecond deadlines (rational rates allowed, no
cumulative drift, per ADR-009).

| Lane | Trigger | Concurrency | Sequential constraint |
| --- | --- | --- | --- |
| **Physics** | Fixed step | One writer of vehicle state | Integrates to each due timestamp first |
| **Control + safety** | Fixed deadline, bounded work | Per-platform parallel for nominal control; deconfliction → safety remain ordered | Never waits for planner/mapping; safety last before actuation |
| **Sensors (per source)** | Per-source schedule + modeled delivery delay | Per source / per platform / per target pair | Capture and arrival are distinct events |
| **Estimation (VIO, localizer)** | Measurement arrival + prediction/expiry deadline | Per platform | Uses actual elapsed time, not a global dt |
| **Fusion** | Available observation batch + track-expiry deadline | Single writer | Association and track lifecycle in canonical order |
| **Mapping** | Evidence batch size **or** max-age deadline | Footprint rasterization parallel; ordered merge into one belief writer | Dirty tiles only; revisioned output |
| **Planning/mission** | Mission/battery/geometry change **or** periodic deadline | Background workers on immutable inputs | Result is a proposal, revalidated by safety before use |
| **Publication** | Display cadence + important changes | Reader of committed revisions | Snapshot = selection of component revisions with ages |

### 4.2 Inside a timestamp: phases with fork-join

ADR-009 already fixes the phase order at one timestamp (propagate → commands →
captures → deliveries → estimators → mapping → mission/planning → control/safety
→ publication). This plan adds one rule on top:

> A **phase is a barrier**. Inside a phase, tasks that touch disjoint state
> partitions run concurrently; their results are merged **in canonical order**
> (stable source id, then sequence) — never in completion order.

Concrete candidates, in expected-value order (to be re-ranked by §9):

1. Observation generation across (node, target) pairs — already key-seeded.
2. Per-platform localizer and VIO updates.
3. Footprint rasterization into a per-worker scratch buffer, then ordered
   reduction into the belief store (reduction order fixed by tile/platform id,
   so floating-point sums are reproducible).
4. Nominal controller sampling and battery accounting per platform.

Kept serial on purpose: command application, mission override precedence,
deconfliction pair resolution (small N, order matters), fusion association,
mission transitions, snapshot freeze.

### 4.3 Across timestamps: revisioned task graph

Every due item becomes a task declaring **inputs by revision** and **outputs by
owned state**. The scheduler may run tasks concurrently only when write sets are
disjoint and input revisions are satisfied. This is what makes the multi-rate
lanes safe to overlap:

- Background jobs (planning, meshing, export) take an immutable input revision,
  run on a bounded pool, and return a proposal tagged with that revision.
- The proposal is applied only at a **recorded application boundary**: if it
  is not ready, simulated time waits (commands stay serviceable) rather than
  applying it at a wall-clock-dependent moment. A stale revision is rejected.
- Conservative lookahead (a lane runs ahead only to the next time its inputs
  could change) is a later optimization, gated on §9 showing it pays; it is not
  part of the first delivery.

## 5. Execution resources

- **One dedicated, bounded compute pool** owned by the session (not a global
  pool), size configurable, default ≤ physical cores − 1. The viewer (Bevy) and
  I/O keep their own resources; domain crates take no dependency on Bevy.
- **Control lane reserved capacity**: bounded work units and, for live use,
  reserved execution capacity. A pool is not a real-time guarantee; live
  deadlines are qualified in S3/S4.
- **Scratch reuse**: per-worker buffers recycled across ticks (P4.10c) so
  parallelism does not add allocation churn.
- Worker count `1` runs the **same scheduler serially**. There is one
  execution path, so the serial result is the reference for parallel runs.
- Sandbox note: this development container has 4 cores, so speed-up numbers
  taken here are indicative only; gates must name target hardware.

## 6. Where rates come from

Three distinct mechanisms, kept separate because only some are deterministic:

| Mechanism | Example | Allowed in deterministic sim | Allowed live |
| --- | --- | --- | --- |
| **Static schedule** | Camera 12 Hz, IMU 100 Hz, control 50 Hz | Yes | Yes |
| **Data-triggered** | Mapping runs at N queued footprints or max-age; planner runs on mission/geometry change; estimator runs on arrival; fused tracks age on expiry deadline even with no input | Yes — trigger is a function of simulated state; trigger reason logged | Yes |
| **Load-adaptive** | Skip/slow display or mapping under CPU pressure | **No.** Wall-clock load must never change causal state. Overload lowers *achieved speed* and is shown | Only under a qualified fallback policy, recorded as a gap/event |

Rate-sensitive logic must be reviewed when rates change: prediction, battery
accounting, timeouts and mission dwell all need actual elapsed time. Today's
legacy battery accounting (including "hover costs nothing") must not drift
silently into `scheduled_v1`; any change is an explicit, versioned behavior
change.

Starting rates are **not chosen here**. P4.10f ("balanced workload
qualification") picks them from measured accuracy/cost curves. A reasonable
first sweep is control ≥ legacy rate, sensing at each sensor's modeled rate,
and mapping/planning at lower rates, with the report showing the accuracy cost
of each reduction.

## 7. Determinism and verification

Required evidence (extends ADR-009's list; statistical-envelope rules from
[VERIFICATION_AND_ROLLOUT](VERIFICATION_AND_ROLLOUT.md) still apply to anything
that is not bit-exact):

1. **Worker-count invariance:** same seed/config → identical committed state and
   replay at 1, 2 and N workers.
2. **Completion-order invariance:** a test executor injects randomized task
   delays and reversed completion order; results unchanged.
3. **Serial equivalence:** `scheduled_v1` with every lane at the legacy rate and
   one worker reproduces `legacy_stages_v1` outputs within the existing
   golden/statistical gates. This is the bridge that proves the new executor
   did not change behavior before rates are changed.
4. **No floating-point reorder:** reductions use fixed-order combine; a mutation
   test swaps the combine order and must fail.
5. **Stale-result rejection**, pending-job restore, bounded queues, old-epoch
   message rejection (already in P4.10d/e exit evidence).
6. **Thread-safety tooling** in CI for the executor core (e.g. loom/Miri-style
   model tests on the small synchronization kernel, ThreadSanitizer on the
   integration run) — to be confirmed available in the CI image.

## 8. Delivery sequence

Maps onto existing packages; new work is marked **(new)**.

| Step | Package | Deliverable | Exit evidence |
| --- | --- | --- | --- |
| 0 | **(new)** P4.10-pre | Per-stage and per-lane wall timers, p50/p95/p99, work counters (platforms, pairs, footprints, tiles touched), in the stage trace | Published Rust profile on named workloads; Amdahl table ranking candidates |
| 1 | P4.10c (pulled forward) | Summed-area/dirty-tile coverage, spatial index, buffer reuse | Lower work counters, equal outputs, measured CPU/alloc change |
| 2 | P4.10a | Integer-time event scheduler, **serial executor**, rational sensor rates, phase barriers | No drift; stable ties; serial-equivalence gate (§7.3) |
| 3 | **(new)** P4.10a.2 | Fork-join inside phases behind `workers = n`; ordered merge | Worker-count and completion-order invariance (§7.1–7.2) |
| 4 | P4.10b | Timestamped envelopes, source epochs, elapsed-time estimators, expiry deadlines, buffered mapping inputs | Capture vs delivery preserved; empty-input track aging; no lost evidence at unequal rates |
| 5 | P4.10d | Bounded background pool, job revisions, cancellation, deterministic application boundaries | Same causal state across worker counts; stale jobs rejected; safety revalidation |
| 6 | P4.10e | Scheduler/sensor/job checkpoints, branch epochs, backpressure, viewer freshness | Restore with pending deliveries/jobs equals uninterrupted run |
| 7 | P4.10f | Workload qualification, rate selection, default promotion | Published accuracy/timing/memory results; no new safety failures; rollback profile kept |

Each step is independently mergeable and leaves `legacy_stages_v1` untouched.
Steps 3 and 5 are optional per candidate: skip any whose step-0 share is too
small to repay the added complexity.

## 9. Gates and how success is judged

- **Decision rule (step 0):** a stage is a parallelisation candidate only if its
  serial share of tick time, combined with its measured parallel fraction,
  predicts a tick-latency gain larger than the run-to-run noise on the named
  hardware. Otherwise reduce its work instead.
- **Targets to be set after step 0** (not asserted here): tick p95/p99, achieved
  sim speed on the balanced workload, peak memory, recording backlog, and
  control-deadline miss count. Viewer p95 ≤ 16.7 ms / p99 ≤ 25 ms are existing
  *targets*, not measured results.
- **No regression:** accuracy, safety-violation counts and coverage stay within
  the statistical envelopes while rates are reduced.

## 10. Risks and open questions

| Item | Why it matters | Resolve by |
| --- | --- | --- |
| Per-platform VIO/localizer RNG may share a stream | Parallelising would change draws | Audit `vio.rs`, `particles.rs`, `localizer` seeding; key by (platform, tick) like ADR-006 |
| Shared handles (`BatteryLink`, `MissionLink`) are interior-mutable | Hidden shared writers break disjoint-write reasoning | RUST_RUNTIME_ARCHITECTURE already requires replacing them with owned state; do that first |
| Float reduction order in belief/footprint merge | Non-reproducible sums | Fixed-order combine + mutation test |
| Deconfliction is cross-platform | Cannot be naively per-platform | Keep serial; revisit only if pair count grows |
| Multi-rate changes battery/prediction semantics | Silent behavior drift | Elapsed-time review per §6; version flag |
| Small fleets (4 drones, 7 stations) | Per-task overhead may exceed gains | Step-0 measurement; coarse task granularity; serial fallback below a size threshold |
| Pool is not real-time | Live control deadlines | S3/S4 qualification; reserved capacity for control lane |

## 11. Docs to update when accepted

- New ADR (proposed ADR-010): parallel execution model, referencing ADR-009.
- `RUST_RUNTIME_ARCHITECTURE.md` §3/§5: add the phase-barrier/ordered-merge rule
  and the three rate mechanisms.
- `DELIVERY_PLAN.md` P4.10: add P4.10-pre and P4.10a.2; note P4.10c is pulled
  forward.
- `KNOWN_GAPS.md`: add "parallel execution" row (Planned).
