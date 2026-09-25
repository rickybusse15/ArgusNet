# Next refactor: finish Rust ownership, then improve the model

> Historical source review: retain its dated findings as evidence, not a live
> completion list. The 2026-09-23 agreed execution target is
> [RUST_RUNTIME_ARCHITECTURE.md](RUST_RUNTIME_ARCHITECTURE.md), with P4.10 packages
> in [the delivery plan](migration/DELIVERY_PLAN.md). Current command/control
> fixes and remaining lifecycle gaps are distinguished in [KNOWN_GAPS.md](KNOWN_GAPS.md).

Review of checkout `388f74b`, 2026-09-19. This is a source-based proposal, not
an implementation or a fresh performance baseline. Existing local edits to
`VIEWER_UI.md` and the scene-manifest backup were left untouched.

The architectural direction is sound: one world authority, Rust-owned fusion,
domain crates independent of Bevy, and an explicit simulation owner publishing
committed snapshots. The next investment should be **runtime composition and
contract correctness**, followed by measured optimization and separately
versioned accuracy improvements. Another broad rewrite would duplicate work
already present.

## Later integration note

This review's findings describe `388f74b`. Before publishing the plan, main was
fast-forwarded to `ba163ac`, which adds P4.3g landscape construction, P4.6d belief
publication and P4.8a–c mission state/TickWorld/stage bodies. The earlier findings
about missing world builders, belief publication and always-empty snapshots are
historical. Use [the phased plan](migration/DELIVERY_PLAN.md) for current work;
revalidate remaining findings against implementation before fixing them.

## What has actually migrated

| Area | Evidence in this checkout | Remaining work |
|---|---|---|
| World and safety | `argusnet-world`, `safety-engine`, native handles and batched queries | Finish runtime scenario/world assembly; existing noise, grid, preset and semantic kernels are useful building blocks |
| Fusion | `argusnet-core::TrackingEngine`; server owns one engine behind a bounded queue | Call the same engine directly from the Rust simulation; retain gRPC for external ingest/subscribers |
| Localization | `argusnet-localization` contains estimator state and VIO modules; deterministic generators exist in `argusnet-sim` | Compose these into the authoritative tick and checkpoint all causal state |
| Mapping | Footprint rasterization, incremental coverage statistics and a belief-query facade exist | Add a coherent mutable belief owner, revisioned tile changes and publication |
| Planning | Battery, routes, planner, frontier, POI, deconfliction and coordination modules exist | Integrate mission execution with explicit inputs and safety ordering |
| Simulation | `Simulation`, commands, clock, traces and checkpoint headers exist | Default domain stages do no work; snapshots are empty; there is no complete Rust-owned mission |
| Replay/viewer | Renderer-independent `argusnet-replay` DTO crate exists | DTOs still contain presentation precision assumptions; interactive Run still launches Python simulation and scene building |

Sources: [simulation shell](../rust/argusnet-sim/src/simulation.rs),
[simulation dependencies](../rust/argusnet-sim/Cargo.toml),
[mapping](../rust/argusnet-mapping/src/lib.rs),
[planning](../rust/argusnet-planning/src/lib.rs),
[viewer process launch](../rust/argusnet-viewer/src/app.rs),
[integration record](migration/P4_INTEGRATION.md).

`architecture.md`, `STATE_OWNERSHIP.md` and the migration index lag the code in
several places: they still describe implemented crates as planned and describe
proto regeneration as missing, although CI now runs both regeneration and field
parity checks. The integration record is more recent, but also does not prove a
complete runtime exists. Replace conflicting status narratives with one table:
**implemented library / active runtime consumer / verified integration / retired
legacy path**, with an evidence link for each transition.

## Decisions to preserve

- Keep one mutable owner per domain and compose it inside `Simulation`. Keep
  Bevy, protobuf and Python adapters outside the domain algorithms.
- Keep fixed simulation time independent of rendering and playback speed.
- Keep snapshots, replay interchange and causal checkpoints distinct.
- Keep bounded command queues and explicit refusal; make their semantics real
  before exposing them in operator controls.
- Preserve additive replay/proto compatibility and `environment.py` re-exports
  while consumers migrate. Delete code based on callers, not on port completion.
- Keep fixture parity and explicit PRNG/algorithm versions. Parity preserves a
  reference; it does not certify that the reference is physically accurate.

These largely implement existing [ADR-007](adr/007-runtime-composition-and-snapshots.md)
and [ADR-008](adr/008-world-aware-streaming-and-compatibility.md), rather than
introducing a competing architecture.

## Priority findings and changes

### P0 — Repair commands before binding the viewer

Concrete findings in `Simulation::apply_command`, `step`,
`run_apply_commands` and `CommandQueue`:

1. A faulted run accepts a queued Reset, but `step()` returns Faulted before
   draining commands. The direct `reset()` API works; the accepted command
   cannot clear the fault through this path. Completed runs similarly return
   before processing commands.
2. SingleStep only sets `paused = false`. Subsequent calls keep advancing; it
   does not execute exactly one tick and return to paused state.
3. Duplicate handling replaces the original status with `Duplicate`, losing
   whether the original command was accepted or rejected and its refusal
   reason. Dedup lookup precedes run/epoch validation and keys only on an integer.
4. Reset clears the pending queue, but commands already drained into the local
   `due` vector continue executing after it, even though they target the old
   epoch. Epoch/revision preconditions are checked at admission, not execution.
5. SetSpeed accepts nonfinite, zero and negative values. Past effective ticks
   can be acknowledged verbatim even though execution happens later.

Proposed contract: validate identity and payload, admit bounded work, then
revalidate applicable preconditions at the execution boundary. Distinguish
`Accepted`, `Applied`, `Rejected` and `Cancelled`; attach a duplicate flag to
the original result instead of replacing it. Key dedup by run/client/command
identity, define the retry horizon, and reject reuse with a different payload.
Make reset an epoch barrier that cancels remaining old-epoch work with explicit
outcomes. Drain eligible recovery commands while faulted or completed, without
executing domain stages. Use a one-tick budget for SingleStep. Define whether a
past requested tick is rejected or normalized and acknowledge the actual tick.

Tests: fault → queued reset → recovery; completed → reset; pause → single step
→ paused; duplicate rejected command retains reason; ID collision with different
payload; reset plus another due command; queue saturation; invalid speed;
scheduled precondition changes. Existing tests cover pieces, but the faulted
reset test checks acceptance rather than command-driven recovery.

Sources: [commands](../rust/argusnet-sim/src/command.rs),
[simulation](../rust/argusnet-sim/src/simulation.rs),
[transaction tests](../rust/argusnet-sim/tests/tick_transaction.rs).

### P0 — Make checkpoint completeness a composition requirement

`StageHandler` exposes only `run()`: handlers can retain mutable state, but have
no reset/checkpoint/restore contract. `Simulation::reset()` leaves those handlers
intact. The checkpoint carries clock/seed/pending commands, not handler state,
pause/speed or dedup history. `restore()` keeps the receiving object's dedup
history, restores the old epoch, and advances the clock in a loop proportional
to checkpoint tick. The recorded `world_content_hash` is currently assigned a
revision number and is not checked against an actual world hash.

Before adding production stages, introduce an explicit `SimulationState` with
owned subsystem state and typed stage inputs/outputs. Prefer concrete subsystem
fields over an unrestricted plugin framework; retain injectable handlers for
fault tests if useful. Checkpoints must contain every variable affecting future
decisions: estimator/controller state, planner claims, mission and battery state,
fusion history, RNG state where streams are persistent, pending commands and
the required idempotency state. Declare pacing/UI restoration policy explicitly.

Validate the entire checkpoint before mutation: world/config content identity,
versions, tick/header agreement, lengths, pending-command bounds and subsystem
invariants. Restore the clock in O(1). Restoring/branching into an active viewer
session needs a fresh publication epoch so old interpolation cannot bridge the
seek. Distinguish that epoch from the checkpoint's recorded provenance.

Exit: uninterrupted execution equals checkpoint/resume for actual mission
outputs, including a fresh process, with randomized checkpoint ticks, pending
commands and stateful stages. A shell restoring its tick counter is insufficient.

Sources: [checkpoint types and validation](../rust/argusnet-sim/src/snapshot.rs),
[reset/checkpoint/restore](../rust/argusnet-sim/src/simulation.rs).

### P1 — Separate compatibility confidence from physical evidence

Both localizers force confidence to 1.0 at timeout. Rust improves observability
with `LocalizationStatus::TimedOut`, but `fuse_estimates()` admits candidates by
spread alone, counts duplicate platform IDs repeatedly, and marks the result
Converged. These are deliberately preserved legacy semantics. Multiple estimates
using the same map can also share errors; inverse-variance weighting alone does
not establish independent evidence.

Keep a named legacy profile for parity. Introduce a separately versioned physical
profile where timeout is a mission-policy outcome and never manufactured
certainty. Deduplicate inputs, check timestamp freshness and status, propagate
provenance, and gate execution on uncertainty and estimator validity. Evaluate
correlation-aware fusion before selecting a replacement weighting rule.

The Python scan path also builds map observations using terrain queries and
passes node positions as localization priors. Make the provenance explicit:
synthetic sensor generation may use truth; mapping/planning should consume
declared measured or estimated inputs. Do not call the current bridge a verified
truth-isolated physical runtime.

Acceptance: hold observations fixed while perturbing hidden truth and verify
planning/localization invariance; score position error and uncertainty calibration
across seeds, delayed/missing sensors, correlated map errors and ambiguous terrain.
Report timeout, true convergence and mission completion separately. Confidence
must not improve simply because the same platform was listed twice.

Sources: [Rust localizer](../rust/argusnet-localization/src/lib.rs),
[Python localizer](../src/argusnet/localization/engine.py),
[scan accumulation and localization](../src/argusnet/simulation/sim.py).

### P1 — Improve accuracy evaluation before changing filters

`argusnet-core::build_metrics` matches truth by `track_id == target_id`. That is
appropriate for labeled fixtures, but generated association IDs can leave
unlabeled tracks unmatched. An empty error set must not imply good accuracy.

Keep operational fusion health independent of truth. Add an evaluation-only
matching step for unlabeled runs with explicit gates and unmatched penalties;
report matched fraction, false tracks, missed objects, identity switches, position
error and covariance calibration. Add crossing targets, weak bearing geometry,
long coasts and timestamp disorder to the qualification suite. Use these results
to decide whether filter/association changes improve accuracy, rather than
assuming a more complex filter will help.

Source: [fusion metrics](../rust/argusnet-core/src/lib.rs), `build_metrics`.

### P1 — Finish the stream contract already agreed in ADR-008

V2 has bounded channels, sequence/session metadata and drop counts, which are
good foundations. However, resume is a filter on future events, not retained
history; requests do not carry the previous session identity. A client resuming
with a high old sequence after a server restart can skip new-session events.
MissionStatus currently returns zero zones and an empty zone list. Clearing
`frame.truths` alone also leaves truth-derived metrics in the frame; define
whether the capability means hiding entities or withholding all truth-derived
information.

Add a negotiated stream alongside V2, with `(run_id, epoch, sequence)`, simulation
tick/time, world and belief revisions, event cursor and explicit capability
metadata. The resume request must name its session/epoch. Return either bounded
retained history or `ResyncRequired`; never imply recovery that cannot occur.
Use immutable manifests plus bounded, checksummed chunks for world transfer and
base/target revisions for patches. Install a complete revision before displaying
dependent frames. Serve real stored mission state.

Keep latest-value display traffic separate from the ordered event journal and
recording sink. Specify behavior when recording cannot keep up: backpressure or
an explicit recording failure, not silent frame loss. Avoid deep protobuf copies
per local consumer: publish shared domain snapshots and serialize at transport
edges. The server already shares the core frame through `Arc`; extend rather
than replace this work.

Tests: restart with stale resume token, lost patch base, corrupt/oversized chunk,
slow consumer, bounded memory, actual zone state, truth filtering, old/new client
compatibility, and direct-library/transport equivalence. Preserve existing TLS
and signed-ingest behavior; adding read streaming does not authorize remote control.

Sources: [server](../rust/argusnet-server/src/lib.rs),
[protobuf contract](../proto/argusnet/v1/world_model.proto).

### P1 — Keep precision in interchange and validation at the edge

Extracting `argusnet-replay` from Bevy was a good step, but many positions remain
`[f32; 3]`, and live conversion casts protobuf coordinates to f32. Treat that as
viewer legacy, not a general runtime/replay precision requirement. Keep f64
coordinates through domain and canonical interchange; convert to camera-relative
f32 only in rendering. Test large-coordinate scenes with small separations.

The custom covariance reader silently ignores malformed elements and accepts
arbitrary flattened lengths. Preserve documented legacy flat/nested encodings,
but reject malformed numeric content and validate covariance dimensions and
finite values at a canonicalization boundary. Audit fixtures and other producers
before tightening legacy readers; some current tests explicitly use 2×2 data.
Upgrade consumers before defaulting writers to a stricter version.

Use one conversion policy per boundary and semantic round-trip tests, not just
field-name parity. Define units, frame/origin identifiers, covariance ordering,
unknown-vs-zero values and algorithm provenance explicitly. Avoid using replay
DTOs as mutable simulation state.

Source: [replay DTOs and conversions](../rust/argusnet-replay/src/lib.rs).

## Performance work, in order

The historical profile reports 40% of tick-loop time in GridLocalizer, 13% in
VIO, and a separate gRPC timing experiment reports 5.9% of whole-run time.
Startup consumed 42% in that profile. Those figures are different denominators
and workloads, and many kernels have since moved: **re-profile this checkout
before estimating savings**. No new throughput claim is made here.

1. Compose a Rust headless mission using existing libraries, with per-stage
   timings, allocation/byte counts and work counters. Direct fusion removes a
   process boundary; it is not evidence by itself of a large speedup.
2. Finish scenario construction and remove Python/GLB baking from interactive
   Run. Load or construct the world once; render runtime terrain directly. Keep
   GIS and scene exports as offline tools. Measure time to first usable frame.
3. Share coverage work. `score_footprints()` allocates and builds a summed-area
   table on every call. Batch platforms against one immutable coverage revision
   and reuse its table; invalidate on mutation. Compare full rebuild versus dirty
   tile updates before adding a complex incremental index. Existing
   `CoverageGrid` already maintains statistics incrementally—do not reimplement it.
4. Keep revisioned belief tiles and immutable snapshots shared; avoid copying
   full maps into every frame. Stream recording in bounded chunks instead of
   retaining the entire run (`run_simulation` currently appends all frames).
5. Profile routing/LOS under the workloads that use them. Reuse route graphs by
   world/constraint revision where valid. Do not coarsen geometric cache keys
   without checking occlusion and clearance errors.
6. Only then parallelize independent sensor/particle work. Preserve keyed random
   streams and deterministic reduction order; keep mission mutation sequential
   until evidence supports splitting it. Do not introduce async per entity or a
   microservice per subsystem.

Benchmark cold startup, warm tick p50/p95/p99, sustained memory, subscriber
serialization cost, record/export throughput, and viewer CPU/GPU frame times
separately. Cover default scan, occlusion-aware mapping, dense targets, larger
maps and long recordings. Set numeric budgets from a designated machine and
workload; a nonempty render smoke test does not establish a 60 fps budget.

Sources: [historical profile](performance/baseline-profile-2026-09.md),
[footprint scoring](../rust/argusnet-mapping/src/lib.rs),
[coverage aggregates](../rust/argusnet-mapping/src/coverage.rs).

## Delivery sequence and exit gates

| Slice | Deliverable | Required evidence |
|---|---|---|
| 1. Foundation correctness | Command lifecycle fixes; explicit owned runtime state; checkpoint validation; current ownership table | Recovery/single-step/dedup tests; stateful reset and fresh-process restore equivalence |
| 2. Complete vertical mission | Rust scenario construction, sensing → fusion → mapping/localization → planning/safety → mission → snapshot, preserving traced causal order | One representative Rust-only mission with nonempty snapshots; then the existing mission/option fixture matrix; terrain-floor and safety-order invariants |
| 3. Belief and physical semantics | Mutable belief revisions; truth capabilities; separate legacy/physical profiles; calibrated localization and evaluation | Truth perturbation tests, duplicate/correlation cases, multi-seed error and uncertainty envelopes |
| 4. Viewer and live protocol | Dedicated simulation worker; bounded snapshots/commands; real pause/step/reset; world-aware stream | Responsive UI during slow ticks/builds; epoch-safe seek; restart/resync tests; frame and memory budgets |
| 5. Retire and optimize | Remove migrated Python execution paths; strict conversion boundaries; measured hot-path changes | Consumer reachability audit, installed-package/export compatibility, benchmark comparison on identical workloads |

Each slice can be multiple small PRs. Keep algorithmic changes separate from
parity ports so regressions can be attributed. Implement world assembly and
belief ownership as prerequisites for their dependent runtime features, not as
post-integration cleanup. Do not block an initial legacy-profile headless mission
on every future estimator improvement.

Operator-facing changes should include explicit build/run/fault/resync states,
acknowledged command outcomes, localization timeout/degraded labels, frame age
and dropped-frame counts, and an actionable error when world data is incomplete.
Configuration should resolve to a reproducible artifact carrying scenario,
algorithm/seed versions and world identity. Keep Python for evaluation,
reporting, GIS and exporters where it remains useful.

## Review limits

Inspected source, ADRs, migration records, fixtures/test structure and CI wiring.
Attempted `cargo test -p argusnet-sim --offline`, but Cargo was unavailable on
PATH; the conventional user Cargo path also could not execute. No runtime tests
or fresh benchmarks were completed. Historical test totals in migration docs
are historical evidence only. Findings above about control flow are static code
findings; proposed fixes require the named regression tests before release.

## Adopted documentation follow-through

This source review remains the findings record. The [roadmap](ROADMAP.md) and
[migration delivery plan](migration/DELIVERY_PLAN.md) incorporate its corrective
work. [Capability delivery](CAPABILITY_DELIVERY.md) adds the subsequently discussed
scorecards, failure scenarios, operator explanations, comparisons, energy-aware
replanning, persistent memory and adaptive inspection. The expanded
[viewer specification](VIEWER_UI.md) supplies their Operate / Plan / Review
surfaces and governs visual/interaction acceptance. Documentation adoption does
not mark any implementation package complete.
