# Verification, Determinism and Rollout

This document defines evidence required to close the work packages in
[DELIVERY_PLAN.md](DELIVERY_PLAN.md). It does not retroactively claim that all
these gates exist today. The current branch's tested measurements are in
[the implementation report](../performance/phase2-phase4-first-slices.md).

## 1. Evidence levels and change discipline

Track five distinct states: implemented locally, locally verified, pushed,
remote checks passed, merged/released. A successful push is not a green CI run;
branch protection is not assumed unless its configuration is inspected.

Each implementation PR records package ID, base commit, ownership change,
contract/algorithm versions, behavior differences, exact test commands/results,
benchmark machine/profile and rollback unit. Include an example before/after
payload for boundary changes. Keep mechanical movement, semantic fixes and golden
re-baselines in distinct commits. A dependency pin/toolchain update is reviewed
separately from a runtime algorithm change whenever possible.

Do not widen a tolerance because a port fails. First classify the difference as
serialization/ordering, numeric roundoff, changed random stream, intentional bug
fix or regression. Publish the evidence and update the relevant ADR/fixture only
when the difference is understood. Fixture generation scripts and provenance
must be committed, and tests must run from a clean checkout/archive at least
once before declaring a new fixture-based boundary complete.

## 2. Determinism contract and random streams

ADR-006 remains authoritative: fixed Rust seed/config is deterministic, with
cross-platform reproducibility as a release criterion. Current same-machine
parity tests do **not** prove that criterion. Do not silently replace it with a
numeric tolerance between Rust platforms. If exact cross-platform runtime state
cannot be achieved at acceptable cost, propose an explicit amendment before
accepting weaker behavior; this is a known delivery risk, not a solved property.

The planned seed contract, `argusnet.seed/v1`, is:

1. Root seed is an unsigned 64-bit integer. Inputs encode little-endian root
   seed and tick, then length-prefixed UTF-8 domain/platform/subject IDs and
   named stream purpose. Empty IDs have explicit zero length. Do not concatenate
   delimiter-separated labels or use process-randomized language hashers.
2. SHA-256 over the domain tag plus canonical input bytes produces a 256-bit
   key. Names distinguish sensing acceptance, bearing noise, bias, particle
   scatter, VIO features and world generation. World generation uses a stable
   generation index rather than wall time. IDs and draw purposes are versioned.
3. Use ChaCha12 with that key and a documented zero initial stream/block state.
   Pin its Rust implementation and test vectors in the P4.1 manifest/lockfile PR;
   no dependency version is guessed by this plan. Specify output word order and
   byte-to-u64 assembly in the fixture. Never seed from thread ID, entity spawn
   order, wall clock or a platform-dependent hash.
4. Uniform doubles use the top 53 bits of a specified u64 word multiplied by
   `2^-53`, giving [0,1). Normal draws use a versioned Box-Muller pair: `u1=1-u`,
   `u2=v`, `r=sqrt(-2*ln(u1))`, then `r*cos(2*pi*u2)` and `r*sin(2*pi*u2)`.
   Fix the pi constant's bit pattern, pair ordering and whether an unused second
   sample is consumed. Do not inherit an unspecified distribution implementation.
5. Pin and audit the transcendental math implementation used by deterministic
   runtime stages. Avoid implicit platform library variation, unordered floating
   reductions and undocumented FMA changes. Canonical output normalizes signed
   zero, rejects nonfinite state and sorts unordered records. Dynamic SIMD or
   parallel reductions must prove conformance before activation.

The exact seed encoding, algorithm/library pins and fixture vectors must land
before a production caller uses this contract. This is an intentional
stochastic re-baseline against Python, not a claim to emulate NumPy. The current
native scoring slice does not use this new RNG and preserves NumPy-generated
particle offsets.

Stateful processes (bias drift, integrators) own prior state explicitly; keyed
innovations do not remove their need for checkpointing. Adding/reordering an
unrelated entity cannot change another entity's stream. Physical interaction
can still change its trajectory; order-independence tests distinguish random
stream identity from legitimate coupled dynamics.

Required deterministic tests:

- Same seed/config/world/input journal produces identical canonical state and
  event sequences on repeated runs; runtime timestamp metadata is excluded.
- Node/target insertion order and worker scheduling do not change results.
- Record/replay inputs and checkpoint/restore match uninterrupted execution.
- Pause, render FPS, recording enabled/disabled and playback speed do not change
  final simulation state at a fixed tick.
- Linux x86_64 and macOS arm64 run shared RNG, math, world and full-state fixtures;
  compare canonical bytes/hashes. Add Windows when supported for runtime release.
- Old seed/algorithm versions are explicitly rejected or replayed by a retained
  compatible reader; never relabel changed output with an old version.

## 3. Python-to-Rust statistical qualification

Use the same resolved config, world-generation version, timestep, duration and
observation-source profile. Run baseline and candidate sequentially or with
isolated resources. Freeze the baseline commit, machine metadata and source
inputs. Begin with at least 30 predetermined seeds per supported scenario/profile;
use a separate held-out set for final acceptance. Estimate variability first and
increase sample size where intervals are too wide; a low-powered non-significant
result is not evidence of equivalence.

Profiles must include default and scan-map-inspect, occlusion-aware mapping,
frontier exploration, safety off/on, VIO pose substitution, low battery/egress,
missing coverage, dense obstacles and weather/vegetation. Use shorter boundary
fixtures in ordinary CI and full-duration multi-seed qualification in a scheduled
or release job. Rare events require targeted adversarial fixtures in addition to
random sampling.

Initial acceptance margins below are **proposed starting margins to calibrate
and freeze before evaluating a port**, not retrospective permission to change
behavior. Canonical scenarios may need stricter margins. Record the sample count,
confidence interval and reason for any different margin in the baseline PR.

| Metric | Starting equivalence/non-inferiority margin | Always inspect |
| --- | --- | --- |
| Observation acceptance and common rejection rates | ±1 percentage point absolute | Per reason/node/sensor, not just aggregate |
| Track position RMSE | No >5% relative increase with a predeclared 0.25 m absolute floor | p95 error, lost tracks and geometry subsets |
| Coverage at fixed tick | ±1 percentage point | Unknown mask and spatial differences |
| Mission completion rate | No >2 percentage point decrease | Failure mode counts and confidence interval |
| Completion time / battery use | No >5% increase | Tail latency and reserve violations |
| Localization confidence/calibration | Baseline-derived interval fixed before port | False convergence and timeout classification |
| Hard safety/terrain penetration | No new violating cases in the fixture corpus | Worst case, including opt-in mode and fallback behavior |

Require the paired bootstrap 95% interval for a difference/ratio to fit inside
its predeclared margin (or below its one-sided non-inferiority bound). Bound and
seed the bootstrap. For counts/rare rates use appropriate exact/binomial
intervals and targeted tests rather than a misleading normal approximation.
Predeclare primary metrics and account for multiple primary comparisons; do not
select favorable metrics after seeing results. Rejection categories with tiny
counts are reported explicitly rather than divided by unstable denominators.

An intentional bug correction such as the Phase 3 reverse-DDA fix is reviewed
with a minimal reproducer and expected new behavior. A distribution shift is
never explained only as "Rust is different." Once approved, regenerate goldens
in an isolated commit with a manifest of old/new counts and rationale. Existing
frozen gRPC transport fixtures change only when their own input/output contract
changes; retain old-reader fixtures even when adding new simulation baselines.

## 4. Required test matrix

| Layer | Required tests | Work packages gated |
| --- | --- | --- |
| Value/DTO | All-field round trips, null/unset/unknown fields, invalid shapes, old/new schema readers | G0, P4.0, P4.8 |
| Protobuf | Reproducible Python generation, Rust build, descriptor compatibility, old/new client matrix | G1, P5.3 |
| World/native | Existing terrain/collision/LOS fixtures, missing tiles, revisions, invalid handles, clone/drop, panic/error containment, concurrency | P3 hardening, P4.3 |
| Localization | Recorded offsets, covariance/status/reset, timeout, GNSS/VIO gaps and math conformance | P4.2, P4.4 |
| Sensing | Versioned source contract, gate/rejection stages, per-pair seeds, invalid input and truth isolation | P4.5 |
| Mapping | Raster axis/bounds/seams, aggregate-vs-full stats, dirty tiles, observation provenance | P4.6 |
| Planning/safety | Causal trace, stale revisions, unknown world, low energy, override ordering, physical floor | P4.7, P4.8 |
| Runtime | Direct/gRPC fixture parity, deterministic journal, tick failure atomic publication and reset | P4.1, P4.8, P4.9 |
| Viewer ECS | Stable identity, allocation/asset lifecycle, reload/seek/visibility, viewport/picking | P2.0-P2.4 |
| Rendering | WGSL pipeline, fixed visual fixtures, terrain/belief pixel correspondence, LOD seams | P2.1-P2.6, P5.1 |
| Worker/recording | Saturated queues, command ack/dedup, cancellation, no silent event loss, slow disk | P5.0, P5.2 |
| Streaming | Session restart, sequence gaps, corrupt/oversized chunks, old clients, TLS and truth capability | P5.3 |
| Packaging | Clean sdist/wheel, no source-tree path assumptions, Rust-only viewer/CLI, Python-only analysis | P4.9, P5.4, P6 |

For code-changing packages retain the workspace build/test/Clippy/fmt and
Python pytest/Ruff/mypy checks until the corresponding Python modules are
removed. Include replay-only viewer builds. Test installed artifacts, not only
`PYTHONPATH=src`. Run relevant meaningful targeted tests during iteration, full
required checks before merge, and no unnecessary repeated full runs for a
documentation-only follow-up.

## 5. Performance gates and visual evidence

Performance reports MUST separate build time, world construction, time-to-first-
usable-world, tick latency, presentation CPU, GPU time, shader warmup and native/
transport/recording overhead. Record release/debug profile, OS, CPU/GPU/driver,
resolution, quality tier, scenario hash, entities and sample duration.

G2 establishes a reference GPU machine and two fixed workloads: representative
mission and declared stress scale. For 60 Hz interactive acceptance, initial
budgets are p95 render frame <=16.7 ms and p99 <=25 ms over a warmed 120-second
run, with no unexplained >100 ms stall. Measure simulation independently: p95
tick processing below its fixed dt at 1x, bounded backlog and no dropped ticks.
These are target gates to calibrate on the named reference hardware; they are
not assertions about arbitrary laptops or lavapipe. Warmup and startup timing
are measured separately, never discarded without reporting.

Keep deterministic work counters as machine-independent regression gates. Use a
same-machine before/after timing comparison with a declared regression threshold
(initially 20%) and enough repetitions to characterize variance. If CI timing is
unreliable, use a dedicated runner and publish artifacts; do not replace its
golden with sentinel values. Query-kernel speedups are labelled as such and do
not imply full runtime throughput gains.

For image regression, use fixed scene/replay/epoch/camera/viewport/quality and
freeze shader animation time. Store small baseline PNGs plus manifest hashes,
and publish actual/baseline/diff artifacts. Compare perceptual error and semantic
regions (terrain visible, expected marker, observed/unobserved mask) with
thresholds measured from repeated runs on the same renderer. Maintain separate
hardware/software baselines if required; do not demand bit-identical pixels
across unrelated GPU drivers. Baseline updates require visual inspection and a
reason, not automatic acceptance. A shader compilation error always fails even
if the terrain still renders.

Long-run lifecycle tests seek/reload/toggle hundreds of times and assert bounded
entity, material, GPU tile, native handle and snapshot/checkpoint counts. Memory
budgets are per subsystem with observable eviction; monotonic growth is a failure
unless it is an explicitly bounded recording/history buffer.

## 6. Protocol compatibility and fault injection

Maintain a checked-in compatibility table for replay major/minor, checkpoint
format/runtime version, C ABI version/capabilities, world bundle axis/encoding
and V2/V3 stream clients. Test oldest supported reader and newest writer in both
directions where promised. Additive fields do not automatically guarantee
semantic compatibility: verify unset defaults and unknown enum behavior.

Fault tests cover missing world chunks, wrong checksum, revision races, duplicate
commands, stale epochs, full channels, paused consumers, recorder disk-full,
closed UI, cancellation while building, daemon restart and unavailable TLS files.
Expected result includes explicit error, retained last valid world/snapshot,
released resources and documented retry behavior. Do not turn corrupt terrain
into a successful zero-height query or a reconnect gap into interpolated motion.

Upgrade protocol in this order: tolerant readers/capability handshake, new
producer behind explicit selection, dual-version integration tests, then default
selection after qualification. Keep V2 until supported clients have a migration
path. Do not add remote actuation during a data-stream upgrade.

## 7. Rollout and rollback units

| Stage | Enablement | Rollback |
| --- | --- | --- |
| Native leaf/estimator port | Python delegates after exact/qualified parity | Revert the port and its adapters as one versioned unit; never silently fall back at runtime |
| Rust full CLI | Explicit development command/runtime choice | Retain supported Python release during qualification; outputs label engine version |
| Rust viewer worker | Explicit capability/config until acceptance | Revert to a known compatible release; do not leave two independently mutating worlds in one viewer |
| New live protocol | Capability-negotiated alongside V2 | Client selects supported old protocol with explicit missing-world limitations |
| Default Rust entry point | Release migration notes and installed-artifact tests | Reinstall previous compatible package versions, preserve replays/configs |
| Python module removal | After caller inventory and compatibility release | Restore the matching package/runtime release, not arbitrary individual modules |

No destructive conversion of old replays/checkpoints is part of rollout. New
exports write separate versioned artifacts. Checkpoints may be runtime-version
specific; mismatches fail rather than claiming to resume accurately. Keep user
configuration and original artifacts intact during upgrade.

## 8. Release gates and unresolved decisions

| Risk / decision | Owner role | Deadline / resolution evidence |
| --- | --- | --- |
| Exact cross-platform floating state under ADR-006 | Runtime maintainer | Before P4.2 stochastic switch; conformance artifacts or explicit ADR amendment |
| Statistical thresholds and canonical seed corpus | Evaluation maintainer | Before evaluating each semantic port |
| Reference GPU/quality tiers and memory ceilings | Viewer maintainer | G2 before effects/default 60 fps claims |
| Rich DTO fields vs core model differences | Protocol + runtime reviewers | P4.0 field matrix before deletion |
| Legacy timeout/true-world planner behavior | Mission + safety reviewers | Explicit profile during port; separate behavior PR before new physical-mode claims |
| Bounded world transfer/chunk encoding and authentication capabilities | Protocol maintainer | P5.3 proto review and fault tests |
| Python/Cargo command-name collision and wheel targets | Release maintainer | P4.9 coexistence plan; P6.2 final ownership |
| Recorder/checkpoint memory budgets on large worlds | Runtime + viewer reviewers | P5.2 stress evidence before rewind default |
| Native global registry contention | Native maintainer | Profile with multiple active consumers before expanding locks/threads |

The role names identify responsibility, not currently assigned people. Each work
package PR must name actual owners/reviewers. None of the open calibrations blocks
publishing this plan; it blocks the dependent implementation's acceptance.

Final migration acceptance requires one authority per runtime concern, Rust-only
interactive boot and simulation, preserved offline Python workflows, versioned
interchange compatibility, tested failure recovery, named-hardware performance
and updated user-facing docs. Remaining product capabilities such as richer
inspection intelligence or persistent spatial memory stay in the product
roadmap; this runtime migration must not falsely mark them complete.

## 9. Next-refactor and feature acceptance additions

These extend existing gates; they are required evidence for the relevant new
slice, not claims that the tests already exist.

| Scope | Required regression / acceptance |
| --- | --- |
| Commands | Faulted/completed queued reset; paused SingleStep stays paused after one tick; duplicate rejection reason; payload/ID collision; reset barrier in a due batch; invalid speed; scheduled precondition changes |
| Checkpoints | Stateful handler reset; fresh-process resume equivalence; pending/dedup state; corrupt world identity and oversized queue rejected before mutation; seek invalidates interpolation |
| Physical semantics | Fixed observations with perturbed hidden truth; timeout separate from convergence; duplicate/correlated localization inputs; delayed and missing measurements |
| Fusion evaluation | Unlabeled matching, unmatched fraction, false/missed tracks and identity changes; missing scores never become zero error; covariance calibration |
| DTO / streaming | Large-coordinate small-separation precision; malformed covariance; session restart with stale resume; missing patch base; truth-derived metric filtering |
| F1–F4 | Outcome reason attribution, repeatable fault schedules, decision chain completeness despite display drops, paired seed/config comparisons including failed runs |
| F5–F7 | Reserve-triggered return/handoff without thrashing; two-mission persistent site reload/change alignment; insufficient evidence triggers bounded safe revisit or explicit unmet result |
| Viewer | Operate/Plan/Review source semantics; no fabricated backend fields; acknowledged command lifecycle; accessibility and fixed visual/lifecycle fixtures from VIEWER_UI |

Use [feature-specific criteria](../CAPABILITY_DELIVERY.md) and
[viewer acceptance](../VIEWER_UI.md). Viewer p95 ≤16.7 ms and p99 ≤25 ms are target
budgets on named hardware/workloads, not measured results. Record cold startup,
warm CPU/GPU frames, tick latency, recording backlog and memory independently.
Historical localization/gRPC percentages do not establish current optimization
priority. Benchmark shared coverage tables and tile updates before adding
parallel execution or complex caches.

## 10. Expanded viewer/system fixtures

Adopt [the feature UI acceptance matrix](../VIEWER_FEATURE_INTEGRATION.md):
overhang/underpass occupancy, vertically/time-separated paths, lost/recovered
localization, disconnected platform with unobserved landing, stale prior map,
rejected/expired command, insufficient evidence, misaligned change comparison,
recording gaps, model out-of-distribution result and unavailable export.

Verify identical typed selection and revision/time in Perspective, Top-down and
Section; clipping must not imply absence. Only the timeline changes source time.
Plan edits require vertical/time validation; Review differences retain alignment
uncertainty. SIL/HIL results show frozen versions and predeclared deadlines,
containment, fallback and measured terminal outcomes. These complement rather
than replace the system plan's staged qualification ladder; UI tests alone do
not qualify aircraft behavior.
