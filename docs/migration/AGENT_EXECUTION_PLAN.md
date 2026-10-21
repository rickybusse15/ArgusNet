# Agent execution plan: finishing the ArgusNet target

> **Status: proposed plan, not accepted.** It sequences work that
> [DELIVERY_PLAN.md](DELIVERY_PLAN.md), [CAPABILITY_DELIVERY.md](../CAPABILITY_DELIVERY.md),
> [AUTONOMOUS_SYSTEM_PLAN.md](../AUTONOMOUS_SYSTEM_PLAN.md),
> [VIEWER_UI.md](../VIEWER_UI.md) and [PARALLEL_TICK_PLAN.md](PARALLEL_TICK_PLAN.md)
> already specify. It adds no new product requirement and **does not renumber**
> any P/F/S/G identifier. Where this plan and an acceptance contract in those
> documents disagree, the contract wins and this plan is wrong.
>
> **Written 2026-10-03** against `main` at `0b1032c`. The starting state in §1
> was read from the repository; everything marked *unverified* must be checked
> by the executing agent before it is relied on.

## Contents

0. Operating protocol for the executing agent
1. Verified starting state
2. Workstreams, dependencies and waves
3. Wave 0 — housekeeping and decisions
4. WS-A — finish the Rust runtime (S0)
5. WS-B — scheduled and parallel execution (P4.10)
6. WS-C — viewer engine and instrument shell (P2.4–P2.6)
7. WS-D — viewer runtime integration (P5)
8. WS-E — guardrails (G0, G2, G3)
9. WS-F — product features and client contracts (F1–F7, S1/S2)
10. WS-G — Python reduction (P6)
11. WS-H — hardware track (S3–S8): contract work only
12. Cross-cutting checklists and commands
13. Risks
14. Decision register (human-owned)
15. Definition of done for the whole target

---

## 0. Operating protocol for the executing agent

### 0.1 Non-negotiable rules

These come from `CLAUDE.md` and the accepted ADRs. A task that appears to need
breaking one stops and escalates (§0.5).

1. **Rust is the source of truth** for fused object state. No duplicated fusion
   math in Python.
2. **Proto changes** update Python bindings (`src/argusnet/v1/`) **and** Rust
   (`argusnet-proto`, `argusnet-core`, `argusnet-server`) in one change, using
   `python3 scripts/regenerate_proto.py` and verifying with
   `python3 scripts/regenerate_proto.py --check` and
   `python3 scripts/check_field_parity.py`. Never hand-edit generated files.
3. **Replay metadata changes are additive** unless viewer and tests change
   together. `environment.py` keeps re-exporting moved symbols.
4. **Determinism:** fixed seed + config ⇒ identical committed state. Worker
   count, pacing, render rate and recording on/off must not change it
   (ADR-006, ADR-009). Never call the platform libm in a draw path (ADR-006 §6).
5. **Behavior changes are separate from mechanical ports.** A refactor slice
   proves it is behavior-neutral (§12.3). A behavior change gets its own
   version label, rebaselined fixtures in a commit that changes nothing else,
   and a rollback boundary.
6. **Truth is a capability.** Only `Stage::may_read_truth` stages and
   evaluation receive it. Never widen that list to make something work.
7. **Never present a missing field as zero, empty, or confident.** Unknown,
   stale and unavailable are distinct states everywhere (viewer, replay,
   reports).
8. **Physical collision never pushes an entity below terrain.**
9. **No hardware claim.** Nothing in this plan may be labelled SIL, HIL,
   scheduler-qualified or aircraft-ready without the evidence in
   [HARDWARE_INTEGRATION.md](../HARDWARE_INTEGRATION.md).
10. **Do not skip, disable or quarantine a test to get green.** Do not widen an
    acceptance envelope to make a scenario pass (the `sar_person_search` case
    in §3 is a decision for a human, not a tolerance to loosen).
11. **Meters, radians, Z-up/datum conventions** as in `CLAUDE.md`.

### 0.2 Session start checklist

Run these before writing code, and record the results in the first PR's body.

```bash
git fetch origin main && git status -sb && git log --oneline -10
# which slice docs / STATUS rows exist for the package you are about to touch
grep -n "<package id>" docs/migration/STATUS.md docs/migration/DELIVERY_PLAN.md
scripts/ci-local.sh --list
scripts/ci-local.sh --quick        # python, rust-fast, rust-lint: your baseline
```

If the baseline is red on `main`, **stop and report which check and on which
commit before adding changes** — do not mix baseline repair with a slice.
Needs system packages for Bevy builds: `pkg-config libasound2-dev libudev-dev`.

### 0.3 Slice discipline

- One PR closes **one package or one clearly named sub-slice** (`P4.10a.2`,
  `P2.5c`). A slice that needs a prerequisite that is not merged says so in
  its description and is stacked on it, as #128/#129 are.
- Every slice starts with a **handoff block** (copy into the slice doc):

  ```text
  Package:        <ID>            Base commit: <sha>
  Inputs:         <contract/schema versions consumed>
  Outputs:        <contract/schema versions produced>
  Crates/files:   <list>          Downstream consumers: <list>
  Verify with:    <exact commands>  Fixtures: <paths>
  Known limits:   <list>          Rollback boundary: <what to revert>
  ```
- Every slice adds a **slice doc** `docs/migration/<NAME>.md` containing a
  verification table (check → result, with the command), following the
  existing slice docs (`TICK_STAGES.md`, `PICKING_AND_SELECTION.md`).
  `STATUS.md` rows change **only** when the slice doc's verification table does.
- Update `STATE_OWNERSHIP.md`, `KNOWN_GAPS.md` and `docs/changelog.md` in the
  same PR when ownership or implemented behavior changes. No "implemented"
  label may describe a proposed interface.
- Size guide: ≤ ~1,000 changed lines of non-generated, non-fixture code. Split
  mechanical moves (no logic) from logic changes.

### 0.4 Evidence vocabulary

Reports use exactly these words, as `DELIVERY_PLAN.md` §2 requires:
**locally verified** → **pushed** → **PR checks passing** → **merged**. Add
**released** only for an installed package. "Tests pass" without naming the
command and the machine is not evidence. A software-rendered (Lavapipe) run
validates shaders, not frame rate; a desktop benchmark is not an onboard
readiness result.

### 0.5 When to stop and ask a human

Stop, write down the question with options and a recommendation, and continue
on an unblocked slice. Items already known to need a human are in §14. In
addition, always stop for:

- anything that would change an acceptance envelope, golden baseline, schema
  version or wire contract beyond the slice's declared scope;
- a finding that contradicts an ADR (write an ADR amendment proposal);
- requiring named reference hardware, a licensed font, a signing key or any
  credential;
- merging, approving, force-pushing, or deleting history (agents prepare PRs;
  humans merge);
- any step that would make the viewer capable of commanding a real aircraft.

### 0.6 One agent or several

The delivery plan does not require multiple agents. If several run, use these
**file-ownership lanes** so no two touch the same file concurrently; protocol
and DTO changes are reviewed jointly.

| Lane | Owns | Must not edit |
| --- | --- | --- |
| Runtime | `rust/argusnet-sim`, `argusnet-core`, `argusnet-planning`, `argusnet-mapping`, `argusnet-localization`, `argusnet-sensing`, `safety-engine`, `argusnet-replay` | viewer, proto without joint review |
| Viewer | `rust/argusnet-viewer` | domain crates |
| Protocol | `proto/`, `argusnet-proto`, `argusnet-server`, `src/argusnet/v1/`, `src/argusnet/adapters/` | — (joint review required) |
| Evaluation | `src/argusnet/evaluation/`, report/scorecard schemas | runtime stage code |
| Infra | `.github/`, `scripts/`, `docs/` index files | domain logic |

Within the Runtime lane, **WS-B0 (timers) lands before WS-A1 (ownership
refactor)** because both edit `simulation.rs`/`stage.rs`.

### 0.7 PR description template for agent PRs

```text
## Package
<ID> — <one line>. Stacked on: <PR or "none">.

## What changes
- behavior-neutral / behavior change (version label: ...)
- files and crates

## Verification
| Check | Command | Result (locally verified / CI) |

## Evidence of non-regression
<golden hashes unchanged | rebaselined in commit <sha> only>

## Not done / known limits
## Rollback boundary
```

---

## 1. Verified starting state

> This section is the snapshot the plan started from (2026-10-03). It is not
> kept current: [STATUS.md](STATUS.md) is the live record of what has merged
> and what is open.

Read from the repository on 2026-10-03 unless marked *unverified*.

### 1.1 What exists

| Area | State | Source |
| --- | --- | --- |
| Rust tick substrate | Eleven-stage pipeline, `StageHandler` trait, truth capability, trace, command lifecycle (`CommandEnvelope`, `CommandAck`, `AckStatus`, `CommandQueue`), format-2 control checkpoints | `rust/argusnet-sim/src/{stage,simulation,command,world}.rs` |
| Stage bodies | Battery, controllers, mission overrides, deconfliction, safety monitor, VIO/pose, observations, fusion, mapping/localization, mission transitions, replay writer | `git log` P4.8c–P4.9d |
| Rust CLI | `argusnet-rs` runs and builds scenario bundles with no Python; replays carry viewer data; 6 of 7 canonical scenarios pass every declared envelope | `STATUS.md`, `PHASE_ACCEPTANCE.md`, `SAFETY_AND_PROFILES.md` |
| Native builder | Terrain, land cover, obstacles, scenario entities, numpy-compatible RNG halves | `NATIVE_BUILDER.md` |
| Viewer P2.0–P2.3 | Snapshot adapter, belief material, sensor/zone/trail geometry, occlusion-aware picking, GPU pixel probes | merged (`25a1985`, `fe09db1`) |
| Guardrails | G1 (proto regeneration + field parity) done | `STATUS.md` |

### 1.2 Open, in flight, or stale

| Item | State |
| --- | --- |
| PR #128 (P2.4a time controls) and #129 (P2.4b camera, stacked) | open; CI pending when last checked |
| PR #130 (CI: keep cargo bin on self-hosted runner), #131–#133 (Dependabot) | open |
| PR #134 (parallel tick plan, this doc's sibling) | open draft |
| `STATUS.md` rows | **stale**: still say "local" for work now merged; #128 updates part of it |
| `KNOWN_GAPS.md`, `RUST_RUNTIME_MIGRATION.md`, `STATE_OWNERSHIP.md` banners | dated 2026-09-23 / baseline `ec9c2f7`; likely understate progress |
| `sar_person_search` envelope `covariance_reduction_mean` | **fails** by metric definition; needs a human decision (D-1) |
| Production ownership | `Arc<Mutex<Shared>>` link handles remain in ten files: `battery.rs`, `controller.rs`, `overrides.rs`, `deconflict.rs`, `safety.rs`, `vio.rs`, `observe.rs`, `fusion.rs`, `mapping.rs`, `mission.rs` (`grep -rl 'Arc<Mutex' rust/argusnet-sim/src`); the architecture says to replace them with owned state |
| Production-constructor check | No code found that rejects a missing required stage (grep for "required"/"production" in `argusnet-sim/src` found none) — *verify before A1.3* |
| Reset/restore completeness | Partial per `KNOWN_GAPS.md`; `CAUSAL_STATE.md` audit exists for stateful stages |
| Scheduler / parallel execution | Partial: integer-time scheduler and phased lane executors (B2.1–B2.3), per-stage timers (B0), fork-join workers (B3), timestamped delivery (B4 first slice), the mission lane at its own rate under `scheduled_v1` (B2.4). No background job pool (B5) and no pending-job restore (B6) |
| Guardrails G0, G2, G3 | open |
| Viewer P2.5 shell, P2.6 acceptance | not started (no PR seen) |
| Viewer P5 | D1, D2 and D5's viewer part done: `--run` and Tools → Scenario run the Rust simulation in process and scenes are built in Rust; the viewer starts no process (`app::scenario_run_tests::the_viewer_starts_no_processes`) |
| F1–F7, S1/S2 contracts | not started except what pre-exists: `src/argusnet/evaluation/{scorecard,reports,benchmarks}.py` are the F1/F4 starting points |
| Hardware S3–S8 | planning docs only; no hardware profile frozen |
| Open question from `STATUS.md` | confirm P4.8d.4 ("world-backed stage 1 + Python-vs-Rust stage-1 parity") was closed by P4.8e.x |

---

## 2. Workstreams, dependencies and waves

### 2.1 Workstreams

| ID | Workstream | Closes | Lane |
| --- | --- | --- | --- |
| WS-0 | Housekeeping and decisions | — | Infra |
| WS-A | Finish the Rust runtime | S0, P4.8 gate, P4.9 residue | Runtime |
| WS-B | Scheduled and parallel execution | P4.10a–f | Runtime |
| WS-C | Viewer engine and shell | P2.4–P2.6 | Viewer |
| WS-D | Viewer runtime integration | P5.0–P5.5 | Viewer + Protocol |
| WS-E | Guardrails | G0, G2, G3 | Infra |
| WS-F | Product features and client contracts | F1–F7, S1, S2 | Evaluation + Runtime + Viewer |
| WS-G | Python reduction | P6.0–P6.4 | Infra + Evaluation |
| WS-H | Hardware track | S3–S8 contracts only | Runtime + Infra |

### 2.2 Critical path

```text
WS-0 ─► WS-A (A1 ownership, A2 restore, A3 qualification, A4 journal)
          │
          ├─► WS-D P5.0 (needs A1+A3) ─► P5.2 (needs A2,A4) ─► P5.4 ─► P5.5 ─► WS-G
          │
          └─► WS-B (B0 timers → B2 scheduler → B3 parallel → B4..B7) ──► promotion
WS-C (P2.4 → P2.5 → P2.6) runs alongside; feeds P5.5
WS-E G2 precedes any expensive-effect default; G0 precedes the first wire change
WS-F F1/F2/F3 replay-first slices can start in Wave 1; F5–F7 need A + B contracts
```

### 2.3 Waves

A wave is a set of slices that may proceed concurrently (disjoint lanes). A wave
gate must hold before the next wave's dependent slices start.

| Wave | Slices | Gate to leave the wave |
| --- | --- | --- |
| 0 | 0.1–0.5 | `main` green; STATUS refreshed; decision requests filed |
| 1 | B0; E-G0, E-G2; C1 (tokens, fonts); F-S1 contract drafts; G-P6.0 inventory (read-only) | Rust per-stage profile published; G2 baselines recorded; tokens merged |
| 2 | A1, A2; C2–C4; B1, B2; F1, F2, F3 (replay-first); E-G3 | Reset == fresh and restore == uninterrupted across the 7 scenarios; serial scheduler equals legacy |
| 3 | A3, A4; B3, B4; C5–C7; D1; F4, F5 | Worker-count invariance; `argusnet-rs` qualified over 30 seeds + held-out; P5.0 controls acknowledged |
| 4 | B5, B6; C8; D2, D3; F7 (single-mission), F6 design + S2 spatial contract | Pending-job restore equals uninterrupted; seek/branch semantics tested |
| 5 | B7; D4, D5, D6; G-P6.1–P6.4; H contract artifacts | Scheduled promotion decision; no Python descendant of the viewer; installed-package gates |

---

## 3. Wave 0 — housekeeping and decisions

### 0.1 Land the open PRs

**Goal:** get `main` to include #128, #129, #130, #134 and the Dependabot bumps
before dependent work starts.
**Do:** watch CI on each; fix only failures caused by the PR; stack order is
#128 → #129; Dependabot bumps merge independently. Agents do **not** merge or
approve — request human merge.
**Exit:** `main` includes them, `scripts/ci-local.sh --quick` green on `main`.

### 0.2 Baseline health run

**Do:** on current `main`, run `scripts/ci-local.sh` (all jobs). Record per-job
result. For `rust-viewer`, also run the three GPU probe tests with
`ARGUSNET_GPU_TESTS=1` (they need software Vulkan; see `ci.yml`).
**Exit:** a dated table in `STATUS.md` ("Baseline run") listing job → pass/fail/skip
with the commit. A **skip is not a pass**.

### 0.3 Refresh the stale status documents

**Docs-only slice.** Do not change a row unless a command in §0.2 backs it.
1. `docs/migration/STATUS.md`: move merged rows from *local* to *done*, with
   the merge commit (find via `git log --merges --first-parent`).
2. Update the dated banners in `KNOWN_GAPS.md`,
   `RUST_RUNTIME_MIGRATION.md`, `STATE_OWNERSHIP.md`, `DELIVERY_PLAN.md` §1
   baseline table, and `AUTONOMOUS_SYSTEM_PLAN.md` "Current phase integration".
3. Confirm and record whether P4.8d.4 is closed.
4. Do **not** edit acceptance contracts.
**Exit:** every changed claim cites a commit or command; diff reviewed for
"implemented" wording on anything proposed.

### 0.4 Decision requests (see §14)

File each decision in §14 as an ADR stub (`docs/adr/010-…`, `011-…`) using
`docs/adr/000-template.md`: context, options, a recommendation, consequences.
Agents propose; humans accept. Do not block unrelated slices on them.

### 0.5 ADR-010: parallel execution model

Promote [PARALLEL_TICK_PLAN.md](PARALLEL_TICK_PLAN.md) §4–§8 into an ADR
extending ADR-009: phase barriers, ordered merge, revisioned background jobs,
three rate mechanisms (static/data-triggered/load-adaptive, the last excluded
from deterministic simulation).
Drafted as [ADR-010](../adr/010-parallel-execution-model.md) (*proposed*).
**Exit:** ADR merged as *proposed*; B3+ cite it.

---

## 4. WS-A — finish the Rust runtime (S0)

**Target (S0):** a nonempty representative mission owned by Rust, direct fusion,
immutable snapshots, complete checkpoints, durable events, no Python callback
inside a tick. Reference: [DELIVERY_PLAN.md](DELIVERY_PLAN.md) §6 "P4.8 completion
gate" and "P4.9".

### A1 — Production ownership and lifecycle

#### A1.1 Inventory the shared links (read-only)
**Do:** list every `Arc<Mutex<…>>` link (`BatteryLink`, controller,
`OverrideLink`, deconflict, `SafetyLink`, `VioLink`, `ObserveLink`, fusion,
mapping, `MissionLink`; re-run the grep in §1.2 for any added since), and for each:
who constructs it, who reads/writes outside the stage, and what that caller
actually needs (seed fleet, read fractions, take events, watch mission).
**Output:** `docs/migration/SESSION_OWNERSHIP.md` with the table plus a target
design: a `Subsystems` struct owned by the session, narrow borrowed inputs
(`&BatteryState`, `&mut MissionState`) passed through `TickContext`, no
interior-mutable aliasing between stage and caller.
**Exit:** reviewed design; every external caller (CLI, tests, builder, viewer)
has a replacement access path (query methods on `Simulation` or a read-only
`SessionView`).

#### A1.2 Replace links one at a time
**Do (per link, one PR each, in this order: battery → controller → overrides →
deconflict → safety → VIO → observe → fusion → mapping → mission, i.e. stage
order, except that mission goes last because mapping watches it):**
1. Introduce owned state + accessor on `Simulation`; keep the old link as a
   thin shim delegating to it.
2. Migrate callers; delete the shim.
3. Keep stage-checkpoint payloads **byte-identical** (`checkpoint_state`/
   `restore_state` contract), so existing checkpoints still load.
**Tests (every PR):**
- the existing stage unit tests unchanged;
- `rust/argusnet-sim/tests/{mission_tick,assembled_resume,tick_transaction}.rs`
  unchanged and green;
- **trace-equality test (new, once, reused):** run the 7 canonical scenarios,
  hash the `StageTrace` and the written replay; hashes equal the pre-refactor
  hashes captured in the slice doc.
**Pitfalls:** do not change stage order or event emission timing; the battery
"first tick is free" and "no hover cost" behaviors are pinned and **must not
change** here (D-3 is a separate decision).
**Exit:** no `Arc<Mutex<Shared>>` link between stage and external caller remains;
trace hashes unchanged.

#### A1.3 Production constructor
**Do:** add `Simulation::production(config) -> Result<Simulation, BuildError>`
that fails when a required stage handler is `UnportedStage`; rename the current
empty constructor to a test-only one (`#[cfg(test)]` or `Simulation::bare()`
documented as tests-only). The CLI and (later) the viewer worker call only
`production`.
**Tests:** constructing without each required stage fails with the stage name;
a full bundle constructs; `argusnet-rs` still passes `rust/argusnet-cli/tests/cli.rs`.
**Exit:** "missing required production stages are rejected" is a tested property.

### A2 — Reset equals fresh; restore equals uninterrupted

**Goal:** close "subsystem-wide reset/restore" in `KNOWN_GAPS.md`.
**Depends on:** A1.2 (owned state makes reconstruction tractable).

#### A2.1 Reset reconstructs everything
**Do:** `Simulation::reset(&mut self)` rebuilds controllers, VIO/fusion,
sensor biases, mission/safety/battery and belief from the **resolved
configuration**, not from mutated state. Epoch increments; old-epoch events and
pending commands are discarded and reported.
**Test:** for each canonical scenario, `run(N)` → `reset()` → `run(N)` ==
fresh `run(N)` (replay hash equal, including RNG-derived output).

#### A2.2 Checkpoint completeness and identity
**Do:** every stateful handler participates in the checkpoint contract
(`CAUSAL_STATE.md` is the inventory — add anything missing: RNG stream
positions, pending commands/dedup history, VIO filters, mapping accumulators,
particle sets). A checkpoint carries a **content hash of the resolved world and
config**; `restore` validates it **before mutating anything**. Old formats that
cannot resume a full mission through defaults are rejected with a versioned
error, never silently defaulted.
**Tests:**
- property test: for random tick `k` per scenario, `run(N)` ==
  `run(k) → checkpoint → fresh construct → restore → run(N-k)` (replay hash);
- restore with a different world hash fails with no state change (assert state
  hash before == after);
- truncated/corrupt checkpoint bytes fail cleanly.
**Pitfalls:** restore must not rely on `Default` for anything causal; add a
mutation test that zeroes one field and must make the equality test fail.
**Exit:** `KNOWN_GAPS.md` row "Subsystem-wide reset/restore" → Implemented, with
the test names.

### A3 — Qualification residue

#### A3.1 Resolve the `sar_person_search` envelope (needs D-1)
**Do after the decision:** implement the chosen option (redefine the metric,
move it to a different metric family, or fix a real defect it exposed).
Document in `SAFETY_AND_PROFILES.md`. **Never** widen the tolerance silently.
**Exit:** 7/7 canonical scenarios scored; the envelope change (if any) is its
own commit.

#### A3.2 Statistical qualification
**Do:** per `VERIFICATION_AND_ROLLOUT.md` §3: ≥30 predetermined seeds per
supported scenario/profile, a **separate held-out set** for final acceptance,
profiles listed there (default, scan-map-inspect, occlusion-aware mapping,
frontier, safety on/off, VIO substitution, low battery/egress, dense obstacles,
weather). Use `scripts/run_phase_acceptance.py` (release build); extend it for
the profile list and the held-out split. Short fixtures in ordinary CI;
full-duration runs in a scheduled job (`nightly-bench.yml` is the precedent).
**Exit:** published report in `docs/performance/` with variance estimates;
failed runs and missing measurements listed, not dropped.

#### A3.3 Cross-platform determinism
**Do:** add a job that hashes canonical-bytes outputs of RNG, math, world and
full-state fixtures on Linux x86_64 and macOS arm64 (ADR-006). Record the
explicitly documented world-generation exceptions rather than hiding them.
*Needs a macOS runner — escalate if unavailable.*

#### A3.4 Truth-isolation tests
**Do:** physical-mode test: hold measurements fixed, change hidden truth,
assert planning/mapping/safety outputs identical. Assert
`Stage::may_read_truth` is exactly {VIO/pose, observations}.

### A4 — Durable event journal, recording and crash recovery

**Do:**
1. Ordered **event journal** separate from presentation frames, with a durable
   cursor and explicit "incomplete" state.
2. Immutable **run configuration** written at start; checkpoints bind to it.
3. Recorder backpressure: for deterministic sim, required-recording pressure
   **suspends advancement at a committed boundary** and reports it; never drops
   events.
**Tests (from the audit gates):** mid-write crash, journal truncation/corruption,
missing checkpoint content, immutable-config mismatch, old-session messages.
**Exit:** restart-after-crash resumes from the last durable cursor or reports an
explicit incomplete artifact.

---

## 5. WS-B — scheduled and parallel execution (P4.10)

**Specification:** [RUST_RUNTIME_ARCHITECTURE.md](../RUST_RUNTIME_ARCHITECTURE.md) §3–§5,
[ADR-009](../adr/009-independently-scheduled-runtime.md),
[PARALLEL_TICK_PLAN.md](PARALLEL_TICK_PLAN.md). `legacy_stages_v1` stays
untouched throughout; `scheduled_v1` is opt-in until B7.

### B0 — Instrumentation (P4.10-pre)

**Do (first, small, additive):**
- Per-stage wall time in `StageRecord` (use `std::time::Instant`; **excluded
  from determinism hashes and replay equality**), plus work counters
  (platforms, observation pairs, footprints, tiles touched, particles).
- A `--timing` option on `argusnet-rs sim` printing p50/p95/p99 per stage and
  startup/tick/replay-write split.
- Allocation counters where the pattern in
  `rust/argusnet-viewer/tests/presentation_budget.rs` can be reused.
**Run:** the 7 canonical scenarios on a **release** build, ≥5 runs, same
machine; record machine metadata.
**Output:** `docs/performance/rust-tick-profile-<date>.md`: per-stage share
table and an **Amdahl table** ranking parallelisation candidates. This replaces
the Python-era profile as the basis for every B-slice decision.
**Exit:** profile merged; timers confirmed not to perturb replay hashes.

### B1 — Work reduction first (P4.10c, pulled forward)

**Do, in profile order (only items the B0 table shows are material):**
1. Shared coverage summed-area tables / dirty tiles (`argusnet-mapping`);
   `FOOTPRINT_RASTERIZATION.md` and `COVERAGE_AGGREGATES.md` are the starting
   docs.
2. Spatial index for obstacle/terrain queries on per-tick hot paths.
3. Buffer reuse for per-tick `Vec` allocations in stage bodies.
4. No-input skips (a stage with no new input reports `NoWork`).
**Tests:** unchanged-input outputs bit-identical; work counters drop;
before/after CPU and allocation numbers (five runs, same machine).
**Exit:** counters and timings in the slice doc; golden hashes unchanged.

### B2 — Scheduler core, serial executor (P4.10a)

**Where:** a new module/crate under `argusnet-sim` (`scheduler/`), no Bevy.
**Types (sketch; adjust to idiom):**

```rust
pub struct SimTime(i64);                       // integer ns since run epoch; checked ops only
pub enum Schedule { Period(NonZeroU64 /*ns*/), Rational { hz_num: NonZeroU64, hz_den: NonZeroU64 }, Once(SimTime) }
fn nth_deadline(epoch: SimTime, s: &Schedule, n: u64) -> Result<SimTime, TimeError>; // u128 intermediate, ceil to next ns, never add a rounded period
pub enum Phase { Propagate, Commands, Captures, Deliveries, Estimators, Mapping, MissionPlanning, ControlSafety, Publication } // order in `RUST_RUNTIME_ARCHITECTURE.md` §3 / ADR-009
pub struct EventKey { time: SimTime, phase: Phase, source: SourceId, seq: u64 } // total order
```

**Do:**
1. Checked integer-time arithmetic and rational schedules; reject unsupported
   rates (unrepresentable distinct deadlines, unsupported budget) at
   configuration time.
2. Event queue ordered by `EventKey`; a newly created event cannot run in a
   phase already closed at that timestamp — defer to the next eligible
   boundary and record the effective time.
3. Serial executor running phases in ADR-009 order; reset discards the old
   epoch's remaining events.
4. Declared sampling for off-boundary sensor captures (exact evaluation or
   declared interpolation); **reject** profiles the physical model cannot
   support. No pose reuse with a wrong timestamp.
5. Per-subsystem **actual elapsed time**, not a universal dt (review battery
   accounting, prediction, timeouts, mission dwell). Do not change legacy
   battery semantics inside `legacy_stages_v1`.
**Tests:**
- no cumulative drift over long virtual runs for rational rates (e.g. 30 Hz
  against a 50 Hz physics step); overflow checks; stable ties;
- **serial-equivalence gate:** every lane at the legacy rate, one worker ⇒
  outputs equal `legacy_stages_v1` (bit-exact where the old contract was
  bit-exact, otherwise inside the existing envelopes);
- pacing/render rate/recording on-off do not change causal state;
- the mixed 200/30/5 Hz synthetic-source fixture from
  `VERIFICATION_AND_ROLLOUT.md` (scheduler tests, not hardware claims).
**Exit:** P4.10a exit evidence in `DELIVERY_PLAN.md` met; behavior version
label `scheduled_v1` recorded in replay metadata (additive).

### B3 — Fork-join inside phases (P4.10a.2, new)

**Gate:** only for candidates B0's Amdahl table supports (decision rule in `PARALLEL_TICK_PLAN.md` §9).
**Do:**
1. A session-owned `ComputePool { workers }` (dedicated, bounded; **not** a
   global pool). `workers = 1` runs the **same** executor serially.
2. `par_map_ordered` / `ordered_reduce` helpers: results gathered by input
   index and combined in a fixed order. **No unordered floating-point
   reductions.**
3. Audit RNG keying for VIO, localizer/particle filters and observations:
   streams must be keyed by `(platform, tick)` like ADR-006's per-
   `(node, target, tick)` observation streams. Fix shared streams in a
   **separate behavior-versioned** slice first.
4. Apply to candidates in measured-value order (expected: observation pairs,
   per-platform localizer/VIO, footprint rasterization with ordered reduction,
   per-platform controllers/battery). Keep command application, override
   precedence, deconfliction resolution, fusion association, mission
   transitions and snapshot freeze serial.
**Tests (gates):**
- identical committed state and replay at 1, 2 and N workers;
- randomized task delays and reversed completion order (test executor);
- reduction-order mutation test fails;
- tiny-fleet fallback: below a size threshold the serial path is used.
**Exit:** measured tick-latency gain beats run-to-run noise on named hardware,
or the candidate is **not** parallelised and the reason is recorded.

### B4 — Measurement envelopes and estimator timing (P4.10b)

**Do:** timestamped `SensorEnvelope` (source/sequence/epoch, acquisition,
availability and receipt times, frame and calibration identity, uncertainty,
clock quality); per-source bounded delivery queues; fusion uses acquisition
time; elapsed-time estimator propagation; prediction/expiry deadlines so tracks
age with **empty** observation intervals; buffered mapping inputs with
batch-size/max-age triggers; explicit late/drop/duplicate policy; track source
identity and evidence-vs-prediction age.
**Tests:** capture vs delivery preserved; duplicate sequence; source epoch
reset; unsupported delayed profile rejected before run; no lost evidence when
sensing outruns mapping; no-observation expiry.
**Exit:** P4.10b evidence; every dropped item has a recorded reason.

### B5 — Bounded background jobs (P4.10d)

**Do:** bounded job queue and pool; jobs take an **input revision** and return a
proposal tagged with it; cancellation of superseded jobs; **deterministic
application boundaries** (if not ready, simulated time waits while commands stay
serviceable); stale-revision rejection; safety revalidation of any applied
trajectory. Live jobs get qualified deadlines/fallbacks — out of scope for
desktop acceptance.
**Tests:** same causal state across worker counts and shuffled completion;
stale/epoch/frame mismatch rejected; Stop/Pause responsive during pending work.

### B6 — Scheduler checkpoints and recording (P4.10e)

**Do:** checkpoints include schedules, delivery queues, source epochs, pending
jobs and their input revisions; branch publication epochs; recorder
backpressure; viewer-freshness data (per-component as-of time).
**Tests:** restore with delayed deliveries and pending jobs equals uninterrupted
run; old-epoch messages ignored; bounded history.

### B7 — Workload qualification and promotion (P4.10f)

**Do:** define the balanced workload suite; sweep rates (control ≥ legacy,
sensors at modeled rates, mapping/planning lower); publish accuracy-vs-cost
curves; apply the performance standard (≥10% balanced-suite CPU improvement,
five-run same-machine comparison, per-workload no-regression, per-task
deadline reporting); retain the rollback profile.
**Exit:** published accuracy, timing, memory and usability results; no new
safety failures. **Promotion to default is a human decision (D-6).**

---

## 6. WS-C — viewer engine and instrument shell (P2.4–P2.6)

**Specification:** [VIEWER_UI.md](../VIEWER_UI.md) (flat, typography-led
instrument; §4 anatomy, §5 visual system, §12 failure states, §13 access,
§15 tiers). Replay fixtures drive all of it first; no backend invention.
**Prerequisite:** #128 and #129 merged.

### C0 — G2 baselines (shared with WS-E)
See E2. **Must precede** any default-on expensive effect (SSAO, fog, shadows).

### C1 — Foundation: tokens, fonts, status vocabulary (P2.5a)

**Do:**
1. `rust/argusnet-viewer/src/theme.rs` (new): the palette from VIEWER_UI §5.3
   (window `#0A0C0D`, chrome `#101315`, canvas `#0C1012`, rules `#363B3E` /
   `#282E31`, text `#E2E2DF` / `#A5AAAC` / `#687175`, accent `#D2AD62`),
   spacing scale (4 px base; 8/12/16/20/24/32), type roles (§5.2 table), and
   one function applying them to `egui::Visuals` (zero radius, no shadows
   except temporary overlays, 1 px rules).
2. Font loading with a **documented metric-compatible fallback** — never the
   default egui face silently (needs D-2).
3. A status vocabulary enum (`Nominal`, `Unknown`, `Stale`, `Degraded`,
   `Critical`, `Unavailable`, `Pending`) with label + shape/pattern + color so
   color is never the only cue.
4. A `Field<T>` presentation type: `Value(T) | Unavailable(Reason) |
   Stale { value, age }` used by every readout (zero never substitutes missing).
**Tests:** unit tests compute WCAG AA contrast for every text/background token
pair; tokens are the only place colors are defined (grep-based test that `ui.rs`
has no stray `Color32::from_rgb` literals after C2); tabular numeral rendering
check.
**Exit:** merged tokens + slice doc `VIEWER_TOKENS.md`.

### C2 — Shell skeleton (P2.5b)

**C2.0 (mechanical):** split the large `ui.rs` into `shell/`, `inspectors/`,
`timeline/`, `commands/` (VIEWER_UI §11.3). **No behavior change**; headless
render and picking probes must stay green.
**C2.1:** header (64 px), status/freshness line (28 px), mission rail (194 px),
inspector (252 px), timeline (100 px), canvas as the remainder, at the 1440×900
reference; opaque, flat, hairline-separated, asymmetric; responsive rules from
§4.4 (collapse rail first; never hide a critical alert).
**C2.2:** persistent **source indicator** state machine producing exactly the
strings in VIEWER_UI §3 (`LIVE`, `LIVE / HISTORY`, `SIM / RUNNING|PAUSED|FAULTED`,
`SIL`/`HIL`, `REPLAY`, `EVALUATION / TRUTH ENABLED`); it is always visible.
**C2.3:** Operate / Plan / Review workspaces, Perspective / Top-down / Section
projections and Operational / Reconstruction / Safety presets as **three
distinct controls**; selection preserved across all of them. Plan and Review
show explicit "unavailable until <contract>" content, not mock data.
**Tests:** layout test at 1440×900 and at the reduced-width rules; selection
survives workspace/projection/preset changes; source indicator never disappears
when a drawer closes or a platform is selected; keyboard path for every pointer
action; headless screenshot baselines.

### C3 — Typed inspectors (P2.5c)

**Do:** platform inspector with the 11 ordered sections of §7.2 over replay
data, each field a `Field<T>`; legacy estimator confidence/timeout labelled
**legacy** and never mapped to physical integrity; map-cell inspector (§7.3);
event/safety-event inspector (§7.4) with "jump to preceding/following
snapshot"; indexed fleet list as the keyboard equivalent of picking.
**Tests:** fields absent in a replay show *unavailable*; same typed identity
from world pick and fleet list; same-label/different-kind IDs stay distinct;
selection clears on epoch change.

### C4 — Timeline (P2.5d)

**Do:** make the bottom timeline the **only** time-control surface (remove any
duplicate play/pause/step/speed from the canvas toolbar); source-aware time and
mission elapsed, tick/epoch, available history extent (using P2.4a's
`trimmed_before_s` and unavailable ranges), event marks grouped by mission /
safety / command / evidence / fault, recording state, selected-event summary.
Keyboard shortcuts invoke the same actions as the buttons.

### C5 — Projections and section view (P2.5e)

**Do:** Top-down and Section projections with plane/frame, offset, thickness
and clip extent; "clipped is not observed-free" indicator; all projections
resolve one typed selection, full 3D coordinates, revision and source time.
**Tests:** an overhang/underpass fixture stays distinguishable; vertical vs
time separation visible with the timeline.

### C6 — Layer presets and quality tiers (P2.5f)

**Do:** Operational / Reconstruction / Safety presets per §6.2; low / standard /
high tiers with shared semantics (§15). **Invariant test:** in every tier, safety
geometry, unknown-space encoding, selection identity and essential uncertainty
remain present; bloom off by default; fog/SSAO/shadows scale with scene span and
need G2 evidence to be on by default.

### C7 — Failure states, accessibility, preferences (P2.5g)

**Do:** first-class presentations for every state listed in §12 (no source,
loading manifest, verifying world, incompatible world, belief stale/awaiting
base, reconnecting, lag/drops, paused/slow, command pending/busy/rejected/
stale, recording backpressure/incomplete/disk failure, sim fault, TLS/auth/
capability failure, unsupported version), each preserving the last coherent
view and naming what is stale. UI scale independent of world scale, reduced
motion, alert ordering by severity (announced once, not per frame), keybinding
reference with conflict-checked remapping (builds on #129's `bindings.rs`).
**Tests:** one fixture per state; keyboard-only navigation; contrast; reduced
motion.

### C8 — Graphics acceptance (P2.6)

**Do:** fixed replay/world fixtures through the real renderer in all tiers;
perceptual + semantic diff thresholds; pixel probes agree with CPU queries at
observed/unknown borders and tile/LOD seams; resource-growth tests across
repeated seek/reload/source-switch/toggle (entity, material, tile, trail
counts flat); frame budget **on named hardware** (p95 ≤ 16.7 ms, p99 ≤ 25 ms,
no unexplained stall > 100 ms). Lavapipe remains a shader-correctness gate.
**Escalate:** named reference hardware and baseline-image approval (D-5).

---

## 7. WS-D — viewer runtime integration (P5)

**Specification:** [DELIVERY_PLAN.md](DELIVERY_PLAN.md) §7, VIEWER_UI §9 & §11,
ADR-007, ADR-008. The viewer consumes immutable snapshots and submits typed
commands; it never owns runtime state.

### D1 — Worker and command integration (P5.0)

**Depends on:** A1.3 (production constructor), A3 (qualified CLI path), C2 (shell).
**Do:**
1. A `SimWorker` thread owning a `Simulation::production(...)`; the render
   thread never runs a tick.
2. Bounded command channel carrying the **existing** `CommandEnvelope`
   (`argusnet-sim/src/command.rs`); acknowledgements use `CommandAck` /
   `AckStatus` / `RejectReason`. The UI shows `PENDING ACK` until Applied,
   Rejected or Cancelled and never infers success from motion; duplicate
   responses keep the original result with a retry flag; no blind retry of
   non-idempotent commands.
3. Snapshot publication is **latest-wins** (immutable); the event journal is a
   separate ordered, lossless channel (A4).
4. Run, pause/resume, speed, single-step (one control interval including
   intervening sensor events), stop, reset; **expose only completed controls**
   and disable unsupported ones with a reason.
5. World preparation is cancellable and off the render thread; a failed load
   never shows a half-new world; loading a second scenario releases old
   workers/handles/assets only when readers are finished.
**Tests (from the plan):** rapid Run/Stop; repeated reload; queue saturation
(explicit Busy); close-during-load; pause acknowledgement latency during world
loading; no tick dropped under slow render; render rate does not change final
state at a fixed tick.
**Exit:** controls work against the qualified legacy Rust model; P5.0 acceptance.

### D2 — Rust world boot and terrain LOD (P5.1)
**Do:** mesh terrain directly from immutable `argusnet-world` views with
camera-driven LOD, skirts/stitching, deterministic tile bounds, async
scheduling and upload but **atomic world-revision publication**; cache by
content hash + LOD + render-origin transform; byte budgets and eviction; near
terrain never disappears during coarse→fine replacement. `.smartscene` stays
import/export only. **Tests:** synthetic boot with no Python executable;
imported preprocessed world with no exporter installed; sampling parity at LOD
seams.

### D3 — Time, checkpoints, recording (P5.2)
**Do:** bounded snapshot/checkpoint history; deterministic seek = restore
nearest checkpoint + replay logged inputs; a new command after seek creates a
**new epoch/branch** preserving the original; UI shows available-history bounds
and recording backpressure/failure. Replay seek and live-history review never
mutate a runtime and never imply rewinding the external world.
**Tests:** uninterrupted == checkpoint-restore-to-same-tick including safety
escalation, RNG, pending commands, VIO, mapping; rate changes do not change
final deterministic state (relies on A2, A4).

### D4 — World-aware live protocol (P5.3)
**Do:** implement ADR-008 additively: capability negotiation, world manifest
and chunks, revisioned frames and belief patches, resync, bounded retention;
store mission zones/status at the server authority (remove the hardcoded empty
zone response). **Follow rule 2 of §0.1 exactly** (Python bindings + `argusnet-
proto` + `argusnet-core` + `argusnet-server` + viewer together; run G1 checks).
**Tests:** old V2/new client combinations; reordered/missing chunks; corrupt
hashes; restarts; replayed sequences; base-revision gaps; slow readers;
non-loopback TLS. No remote control is added.

### D5 — Remove the subprocess/runtime export path (P5.4)
**Do:** delete `simulation_python`, `spawn_simulation_child`,
`spawn_build_scene_child`; remove Python daemon auto-build from library
imports; server-selected port 0 plus readiness handshake where local daemon
launch remains for standalone clients. **Gate:** only after D1 and D2 pass.
**Test:** viewer startup invokes no Python, Cargo or GLB export (process-tree
assertion).

### D6 — Interactive acceptance (P5.5)
**Do:** from a clean installed build, press Run on the designated performance
scenario; verify pause/step/seek, recording, overlays, picking, camera modes,
restart under load; no Python descendant of the viewer; record render CPU/GPU
timing separately from sim ticks and catch-up lag; memory stability and
graceful cancellation. **Escalate:** reference hardware (D-5).

---

## 8. WS-E — guardrails

### E1 — G0 semantic contract inventory
**Do:** extend `scripts/check_field_parity.py` coverage from fields to **enum
semantics and unsupported schema versions**; add installed-wheel and
clean-checkout contract jobs (`scripts/check_installed_contract.py` is the
precedent). **Exit:** every payload is preserved, normalized explicitly, or
rejected — no silent missing-field success. **Must precede the first wire
change in WS-D4/WS-F.**

### E2 — G2 renderer fixtures and frame instrumentation
**Do:** fixtures for overview, close view, low pitch, split mode, high entity
count; warm-up-aware CPU/GPU frame instrumentation; split software smoke from
hardware timing; record hardware/profile/viewport with each artifact.
**Exit:** baseline artifacts under `docs/performance/`. **Gates** turning
expensive effects on by default.

### E3 — G3 scanners and provenance
**Do:** triage `pip-audit`, `bandit`, `cargo-audit` findings (each resolved or
documented with owner and expiry), keep pins, add artifact/provenance checks;
enable required checks **one at a time**, never blanket-ignore.

---

## 9. WS-F — product features and client contracts

**Rule:** each feature ships as a vertical slice — domain result, recorded
contract, viewer workflow, acceptance scenario together — and the viewer shows
*unavailable* rather than fabricating missing backend state.
([CAPABILITY_DELIVERY.md](../CAPABILITY_DELIVERY.md),
[VIEWER_FEATURE_INTEGRATION.md](../VIEWER_FEATURE_INTEGRATION.md).)

### S1 — Canonical contracts and client-output skeleton
**Do:** versioned schemas (alongside `docs/replay-schema.json`) for site,
mission, platform/sensor, localization, spatial revision, trajectory, safety,
evidence, reconstruction, finding, change and report records, with explicit
unavailable fields; one replay-derived client package. **Design these before
building Plan/Review UI.** Ids and revision keys follow VIEWER_UI §10.2
(`session_id / run_id / epoch / tick / world_revision / belief_revision /
sequence`).

### F1 — Mission outcome scorecards
**Start:** `src/argusnet/evaluation/scorecard.py`, `reports.py`.
**Do:** objective outcomes, evidence sufficiency, energy, violations,
timeout/degradation, completion/failure **reasons**; unmatched/missing metrics
are *unavailable*, never success or zero error; every result links to event and
evidence IDs. **Viewer:** Review summary, Reporting export.
**Tests:** completed, aborted, incomplete runs each yield attributable reasons.

### F2 — Sensor and link failure scenarios
**Start:** existing noise/health/scenario helpers (`evaluation/scenarios.py`,
`world/weather.py`, sensing health).
**Do:** manifest-recorded scheduled outages, delay/reordering, calibration
drift, degraded positioning, recovery; deterministic for a fixed manifest
(seed, schedule, versions, policy). Needs B4 for delivery timing realism.
**Tests:** hold/return/abort/recovery within bounded deadlines; no terrain
penetration; predeclared detection deadline, allowed transient, containment
bound and expected terminal state per case (not "no crash").

### F3 — Operator explanations
**Do:** mission/planning/safety emit **typed decision records** (candidate and
selected IDs, reason codes, scores/constraints, input revisions, fallback,
resulting command), journaled (A4) and independent of display-frame drops.
**Viewer:** inspectors answer "why this task/route" with next replanning
trigger. **Tests:** trace task → route → trajectory → safety → applied command;
pending/rejected/cancelled differ from executed.

### F4 — Scenario comparisons
**Start:** `evaluation/benchmarks.py`.
**Do:** paired seed sets across configs/policies; show dispersion, failed-run
count, sample size, metric definitions and runtime cost; identities of
scenario/config/world/algorithm recorded.

### F5 — Battery-aware replanning
**Depends:** A (validated execution), B4 (elapsed-time accounting), D-3.
**Do:** continuous route-to-home reserve feasibility, hysteresis against task
thrashing, handoff, unassigned-task reporting; no team assignment weakens a
per-platform reserve; **no UI-computed reserve authority**.

### F6 — Persistent spatial memory (design first)
**Depends:** one mutable belief owner (A), durable immutable tile/evidence store.
**Do:** ADR for site/map store, stable CRS, content identity, alignment
uncertainty and aging policy; then implement. A stale prior stays
distinguishable from new observation; alignment uncertainty can block a change
claim.

### F7 — Adaptive inspection (single-mission slice first)
**Depends:** inspection quality evaluator + evidence manifests + F5.
**Do:** requested-vs-achieved evidence quality; occluded or insufficient-angle
evidence cannot complete an inspection; safe revisit or an explicit unmet
requirement; energy/time/retry bounds prevent endless revisits.

### S2 — Three-dimensional belief world
**Do (design before code):** an additive spatial-chunk contract (3D bounds,
frame, resolution/LOD, revision, time, content hash, provenance) for
free/occupied/unknown volumes, clearance fields, evidence anchors. Legacy
surface-only sources show volumetric state as **unavailable** — never infer
free airspace above a terrain raster. Planning, safety and presentation resolve
the **same** spatial revision. Needs a D-7 decision on volumetric encoding.

---

## 10. WS-G — Python reduction (P6)

Only after D5/D6 and installed-package acceptance.
- **P6.0** caller inventory: script with `scripts/module_reachability.py` and
  `scripts/module_symbols.py` per module (RUNTIME / test-only / ORPHAN);
  deletion map with evidence; resolve duplicate trajectory builders by behavior
  name and test.
- **P6.1** separate lightweight config/replay readers from the native bridge so
  analysis installs and imports without building Rust.
- **P6.2** Rust owns the production `argusnet` executable (the Cargo binary is
  `argusnet-rs` until then); Python provides `argusnet-analysis`; command
  migration table for every old subcommand/flag/exit code; wrappers must not
  build the binary or search hardcoded interpreter paths.
- **P6.3** delete migrated modules as their last runtime callers vanish;
  relocate exporter-only GLB/GIS first; shrink mypy ignore lists; rewrite
  README, usage, architecture, STATE_OWNERSHIP, KNOWN_GAPS, performance docs.
- **P6.4** acceptance: Rust-only runtime install, Python-only analysis
  install, legacy replay/scene import, glTF export in a conformant consumer,
  upgrade from the compatibility release. Retain legacy execution through
  scheduled qualification and the following release.

---

## 11. WS-H — hardware track (S3–S8): contract work only

The agent **does not** claim SIL/HIL/aircraft results. Permitted work:
1. A **hardware-profile template** document (autopilot, board, transport,
   sensor rates, OS, deadlines, acceptance bounds, clock-conversion policy,
   watchdog response). Leave choices blank until D-4.
2. A **device-adapter trait** and recorded-packet **contract fixtures**
   (frames/units/time conversion; invalid and unsupported versions fail
   explicitly), clearly labelled *contract fixtures, not SIL*.
3. The **fault matrix** as data (HARDWARE/AUTONOMOUS docs list: deadline/crash,
   propulsion, actuators, battery, sensor dropout/freeze/bias, GNSS, VIO
   divergence, clock error, CPU overload, watchdog, storage, RF delay/loss,
   stale/replayed commands, wrong world revision, severe weather, multi-
   platform conflict), each with predeclared detection deadline, allowed
   transient, containment bound, fallback, terminal state, evidence.
4. Viewer read-only supervision surfaces (source identity, freshness, lost-link
   last-known state) against those fixtures.
Everything else (actual SIL/HIL, contained tests, field envelopes, fleet, ML)
waits on human decisions and physical evidence.

---

## 12. Cross-cutting checklists and commands

### 12.1 Commands

```bash
scripts/ci-local.sh --quick              # python, rust-fast, rust-lint
scripts/ci-local.sh rust-viewer          # viewer tests + headless render smoke
cargo test --workspace --exclude argusnet-viewer
cargo test -p argusnet-viewer
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
ARGUSNET_GPU_TESTS=1 cargo test -p argusnet-viewer --test belief_probes   -- --test-threads=1
ARGUSNET_GPU_TESTS=1 cargo test -p argusnet-viewer --test geometry_probes -- --test-threads=1
ARGUSNET_GPU_TESTS=1 cargo test -p argusnet-viewer --test picking_probes  -- --test-threads=1
python3 -m pytest tests/ -q --timeout=120
python3 scripts/regenerate_proto.py --check && python3 scripts/check_field_parity.py
python3 scripts/run_phase_acceptance.py  # release build of argusnet-rs; see PHASE_ACCEPTANCE.md
```

Docs-only slices: inspect the diff, check local links/anchors, confirm example
commands against current parsers, and **do not report runtime tests as passed
because no source changed** (`LOCAL_CI.md`).

### 12.2 Wire/schema change checklist
1. G1/G0 checks green before you start.
2. Edit `.proto`; run `python3 scripts/regenerate_proto.py`.
3. Update `argusnet-proto`, `argusnet-core`, `argusnet-server`, Python adapter,
   viewer conversion, and add a sample payload + migration example.
4. Update replay schema/`FIELD_PARITY.md` disposition for new fields
   (additive).
5. Compatibility test: old client ↔ new server and the reverse.
6. `regenerate_proto.py --check` and field-parity in CI.

### 12.3 "Behavior-neutral" proof
Capture before/after: replay hash, `StageTrace` hash, checkpoint bytes (where
unchanged by contract), and the 7-scenario acceptance table. All identical ⇒
neutral. Any difference ⇒ it is a behavior change: separate PR, version label,
rebaseline commit, rollback boundary.

### 12.4 Performance measurement protocol
Release build, quiesced machine, **≥5 runs**, same machine for baseline and
candidate, record hardware/OS/commit/config, report p50/p95/p99 and worst
observed, keep failed runs. Report CPU presentation, GPU, sim step/task,
transport/native, world-build, shader warmup and recording backlog
**separately**. Historical percentages (the Python-era 40% localization figure)
are not current priorities.

### 12.5 Documentation convergence checklist (every slice)
`STATE_OWNERSHIP.md`, `KNOWN_GAPS.md`, `docs/changelog.md`, the slice doc, the
`STATUS.md` row (only with verification), ADR if a decision was made.

---

## 13. Risks

| Risk | Mitigation |
| --- | --- |
| A1 ownership refactor perturbs behavior | Trace/replay hash equality per link; stage-checkpoint bytes unchanged; one link per PR |
| Scope creep from "finish everything" | Waves + gates; one package per PR; unavailable states instead of mocks |
| Parallelism changes results | Single executor path; ordered merge; worker-count and completion-order invariance as merge gates; RNG audit before B3 |
| Parallelism gives no gain on small fleets | B0 Amdahl table decides; serial fallback; "not parallelised" is an acceptable outcome |
| Viewer redesign drifts from spec | Tokens as the only color source; invariants tests per tier; replay-fixture baselines; no card/glass/neon |
| Baselines rebaselined to hide regressions | Rebaseline only in dedicated commits; reviewer sees a pure diff |
| Python-era numbers misguide priorities | B0 profile replaces them before any optimisation |
| Hardware claims leak into docs | §0.1 rule 9; labels "contract fixture" vs "SIL/HIL"; Hardware track limited to §11 |
| Stale status docs mislead agents | Wave 0.3; every status change cites a command or commit |
| CI minutes exhausted look like repo breakage | `LOCAL_CI.md`: jobs failing within seconds with no steps are a runner problem, use `ci-local.sh` |
| Wire change breaks old clients | §12.2; compat tests both directions |
| Long-run resource growth in viewer | C8 resource-growth tests across reload/seek/toggle |

---

## 14. Decision register (human-owned)

The agent prepares an ADR stub with options and a recommendation for each; it
does not decide. **2026-10-03:** the maintainer asked the agent to proceed on
its recommendation where a question arises; decisions taken that way are marked
*decided on recommendation* and remain open to reversal.

| ID | Decision | Needed by | Options | Recommendation |
| --- | --- | --- | --- | --- |
| D-1 | `sar_person_search` `covariance_reduction_mean`: the envelope fails because of the metric's definition | A3.1 | (a) redefine the metric to measure what the scenario intends; (b) replace with a different metric; (c) treat as a real behavior defect and fix | Read `SAFETY_AND_PROFILES.md` analysis; prefer (a) or (b) with a rebaseline commit. Do **not** widen the tolerance. **Decided (maintainer): (a)**, covariance at last update with ≥5 qualifying tracks (#137) |
| D-2 | Typeface and licence (spec names Helvetica / Neue Haas Grotesk, both commercial) | C1 | Licence the named face; or bundle an open metric-compatible face (candidates to evaluate and licence-check: Arimo/Liberation Sans/Nimbus-class) | Open, metric-compatible default now; swap if a licence is bought. Verify each licence before bundling. **Decided (maintainer): Arimo, OFL 1.1** (#142) |
| D-3 | Hover energy: the reference charges nothing for hovering (documented in `TICK_STAGES.md`) | F5 | Keep as legacy profile and add a physical-profile hover term; or change globally | New versioned physical profile; leave legacy untouched. **Decided (maintainer, 2026-10-05): as recommended.** `EnergyProfile::PhysicalV1` (`argusnet-rs sim --energy-profile physical_v1`) charges the legacy motion energy plus hover power for the part of each tick not spent moving; `Legacy` stays the default and unchanged. Default promotion waits on F5 qualifying with it |
| D-4 | First supported autopilot / vehicle class and hardware profile | WS-H beyond §11 | — | Not for the agent to choose |
| D-5 | Named reference hardware for frame budgets and baseline-image approval | C8, D6 | — | Human supplies; record in artifacts. **Decided (maintainer, 2026-10-20): the maintainer's Apple M5 Pro (18 cores, 64 GB, macOS, Metal).** First budget recorded on it: [r4-frame-budget-2026-10-21](../performance/r4-frame-budget-2026-10-21.md), worst p95 6.3 ms. Baseline-image approval is still open |
| D-6 | Promote `scheduled_v1` to default | B7 | Promote / keep opt-in another release | Only after B7 evidence; keep rollback profile |
| D-7 | Volumetric belief encoding and chunk format | S2 | Sparse voxel / octree / other | ADR with a cost comparison on a named workload |
| D-8 | Merge strategy and who merges | all | — | Humans merge; agents prepare stacked PRs. **Decided (maintainer): the agent merges any PR green on local CI**, with the evidence posted on the PR first, while GitHub runners are unavailable |
| D-9 | macOS runner availability for cross-platform determinism | A3.3 | Add runner / scheduled external job / defer | Escalate; do not mark the gate passed without it |
| D-10 | Acceptance reference: the Python reference's draws are platform-dependent, so a reference recomputed per machine flips `sar_person_search`'s RMSE envelope by platform | A3.1 | Fix one reference platform; or record the reference once and score Rust against the record | **Decided on recommendation: stored** (`docs/performance/p49c-reference.json`, tagged with its platform and the reference code's hash; `--reference live --record-reference` re-records) |
| D-12 | `covariance_reduction_mean`'s envelope: over 30 and 10 seeds its standard error (0.07–0.21) is of the order of the 0.10 envelope, with heavy tails on both sides, so it fails without a consistent direction (A3.2 report) | A3.2 | Judge it by a robust statistic (median / trimmed mean) declared before the next run; pre-declare a larger seed count; or size the envelope from its measured variance | **Decided by the project lead (2026-10-04): the cross-seed median, envelope kept at 0.10**, declared in code and in PHASE_ACCEPTANCE.md before any run on a fresh 30-seed confirmation set (`--seed-set confirmation`) that judges it; the result is reported as it is |
| D-13 | C6 quality tiers: shadows and distance fog were on by default before the rule that an expensive effect needs G2 baselines to be on by default, and G2 (E2) is still open | C6 | Turn them off by default until G2; or keep them on in the standard tier, pending G2 | **Decided by the project lead (2026-10-04): kept on in standard, pending G2**, since turning them off would change the default look twice; bloom, never baselined, is off by default (high tier only). E2 decides whether they stay on. E2's software baseline (2026-10-04: Standard 2.2–3.0× Low on llvmpipe) is relative CPU-rasteriser cost, not hardware evidence, so the decision stands until the fixtures run on reference hardware (D-5). The belief overlay no longer depends on any of them (composited after lighting and fog; probes run in every tier) |
| D-14 | `sensor_type`: the Rust observation model treats thermal, lwir, mwir and ir as a thermal band and anything else as optical; the replay schema allowed only optical and radar; configs took any string; Python's thermal sensing already writes "thermal" | E1.2 | Constrain configs to the schema's two values; or widen the schema to what the runtime models and refuse everything else | **Decided by the project lead (2026-10-04): one vocabulary, optical, radar, thermal, lwir, mwir, ir** (`SENSOR_TYPES` in Python and `argusnet-types`), the schema's enum is exactly it, Python configs and Rust bundles refuse anything else |
| D-11 | Stage 5 (VIO) in the production constructor: production does not assemble it | A1.3 | Require it and assemble VIO; install a declared-absent handler; or leave it out of the required set | **Decided on recommendation: not required** (`Stage::PRODUCTION_CORE`); a run that assembles VIO requires it explicitly |

---

## 15. Definition of done for the whole target

All of the following, each with evidence in a slice doc:

1. **S0:** a nonempty representative mission is owned by Rust; direct fusion;
   immutable snapshots; reset equals fresh; restore equals uninterrupted;
   durable journal; no Python callback in a tick; 7/7 canonical scenarios
   within declared envelopes; 30-seed + held-out qualification published.
2. **P4.10:** `scheduled_v1` qualified (B7) with worker-count and completion-
   order invariance, pending-job restore, and published accuracy/timing/memory
   results; the promotion decision made by a human; rollback profile retained.
3. **Viewer:** the Operate/Plan/Review shell, typed inspectors, single timeline,
   three projections, presets, tiers, failure states and accessibility meet
   VIEWER_UI §16; P2.6 graphics acceptance passed on named hardware.
4. **P5:** worker + acknowledged commands, Rust world boot with LOD, seek/branch,
   world-aware live protocol, Python subprocess path removed, interactive
   acceptance from a clean installed build with no Python descendant.
5. **Guardrails:** G0, G2, G3 closed.
6. **Features:** F1–F4 replay/evaluation slices shipped; F5, F7 (single
   mission) with the validated pipeline; F6 and S2 at least at accepted-ADR
   plus first implemented slice; each shows unavailable states honestly.
7. **P6:** Python reduced to analysis, packaging acceptance passed, docs
   rewritten against the shipped system, status documents current.
8. **Hardware:** contract artifacts of §11 exist; **no** SIL/HIL/aircraft claim
   is made until separate qualification evidence exists.

A merged port is not an installed release. Archive migration checklists only
after the release contains the work.
