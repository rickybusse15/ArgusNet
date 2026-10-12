# Capability delivery and viewer integration

Status (2026-10-12): F1, F3 and F4 have shipped as replay and mission slices, and
F2 has its first slice; F5–F7 are planned (see [migration status](migration/STATUS.md)
for each slice's evidence). For the planned ones, existing modules are starting
points, not evidence that these end-to-end features ship.
This document translates the [roadmap](ROADMAP.md) and
[source review](NEXT_REFACTOR_REVIEW.md) into feature acceptance criteria.
Migration P2/P4/P5/P6 identifiers retain their existing meaning; F1–F7 below are
feature identifiers, not new migration phases.

## Delivery order

Start F1 scorecards, F2 failure scenarios and F3 explanations against the current
replay/evaluation path while repairing runtime foundations. Add F4 comparisons
once reports and scenario identities are stable. Integrate F5 energy contingencies
with the validated Rust execution pipeline. F6 persistent memory and F7 adaptive
inspection follow revisioned belief ownership and evidence contracts. These are
vertical slices: domain result, recorded contract, viewer workflow and acceptance
scenario ship together. Prototype UI with explicit fixtures; never fabricate
missing backend state.

## Feature contracts and acceptance

| ID / feature | Existing foundation and change | Owner / prerequisites | Viewer surface | Exit evidence |
| --- | --- | --- | --- | --- |
| F1 Mission outcome scorecards | Extend evaluation/reports beyond summary metrics with objective outcomes, evidence sufficiency, energy, violations, timeout/degradation and completion/failure reasons | Offline evaluation consumes replay + ordered events; initial replay slice can precede P4.8, full causal report needs typed mission events | Review summary and Reporting; Operate shows current progress | Completed, aborted and incomplete runs produce attributable reasons; missing/unmatched metrics are unavailable, never success or zero error; every result links to supporting events/evidence |
| F2 Sensor and link failure scenarios | Extend existing noise/health/scenario helpers with scheduled outages, delay/reordering, calibration drift, degraded positioning and recovery | Scenario/observation-source owner; mission/safety own responses; manifest records seed, schedule, versions and policy | Scenario Studio configuration, Operate degraded state, Review timeline | Repeatable injection and delivered measurements for a fixed manifest; hold/return/abort/recovery behavior checked with bounded deadlines and no terrain penetration |
| F3 Operator explanations | Extend selection and event views with task-selection reason, route constraints, uncertainty and replanning trigger | Mission/planning/safety emit typed decisions with input revisions, candidate/selected IDs, reason codes, relevant scores/constraints, fallback and resulting command | Operate platform/map/event inspectors; Plan proposal review; Review causal timeline | Operator can trace task → route → trajectory → safety → applied command; pending/rejected/cancelled differ from executed; skipped display frames do not lose explanations |
| F4 Scenario comparisons | Extend benchmark aggregation to paired comparisons across seeds, sensor configurations and planning policies | Evaluation runner consumes immutable scenario/config/world/algorithm identities and F1 reports; F2 adds stress cases | Scenario Studio setup, Review comparisons, Reporting export | Same seed set and workload per variant; show dispersion, failed runs, sample count, metric definitions and runtime cost; changes in algorithms or worlds remain visible |
| F5 Battery-aware replanning | Extend existing battery, energy-aware POI assignment and handoff with continuous route-to-home reserve feasibility and task reassignment | Planning/battery + mission execution + safety; P4.7/P4.8 integration, belief/localization revisions and declared reserve policy | Operate reserve/contingency inspector; Plan feasibility; Review handoff reasons | Increasing return cost or reducing battery triggers safe replan/return; unfinishable tasks transfer or remain explicitly unassigned; hysteresis prevents task thrashing; no team assignment weakens per-platform reserve |
| F6 Persistent spatial memory | Extend indexing foundations into reusable site maps, evidence and mission history with versioned alignment and change candidates | One mutable belief owner; durable immutable tile/evidence store and site/spatial index; stable CRS, content identity, provenance and aging policy | Site catalog, Plan prior selection, Review synchronized prior/current comparison | Two missions reload the same site; known scene changes are attributable; stale priors remain distinguishable from new observations; alignment uncertainty can prevent a change claim; restart preserves references |
| F7 Adaptive inspection | Extend POI dwell/coverage completion into requested-versus-achieved evidence quality and safe revisits for missing viewpoints or uncertainty | Inspection quality evaluator + evidence manifests + task pipeline + F5; F6 enables cross-mission revisits, but is not required for a single-mission slice | Operate unmet criteria and revisit reason; Plan viewpoints; Review evidence acceptance | Occluded/insufficient-angle evidence cannot complete inspection; safe useful revisit is selected or an unmet requirement is reported; energy/time/retry bounds prevent endless revisits |

## Scheduling and delivery amendment (2026-09-23)

All F1–F7 features use [the Rust runtime contracts](RUST_RUNTIME_ARCHITECTURE.md).
F1/F4 compare elapsed source time and actual work across execution profiles;
F2 preserves acquisition/delivery timing and supported clock/failure models;
F3 reads durable decisions even when display snapshots are replaced. F5 runs
reserve/availability checks without waiting for visualization; F6/F7 consume
versioned evidence and bounded reconstruction jobs. No feature introduces a
second map/mission writer or reads hidden truth to improve its score.

Ship P5.0 basic controls alongside P4.10, then expose freshness, delayed work,
recording gaps and unavailable timing capabilities in Operate/Review/Diagnostics.
SIL/HIL/live feature promotion additionally needs [hardware evidence](HARDWARE_INTEGRATION.md).

## Architecture that supports the features

1. Truth belongs to synthetic sensing, physical collision and evaluation through
   explicit capabilities. Mapping/localization/planning/safety decisions use
   declared prior, observed and estimated inputs. Collision ground truth does
   not grant the planner truth access. Physical-mode tests hold measurements
   fixed while changing hidden truth.
2. Mission intent becomes a candidate task, route/viewpoint, trajectory, safety
   decision, then executable command. The mission executive owns transitions;
   domain planners propose actions; the viewer submits intent and never executes
   a route directly. Preserve legacy behavior only under its named profile.
3. A single belief-map owner publishes immutable revisioned views and dirty tile
   patches to planners, localization and presentation. Persistent history stores
   those revisions; it is not a second mutable mission map.
4. Replaceable display snapshots and a durable ordered decision/event journal
   have separate loss policies. Recording reports backpressure or incomplete
   output explicitly. All F1/F3/F7 explanations refer to event and evidence IDs.
5. Fusion has no need for scoring truth. Evaluation matches unlabeled tracks to
   truth separately and reports unmatched objects, false tracks, identity changes,
   error and uncertainty calibration. Correlated localization estimates and
   timeouts must not manufacture confidence in the physical profile.

## Viewer placement in the system

[VIEWER_UI.md](VIEWER_UI.md) is the target presentation and interaction
specification. Its Operate / Plan / Review shell supersedes the older list of
peer workspaces; Diagnostics, Scenario Studio, Reporting and Site catalog are
secondary destinations. Its visual language, accessibility requirements, quality
tiers and budgets apply across these features. It is not proof those screens exist.

| Viewer specification work | Migration dependency | Product integration |
| --- | --- | --- |
| Tokens, fonts, shell, inspectors, replay timeline, picking, camera, fixed visual fixtures | P2.0–P2.6; replay fixtures allow early delivery | F1/F3 review prototypes, F4 comparison surfaces |
| Belief rendering and coherent revision install | P2.1 + P4.6 belief owner; P5.1 world boot | F5 feasibility and F6 map history; unknown/stale/unavailable remain distinct |
| Simulation worker, acknowledged controls, pause/step/seek/record | Repaired P4.1 commands/checkpoints + P4.8/P4.9 + P5.0/P5.2 | Interactive mission supervision; no Python/GLB boot after P5.4 |
| World-aware live sources and actual mission metadata | P5.3 + ADR-008 | Same Operate/Review semantics with explicit live-history context; read-only live by default |
| Plan authoring, evidence Review, persistent Site catalog | Typed mission/safety decisions + F5/F6/F7 contracts | Validated proposals, attributable inspection completion, prior/current comparison |

Rendered state remains derived. The viewer owns camera, selection, layout and
filters only. Checkpoints are private causal runtime state, not UI DTOs. Keep
f64 spatial interchange and one camera-relative f32 rendering conversion.

## Release criteria

Each feature must name its supported execution modes, unavailable fields,
contract versions, scenario fixture, owning subsystem and operator workflow.
Use the [migration verification gates](migration/VERIFICATION_AND_ROLLOUT.md)
for compatibility, truth isolation, restore, stream recovery and performance.
Physical semantics get new algorithm/profile versions rather than silently
rewriting parity fixtures. Choose numeric quality, energy and timing thresholds
per recorded scenario/reference hardware before evaluating the implementation.

## Expanded 3D/autonomous viewer scope

The supplied [autonomous system plan](AUTONOMOUS_SYSTEM_PLAN.md) adds S0–S8
system gates and client products. [VIEWER_FEATURE_INTEGRATION.md](VIEWER_FEATURE_INTEGRATION.md)
is the feature-level UI handoff: F1–F7 plus volumetric belief, reconstruction,
localization integrity, onboard health/contingencies, fleet reservations, SIL/HIL,
ML provenance, offline synchronization and governance. It specifies selections,
visible fields, backend owners, missing-data behavior and acceptance scenarios.

F1 expands into mission assurance including landing/terminal outcome; F2 expands
into qualification cases with explicit detection/containment bounds; F3 explains
onboard autonomy and recovery; F4 includes frozen SIL/HIL build identities; F5
shows reserve-valid alternatives; F6 compares aligned 3D revisions; F7 judges
inspection through evidence. These extensions retain existing feature IDs.
