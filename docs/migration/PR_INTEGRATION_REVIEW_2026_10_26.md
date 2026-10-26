# Open PR integration review — 2026-10-26

Base: `8b62bfe`. Integration PR: #305. Review examined changed source, declared
behavior changes, tests, dependencies and remote check results. Original heads
are preserved by merge commits; the integration must also use a merge commit.

| PR | Scope | Disposition |
| --- | --- | --- |
| #303 | Sweep physical steps against structures | Integrated; preserves the declared no-prior/non-reference behavior change |
| #293 | Transport, contained paths, bounded parsing | Integrated; fixed the formatting failure that stopped Python CI |
| #279 | Unified real-world import and canonical site | Integrated; new import readers also retain #293's size caps |
| #282 → #283 → #284 | Record contracts, source-bound Review, status | Integrated in order; canonical JSON and explicit unavailable values retained |
| #285 → #286 | Energy contingencies and flown energy | Integrated in order; profiles remain opt-in |
| #287 | Calibration drift and replay fault metadata | Integrated; deterministic schedule semantics retained |
| #288 | Learned-wire replay and display | Integrated; learned and true geometry remain distinct |
| #289 | Mission decision inspector | Integrated; fixed interaction with #297 |
| #290 | Energy reserve display | Integrated; fixed interaction with #286 |
| #292 | Cloud setup documentation | Integrated; environment-specific paths remain labeled |
| #294 | Belief/truth switch and scale | Integrated; live truth remains unavailable |
| #295 | Timing diagnostics | Integrated; measured speed remains separate from requested speed |
| #296 | Allocation display | Integrated; draws only recorded assignments/positions |
| #297 | Branch simulation history | Integrated; bounded checkpoint count is not a byte budget |
| #298 | Rust cache action update | Integrated; action remains pinned by commit |
| #302 | Quality requirements and deficit mapping | Integrated; unmet/unreachable cells remain distinct from met cells |
| #301 → #304 | Navigation camera, rig and moving pod | Integrated in order; assumed profiles and unmodeled sensor outputs remain explicit |
| #299, #300 | egui_plot / egui_extras minor upgrades | Deferred: both fail viewer compilation with incompatible egui types |

## Findings fixed in integration

1. **Abandoned decisions after history branching.** #289 kept a decision log
   independently of frames, while #297 truncated only frames. Branching now
   discards decisions after the last retained frame, including all decisions
   when no frame is retained. A regression test covers both cases.
2. **Reserve ledger versus committed battery.** #290 sampled stage-one energy,
   while #286 charges actual flown motion at commit. `flown_v1` now publishes
   remaining energy and reserve from the committed position; a regression test
   compares every mobile node's record with its committed battery fraction.
3. **New importer bypassed bounded reads.** #279's source reads/hashes predated
   #293. The integrated import and Overpass entrypoints now use capped reads and
   chunked file hashing.
4. **Projection fixture platform difference.** PROJ 9.8.1 differs from the
   committed footprint coordinates by at most 0.676 nanometres, and its area-error
   diagnostic by 5.94e-11. The fixture permits 1e-9 m only on projected coordinates
   and 1e-10 on that diagnostic; tags, heights, identity and topology stay exact.

## Deferred dependency updates

Both helper upgrades introduce egui 0.37 types into a viewer whose
`bevy_egui 0.43.1` requires egui 0.36. Remote viewer and lint jobs fail with
type mismatches. The available bridge releases through 0.43.1 all require 0.36;
merging these independently would break the viewer. Keep #299/#300 open until a
coordinated compatible bridge upgrade can be qualified. Other passing checks on
those PRs do not override the compiler failures.

## Next delivered workflow

`scripts/real_site_inspection.py` implements the next recommended outcome:
canonical site → self-built simulation → verified client package → Review scene,
with repeated seeds and recorded outcomes. See [the workflow guide](../REAL_SITE_INSPECTION.md).
Mission quality thresholds, truth isolation, scheduler completion and hardware
qualification are not implied by this integration.
