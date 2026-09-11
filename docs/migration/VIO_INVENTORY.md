# P4.4 inventory: what in the localization package is actually wired

**Status:** measured 2026-09-10, before any VIO code moves.

P4.4 opens with an instruction rather than a task: *"Inventory which helper
modules are actually wired before porting stubs."* This is that inventory, and
it changes the size of the job by more than half.

## Method

Reachability by **transitive import traversal** from the four
`[project.scripts]` entry points — `argusnet.cli.main`,
`argusnet.simulation.sim`, `argusnet.evaluation.export`,
`argusnet.cli.scene` — following the import graph rather than grepping for
names.

That distinction mattered. A first pass grepped `sim.py` for each module name
and reported 31 hits for `state`, suggesting it was heavily used. Those hits
were local variables — `node_state`, `step_localization_state` — and `sim.py`
never imports `localization.state` at all. The `LocalizationState` type it does
use is defined in `core/types.py:114`. Counting references is not the same as
following imports, and here the two disagree by 759 lines.

## Result

| module | lines | reached from the runtime? |
|---|---|---|
| `engine.py` | 216 | **yes** — `sim.py` (already ported, #63/#67) |
| `vio.py` | 521 | **yes** — `sim.py` imports `EKFVIO`, `VisualFeature` |
| `transforms.py` | 120 | **yes** — via `vio.py` |
| `state.py` | 759 | no — tests only (see below) |
| `relocalization.py` | 148 | no |
| `pose_graph.py` | 144 | no |
| `gnss_fusion.py` | 142 | no |
| `odometry.py` | 122 | no |
| `loop_closure.py` | 115 | no |

**Live runtime surface: 857 lines. Unreachable: 1,430.**

### Three findings worth stating separately

**`state.py` is test-only, not dead.** Its sole importer in `src/` is
`mapping/fusion.py`, a re-export shim that nothing imports. But three test
files import it directly. So it is exercised, just never by the running
simulation. That is a different category from the rest and deserves a
different decision.

**Five modules are orphaned outright** — no runtime importer *and* no test
importer: `relocalization`, `pose_graph`, `gnss_fusion`, `loop_closure`,
`odometry`, 671 lines between them.

**`odometry` appears in `sim.py` exactly once, inside a comment**
("ground-station nodes don't need odometry"). Its only importer anywhere is
`gnss_fusion.py`, which itself has none. It is reachable only through code
that is itself unreachable.

**`indexing/` (800 lines) is reachable only from `loop_closure`**, and
`loop_closure` imports `indexing` while `indexing`'s only importer is
`loop_closure`. A closed cycle with no entry point from anywhere else.

## What this means for P4.4

The port surface is **`vio.py` + `transforms.py`, 641 lines** — not the 2,292
in the package. Porting the rest would be roughly 1,430 lines of work
producing nothing the simulation can call, which is precisely the outcome
P4.4's opening instruction exists to prevent.

P4.4 also says *"no manufactured confidence on a failed estimator"* and
*"define invalid or stale pose behavior explicitly"*. Both apply to `vio.py`
and are the substance of the port; neither depends on the orphaned modules.

## What this does **not** claim

Not that the unreachable modules should be deleted. Phase 6 owns that
decision, and it needs more than reachability:

- `state.py`'s triangulation functions have real tests. Deleting it deletes
  those, and someone should decide whether the capability is wanted before the
  code goes.
- The five orphans may be partial implementations of a planned capability
  rather than abandoned ones. This inventory says nothing about intent.

What it does establish is that none of them is load-bearing today, so none of
them blocks or belongs in the VIO port.

## Reproducing this

The traversal is a dozen lines of `ast` walking from the entry points, and it
is worth re-running rather than trusting this table after the package changes.
The check that matters is the one that caught the `state.py` error: compare
what the import graph reaches against what a name-grep suggests, and when they
disagree, believe the graph.
