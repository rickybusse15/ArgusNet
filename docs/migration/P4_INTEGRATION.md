# P4 integration tree

**Status: the five P4 stacks compose. Verified, not assumed.**

## Why this branch exists

P4 has been delivered as five independent stacks, each branched from `main` at
the P4.6a merge (`0518204`) and none merged since:

| Stack | Head branch | Commits | Covers |
|---|---|---|---|
| Localization | `claude/p42b-localizer-prng` | 1 | P4.2b |
| VIO / pose | `claude/p44c-vio-update` | 3 | P4.4a–c |
| Sensing | `claude/p45b-rejection-cascade` | 3 | P4.5a–b |
| Mapping | `claude/p46b-coverage-aggregates` | 1 | P4.6b |
| Planning | `claude/p47h-coordination` | 7 | P4.7a–h |

P4.8 — *compose tick, mission state and replay* — cannot start from any one of
them. It needs the localizer, the pose pipeline, the sensing cascade, the
mapping aggregates and the eight planning modules **in one tree**, and until
now nothing had ever put them there. Whether they even compose was an open
question, and the answer mattered before more work was stacked on top.

## They compose

All five merged into `main` with **no conflicts**, textual or semantic:

```
OK  claude/p42b-localizer-prng
OK  claude/p44c-vio-update
OK  claude/p45b-rejection-cascade
OK  claude/p46b-coverage-aggregates
OK  claude/p47h-coordination
```

A clean text merge proves nothing on its own, so the combined tree was put
through the full gate rather than eyeballed:

| Gate | Result |
|---|---|
| `cargo test --workspace --exclude argusnet-viewer` | **393 passing**, exit 0 |
| `cargo test -p argusnet-viewer` | **99 passing**, exit 0 |
| `cargo clippy --workspace --all-targets -- -D warnings` | exit 0 |
| `cargo fmt --all -- --check` | exit 0 |
| `python3 -m pytest tests/ -q` | **981 passed**, 9 skipped, 65 subtests, exit 0 |

Each exit code was read directly from cargo and pytest, not from the tail of a
pipeline — `exit=$?` after `cmd | tail` reports `tail`'s status, which has
misreported a Rust result in this repo before.

## What this does and does not settle

**Settled:** the five stacks are mutually compatible at their current heads.
Nothing in the parallel work has to be redone before P4.8, and no stack has
silently broken another's tests.

**Not settled:** this branch adds no behaviour. It is a merge commit and this
document. It does not compose a tick, and it is not evidence that the tick
*can* be composed — only that the parts it needs are simultaneously available.

**Not addressed here:** two P4 items remain unstarted, and P4.8's exit
criterion ("a full representative run is owned by Rust") depends on both.

* **P4.3 — runtime world construction.** Procedural shaping is still Python.
  Until it moves, a Rust-owned run still needs a Python process to build its
  world, which is exactly what P4.8's exit criterion rules out.
* **P4.6c — the belief half of mapping.** `argusnet-mapping` carries the
  footprint rasterizer and the coverage aggregates; belief revisions and dirty
  tile patches are not there yet. P2.1b in the graphics lane is already waiting
  on them.

## Note on CI

This tree has not been verified on GitHub Actions: the account's hosted minutes
are exhausted, so no workflow runs. Every result above was produced locally.
`scripts/ci-local.sh` reproduces the six CI jobs on a developer machine, and
the workflow reads `vars.CI_RUNNER` per job so a self-hosted runner can take
over without further changes.
