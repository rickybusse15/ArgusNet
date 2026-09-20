# P4.8b: the tick's shared world, and truth as a type

**Status:** delivered 2026-09-16. Second slice of P4.8, after P4.8a (the mission
state machine).

## The blocker this removes

P4.1 built the tick transaction — clock, stages, commands, fault semantics,
trace — and left the stage bodies pluggable. But `TickContext` carried a tick
number, a timestamp, a seed and a notes buffer: everything a stage needs to
*describe* what it did and nothing it needs to *do* anything.

So no domain stage could be wired in at all, and `commit_current` published
`RuntimeSnapshot::empty` on every tick of every run. Every ported crate —
sensing, mapping, planning, localization, the VIO — had nowhere to attach.
`TickWorld` is the state they attach to.

## One list, mutated in place, frozen once

P4.8: *"Eliminate repeated dataclass-style list rebuilds; freeze once at the
snapshot boundary."*

The reference rebuilds `node_states` **up to seven times per tick** —
battery → frontier → POI → egress → deconfliction → safety → frame — each one a
full `dataclasses.replace` over every platform, to change one field of one of
them. A faithful transcription would preserve that shape and waste the
migration.

Here there is one `Vec<NodeState>`. Stages take `&mut` and edit the platforms
they are changing. The single clone happens at `TickWorld::freeze`, which is
stage 10 and nowhere else.

"Once" is a **tested property**, not an intention:
`nine_stages_edit_one_list_and_it_is_frozen_once` installs an editing handler on
every one of stages 1–9, and asserts `freezes_this_tick() == 1` with the
untouched platform still untouched. A mutant that freezes twice is caught, and
so is one that stops resetting the counter between ticks.

## Truth is a capability, and the capability is a type

`ARCHITECTURE_AND_PROTOCOLS.md` §3 makes truth a capability rather than a
convention. A comment asking stages not to read truth is worth nothing; a stage
that *cannot name* truth is worth something.

So truth does not live on `TickWorld` at all. It reaches a stage only as
`TickContext::truth`, which the simulation populates **only** for the stages
`Stage::may_read_truth` admits:

| stage | truth | why |
|---|---|---|
| `UpdateVioAndPose` | **yes** | builds synthetic IMU features from the true world |
| `GenerateObservations` | **yes** | is the sensor model |
| every other stage | `None` | there is no other path |

`None`, not an empty slice — "you may not look" and "there is nothing to see"
are different facts and a stage should be able to tell them apart.

`may_read_truth` is a `match` over **every** variant rather than a default, so a
new stage cannot join the truth list by omission — and cannot leave it silently
either. Removing `GenerateObservations` from the `true` arm does not fail a
test; it fails to **compile**.

That also makes the snapshot's `truth_filtered` flag honest. It used to be a
hardcoded `true` beside an empty snapshot. Now the snapshot is built from
`TickWorld`, which never held truth, so the flag is a statement about the type
rather than about a filtering step someone has to remember to run.

## Three smaller decisions

- **A failed platform lookup is not a mutation.** `platform_mut` returns `None`
  for an unknown id rather than inserting, and does not bump the revision. A
  stage naming a platform that is not in the fleet has a bug; silently creating
  one would hide it, and the integration test asserts it faults the run
  instead.
- **A reset clears truth as well as the world.** A reset is a new epoch, so the
  previous epoch's platforms are not this run's — and leaving stale truth behind
  would let a fresh run sense the *old* world for one tick. Both halves are
  separately mutation-tested.
- **The world revision keeps climbing across a clear**, for the same reason the
  run's does: it orders snapshots a consumer may receive out of order, and
  restarting it would make a stale snapshot look newer than a current one.

## A gap in `argusnet-types`, fixed in passing

`NodeState`, `TruthState`, `TrackState` and `BearingObservation` all carry
`Vector3<f64>`, and `argusnet-types` did not re-export it — `Matrix6` was
exposed, `Vector3` was not. So a consumer could not construct any of them
without taking its own `nalgebra` dependency and keeping the version in step by
hand; two copies of the type would not unify, and the error would be about a
version mismatch rather than about the code. For a crate whose entire job is
"one authority for value types", that is a real hole. `pub use
nalgebra::Vector3;`.

## Verification

- 7 unit tests on `TickWorld`, 9 integration tests driving the real pipeline.
- **12 mutants, 12 caught.** One of them — removing a stage from the truth
  capability — is caught at compile time by the exhaustive match, which is a
  stronger guarantee than a failing test.

## What comes next

Stage bodies. Every crate P4.2–P4.7 ported now has somewhere to attach:
controllers and battery at stage 1, the mission overrides at 2, deconfliction at
3, the safety monitor at 4, VIO at 5, `build_observations` at 6, fusion at 7,
coverage/belief/localization at 8, and P4.8a's mission machine already at 9.

Then the versioned Rust replay writer, which is what turns a committed snapshot
into the document Python evaluation reads back under strict schema validation.
