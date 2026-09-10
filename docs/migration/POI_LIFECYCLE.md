# POI lifecycle — P4.7f

Ports `planning/poi.py` — `POIManager`, `POIAssignmentContext` and the
`InspectionPOI`/`POIStatus` pair — into `argusnet-planning`.

`POIManager` is stateful, so the fixture records **sequences** of calls with
the result of each and the full status list after it, rather than isolated
inputs and outputs.

## Construction order is part of the contract

The reference keeps its POIs in a `dict`, which iterates in insertion order,
and three behaviours depend on that:

| behaviour | consequence |
|---|---|
| `assign_nearest` sorts by `(-priority, distance)` with `list.sort`, which is **stable** | two POIs at the same priority and distance keep construction order |
| `assign_energy_aware` scans in construction order, keeping the **first** strictly-better score | ditto, for the energy path |
| `check_completions` walks in construction order | ids come back in that order, *not* the order they finished |

POIs are therefore held in a `Vec`. A `HashMap` passes every individual method
and still returns a different POI, and a different completion order, on the
cases that matter. The completion case is built so `second` reaches its dwell
first in time and is still reported second.

## Quirks reproduced deliberately

- **`assign_nearest` reads the RAW priority.** `rescore_from_map` only affects
  `assign_energy_aware`. The asymmetry is in the reference and is only visible
  when the raw and boosted winners differ, so the fixture arranges exactly
  that.
- **`POIStatus.dwell_accumulated_s` reads zero while a platform hovers.** The
  running dwell lives in a separate `_dwell_acc` dict and is copied into the
  status only at completion, so a caller watching that field for progress sees
  nothing until the POI is already done. The port keeps the two separate for
  the same reason: merging them would quietly change what a status means.
- **`trigger_handoff` silently replaces an assignment the receiving platform
  already held**, leaving the old POI Active with a stale drone id.
- **`request_team_assign` records only one drone in the status**, so a reader
  of the statuses alone cannot tell a POI is being worked by two — though
  completion correctly releases both.
- **The 80% affordability factor** is a second, unnamed margin on top of
  `battery_reserve_fraction`, applied to whatever the first leaves.

## The rounding trap, again — and it is subtler here

`rescore_from_map` computes `round((1 - density) * 3)`, and Python's `round()`
is banker's. The port uses `round_ties_even()`, as the route cache key does in
P4.7d.

Reaching a tie takes more care than it looks. The obvious density of 5/6 gives
`(1 - 5/6) * 3 == 0.4999999999999999`, **not** `0.5` — so both rounding rules
send it to 0 and the case proves nothing. The first version of this fixture did
exactly that and passed its own guard, because the winner happened to be the
same either way.

A density of `0.1666666666666666` — one 6 short of the nearest double to 1/6 —
makes the product land exactly on `2.5`, where the rules genuinely differ
(banker's 2, half-away 3). The two POIs are then arranged so that single boost
decides the assignment, and the generator refuses to emit the case if the
density stops landing on a tie.

## Verification

Fixture: `tests/fixtures/poi.json`, from `scripts/capture_poi_fixture.py`,
which drives the reference and records every step. Regenerates
byte-identically.

The replay checks each call's result **and the full status list after it**.
Comparing only return values would miss a port that answered correctly while
leaving its bookkeeping wrong — the mistake that cost two rounds in P4.4.

**The generator refuses to emit a case that does not do what it claims:** no
successful *and* rejected handoff, no successful *and* rejected team
assignment, no refused assignment, no `check_completions` returning more than
one id (without which the reported order is unobservable), a rescoring case
whose winner does not turn on the boost, or an `assign_nearest` case where the
raw and boosted priorities agree. Three of those guards fired during
development.

### Mutation testing: 34 mutations, each verified to have reached disk

**27 caught by a named test, 6 proven equivalent, 1 open gap.** The first pass
caught 21.

Two findings came out of the first replay rather than the mutations:

- **`(1 - 5/6) * 3` is not `0.5`.** The rounding case never discriminated, and
  its guard was too weak to notice — it checked the winner, which was the same
  under both rules.
- **The port published the running dwell into the status.** The reference does
  not, until completion. Caught immediately by comparing full statuses rather
  than return values.

### Six proven equivalent mutants

Asserted in `five_recorded_equivalent_mutants_in_the_poi_manager` and its
neighbours, each resting on an invariant rather than on the fixture:

1. **The completed check in `existing_assignment`** and
2. **the Active check in `trigger_handoff`.** An entry in the assignment map
   can only point at an *Active* POI: assignment sets the state in the same
   breath, and completion removes every entry pointing at the POI.
3. **The dwell reset on assignment.** A POI is only assigned while Pending, and
   dwell is credited only to a drone's *assigned* POI, which is Active — so a
   Pending POI's dwell is always already zero.
4. **The priority tier scale of 10.** The affordability gate caps cost at 80%
   of usable energy, so the energy term is at most 0.8 while priorities differ
   by at least 1; any scale of 1 or more orders candidates identically.
5. **Replacing an arrival time of exactly zero.** Nothing writes
   `arrival_time_s` before completion, so the zero branch is unreachable —
   kept because the reference's `arrival_time_s or timestamp_s` would behave
   this way if a writer ever appeared.
6. **Rescoring non-pending POIs.** Effective priorities are read only for
   Pending candidates, and a POI never returns to Pending.

### One open gap, not dressed up as equivalent

Swapping `sort_by` for `sort_unstable_by` in `assign_nearest` **is not
caught.** `assign_nearest` returns only `pending[0]`, so a reorder is invisible
unless it disturbs the minimum, and it does not at any shape this fixture can
express.

Measured rather than assumed: `sort_unstable_by` preserves order for a handful
of elements whatever their keys, and for *any* number when every key is equal —
pdqsort's all-equal partition never moves anything. It begins reordering at
around forty elements once several distinct groups exist, which is why the tie
set is 48 POIs across twelve groups rather than the five, then twenty-four, it
held earlier. Neither of those earlier shapes could have caught it at all.

So the requirement is pinned as a *case* — the reference's stable sort returns
the first-constructed member of the winning group, and the port agrees — but
not as a *mutation*. `sort_unstable_by` guarantees nothing about equal keys, so
a future toolchain could make this diverge silently.
