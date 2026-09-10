# Deconfliction — P4.7g

Ports `planning/deconfliction.py` — `DeconflictionConfig`,
`CorridorWindowAllocator` and `DeconflictionLayer.resolve_step` — into
`argusnet-planning`. Terrain and exclusion zones arrive as closures, keeping
the layer a pure function of its inputs.

## Order decides the outcome three times over

**The lexicographic tie-break yields the LARGER id.** When two platforms share
a role priority, `_yielder` returns `(a, b) if a > b else (b, a)` — so `bravo`
gives way to `alpha`, not the other way round. A port that yielded the smaller
id passes every case where the roles differ and quietly swaps the two whenever
they do not.

**The pairwise pass writes back in place.** It walks `sorted(ids)` and stores
each adjustment immediately, so a later pair sees an earlier pair's result.
Three platforms in a row do not resolve to what three independent pairs would
give, and the fixture has a case that shows it.

**Corridor membership follows the PROPOSED order**, not sorted order, and a
`held` set short-circuits later pairs — so which platform ends up holding
depends on the order the caller supplied. Reproduced with an order-preserving
input rather than a map.

Events are compared **in order**, not as a set: a set comparison would accept
the right conflicts resolved in the wrong sequence, which is exactly what the
in-place writeback makes observable.

## Verification

Fixture: `tests/fixtures/deconfliction.json`, from
`scripts/capture_deconfliction_fixture.py`, which drives the reference and
records the full adjusted state and event list per case. Regenerates
byte-identically. 35 cases, 29 events.

**The generator refuses to emit a set that does not exercise what it claims:**
every resolution present, at least one clean case with no events, at least one
case with more than one event (without which the event order is unobservable),
a tie case whose yielder is the larger id, a corridor window that genuinely
flips the holder, a derived window that chooses a different holder than the
default would, a `held` set that actually short-circuits a second hold, and a
blocked-side case that stays lateral. **Four of those guards fired during
development.**

### Mutation testing: 36 mutations, each verified to have reached disk

**32 caught by a named test, 4 proven equivalent.** The first pass caught 23.

Two of the survivors needed geometry built specifically for them, and both
took a second attempt:

- **The opposite-side flip.** Blocking the first side usually falls through to
  a *vertical* offset rather than the mirror, because the first push already
  points away from the peer and its mirror points toward it — still too close.
  The flip only lands laterally when the two platforms are far apart but
  closing head-on, so the conflict is predicted rather than current and there
  is room on both sides. The first attempt at this case produced a vertical
  offset and never reached the branch.
- **The flip-away step itself.** Dropping it is normally *absorbed*: the
  un-flipped push heads toward the peer, the still-close check rejects it, and
  the opposite-side fallback lands on exactly the side the flip would have
  chosen. It only shows when the toward-the-peer side is also viable, which
  needs a head-on approach with a small lateral offset — enough for the
  perpendicular to have a positive component toward the peer, and enough
  clearance that both sides survive the separation check.

The rest were ordinary boundary gaps, each closed with a case: separation
exactly at the minimum (the guard is `>=`), a corridor member exactly at the
half-width (`>` width), a single-element active window (always active),
crossing traffic with no along-axis speed, a platform that would be held twice,
a window derived from corridor length rather than the default, and ids supplied
out of sorted order.

### Four proven equivalent mutants

Asserted in `four_recorded_equivalent_mutants_in_the_deconfliction_layer`:

1. **`t_star` for a stationary pair.** With zero relative velocity the
   separation is `r0 + vr * t` for any `t`, so 0 and the full look-ahead agree.
2. **The `max(margin)` floor on the push magnitude.** The code only reaches it
   when `pred_sep < min_sep`, so `min_sep - pred_sep` is already positive and
   `+ 1.0` puts the magnitude above the floor. It can never bind.
3. **The direction of the stationary fallback.** Starting the push from
   `peer - y` rather than `y - peer` is corrected by the very next step, which
   flips any perpendicular pointing at the peer.
4. **The `len() < 2` early return.** With one platform the pairwise loop makes
   no pairs and a corridor needs two members, so the function already does
   nothing.
