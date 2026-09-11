# Frontier exploration — P4.7e

Ports `planning/frontier.py` — `FrontierConfig`, `ClaimedCells`,
`select_frontier_cell` and `find_gap_cells` — into `argusnet-planning`.

The coverage grid arrives as a plain `CoverageView` rather than through the
mapping crate. The selection is a pure function over counts, bounds and
claims, so keeping it that way makes the tie-breaking testable and avoids
depending on P4.6b, which is still in flight.

## Two behaviours that decide the answer

**Ties go to the lowest `(i, j)`.** The reference scores every candidate and
takes `np.argmax`, which returns the *first* maximum, over candidates from
`np.where(grid == 0)` — row-major, so `i` ascending then `j`. At low coverage
the gradient term is off and equidistant cells score identically, which makes
ties the ordinary case rather than an edge case. A port that kept the last
maximum, or walked the grid column-major, sends platforms somewhere else
entirely. Both are caught, by a scene built to tie: a 3×3 with only the centre
covered, and a platform standing on it, leaves the four orthogonal neighbours
scoring identically — and the reference picks `(0, 1)`, first in row-major
order.

**A hole is not a frontier.** `find_gap_cells` returns only *enclosed*
uncovered regions, because `binary_fill_holes` fills background that cannot
reach the border. This is a 4-connected flood fill inward from every border
cell, matching that function's default structure. Two scenes pin it: one where
the only path out is diagonal (enclosed under 4-connectivity, reachable under
8), and one whose only opening is through the *interior* of the bottom row
(missed by a fill seeded from fewer than all four edges).

## A float difference taken deliberately

The reference computes the coverage gradient as
`uniform_filter(covered_mask, size=3, mode="constant") * 9`. Its own comment
calls that a neighbour count, and mathematically it is — but it is evaluated
as two separable `1/3` passes, so it lands *near* the integer rather than on
it. The captured fixture holds `0.9999999999999991` where the count is 1, and
`-8.326672684688674e-16` where it is 0. **Nine of the nineteen distinct
gradient values in the fixture are not integers.**

This port counts directly. The difference can only change an outcome by
flipping a tie, so rather than assume it is harmless the fixture records how
far ahead the winner was in every case, and a test checks each margin against
the noise. **The narrowest margin is 9.2e-2 — thirteen orders of magnitude
above the largest deviation observed.** If a future case ever came within that
band, the test would say so instead of the port quietly disagreeing.

## Not carried across

`find_gap_cells` takes a `threshold: float = 0.70` parameter and never reads
it. Its only caller uses the returned list's *length*, comparing it against
`gap_fill_min_fraction` of the grid, so the parameter is dead in both senses.
Dropped rather than reproduced.

## Verification

Fixture: `tests/fixtures/frontier.json`, from
`scripts/capture_frontier_fixture.py`, which drives the reference and reads
its results back. Regenerates byte-identically.

**The generator refuses to emit a case that does not do what it claims** — it
raises if no scene turns the gradient on, if none leaves it off, if no
selection returns `None`, if fewer than four distinct cells are chosen, if no
scene has an enclosed hole or none is hole-free, if the edge notch is
misreported as a hole, or — added after the first pass — **if no selection is
an exact tie**, since without one the first-index rule the fixture claims to
pin is never exercised.

### Mutation testing: 22 mutations, each verified to have reached disk

**20 caught by a named test, 2 proven equivalent.** The first pass caught 14.

Two survivors were **a fault in my own test, not in the fixture.**
`the_gradient_ramp_matches_python` recomputed the ramp formula *inside the
test* and compared that against the recorded weight — so mutating the ramp in
the source could not fail it. Widening the ramp from 0.50 to 1.0 and dropping
its upper clamp both passed, and neither changed any selected cell either. The
ramp is now a named function the test calls, which is the same rule the
generators follow: a check that reimplements what it checks cannot fail.

The rest were ordinary fixture gaps, each closed with a scene built for it: a
claim at exactly the exclusion radius (rather than one excluding everything
and falling through), a grid large enough that normalising distance by the
diagonal decides between a near, poorly-surrounded cell and a far,
well-surrounded one, and the two flood-fill scenes above.

### Two proven equivalent mutants

Asserted in `two_recorded_equivalent_mutants_in_the_frontier`:

1. **Skipping the neighbour count when the gradient weight is zero.** The
   weight multiplies the count and `0.0 * n == 0.0` for every finite `n`, so
   the branch is an optimisation rather than a behaviour.
2. **The `max(1.0)` floor on the distance normaliser.** It is
   `hypot(nx, ny)`, and a grid has at least one cell on each axis, so the
   value is at least `hypot(1, 1)` — already above 1. Checked across every
   grid up to 64×64.
