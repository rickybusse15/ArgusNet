# Land-cover polygon overlay (P4.3f)

`src/argusnet/world/procedural.py::build_land_cover_layer` →
`rust/argusnet-world/src/land_cover_build.rs`.

Sixth and last slice of P4.3, and the tail of the land-cover builder: the two
loops that let obstacle footprints and caller-supplied patches repaint cells the
classification has already decided. P4.3d covered the seven mask rules and was
captured with `obstacles=()` and `patches=()` precisely so these loops were
no-ops there.

Fixture: `tests/fixtures/polygon_overlay.json` — 14 scenes, 216 cells each.

No new geometry. `point_in_polygon` was ported in P4.3a and is shared with the
planner; this slice is composition, and because it draws no random numbers it is
bit-exact on every platform.

## The fixture is a difference, not a snapshot

Each scene runs the real `build_land_cover_layer` **twice** on identical inputs:
once with the obstacles and patches, once without. Everything upstream is
deterministic, so the two rasters differ by the overlay and nothing else. The
fixture stores the empty run as `base_*` and the full run as `final_*`, and the
port's job is to turn one into the other.

That shape matters. A fixture that recorded only the final grid could be
satisfied by a port that reproduced the *classification* and got the overlay
wrong, since the classification is most of the answer. Recording the difference
makes the overlay the only thing under test.

## Four rules, and the wrong port that each one catches

| Rule | Wrong port it rules out |
|---|---|
| **Obstacles run before patches** | Swapping the loops leaves every individual rule correct and changes the map wherever they overlap. A patch exists to override what the generator produced; it can only do that by running last. |
| **Class assigns, density takes `max`** | This is what separates the overlay from the `road` rule twenty lines above it, which assigns density **zero**. Two ways to put a road on the map, two different densities: the road mask clears the foliage, a `ROAD` *patch* keeps it. |
| **A forest stand is skipped under `suppress_vegetation`; a building is not** | The flag removes the trees and leaves the town. A patch is not gated at all, so a `FOREST` patch lands even with vegetation suppressed — the flag suppresses *generated* vegetation, not vegetation the caller named. |
| **An obstacle with no footprint is silently dropped** | The reference's `else: continue` catches exactly the cylinders, which have no `footprint_xy_m` attribute. A cylindrical obstacle occludes sight and blocks movement and contributes nothing whatever to land cover. |

Two numeric traps round the wrong way if transcribed casually:
`int(np.clip(obstacle.density, 0.0, 1.0) * 255)` **truncates** — 0.5 gives 127,
not 128 — and `int(np.clip(patch.density, 0, 255))` clamps a density that a bare
`as u8` cast would wrap (300 → 44).

## The forest test is on `kind`, not on the blocker type

The reference asks `isinstance(obstacle, ForestStand)`. A `PolygonPrism`
constructed with `blocker_type="vegetation"` is **not** a `ForestStand`, so it
takes the urban branch — and under suppression it stays on the map while a real
stand disappears. `kind` is the field that survives that distinction into Rust;
`blocker_type` is not, and a port that dispatches on it paints forest where the
reference paints a building. `vegetation_prism_is_not_a_stand` is the scene that
says so.

## A 1-ULP divergence found here, and deliberately not chased

`OrientedBox.footprint_xy_m()` rotates its corners with `corners @ rot.T`, which
numpy hands to BLAS — and BLAS accumulates the two-term dot product with a
**fused multiply-add**, `fma(y, -sin, x * cos)`: one rounding where Rust's
`x * cos - y * sin` has two. The derived vertices differ by up to
**2.8e-14 m**.

Reproducing it was checked and it does reproduce, exactly, with one `mul_add`.
It is not done, on ADR-006 Decision 1's own reasoning: whether numpy's matmul
fuses depends on the BLAS vendor, the matrix size and the CPU's feature set, so
a `mul_add` here would match one build and silently mismatch the next. That is a
different case from P4.3e's PRNG, where the stream is defined by a published
algorithm rather than by a vendor.

**It cannot change a classification.** `point_in_polygon` treats anything within
`SEGMENT_EPSILON_M` — 1e-6 m — of an edge as on the boundary and inside, so a
vertex shift eight orders of magnitude inside that band cannot move a cell
across it: a point close enough to be at risk was already answered by the
boundary short-circuit, identically, under both spellings. The overlay
comparison passes bit-for-bit on the same fixture, which is the observation
rather than the argument. `polygon_overlay_parity` carries the tolerance, the
reasoning, and a compile-time check that the tolerance stays three orders of
magnitude inside the boundary band.

Note that **production is unaffected today**: `OrientedBox.to_metadata()` sends
Python's own derived footprint to the native side, so the Rust constructor is
not on that path. It becomes live when the Rust runtime builds its own scenes.

## Mutation testing

Twenty mutations of the production code, run against the whole crate suite:
**19 killed**.

The survivor is honest and is recorded in the code: replacing
`density.clamp(0.0, 1.0)` with `density.max(0.0)` changes nothing, because
Rust's `as u8` **saturates** — `2.0 * 255.0` and `1.0 * 255.0` both land on 255.
The clamp is not redundant in the reference, where `int(2.0 * 255)` is 510 and
assigning that into a `uint8` raster raises. Two different mechanisms, one
behaviour. The clamp stays, with a comment saying why no test can reach it, and
`a_stand_density_above_one_saturates_rather_than_wrapping` pins the *behaviour*
so a future change to the cast or the rounding cannot quietly turn a dense
canopy into a sparse one.

The capture script carries the same twelve mutations on the Python side and
refuses to write a fixture unless each one is caught by at least one scene, and
unless its own re-implementation reproduces the recorded `final_*` first. A
guard that asserts a proxy has caught nothing four times in this migration; this
one asserts the final classification.

## What is left of `build_land_cover_layer`

Everything except the caller. The classification (P4.3d), the mask resampler
(P4.3d), the moisture noise (P4.3a) and now the overlay are all ported; what
remains is the function that computes the cell centres, samples the terrain for
heights and slopes, and calls them in order. That belongs with
`build_terrain_layer` and `build_landscape` in the next slice, since all three
are composition over ported parts.
