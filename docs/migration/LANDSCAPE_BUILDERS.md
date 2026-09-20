# Landscape builders (P4.3g)

`src/argusnet/world/procedural.py` → `rust/argusnet-world/src/landscape.rs`.

The slice that makes P4.3 load-bearing. P4.3a–f ported the pieces and verified
each against its own fixture in isolation; this wires them into the three
functions the rest of the system calls — `build_terrain_layer`,
`build_land_cover_layer` and `build_landscape`.

There is no new mathematics. Every line calls something already under test,
which is why the fixture compares whole builder outputs rather than re-checking
any individual formula.

Fixture: `tests/fixtures/landscape.json` — 9 scenes.

## Five seams no earlier fixture could reach

A slice that is pure composition needs a different kind of test. The scenes are
chosen against the *joins*, not against the formulas.

| Seam | What goes wrong if it is mis-wired |
|---|---|
| **`effective_seed`** | With no explicit seed it is `stable_seed("terrain", preset)` — a different derivation from the `stable_seed("profile", preset, seed)` that `procedural_terrain_grid` then applies to it, and different again from `stable_seed("land-cover", preset, seed)` for the moisture field. **Three stable seeds in one build, each feeding the next.** Collapse any two and every scene is plausible and wrong. |
| **The cover resolution is derived** | `max(10.0, terrain_resolution * 2.0)`. Terrain and land cover are rasterised at *different* resolutions, which is what makes the mask resampler do real work. A port that shares one resolution passes every isolated fixture and builds the wrong world. Both halves of the `max` bind in the fixture. |
| **Heights come from the tiled field** | `from_height_grid` resamples into padded tiles, so a height read back is not bit-identical to the grid it was built from. Sampling the source grid instead is within a hair everywhere and wrong at every tile seam. |
| **The gradient step** | `slope_rad_at_many` uses `max(resolution * 0.5, 0.25)` — while `TerrainModel.gradient_at` defaults to `0.5` and `slope_rad_at` to `1.0` and passes it down. Three steps for "the gradient" in one package; this pins which one land cover gets. |
| **`SeasonalVariation.from_month`** | The table lives in `environment.py`, far from everything else here. Winter's `0.15` is the one factor whose product with 190 is not an integer — `28.5`, which numpy's `uint8` cast truncates to 28 where a round gives 29. |

## Multi-tile coverage without a multi-megabyte fixture

Testing the tile seam needs a world spanning more than one tile. At the default
`tile_size_cells = 256` and 4 m resolution a tile spans **1 km**, so a
multi-tile scene needs a 2.4 km world — and that first capture produced a
**4.7 MB** fixture, most of it 240,000 mask cells per mask across six masks, to
exercise one seam.

`tile_size_cells` is a config field with a floor of 2. Setting it to **32**
gives 3.6 tiles across the same small world. The seam is what is under test,
not the acreage. The fixture is 669 KB.

The semantic masks are recorded as **coverage fractions**, not bitmaps, for the
same reason: `terrain_grid.json` already pins them cell-for-cell from P4.3e, and
repeating that here would add megabytes to re-assert something proven. What is
*not* proven elsewhere is that those masks reach the classifier at all — and the
recorded `classes` carry that, because a mis-wired resample changes the map.

## Result

Every scene reproduces the reference. Worst divergence across all nine:

- **heights 8.5e-14 m**, through the terrain's `exp` envelopes and a bilinear
  tile resample
- **slopes 1.2e-14 rad**, adding an `atan`

Both sit inside ADR-006's bound for numpy's `exp`, and land cover matches
**exactly** — every class and every density byte, on every cell of every scene.
That is the useful result: the floating-point difference is real and does not
reach the classification.

## What is deliberately not ported

`terrain_source` accepts `procedural`, `dem` and `hybrid`. Only `procedural` is
built here. P4.3's own scope note says *"DEM/GIS decoding may stay in offline
import tools, but the Rust runtime accepts a documented preprocessed world
bundle directly"*, and the DEM path pulls in GeoTIFF reading and CRS projection
— a large dependency surface serving an import step, not the tick loop.

The runtime requirement P4.3 actually states is *"Pure synthetic defaults MUST
need neither Python nor GLB baking"*, and that is what this delivers.
`build_terrain_layer` takes the source as an explicit `TerrainSource` argument
rather than a string, so `dem` and `hybrid` cannot be passed at all — a
compile-time refusal instead of a runtime one that might be mistaken for
"procedural" and silently ignore the DEM the caller asked for.

## What this unblocks

P4.3 is now complete *and* wired. A representative world constructs in Rust from
a preset and a seed with no Python process, which is P4.3's stated exit
condition. P5.1 (Rust world boot and terrain LOD) can call these directly
instead of shelling out to `build-scene`.
