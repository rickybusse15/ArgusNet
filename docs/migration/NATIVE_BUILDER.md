# P4.9b: the Rust-native scenario builder

**Status:** P4.9b.1 (the world half) delivered 2026-09-23, locally verified
only (not pushed). P4.9b.2 (the scenario half) is open. See
[STATUS.md](STATUS.md).

P4.9's exit requires "no Python runtime dependency", with versioned config
files as the canonical input. Since P4.8e.1, a scenario reaches the Rust
runtime as a bundle, and only Python could build one: `build_default_scenario`,
then `export_bundle`. P5.0 and P5.1 then load scenarios in the viewer and
build their worlds in Rust. The native builder closes that gap. It reads an
`argusnet.scenario-config` document and produces the same bundle Python would.

## The config document

```json
{"format": "argusnet.scenario-config", "version": 1, "seed": 7,
 "options": {"map_preset": "small", "terrain_preset": "alpine"}}
```

`options` is `ScenarioOptions`, field for field, with the reference's
defaults. Unknown fields are refused, so a typo is an error rather than a
silently defaulted option. DEM and hybrid terrain are refused as
`Unsupported`. P4.3 keeps GeoTIFF decoding an import step, not a runtime
build.

## P4.9b.1: the world half

`builder::build_world(options)` builds the map bounds, the procedural terrain
(P4.3's port), the environment, and the land cover. The environment is the
obstacles and patches of `build_default_environment`,
`build_obstacle_cluster`, `_preset_environment_profile` and
`_filter_obstacles_for_landscape`, ported line for line. From those,
`BuiltWorld` writes:

* the bundle's `world` section: terrain tiles, obstacle documents,
  land-cover tiles, the optical visibility profile and the sensing
  clearance;
* the world keys of `presentation`: terrain metadata with its viewer mesh,
  terrain summary, obstacles, land-cover metadata and legend,
  environment id, CRS, bounds and the ENU origin.

Obstacles are built as their **metadata documents**, which is what a bundle
carries. The land-cover pass gets its `Obstacle`s from those documents
through `ObstacleMetadata::into_obstacle`, which is the runtime's own path
from a bundle.

### Parity is exact, except where numpy's platform functions reach

Everything that does not derive from a terrain height is compared
**exactly**: structure, identifiers, kinds, every footprint coordinate,
counts, masks and their coverage fractions, every land-cover class and
density cell, and the seeds.

Heights, and the values computed from them, are compared **within 1e-12 m**.
That covers obstacle base and top elevations, the terrain's min and max, and
the viewer mesh. The worst difference measured is 2.0e-13 m. This is P4.3's
finding carried forward: numpy's `exp` differs from the platform's on about
5% of inputs, and tiling spreads the resulting ulp-level height differences
to about half the tile samples. Bit-identical heights would mean
reimplementing numpy's `exp`. That would still not give a single target,
because numpy's `exp` is platform-specific: the "Python value" differs
between machines.

## Findings

* **numpy's matmul fuses.** `OrientedBox.footprint_xy_m` is
  `corners @ rotation.T + centre`. On the reference platform numpy computes
  each output as `fma(y, r, x * c)`, then adds the centre. That was tested
  against both association orders and both FMA placements, and only this
  one reproduces every corner. The world crate's own `Obstacle::oriented_box`
  associates differently and would be one ulp off, so the builder has its
  own `box_footprint`. The runtime is unaffected, because it reads
  footprints from metadata. Like `exp`, this is the platform's kernel: on a
  machine whose matmul does not fuse, the reference itself would change.
* **Two different `_stable_seed`s.** `sim.py`'s FNV-1 hash (land cover,
  targets) and `procedural.py`'s blake2b hash (terrain) both hash
  `str(part)`, and a float part hashes as Python prints it (`"1.0"`, not
  Rust's `"1"`). `python_float_repr` reproduces CPython's `repr`, and a unit
  test pins it, including the exponent cases.
* **The ground-contact pad is 3 m, not 1.** It lives in
  `SimulationConstants.dynamics`. The first port assumed 1 m, and the parity
  test caught it as a flat 2 m error on every top elevation.
* **The landscape filter's rarer branches are reachable, but only by
  seed.** At the default seeds no building is dropped for water alone and no
  forest stand for water. A search over terrain seeds found both (alpine 35,
  river valley 0), and those seeds are now fixture cases.
* **The lake ellipses match.** `np.cos`/`np.sin` over an angle array could
  have diverged from libm in the same way `exp` does. They do not on these
  inputs: every land-cover cell of the lake-district cases matches.

## Verification

| Test | Shows |
| --- | --- |
| `builder_world::every_preset_builds_the_references_world` | 27 cases (`scripts/capture_scenario_world_fixture.py`): every terrain preset, including the unknown-name fall-through; every map scale from small to operational; the large, xlarge and regional clusters and corridor walls; clean terrain; winter; explicit seeds and detail strength; a 2.5 m terrain, where the land cover's 10 m floor applies; and searched seeds that reach the landscape filter's rarer branches (a building dropped for water alone, for water and slope, a forest stand dropped for standing half in water) and a tower whose lowest ground is its centre sample. For each case: obstacle documents (all five kinds, up to 21 obstacles), both land-cover rasters cell for cell, land-cover metadata, terrain summary and a viewer-mesh sample. 7,010 numbers exact; 6,792 heights within tolerance, worst 2.0e-13 m |
| `builder_world::box_and_wall_footprints_match_the_references_arithmetic` | 300 random boxes and 300 random walls, footprinted by the reference's own `OrientedBox` and `WallSegment`: bit for bit |
| `builder_world::the_small_alpine_bundles_world_and_presentation_are_rebuilt` | the whole `world` and `presentation` sections of all three committed bundles: terrain tiles, land-cover tiles, the full viewer mesh |
| `builder::tests`, `builder::environment::tests` | `python_float_repr` against CPython, including exponents; `linspace`'s endpoint correction, where `i * step + start` overshoots; the mask lookup's half-to-even rounding and clamping; config defaults and the refusal of unknown options |

### Mutation round

20 mutants. **18 caught and 2 equivalent**, after a strengthening pass. The
first pass caught 11. The survivors were behaviour the preset matrix did not
reach, so the matrix grew rather than the claim shrinking:

* the fused matmul, per axis: a 300-box geometry test;
* `hypot` for a wall's length: 300 random walls;
* the cylinder's centre sample, the landscape filter's water branches, and
  the land cover's resolution floor: searched seeds and a 2.5 m case;
* the mask lookup's rounding and `linspace`'s endpoint: unit tests. No preset
  reaches either edge.

The two equivalents:

* **The wall offset's association.** `(n * t) * 0.5` and `n * (t * 0.5)` are
  always equal, because scaling by 0.5 is exact in binary floating point
  (outside subnormals).
* **A steep fraction of exactly 0.5 for a building or wall.** They are sampled
  at five points (four corners and the centroid), and a cylinder at one, so
  0.5 cannot occur. The `>` in the filter is still the reference's.

### Gate

Pinned toolchain 1.98.0-aarch64, exit codes read directly:

| Check | Result |
| --- | --- |
| `cargo test --workspace` | 1067 passed, 0 failed |
| `cargo clippy --workspace --all-targets -- -D warnings` | clean |
| `cargo fmt --all -- --check` | clean |
| `ruff check src/argusnet scripts tests`, `ruff format --check` on touched files | clean |
| `pytest tests/` (`ARGUSNET_REQUIRE_RS=1`) | 1022 passed, 9 skipped, 0 failed |
| CI | **not run**: nothing is pushed |

## P4.9b.2a: numpy's `integers`, and the target behaviours

**Delivered 2026-09-23, locally verified only.**

The scenario half draws from numpy beyond what P4.3 ported. The behaviours
and mission zones call `Generator.integers`, and `NumpyRng` now reproduces it:

* **PCG64's half-word buffer.** `next_uint32` returns the low half of a fresh
  64-bit draw and keeps the high half for the next call. `next_uint64` and
  `next_double` neither read nor clear it, so an `integers` draw's second
  half can come back after any number of doubles.
* **Lemire's bounded method,** with its rejection loop, and the exact
  `2^32 - 1` path. Ranges wider than 32 bits are refused, because no caller
  uses them.

`standard_normal` (numpy's ziggurat) is **not** needed. Its only user,
`TurbulenceModel`, is referenced nowhere outside `behaviors.py`, and neither
are the policy behaviours.

`builder::behaviors` ports `build_target_trajectory` and the eight presets it
reaches: loiter, transit (Hermite, then a loiter), evasive (nap-of-earth
windows over a sinusoid), expanding-square search, sinusoid, racetrack,
waypoint patrol, and the `mixed` composite with its 2 s smooth-step blends.
The evaluation order follows the reference wherever it can change the last
bit: numpy's elementwise expressions are separate passes, `**` is C `pow`,
and float `%` takes the divisor's sign.

### numpy's `uniform` is platform-dependent too

numpy 2.5 on arm64 computes `uniform` as `fma(high - low, u, low)`. That is
true on 4,000 of 4,000 draws tested, where the unfused form agrees on only
55%. The committed `numpy_rng.json` `uniform` section was recorded on a build
that does not fuse, and `NumpyRng::uniform` reproduces that build. Rather
than chase each machine's contraction, the parity fixtures separate the two
effects:

* `numpy_rng.json`'s new `mixed` section interleaves `integers` with
  `random()`. `next_double` is integer-exact everywhere, so the section is
  platform-independent. The capture script's `--sections` flag adds it
  without overwriting the committed `uniform` section, which a fusing build
  would move by one ulp in about 45% of its values.
* `behaviors.json` is recorded with `default_rng` wrapped so `uniform` is the
  unfused form. The behaviour logic is then pinned bit for bit, and the
  platform's FMA is left to the builder's end-to-end comparison.

### Verification

| Test | Shows |
| --- | --- |
| `numpy_rng_parity::integers_interleaved_with_doubles_match_numpy` | 684 `integers` draws over 19 seeds, interleaved with `random()`, including the rejection loop (a range of 2^31 + 1) and the 2^32 - 1 path |
| `behaviors_parity::every_preset_matches_the_reference_bit_for_bit` | all eight presets, three bounds, four seeds each, 10,464 samples plus a dense sweep of both `mixed` blend windows. Every position and velocity component is exact |

### Mutation round

16 mutants. **15 caught and 1 equivalent.** The first pass left `pow` against
`powi` in the blend: the 2.5 s sample grid put too few points inside the 2 s
windows, so a dense blend sweep was added. The equivalent is `hypot` for a
patrol leg's length. Every patrol leg is axis-aligned, so one component is
exactly zero, and both forms return `|dx|` exactly.

## Not in this slice (P4.9b.2)

* **The scenario half.** Ground stations, targets and their behaviours
  (`build_behavior_trajectory`, weather-adjusted), drones and their
  controllers, roles, mission zones, the POIs, the weather preset and the
  sensors, plus the assignment keys and `scenario_options` of
  `presentation`. The random draws they make go beyond what P4.3 ported
  (`default_rng` → `uniform`/`random`). `Generator.integers` is needed, and
  one behaviour draws `standard_normal`: numpy's ziggurat.
* **`argusnet-rs build`.** A command that writes a bundle from a config,
  once a whole bundle can be built.

## Next

**P4.9b.2**: the scenario half, and `argusnet-rs build --config`.
