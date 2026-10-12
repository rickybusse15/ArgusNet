# Terrain winding and exposure, 2026-10-12

Every scene package's terrain was wound facing down (clockwise seen from
above) while its normals pointed up. Its material is double-sided, so it drew,
but Bevy lit it as a back face: the normal was flipped, the 12 000 lux sun
contributed nothing, and the ground was lit by the ambient term alone. The
camera had been opened three stops (EV100 9) to make up for the dark ground,
which saturated every surface the sun did reach (buildings, and the drones'
self-built map, which was wound correctly) to white.

The fix, in one change:

- both scene writers (`src/argusnet/world/_scene_geometry.py`,
  `rust/argusnet-sim/src/scene_build.rs`) wind every terrain triangle
  counter-clockwise from above, `[a, b, c, b, d, c]` per grid quad;
- the committed `scene.smartscene/terrain/terrain-base.glb` has its index
  buffer rewritten the same way (every triangle's second and third index
  swapped; vertices untouched);
- both 3-D cameras use EV100 12, Bevy's `Exposure::OVERCAST`, which matches
  the key light, and the reconstruction camera uses the main camera's
  `AcesFitted` tonemapping.

Two things the dark ground had hidden, found by the GPU probes once it was lit:

- **Holograms were not additive.** `AlphaMode::Add` blends premultiplied (one,
  one-minus-source-alpha). `hologram.wgsl` returned its weight (0.85) as alpha,
  so every sensor volume and zone wall covered 85% of the ground behind it
  instead of adding light to it: a field-of-view sector darkened the sunlit
  ground it lay over (`geometry_probes`: "an additive volume only adds
  light"). Volumes and walls now write their weighted colour with alpha 0, as
  Bevy's own additive output does. Marker glyphs keep the covering blend on
  purpose (`HologramMaterial::glyph`): they are information and must keep their
  colour over bright ground.
- **The belief overlay took the ground's hue.** It is blended over the lit
  terrain with a partial weight, and warm sunlit ground pulled every class
  toward it (`belief_probes`: unknown hatch blue over red fell from at least
  1.21 to 1.02). Under the overlay the terrain now shows as its luminance at
  70% (`BELIEF_UNDERLAY`): relief still reads, the colour cannot shift a class.
  The smallest blue-over-red ratio is back to 1.23.

## Mean frame value (0-255), G2 fixtures, Standard tier, 650 x 480

| Fixture | Before (main) | Winding fixed, old exposure | After |
| --- | --- | --- | --- |
| overview | 50.1 | 121.4 | 68.0 |
| close | 97.4 | 229.8 | 133.8 |
| low_pitch | 60.4 | 69.6 | 63.9 |
| split | 40.0 | 79.6 | 51.3 |
| high_entity | 63.7 | 93.9 | 71.1 |

The middle column is the "pale" terrain: correct winding under the old
exposure. Exposures from EV100 10.5 to 12.5 were rendered; 12 matches the
light by name and keeps the buildings' lit and shaded faces apart.

Frame time is unchanged (software renderer, p50 ms before / after: overview
74.5 / 57.4, close 89.3 / 92.3, low pitch 53.9 / 48.1, split 82.0 / 78.1,
high entity 67.2 / 64.5; five frames each, so within run-to-run noise).

Stills, before on the left and after on the right, downscaled:
`overview-`, `close-`, `low_pitch-`, `split-` and `high_entity-before-after.png`.
`self-built-map-split.png` is the split view's reconstruction pane after a
180 s `--world-knowledge none` run on the regional map, at the new exposure.
