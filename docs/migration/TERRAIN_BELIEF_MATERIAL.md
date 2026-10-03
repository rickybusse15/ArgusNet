# P2.1: the terrain and belief material

**Status:** local 2026-09-24; merged 2026-10-01 in #122 (`badb1e1`). See [STATUS.md](STATUS.md).

DELIVERY_PLAN P2.1: *"Author an embedded WGSL material using Z-up
elevation/slope bands, anti-aliased contours and land-cover tint. Supply a
typed `BeliefRenderSnapshot` ... Sampling/packing lives in one adapter;
shaders MUST NOT infer coverage from truth terrain color or distance to a
drone. ... When a replay lacks belief data, render an explicit
unavailable/unknown state ... Upload only changed tiles, cap texture
dimensions/memory and expose failed/unavailable uploads. Validate pixel
probes against CPU belief queries at boundaries, stale/unknown cells and LOD
seams."*

The elevation and slope bands and the anti-aliased contours already existed.
This slice adds belief, land cover, the upload path, and the probes.

## Belief

**One adapter.** `belief_render::BeliefRenderSnapshot` takes
`argusnet-mapping`'s `BeliefTilePatch`es, P4.6's publication, and nothing
else. It is never given terrain, truth or drone positions, so neither it nor
the shader can infer coverage from them.

* **Class and confidence are the query's.** Each cell is judged at its
  centre, exactly as `WorldBeliefQuery::cell_ij` judges it. The last cell of a
  ragged grid is *outside the geofence*; zero coverage is *unknown*; then the
  0.6 and 0.4 obstacle-probability thresholds; confidence is
  `min(1, visits / 5) * (1 - p)`. A test compares every cell of a ragged
  fixture against the query.
* **Channels are kept apart.** Class and observed are `Rg8Uint`, read with
  `textureLoad` and never filtered. Confidence and last-observed time are
  `Rg32Float`, also loaded. Confidence is interpolated in the shader:
  bilinear between the four nearest cell centres **among observed cells
  only**, renormalised. `BeliefRenderSnapshot::sample` is the same arithmetic
  on the CPU. Time is never interpolated.
* **No seams.** Tiles are sub-rectangles of one texture per channel, so the
  interpolation reads across a tile boundary as it reads across any other.
  There is no halo to maintain.
* **Off the grid is no data, not the edge cell.** `GridBounds.xy_to_ij`
  clamps, which is right for a query and wrong for a pixel.
* **Unknown and unavailable are drawn, never faked.** An unobserved cell is a
  violet diagonal hatch. With the overlay on and no belief data at all
  (`BeliefLayer::Unavailable`, the state of every replay the repository ships
  today), the whole terrain is hatched, and the overlay panel says why:
  *"Belief unavailable: this source carries no belief data."* Stale cells
  (older than 60 simulation seconds) fade toward grey.
* **Caps.** 2,048 cells a side and 48 MiB across both channels. A larger
  grid is `Unavailable(TooLarge)`, with its dimensions in the message.
  Patches that do not fit the grid are refused and counted; patches older
  than a tile's current revision are ignored and counted.

**Uploads.** `terrain_rasters::TerrainRasters` creates each image whole when
its geometry appears or changes, as `RENDER_WORLD`-only assets. After that, a
changed tile is written straight into the existing GPU texture by a
render-world system (`RenderQueue::write_texture`). Editing the `Image`
asset would re-upload the whole texture. Uploads that cannot be made are
counted, not dropped: tiles written, tiles waiting for their texture, and
tiles refused (outside the texture, a wrong byte count, or waiting more than
600 frames). The overlay panel shows the counts.

## Land cover

A scene built from an environment now carries `metadata/land_cover.json`,
declared in the manifest's `metadata.land_cover` (additive). It holds the
bundle's own `land_cover` payload, plus the legend and each class's style
colour. `land_cover_raster::LandCoverRaster` rebuilds the runtime's
`argusnet_world::land_cover::LandCoverLayer` from it and samples that into an
`R8Uint` texture, so the class is the runtime's, by the runtime's rule. A
ragged last cell is sampled at its centre clamped into the bounds, since the
layer answers "no class" past the edge. The tint is deliberately subtle. A
scene without the file, including the committed one, gets no tint, and the
panel says so. A declared file that is missing or unreadable fails the load,
as a declared replay does.

## Headless renders now fail on a broken shader

Wiring this up surfaced a gap in the CI smoke test. A terrain shader that
failed to compile (a WGSL reserved word) still produced a plausible PNG, and
the smoke checks would have passed it. The headless readback now reports
pipelines that failed for good, and such a capture is refused with a non-zero
exit. So is a PNG that cannot be written, which used to log and exit 0.
Verified by reintroducing the reserved word: exit 1, no PNG.

## Verification

### Pixel probes

`tests/belief_probes.rs`, four real headless renders of the committed scene
(`ARGUSNET_GPU_TESTS=1`; the CI viewer job now runs them on lavapipe). Each
probe projects a world point through the live camera after capture and
records the 9×9 pixels around it. The checks are on which channel dominates,
and on the hatch's stripes, because lighting and tonemapping change absolute
colours.

| Render | What must hold |
| --- | --- |
| Belief over the scene in blocks | the CPU class at every probe is what the block says; the pixels are green (safe), red (obstacle), amber (uncertain), striped violet (unknown), desaturated green (stale), dark grey (outside, the ragged column) |
| Overlay on, no belief | every probe is the unavailable hatch |
| A revision arriving after the textures exist | the changed cells redraw (including an off-diagonal tile, so swapped tile origins are caught); exactly the revision's 4 tiles were written, none waiting or refused |
| A scene with a synthetic land-cover file | forest tints greener and water bluer than untinted open ground, where the runtime layer says they are |

The first probe run caught a real bug. The belief mode was written into the
fourth uniform lane, not the one the shader reads, so the overlay silently
drew nothing. A CPU test now pins the lane as well.

### Mutation round

21 mutants. 17 caught at first. Three survivors were test gaps and got
fixes: a land-cover extent that was not really ragged, a palette test that
did not reach a high class code, and a late-revision probe that could not
see swapped tile origins. All three are now caught. **One survivor
remains:** making the shader's confidence interpolation include unobserved
neighbours. At the probe resolution a cell is about 15 pixels, too coarse to
resolve the sub-cell blend near an unobserved cell. The rule is pinned on the
CPU (`confidence_interpolates_among_observed_cells_only`), and the shader
follows the same structure, but no rendered test distinguishes the two.

### Gate

| Check | Result |
| --- | --- |
| `cargo test --workspace`; GPU probes | 1131 passed, 0 failed; 4 passed |
| clippy `-D warnings` (workspace, and viewer without `live-stream`); `cargo fmt --check` | clean |
| `ruff check`, `ruff format --check`; `pytest tests/` (`ARGUSNET_REQUIRE_RS=1`) | clean; 1049 passed, 9 skipped |
| CI's headless smoke, locally | exit 0, 10,698 colours, 257 green track pixels: unchanged, since belief is off by default and the committed scene has no land cover |
| CI | **not run**: nothing is pushed |

## Not in this slice

* **Belief from a real run.** No replay carries belief, and the runtime's
  stage 8 does not yet publish `BeliefRevision`s to the viewer. The adapter's
  `apply_revision` is the seam P5.0's snapshot protocol plugs into.
* **LOD seams.** The terrain is one mesh at one level of detail today, so
  there is no seam to probe. Belief itself has no seams, by construction.
