# Runtime Terrain Interface Contract

This document defines the terrain query API, generation entrypoint, role
separation, and caching strategy for terrain access across Python and Rust.

---

## 1. Role Separation: Visual vs Analytic Terrain

The codebase already maintains two terrain representations. This section makes
the boundary explicit and normative.

### 1.1 Visual Terrain (render path)

**Owner:** `tracker-viewer` (Bevy).

**Purpose:** Rendering only. Vertices, normals, UVs, texture coordinates,
level-of-detail meshes for the Bevy scene.

**Source:** `TerrainLayer.viewer_mesh()` in `environment.py` generates a
downsampled grid (capped at 128 × 128 by default) suitable for GPU upload.
The `.smartscene` package carries this as `terrain.viewer_mesh` JSON.

**Invariant:** Visual terrain MAY differ from analytic terrain in resolution
and smoothing. The viewer MUST NOT call analytic query functions for rendering
decisions. Physics and path planning are not the viewer's responsibility.

### 1.2 Analytic Terrain (simulation / planning / tracking path)

**Owner:** `environment.py` (`TerrainLayer`) at runtime.

**Construction:** `world/procedural.py` exposes `TerrainBuildConfig` and
`build_terrain_layer(config, bounds)`. Supported sources are:
- `procedural`: deterministic layered terrain by preset and seed.
- `dem`: GeoTIFF DEM promoted into a runtime `TerrainLayer`.
- `hybrid`: DEM base elevation plus deterministic procedural detail.

**Purpose:** Authoritative source for:
- height queries used by altitude clamping (`clamp_altitude`)
- gradient/slope used by path planning and obstacle detection
- terrain-intersection tests in LOS raycasts (`EnvironmentQuery._terrain_intersection`)
- minimum terrain clearance enforcement during simulation (`DynamicsConfig.terrain_following_agl_m`)
- communications shadow (LOS blocked by terrain)

**Invariant:** All physics, safety constraint evaluation, and sensor modelling
MUST use the analytic representation. Visual resolution is irrelevant to these
queries.

---

## 2. Existing Implementation Mapping

| Interface method (formal name below) | Current Python implementation | Notes |
|--------------------------------------|-------------------------------|-------|
| `height_at(x, y)` | `TerrainLayer.height_at` | Runtime scalar query, clamped to `ground_plane_m` |
| `height_at_many(xy)` | `TerrainLayer.height_at_many` | Runtime batch query over shape `(..., 2)` |
| `analytic_height_at(x, y)` | `TerrainModel.analytic_height_at` | Legacy construction helper |
| `gradient_at(x, y)` | `TerrainModel.gradient_at` / `TerrainLayer.gradient_at` | Central-difference, delta configurable |
| `gradient_at_many(xy)` | `TerrainLayer.gradient_at_many` | Batch central-difference query |
| `normal_at(x, y)` | `TerrainLayer.normal_at` | Derived from gradient |
| `curvature_at(x, y)` | `TerrainModel.curvature_at` / `TerrainLayer.curvature_at` | Laplacian approximation |
| `los_raycast(origin, target)` | `EnvironmentQuery._terrain_intersection` | Returns first hit |
| `comms_shadow(origin, target)` | `EnvironmentQuery.los` (`blocker_type == "terrain"`) | Folded into sensor LOS |
| `land_cover_at(x, y)` | `LandCoverLayer.land_cover_at` | Returns `LandCoverClass` enum |
| `clamp_altitude(xy, z, min_agl)` | `TerrainModel.clamp_altitude` / `TerrainLayer.clamp_altitude` | Both exist |

---

## 3. TerrainQuery Trait (Rust)

The Rust terrain engine provides a `TerrainQuery` trait and `GridTerrain`
backend that match Python `TerrainLayer` bilinear interpolation semantics.

```rust
/// Analytic terrain query interface.
///
/// Implementors provide point queries and segment intersection tests over
/// the local-ENU coordinate frame (metres, XY projected, Z above datum).
/// All coordinates are f64 metres; angles are radians.
pub trait TerrainQuery: Send + Sync {
    // -----------------------------------------------------------------------
    // Point queries
    // -----------------------------------------------------------------------

    /// Return ground elevation at (x_m, y_m), clamped to ground_plane_m.
    fn height_at(&self, x_m: f64, y_m: f64) -> f64;

    /// Return raw analytic elevation without floor clamping.
    /// Useful for curvature and feature evaluation below the ground plane.
    fn analytic_height_at(&self, x_m: f64, y_m: f64) -> f64;

    /// Return the terrain surface normal at (x_m, y_m) as a unit vector [nx, ny, nz].
    fn normal_at(&self, x_m: f64, y_m: f64) -> [f64; 3];

    /// Return the XY gradient [dz/dx, dz/dy] at (x_m, y_m) using
    /// central differences with the given finite-difference delta (metres).
    fn gradient_at(&self, x_m: f64, y_m: f64, delta_m: f64) -> [f64; 2];

    /// Return the scalar surface curvature (Laplacian approximation) at (x_m, y_m).
    /// Positive = convex (hilltop), negative = concave (valley).
    fn curvature_at(&self, x_m: f64, y_m: f64, delta_m: f64) -> f64;

    /// Return the slope magnitude (radians) at (x_m, y_m).
    fn slope_rad_at(&self, x_m: f64, y_m: f64) -> f64;

    /// Clamp altitude z_m so that it is at least min_agl_m above the terrain.
    fn clamp_altitude(&self, x_m: f64, y_m: f64, z_m: f64, min_agl_m: f64) -> f64;

    // -----------------------------------------------------------------------
    // Segment / LOS queries
    // -----------------------------------------------------------------------

    /// Test whether the segment from `origin` to `target` (each [x, y, z] metres)
    /// intersects the terrain surface, applying `clearance_m` as a vertical
    /// safety margin above the terrain.
    ///
    /// Returns `Some(TerrainHit)` at the first intersection, `None` if clear.
    fn los_raycast(
        &self,
        origin: [f64; 3],
        target: [f64; 3],
        clearance_m: f64,
    ) -> Option<TerrainHit>;

    /// Test whether the communications link between `tx_xyz` and `rx_xyz`
    /// is shadowed by terrain (ignores obstacles; terrain only).
    ///
    /// Returns `true` if terrain blocks the direct path.
    fn comms_shadow(&self, tx_xyz: [f64; 3], rx_xyz: [f64; 3]) -> bool;

    // -----------------------------------------------------------------------
    // Metadata
    // -----------------------------------------------------------------------

    /// Return the bounds of the terrain coverage area.
    fn bounds(&self) -> TerrainBounds;

    /// Return the ground-plane floor (absolute minimum elevation, metres).
    fn ground_plane_m(&self) -> f64;
}

/// Result of a terrain LOS intersection test.
#[derive(Clone, Debug)]
pub struct TerrainHit {
    /// Parametric distance along the segment [0.0, 1.0] where the hit occurred.
    pub t: f64,
    /// World-space hit point [x, y, z] (metres).
    pub point_m: [f64; 3],
    /// Terrain height at the hit XY location (metres, including clearance).
    pub terrain_z_m: f64,
}

/// Bounding box of terrain coverage.
#[derive(Clone, Debug)]
pub struct TerrainBounds {
    pub x_min_m: f64,
    pub x_max_m: f64,
    pub y_min_m: f64,
    pub y_max_m: f64,
    pub z_min_m: f64,
    pub z_max_m: f64,
}
```

---

## 4. Curvature

**Definition:** Laplacian approximation using the four-point stencil:

```
kappa(x, y) = (h(x+d,y) + h(x-d,y) + h(x,y+d) + h(x,y-d) - 4*h(x,y)) / d^2
```

where `d = delta_m`. Positive values indicate convex surfaces (hilltops);
negative values indicate concave surfaces (valleys).

This exists on both `TerrainModel` and `TerrainLayer`.

---

## 5. Caching Strategy

### 5.1 Existing mechanism

`TerrainLayer` holds an LRU tile cache keyed by `(lod, tx, ty)` and a
`viewer_mesh(max_dimension)` cache keyed by mesh dimension. Runtime hot paths
should prefer `height_at_many`, `gradient_at_many`, and `slope_rad_at_many`
instead of scalar loops.

### 5.2 Required additions

#### Terrain construction cache

Generated terrain is immutable after construction. Rebuild terrain when source,
preset, seed, DEM path, detail strength, resolution, bounds, or season changes.

#### Rust-side cache

Rust reads fixed-resolution grids through `GridTerrain`. Line-of-sight marching
uses adaptive sample counts based on segment length rather than a fixed sample
count.

### 5.3 Cache invalidation rules

| Event | Action |
|-------|--------|
| New `TerrainLayer` loaded | Discard tile and viewer-mesh caches |
| Terrain preset changed via CLI | Both of the above |
| Scene hot-reload in viewer | Re-bake Rust shared-memory grid |

---

## 6. Communications Shadow Semantics

A communications shadow query differs from a sensor LOS query in two ways:

1. **No obstacle attenuation.** Buildings and trees do not attenuate the
   datalink the same way they attenuate an optical sensor. A comms shadow is
   terrain-only (Earth surface masks the signal).
2. **Fresnel-zone margin.** The effective clearance for RF propagation is
   larger than the 1 m optical terrain clearance. The default Fresnel margin
   for the comms shadow check is `5.0 m` above the terrain surface.

The existing `EnvironmentQuery.los` folds both cases together. The new
`TerrainQuery.comms_shadow` is a separate, dedicated method so callers are
explicit about which check they need.

---

## 7. Out-of-Bounds Behaviour

Queries for coordinates outside `TerrainBounds`:

| Method | Behaviour |
|--------|-----------|
| `height_at` | Return `ground_plane_m` |
| `analytic_height_at` | Return `ground_plane_m` |
| `gradient_at` | Return `[0.0, 0.0]` |
| `normal_at` | Return `[0.0, 0.0, 1.0]` (flat up) |
| `curvature_at` | Return `0.0` |
| `los_raycast` | Return `Some(TerrainHit)` with `t=0.0` if origin is outside, else treat the out-of-bounds portion as unobstructed (same as `out_of_coverage` in current Python) |
| `comms_shadow` | Return `true` (conservative: assume shadowed when out of known area) |

This matches the conservative fallback in `EnvironmentQuery` where
`out_of_coverage` is treated as a hard blocker.
