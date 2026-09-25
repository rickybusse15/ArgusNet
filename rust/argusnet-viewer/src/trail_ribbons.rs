//! Trails as persistent ribbons, sampled by time and distance (P2.2).
//!
//! # Sampling that does not depend on how you got here
//!
//! [`TrailStore`] keeps each entity's trail from the **recorded** frames.
//! Every decision looks only at a frame and the one recorded before it, never
//! at the chain of samples kept so far. So playing up to a frame and seeking
//! straight to it give the same trail, and a test holds the two equal across
//! random seeks. The one exception is the global sample budget: if it bound
//! earlier in the window and was later relieved (entities left), playback has
//! already dropped samples a rebuild keeps. For each entity in each recorded
//! frame:
//!
//! * **absent** from the frame: its trail is removed (an ID that disappears
//!   leaves no trail behind to be mistaken for its successor's);
//! * **teleported** (the step from the previous frame implies more than
//!   [`TELEPORT_SPEED_MPS`], or it was absent there): the trail restarts;
//! * **sampled** when it enters a new [`SAMPLE_INTERVAL_S`] bucket of
//!   simulation time, or has moved more than [`SAMPLE_DISTANCE_M`] since the
//!   previous frame.
//!
//! Samples older than [`HORIZON_S`] are dropped, each trail keeps at most
//! [`MAX_SAMPLES_PER_TRAIL`], and past [`SAMPLE_BUDGET`] across all trails the
//! globally oldest go first. A new epoch clears everything.
//!
//! # Ribbons
//!
//! Each trail is one mesh: two crossed strips, horizontal and vertical, so it
//! reads from overhead and from low pitch alike without being rebuilt as the
//! camera turns. Colour runs from blue (low) to red (high) by altitude, and
//! alpha from faint (old) to full (new), as the gizmo trails did. The width
//! is a few screen pixels at the trail's newest point, recomputed only when
//! the view's scale moves by more than a quarter. A mesh is rewritten only
//! when its trail changed, which happens when the source frame does, never
//! per rendered frame. Trails end at the newest recorded sample, at most one
//! frame behind the interpolated glyph. They are unlit and additive (alpha
//! scales the light they add), depth-tested, write no depth, and carry
//! [`crate::sensor_volumes::PresentationOnly`], which picking excludes.

use std::collections::{BTreeMap, HashMap, VecDeque};

use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;

use crate::app::MainCamera;
use crate::glyphs::GlyphProjection;
use crate::presentation::{CurrentRuntimeMarkers, EntityKey};
use crate::replay::{MarkerKind, ReplayFrame, ReplayState};
use crate::sensor_volumes::PresentationOnly;
use crate::state::RuntimeOverlayVisibility;

pub const SAMPLE_INTERVAL_S: f32 = 0.5;
pub const SAMPLE_DISTANCE_M: f32 = 25.0;
pub const TELEPORT_SPEED_MPS: f32 = 200.0;
pub const HORIZON_S: f32 = 60.0;
pub const MAX_SAMPLES_PER_TRAIL: usize = 240;
pub const SAMPLE_BUDGET: usize = 20_000;
/// Ribbon width, in screen pixels at the trail's newest point.
pub const RIBBON_WIDTH_PX: f32 = 3.0;

/// One trail sample.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrailSample {
    pub t: f32,
    pub position: Vec3,
}

/// Every entity's trail. See the module docs.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct TrailStore {
    trails: BTreeMap<EntityKey, VecDeque<TrailSample>>,
    /// The epoch, frame index and frame count the store reflects.
    at: Option<(u64, usize, usize)>,
    /// Trails changed since the last [`Self::take_changed`].
    changed: Vec<EntityKey>,
}

impl TrailStore {
    pub fn trail(&self, key: EntityKey) -> Option<&VecDeque<TrailSample>> {
        self.trails.get(&key)
    }

    pub fn len(&self) -> usize {
        self.trails.len()
    }

    pub fn is_empty(&self) -> bool {
        self.trails.is_empty()
    }

    pub fn total_samples(&self) -> usize {
        self.trails.values().map(VecDeque::len).sum()
    }

    pub fn keys(&self) -> impl Iterator<Item = EntityKey> + '_ {
        self.trails.keys().copied()
    }

    /// Keys whose trails changed (or vanished) since the last call.
    pub fn take_changed(&mut self) -> Vec<EntityKey> {
        let mut changed = std::mem::take(&mut self.changed);
        changed.sort();
        changed.dedup();
        changed
    }

    /// Bring the store up to the replay's current frame: one step when it is
    /// the next frame, a rebuild over the horizon otherwise. Returns whether
    /// anything changed.
    pub fn update(&mut self, replay: &ReplayState, markers: &CurrentRuntimeMarkers) -> bool {
        let Some(document) = replay.document.as_ref() else {
            return self.clear_all();
        };
        if document.frames.is_empty() {
            return self.clear_all();
        }
        let index = replay.frame_index.min(document.frames.len() - 1);
        let at = (markers.epoch(), index, document.frames.len());
        if self.at == Some(at) {
            return false;
        }
        let next = match self.at {
            Some((epoch, previous, count)) => {
                epoch == at.0 && count == at.2 && index == previous + 1
            }
            None => false,
        };
        if next {
            self.step(
                &document.frames[index - 1],
                &document.frames[index],
                markers,
            );
        } else {
            self.rebuild(&document.frames, index, markers);
        }
        self.at = Some(at);
        true
    }

    fn clear_all(&mut self) -> bool {
        let had = !self.trails.is_empty() || self.at.is_some();
        self.changed.extend(self.trails.keys().copied());
        self.trails.clear();
        self.at = None;
        had
    }

    fn rebuild(&mut self, frames: &[ReplayFrame], index: usize, markers: &CurrentRuntimeMarkers) {
        self.changed.extend(self.trails.keys().copied());
        self.trails.clear();
        let now = frames[index].timestamp_s;
        let mut start = index;
        while start > 0 && frames[start - 1].timestamp_s >= now - HORIZON_S {
            start -= 1;
        }
        // The frame before the window, if any, is the first frame's "previous".
        let mut previous = start.checked_sub(1).map(|i| &frames[i]);
        for frame in &frames[start..=index] {
            match previous {
                Some(before) => self.step(before, frame, markers),
                None => self.first(frame, markers),
            }
            previous = Some(frame);
        }
    }

    /// The first frame of a replay: every entity starts a trail.
    fn first(&mut self, frame: &ReplayFrame, markers: &CurrentRuntimeMarkers) {
        for (key, position) in frame_positions(frame, markers) {
            let trail = self.trails.entry(key).or_default();
            trail.clear();
            trail.push_back(TrailSample {
                t: frame.timestamp_s,
                position,
            });
            self.changed.push(key);
        }
        self.trim(frame.timestamp_s);
    }

    fn step(&mut self, before: &ReplayFrame, frame: &ReplayFrame, markers: &CurrentRuntimeMarkers) {
        let previous: HashMap<EntityKey, Vec3> = frame_positions(before, markers).collect();
        let current: HashMap<EntityKey, Vec3> = frame_positions(frame, markers).collect();
        // Gone: no trail.
        let gone: Vec<EntityKey> = self
            .trails
            .keys()
            .filter(|key| !current.contains_key(key))
            .copied()
            .collect();
        for key in gone {
            self.trails.remove(&key);
            self.changed.push(key);
        }
        let dt = (frame.timestamp_s - before.timestamp_s).max(1e-6);
        let mut keys: Vec<EntityKey> = current.keys().copied().collect();
        keys.sort();
        for key in keys {
            let position = current[&key];
            let sample = TrailSample {
                t: frame.timestamp_s,
                position,
            };
            let trail = self.trails.entry(key).or_default();
            match previous.get(&key) {
                Some(before_position) => {
                    let step = position.distance(*before_position);
                    if step / dt > TELEPORT_SPEED_MPS {
                        trail.clear();
                        trail.push_back(sample);
                        self.changed.push(key);
                    } else {
                        let bucket = |t: f32| (t / SAMPLE_INTERVAL_S).floor() as i64;
                        // Local to this frame and the one before: never
                        // "the trail is empty", which depends on history.
                        if bucket(frame.timestamp_s) != bucket(before.timestamp_s)
                            || step > SAMPLE_DISTANCE_M
                        {
                            trail.push_back(sample);
                            self.changed.push(key);
                        }
                    }
                }
                // Appeared: a fresh trail.
                None => {
                    trail.clear();
                    trail.push_back(sample);
                    self.changed.push(key);
                }
            }
        }
        self.trim(frame.timestamp_s);
    }

    fn trim(&mut self, now: f32) {
        for (key, trail) in &mut self.trails {
            let before = trail.len();
            while trail.front().is_some_and(|s| s.t < now - HORIZON_S) {
                trail.pop_front();
            }
            while trail.len() > MAX_SAMPLES_PER_TRAIL {
                trail.pop_front();
            }
            if trail.len() != before {
                self.changed.push(*key);
            }
        }
        // Past the budget: the globally oldest samples go first, ties by key.
        let mut excess = self.total_samples().saturating_sub(SAMPLE_BUDGET);
        while excess > 0 {
            let oldest = self
                .trails
                .iter()
                .filter_map(|(key, trail)| trail.front().map(|s| (s.t, *key)))
                .min_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            let Some((_, key)) = oldest else {
                break;
            };
            if let Some(trail) = self.trails.get_mut(&key) {
                trail.pop_front();
            }
            self.changed.push(key);
            excess -= 1;
        }
    }
}

/// A frame's entities and their recorded positions, keyed as presented.
fn frame_positions<'a>(
    frame: &'a ReplayFrame,
    markers: &'a CurrentRuntimeMarkers,
) -> impl Iterator<Item = (EntityKey, Vec3)> + 'a {
    let tracks = frame
        .tracks
        .iter()
        .map(|t| (MarkerKind::Track, t.track_id.as_str(), t.position));
    let truths = frame
        .truths
        .iter()
        .map(|t| (MarkerKind::Truth, t.target_id.as_str(), t.position));
    let nodes = frame
        .nodes
        .iter()
        .map(|n| (MarkerKind::Node, n.node_id.as_str(), n.position));
    tracks
        .chain(truths)
        .chain(nodes)
        .filter_map(move |(kind, id, position)| {
            markers
                .key_of(kind, id)
                .map(|key| (key, Vec3::from_array(position)))
        })
}

/// The ribbon for one trail: a horizontal and a vertical strip along it.
/// `None` for fewer than two samples.
pub fn ribbon_mesh(
    trail: &VecDeque<TrailSample>,
    width_m: f32,
    z_range: (f32, f32),
) -> Option<Mesh> {
    if trail.len() < 2 {
        return None;
    }
    let n = trail.len();
    let half = width_m * 0.5;
    let (z_low, z_high) = z_range;
    let span = (z_high - z_low).max(1.0);
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(n * 4);
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity(n * 4);
    let mut colors: Vec<[f32; 4]> = Vec::with_capacity(n * 4);
    for (i, sample) in trail.iter().enumerate() {
        let p = sample.position;
        let neighbour = if i + 1 < n {
            trail[i + 1].position - p
        } else {
            p - trail[i - 1].position
        };
        let dir = neighbour.normalize_or(Vec3::X);
        let flat = Vec3::new(-dir.y, dir.x, 0.0).normalize_or(Vec3::Y);
        let t = i as f32 / (n - 1) as f32;
        let hue = 240.0 * (1.0 - ((p.z - z_low) / span).clamp(0.0, 1.0));
        let color = LinearRgba::from(Color::hsla(hue, 1.0, 0.55, 0.1 + 0.9 * t)).to_f32_array();
        for offset in [flat * half, -flat * half, Vec3::Z * half, -Vec3::Z * half] {
            positions.push((p + offset).to_array());
            normals.push([0.0, 0.0, 1.0]);
            colors.push(color);
        }
    }
    let mut indices: Vec<u32> = Vec::with_capacity((n - 1) * 12);
    for i in 0..(n - 1) as u32 {
        let (a, b) = (i * 4, (i + 1) * 4);
        // Horizontal strip (vertices 0, 1), then vertical (2, 3).
        for (l, r) in [(0, 1), (2, 3)] {
            indices.extend_from_slice(&[a + l, a + r, b + l, a + r, b + r, b + l]);
        }
    }
    Some(
        Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        )
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
        .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, colors)
        .with_inserted_indices(Indices::U32(indices)),
    )
}

/// Whether ribbon widths must be recomputed: the first time, or when the
/// view's scale has moved by more than a quarter since they last were.
pub fn needs_rescale(cut: Option<f32>, scale: f32) -> bool {
    cut.is_none_or(|cut| (scale / cut - 1.0).abs() > 0.25)
}

/// The ribbons' shared material: unlit vertex colour, both faces.
///
/// Additive, like the glyphs, volumes and walls, so every transparent layer
/// commutes. Blended, the ribbon's order against the glyph it ends at hung
/// on near-equal sort distances, and an unrelated mesh could flip it (P2.3).
pub fn ribbon_material() -> StandardMaterial {
    StandardMaterial {
        base_color: Color::WHITE,
        unlit: true,
        alpha_mode: AlphaMode::Add,
        double_sided: true,
        cull_mode: None,
        ..default()
    }
}

#[derive(Component, Debug, Clone, Copy)]
pub struct TrailRibbon {
    pub key: EntityKey,
}

/// The trail store and the ribbon entities drawing it.
#[derive(Resource, Default)]
pub struct TrailRibbons {
    pub store: TrailStore,
    entities: HashMap<EntityKey, (Entity, Handle<Mesh>)>,
    material: Option<Handle<StandardMaterial>>,
    /// The view scale (metres per pixel at the origin) widths were cut for.
    width_scale: Option<f32>,
    mesh_writes: u64,
}

impl TrailRibbons {
    pub fn ribbon_count(&self) -> usize {
        self.entities.len()
    }

    /// Ribbon meshes written so far.
    pub fn mesh_writes(&self) -> u64 {
        self.mesh_writes
    }
}

#[allow(clippy::too_many_arguments)]
pub fn sync_trail_ribbons(
    mut commands: Commands,
    markers: Res<CurrentRuntimeMarkers>,
    replay: Res<ReplayState>,
    visibility: Res<RuntimeOverlayVisibility>,
    camera: Query<(&GlobalTransform, &Projection, &Camera), With<MainCamera>>,
    mut ribbons: ResMut<TrailRibbons>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut shown: Query<&mut Visibility, With<TrailRibbon>>,
) {
    let ribbons = ribbons.as_mut();
    ribbons.store.update(&replay, &markers);
    let projection = camera
        .single()
        .map(|(t, p, c)| GlyphProjection::from_camera(t, p, c.logical_viewport_size().map(|s| s.y)))
        .unwrap_or_default();
    // Metres a pixel covers at the scene origin: the scale widths follow.
    let scale = projection.radius(Vec3::ZERO, 1.0).max(1e-6);
    let rescale = needs_rescale(ribbons.width_scale, scale);
    let mut changed = ribbons.store.take_changed();
    if rescale {
        ribbons.width_scale = Some(scale);
        changed = ribbons
            .store
            .keys()
            .chain(ribbons.entities.keys().copied())
            .collect();
        changed.sort();
        changed.dedup();
    }
    let visible = if visibility.trails {
        Visibility::Inherited
    } else {
        Visibility::Hidden
    };
    for mut v in &mut shown {
        v.set_if_neq(visible);
    }
    if changed.is_empty() {
        return;
    }
    let material = ribbons
        .material
        .get_or_insert_with(|| materials.add(ribbon_material()))
        .clone();
    let z_range = ribbons
        .store
        .trails
        .values()
        .flatten()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), s| {
            (lo.min(s.position.z), hi.max(s.position.z))
        });
    for key in changed {
        let mesh = ribbons.store.trail(key).and_then(|trail| {
            let head = trail.back().map_or(Vec3::ZERO, |s| s.position);
            ribbon_mesh(trail, projection.radius(head, RIBBON_WIDTH_PX), z_range)
        });
        match (mesh, ribbons.entities.get(&key).cloned()) {
            (Some(mesh), Some((_, handle))) => {
                if let Some(mut existing) = meshes.get_mut(&handle) {
                    *existing = mesh;
                }
                ribbons.mesh_writes += 1;
            }
            (Some(mesh), None) => {
                let handle = meshes.add(mesh);
                let entity = commands
                    .spawn((
                        TrailRibbon { key },
                        PresentationOnly,
                        Mesh3d(handle.clone()),
                        MeshMaterial3d(material.clone()),
                        Transform::IDENTITY,
                        visible,
                    ))
                    .id();
                ribbons.entities.insert(key, (entity, handle));
                ribbons.mesh_writes += 1;
            }
            (None, Some((entity, handle))) => {
                commands.entity(entity).despawn();
                meshes.remove(&handle);
                ribbons.entities.remove(&key);
            }
            (None, None) => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::replay::ReplayDocument;

    /// `n` frames, `dt` apart. drone-a moves steadily; drone-b exists only
    /// from frame 10 to 30; drone-c jumps 5 km at frame 20.
    fn replay(n: usize, dt: f32) -> ReplayState {
        let frames: Vec<serde_json::Value> = (0..n)
            .map(|f| {
                let t = f as f32 * dt;
                let mut nodes = vec![serde_json::json!({"node_id": "drone-a",
                    "position": [t * 10.0, 0.0, 100.0 + t], "velocity": [10.0, 0.0, 1.0], "is_mobile": true})];
                if (10..30).contains(&f) {
                    nodes.push(serde_json::json!({"node_id": "drone-b",
                        "position": [0.0, t * 5.0, 50.0], "velocity": [0.0, 5.0, 0.0], "is_mobile": true}));
                }
                let jump = if f >= 20 { 5000.0 } else { 0.0 };
                nodes.push(serde_json::json!({"node_id": "drone-c",
                    "position": [jump, 100.0 + t, 80.0], "velocity": [0.0, 1.0, 0.0], "is_mobile": true}));
                serde_json::json!({"timestamp_s": t, "tracks": [], "truths": [], "nodes": nodes})
            })
            .collect();
        let document: ReplayDocument =
            serde_json::from_value(serde_json::json!({ "frames": frames })).unwrap();
        ReplayState::new(Some(document))
    }

    fn at(
        replay: &mut ReplayState,
        markers: &mut CurrentRuntimeMarkers,
        store: &mut TrailStore,
        frame: usize,
    ) {
        replay.step_to(frame);
        markers.apply(replay);
        store.update(replay, markers);
    }

    /// Playing to a frame and seeking straight to it give the same trails.
    #[test]
    fn playback_and_seeking_agree() {
        let mut replay = replay(400, 0.25);
        let mut markers = CurrentRuntimeMarkers::default();
        let mut played = TrailStore::default();
        let mut seed = 0x9e37_79b9_u32;
        for frame in 0..400 {
            at(&mut replay, &mut markers, &mut played, frame);
            if frame % 37 == 0 || frame == 25 || frame == 399 {
                let mut sought = TrailStore::default();
                let mut other_markers = CurrentRuntimeMarkers::default();
                // Arrive by a detour: somewhere random first.
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                at(
                    &mut replay,
                    &mut other_markers,
                    &mut sought,
                    seed as usize % 400,
                );
                at(&mut replay, &mut other_markers, &mut sought, frame);
                // Keys are interned per marker table; compare by label.
                let by_label = |store: &TrailStore,
                                m: &CurrentRuntimeMarkers|
                 -> BTreeMap<String, Vec<TrailSample>> {
                    store
                        .keys()
                        .map(|k| {
                            (
                                m.ids().name(k.id()).to_string(),
                                store.trail(k).unwrap().iter().copied().collect(),
                            )
                        })
                        .collect()
                };
                assert_eq!(
                    by_label(&played, &markers),
                    by_label(&sought, &other_markers),
                    "frame {frame}"
                );
                replay.step_to(frame);
                markers.apply(&replay);
            }
        }
    }

    #[test]
    fn trails_sample_by_time_and_respect_the_horizon() {
        let mut replay = replay(400, 0.25);
        let mut markers = CurrentRuntimeMarkers::default();
        let mut store = TrailStore::default();
        for frame in 0..=399 {
            at(&mut replay, &mut markers, &mut store, frame);
        }
        let a = markers.key_of(MarkerKind::Node, "drone-a").unwrap();
        let trail = store.trail(a).unwrap();
        // 60 s at one sample per 0.5 s (drone-a moves 2.5 m a frame).
        assert!((119..=121).contains(&trail.len()), "{}", trail.len());
        assert!(trail.front().unwrap().t >= 99.75 - HORIZON_S);
        assert!(trail
            .iter()
            .zip(trail.iter().skip(1))
            .all(|(x, y)| y.t - x.t >= SAMPLE_INTERVAL_S - 1e-4));
    }

    #[test]
    fn a_teleport_restarts_and_a_disappearance_removes() {
        let mut replay = replay(40, 0.25);
        let mut markers = CurrentRuntimeMarkers::default();
        let mut store = TrailStore::default();
        for frame in 0..=25 {
            at(&mut replay, &mut markers, &mut store, frame);
        }
        let b = markers.key_of(MarkerKind::Node, "drone-b").unwrap();
        let c = markers.key_of(MarkerKind::Node, "drone-c").unwrap();
        assert!(store.trail(b).is_some());
        let trail_c = store.trail(c).unwrap();
        assert!(
            trail_c.iter().all(|s| s.position.x >= 4999.0),
            "restarted at the jump"
        );
        for frame in 26..=35 {
            at(&mut replay, &mut markers, &mut store, frame);
        }
        assert!(store.trail(b).is_none(), "drone-b left at frame 30");
    }

    #[test]
    fn the_sample_budget_holds_and_a_new_epoch_clears() {
        // 120 entities for 60 s at 0.5 s is 14,400 samples; 200 is 24,000,
        // past the budget.
        let frames: Vec<serde_json::Value> = (0..240)
            .map(|f| {
                let t = f as f32 * 0.5;
                let nodes: Vec<_> = (0..200)
                    .map(|i| serde_json::json!({"node_id": format!("d{i}"),
                        "position": [i as f32 * 100.0, t, 10.0], "velocity": [0.0, 1.0, 0.0], "is_mobile": true}))
                    .collect();
                serde_json::json!({"timestamp_s": t, "tracks": [], "truths": [], "nodes": nodes})
            })
            .collect();
        let mut replay = ReplayState::new(Some(
            serde_json::from_value(serde_json::json!({ "frames": frames })).unwrap(),
        ));
        let mut markers = CurrentRuntimeMarkers::default();
        let mut store = TrailStore::default();
        at(&mut replay, &mut markers, &mut store, 239);
        assert_eq!(store.total_samples(), SAMPLE_BUDGET);
        replay.begin_epoch(None);
        markers.apply(&replay);
        assert!(store.update(&replay, &markers));
        assert!(store.is_empty());
    }

    /// One node moving along x by `step(f)` per frame, `dt` apart.
    fn moving(n: usize, dt: f32, step: impl Fn(usize) -> f32) -> ReplayState {
        let mut x = 0.0;
        let frames: Vec<serde_json::Value> = (0..n)
            .map(|f| {
                if f > 0 {
                    x += step(f);
                }
                serde_json::json!({"timestamp_s": f as f32 * dt, "tracks": [], "truths": [],
                    "nodes": [{"node_id": "d", "position": [x, 0.0, 50.0], "velocity": [0.0, 0.0, 0.0], "is_mobile": true}]})
            })
            .collect();
        ReplayState::new(Some(
            serde_json::from_value(serde_json::json!({ "frames": frames })).unwrap(),
        ))
    }

    fn trail_after(mut replay: ReplayState, frames: usize) -> Vec<TrailSample> {
        let mut markers = CurrentRuntimeMarkers::default();
        let mut store = TrailStore::default();
        for frame in 0..frames {
            at(&mut replay, &mut markers, &mut store, frame);
        }
        let key = markers.key_of(MarkerKind::Node, "d").unwrap();
        store.trail(key).unwrap().iter().copied().collect()
    }

    /// A teleport is judged by speed, not distance: 750 m over 5 s is
    /// flying, 50 m in 0.1 s is not.
    #[test]
    fn a_teleport_is_a_speed_not_a_distance() {
        let fast = trail_after(moving(6, 5.0, |_| 750.0), 6);
        assert_eq!(fast.len(), 6, "a 150 m/s platform restarted its trail");
        let jump = trail_after(moving(10, 0.1, |f| if f == 5 { 50.0 } else { 1.0 }), 10);
        assert!(
            jump.iter().all(|s| s.position.x >= 54.0),
            "a 500 m/s jump kept its trail: {jump:?}"
        );
    }

    /// Within one time bucket, a step past the sample distance is sampled.
    #[test]
    fn a_long_step_is_sampled_within_a_bucket() {
        // Frames 0.2 s apart, 30 m each (150 m/s): every frame moves past
        // 25 m, though most stay in the same 0.5 s bucket.
        let trail = trail_after(moving(10, 0.2, |_| 30.0), 10);
        assert_eq!(trail.len(), 10);
        // And each trail is capped, however fast it samples: 400 frames at
        // 0.2 s is 80 s, one sample a frame, inside the horizon for 300.
        let trail = trail_after(moving(400, 0.2, |_| 30.0), 400);
        assert_eq!(trail.len(), MAX_SAMPLES_PER_TRAIL);
    }

    /// Every presentation layer is additive, so none of their draw orders
    /// matter; a blended ribbon made the order at its head visible.
    #[test]
    fn ribbons_are_additive_like_every_other_layer() {
        let material = ribbon_material();
        assert_eq!(material.alpha_mode, AlphaMode::Add);
        assert!(material.unlit);
        assert_eq!(material.cull_mode, None);
    }

    #[test]
    fn widths_are_recomputed_past_a_quarter_change_in_scale() {
        assert!(needs_rescale(None, 1.0));
        assert!(!needs_rescale(Some(1.0), 1.2));
        assert!(!needs_rescale(Some(1.0), 0.8));
        assert!(needs_rescale(Some(1.0), 1.3));
        assert!(needs_rescale(Some(1.0), 0.7));
    }

    #[test]
    fn a_ribbon_is_two_crossed_strips_fading_in() {
        let trail: VecDeque<TrailSample> = (0..5)
            .map(|i| TrailSample {
                t: i as f32,
                position: Vec3::new(i as f32 * 10.0, 0.0, 50.0),
            })
            .collect();
        let mesh = ribbon_mesh(&trail, 2.0, (0.0, 100.0)).unwrap();
        assert_eq!(mesh.count_vertices(), 20);
        assert_eq!(mesh.indices().unwrap().len(), 4 * 12);
        let Some(bevy::mesh::VertexAttributeValues::Float32x4(colors)) =
            mesh.attribute(Mesh::ATTRIBUTE_COLOR)
        else {
            panic!("colours");
        };
        assert!(colors[0][3] < colors[19][3], "older is fainter");
        // Altitude runs blue (low) to red (high).
        let climbing: VecDeque<TrailSample> = [0.0, 100.0]
            .iter()
            .enumerate()
            .map(|(i, z)| TrailSample {
                t: i as f32,
                position: Vec3::new(i as f32 * 10.0, 0.0, *z),
            })
            .collect();
        let mesh = ribbon_mesh(&climbing, 2.0, (0.0, 100.0)).unwrap();
        let Some(bevy::mesh::VertexAttributeValues::Float32x4(colors)) =
            mesh.attribute(Mesh::ATTRIBUTE_COLOR)
        else {
            panic!("colours");
        };
        assert!(colors[0][2] > colors[0][0], "low is blue: {:?}", colors[0]);
        assert!(colors[4][0] > colors[4][2], "high is red: {:?}", colors[4]);
        assert!(ribbon_mesh(&trail.iter().take(1).copied().collect(), 2.0, (0.0, 1.0)).is_none());
    }
}
