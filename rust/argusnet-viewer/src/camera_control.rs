//! Camera behaviours for the main view (P2.4b): orbit, damped free-fly,
//! follow and chase of the selection, focus, bookmarks.
//!
//! # One step, explicit inputs
//!
//! [`CameraControl::step`] is the whole behaviour: given this frame's
//! actions, pointer look and scroll, the render delta, and what the world
//! says (the selection, where tracked entities are, the ground), it updates
//! the orbit and returns the camera's transform. The system around it only
//! gathers those inputs, so every behaviour is tested without a window.
//!
//! * **Render time.** Every rate and every damping constant uses the render
//!   delta (`Time<Real>`), never simulation time: pausing or speeding up a
//!   replay does not freeze or race the camera.
//! * **Damped.** Focus, follow, chase, bookmarks and reset move the orbit
//!   toward a goal exponentially, `1 - exp(-dt / tau)` a frame, which is the
//!   same path at any frame rate. Direct mouse orbiting stays immediate.
//! * **Free-fly** (`G`): `W A S D` along the view, `E`/`Q` up and down,
//!   right-drag to look, scroll to scale speed, `Shift` for four times. Speed
//!   follows height above the ground, from 5 m/s up to a quarter of the
//!   scene's span per second. The eye is kept 2 m over the ground, within one
//!   span of the scene, and under the orbit's ceiling, so the far plane,
//!   which is sized from that ceiling, always holds the scene. Pitch stops
//!   short of straight up and down. Leaving free-fly turns the view back
//!   into an orbit around the ground point it was looking at.
//! * **Follow** (`L`) keeps the orbit's focus on the selected entity; orbit
//!   and zoom still work around it. **Chase** (`C`) also swings the view
//!   behind its direction of travel. Losing the entity (it leaves the frame
//!   or the run changes) ends the mode, settles the camera on its last known
//!   position (never the origin), and says so.
//! * **Focus** (`F`) moves the orbit onto the selection and closes in to at
//!   most 600 m. **Bookmarks**: `1`-`9` recall, with `Ctrl` or `Cmd` held,
//!   save. **Reset** (`R`) returns home. Each of these leaves free-fly.
//! * **Escape** leaves a transient mode (free-fly, follow, chase) first; only
//!   an Escape the camera did not use clears the selection.
//! * **Focused viewport.** Keys move the main camera only while the pointer
//!   is over it or off the window, and never while the UI has the pointer.

use std::f32::consts::{PI, TAU};

use bevy::input::mouse::{MouseMotion, MouseWheel};
use bevy::prelude::*;

use crate::app::MainCamera;
use crate::bindings::{Action, ActionSet, Bindings, KeyboardGate};
use crate::orbit_camera::{OrbitCamera, MAX_PITCH_RAD, MIN_PITCH_RAD};
use crate::picking::PointerInput;
use crate::prefs::{Bookmark, ViewerPrefs};
use crate::presentation::{CurrentRuntimeMarkers, EntityKey};
use crate::schema::{Bounds2d, ScenePackage};
use crate::state::{LoadedMissionZones, SelectionState};

/// Time constant of the damped transitions, seconds.
pub const FOCUS_TAU_S: f32 = 0.18;
/// Time constant of free-fly acceleration, seconds.
pub const FLY_TAU_S: f32 = 0.15;
/// Focus closes the orbit in to at most this, metres.
pub const FOCUS_RADIUS_M: f32 = 600.0;
/// The orbit pitch chase settles at.
pub const CHASE_PITCH_RAD: f32 = 0.35;
/// Free-fly pitch stops this short of straight up or down.
pub const FLY_MAX_PITCH_RAD: f32 = 1.48;
/// Free-fly keeps the eye at least this far over the ground, metres.
pub const FLY_CLEARANCE_M: f32 = 2.0;
/// How long a camera notice stays on screen, seconds.
pub const NOTICE_S: f32 = 4.0;
/// Below this horizontal speed a chased entity keeps its last heading.
const HEADING_SPEED_MPS: f32 = 0.5;
const LOOK_RAD_PER_PX: f32 = 0.004;

/// `current` moved toward `goal` over `dt`: exponential, so two half steps
/// land exactly where one whole step does.
pub fn approach(current: f32, goal: f32, dt: f32, tau: f32) -> f32 {
    goal + (current - goal) * (-dt.max(0.0) / tau).exp()
}

/// As [`approach`], the short way round the circle.
pub fn approach_angle(current: f32, goal: f32, dt: f32, tau: f32) -> f32 {
    let delta = (goal - current + PI).rem_euclid(TAU) - PI;
    current + delta * (1.0 - (-dt.max(0.0) / tau).exp())
}

/// An orbit pose: what a bookmark holds and a goal aims at.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrbitPose {
    pub focus: Vec3,
    pub radius: f32,
    pub yaw: f32,
    pub pitch: f32,
}

impl OrbitPose {
    pub fn of(orbit: &OrbitCamera) -> Self {
        Self {
            focus: orbit.focus,
            radius: orbit.radius,
            yaw: orbit.yaw,
            pitch: orbit.pitch,
        }
    }

    pub fn from_bookmark(bookmark: &Bookmark) -> Self {
        Self {
            focus: Vec3::from_array(bookmark.focus),
            radius: bookmark.radius,
            yaw: bookmark.yaw,
            pitch: bookmark.pitch,
        }
    }

    pub fn to_bookmark(self) -> Bookmark {
        Bookmark {
            focus: self.focus.to_array(),
            radius: self.radius,
            yaw: self.yaw,
            pitch: self.pitch,
        }
    }
}

/// The orbit that puts the eye at `eye` looking at `focus`, with its pitch
/// clamped to the orbit's range.
pub fn orbit_from_eye(eye: Vec3, focus: Vec3) -> OrbitPose {
    let offset = eye - focus;
    let radius = offset.length().max(1e-3);
    OrbitPose {
        focus,
        radius,
        yaw: offset.y.atan2(offset.x),
        pitch: (offset.z / radius)
            .clamp(-1.0, 1.0)
            .asin()
            .clamp(MIN_PITCH_RAD, MAX_PITCH_RAD),
    }
}

/// Free-fly speed, m/s: height above the ground, between 5 m/s and a
/// quarter of the scene's span, times the scroll multiplier, four times
/// with `Faster` held.
pub fn fly_speed(altitude_m: f32, span_m: f32, multiplier: f32, faster: bool) -> f32 {
    let ceiling = (span_m * 0.25).max(5.0);
    altitude_m.clamp(5.0, ceiling) * multiplier * if faster { 4.0 } else { 1.0 }
}

/// Where free-fly may put the eye: over the ground, within a span of the
/// scene, under `ceiling_m`.
pub fn clamp_fly(
    position: Vec3,
    ground_m: f32,
    bounds: &Bounds2d,
    span_m: f32,
    ceiling_m: f32,
) -> Vec3 {
    Vec3::new(
        position
            .x
            .clamp(bounds.x_min_m - span_m, bounds.x_max_m + span_m),
        position
            .y
            .clamp(bounds.y_min_m - span_m, bounds.y_max_m + span_m),
        position
            .z
            .max(ground_m + FLY_CLEARANCE_M)
            .min(ceiling_m.max(ground_m + FLY_CLEARANCE_M)),
    )
}

/// The unit direction of a yaw and pitch, Z up.
pub fn fly_direction(yaw: f32, pitch: f32) -> Vec3 {
    Vec3::new(
        pitch.cos() * yaw.cos(),
        pitch.cos() * yaw.sin(),
        pitch.sin(),
    )
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CameraMode {
    Orbit,
    FreeFly,
    Follow(EntityKey),
    Chase(EntityKey),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Fly {
    pub position: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub velocity: Vec3,
    /// Scroll's speed scale, 0.1 to 10.
    pub multiplier: f32,
}

/// What a goal still has to reach. Fields not set are left alone.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct Goal {
    focus: Option<Vec3>,
    radius: Option<f32>,
    yaw: Option<f32>,
    pitch: Option<f32>,
}

/// A tracked entity, as the camera needs it.
#[derive(Debug, Clone, PartialEq)]
pub struct Tracked {
    pub position: Vec3,
    pub velocity: Vec3,
    /// Shared with the marker table: tracking every frame copies nothing.
    pub label: std::sync::Arc<str>,
}

/// This frame's input to the camera.
#[derive(Debug, Clone, Default)]
pub struct CameraInputs {
    /// Render delta, seconds.
    pub dt: f32,
    /// Actions whose key went down this frame.
    pub pressed: ActionSet,
    /// Actions whose key is held.
    pub held: ActionSet,
    /// Ctrl or Cmd is held: a bookmark key saves.
    pub save_modifier: bool,
    /// Right-drag this frame, pixels (free-fly look).
    pub look: Vec2,
    /// Scroll this frame, notches (free-fly speed).
    pub scroll: f32,
    /// The pointer is over the main view, or off the window, and not on the
    /// UI: keys and look move this camera.
    pub over_view: bool,
}

/// What the camera may ask of the world.
pub struct CameraWorld<'a> {
    pub selection: Option<EntityKey>,
    pub epoch: u64,
    pub tracked: &'a dyn Fn(EntityKey) -> Option<Tracked>,
    pub ground: &'a dyn Fn(f32, f32) -> f32,
    pub bookmark: &'a dyn Fn(u8) -> Option<OrbitPose>,
    pub bounds: Bounds2d,
    pub span_m: f32,
}

/// What a step produced besides the orbit.
#[derive(Debug, Clone, PartialEq)]
pub struct CameraOutput {
    pub transform: Transform,
    /// The camera used this frame's Escape; the selection keeps it.
    pub escape_used: bool,
    /// Save this pose to a bookmark slot.
    pub save_bookmark: Option<(u8, OrbitPose)>,
}

/// The main camera's behaviour state. See the module docs.
#[derive(Resource, Debug, Clone)]
pub struct CameraControl {
    pub mode: CameraMode,
    pub home: OrbitPose,
    goal: Goal,
    fly: Fly,
    /// The run a follow or chase began in.
    epoch: u64,
    tracked_label: std::sync::Arc<str>,
    last_known: Option<Vec3>,
    heading: f32,
    notice: Option<(String, f32)>,
}

impl CameraControl {
    pub fn new(home: OrbitPose) -> Self {
        Self {
            mode: CameraMode::Orbit,
            home,
            goal: Goal::default(),
            fly: Fly {
                position: Vec3::ZERO,
                yaw: 0.0,
                pitch: 0.0,
                velocity: Vec3::ZERO,
                multiplier: 1.0,
            },
            epoch: 0,
            tracked_label: std::sync::Arc::from(""),
            last_known: None,
            heading: 0.0,
            notice: None,
        }
    }

    pub fn fly(&self) -> &Fly {
        &self.fly
    }

    /// A short line for the status bar.
    pub fn mode_label(&self) -> String {
        match self.mode {
            CameraMode::Orbit => "Orbit".into(),
            CameraMode::FreeFly => format!("Free-fly ×{:.1}", self.fly.multiplier),
            CameraMode::Follow(_) => format!("Following {}", self.tracked_label),
            CameraMode::Chase(_) => format!("Chasing {}", self.tracked_label),
        }
    }

    /// The current notice, if one is still showing.
    pub fn notice(&self) -> Option<&str> {
        self.notice.as_ref().map(|(text, _)| text.as_str())
    }

    fn say(&mut self, text: impl Into<String>) {
        self.notice = Some((text.into(), NOTICE_S));
    }

    fn enter_free_fly(&mut self, orbit: &OrbitCamera) {
        let eye = orbit.eye_position();
        let direction = (orbit.focus - eye).normalize_or(Vec3::NEG_Z);
        self.fly = Fly {
            position: eye,
            yaw: direction.y.atan2(direction.x),
            pitch: direction
                .z
                .clamp(-1.0, 1.0)
                .asin()
                .clamp(-FLY_MAX_PITCH_RAD, FLY_MAX_PITCH_RAD),
            velocity: Vec3::ZERO,
            multiplier: self.fly.multiplier,
        };
        self.goal = Goal::default();
        self.mode = CameraMode::FreeFly;
    }

    /// The orbit around what free-fly was looking at: the ground point on the
    /// view ray, or the previous orbit distance ahead when looking up. The
    /// focus stays on that point; only the radius is held to the orbit's
    /// range, so an eye closer to the ground than the orbit's minimum backs
    /// off along the ray rather than pushing the focus underground.
    fn fly_as_orbit(&self, orbit: &OrbitCamera, world: &CameraWorld) -> OrbitPose {
        let direction = fly_direction(self.fly.yaw, self.fly.pitch);
        let eye = self.fly.position;
        let ground = (world.ground)(eye.x, eye.y);
        let distance = if direction.z < -0.02 {
            ((eye.z - ground) / -direction.z).min(orbit.max_radius)
        } else {
            orbit.radius
        };
        let focus = eye + direction * distance;
        let radius = distance.clamp(orbit.min_radius, orbit.max_radius);
        orbit_from_eye(focus - direction * radius, focus)
    }

    fn leave_free_fly(&mut self, orbit: &mut OrbitCamera, world: &CameraWorld) {
        if self.mode != CameraMode::FreeFly {
            return;
        }
        let pose = self.fly_as_orbit(orbit, world);
        orbit.focus = pose.focus;
        orbit.radius = pose.radius.clamp(orbit.min_radius, orbit.max_radius);
        orbit.yaw = pose.yaw;
        orbit.pitch = pose.pitch;
        self.mode = CameraMode::Orbit;
    }

    fn aim(&mut self, pose: OrbitPose) {
        self.goal = Goal {
            focus: Some(pose.focus),
            radius: Some(pose.radius),
            yaw: Some(pose.yaw),
            pitch: Some(pose.pitch),
        };
    }

    fn track(
        &mut self,
        mode: CameraMode,
        key: EntityKey,
        world: &CameraWorld,
        orbit: &mut OrbitCamera,
    ) {
        let Some(tracked) = (world.tracked)(key) else {
            self.say("Nothing selected to follow");
            return;
        };
        self.leave_free_fly(orbit, world);
        self.mode = mode;
        self.epoch = world.epoch;
        self.tracked_label = tracked.label.clone();
        self.last_known = Some(tracked.position);
        self.heading = orbit.yaw - PI;
        let verb = if matches!(mode, CameraMode::Chase(_)) {
            "Chasing"
        } else {
            "Following"
        };
        self.say(format!("{verb} {}", tracked.label));
    }

    /// One frame. See the module docs.
    pub fn step(
        &mut self,
        input: &CameraInputs,
        world: &CameraWorld,
        orbit: &mut OrbitCamera,
    ) -> CameraOutput {
        let dt = input.dt.max(0.0);
        if let Some((_, left)) = self.notice.as_mut() {
            *left -= dt;
            if *left <= 0.0 {
                self.notice = None;
            }
        }
        let pressed = |action| input.pressed.contains(action);
        let mut output = CameraOutput {
            transform: Transform::IDENTITY,
            escape_used: false,
            save_bookmark: None,
        };

        if pressed(Action::Escape) && self.mode != CameraMode::Orbit {
            self.leave_free_fly(orbit, world);
            self.mode = CameraMode::Orbit;
            output.escape_used = true;
        }
        if pressed(Action::FreeFly) {
            if self.mode == CameraMode::FreeFly {
                self.leave_free_fly(orbit, world);
            } else {
                self.enter_free_fly(orbit);
            }
        }
        if pressed(Action::ResetCamera) {
            self.leave_free_fly(orbit, world);
            self.mode = CameraMode::Orbit;
            self.aim(self.home);
        }
        if pressed(Action::Focus) {
            match world.selection.and_then(|key| (world.tracked)(key)) {
                Some(tracked) => {
                    self.leave_free_fly(orbit, world);
                    if self.mode == CameraMode::FreeFly {
                        self.mode = CameraMode::Orbit;
                    }
                    self.goal.focus = Some(tracked.position);
                    self.goal.radius = Some(orbit.radius.min(FOCUS_RADIUS_M).max(orbit.min_radius));
                }
                None => self.say("Nothing selected to focus"),
            }
        }
        for (action, chase) in [(Action::Follow, false), (Action::Chase, true)] {
            if !pressed(action) {
                continue;
            }
            let active = match self.mode {
                CameraMode::Follow(_) => !chase,
                CameraMode::Chase(_) => chase,
                _ => false,
            };
            if active {
                self.mode = CameraMode::Orbit;
                continue;
            }
            match world.selection {
                Some(key) => {
                    let mode = if chase {
                        CameraMode::Chase(key)
                    } else {
                        CameraMode::Follow(key)
                    };
                    self.track(mode, key, world, orbit);
                    if chase {
                        self.goal.pitch = Some(CHASE_PITCH_RAD);
                    }
                }
                None => self.say("Nothing selected to follow"),
            }
        }
        for slot in 1..=9 {
            if !pressed(Action::Bookmark(slot)) {
                continue;
            }
            if input.save_modifier {
                let pose = if self.mode == CameraMode::FreeFly {
                    self.fly_as_orbit(orbit, world)
                } else {
                    OrbitPose::of(orbit)
                };
                output.save_bookmark = Some((slot, pose));
                self.say(format!("Saved bookmark {slot}"));
            } else if let Some(pose) = (world.bookmark)(slot) {
                self.leave_free_fly(orbit, world);
                self.mode = CameraMode::Orbit;
                self.aim(pose);
            } else {
                self.say(format!("Bookmark {slot} is empty: Ctrl+{slot} saves one"));
            }
        }

        // Follow and chase: keep the focus on the entity, or lose it.
        if let CameraMode::Follow(key) | CameraMode::Chase(key) = self.mode {
            let found = (world.epoch == self.epoch)
                .then(|| (world.tracked)(key))
                .flatten();
            match found {
                Some(tracked) => {
                    self.last_known = Some(tracked.position);
                    self.goal.focus = Some(tracked.position);
                    if matches!(self.mode, CameraMode::Chase(_)) {
                        let ground_speed = tracked.velocity.truncate().length();
                        if ground_speed > HEADING_SPEED_MPS {
                            self.heading = tracked.velocity.y.atan2(tracked.velocity.x);
                        }
                        self.goal.yaw = Some(self.heading + PI);
                    }
                }
                None => {
                    let label = std::sync::Arc::clone(&self.tracked_label);
                    self.mode = CameraMode::Orbit;
                    self.goal = Goal {
                        focus: self.last_known,
                        ..Goal::default()
                    };
                    self.say(format!("Lost {label}: holding its last known position"));
                }
            }
        }

        if self.mode == CameraMode::FreeFly {
            self.fly_step(input, world, orbit);
            let direction = fly_direction(self.fly.yaw, self.fly.pitch);
            output.transform =
                Transform::from_translation(self.fly.position).looking_to(direction, Vec3::Z);
            return output;
        }

        // Damped orbit goals. Settled fields are dropped; follow and chase
        // set theirs again every frame.
        let goal = self.goal;
        if let Some(focus) = goal.focus {
            orbit.focus = Vec3::new(
                approach(orbit.focus.x, focus.x, dt, FOCUS_TAU_S),
                approach(orbit.focus.y, focus.y, dt, FOCUS_TAU_S),
                approach(orbit.focus.z, focus.z, dt, FOCUS_TAU_S),
            );
            if orbit.focus.distance(focus) < 0.01 {
                self.goal.focus = None;
            }
        }
        if let Some(radius) = goal.radius {
            orbit.radius = approach(orbit.radius, radius, dt, FOCUS_TAU_S)
                .clamp(orbit.min_radius, orbit.max_radius);
            if (orbit.radius - radius).abs() < 0.01 {
                self.goal.radius = None;
            }
        }
        if let Some(yaw) = goal.yaw {
            orbit.yaw = approach_angle(orbit.yaw, yaw, dt, FOCUS_TAU_S);
            if ((yaw - orbit.yaw + PI).rem_euclid(TAU) - PI).abs() < 1e-4 {
                self.goal.yaw = None;
            }
        }
        if let Some(pitch) = goal.pitch {
            orbit.pitch =
                approach(orbit.pitch, pitch, dt, FOCUS_TAU_S).clamp(MIN_PITCH_RAD, MAX_PITCH_RAD);
            if (orbit.pitch - pitch).abs() < 1e-4 {
                self.goal.pitch = None;
            }
        }
        output.transform =
            Transform::from_translation(orbit.eye_position()).looking_at(orbit.focus, Vec3::Z);
        output
    }

    fn fly_step(&mut self, input: &CameraInputs, world: &CameraWorld, orbit: &OrbitCamera) {
        let dt = input.dt.max(0.0);
        let held = |action| input.over_view && input.held.contains(action);
        if input.over_view {
            self.fly.yaw -= input.look.x * LOOK_RAD_PER_PX;
            self.fly.pitch = (self.fly.pitch - input.look.y * LOOK_RAD_PER_PX)
                .clamp(-FLY_MAX_PITCH_RAD, FLY_MAX_PITCH_RAD);
            self.fly.multiplier =
                (self.fly.multiplier * 1.25_f32.powf(input.scroll)).clamp(0.1, 10.0);
        }
        let forward = fly_direction(self.fly.yaw, self.fly.pitch);
        let right = forward.cross(Vec3::Z).normalize_or(Vec3::X);
        let axis = |plus, minus| f32::from(u8::from(held(plus))) - f32::from(u8::from(held(minus)));
        let wish = forward * axis(Action::MoveForward, Action::MoveBack)
            + right * axis(Action::MoveRight, Action::MoveLeft)
            + Vec3::Z * axis(Action::MoveUp, Action::MoveDown);
        let ground = (world.ground)(self.fly.position.x, self.fly.position.y);
        let speed = fly_speed(
            self.fly.position.z - ground,
            world.span_m,
            self.fly.multiplier,
            held(Action::Faster),
        );
        let target = wish.normalize_or_zero() * speed;
        let blend = 1.0 - (-dt / FLY_TAU_S).exp();
        self.fly.velocity += (target - self.fly.velocity) * blend;
        let moved = self.fly.position + self.fly.velocity * dt;
        let ground = (world.ground)(moved.x, moved.y);
        self.fly.position = clamp_fly(
            moved,
            ground,
            &world.bounds,
            world.span_m,
            ground + orbit.max_radius,
        );
    }
}

/// Set when the camera used this frame's Escape, so the selection keeps it.
#[derive(Resource, Debug, Default, Clone, Copy, PartialEq)]
pub struct EscapeUsed(pub bool);

/// The input side of the camera system.
#[derive(bevy::ecs::system::SystemParam)]
pub struct CameraInputParams<'w> {
    time: Res<'w, Time<Real>>,
    keys: Res<'w, ButtonInput<KeyCode>>,
    buttons: Res<'w, ButtonInput<MouseButton>>,
    gate: Res<'w, KeyboardGate>,
    bindings: Res<'w, Bindings>,
    pointer: Res<'w, PointerInput>,
}

/// What the camera system reads of the world.
#[derive(bevy::ecs::system::SystemParam)]
pub struct CameraWorldParams<'w> {
    markers: Res<'w, CurrentRuntimeMarkers>,
    selection: Res<'w, SelectionState>,
    zones: Res<'w, LoadedMissionZones>,
    scene: Res<'w, ScenePackage>,
}

/// Gathers this frame's inputs, steps [`CameraControl`], and places the main
/// camera. Saves a bookmark when asked.
#[allow(clippy::too_many_arguments)]
pub fn drive_main_camera_system(
    inputs: CameraInputParams,
    mut motion: MessageReader<MouseMotion>,
    mut wheel: MessageReader<MouseWheel>,
    reads: CameraWorldParams,
    mut prefs: ResMut<ViewerPrefs>,
    mut control: ResMut<CameraControl>,
    mut escape: ResMut<EscapeUsed>,
    mut camera: Query<(&mut OrbitCamera, &mut Transform, &Camera), With<MainCamera>>,
) {
    let CameraInputParams {
        time,
        keys,
        buttons,
        gate,
        bindings,
        pointer,
    } = inputs;
    let CameraWorldParams {
        markers,
        selection,
        zones,
        scene,
    } = reads;
    let Ok((mut orbit, mut transform, camera)) = camera.single_mut() else {
        return;
    };
    let over_view = !pointer.over_ui
        && pointer.position.is_none_or(|p| {
            camera
                .logical_viewport_rect()
                .is_some_and(|rect| crate::picking::viewport_contains(rect, p))
        });
    let drag: Vec2 = motion.read().map(|m| m.delta).sum();
    let scroll: f32 = wheel.read().map(|w| w.y).sum();
    let free_fly = control.mode == CameraMode::FreeFly;
    let input = CameraInputs {
        dt: time.delta_secs(),
        pressed: Action::ALL
            .iter()
            .copied()
            .filter(|a| bindings.just_pressed(&gate, &keys, *a))
            .collect(),
        held: Action::ALL
            .iter()
            .copied()
            .filter(|a| bindings.held(&gate, &keys, *a))
            .collect(),
        save_modifier: keys.any_pressed([
            KeyCode::ControlLeft,
            KeyCode::ControlRight,
            KeyCode::SuperLeft,
            KeyCode::SuperRight,
        ]),
        look: if free_fly && buttons.pressed(MouseButton::Right) {
            drag
        } else {
            Vec2::ZERO
        },
        scroll: if free_fly { scroll } else { 0.0 },
        over_view,
    };
    let tracked = |key: EntityKey| {
        markers.get(key).map(|m| Tracked {
            position: m.position,
            velocity: m.velocity,
            label: std::sync::Arc::clone(&m.label),
        })
    };
    let floor = scene
        .environment
        .terrain_summary
        .as_ref()
        .and_then(|s| s.min_height_m)
        .unwrap_or(0.0);
    let ground = |x: f32, y: f32| {
        zones
            .terrain_mesh
            .as_ref()
            .and_then(|m| m.height_at(x, y))
            .unwrap_or(floor)
    };
    let bookmark = |slot: u8| prefs.bookmark(slot).map(|b| OrbitPose::from_bookmark(&b));
    let bounds = scene.environment.bounds_xy_m.clone();
    let span = bounds.span_xy();
    let world = CameraWorld {
        selection: selection
            .selected
            .as_ref()
            .filter(|s| s.epoch == markers.epoch())
            .map(|s| s.key),
        epoch: markers.epoch(),
        tracked: &tracked,
        ground: &ground,
        bookmark: &bookmark,
        bounds,
        span_m: span[0].max(span[1]),
    };
    let output = control.step(&input, &world, &mut orbit);
    transform.set_if_neq(output.transform);
    escape.set_if_neq(EscapeUsed(output.escape_used));
    if let Some((slot, pose)) = output.save_bookmark {
        if let Err(error) = prefs.set_bookmark(slot, pose.to_bookmark()) {
            control.say(format!("Could not save bookmark {slot}: {error}"));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::presentation::IdIndex;

    fn bounds() -> Bounds2d {
        Bounds2d {
            x_min_m: -2000.0,
            x_max_m: 2000.0,
            y_min_m: -2000.0,
            y_max_m: 2000.0,
        }
    }

    fn orbit() -> OrbitCamera {
        OrbitCamera {
            focus: Vec3::new(0.0, 0.0, 10.0),
            radius: 3000.0,
            yaw: 0.8,
            pitch: 1.0,
            min_radius: 40.0,
            max_radius: 20_000.0,
        }
    }

    fn key(n: usize) -> EntityKey {
        EntityKey::Platform(IdIndex::for_tests(n))
    }

    /// A world with one drone at `at` moving east, selectable as `key(1)`.
    struct Scene {
        drone: Option<Vec3>,
        velocity: Vec3,
        epoch: u64,
        selected: bool,
        bookmarks: Vec<(u8, OrbitPose)>,
    }

    impl Scene {
        fn new() -> Self {
            Self {
                drone: Some(Vec3::new(500.0, -300.0, 120.0)),
                velocity: Vec3::new(10.0, 0.0, 0.0),
                epoch: 1,
                selected: true,
                bookmarks: Vec::new(),
            }
        }

        fn step(
            &self,
            control: &mut CameraControl,
            orbit: &mut OrbitCamera,
            input: &CameraInputs,
        ) -> CameraOutput {
            let tracked = |k: EntityKey| {
                (k == key(1))
                    .then_some(())
                    .and(self.drone)
                    .map(|p| Tracked {
                        position: p,
                        velocity: self.velocity,
                        label: std::sync::Arc::from("drone-1"),
                    })
            };
            let ground = |_: f32, _: f32| 5.0;
            let bookmark = |slot: u8| {
                self.bookmarks
                    .iter()
                    .find(|(s, _)| *s == slot)
                    .map(|(_, p)| *p)
            };
            let world = CameraWorld {
                selection: self.selected.then_some(key(1)),
                epoch: self.epoch,
                tracked: &tracked,
                ground: &ground,
                bookmark: &bookmark,
                bounds: bounds(),
                span_m: 4000.0,
            };
            control.step(input, &world, orbit)
        }
    }

    fn press(actions: &[Action]) -> CameraInputs {
        CameraInputs {
            dt: 1.0 / 60.0,
            pressed: actions.iter().copied().collect(),
            over_view: true,
            ..Default::default()
        }
    }

    fn idle(dt: f32) -> CameraInputs {
        CameraInputs {
            dt,
            over_view: true,
            ..Default::default()
        }
    }

    fn settle(
        scene: &Scene,
        control: &mut CameraControl,
        orbit: &mut OrbitCamera,
        seconds: f32,
    ) -> CameraOutput {
        let mut out = scene.step(control, orbit, &idle(0.0));
        for _ in 0..(seconds * 60.0) as usize {
            out = scene.step(control, orbit, &idle(1.0 / 60.0));
        }
        out
    }

    /// Damping is exponential in render time: the same path at any frame
    /// rate, and the short way round for angles.
    #[test]
    fn damping_is_frame_rate_independent() {
        let one = approach(0.0, 10.0, 0.1, FOCUS_TAU_S);
        let two = approach(
            approach(0.0, 10.0, 0.05, FOCUS_TAU_S),
            10.0,
            0.05,
            FOCUS_TAU_S,
        );
        assert!((one - two).abs() < 1e-5, "{one} vs {two}");
        assert!(one > 0.0 && one < 10.0);
        assert_eq!(approach(3.0, 10.0, 0.0, FOCUS_TAU_S), 3.0);
        let a = approach_angle(3.0, -3.0, 0.1, FOCUS_TAU_S);
        assert!(a > 3.0, "went the long way: {a}");
    }

    #[test]
    fn orbit_from_eye_inverts_eye_position() {
        let o = orbit();
        let pose = orbit_from_eye(o.eye_position(), o.focus);
        assert!((pose.radius - o.radius).abs() < 1e-2);
        assert!((pose.yaw - o.yaw).abs() < 1e-4);
        assert!((pose.pitch - o.pitch).abs() < 1e-4);
        // An eye below its focus clamps to the orbit's lowest pitch.
        let low = orbit_from_eye(Vec3::new(100.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 50.0));
        assert_eq!(low.pitch, MIN_PITCH_RAD);
    }

    #[test]
    fn fly_speed_follows_altitude_within_bounds() {
        assert_eq!(fly_speed(1.0, 4000.0, 1.0, false), 5.0, "floor");
        assert_eq!(fly_speed(200.0, 4000.0, 1.0, false), 200.0);
        assert_eq!(
            fly_speed(50_000.0, 4000.0, 1.0, false),
            1000.0,
            "a quarter of the span"
        );
        assert_eq!(fly_speed(200.0, 4000.0, 0.5, true), 400.0);
        let p = clamp_fly(
            Vec3::new(1e6, -1e6, -50.0),
            5.0,
            &bounds(),
            4000.0,
            20_000.0,
        );
        assert_eq!(p, Vec3::new(6000.0, -6000.0, 7.0));
        let high = clamp_fly(Vec3::new(0.0, 0.0, 1e6), 5.0, &bounds(), 4000.0, 20_000.0);
        assert_eq!(high.z, 20_000.0);
    }

    /// Focus moves the orbit onto the selection, damped, and closes in;
    /// with nothing selected it says so and does nothing.
    #[test]
    fn focus_settles_on_the_selection() {
        let scene = Scene::new();
        let mut control = CameraControl::new(OrbitPose::of(&orbit()));
        let mut o = orbit();
        scene.step(&mut control, &mut o, &press(&[Action::Focus]));
        assert!(
            o.focus.distance(scene.drone.unwrap()) > 1.0,
            "a jump, not damped"
        );
        settle(&scene, &mut control, &mut o, 3.0);
        assert!(o.focus.distance(scene.drone.unwrap()) < 0.05);
        assert!((o.radius - FOCUS_RADIUS_M).abs() < 0.05);
        let empty = Scene {
            selected: false,
            ..Scene::new()
        };
        let before = o.focus;
        empty.step(&mut control, &mut o, &press(&[Action::Focus]));
        assert_eq!(o.focus, before);
        assert_eq!(control.notice(), Some("Nothing selected to focus"));
    }

    /// Follow tracks the entity; losing it ends the mode at its last known
    /// position, never the origin, and says so; a new run is a loss too.
    #[test]
    fn follow_tracks_and_a_loss_holds_the_last_known_position() {
        let mut scene = Scene::new();
        let mut control = CameraControl::new(OrbitPose::of(&orbit()));
        let mut o = orbit();
        scene.step(&mut control, &mut o, &press(&[Action::Follow]));
        assert_eq!(control.mode, CameraMode::Follow(key(1)));
        assert_eq!(control.mode_label(), "Following drone-1");
        settle(&scene, &mut control, &mut o, 2.0);
        scene.drone = Some(Vec3::new(700.0, -300.0, 120.0));
        settle(&scene, &mut control, &mut o, 2.0);
        assert!(o.focus.distance(Vec3::new(700.0, -300.0, 120.0)) < 0.1);
        scene.drone = None;
        scene.step(&mut control, &mut o, &idle(1.0 / 60.0));
        assert_eq!(control.mode, CameraMode::Orbit);
        assert_eq!(
            control.notice(),
            Some("Lost drone-1: holding its last known position")
        );
        settle(&scene, &mut control, &mut o, 2.0);
        assert!(
            o.focus.distance(Vec3::new(700.0, -300.0, 120.0)) < 0.1,
            "not held: {}",
            o.focus
        );

        // A new epoch ends it the same way, with the entity still present.
        let mut scene = Scene::new();
        scene.step(&mut control, &mut o, &press(&[Action::Follow]));
        assert_eq!(control.mode, CameraMode::Follow(key(1)));
        scene.epoch = 2;
        scene.step(&mut control, &mut o, &idle(1.0 / 60.0));
        assert_eq!(control.mode, CameraMode::Orbit);
    }

    /// Chase swings the view behind the direction of travel, and keeps the
    /// last heading while the entity hovers.
    #[test]
    fn chase_sits_behind_the_heading() {
        let mut scene = Scene::new();
        let mut control = CameraControl::new(OrbitPose::of(&orbit()));
        let mut o = orbit();
        scene.step(&mut control, &mut o, &press(&[Action::Chase]));
        settle(&scene, &mut control, &mut o, 3.0);
        // Moving east: the eye is west of the drone, at the chase pitch.
        let eye = o.eye_position();
        assert!(eye.x < o.focus.x - 100.0, "eye {eye} focus {}", o.focus);
        assert!((eye.y - o.focus.y).abs() < 1.0);
        assert!((o.pitch - CHASE_PITCH_RAD).abs() < 1e-3);
        scene.velocity = Vec3::new(0.0, 0.1, 0.0);
        settle(&scene, &mut control, &mut o, 1.0);
        assert!(
            o.eye_position().x < o.focus.x - 100.0,
            "a hover turned the chase"
        );
        scene.step(&mut control, &mut o, &press(&[Action::Chase]));
        assert_eq!(control.mode, CameraMode::Orbit, "chase toggles off");
    }

    /// Escape leaves a transient mode first and says it used the key; in
    /// plain orbit it does not use it, so the selection clears.
    #[test]
    fn escape_leaves_the_mode_before_the_selection() {
        let scene = Scene::new();
        let mut control = CameraControl::new(OrbitPose::of(&orbit()));
        let mut o = orbit();
        scene.step(&mut control, &mut o, &press(&[Action::Follow]));
        let out = scene.step(&mut control, &mut o, &press(&[Action::Escape]));
        assert!(out.escape_used);
        assert_eq!(control.mode, CameraMode::Orbit);
        let out = scene.step(&mut control, &mut o, &press(&[Action::Escape]));
        assert!(!out.escape_used);
    }

    /// Free-fly starts where the orbit eye is, looking where it looked; keys
    /// move it along the view at a damped, altitude-scaled speed; it stays
    /// over the ground; leaving it makes an orbit around the ground point it
    /// looked at, with no jump of the eye.
    #[test]
    fn free_fly_enters_moves_and_leaves_without_a_jump() {
        let scene = Scene::new();
        let mut control = CameraControl::new(OrbitPose::of(&orbit()));
        let mut o = orbit();
        let eye = o.eye_position();
        let out = scene.step(&mut control, &mut o, &press(&[Action::FreeFly]));
        assert_eq!(control.mode, CameraMode::FreeFly);
        assert!(
            out.transform.translation.distance(eye) < 1e-3,
            "entering moved the eye"
        );
        let looking = (o.focus - eye).normalize();
        assert!(out.transform.forward().as_vec3().dot(looking) > 0.999);

        let mut forward = idle(1.0 / 60.0);
        forward.held.insert(Action::MoveForward);
        let start = control.fly().position;
        let first = scene
            .step(&mut control, &mut o, &forward)
            .transform
            .translation;
        // Damped: the first frame reaches 1 - exp(-dt / tau), about a tenth,
        // of full speed.
        let full = fly_speed(start.z - 5.0, 4000.0, 1.0, false) / 60.0;
        let damped = 1.0 - (-(1.0 / 60.0) / FLY_TAU_S).exp();
        assert!(
            (first.distance(start) - full * damped).abs() < full * 0.01,
            "no damping: {} of {full}",
            first.distance(start)
        );
        for _ in 0..600 {
            scene.step(&mut control, &mut o, &forward);
        }
        let end = control.fly().position;
        assert!((end - start).dot(looking) > 100.0, "did not fly forward");
        assert!(
            end.z >= 5.0 + FLY_CLEARANCE_M - 1e-3,
            "went underground: {end}"
        );

        // Keys do nothing while the pointer is over another viewport or the
        // UI: from rest, holding forward there moves nothing at all.
        control.fly.velocity = Vec3::ZERO;
        let mut elsewhere = forward.clone();
        elsewhere.over_view = false;
        let before = control.fly().position;
        for _ in 0..30 {
            scene.step(&mut control, &mut o, &elsewhere);
        }
        assert_eq!(control.fly().position, before, "moved by keys elsewhere");

        // Looking stops short of straight up and down.
        let mut look = idle(1.0 / 60.0);
        look.look = Vec2::new(0.0, 1e6);
        scene.step(&mut control, &mut o, &look);
        assert_eq!(control.fly().pitch, -FLY_MAX_PITCH_RAD);
        look.look = Vec2::new(0.0, -1e6);
        scene.step(&mut control, &mut o, &look);
        assert_eq!(control.fly().pitch, FLY_MAX_PITCH_RAD);

        // At the ground (2 m up, the ground point 3.5 m ahead): the focus
        // is that point, and the eye backs off to the orbit's minimum.
        control.fly.velocity = Vec3::ZERO;
        control.fly.pitch = -0.6;
        scene.step(&mut control, &mut o, &press(&[Action::FreeFly]));
        assert_eq!(control.mode, CameraMode::Orbit);
        assert!(
            (o.focus.z - 5.0).abs() < 1e-2,
            "focus underground: {}",
            o.focus
        );
        assert_eq!(o.radius, o.min_radius);

        // Higher up, leaving does not move the eye at all.
        scene.step(&mut control, &mut o, &press(&[Action::FreeFly]));
        control.fly.position.z = 400.0;
        control.fly.velocity = Vec3::ZERO;
        control.fly.pitch = -0.6;
        let eye = control.fly().position;
        let out = scene.step(&mut control, &mut o, &press(&[Action::FreeFly]));
        assert!(
            out.transform.translation.distance(eye) < 0.05,
            "leaving jumped the eye"
        );
        assert!(
            (o.focus.z - 5.0).abs() < 1e-2,
            "focus is the ground point looked at"
        );
    }

    /// The far plane, sized from the orbit's ceiling, holds the whole scene
    /// from anywhere free-fly may put the eye: the worst case is a corner a
    /// span outside the scene, at the ceiling, looking at the far corner.
    #[test]
    fn the_far_plane_holds_the_scene_from_every_free_fly_position() {
        let b = Bounds2d {
            x_min_m: -2160.0,
            x_max_m: 2160.0,
            y_min_m: -1980.0,
            y_max_m: 2160.0,
        };
        let o = OrbitCamera::from_bounds(&b, Some(0.0), Some(237.0));
        let span = (b.x_max_m - b.x_min_m).max(b.y_max_m - b.y_min_m);
        let far = o.far_plane(span);
        let eye = clamp_fly(Vec3::new(-1e9, -1e9, 1e9), 0.0, &b, span, o.max_radius);
        let farthest = Vec3::new(b.x_max_m, b.y_max_m, 0.0);
        assert!(
            eye.distance(farthest) < far,
            "{} past a far plane of {far}",
            eye.distance(farthest)
        );
    }

    /// Scroll scales free-fly speed within bounds.
    #[test]
    fn scroll_scales_free_fly_speed() {
        let scene = Scene::new();
        let mut control = CameraControl::new(OrbitPose::of(&orbit()));
        let mut o = orbit();
        scene.step(&mut control, &mut o, &press(&[Action::FreeFly]));
        let mut up = idle(1.0 / 60.0);
        up.scroll = 3.0;
        scene.step(&mut control, &mut o, &up);
        assert!((control.fly().multiplier - 1.25_f32.powi(3)).abs() < 1e-4);
        up.scroll = 100.0;
        scene.step(&mut control, &mut o, &up);
        assert_eq!(control.fly().multiplier, 10.0);
    }

    /// Ctrl+digit asks to save the pose; the digit alone recalls it, damped,
    /// leaving free-fly; an empty slot says how to fill it; reset goes home.
    #[test]
    fn bookmarks_save_recall_and_reset_goes_home() {
        let mut scene = Scene::new();
        let home = OrbitPose::of(&orbit());
        let mut control = CameraControl::new(home);
        let mut o = orbit();
        let mut save = press(&[Action::Bookmark(4)]);
        save.save_modifier = true;
        let out = scene.step(&mut control, &mut o, &save);
        let (slot, pose) = out.save_bookmark.unwrap();
        assert_eq!(slot, 4);
        assert_eq!(pose, OrbitPose::of(&o));

        let target = OrbitPose {
            focus: Vec3::new(100.0, 200.0, 30.0),
            radius: 800.0,
            yaw: -1.0,
            pitch: 0.7,
        };
        scene.bookmarks.push((4, target));
        scene.step(&mut control, &mut o, &press(&[Action::FreeFly]));
        scene.step(&mut control, &mut o, &press(&[Action::Bookmark(4)]));
        assert_eq!(control.mode, CameraMode::Orbit);
        settle(&scene, &mut control, &mut o, 3.0);
        let now = OrbitPose::of(&o);
        assert!(
            now.focus.distance(target.focus) < 0.05 && (now.radius - target.radius).abs() < 0.05
        );
        assert!((now.yaw - target.yaw).abs() < 1e-3 && (now.pitch - target.pitch).abs() < 1e-3);

        scene.step(&mut control, &mut o, &press(&[Action::Bookmark(7)]));
        assert_eq!(
            control.notice(),
            Some("Bookmark 7 is empty: Ctrl+7 saves one")
        );
        scene.step(&mut control, &mut o, &press(&[Action::ResetCamera]));
        settle(&scene, &mut control, &mut o, 3.0);
        assert!(OrbitPose::of(&o).focus.distance(home.focus) < 0.05);
    }

    /// Notices expire in render time.
    #[test]
    fn notices_expire() {
        let scene = Scene {
            selected: false,
            ..Scene::new()
        };
        let mut control = CameraControl::new(OrbitPose::of(&orbit()));
        let mut o = orbit();
        scene.step(&mut control, &mut o, &press(&[Action::Focus]));
        assert!(control.notice().is_some());
        scene.step(&mut control, &mut o, &idle(NOTICE_S + 0.1));
        assert!(control.notice().is_none());
    }
}
