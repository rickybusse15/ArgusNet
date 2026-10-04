//! Quality tiers (C6, VIEWER_UI §15): low, standard and high share one
//! meaning. A tier changes how the world is rendered (shadows, distance fog,
//! bloom, multisampling), never what is shown: it has no access to the layer
//! toggles, the selection, the zones or the terrain's unknown-space encoding,
//! so no tier can remove safety geometry, unknown space, selection identity or
//! uncertainty. [`apply_quality_tier_system`] is the only writer, and its
//! parameters are the camera and the lights.
//!
//! Bloom is off unless the high tier is chosen. Shadows and fog predate the
//! rule that an expensive effect needs G2 baselines to be on by default; the
//! standard tier keeps them, pending those baselines (decision D-13).

use bevy::light::DirectionalLight;
use bevy::pbr::{DistanceFog, FogFalloff};
use bevy::post_process::bloom::{Bloom, BloomPrefilter};
use bevy::prelude::*;
use bevy::render::view::Msaa;

use crate::app::MainCamera;

/// The rendering quality tier.
#[derive(Resource, Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub enum QualityTier {
    Low,
    #[default]
    Standard,
    High,
}

/// What a tier turns on. Rendering effects only, by construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TierEffects {
    pub shadows: bool,
    pub fog: bool,
    pub bloom: bool,
    pub msaa: Msaa,
}

impl QualityTier {
    pub const ALL: [QualityTier; 3] = [QualityTier::Low, QualityTier::Standard, QualityTier::High];

    pub fn label(self) -> &'static str {
        match self {
            QualityTier::Low => "Low",
            QualityTier::Standard => "Standard",
            QualityTier::High => "High",
        }
    }

    /// The name a preferences file uses.
    pub fn name(self) -> &'static str {
        match self {
            QualityTier::Low => "low",
            QualityTier::Standard => "standard",
            QualityTier::High => "high",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|tier| tier.name() == name)
    }

    pub fn effects(self) -> TierEffects {
        match self {
            QualityTier::Low => TierEffects {
                shadows: false,
                fog: false,
                bloom: false,
                msaa: Msaa::Off,
            },
            QualityTier::Standard => TierEffects {
                shadows: true,
                fog: true,
                bloom: false,
                msaa: Msaa::Sample4,
            },
            QualityTier::High => TierEffects {
                shadows: true,
                fog: true,
                bloom: true,
                msaa: Msaa::Sample4,
            },
        }
    }
}

/// The fog the main camera carries when its tier has fog: the sky's horizon
/// colour, with distances filled in each frame by the fog sync, which tracks
/// the camera's distance and the scene span.
pub fn main_fog(color: Color) -> DistanceFog {
    DistanceFog {
        color,
        falloff: FogFalloff::Linear {
            start: 0.0,
            end: f32::MAX,
        },
        ..default()
    }
}

/// The high tier's bloom. `NATURAL` thresholds at zero and blooms everything,
/// so plain white roofs smeared like beacons; thresholding just above display
/// white keeps the glow on genuinely over-bright (emissive) surfaces.
pub fn main_bloom() -> Bloom {
    Bloom {
        prefilter: BloomPrefilter {
            threshold: 1.0,
            threshold_softness: 0.4,
        },
        ..Bloom::NATURAL
    }
}

/// Carry the operator's choice (`ViewerUiState::quality`, set through the
/// shell's commands) to the tier, and remember it. On the first run the
/// choice is seeded from the tier the viewer started with (command line or
/// preferences), so starting never overwrites it.
pub fn sync_quality_choice_system(
    mut ui_state: ResMut<crate::state::ViewerUiState>,
    mut tier: ResMut<QualityTier>,
    prefs: Option<ResMut<crate::prefs::ViewerPrefs>>,
    mut seeded: Local<bool>,
) {
    if !*seeded {
        *seeded = true;
        if ui_state.quality != *tier {
            ui_state.quality = *tier;
        }
        return;
    }
    if ui_state.quality == *tier {
        return;
    }
    *tier = ui_state.quality;
    if let Some(mut prefs) = prefs {
        if let Err(problem) = prefs.set_quality_tier(*tier) {
            warn!("could not save the quality tier: {problem}");
        }
    }
}

/// The main camera, with what a tier may change on it.
type TieredCamera<'a> = (
    Entity,
    Ref<'a, MainCamera>,
    Has<DistanceFog>,
    Has<Bloom>,
    &'a Msaa,
);

/// Apply the tier to the main camera and the lights, when it changes and
/// when the camera appears.
pub fn apply_quality_tier_system(
    mut commands: Commands,
    tier: Res<QualityTier>,
    cameras: Query<TieredCamera>,
    mut lights: Query<&mut DirectionalLight>,
) {
    let Ok((camera, marker, has_fog, has_bloom, msaa)) = cameras.single() else {
        return;
    };
    if !tier.is_changed() && !marker.is_added() {
        return;
    }
    let effects = tier.effects();
    let mut camera = commands.entity(camera);
    match (effects.fog, has_fog) {
        (true, false) => {
            camera.insert(main_fog(crate::sky::HORIZON_COLOR));
        }
        (false, true) => {
            camera.remove::<DistanceFog>();
        }
        _ => {}
    }
    match (effects.bloom, has_bloom) {
        (true, false) => {
            camera.insert(main_bloom());
        }
        (false, true) => {
            camera.remove::<Bloom>();
        }
        _ => {}
    }
    if *msaa != effects.msaa {
        camera.insert(effects.msaa);
    }
    for mut light in &mut lights {
        if light.shadow_maps_enabled != effects.shadows {
            light.shadow_maps_enabled = effects.shadows;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{MissionOverlaySettings, RuntimeOverlayVisibility};

    fn app(tier: QualityTier) -> App {
        let mut app = App::new();
        app.insert_resource(tier)
            .init_resource::<RuntimeOverlayVisibility>()
            .init_resource::<MissionOverlaySettings>()
            .add_systems(Update, apply_quality_tier_system);
        app.world_mut().spawn((MainCamera, Msaa::Sample4));
        app.world_mut().spawn(DirectionalLight {
            shadow_maps_enabled: true,
            ..default()
        });
        app
    }

    fn camera(app: &mut App) -> (bool, bool, Msaa) {
        let mut query = app
            .world_mut()
            .query_filtered::<(Has<DistanceFog>, Has<Bloom>, &Msaa), With<MainCamera>>();
        let (fog, bloom, msaa) = query.single(app.world()).unwrap();
        (fog, bloom, *msaa)
    }

    fn shadows(app: &mut App) -> bool {
        let mut query = app.world_mut().query::<&DirectionalLight>();
        query.single(app.world()).unwrap().shadow_maps_enabled
    }

    #[test]
    fn bloom_is_off_by_default_and_only_high_turns_it_on() {
        assert_eq!(QualityTier::default(), QualityTier::Standard);
        assert!(!QualityTier::default().effects().bloom);
        let bloomed: Vec<QualityTier> = QualityTier::ALL
            .into_iter()
            .filter(|tier| tier.effects().bloom)
            .collect();
        assert_eq!(bloomed, [QualityTier::High]);
    }

    #[test]
    fn each_tier_reaches_the_camera_and_lights() {
        let mut app = app(QualityTier::Low);
        app.update();
        assert_eq!(camera(&mut app), (false, false, Msaa::Off));
        assert!(!shadows(&mut app));
        *app.world_mut().resource_mut::<QualityTier>() = QualityTier::High;
        app.update();
        assert_eq!(camera(&mut app), (true, true, Msaa::Sample4));
        assert!(shadows(&mut app));
        *app.world_mut().resource_mut::<QualityTier>() = QualityTier::Standard;
        app.update();
        assert_eq!(camera(&mut app), (true, false, Msaa::Sample4));
    }

    /// §15: a lower tier may not remove safety geometry, unknown-space
    /// encoding, selection identity or essential uncertainty. Those live in
    /// the layer toggles and the selection, which switching tiers leaves
    /// untouched.
    #[test]
    fn no_tier_changes_what_is_shown() {
        let mut app = app(QualityTier::Standard);
        app.update();
        let visibility = app.world().resource::<RuntimeOverlayVisibility>().clone();
        let overlay = app.world().resource::<MissionOverlaySettings>().clone();
        for tier in QualityTier::ALL {
            *app.world_mut().resource_mut::<QualityTier>() = tier;
            app.update();
            let world = app.world();
            assert_eq!(world.resource::<RuntimeOverlayVisibility>(), &visibility);
            assert_eq!(world.resource::<MissionOverlaySettings>(), &overlay);
        }
    }

    #[test]
    fn the_operators_choice_reaches_the_tier_but_startup_does_not_override_it() {
        let mut app = App::new();
        app.insert_resource(QualityTier::Low)
            .init_resource::<crate::state::ViewerUiState>()
            .add_systems(Update, sync_quality_choice_system);
        app.update();
        assert_eq!(
            app.world()
                .resource::<crate::state::ViewerUiState>()
                .quality,
            QualityTier::Low,
            "seeded from the starting tier"
        );
        app.world_mut()
            .resource_mut::<crate::state::ViewerUiState>()
            .quality = QualityTier::High;
        app.update();
        assert_eq!(*app.world().resource::<QualityTier>(), QualityTier::High);
    }

    #[test]
    fn tier_names_round_trip() {
        for tier in QualityTier::ALL {
            assert_eq!(QualityTier::from_name(tier.name()), Some(tier));
        }
        assert_eq!(QualityTier::from_name("ultra"), None);
    }
}
