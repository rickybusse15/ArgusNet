//! Keyboard bindings: one remappable table for every viewer action (P2.4b).
//!
//! Every keyboard action the viewer takes is an [`Action`], bound to one key
//! in [`Bindings`]. The defaults match `docs/VIEWER_UI.md`'s table. A viewer
//! preferences file may remap them by name (`"focus": "KeyF"`); an unknown
//! action or key, or a key already bound to another action, is refused and
//! reported, and the default stays.
//!
//! **Nothing fires while typing.** [`KeyboardGate`] is computed once a frame,
//! first thing, from egui; every system that reads a binding goes through
//! [`Bindings::just_pressed`] or [`Bindings::held`] with the gate, so a key
//! typed into a text field never also plays, steps or flies.

use std::collections::BTreeMap;

use bevy::prelude::*;
use bevy_egui::EguiContexts;

/// Everything the keyboard can do.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Action {
    PlayPause,
    StepBack,
    StepForward,
    First,
    Last,
    PreviousEvent,
    NextEvent,
    ResetCamera,
    CycleView,
    NextSelection,
    /// Clears a transient mode, then the selection (VIEWER_UI §12).
    Escape,
    Focus,
    Follow,
    Chase,
    FreeFly,
    MoveForward,
    MoveBack,
    MoveLeft,
    MoveRight,
    MoveUp,
    MoveDown,
    Faster,
    /// Operate -> Plan -> Review (C2.3).
    CycleWorkspace,
    /// The next available projection.
    CycleProjection,
    /// Operational -> Reconstruction -> Safety.
    CyclePreset,
    /// Open or close the tool panel.
    ToggleToolPanel,
    /// Recall bookmark 1-9; with Ctrl or Cmd held, save it.
    Bookmark(u8),
}

impl Action {
    /// Every action, in a stable order. A constant, so reading every binding
    /// each frame allocates nothing.
    pub const ALL: [Action; 35] = [
        Action::PlayPause,
        Action::StepBack,
        Action::StepForward,
        Action::First,
        Action::Last,
        Action::PreviousEvent,
        Action::NextEvent,
        Action::ResetCamera,
        Action::CycleView,
        Action::NextSelection,
        Action::Escape,
        Action::Focus,
        Action::Follow,
        Action::Chase,
        Action::FreeFly,
        Action::MoveForward,
        Action::MoveBack,
        Action::MoveLeft,
        Action::MoveRight,
        Action::MoveUp,
        Action::MoveDown,
        Action::Faster,
        Action::CycleWorkspace,
        Action::CycleProjection,
        Action::CyclePreset,
        Action::ToggleToolPanel,
        Action::Bookmark(1),
        Action::Bookmark(2),
        Action::Bookmark(3),
        Action::Bookmark(4),
        Action::Bookmark(5),
        Action::Bookmark(6),
        Action::Bookmark(7),
        Action::Bookmark(8),
        Action::Bookmark(9),
    ];

    /// This action's place in [`Action::ALL`].
    pub fn index(self) -> usize {
        Self::ALL
            .iter()
            .position(|action| *action == self)
            .expect("every action is listed")
    }

    /// Every action, in a stable order.
    pub fn all() -> Vec<Action> {
        Self::ALL.to_vec()
    }

    /// The name a preferences file uses.
    pub fn name(self) -> String {
        match self {
            Action::PlayPause => "play_pause".into(),
            Action::StepBack => "step_back".into(),
            Action::StepForward => "step_forward".into(),
            Action::First => "first".into(),
            Action::Last => "last".into(),
            Action::PreviousEvent => "previous_event".into(),
            Action::NextEvent => "next_event".into(),
            Action::ResetCamera => "reset_camera".into(),
            Action::CycleView => "cycle_view".into(),
            Action::NextSelection => "next_selection".into(),
            Action::Escape => "escape".into(),
            Action::Focus => "focus".into(),
            Action::Follow => "follow".into(),
            Action::Chase => "chase".into(),
            Action::FreeFly => "free_fly".into(),
            Action::MoveForward => "move_forward".into(),
            Action::MoveBack => "move_back".into(),
            Action::MoveLeft => "move_left".into(),
            Action::MoveRight => "move_right".into(),
            Action::MoveUp => "move_up".into(),
            Action::MoveDown => "move_down".into(),
            Action::Faster => "faster".into(),
            Action::CycleWorkspace => "cycle_workspace".into(),
            Action::CycleProjection => "cycle_projection".into(),
            Action::CyclePreset => "cycle_preset".into(),
            Action::ToggleToolPanel => "toggle_tool_panel".into(),
            Action::Bookmark(n) => format!("bookmark_{n}"),
        }
    }

    pub fn from_name(name: &str) -> Option<Action> {
        Action::all()
            .into_iter()
            .find(|action| action.name() == name)
    }

    fn default_key(self) -> KeyCode {
        match self {
            Action::PlayPause => KeyCode::Space,
            Action::StepBack => KeyCode::ArrowLeft,
            Action::StepForward => KeyCode::ArrowRight,
            Action::First => KeyCode::Home,
            Action::Last => KeyCode::End,
            Action::PreviousEvent => KeyCode::BracketLeft,
            Action::NextEvent => KeyCode::BracketRight,
            Action::ResetCamera => KeyCode::KeyR,
            Action::CycleView => KeyCode::KeyM,
            Action::NextSelection => KeyCode::Tab,
            Action::Escape => KeyCode::Escape,
            Action::Focus => KeyCode::KeyF,
            Action::Follow => KeyCode::KeyL,
            Action::Chase => KeyCode::KeyC,
            Action::FreeFly => KeyCode::KeyG,
            Action::MoveForward => KeyCode::KeyW,
            Action::MoveBack => KeyCode::KeyS,
            Action::MoveLeft => KeyCode::KeyA,
            Action::MoveRight => KeyCode::KeyD,
            Action::MoveUp => KeyCode::KeyE,
            Action::MoveDown => KeyCode::KeyQ,
            Action::Faster => KeyCode::ShiftLeft,
            Action::CycleWorkspace => KeyCode::KeyO,
            Action::CycleProjection => KeyCode::KeyV,
            Action::CyclePreset => KeyCode::KeyP,
            Action::ToggleToolPanel => KeyCode::KeyT,
            Action::Bookmark(n) => DIGITS[usize::from(n.clamp(1, 9)) - 1],
        }
    }
}

const DIGITS: [KeyCode; 9] = [
    KeyCode::Digit1,
    KeyCode::Digit2,
    KeyCode::Digit3,
    KeyCode::Digit4,
    KeyCode::Digit5,
    KeyCode::Digit6,
    KeyCode::Digit7,
    KeyCode::Digit8,
    KeyCode::Digit9,
];

/// The keys a preferences file may name, by their `KeyCode` names.
pub fn key_from_name(name: &str) -> Option<KeyCode> {
    let letters = [
        KeyCode::KeyA,
        KeyCode::KeyB,
        KeyCode::KeyC,
        KeyCode::KeyD,
        KeyCode::KeyE,
        KeyCode::KeyF,
        KeyCode::KeyG,
        KeyCode::KeyH,
        KeyCode::KeyI,
        KeyCode::KeyJ,
        KeyCode::KeyK,
        KeyCode::KeyL,
        KeyCode::KeyM,
        KeyCode::KeyN,
        KeyCode::KeyO,
        KeyCode::KeyP,
        KeyCode::KeyQ,
        KeyCode::KeyR,
        KeyCode::KeyS,
        KeyCode::KeyT,
        KeyCode::KeyU,
        KeyCode::KeyV,
        KeyCode::KeyW,
        KeyCode::KeyX,
        KeyCode::KeyY,
        KeyCode::KeyZ,
    ];
    if let Some(letter) = name.strip_prefix("Key") {
        let mut chars = letter.chars();
        if let (Some(c @ 'A'..='Z'), None) = (chars.next(), chars.next()) {
            return Some(letters[(c as u8 - b'A') as usize]);
        }
    }
    if let Some(digit) = name.strip_prefix("Digit") {
        return match digit.parse::<usize>() {
            Ok(0) => Some(KeyCode::Digit0),
            Ok(n @ 1..=9) => Some(DIGITS[n - 1]),
            _ => None,
        };
    }
    Some(match name {
        "Space" => KeyCode::Space,
        "Tab" => KeyCode::Tab,
        "Escape" => KeyCode::Escape,
        "Enter" => KeyCode::Enter,
        "Backspace" => KeyCode::Backspace,
        "ArrowLeft" => KeyCode::ArrowLeft,
        "ArrowRight" => KeyCode::ArrowRight,
        "ArrowUp" => KeyCode::ArrowUp,
        "ArrowDown" => KeyCode::ArrowDown,
        "Home" => KeyCode::Home,
        "End" => KeyCode::End,
        "PageUp" => KeyCode::PageUp,
        "PageDown" => KeyCode::PageDown,
        "BracketLeft" => KeyCode::BracketLeft,
        "BracketRight" => KeyCode::BracketRight,
        "Comma" => KeyCode::Comma,
        "Period" => KeyCode::Period,
        "Slash" => KeyCode::Slash,
        "Semicolon" => KeyCode::Semicolon,
        "Quote" => KeyCode::Quote,
        "Minus" => KeyCode::Minus,
        "Equal" => KeyCode::Equal,
        "ShiftLeft" => KeyCode::ShiftLeft,
        "ShiftRight" => KeyCode::ShiftRight,
        "ControlLeft" => KeyCode::ControlLeft,
        "AltLeft" => KeyCode::AltLeft,
        _ => return None,
    })
}

/// A set of actions, as a bitset: building one every frame, with keys
/// held, allocates nothing.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ActionSet(u64);

impl ActionSet {
    pub fn insert(&mut self, action: Action) {
        self.0 |= 1 << action.index();
    }

    pub fn contains(&self, action: Action) -> bool {
        self.0 & (1 << action.index()) != 0
    }

    pub fn is_empty(&self) -> bool {
        self.0 == 0
    }
}

impl FromIterator<Action> for ActionSet {
    fn from_iter<I: IntoIterator<Item = Action>>(iter: I) -> Self {
        let mut set = Self::default();
        for action in iter {
            set.insert(action);
        }
        set
    }
}

/// Which key each action is on.
#[derive(Resource, Debug, Clone, PartialEq)]
pub struct Bindings {
    keys: BTreeMap<Action, KeyCode>,
}

impl Default for Bindings {
    fn default() -> Self {
        Self {
            keys: Action::all()
                .into_iter()
                .map(|action| (action, action.default_key()))
                .collect(),
        }
    }
}

impl Bindings {
    pub fn key(&self, action: Action) -> KeyCode {
        self.keys[&action]
    }

    /// The defaults with `overrides` (action name -> key name) applied, and
    /// what was refused. An override that names an unknown action or key, or
    /// a key another action already has, is refused; its action keeps its
    /// default.
    pub fn with_overrides(overrides: &BTreeMap<String, String>) -> (Self, Vec<String>) {
        let mut bindings = Self::default();
        let mut problems = Vec::new();
        for (action_name, key_name) in overrides {
            let Some(action) = Action::from_name(action_name) else {
                problems.push(format!("unknown action {action_name:?}"));
                continue;
            };
            let Some(key) = key_from_name(key_name) else {
                problems.push(format!("{action_name}: unknown key {key_name:?}"));
                continue;
            };
            let clash = bindings
                .keys
                .iter()
                .find(|(other, bound)| **other != action && **bound == key)
                .map(|(other, _)| *other);
            if let Some(other) = clash {
                problems.push(format!(
                    "{action_name}: {key_name} is already {}",
                    other.name()
                ));
                continue;
            }
            bindings.keys.insert(action, key);
        }
        (bindings, problems)
    }

    /// Whether `action`'s key went down this frame, unless the UI is taking
    /// the keyboard.
    pub fn just_pressed(
        &self,
        gate: &KeyboardGate,
        keys: &ButtonInput<KeyCode>,
        action: Action,
    ) -> bool {
        !gate.typing && keys.just_pressed(self.key(action))
    }

    /// Whether `action`'s key is held, unless the UI is taking the keyboard.
    pub fn held(&self, gate: &KeyboardGate, keys: &ButtonInput<KeyCode>, action: Action) -> bool {
        !gate.typing && keys.pressed(self.key(action))
    }
}

/// Whether the UI has the keyboard this frame. Set first in the frame.
#[derive(Resource, Debug, Clone, Copy, Default, PartialEq)]
pub struct KeyboardGate {
    pub typing: bool,
}

/// Asks egui, once a frame, whether a text field has the keyboard. Headless
/// has no egui: nothing is typing there.
pub fn keyboard_gate_system(contexts: Option<EguiContexts>, mut gate: ResMut<KeyboardGate>) {
    let typing = contexts
        .and_then(|mut contexts| {
            contexts
                .ctx_mut()
                .ok()
                .map(|ctx| ctx.egui_wants_keyboard_input())
        })
        .unwrap_or(false);
    gate.set_if_neq(KeyboardGate { typing });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_action_set_holds_exactly_what_was_inserted() {
        let set: ActionSet = [Action::Focus, Action::Bookmark(9)].into_iter().collect();
        for action in Action::ALL {
            assert_eq!(
                set.contains(action),
                matches!(action, Action::Focus | Action::Bookmark(9))
            );
        }
        assert!(ActionSet::default().is_empty());
        assert!(Action::ALL.len() <= 64, "the bitset holds 64");
    }

    #[test]
    fn defaults_bind_every_action_to_a_distinct_key() {
        let bindings = Bindings::default();
        let mut keys: Vec<KeyCode> = Action::all().into_iter().map(|a| bindings.key(a)).collect();
        let n = keys.len();
        keys.sort_by_key(|k| format!("{k:?}"));
        keys.dedup();
        assert_eq!(keys.len(), n, "two actions share a default key");
        assert_eq!(bindings.key(Action::Focus), KeyCode::KeyF);
        assert_eq!(bindings.key(Action::Bookmark(1)), KeyCode::Digit1);
        assert_eq!(bindings.key(Action::Bookmark(9)), KeyCode::Digit9);
    }

    #[test]
    fn every_action_round_trips_its_name_and_every_default_key_its_name() {
        for action in Action::all() {
            assert_eq!(Action::from_name(&action.name()), Some(action));
            let key = Bindings::default().key(action);
            assert_eq!(key_from_name(&format!("{key:?}")), Some(key), "{action:?}");
        }
        assert_eq!(key_from_name("KeyZ"), Some(KeyCode::KeyZ));
        assert_eq!(key_from_name("Key"), None);
        assert_eq!(key_from_name("KeyAA"), None);
        assert_eq!(key_from_name("Digit10"), None);
    }

    #[test]
    fn overrides_apply_and_bad_ones_are_refused_and_reported() {
        let overrides: BTreeMap<String, String> = [
            ("focus", "KeyZ"),        // fine
            ("chase", "KeyW"),        // taken by move_forward
            ("teleport", "KeyT"),     // no such action
            ("follow", "Hyperspace"), // no such key
        ]
        .into_iter()
        .map(|(a, k)| (a.to_string(), k.to_string()))
        .collect();
        let (bindings, problems) = Bindings::with_overrides(&overrides);
        assert_eq!(bindings.key(Action::Focus), KeyCode::KeyZ);
        assert_eq!(
            bindings.key(Action::Chase),
            KeyCode::KeyC,
            "a clash keeps the default"
        );
        assert_eq!(bindings.key(Action::Follow), KeyCode::KeyL);
        assert_eq!(problems.len(), 3, "{problems:?}");
        assert!(problems.iter().any(|p| p.contains("already move_forward")));
    }

    /// Nothing fires while a text field has the keyboard.
    #[test]
    fn typing_blocks_every_binding() {
        let bindings = Bindings::default();
        let mut keys = ButtonInput::<KeyCode>::default();
        keys.press(KeyCode::KeyF);
        keys.press(KeyCode::KeyW);
        let free = KeyboardGate { typing: false };
        let typing = KeyboardGate { typing: true };
        assert!(bindings.just_pressed(&free, &keys, Action::Focus));
        assert!(bindings.held(&free, &keys, Action::MoveForward));
        assert!(!bindings.just_pressed(&typing, &keys, Action::Focus));
        assert!(!bindings.held(&typing, &keys, Action::MoveForward));
    }
}
