# P2.5a: visual tokens, the bundled face, status vocabulary

**Status:** 2026-10-03. First slice of P2.5 (the mission-control shell). See
[STATUS.md](STATUS.md). Specification: [VIEWER_UI.md](../VIEWER_UI.md) §5
(visual system), §12 (failure states) and §13 (accessibility).

The shell slices that follow (P2.5b onward) draw from these tokens instead of
literals. This slice builds the foundation and swaps it in for the old ad-hoc
style. It does not yet rebuild the panels, which still use their own colours.

## What it adds

`rust/argusnet-viewer/src/theme.rs`:

| Piece | What |
| --- | --- |
| Palette | §5.3's nine tokens (window, chrome, canvas, two rules, three text levels, accent) plus `CRITICAL`, reserved for blocked execution, faults and immediate safety action |
| Spacing | 4 px base; 8, 12, 16, 20, 24, 32; 1 px rules |
| Type roles | §5.2's six roles (object identity, workspace title, section title, body, utility, numeric) with size, tracking, weight and colour; utility labels are uppercased |
| Visuals | `apply_visuals`: dark, opaque chrome panels, zero corner radius everywhere, 1 px rules, no window shadow, accent for selection, focus and active state, idle → hover one contrast step brighter (§5.4, §5.6) |
| Face | Arimo Regular and Medium, compiled in (below) |
| `Status` | NOMINAL, UNKNOWN, STALE, DEGRADED, CRITICAL, UNAVAILABLE, PENDING: each with an uppercase label, a distinct shape (● ○ ◌ ▲ ■ — …) and a colour, so colour is never the only cue |
| `Field<T>` | `Value`, `Unavailable(reason)` or `Stale { value, age_s }`. A readout built on it cannot show a missing value as `0` or empty; `Option::None` converts to "not reported" |

`install_theme_system` installs the fonts and visuals once, on the first frame
with an egui context, ahead of the operator UI. The old
`configure_operator_style` (4 px radii, egui's default palette, re-applied every
frame) is gone.

## The face (decision D-2)

§5.2 names Helvetica / Neue Haas Grotesk, both commercial. The maintainer
chose to bundle an open, metric-compatible face now:
**Arimo 1.341**, from Google Fonts, under the **SIL Open Font License 1.1**
(`rust/argusnet-viewer/assets/fonts/OFL.txt`, with the copyright notice; no
Reserved Font Name is declared). Arimo's advance widths match
Helvetica/Arial, and its digits are tabular, which §5.2 requires for time,
coordinates and telemetry.

- It is `include_bytes!`'d, so it cannot fail to load at run time, and the UI
  never drops to egui's face silently (§5.2). egui's default faces remain
  *behind* Arimo, only as glyph fallback.
- Arimo has no 300 weight. The roles §5.2 sets at 300 (object identity,
  numeric) use Regular (400); the section title and utility roles use
  Medium (500).
- A licensed Helvetica can replace it by changing the two `include_bytes!`
  paths and the licence file.

## One deviation from §5.3, for §13

§5.3's muted text `#687175` reaches only **3.74–3.93:1** on the three grounds,
below the **WCAG AA 4.5:1** that §13 requires for all UI text. The token is the
same hue lifted uniformly (+14 per channel) to the first value that clears AA
on every ground: **`#767F83`** (worst 4.56:1, on chrome). Secondary text stays
at 7.9:1, so the three-level hierarchy holds. §5.3 calls its table a
reference palette; §13 is a requirement.

## Verification

| Check | Command | Result (linux-x86_64, locally verified) |
| --- | --- | --- |
| Contrast | `theme::tests::text_on_every_ground_meets_wcag_aa` | primary, secondary, muted, accent and critical meet 4.5:1 on window, chrome and canvas |
| WCAG formula | `contrast_matches_the_wcag_reference_points` | black/white 21:1; identity 1:1 |
| Status | `every_status_differs_by_label_and_shape_not_only_colour` | 7 distinct labels and shapes; red only for critical |
| Fields | `a_missing_field_never_reads_as_a_number` | unavailable shows no digit; stale shows its age; stale is not current |
| Face | `the_bundled_face_is_installed_and_its_digits_are_tabular` | Arimo first in both families; all 10 digits the same advance; every status glyph in Arimo's own character map |
| Visuals | `visuals_are_square_flat_and_from_the_palette` | zero radius, 1 px rules, no window shadow, chrome panels, accent on active |
| Viewer | `scripts/ci-local.sh rust-viewer` (tests, binary, headless render) | see the PR |
| Lint | `cargo clippy -p argusnet-viewer --all-targets -D warnings`, `cargo fmt --check` | see the PR |

## Not done here

- The panels still use their own colours (54 `Color32::from_rgb` literals in
  `ui.rs`). P2.5b's shell rebuild moves them onto the tokens, and adds the
  test that `ui.rs` defines no colours of its own.
- Reduced motion and UI scale are P2.5g.
