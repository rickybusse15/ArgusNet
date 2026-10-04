# G2 renderer fixtures: software baseline, 2026-10-04

**What this is.** The first G2 artifacts (E2): the five fixtures VIEWER_UI and
DELIVERY_PLAN name (overview, close view, low pitch, split mode, high entity
count) rendered in each quality tier through the real headless renderer, with
warm-up-aware frame statistics. **What it is not:** the 60 Hz acceptance
budget. That is calibrated on named reference GPU hardware
(VERIFICATION_AND_ROLLOUT.md §5, decision D-5), which this machine does not
have: the adapter is **llvmpipe (Mesa 25.2.8, LLVM 20.1.2), a software
renderer**, and every run is labelled `renderer_class: software`.

**How.** `python scripts/render_fixtures.py OUT --viewer target/release/argusnet-viewer`
(release build, 650×480, 120 timed frames per run after the still). Each run
writes `<fixture>-<tier>.png` and `<fixture>-<tier>.frames.json`
(`argusnet-viewer --headless --frame-stats N`). Warm-up is reported, never
discarded: assets, the frame something was first drawn, and the frame the
capture settled on precede the timed frames. The timed loop runs unpaced, so a
sample is the frame's own cost, including the per-frame readback headless
performs. GPU time is not recorded yet (`gpu_time_ms: null`).

Data: [`g2-software-baseline-linux-x86_64.json`](g2-software-baseline-linux-x86_64.json)
(every run's summary, warm-up, adapter, scene and image hashes; Standard has a
repeat run). Standard-tier stills (downscaled):
[`g2-baseline/`](g2-baseline/).

## Frame time, ms (p50 / p95 / p99 / max; frames over 100 ms)

| Fixture | Low | Standard | High |
| --- | --- | --- | --- |
| overview | 22.9 / 28.3 / 30.3 / 30.5; 0 | 51.0 / 60.9 / 64.2 / 66.1; 0 | 77.4 / 89.2 / 93.5 / 135.8; 1 |
| close | 24.5 / 29.2 / 32.1 / 75.3; 0 | 73.0 / 81.0 / 87.4 / 91.0; 0 | 99.9 / 111.1 / 118.8 / 122.7; 58 |
| low_pitch | 16.9 / 21.7 / 27.6 / 35.1; 0 | 42.1 / 49.5 / 52.4 / 55.6; 0 | 65.6 / 74.3 / 81.3 / 82.1; 0 |
| split | 26.5 / 31.8 / 33.1 / 33.3; 0 | 52.1 / 63.0 / 67.8 / 71.0; 0 | 66.6 / 76.1 / 86.5 / 110.9; 1 |
| high_entity (400 tracks) | 23.3 / 26.6 / 28.5 / 30.5; 0 | 52.6 / 64.1 / 71.5 / 73.9; 0 | 77.3 / 88.8 / 96.9 / 98.4; 0 |

Standard repeated: p50 within 7 % of the first run on every fixture
(overview 50.7, close 75.3, low pitch 39.1, split 52.3, high entity 52.7).

## Reading it

1. **On software, Standard costs 2.2–3.0× Low** (shadows, fog and 4× MSAA),
   and High adds 30–50 % for bloom. This is relative cost on a CPU
   rasteriser, where MSAA and shadow maps are disproportionately expensive; it
   is not evidence about a GPU. **D-13 stands**: shadows and fog stay on in
   Standard until the same fixtures run on reference hardware.
2. **The close view is the worst case**, and in High on software 58 of 120
   frames exceed 100 ms. Not a hardware claim, but it names the fixture to
   watch first when hardware numbers arrive.
3. **400 extra tracks cost little** next to the scene itself (high_entity
   ≈ overview in every tier): presentation is not the bottleneck at this count.
4. **Stills are bit-identical across runs on this renderer** (every
   Standard image hash matched the repeat). `image_sha256` is therefore a
   same-renderer regression key; across drivers, comparisons must be perceptual
   (C8), never bit-identical.

## Open

- Reference hardware and its budgets (D-5); GPU timestamp queries.
- Perceptual and semantic image comparison, and resource-growth tests: C8.
