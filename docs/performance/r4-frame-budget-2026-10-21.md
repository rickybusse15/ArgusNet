# R4 frame budget on the reference machine, 2026-10-21

**Result: met with room to spare.** On the D-5 reference machine, every
realism fixture in every quality tier stays under the C8 budget (p95 ≤
16.7 ms, p99 ≤ 25 ms, no stall over 100 ms). The worst p95 is 6.3 ms, the
worst p99 8.0 ms and the worst frame 10.0 ms, over 75 runs.

**Reference machine (D-5, decided by the maintainer 2026-10-20):** Apple
M5 Pro (18 cores), 64 GB, macOS 27.0, Metal (`renderer_class: hardware`).
Release build of main at `32ba948`.

**How.** `python scripts/render_fixtures.py OUT --viewer target/release/argusnet-viewer
--argusnet-rs target/release/argusnet-rs --fixtures realism_dense realism_dense_close
realism_dense_low_pitch realism_large realism_large_close --width 1280 --height 720
--repeats 5`.
- **Fixtures:** the two realism worlds are built natively from
  scenario configs.
  - `dense`: the `urban_dense` block grid with a power line and 1,894
    scattered trees.
  - `large`: a regional alpine map with 15,000 trees, the builder's cap.
- **Timing:** each run times 120 unpaced frames after warm-up. A sample is
  the whole frame, main and render world, including the per-frame image
  readback that headless performs, so it overstates what an interactive
  frame costs. GPU time is not recorded separately yet.
- **Quiesced:** the self-hosted CI runner shares this machine, so its queue
  was cancelled and the runner was idle for the whole measurement. Load
  averaged about 3 at both start and end. An earlier attempt with the
  runner working showed p95 up to 103 ms, a 20× worse tail; it is not
  recorded.

Data: [`r4-frame-budget-apple-m5-pro.json`](r4-frame-budget-apple-m5-pro.json),
every run's summary, warm-up, adapter, scene and image hashes.

## Frame time, ms (p50 median / p95, p99, max worst over 5 runs; frames over 100 ms)

| Fixture | Low | Standard | High |
| --- | --- | --- | --- |
| realism_dense | 2.3 / 3.0 / 3.5 / 5.0; 0 | 3.1 / 5.1 / 6.5 / 6.6; 0 | 3.7 / 5.7 / 6.2 / 6.2; 0 |
| realism_dense_close | 2.4 / 2.9 / 3.4 / 3.8; 0 | 3.5 / 4.5 / 6.8 / 6.8; 0 | 4.0 / 4.7 / 6.9 / 7.5; 0 |
| realism_dense_low_pitch | 2.3 / 4.5 / 5.3 / 5.7; 0 | 3.2 / 5.1 / 6.9 / 7.4; 0 | 4.0 / 5.8 / 6.8 / 8.0; 0 |
| realism_large | 2.6 / 3.9 / 4.6 / 5.3; 0 | 3.2 / 4.8 / 5.6 / 7.9; 0 | 3.9 / 5.9 / 6.3 / 9.5; 0 |
| realism_large_close | 2.6 / 3.6 / 3.9 / 4.4; 0 | 3.9 / 6.3 / 7.9 / 8.0; 0 | 4.6 / 6.0 / 8.0 / 10.0; 0 |

## What this does not cover

- **GPU time:** not recorded separately; the samples are whole frames.
- **Windowed and interactive frames:** headless only.
- **Reference baseline images:** the image-diff thresholds of C8 are not
  set; the image hashes are in the data for a baseline once one is
  approved.
