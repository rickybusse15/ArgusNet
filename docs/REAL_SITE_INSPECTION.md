# Reproducible real-site inspection workflow

The checked-in West Pleasant View, Colorado sources can now pass through one
workflow: import around one origin, simulate a self-built map, produce a client
package, then open its source-bound report in Review.

From the repository root, with the project Python active:

```bash
cargo build --locked -p argusnet-cli
python scripts/real_site_inspection.py --output runs/real-site --seeds 7 42
cargo run --locked -p argusnet-viewer -- --scene runs/real-site/seed-7/review.smartscene
```

To verify a package's report through the Rust reader without opening a window:

```bash
cargo run --locked -p argusnet-replay --example verify_assurance -- runs/real-site/seed-7/client-package/replay.json
```

Use a new output directory for each campaign. The script preserves both runs of
each seed, requires byte-identical replays, validates the replay schema, verifies
every client-package artifact hash, and builds each Review scene from the
package's annotated replay. It uses `contingency_v1` planning and `flown_v1`
energy accounting. `results.json` records the world ID, source commit, binary
hash, replay hashes, actual mission outcomes, map statistics, energy and missing
report fields. A zero exit status means those reproducibility and contract
checks passed; it does not mean the mission completed or the map met a quality
target. Interrupted or failed campaigns leave their partial evidence in place.

Choose `--duration-s` to change the observation window and `--binary` to select
an explicitly built release executable. Timings from a debug executable must
not be used as production performance claims.

## With thresholds: `--profile field_v1`

```bash
cargo build --release --locked -p argusnet-cli
python scripts/real_site_inspection.py --output runs/real-site-field \
  --profile field_v1 --binary target/release/argusnet-rs --seeds 7 42
```

With a profile, every run flies it (600 s by default, long enough for the
mission to complete) and is scored against the profile's frozen thresholds
(`docs/contracts/field_v1.thresholds.json`): mission complete, every POI done,
drones home, no world contact, map coverage and accuracy, wire clearance,
energy reserve and reproducibility. `results.json` carries each gate's value
and verdict, and the exit status is zero only when every seed passes every
gate. That is one site and the seeds given;
`scripts/qualify_field_profile.py` runs the predetermined and held-out seeds
and is the qualification ([FIELD_PROFILE.md](migration/FIELD_PROFILE.md)).

## What remains

Localization integrity, landing and inspection evidence sufficiency are not
established, and most imported building heights are defaults. The rig, LiDAR
and camera models are assumed, not calibrated. Independent scheduling beyond
the lane rates, hardware qualification and Plan remain separate work.
