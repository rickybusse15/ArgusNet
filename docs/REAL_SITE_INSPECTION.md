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

The remaining acceptance work is to declare and qualify coverage, reconstruction
error, inspection evidence, energy reserve and failure-response thresholds over
held-out seeds. Some scenario setup still consults truth, most imported building
heights are defaults, and localization integrity, landing and inspection evidence
sufficiency are not established. Independent scheduling, hardware qualification
and Plan remain separate work.
