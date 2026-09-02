# `replay_contract_fixture.json`

Real output from `argusnet sim`, trimmed so it can live in the repository, used
as the shared artifact for the cross-language replay contract test.

Replay JSON is the interface between the Python simulation and the Rust viewer,
but until now nothing checked that the two agreed. They did not: the viewer
declared `mode_probability_cv` and `contributing_node_ids` that Python never
wrote, Python wrote `battery_fraction` that the viewer had no field for, and a
`null` safety `reason` — which `docs/replay-schema.json` explicitly permits —
aborted deserialization of the whole document.

Two tests read this one file:

- `tests/test_replay_contract.py` — the **writer** side. Validates the fixture
  against `docs/replay-schema.json` with `jsonschema`, and asserts the simulation
  still emits every field the viewer reads.
- `rust/argusnet-viewer/src/replay.rs` (`contract_fixture_tests`) — the **reader**
  side. Deserializes it and asserts the viewer-facing fields actually populate.

A change on either side that breaks the other fails one of them.

## Regenerating

```bash
python -m argusnet sim \
  --map-preset small --terrain-preset alpine --mission-mode target_tracking \
  --drone-count 2 --target-count 1 --ground-stations 3 --duration-s 12 \
  --seed 7 --replay /tmp/fx.json
```

Then trim: keep the first three frames that carry tracks or observations, drop
`generation_rejections` from each (prose `detail` strings dominate replay size
and are not part of this contract), and remove `meta.terrain.viewer_mesh` (a
~70 KB heightmap covered by the terrain tests instead). Set
`meta.frame_count` to the retained count.

Field *shapes* are exactly as the writer emits them — only whole frames and
whole keys are removed, never rewritten.
