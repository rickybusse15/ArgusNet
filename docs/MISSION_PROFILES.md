# Civilian Mission Profiles

ArgusNet mission profiles are additive, declarative presets for domain workflows. They configure
existing simulation, planning, replay, sensing, and evaluation primitives; they do not implement a
separate fusion stack or domain-specific autonomy.

## Included profiles

`sar_person_search` models authorized missing-person search and localization. It selects target
tracking with a deterministic loitering person target, search-mode drones, an objective search
zone, optical and thermal capability requirements, an acquisition deadline, and localization
confidence criteria. It does not diagnose injuries or perform autonomous triage.

`industrial_asset_survey` models authorized fixed-asset mapping and inspection. It selects the
scan-map-inspect runtime, deterministic optical/thermal/any inspection POIs, a coverage threshold,
an exclusion zone, dwell requirements, and safe egress. It does not classify defects.

Run either profile with:

```bash
argusnet sim --mission-profile sar_person_search
argusnet sim --mission-profile industrial_asset_survey
```

Explicit compatible simulator flags override profile defaults:

```bash
argusnet sim --mission-profile sar_person_search --drone-count 4 --weather-preset fog
```

Contradictory settings fail before scenario construction. For example, SAR requires at least one
target, while industrial survey uses fixed POIs and requires zero moving targets.

## Extension points

- Add runtime objective/state-machine behavior behind the existing mission executor.
- Add victim or asset observation semantics additively to replay and protobuf only when consumers
  are updated together.
- Present profile objectives, sensor requirements, and completion state in the viewer.
- Add domain metrics without replacing generic coverage, localization, safety, or track metrics.
- Map real optical and thermal adapters onto the declared sensor capability requirements.

Rust remains the source of truth for fused tracks. Profile code must not duplicate fusion math.
