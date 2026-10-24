# Navigation camera as the primary avoidance sensor (SR5c, first slice)

**Status:** first slice shipped behind `--nav-camera` in `--world-knowledge none`
runs. Spec: [sensing realism guide §8.4](../SENSING_REALISM_IMPLEMENTATION_GUIDE.md).
Code: `rust/argusnet-sim/src/nav_camera.rs`, the safety layer in
`belief_world.rs`, the speed caps in `controller.rs`.

## What changes with the flag

| Piece | Without `--nav-camera` (unchanged) | With it |
| --- | --- | --- |
| Forward sensing | The range scanner's 9-ray forward fan | A body-fixed camera: 90° × 60° view, 10° below the horizon, 19 × 9 sight lines |
| Range scanner | Rings, nadir and forward fan | Rings and nadir only (`forward_count: 0`): it maps, it no longer looks ahead |
| Obstacle evidence | Mapped structure cells | Mapped structure cells **or** camera-occupied cells (`BeliefSnapshot::collides`) |
| Map figures, frontier, known fraction | From the scanner | Still from the scanner only: camera cells are never "measured" |
| Speed | Each controller's own | Every lane, explorer, follow orbit, POI redirect and rejoin capped to the envelope's `max_speed_mps` |
| Replay | — | `meta.world_knowledge.exploration.nav_camera`, `meta.world_knowledge.nav_camera_envelope`, camera frames in `range_scans` with `sensor: "nav_camera"` (drawn green in the viewer) |

`--nav-camera-profile FILE` replaces the default profile with a
`NavCameraConfig` JSON document. Both flags are refused in a prior run.

## The camera model

The camera is a perception model, not an image renderer:

- **Detection.** A sight line that meets a building, wall or trunk within
  `structure_range_m` (150 m by default), or a crown within `canopy_range_m`
  (100 m), is detected with probability `detection_probability` (0.9) per
  frame.
- **Depth error.** One sigma is `depth_noise_per_m` × range²: about 5 m at
  100 m.
- **Wires.** A wire is detected while it still spans `wire_min_pixels` (1.5)
  pixels of `pixel_ifov_rad` (0.8 mrad). That is 25 m for a 3 cm conductor and
  17 m for a 2 cm one, and only inside the view.
- **Not modelled.** No ground detections: terrain clearance stays with the
  belief terrain. No false alarms (assumed zero). The object a sight line meets
  is taken from the world exactly; a real system detects it from pixels.
- **Heading** is the horizontal velocity, as for the scanner's old forward
  fan: the sim has no separate yaw.

Every number is tagged `evidence: "assumed"`. None is a selected device.

## The envelope and the speed bound

`NavCameraConfig::envelope` shortens each detection range to where the
atmosphere's apparent contrast reaches the camera's threshold (Koschmieder:
`ln(C0/Cth) · V / 3.912`, with the bundle's precipitation-reduced
visibility). Clear air at 15 km changes nothing. At 60 m visibility the
structure range falls to 35 m.

The cap solves `v · latency + v² / (2 · braking) = bounding range − margin`.
Defaults: 0.15 s latency, 3 m/s² braking, 10 m margin. That gives 28.5 m/s at
150 m, and 11.8 m/s in 60 m fog. The bound covers structures. Wires are
reported (`wire_range_m`) but bound speed only when the profile sets
`bound_speed_by_wires`, because a one-camera wire range bounds speed to about
6 m/s. A run that flies faster than its wire range allows therefore says so in
its metadata.

## Measured (2026-10-23, `tt` fixture, 40 m wall across the map, downward pod rings 80/65/50°)

| Run | Structure contacts | Camera detections |
| --- | ---: | ---: |
| Default scanner (with forward fan) | 8 | — |
| Downward pod, no camera | 14 | — |
| Downward pod + camera | 10 | 2296 |

Contacts are not zero with the camera, because the belief planner only pushes
clear of what it believes; routing on the belief is P3 work. The gate
(`tests/nav_camera.rs`) is that the camera does no worse than the pod alone.

## Findings outside this slice

1. **Structures are not swept.** `PhysicalGuard::enforce_from` sweeps a step
   against wires only. A step longer than a structure is thick passes through
   it with no contact recorded: on the `tt` fixture a drone stepped 18 m
   through a 10 m wall. The tests use a 40 m wall for that reason.
   `first_structure_hit` already exists for the sweep.
2. **A push can land on the far side.** `enforce` pushes a platform inside a
   structure to the nearest clear cell, which for a platform past a wall's
   midline is the far side. Combined with (1), a thin wall can be crossed.

## Not in this slice

Degraded-camera faults (outage, glare, night) beyond weather; holding or
returning when the envelope collapses; direction limits (turning into space
the camera has not cleared); pod pointing and the arbiter, which need SR1/SR2.
