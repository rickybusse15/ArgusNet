# Sensor rig and moving pod (SR1, SR2, first pointing slice)

**Status:** first slice shipped behind `--rig-profile` (and `--pointing`) in
`--world-knowledge none` runs. Spec: sensing realism guide §1 (decisions),
§3, §4 and §8.4. Code: `rust/argusnet-sim/src/rig.rs`, `pod.rs`, the refusals in
`belief_world.rs`, and the height caps in `controller.rs`, `waypoint_path.rs`,
`follow_path.rs` and `trajectory.rs`.

## Decisions it builds on (maintainer, 2026-10-23)

| Item | Decision |
| --- | --- |
| Gimbal | 3-axis, no joint limits |
| Pod | LiDAR, two visible cameras, thermal and the mapping IMU |
| LiDAR | Fixed rectangular field of view; its size is unknown, so the profile assumes one |
| Flight | Cruise only for sensing; hover is for take-off and landing |
| Not yet decided | LiDAR field of view, camera roles (stereo or wide/detail), cruise dynamics |

## The rig profile (SR1)

`RigProfile` v1 declares:

- **Sensors:** id, role, attachment (body or pod), lever arm, mount rotation,
  model version, and the model (LiDAR, camera optics, or navigation camera
  profile).
- **Gimbal envelope:** slew rate, settle time and tolerance, attitude noise,
  stow attitude, airframe mask.
- **Calibration:** epoch, and the LiDAR's true boresight bias, which the
  estimator does not know.
- **Mass ledger.**
- **Evidence tag:** `assumed` for every number in `RigProfile::provisional()`.

Validation runs before any tick:

- version, unique ids, finite values, attachments (the navigation camera on the
  body, everything else on the pod), exactly one LiDAR and one navigation
  camera, each model valid;
- the **10 lb scanning assembly** and the **55 lb take-off** limits. Compute,
  storage, cooling and batteries count toward take-off only.

A rig and a stand-alone `--nav-camera` are refused together.

The replay records the rig in `meta.world_knowledge`:

- the profile;
- `rig_identity`: FNV-1a 64 of its canonical JSON;
- `rig_mass_kg`;
- `rig_sensors`: which sensors are **simulated** (LiDAR, navigation camera)
  and which are **declared, not simulated** (visible cameras, thermal, IMU,
  until SR4).

Provisional mass: 2.65 kg scanning assembly, 23.45 kg take-off.

## The pod (SR2)

- **Pose chain.** `T_world_sensor = T_world_body · T_body_base · R_base_pod(t) · T_pod_sensor`.
  Each mount has its own lever arm. The body has heading only: the sim has no
  roll or pitch.
- **Gimbal.** Its attitude is a quaternion in the body frame. With no joint
  limits, nothing needs joint angles, and there is no singularity at nadir. It
  turns along the shortest rotation at `max_rate_rad_s` and is **settled** after
  holding within tolerance for `settle_s`. Its state is checkpointed with stage
  6.
- **Truth vs estimate.** Rays are cast from the true pose (actual attitude,
  true mount, boresight bias). Returns are georeferenced with the estimated
  pose (measured attitude, nominal mount). Bias and attitude noise therefore
  show in the map. A 20 mrad pitch bias takes ground RMSE from 1.41 m to 2.08 m
  on the scan fixture.
- **LiDAR.** A fixed rectangle: assumed 70° × 30°, 24 × 10 rays, 250 m, 5 cm
  noise. At nadir, 70° lies across the track. Rays at or above 5° below the
  body's horizon are blocked by the airframe and counted as occluded.

## Pointing and arbitration

The planner proposes a `PointingPolicy`:

- `nadir` (default);
- `sweep:AMPLITUDE_DEG:PERIOD_S`: rolled side to side about the track;
- `body:YAW,PITCH,ROLL`.

The arbiter commands **stow** below cruise (under 0.5 m/s: take-off, landing,
a hold) and **mission** otherwise. No safety rule takes the pod: the
navigation camera is the avoidance sensor.

The map refuses **stowed** captures, whose rays graze the ground far ahead,
and **unsettled** ones. It counts both: `mapping_state.belief.stowed_scans` and
`unsettled_scans`.

## The planner flies within the LiDAR's reach

`PodSensing::max_mapping_agl_m` gives the highest the aircraft may fly so that
the policy's steepest ray over its whole cycle still reaches the ground:
`max_range · cos θ_max · 0.9 − unknown_margin`. Every lane, explorer, follow
orbit and launch climb is capped to it, swing included. The cap is recorded as
`mapping_agl_cap_m`: 148 m for nadir and 79 m for a 25° sweep with the
provisional rig.

Without the cap, the scan fixture's lanes (230 m above believed ground) took
the pod above its range: 21% known, and the scan never finished.

## Finding fixed here: overrides could leave a platform underground

Stage 2 applied POI and egress override positions without the physics guard;
only rejoin steps were guarded. A flight at mapping height over hills went
into them: 51 m below the ground at worst on the scan fixture. Now an override
position that is below the true ground, inside a structure or inside a wire is
put back through `PhysicalGuard::enforce` and recorded. A landing within the
guard's clearance is left alone, so no-prior runs that never went underground
are byte-for-byte unchanged. Overrides are applied in node order, so contacts
record deterministically.

The fix also reaches **prior-map runs under `efficient_v1`**, whose planning
policy installs the same guard. On the scan fixture two egress legs there ran
into terrain at tick 87 (3.0 m and 11.5 m). They are now held above it and
recorded. The 14 canonical replays use the reference policy and are
unchanged.

## Measured (2026-10-23, `scan` fixture, `--world-knowledge none`)

| Run | Known | Ground RMSE | Buildings found | Mission completes |
| --- | ---: | ---: | ---: | ---: |
| Legacy scanner | 99.9% | 0.65 m | 100% | tick 100 |
| Rig, nadir | 100% | 0.72 m | 100% | tick 96 |
| Rig, 25° sweep | 100% | 0.83 m | 100% | tick 81 |

## Not in this slice

- Visible, thermal and IMU products (SR4). Viewer frusta (SR6).
- Body roll and pitch, vibration and backlash, and a position estimator in the
  georeferencing chain.
- **View planning (SR5d):** choosing pointing from the deficit map (SR5a), and
  energy-weighted views.
- **Cruise-only inspection.** POI dwell is still a hover, so the pod stows
  through it and maps nothing there. Inspection should become cruise passes or
  orbits once cruise dynamics are known.
