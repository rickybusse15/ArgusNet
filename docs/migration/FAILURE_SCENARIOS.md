# F2, first slice: scheduled sensor and link faults

**Status:** 2026-10-05. Module `argusnet-sim/src/faults.rs`; cases
`argusnet-sim/tests/fault_cases.rs`; flag `argusnet-rs sim --faults`.

## The manifest

An `argusnet.fault-manifest` (version 1) lists faults by simulated time:

```json
{
  "format": "argusnet.fault-manifest",
  "version": 1,
  "faults": [
    {"kind": "sensor_outage", "node_id": "drone-east", "start_s": 20.0, "end_s": 40.0},
    {"kind": "link_delay", "node_id": "drone-north", "start_s": 30.0, "end_s": 60.0, "delay_s": 1.0},
    {"kind": "source_restart", "node_id": "drone-west", "at_s": 45.0}
  ]
}
```

| Fault | Effect | Needs |
| --- | --- | --- |
| `sensor_outage` | The node's sensor produces nothing over `[start_s, end_s)`, then returns with a fresh bias state (it restarted). | — |
| `link_delay` | The node's evidence takes `delay_s` to arrive over the interval instead of its profile delay. | B4 delivery (`--fusion-lag-s`); delay plus jitter within the lag |
| `source_restart` | The node's transport restarts: a new source epoch, so its in-flight evidence is dropped as `delivery_stale_epoch`. | B4 delivery |
| `calibration_drift` | The node's bearing calibration drifts at `rate_rad_per_s` over `[start_s, end_s)` and stays where it got to, since nothing recalibrates it. The offset adds to the sensor's bias walk and survives an outage's restart. (Second slice, 2026-10-23.) | — |
| `mapping_outage` | The platform's mapping payload (the rig's pod LiDAR, or the range scanner) returns nothing over `[start_s, end_s)`: no ground, structure or wire evidence from it reaches the map. Its navigation camera goes on seeing. | A run that maps for itself (`--world-knowledge none`, or a profile) |
| `nav_camera_outage` | The platform's forward navigation camera, its primary obstacle sensor, sees nothing over `[start_s, end_s)`. Declared fallback: it brakes at the camera's declared deceleration to a hold in plan, keeps its height control (which reads the believed map), and goes on when the camera is back. Mission flights are delayed while held, never jumped. | A navigation camera and governed flights (`mission_flights`, as in `field_v1`) |
| `gimbal_jam` | The platform's pod gimbal holds where it is over `[start_s, end_s)`, whatever is commanded. Declared fallback, fixed pointing: its encoders still measure the attitude, so a capture counts while the boresight is within 45° of nadir and is refused (counted as unsettled) past it. | A declared rig |
| `gnss_outage` | The aircraft gets no GNSS fix over `[start_s, end_s)`. Its navigation goes on from its IMU and heading, its sigma grows, and what it maps says so. Declared fallback, with `lost_position` (`field_v3`): past a 5 m horizontal sigma it holds, and it goes home 240 s after its last fix ([NAVIGATION.md](NAVIGATION.md#when-it-is-lost)). | `--navigation` |

The manifest is validated against the run before any tick. It is refused
(`argusnet-rs` exit 3, with the reason) when:

- a fault names a node with no sensor;
- an interval is empty or reversed;
- a link fault is given without a fusion lag;
- a drift rate is zero or not finite;
- a delay doesn't fit within the lag.

Accepted manifests are recorded in the replay as `meta.fault_manifest`, so a
run is reproducible from its seed, configuration and manifest. The run is
deterministic: the same manifest gives the same replay bytes.

**How it is applied.** Outages and delays are a pure function of the tick and
are re-applied idempotently before every tick, so they need no checkpoint
state of their own. A restart happens once, at its tick; it is guarded against
a tick re-entered while paused, and its effect (the epoch) lives in the
delivery network's checkpoint. `AssembledRun::restore`, used by `--resume`,
first puts the sensors in the state the schedule has at the checkpoint's tick.
A checkpoint taken mid-outage therefore resumes exactly.

## Cases and their declared expectations

The deadlines come from the fixture's tracker configuration, not from the
runs:

- dt is 0.5 s;
- association is labeled (a track per target);
- a track is dropped after 8 stale ticks (4 s);
- a track is confirmed after 3 updates in 5 ticks (2.5 s).

| Case | Declared expectation | Result |
| --- | --- | --- |
| A. Every sensor out, 10–20 s | **Detection:** every track stale at the first outage frame. **Containment:** no track updated during the outage, and none held once the drop window has passed. **Terminal:** recovered (as restated below). | pass after restatement |
| B. `drone-east` out, 10–30 s | It contributes to nothing during the outage while the other sensors keep fusing; it contributes again within 2.5 s after. | pass |
| C. `drone-east` delayed to 1.0 s, 10–20 s, lag 1.0 s | No delivery drops; every fused observation at its acquisition time; its evidence still fused during the spike. | pass |
| D. `drone-east` restarts at 15 s with 1.0 s delay, lag 1.5 s | Some in-flight evidence dropped, all of it `delivery_stale_epoch` from that drone; fused again after the restart. | pass |
| E. A 2.0 s delay with a 1.0 s lag | Refused before the run, and so is a restart without delivery. | pass |
| F. Outage plus restart | The same replay twice. A checkpoint taken at 12 s, mid-outage, resumes to the straight run's frames exactly. | pass |

**Case A's terminal expectation was restated after its first run.**

- *Declared first:* by the end of the outage plus 2.5 s, at least as many
  fresh tracks as before. That failed: no fresh tracks by 22.5 s.
- *Why it failed:* the unfaulted run of the same seed loses both of its tracks
  at 17–21 s too. No sensor has the targets in view then; it's a natural
  coverage gap in this fixture's geometry. The expectation assumed the
  targets would be visible the moment the outage ended, and that is not the
  tracker's to guarantee.
- *Restated, against evidence and the unfaulted run:* every target is tracked
  within 2.5 s of its first accepted bearing after the outage, and the run
  ends with the same tracks as the unfaulted run. The fault's effect is
  transient. Both hold.
- The containment check also showed that tracks are deleted about 2 s into the
  outage, sooner than the 4 s stale window. The coasting limits end them first.

**End to end.** Demo config with lag 1.5 s, delay 0.25 s and the example
manifest above:

- `drone-east` contributes no fused evidence in [21.5, 41.5) s, which is the
  outage shifted by the lag, and 251 fused bearings after it;
- the restart drops 2 envelopes as stale epoch;
- `delivery_totals` reconcile;
- the replay validates against the schema.

Without delivery, the same manifest is refused (exit 3). Legacy runs are
unchanged: 14/14 replay hashes identical, `--lifecycle` equal.

## Not in this slice

These are the rest of F2 in the plan:

- degraded positioning (VIO and localization). Stage 5's VIO integrates the
  nominal path and is told every frame that it did not move, so a positioning
  fault needs its own design;
- hold, return or abort mission responses to a fault, which need F5's
  reserve logic and mission policy;
- a no-terrain-penetration check under faults: these faults don't move
  platforms, so it becomes relevant with mission responses;
- fault events in the viewer's timeline. Since the second slice, every frame
  names the faults in force (`active_faults`) and each sensor's drift offset
  (`sensor_bias_offsets_rad`).

## Second slice: calibration drift and faults in frames (2026-10-23)

Case E (`a_calibration_drift_turns_one_sensors_bearings_and_stays`) drifts
drone-east at 0.01 rad/s over [10 s, 20 s). Declared before the run, and held:
- every bearing drone-east reports from 10 s on is turned from the unfaulted
  run's by exactly the drift so far, to 1e-6 rad: 0.1 rad from 20 s, held to
  the end;
- every other sensor's observations are bit-identical to the unfaulted run;
- frames name the fault over [10 s, 20 s) only, and carry the offset from its
  first tick on.

The turn is exact because the drift and the walk rotate the same direction
about the same axis. Offsets are a pure function of the tick, re-applied before
every tick and on restore. A run resumed mid-drift or after it matches the
straight run frame for frame. A zero or non-finite rate is refused before the
run. Runs without faults are unchanged (14/14 canonical replays).

## Third slice: a mapping outage against the composed mission (2026-11-13)

The first two slices fault bearing sensors and their links, which a scan
mission does not fly on. `mapping_outage` faults what it does fly on. Case F
(`a_mapping_outage_slows_the_mission_and_nothing_else`,
`tests/field_profile.rs`) takes drone-east's mapping LiDAR out for the first
80 s of the composed mission ([FIELD_PROFILE.md](FIELD_PROFILE.md)). Declared
before the run, and held:

- detection: every frame of the interval names the fault, and the map holds
  under 80% of the unfaulted run's at the interval's end;
- containment: no world contact;
- terminal state: the mission completes with every POI done and every drone
  at its station, and the scan ends only once the map the run made meets the
  gate. As first declared, the mission also completed later than the
  unfaulted run; that held only by chance. Once terrain returns were put on
  the ray (FIELD_PROFILE.md), the faulted run's scan and mission both ended
  earlier (140 s and 207 s, against 192 s and 258 s): with the outage over
  by 80 s, which drone takes which POI decides the order. The expectation
  was restated rather than kept;
- a run checkpointed inside the outage finishes as the uninterrupted one.

The outage is a pure function of the tick, like the others. The payload still
flies its capture (gimbal, clock); only what it would have returned is
withheld. A manifest with a `mapping_outage` is refused in a run with a prior
map, which has no mapping payload.

## Fourth slice: losing the navigation camera (2026-11-15)

The composed mission avoids what it has not mapped with the forward
navigation camera. `nav_camera_outage` takes it away. Speed is capped
fleet-wide from the camera's envelope when the run is assembled, so the
fallback acts where every platform's motion already passes, stage 2's
governor: a platform whose camera is out brakes at the camera's declared
deceleration (3 m/s²) to a hold in plan. Explorers go on from where the
aircraft is once it falls behind them, a held mission flight's plan is
delayed a tick at a time, and a rejoin resumes from where it holds, so
nothing jumps when the camera is back.

Case G (`a_navigation_camera_outage_brakes_to_a_hold_and_the_mission_goes_on`,
`tests/field_profile.rs`) takes drone-east's camera out over [20 s, 60 s)
while it explores. Declared before the run, and held:

- detection: every frame of the interval names the fault, and no other;
- containment: its distance flown in plan each tick never grows during the
  outage and never falls by more than the declared braking allows; it is at
  rest within the time to brake from its cruise speed and stays at rest to
  the end of the interval; no world contact;
- terminal state: the mission completes, every POI done, every drone home;
- a run checkpointed during the hold finishes as the uninterrupted one.

Refused before the run without a navigation camera, or without governed
flights to fall back with.

## Fifth slice: a jammed gimbal (2026-11-15)

`gimbal_jam` holds the pod where it is, whatever is commanded. The sensing
guide asks for a declared fixed-pointing fallback with the gap recorded.
The pod's encoders still measure where it is, so its captures are
georeferenced as always; they count as survey evidence while the boresight
looks within 45° of nadir, and the map refuses them past that, counted as
unsettled. On target or not stays the attitude against the command, so a
pod jammed where it was already pointing does not settle again when freed.

Case H (`a_jammed_gimbal_maps_by_fixed_pointing_or_not_at_all`,
`tests/field_profile.rs`), declared before the run, and held:

- jammed at nadir while cruising (where the profile points it anyway),
  every frame's map is the unfaulted run's: the fallback loses nothing it
  should not;
- jammed from launch, still stowed and looking along the horizon, every
  capture is refused while it lasts (at least 30 more refusals by 60 s),
  the map falls behind, nothing is flown into, the mission completes with
  every POI done and every drone home, and a run resumed inside the jam
  matches;
- refused before the run without a declared rig.

Still not covered: a degraded (rather than lost) camera envelope per
platform, and a jam during a sweep (a policy `field_v1` does not use).

## Sixth slice: a lost sensor sends its drone home (2026-11-22)

Before this, a navigation camera that never came back held its drone where it
stopped until the run ended, and the mission with it. `--fault-return` (and
`field_v2`) gives the two outages an end. It needs mission flights over the
believed surface and a policy that sends drones home on their own.

| Lost | For | Then |
| --- | --- | --- |
| Navigation camera | 10 s | Home at 6 m/s or less, over what the map believes |
| Mapping payload | 60 s | Home at its own speed; its camera still sees |

- The drone brakes and holds first, as before. A sensor back sooner than that
  changes nothing.
- It goes home the way a drone at its reserve does: it gives up any POI it
  holds (`fault_handoff`, `fault_teammate`) and takes nothing again, even if
  the sensor comes back. The decision is `return_home`, with
  `nav_camera_lost` or `mapping_payload_lost`.
- The mission's own recall is flown at the blind speed too, so a drone with no
  camera no longer holds the mission open at the end.
- A drone knows how long its own sensor has been out. Nothing else of the fault
  manifest reaches the mission.

Flying home with no forward camera relies on the map: measured ground, believed
structures and the stand-in for unseen ground, at the usual clearance, with the
physical guard behind it. Something unmapped and taller than the stand-in on
the way would not be seen. Holding until the battery ran out was judged worse.

Case I (`tests/fault_return.rs`), declared before the run, and held: no world
contact; braking as declared, then never over the blind speed; home and staying
there; the mission complete with every POI done; short outages identical to the
run without the option; resume on the way home and reset exact.

West Pleasant View, `field_v2`, 16 runs (seeds 7 and 4451, each of the two
drones, each sensor, lost for good at 60 or 200 s): the faulted drone is home at
the end of every one, with no world contact and no step over 9.4 m. Under
`field_v1` the one case run (seed 7, camera lost at 60 s) ends with the drone
still holding 945 m from home. With two drones, the one left usually cannot
finish the scan in 600 s: 2 of the 16 missions complete. The response saves the
drone, not the schedule.

Not covered: a fault return when every drone has gone home mid-scan (the
mission stays scanning), and a jammed gimbal, which keeps its fixed pointing.
