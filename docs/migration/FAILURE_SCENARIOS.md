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

The manifest is validated against the run before any tick. It is refused
(`argusnet-rs` exit 3, with the reason) when:

- a fault names a node with no sensor;
- an interval is empty or reversed;
- a link fault is given without a fusion lag;
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

- calibration drift faults;
- degraded positioning (VIO and localization);
- hold, return or abort mission responses to a fault, which need F5's
  reserve logic and mission policy;
- a no-terrain-penetration check under faults: these faults don't move
  platforms, so it becomes relevant with mission responses;
- fault events as replay frames: today the manifest is recorded, and each
  drop with its reason.
