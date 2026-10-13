# B4 (P4.10b), first slice: timestamped delivery between capture and fusion

**Status:** 2026-10-04. Module `argusnet-sim/src/delivery.rs`; fusion stage
`argusnet-sim/src/fusion.rs`; CLI flags on `argusnet-rs sim`.

## Why

In the legacy tick an observation is captured, delivered and fused in the same
tick. A real sensor's evidence arrives later than it was captured, by an amount
that depends on the source, and sometimes out of order. Fusion must use the
time the evidence was **acquired**, account for every item it could not use,
and age tracks through intervals with no evidence. F2 (sensor and link failure
scenarios) builds on this.

## Design

**Envelopes.** Stage 6's observations are wrapped in a `SensorEnvelope`:

- source id;
- per-source sequence;
- source epoch (a restart begins a new one);
- acquisition, availability and receipt ticks;
- a capture index that keeps a tick's evidence in production order.

**Delay model.** A `DeliveryProfile` gives each source a delay (a default plus
per-source overrides) and a jitter drawn uniformly per envelope. The draws are
keyed by `(seed, source, epoch, sequence)`, so jitter reorders arrivals but a
run is deterministic for a fixed seed and profile. Times are whole ticks;
seconds are rounded up.

**Fixed-lag fusion.** The tracker (`argusnet-core`) is a forward-only filter.
It triangulates a batch at one time and cannot take a measurement older than
its last update. Delayed-measurement smoothing is out of scope (DELIVERY_PLAN
§6). So fusion runs a fixed lag `L` behind capture: at tick `t` it ingests
exactly the evidence acquired at `t - L`, at that tick's own time.

- Every acquisition tick is ingested once, with or without evidence, so the
  tracker's lifecycle ages tracks by elapsed acquisition time through empty
  intervals.
- Nothing is ever fused out of order.
- Before the first horizon (`t < L`) nothing can have been delivered. Fusion
  ingests an empty batch at the current time; no track exists yet, so nothing
  ages at the wrong time.
- Snapshot tracks are estimates as of `t - L`. Their timestamps say so, and an
  error measured against current truth includes the latency.

**Refused before the run.** A profile whose largest delay plus jitter exceeds
`L` would drop evidence by construction. `DeliveryProfile::validate` refuses
it, and `argusnet-rs` exits 3 (unsupported) with the reason before any tick.

**Nothing lost silently.** An envelope that cannot be fused is dropped with a
reason:

| Code | When |
| --- | --- |
| `delivery_late` | it arrived after fusion had passed its acquisition tick |
| `delivery_duplicate` | its `(epoch, sequence)` was already delivered from that source |
| `delivery_stale_epoch` | it was sent before its source restarted |
| `delivery_overflow` | its source already had `queue_capacity` envelopes queued |

Each drop is appended to the frame's `rejected_observations`; the four codes
are in the replay schema and in CONTRACT_VOCABULARIES.md. The replay's
`meta.delivery_totals` accounts for every captured envelope: `captured = fused
+ dropped + in_transit_at_end`.

**Legacy is untouched.** With no profile, fusion runs exactly as before:
14/14 canonical replay hashes are unchanged, and `--lifecycle` is equal. An
all-zero profile, which goes through the network, reproduces the legacy run
frame for frame. That is the evidence the network itself changes nothing it
should not.

**Checkpoints.** With a profile, the fusion checkpoint holds the engine and
the network: envelopes in flight and received, sequences, epochs, dedup
windows and totals. A checkpoint taken with evidence in transit resumes
exactly. A run without delivery refuses one with it (and the reverse), as does
a run with another profile.

## Using it

```bash
argusnet-rs sim --config run.json --replay out.json \
    --fusion-lag-s 1.25 --delivery-delay-s 0.25 --delivery-jitter-s 0.5 \
    --source-delay ground-alpha=0.75 --delivery-queue-capacity 4096
```

The profile is recorded in `meta.delivery_profile`.

## Evidence

**Tests.** `delivery.rs` unit tests (9):

- zero profile in capture order;
- capture vs delivery vs receipt times;
- duplicate;
- source restart / stale epoch;
- late;
- overflow;
- jitter reorders arrivals but not what fusion sees;
- profile refusal;
- checkpoint round trip.

`tests/delivery_profiles.rs` (6):

- zero profile equals legacy;
- a delayed profile fuses each tick at its acquisition time,
  deterministically, losing nothing;
- checkpoint with evidence in transit resumes exactly;
- an unsupported profile is refused;
- after the sensors go quiet, tracks expire exactly `L` ticks after the
  legacy run's;
- the replay accounts for every envelope.

`replay_vocabulary.rs` checks the drop codes against the schema exhaustively.

**Demo** (`civilian-observation-small-mixed-mixed`, 120 s, dt 0.25 s):

| Run | Fused bearings (replay) | Tracks at end | Error vs current truth | `delivery_totals` |
| --- | --- | --- | --- | --- |
| legacy | 4677 | 3 | 10.0 m | none |
| lag 1.25 s, delay 0.25 s, jitter 0.5 s, ground-alpha 0.75 s | 4615 | 3 | 36.8 m | captured 4853 = fused 4790 + in transit 63 |
| lag 1.0 s, delay 0.75 s, queue capacity 1 | 276 | 2 | 39.6 m | captured 4853 = fused 706 + overflow 4140 + in transit 7 |

The delayed run's extra error is mostly the lag itself: targets move about
25 m/s, and its tracks are estimates as of 1.25 s ago. All three replays
validate against `docs/replay-schema.json`.

## Evidence age and track error

Fixed-lag fusion publishes each track as of its evidence time: a track's
`timestamp_s` is the fusion horizon, `fusion_lag` before the frame. Scoring it
against the frame's truth mixes up two different costs. The first is how good
the estimate is. The second is how old it is. The evaluation now reports them
separately:

- `localisation_rmse_m` scores each track against truth at the track's own
  time. This is the estimator's error.
- `present_error_rmse_m` scores it against truth at the frame time. This is
  what a consumer gets if it takes the track as the present.
- `track_evidence_age_mean_s` is the mean of frame time minus track time.

The two errors are equal without delivery modelling.

The demo scenario (`small` map, 4 drones, 3 `mixed` targets, seed 7, 120 s,
dt 0.25 s) at increasing fusion lag:

| Fusion lag | `localisation_rmse_m` | `present_error_rmse_m` | Evidence age |
|---|---|---|---|
| none (legacy) | 13.5 m | 13.5 m | 0 s |
| 0.25 s | 13.5 m | 18.6 m | 0.25 s |
| 0.5 s | 13.5 m | 24.7 m | 0.5 s |
| 1.5 s | 13.6 m | 47.6 m | 1.5 s |

So the estimate does not degrade with lag. The present error is the targets'
motion over the evidence age.

**Projecting tracks to the frame time was tried and rejected.** Each track was
predicted forward over the lag with the tracker's own IMM filter. That made the
present error worse: a mean of 47.7 m against 41.8 m without projection, at
1.5 s lag. (These are mean errors over tracks and frames; the table above gives
the RMSE, which is 47.6 m without projection at the same lag.) There are two reasons:

- The filter's velocity is poor for these targets. Its mean error is 18.6 m/s
  in the legacy run, against target speeds of about 17 m/s.
- The targets manoeuvre hard. Even extrapolating with the *true* velocity at
  the evidence time misses by 21.9 m on average over 1.5 s.

Scaling the projection back to a quarter or half of the predicted motion
recovered only 1–2 m. A useful present-time estimate needs a better motion
model or faster evidence, not extrapolation. The published tracks therefore
stay at their evidence time, labelled with it, and consumers can see
`now - timestamp_s`.

## Evidence time on the track

`timestamp_s` says when a track's estimate is for, not when it last saw
evidence: a track that coasts is predicted forward over ticks with no
measurement. Each runtime track now records its last measurement time
(`TrackState::evidence_time_s`, from the tracker's `last_update_time_s`, which
the engine checkpoint already carried).

A replay writes it as each track's `evidence_time_s` only in a run whose
tracks can be older than their frame: delivery modelled, or the fusion lane
at its own rate (`AssembledRun::tracks_lag_their_frames`). Legacy replays are
byte for byte unchanged. For a track updated at the fusion horizon it equals
`timestamp_s`; for a coasting one it is earlier. A frame published while the
fusion lane sits out a boundary republishes the last tracks, and their
evidence times say how old they are.

The evaluation reports `track_measurement_age_mean_s`: the mean of frame time
minus `evidence_time_s`. That is `track_evidence_age_mean_s` (frame time minus
track time) plus the time tracks spent coasting. It is `null` for a replay
that does not record evidence times. The viewer's track summary adds an
"Age (s)" column when the frame's tracks record them.

The gRPC wire does not carry it. `TrackState.last_seen_s` on the wire stays an
alias of `timestamp_s` (FIELD_PARITY.md); giving it the evidence time would
change what the Python client receives, which is a contract change of its own.

## Not in this slice (rest of P4.10b)

- Buffered mapping inputs with batch-size and max-age triggers (no lost
  evidence when sensing outruns mapping).
- Per-source capture schedules (rates other than the tick).
- Clock quality and frame or calibration identity on the envelope.
- Evidence time on the gRPC wire (`last_seen_s`). Source identity is already
  on every track (`contributing_node_ids`), and the evidence time now is in the
  replay (above).
- Delayed-measurement smoothing, which would remove the lag's cost. It is a
  separate estimator capability.
