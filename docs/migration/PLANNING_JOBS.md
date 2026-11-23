# Planning jobs (P4.10d and P4.10e, first slices)

**Status:** shipped, opt in. `rust/argusnet-sim/src/scheduler/jobs.rs` (the board),
`reroute.rs` (its first use), `argusnet-rs sim --profile field_v3 --replan-routes`.
Off, nothing changes: `field_v3`'s replays are the campaign's.

## The board

Work that takes longer than a tick is submitted on one and applied on a later one,
declared when it is submitted.

- **Deterministic application.** A result is taken at its declared tick and no
  other: early ones wait, late ones are waited for. Results due together are taken
  in the order they were submitted, whatever order the workers finished in.
- **Bounded.** Past its capacity a submission is refused, not queued, and there is one
  job a key. The caller keeps what it had.
- **Input revisions.** A job carries the revision of what it was worked out from.
  Whether that is still good enough is judged when it lands.
- **Cancellation.** A cancelled job's result is never taken, finished or not.
- **Checkpoints with jobs pending (P4.10e).** A checkpoint holds the pending jobs'
  results, finishing them first. That changes nothing, since their inputs are fixed
  and their ticks still to come, and a resumed run needs none of the inputs.

Deadlines here are simulated time. What a live run does with a job that is late
on its hardware is S3/S4's.

## Its first use: flights planned again

A mission flight whose way was open when it was set is watched. Every `check_s`
(1 s) the way it has left is tested against the map as it is now (P4.10c's affected
routes). One that something newly mapped lies across is planned again on that map,
as a job due `latency_s` (0.5 s) later. It is flown only if the drone is still on
that flight and the way is open on the map as it is *then*. A flight set anew or
dropped cancels its job. Each landing is a `route_replanned` decision.

## Measured (2026-11-23)

The seven tuning seeds on each site, `field_v3` with and without: every gate passes
either way, 14 of 14. Eight of the 14 missions are unchanged to the figures printed.
The others differ by up to 47 s (one, West Pleasant View 1010, 530 to 483 s) and a
few metres of wire clearance either way. On these mild sites a flight's way is seldom
closed under it; it is not in `field_v3`, which was qualified without it.

## Verification

- `jobs.rs` (5): order at 1, 2 and 4 workers, bounds, cancellation, resuming with
  jobs running, malformed state. `reroute.rs` (5): a tower mapped on the way is gone
  round on the declared tick; a way closed since is not flown; cancellation; resume;
  off.
- `tests/field_v3.rs`: the composed run is the same at 1 and 4 workers, and a
  checkpoint taken while planning is pending resumes exactly, at two points.

## Not in these slices

- The mission stage still runs on its lane's boundary: a job cannot interrupt one.
- Delayed sensor evidence and jobs share no queue, and the scheduler's own history is
  not recorded (the rest of P4.10e).
- View planning, map reports and the localizer are still worked out in their tick.
- P4.10f: `scheduled_v1` is not the default, and this does not argue that it should be.
