# F5, first slice: energy contingencies (`contingency_v1`)

**Status:** first slice, 2026-10-23. Opt-in: `argusnet-rs sim --planning-policy
contingency_v1` (also on the viewer's `--run`). Every other policy's replays are
unchanged. The one exception is a fix to `efficient_v1` (below).

## What `efficient_v1` already did

P1–P3 ([PLANNING_EFFICIENCY_PLAN.md](../PLANNING_EFFICIENCY_PLAN.md)) gave the
mission driver most of F5's machinery:
- a drone at its dynamic reserve home flies home on its own (`return_home`,
  `reserve_reached`);
- a POI a drone could not finish and fly home from is never assigned to it;
- a low drone hands its POI to a receiver that can afford it, or releases it
  to the pool;
- a team join must leave the newcomer its reserve.

What it lacked against [CAPABILITY_DELIVERY.md](../CAPABILITY_DELIVERY.md) F5:
- **Giving up was a fixed fraction, not feasibility.** A holder handed its POI
  on below 30 % of capacity, whatever was left to do. Taking new work only
  needed the reserve. So a drone just above 30 % was given a POI it could
  easily finish, then dropped it a few seconds later as it crossed 30 %. On the
  scan fixture at 152 Wh (30.4 %), both drones took their POIs and released
  them 9 and 12 ticks later; nothing was inspected, and the mission was still
  inspecting when the run ended.
- **Unfinishable work was silent.** A POI no drone could afford stayed
  `pending`, and the mission waited for it until the run's time ran out.

## What `contingency_v1` changes

It is `efficient_v1` with three changes, all in stage 9
(`argusnet-sim/src/mission_driver.rs`):

| Rule | `contingency_v1` |
| --- | --- |
| Give work up | Checked every run: give up the POI when what is left of it can no longer be afforded. That is the trip there (none once in dwell range), plus this drone's share of the remaining dwell with the holders already in range, plus the reserve home from the POI. There is no fixed fraction. Reasons: `cannot_finish_handoff` and `cannot_finish_teammate`, or `no_affordable_receiver` when nobody can take it. |
| Take work | The same affordability test, plus a margin of 5 % of capacity (`CONTINGENCY_TAKE_MARGIN_FRACTION`) above the reserve. This covers fleet allocation, handoff receivers and team joins. The band between the take rule and the give-up rule is the hysteresis. A drone that spends energy as its plan said keeps the work. Only something the plan did not price moves it, such as a headwind, a longer way home or a lower charge. |
| Unfinishable work | After allocation, any pending POI that no drone still working (not homeward) could afford from where it is now, with the margin, is marked unassigned. This is a `poi_unassigned` / `no_affordable_drone` decision, with `affordable_again` if that changes. The replay's POI status carries `unassigned_reason: "no_affordable_drone"`, and the scanning→egress gate treats such a POI as closed, so the mission goes home instead of waiting. F1's outcome lists the POI under `objectives.unassigned_pois` and cites the reason in `pois_not_inspected`. |

The unassigned set is checkpointed (absent when empty). A `contingency_v1`
checkpoint resumes only into a `contingency_v1` run. Energy authority stays in
the runtime: the viewer computes no reserve.

## The fix to `efficient_v1`

When the mission's egress ended, `ClearAllEgress` dropped every recall at
once, including that of a drone sent home on its own reserve. That drone is
never sent home twice, so it went back to its lanes and flew until the battery
was empty. Both efficient policies now fly such a drone home again from where
it is (`return_home` / `reserve_return_kept`). Reference runs are untouched.

## Evidence

`rust/argusnet-sim/tests/contingency.rs`: the scan fixture under
`physical_v1`, both drones set to a given charge on the first inspecting tick.

| Case | `efficient_v1` | `contingency_v1` |
| --- | --- | --- |
| 152 Wh | takes poi-00 and poi-01, releases both 9–12 ticks later; 0/3 inspected; still inspecting at the end | no POI given up; 3/3 inspected; complete |
| 160 Wh | 2/3 (poi-02 handed off, then released) | 3/3 |
| 117.5 Wh | 0/3; still inspecting at the end, POIs `pending` with no reason | 3 `poi_unassigned` decisions; mission complete; every POI `pending` with `unassigned_reason` |
| east cut to 100.5 Wh on the way to poi-02 | — | `no_affordable_receiver` at once (north is still busy), north takes poi-02 next run and finishes it; east turns home 3 ticks later on its reserve, and its return survives the egress clear |
| checkpoint at tick 60 with POIs unassigned | — | the resumed frames equal the straight run's; an `efficient_v1` run refuses the checkpoint |

`tests/test_mission_outcome.py` covers F1's reading of the reason.

## Findings, not fixed here

- **Energy is charged on the nominal controller position, not the flown one.**
  Stage 1 accounts battery before stage 2's overrides (the preserved causal
  order, [ARCHITECTURE_AND_PROTOCOLS.md](ARCHITECTURE_AND_PROTOCOLS.md) stage
  1). A drone held at home by an egress is still charged for the lanes its
  controller would fly: on the scan fixture that is 0.25–0.65 Wh/s while it
  sits still, against 0.083 Wh/s of hover. The planner's predictions price
  the flown route, so they and the ledger disagree for an overridden drone.
  Charging the flown position is a declared behaviour change. It belongs in its
  own energy profile, and is F5's next slice.
- **No landed state.** A drone that is home keeps drawing hover power under
  `physical_v1`. A long run drains it whatever the planner does.

## Open

- Charging the flown position (above), then re-measuring the margin.
- The Operate reserve/contingency inspector and Plan feasibility view: the
  decisions and `unassigned_reason` are their inputs.
- Qualification over seeds (F4's `compare_scenarios.py`): `contingency_v1`
  against `efficient_v1` on POI completion and reserve violations.
- F2's hold/return/abort responses, which can now use the same feasibility test.
