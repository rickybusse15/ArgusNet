#!/usr/bin/env python3
"""Capture the Python simulation's per-tick stage order, mechanically.

P4.1 asks for a machine-readable stage trace against the Python reference
*before* the orchestrator is extracted, so that "did the Rust port preserve the
causal order?" is a question with an answer instead of a side-by-side read of
two implementations.

This script does **not** edit ``sim.py``. It wraps the methods each stage
characteristically calls and records the order they fire in, per tick. Two
reasons for observing rather than instrumenting:

1. ``run_simulation`` is ~1,300 lines and its output is pinned by bit-identical
   golden files. Additive tracing calls would be low-risk but not no-risk, and
   the risk is borne by every future run, not just this one.
2. A trace derived from what the code *does* is better evidence than a trace
   derived from where someone remembered to put a probe. If a stage stops
   calling its anchor, the trace changes, which is the signal we want.

What this cannot see
--------------------
A stage that runs but does no work makes no anchor call, so it is absent from
the trace rather than recorded as "ran, nothing to do". The Rust harness draws
that distinction (``StageOutcome::NoWork`` versus never appearing) and this
capture cannot, so **absence here means "did no work", not "did not run"**.
Comparing the two therefore compares the order and presence of *active* stages.
That is the property §4 constrains; it is not the whole trace.

Usage
-----
    python3 scripts/capture_stage_trace.py --path default --out trace.json
    python3 scripts/capture_stage_trace.py --all --out-dir tests/fixtures/stage_traces
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


@dataclass
class Probe:
    """One stage anchor: the call whose occurrence means the stage did work."""

    stage: str
    """Matches the Rust `Stage` rendering in argusnet-sim/src/stage.rs."""
    module: str
    qualname: str
    """`Class.method` or a bare function name."""

    def resolve(self) -> tuple[object, str] | None:
        """Return `(owner, attribute)` to patch, or None if it is not present."""
        import importlib

        try:
            module = importlib.import_module(self.module)
        except ImportError:
            return None
        parts = self.qualname.split(".")
        owner: object = module
        for part in parts[:-1]:
            owner = getattr(owner, part, None)
            if owner is None:
                return None
        if not hasattr(owner, parts[-1]):
            return None
        return owner, parts[-1]


# Anchors, in the causal order §4 fixes. The order here is documentation; the
# trace records whatever order the code actually produces, which is the point.
PROBES: list[Probe] = [
    Probe(
        "sample_controllers_and_battery",
        "argusnet.simulation.sim",
        "BatteryModel.segment_cost_wh",
    ),
    Probe("deconflict", "argusnet.planning.deconfliction", "DeconflictionLayer.resolve_step"),
    Probe("safety_monitor", "argusnet.safety.native", "SafetyMonitor.process"),
    Probe("update_vio_and_pose", "argusnet.localization.vio", "EKFVIO.process_image"),
    Probe("generate_observations", "argusnet.simulation.sim", "build_observations"),
    Probe(
        "update_mapping_and_localization",
        "argusnet.mapping.coverage",
        "CoverageMap.mark_circular",
    ),
    Probe("evaluate_mission_transitions", "argusnet.simulation.sim", "MissionExecutor.step"),
]

# The first statement of every tick, so it is the tick boundary.
TICK_BOUNDARY = Probe("__tick__", "argusnet.simulation.sim", "ScenarioDefinition.node_states")


@dataclass
class Recorder:
    """Collects `(tick, stage)` in call order."""

    tick: int = -1
    records: list[dict] = field(default_factory=list)
    unavailable: list[str] = field(default_factory=list)
    seen_this_tick: set[str] = field(default_factory=set)

    def begin_tick(self) -> None:
        self.tick += 1
        self.seen_this_tick = set()

    def note(self, stage: str) -> None:
        if self.tick < 0:
            # A stage anchor fired during scenario construction, before the
            # first tick. Recording it as tick 0 would be a lie; setup is not
            # part of the tick transaction.
            return
        if stage in self.seen_this_tick:
            # Stages fire once per platform or per target. The trace records
            # stage *order*, not call counts, so the first occurrence is what
            # places the stage in the sequence.
            return
        self.seen_this_tick.add(stage)
        self.records.append({"tick": self.tick, "stage": stage})


def install(recorder: Recorder) -> list[tuple[object, str, object]]:
    """Patch every resolvable anchor. Returns undo information."""
    undo: list[tuple[object, str, object]] = []

    def wrap(probe: Probe, is_boundary: bool) -> None:
        resolved = probe.resolve()
        if resolved is None:
            recorder.unavailable.append(f"{probe.module}.{probe.qualname}")
            return
        owner, attribute = resolved
        original = getattr(owner, attribute)

        def traced(*args, **kwargs):
            if is_boundary:
                recorder.begin_tick()
            else:
                recorder.note(probe.stage)
            return original(*args, **kwargs)

        traced.__name__ = getattr(original, "__name__", attribute)
        setattr(owner, attribute, traced)
        undo.append((owner, attribute, original))

    wrap(TICK_BOUNDARY, is_boundary=True)
    for probe in PROBES:
        wrap(probe, is_boundary=False)
    return undo


def uninstall(undo: list[tuple[object, str, object]]) -> None:
    for owner, attribute, original in undo:
        setattr(owner, attribute, original)


# The paths to capture.
#
# P4.1 names seven: "default, scan-map-inspect, opt-in safety, VIO, frontier,
# POI and egress". Only some of those are configurations. Checked against
# `ScenarioOptions` and `run_simulation` rather than assumed:
#
# - `scan_map_inspect` is the *default* `mission_mode`, so "default" and
#   "scan-map-inspect" are the same run. `target_tracking` is the other mode
#   and is captured instead, since a second copy of the default path would be
#   evidence of nothing.
# - `use_vio_position` lives on `ScenarioDefinition`, not `ScenarioOptions`, and
#   the scenario is frozen -- hence the separate override below.
# - **POI and egress are runtime mission phases, not options.** `_scan_phase`
#   reaches `"egress"` only after coverage and battery conditions are met, and
#   POI trajectories appear when the mission generates them. There is no flag to
#   set. Capturing those two needs a long enough run to reach the phase, which
#   is why they are marked `requires_phase` rather than given a fake toggle.
@dataclass
class PathSpec:
    options: dict = field(default_factory=dict)
    """`ScenarioOptions` fields."""
    scenario_overrides: dict = field(default_factory=dict)
    """`ScenarioDefinition` fields applied after construction."""
    requires_phase: str | None = None
    """A mission phase this path only exercises once reached at runtime."""
    steps_hint: int | None = None


PATHS: dict[str, PathSpec] = {
    "default": PathSpec(),
    # `target_count` defaults to 0, and target_tracking mode with nothing to
    # track raises KeyError('') inside build_default_scenario. Found by running
    # it, not by reading the dataclass.
    "target_tracking": PathSpec(options={"mission_mode": "target_tracking", "target_count": 3}),
    "safety_blocking": PathSpec(options={"safety_blocking": True}),
    "frontier": PathSpec(options={"frontier_exploration": True}),
    "occlusion_aware": PathSpec(options={"occlusion_aware_mapping": True}),
    "vio": PathSpec(scenario_overrides={"use_vio_position": True}),
    "poi": PathSpec(requires_phase="inspecting", steps_hint=600),
    "egress": PathSpec(requires_phase="egress", steps_hint=900),
}


def capture(path_name: str, steps: int, seed: int) -> dict:
    """Run one path and return its trace document."""
    import dataclasses

    from argusnet.simulation import sim as sim_module

    spec = PATHS[path_name]
    if spec.steps_hint is not None and steps < spec.steps_hint:
        steps = spec.steps_hint

    recorder = Recorder()
    undo = install(recorder)
    try:
        scenario = sim_module.build_default_scenario(
            options=sim_module.ScenarioOptions(**spec.options), seed=seed
        )
        if spec.scenario_overrides:
            # ScenarioDefinition is frozen, per the repo's convention for shared
            # runtime state.
            scenario = dataclasses.replace(scenario, **spec.scenario_overrides)
        sim_module.run_simulation(
            scenario,
            sim_module.SimulationConfig(steps=steps, dt_s=0.1, seed=seed),
        )
    finally:
        uninstall(undo)

    ticks: dict[int, list[str]] = {}
    for record in recorder.records:
        ticks.setdefault(record["tick"], []).append(record["stage"])

    return {
        "path": path_name,
        "source": "src/argusnet/simulation/sim.py::run_simulation",
        "note": (
            "Observed stage order, captured by wrapping each stage's anchor call. "
            "A stage that did no work makes no anchor call and is absent -- absence "
            "means 'no work', not 'did not run'. See scripts/capture_stage_trace.py."
        ),
        "seed": seed,
        "steps": steps,
        "requires_phase": PATHS[path_name].requires_phase,
        "unavailable_anchors": recorder.unavailable,
        "stage_order_by_tick": {str(tick): stages for tick, stages in sorted(ticks.items())},
    }


def render(document: dict) -> str:
    """Serialize with each tick's stage list on one line.

    `json.dumps(indent=2)` puts every stage on its own line, which for a
    900-tick egress capture was most of the file. A tick's stage list is read as
    one unit, so it is written as one line; the fixtures went from 1.1 MiB to
    under 300 KiB with no loss.
    """
    ticks = dict(document["stage_order_by_tick"])
    head = {key: value for key, value in document.items() if key != "stage_order_by_tick"}
    rendered = json.dumps(head, indent=2)[:-2]
    rendered += ',\n  "stage_order_by_tick": {\n'
    rendered += ",\n".join(f'    "{tick}": {json.dumps(stages)}' for tick, stages in ticks.items())
    rendered += "\n  }\n}\n"
    return rendered


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", choices=sorted(PATHS), help="Capture one path.")
    parser.add_argument("--all", action="store_true", help="Capture every path.")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out", type=Path, help="Write one path's trace here.")
    parser.add_argument("--out-dir", type=Path, help="Write every path's trace here.")
    args = parser.parse_args()

    if not args.path and not args.all:
        parser.error("pass --path NAME or --all")

    names = sorted(PATHS) if args.all else [args.path]
    failures: list[str] = []
    for name in names:
        try:
            document = capture(name, args.steps, args.seed)
        except Exception as error:  # noqa: BLE001 - one path must not sink the rest
            print(f"error: {name}: {type(error).__name__}: {error}", file=sys.stderr)
            failures.append(name)
            continue
        if document["unavailable_anchors"]:
            print(
                f"warning: {name}: anchors not found: "
                + ", ".join(document["unavailable_anchors"]),
                file=sys.stderr,
            )
        rendered = render(document)
        if args.out_dir:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            destination = args.out_dir / f"{name}.json"
            destination.write_text(rendered)
            print(f"wrote {destination}")
        elif args.out:
            args.out.write_text(rendered)
            print(f"wrote {args.out}")
        else:
            print(rendered)

    if failures:
        print(f"failed to capture: {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
