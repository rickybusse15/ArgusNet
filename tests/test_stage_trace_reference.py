"""The Python tick loop must run its stages in the order §4 fixes.

`docs/migration/ARCHITECTURE_AND_PROTOCOLS.md` §4 states the causal order as a
migration constraint on the *Rust* port. That constraint is only worth anything
if the Python reference actually has the order it claims, and nothing checked
that until now — the table was written by reading the loop, and reading is how
the three matrix errors in G0 happened.

These tests read the captured traces in `tests/fixtures/stage_traces/`, which
`scripts/capture_stage_trace.py` produces by observing which stage anchors fire,
in what order, per tick. They assert two things:

1. every observed order is a subsequence of the documented order, so no tick
   ever runs a stage out of place;
2. the constraints §4 states in prose — safety after deconfliction, controllers
   before overrides, VIO before observation generation — hold in the runs where
   both stages are active.

A stage that does no work on a tick makes no anchor call and is simply absent,
so these are subsequence checks rather than equality checks. That is a real
limit and it is why the assertions are phrased the way they are: absence proves
nothing, but a stage appearing *after* one that should follow it proves the
order is wrong.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures" / "stage_traces"

# The order from ARCHITECTURE_AND_PROTOCOLS.md §4, written out rather than
# imported, so that editing the source of truth does not silently edit the test.
# Stages 0 and 10 are the tick transaction boundary and have no Python anchor:
# the loop does not admit operator commands, and it assembles a frame rather
# than committing a revision.
DOCUMENTED_ORDER = [
    "sample_controllers_and_battery",  # 1
    "apply_mission_overrides",  # 2
    "deconflict",  # 3
    "safety_monitor",  # 4
    "update_vio_and_pose",  # 5
    "generate_observations",  # 6
    "update_fusion",  # 7
    "update_mapping_and_localization",  # 8
    "evaluate_mission_transitions",  # 9
]


def trace_files() -> list[Path]:
    return sorted(FIXTURES.glob("*.json"))


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def is_subsequence(observed: list[str], reference: list[str]) -> bool:
    iterator = iter(reference)
    return all(stage in iterator for stage in observed)


def test_fixtures_exist():
    """A missing fixture directory would make every test below vacuous."""
    files = trace_files()
    assert files, f"no captured traces in {FIXTURES}; run scripts/capture_stage_trace.py"


@pytest.mark.parametrize("path", trace_files(), ids=lambda p: p.stem)
def test_every_tick_follows_the_documented_stage_order(path: Path):
    document = load(path)
    assert not document["unavailable_anchors"], (
        f"{path.stem}: anchors did not resolve: {document['unavailable_anchors']}. "
        "An unresolved anchor makes its stage invisible, so the trace would pass "
        "this test by having nothing to check."
    )

    for tick, observed in document["stage_order_by_tick"].items():
        assert is_subsequence(observed, DOCUMENTED_ORDER), (
            f"{path.stem} tick {tick} ran stages out of order.\n"
            f"  observed:   {observed}\n"
            f"  documented: {DOCUMENTED_ORDER}"
        )


@pytest.mark.parametrize("path", trace_files(), ids=lambda p: p.stem)
def test_a_trace_actually_observed_something(path: Path):
    """Guards the way this suite could pass for the wrong reason.

    An empty trace is a subsequence of anything. If the capture silently stopped
    observing — a renamed method, a refactor that moved an anchor — every
    ordering test above would still pass.
    """
    document = load(path)
    ticks = document["stage_order_by_tick"]
    assert ticks, f"{path.stem}: no ticks recorded"
    assert any(len(stages) >= 3 for stages in ticks.values()), (
        f"{path.stem}: no tick recorded three or more stages, so the ordering "
        "assertions had almost nothing to constrain"
    )


def test_safety_runs_after_deconfliction():
    """§4 states this in prose; the safety path is where both stages are active."""
    path = FIXTURES / "safety_blocking.json"
    if not path.exists():
        pytest.skip("safety_blocking trace not captured")
    document = load(path)

    checked = 0
    for tick, observed in document["stage_order_by_tick"].items():
        if "safety_monitor" in observed and "deconflict" in observed:
            assert observed.index("safety_monitor") > observed.index("deconflict"), (
                f"tick {tick}: safety ran before deconfliction, contradicting §4"
            )
            checked += 1
    assert checked > 0, (
        "the safety path recorded no tick where both stages were active, so this "
        "assertion checked nothing"
    )


def test_vio_runs_before_observation_generation():
    """The loop's own comment says VIO "must happen before ingest_frame" so that
    ``use_vio_position`` can substitute positions into the tracker request.
    Reversing them would feed the tracker ground truth while claiming drift.
    """
    checked = 0
    for path in trace_files():
        document = load(path)
        for tick, observed in document["stage_order_by_tick"].items():
            if "update_vio_and_pose" in observed and "generate_observations" in observed:
                assert observed.index("update_vio_and_pose") < observed.index(
                    "generate_observations"
                ), f"{path.stem} tick {tick}: observations were generated before VIO updated"
                checked += 1
    assert checked > 0, "no trace exercised both VIO and observation generation"


def test_the_first_tick_has_no_battery_accounting():
    """A real property of the reference, recorded so the port reproduces it.

    Battery cost is charged per travelled segment, and on tick 0 there is no
    previous position, so stage 1 does no work. A Rust port that charged a
    segment on the first tick would drain batteries slightly faster than the
    reference for the whole run.
    """
    path = FIXTURES / "default.json"
    if not path.exists():
        pytest.skip("default trace not captured")
    document = load(path)
    first = document["stage_order_by_tick"].get("0")
    assert first is not None, "no tick 0 recorded"
    assert "sample_controllers_and_battery" not in first, (
        "tick 0 charged battery; the reference has no previous position to charge a segment against"
    )
    second = document["stage_order_by_tick"].get("1")
    assert second is not None and "sample_controllers_and_battery" in second, (
        "tick 1 did not charge battery, so the assertion above may be measuring "
        "a broken anchor rather than a real property"
    )
