from __future__ import annotations

import contextlib
from collections.abc import Iterable, Sequence
from dataclasses import replace
from pathlib import Path

from argusnet.evaluation.benchmarks import BenchmarkConfig
from argusnet.evaluation.metrics import EvaluationReport, check_pass_fail, evaluate_replay
from argusnet.evaluation.replay import load_replay_document

CANONICAL_SEEDS: tuple[int, ...] = (7, 42, 137, 9999, 31415)


def _config(
    name: str,
    *,
    mission_type: str,
    difficulty: float,
    duration_s: float,
    map_preset: str,
    terrain_preset: str,
    drone_count: int,
    sim_args: dict[str, object],
    tags: Sequence[str] = (),
) -> BenchmarkConfig:
    return BenchmarkConfig(
        name=name,
        mission_type=mission_type,
        difficulty=difficulty,
        seeds=list(CANONICAL_SEEDS),
        duration_s=duration_s,
        tags=list(dict.fromkeys(("canonical", *tags))),
        family="argusnet-performance-v1",
        map_preset=map_preset,
        terrain_preset=terrain_preset,
        drone_count=drone_count,
        sim_args=sim_args,
    )


CANONICAL_SCENARIOS: tuple[BenchmarkConfig, ...] = (
    _config(
        "baseline_coverage",
        mission_type="coverage",
        difficulty=0.20,
        duration_s=60.0,
        map_preset="medium",
        terrain_preset="default",
        drone_count=2,
        sim_args={
            "mission_mode": "scan_map_inspect",
            "target_count": 0,
            "drone_mode": "mixed",
            "poi_count": 3,
            "scan_coverage_threshold": 0.55,
        },
        tags=("benchmark_fast",),
    ),
    _config(
        "intercept_stress",
        mission_type="intercept",
        difficulty=0.65,
        duration_s=90.0,
        map_preset="medium",
        terrain_preset="alpine",
        drone_count=3,
        sim_args={
            "mission_mode": "target_tracking",
            "target_count": 4,
            "target_motion": "mixed",
            "drone_mode": "follow",
            "ground_stations": 8,
        },
        tags=("benchmark_fast",),
    ),
    _config(
        "persistent_long",
        mission_type="persistent_surveillance",
        difficulty=0.75,
        duration_s=300.0,
        map_preset="regional",
        terrain_preset="alpine",
        drone_count=5,
        sim_args={
            "mission_mode": "scan_map_inspect",
            "target_count": 0,
            "drone_mode": "mixed",
            "poi_count": 5,
            "scan_coverage_threshold": 0.65,
            "streaming": True,
            "max_frames_in_memory": 500,
        },
        tags=("benchmark_slow",),
    ),
    _config(
        "search_acquisition",
        mission_type="search",
        difficulty=0.45,
        duration_s=90.0,
        map_preset="medium",
        terrain_preset="urban_flat",
        drone_count=3,
        sim_args={
            "mission_mode": "target_tracking",
            "target_count": 3,
            "target_motion": "mixed",
            "drone_mode": "search",
            "ground_stations": 6,
        },
        tags=("benchmark_fast",),
    ),
    _config(
        "planner_adversarial",
        mission_type="planner_adversarial",
        difficulty=0.85,
        duration_s=120.0,
        map_preset="medium",
        terrain_preset="urban_flat",
        drone_count=4,
        sim_args={
            "mission_mode": "scan_map_inspect",
            "target_count": 0,
            "drone_mode": "mixed",
            "poi_count": 6,
            "scan_coverage_threshold": 0.75,
            "weather_preset": "heavy_rain",
        },
        tags=("benchmark_slow",),
    ),
)

_SCENARIOS_BY_NAME = {config.name: config for config in CANONICAL_SCENARIOS}


def benchmark_fast(duration_s: float | None = 5.0) -> list[BenchmarkConfig]:
    """Return the PR-safe canonical smoke subset with one seed per scenario."""
    configs = [
        _SCENARIOS_BY_NAME["baseline_coverage"],
        _SCENARIOS_BY_NAME["intercept_stress"],
        _SCENARIOS_BY_NAME["search_acquisition"],
    ]
    return [
        replace(
            config,
            seeds=[7],
            duration_s=config.duration_s if duration_s is None else duration_s,
        )
        for config in configs
    ]


def benchmark_slow(seeds: Sequence[int] = CANONICAL_SEEDS) -> list[BenchmarkConfig]:
    """Return the nightly canonical scenario sweep."""
    return [replace(config, seeds=list(seeds)) for config in CANONICAL_SCENARIOS]


def list_scenarios() -> list[str]:
    return [config.name for config in CANONICAL_SCENARIOS]


def get_scenario(name: str) -> BenchmarkConfig:
    try:
        return _SCENARIOS_BY_NAME[name]
    except KeyError as exc:
        known = ", ".join(list_scenarios())
        raise KeyError(f"Unknown benchmark scenario '{name}'. Known scenarios: {known}") from exc


def configs_for_suite(suite: str, *, seeds: Sequence[int] | None = None) -> list[BenchmarkConfig]:
    if suite == "fast":
        configs = benchmark_fast()
    elif suite == "slow":
        configs = benchmark_slow(seeds or CANONICAL_SEEDS)
    else:
        config = get_scenario(suite)
        configs = [replace(config, seeds=list(seeds or config.seeds))]
    if seeds is not None and suite == "fast":
        configs = [replace(config, seeds=list(seeds)) for config in configs]
    return configs


def _append_arg(argv: list[str], name: str, value: object) -> None:
    flag_name = {"scan_coverage_threshold": "scan-threshold"}.get(name, name.replace("_", "-"))
    flag = "--" + flag_name
    if isinstance(value, bool):
        if value:
            argv.append(flag)
        return
    argv.extend([flag, str(value)])


def _sim_argv(config: BenchmarkConfig, seed: int, replay_path: Path) -> list[str]:
    argv = [
        "--duration-s",
        str(config.duration_s),
        "--seed",
        str(seed),
        "--replay",
        str(replay_path),
        "--map-preset",
        str(config.map_preset or "medium"),
        "--terrain-preset",
        str(config.terrain_preset or "default"),
        "--drone-count",
        str(config.drone_count or 2),
    ]
    for key, value in config.sim_args.items():
        _append_arg(argv, key, value)
    return argv


def run_config_with_sim(
    config: BenchmarkConfig,
    seed: int,
    output_dir: str | Path,
    *,
    quiet: bool = True,
) -> EvaluationReport:
    """Run one benchmark seed through ``argusnet sim`` and evaluate its replay."""
    from argusnet.simulation import sim

    run_dir = Path(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    replay_path = run_dir / "replay.json"
    sim_log_path = run_dir / "sim.log"

    args = sim.parse_args(_sim_argv(config, seed, replay_path))
    if quiet:
        with sim_log_path.open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
            sim.run_from_args(args)
    else:
        sim.run_from_args(args)

    replay = load_replay_document(str(replay_path))
    replay_size_mb = replay_path.stat().st_size / (1024.0 * 1024.0)
    meta = replay.setdefault("meta", {})
    if isinstance(meta, dict):
        performance = meta.setdefault("performance", {})
        if isinstance(performance, dict):
            performance["replay_size_mb"] = replay_size_mb

    report = evaluate_replay(
        replay,
        scenario_name=config.name,
        mission_type=config.mission_type,
        difficulty=config.difficulty,
        seed=seed,
    )
    report = check_pass_fail(report)
    if report.replay_size_mb is None:
        report = replace(report, replay_size_mb=replay_size_mb)
    return report


def iter_config_runs(configs: Iterable[BenchmarkConfig]) -> Iterable[tuple[BenchmarkConfig, int]]:
    for config in configs:
        for seed in config.seeds:
            yield config, int(seed)
