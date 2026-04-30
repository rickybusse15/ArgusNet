"""Benchmark runner for ArgusNet evaluation.

Provides a harness for running simulation scenarios at multiple seeds,
collecting EvaluationReport objects, and aggregating results across runs.
"""

from __future__ import annotations

import json
import math
import os
import platform
import statistics
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from argusnet.evaluation.metrics import EvaluationReport, report_to_dict

__all__ = [
    "DEFAULT_SCENARIO_SEEDS",
    "BenchmarkScenario",
    "BenchmarkConfig",
    "BenchmarkRun",
    "BenchmarkSuite",
    "AggregatedResult",
    "ScenarioFamily",
    "aggregate_reports",
    "build_performance_summary",
    "capture_environment_metadata",
    "compute_timing_summary",
    "default_benchmark_configs",
    "get_scenario_family",
    "get_scenario_families",
    "get_scenario_metadata",
    "list_scenario_family_ids",
    "normalize_command",
    "write_performance_summary",
]

DEFAULT_SCENARIO_SEEDS: tuple[int, ...] = (7, 17, 29)


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for a single benchmark scenario."""

    name: str
    """Unique benchmark name."""

    mission_type: str = "surveillance"
    """Mission type passed to scenario builder."""

    difficulty: float = 0.5
    """Difficulty level [0, 1]."""

    seeds: list[int] = field(default_factory=lambda: [42, 43, 44])
    """Random seeds to run."""

    duration_s: float = 120.0
    """Simulation duration per seed (seconds)."""

    tags: list[str] = field(default_factory=list)
    """Optional categorisation tags."""

    family: str | None = None
    """Scenario family identifier, when this config comes from a benchmark family."""

    map_preset: str | None = None
    """Map preset requested by the deterministic scenario metadata."""

    terrain_preset: str | None = None
    """Terrain preset requested by the deterministic scenario metadata."""

    drone_count: int | None = None
    """Number of drones requested by the deterministic scenario metadata."""

    pass_criteria: list[str] = field(default_factory=list)
    """Human-readable primary pass criteria for this scenario."""

    sim_args: dict[str, object] = field(default_factory=dict)
    """Additional simulator CLI arguments used by operational benchmark runners."""


@dataclass(frozen=True)
class BenchmarkScenario:
    """Deterministic metadata for one documented benchmark scenario."""

    name: str
    family: str
    map_preset: str
    terrain_preset: str
    drone_count: int
    difficulty: float
    mission_type: str
    seeds: tuple[int, ...] = DEFAULT_SCENARIO_SEEDS
    duration_s: float = 120.0
    pass_criteria: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()

    def to_config(
        self,
        *,
        seeds: Sequence[int] | None = None,
        duration_s: float | None = None,
    ) -> BenchmarkConfig:
        """Convert scenario metadata into the existing benchmark-runner config."""
        config_tags = list(dict.fromkeys((self.family, *self.tags)))
        return BenchmarkConfig(
            name=self.name,
            mission_type=self.mission_type,
            difficulty=self.difficulty,
            seeds=list(seeds if seeds is not None else self.seeds),
            duration_s=float(duration_s if duration_s is not None else self.duration_s),
            tags=config_tags,
            family=self.family,
            map_preset=self.map_preset,
            terrain_preset=self.terrain_preset,
            drone_count=self.drone_count,
            pass_criteria=list(self.pass_criteria),
        )


@dataclass(frozen=True)
class ScenarioFamily:
    """A documented benchmark family with deterministic scenario definitions."""

    family_id: str
    purpose: str
    primary_pass_criteria: tuple[str, ...]
    scenarios: tuple[BenchmarkScenario, ...]

    def to_configs(
        self,
        *,
        seeds: Sequence[int] | None = None,
        duration_s: float | None = None,
    ) -> list[BenchmarkConfig]:
        """Return benchmark configs for all scenarios in deterministic order."""
        return [
            scenario.to_config(seeds=seeds, duration_s=duration_s)
            for scenario in self.scenarios
        ]


@dataclass
class BenchmarkRun:
    """A single executed benchmark (one seed)."""

    config: BenchmarkConfig
    seed: int
    report: EvaluationReport | None = None
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.report is not None and self.error is None


@dataclass(frozen=True)
class AggregatedResult:
    """Aggregated statistics across multiple seeds for a benchmark."""

    name: str
    total_runs: int
    successful_runs: int
    pass_rate: float

    localisation_rmse_mean: float | None
    localisation_rmse_std: float | None
    track_continuity_mean: float | None
    track_continuity_std: float | None
    mission_completion_mean: float | None

    failed_seeds: list[int]
    errors: list[str]


def _scenario(
    name: str,
    family: str,
    map_preset: str,
    terrain_preset: str,
    drone_count: int,
    difficulty: float,
    mission_type: str,
    pass_criteria: Sequence[str],
) -> BenchmarkScenario:
    return BenchmarkScenario(
        name=name,
        family=family,
        map_preset=map_preset,
        terrain_preset=terrain_preset,
        drone_count=drone_count,
        difficulty=difficulty,
        mission_type=mission_type,
        pass_criteria=tuple(pass_criteria),
        tags=(map_preset, terrain_preset, mission_type),
    )


_MAPPING_CRITERIA = (
    "map_coverage_fraction >= scan_coverage_threshold",
    "enclosed_gap_fraction <= gap_fill_min_fraction",
    "no unsafe route acceptance",
)
_LOCALIZATION_CRITERIA = (
    "localization confidence reaches configured threshold",
    "position uncertainty trends downward",
    "timeout use is reported when convergence does not happen naturally",
)
_INSPECTION_CRITERIA = (
    "all required POIs complete",
    "inspection does not start before localization confidence is sufficient",
    "egress reaches complete",
)
_COORDINATION_CRITERIA = (
    "no unresolved separation violations",
    "POI assignments remain explainable",
    "completed replay includes mapping, localization, inspection, and egress state",
)
_REVISIT_CRITERIA = (
    "prior POIs or map regions are retrieved from the index",
    "localization confidence is established before revisit routing",
    "evidence/change records are persisted when enabled",
)

_FAMILY_DEFINITIONS: tuple[ScenarioFamily, ...] = (
    ScenarioFamily(
        family_id="mapping_coverage",
        purpose="Verify map coverage, gap detection, terrain handling, and replay output.",
        primary_pass_criteria=_MAPPING_CRITERIA,
        scenarios=(
            _scenario(
                "mapping_small",
                "mapping_coverage",
                "small",
                "default",
                1,
                0.1,
                "mapping",
                _MAPPING_CRITERIA,
            ),
            _scenario(
                "mapping_medium_multi",
                "mapping_coverage",
                "medium",
                "default",
                3,
                0.3,
                "mapping",
                _MAPPING_CRITERIA,
            ),
            _scenario(
                "mapping_alpine",
                "mapping_coverage",
                "medium",
                "alpine",
                3,
                0.5,
                "mapping",
                _MAPPING_CRITERIA,
            ),
        ),
    ),
    ScenarioFamily(
        family_id="localization_recovery",
        purpose="Exercise startup localization, weak-prior localization, and timeout handling.",
        primary_pass_criteria=_LOCALIZATION_CRITERIA,
        scenarios=(
            _scenario(
                "localize_known_launch",
                "localization_recovery",
                "small",
                "default",
                1,
                0.1,
                "localization",
                _LOCALIZATION_CRITERIA,
            ),
            _scenario(
                "localize_weak_prior",
                "localization_recovery",
                "medium",
                "default",
                2,
                0.4,
                "localization",
                _LOCALIZATION_CRITERIA,
            ),
            _scenario(
                "localize_after_revisit",
                "localization_recovery",
                "medium",
                "alpine",
                2,
                0.6,
                "localization",
                _LOCALIZATION_CRITERIA,
            ),
        ),
    ),
    ScenarioFamily(
        family_id="inspection_workflow",
        purpose=(
            "Verify map-relative POI assignment, dwell completion, egress, "
            "and inspection events."
        ),
        primary_pass_criteria=_INSPECTION_CRITERIA,
        scenarios=(
            _scenario(
                "inspect_single_poi",
                "inspection_workflow",
                "small",
                "default",
                1,
                0.2,
                "inspection",
                _INSPECTION_CRITERIA,
            ),
            _scenario(
                "inspect_multi_poi",
                "inspection_workflow",
                "medium",
                "default",
                3,
                0.4,
                "inspection",
                _INSPECTION_CRITERIA,
            ),
            _scenario(
                "inspect_occluded_area",
                "inspection_workflow",
                "medium",
                "urban_flat",
                3,
                0.6,
                "inspection",
                _INSPECTION_CRITERIA,
            ),
        ),
    ),
    ScenarioFamily(
        family_id="coordination_stress",
        purpose=(
            "Verify multi-drone coverage distribution, POI workload sharing, deconfliction, "
            "and communication assumptions."
        ),
        primary_pass_criteria=_COORDINATION_CRITERIA,
        scenarios=(
            _scenario(
                "coordination_two_drone",
                "coordination_stress",
                "medium",
                "default",
                2,
                0.3,
                "map_localize_inspect",
                _COORDINATION_CRITERIA,
            ),
            _scenario(
                "coordination_dense_pois",
                "coordination_stress",
                "medium",
                "urban_flat",
                4,
                0.6,
                "map_localize_inspect",
                _COORDINATION_CRITERIA,
            ),
            _scenario(
                "coordination_large_area",
                "coordination_stress",
                "regional",
                "alpine",
                5,
                0.7,
                "map_localize_inspect",
                _COORDINATION_CRITERIA,
            ),
        ),
    ),
    ScenarioFamily(
        family_id="revisit_and_change",
        purpose="Prepare repeat-inspection and change-detection benchmark coverage.",
        primary_pass_criteria=_REVISIT_CRITERIA,
        scenarios=(
            _scenario(
                "revisit_known_poi",
                "revisit_and_change",
                "small",
                "default",
                1,
                0.2,
                "revisit",
                _REVISIT_CRITERIA,
            ),
            _scenario(
                "revisit_multi_session",
                "revisit_and_change",
                "medium",
                "default",
                2,
                0.5,
                "revisit",
                _REVISIT_CRITERIA,
            ),
            _scenario(
                "revisit_changed_area",
                "revisit_and_change",
                "medium",
                "urban_flat",
                3,
                0.7,
                "revisit",
                _REVISIT_CRITERIA,
            ),
        ),
    ),
)

_FAMILIES_BY_ID: dict[str, ScenarioFamily] = {
    family.family_id: family for family in _FAMILY_DEFINITIONS
}
_SCENARIOS_BY_NAME: dict[str, BenchmarkScenario] = {
    scenario.name: scenario for family in _FAMILY_DEFINITIONS for scenario in family.scenarios
}


def list_scenario_family_ids() -> list[str]:
    """Return documented scenario-family identifiers in deterministic order."""
    return [family.family_id for family in _FAMILY_DEFINITIONS]


def get_scenario_families() -> dict[str, ScenarioFamily]:
    """Return all scenario families keyed by family ID."""
    return dict(_FAMILIES_BY_ID)


def get_scenario_family(family_id: str) -> ScenarioFamily:
    """Return one scenario family by ID."""
    try:
        return _FAMILIES_BY_ID[family_id]
    except KeyError as exc:
        known = ", ".join(list_scenario_family_ids())
        raise KeyError(f"Unknown scenario family '{family_id}'. Known families: {known}") from exc


def get_scenario_metadata(scenario_name: str) -> BenchmarkScenario:
    """Return deterministic metadata for one benchmark scenario by name."""
    try:
        return _SCENARIOS_BY_NAME[scenario_name]
    except KeyError as exc:
        known = ", ".join(sorted(_SCENARIOS_BY_NAME))
        raise KeyError(
            f"Unknown benchmark scenario '{scenario_name}'. Known scenarios: {known}"
        ) from exc


def default_benchmark_configs(
    family_ids: Sequence[str] | None = None,
    *,
    seeds: Sequence[int] | None = None,
    duration_s: float | None = None,
) -> list[BenchmarkConfig]:
    """Return deterministic configs for the requested scenario families."""
    ids = list(family_ids) if family_ids is not None else list_scenario_family_ids()
    configs: list[BenchmarkConfig] = []
    for family_id in ids:
        configs.extend(
            get_scenario_family(family_id).to_configs(seeds=seeds, duration_s=duration_s)
        )
    return configs


def aggregate_reports(
    name: str,
    runs: list[BenchmarkRun],
) -> AggregatedResult:
    """Aggregate a list of :class:`BenchmarkRun` objects into statistics."""
    successful = [r for r in runs if r.succeeded]
    total = len(runs)
    n_ok = len(successful)

    reports = [r.report for r in successful if r.report is not None]

    def _mean_std(vals: list[float]):
        if not vals:
            return None, None
        return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)

    rmse_vals = [r.localisation_rmse_m for r in reports if r.localisation_rmse_m is not None]
    cont_vals = [r.track_continuity_mean for r in reports]
    comp_vals = [r.mission_completion_rate for r in reports]

    rmse_mean, rmse_std = _mean_std(rmse_vals)
    cont_mean, cont_std = _mean_std(cont_vals)
    comp_mean, _ = _mean_std(comp_vals)

    pass_rate = sum(1 for r in reports if r.passed) / max(n_ok, 1)

    failed_seeds = [r.seed for r in runs if not r.succeeded]
    errors = [r.error for r in runs if r.error]

    return AggregatedResult(
        name=name,
        total_runs=total,
        successful_runs=n_ok,
        pass_rate=pass_rate,
        localisation_rmse_mean=rmse_mean,
        localisation_rmse_std=rmse_std,
        track_continuity_mean=cont_mean,
        track_continuity_std=cont_std,
        mission_completion_mean=comp_mean,
        failed_seeds=failed_seeds,
        errors=errors,
    )


def _percentile(sorted_values: Sequence[float], percentile: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])

    rank = (len(sorted_values) - 1) * (percentile / 100.0)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(sorted_values[low])

    weight = rank - low
    return float(sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight)


def _finite_timing_values(timings_s: Sequence[float]) -> list[float]:
    values: list[float] = []
    for timing in timings_s:
        value = float(timing)
        if not math.isfinite(value):
            raise ValueError("Timing values must be finite.")
        if value < 0.0:
            raise ValueError("Timing values must be non-negative.")
        values.append(value)
    return values


def compute_timing_summary(timings_s: Sequence[float]) -> dict[str, float | int | None]:
    """Compute deterministic frame/run timing statistics in seconds."""
    values = _finite_timing_values(timings_s)
    if not values:
        return {
            "count": 0,
            "min_s": None,
            "max_s": None,
            "mean_s": None,
            "p50_s": None,
            "p95_s": None,
            "p99_s": None,
            "total_s": 0.0,
        }

    sorted_values = sorted(values)
    return {
        "count": len(values),
        "min_s": float(sorted_values[0]),
        "max_s": float(sorted_values[-1]),
        "mean_s": float(statistics.fmean(sorted_values)),
        "p50_s": _percentile(sorted_values, 50.0),
        "p95_s": _percentile(sorted_values, 95.0),
        "p99_s": _percentile(sorted_values, 99.0),
        "total_s": float(sum(sorted_values)),
    }


def normalize_command(command: Sequence[object] | str | None = None) -> list[str]:
    """Return a JSON-friendly command list for performance summaries."""
    if command is None:
        return [str(part) for part in sys.argv]
    if isinstance(command, str):
        return [command]
    return [str(part) for part in command]


def _run_metadata_command(command: Sequence[str], cwd: str | Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            list(command),
            cwd=str(cwd) if cwd is not None else None,
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _memory_total_bytes() -> int | None:
    if not hasattr(os, "sysconf"):
        return None
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        page_count = int(os.sysconf("SC_PHYS_PAGES"))
    except (OSError, TypeError, ValueError):
        return None
    return page_size * page_count


def capture_environment_metadata(cwd: str | Path | None = None) -> dict[str, Any]:
    """Capture Git, OS, Python, Rust, CPU, and memory metadata for a run."""
    commit_sha = _run_metadata_command(("git", "rev-parse", "HEAD"), cwd)
    git_dirty: bool | None = None
    dirty_output = _run_metadata_command(("git", "status", "--porcelain"), cwd)
    if dirty_output is not None:
        git_dirty = bool(dirty_output)

    return {
        "commit_sha": commit_sha,
        "git_dirty": git_dirty,
        "os": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "rust": {
            "rustc": _run_metadata_command(("rustc", "--version"), cwd),
            "cargo": _run_metadata_command(("cargo", "--version"), cwd),
        },
        "hardware": {
            "cpu_count": os.cpu_count(),
            "memory_total_bytes": _memory_total_bytes(),
        },
    }


def _as_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    return dict(asdict(value))


def build_performance_summary(
    *,
    scenario_name: str,
    seed: int,
    command: Sequence[object] | str | None = None,
    timings_s: Sequence[float] = (),
    family: str | None = None,
    report: EvaluationReport | Mapping[str, Any] | None = None,
    aggregate: AggregatedResult | Mapping[str, Any] | None = None,
    environment_metadata: Mapping[str, Any] | None = None,
    generated_at_utc: str | None = None,
    cwd: str | Path | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable ``performance_summary.json`` payload."""
    environment = (
        dict(environment_metadata)
        if environment_metadata is not None
        else capture_environment_metadata(cwd)
    )
    commit_sha = environment.get("commit_sha")
    if commit_sha is None and isinstance(environment.get("git"), Mapping):
        commit_sha = environment["git"].get("commit_sha")  # type: ignore[index]

    summary: dict[str, Any] = {
        "schema_version": 1,
        "scenario_name": scenario_name,
        "family": family,
        "seed": int(seed),
        "command": normalize_command(command),
        "commit_sha": commit_sha,
        "generated_at_utc": generated_at_utc or datetime.now(timezone.utc).isoformat(),
        "timing_s": compute_timing_summary(timings_s),
        "environment": environment,
    }

    if report is not None:
        summary["report"] = (
            report_to_dict(report) if isinstance(report, EvaluationReport) else dict(report)
        )
    if aggregate is not None:
        summary["aggregate"] = _as_mapping(aggregate)

    return summary


def write_performance_summary(
    output_path: str | Path,
    summary: Mapping[str, Any] | None = None,
    **summary_kwargs: Any,
) -> Path:
    """Write a ``performance_summary.json`` payload and return the written path.

    If ``output_path`` points to a directory or has no suffix, the helper writes
    ``performance_summary.json`` inside that directory.
    """
    path = Path(output_path)
    if path.is_dir() or path.suffix == "":
        path = path / "performance_summary.json"

    payload = dict(summary) if summary is not None else build_performance_summary(**summary_kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


RunFn = Callable[[BenchmarkConfig, int], EvaluationReport]


class BenchmarkSuite:
    """Collection of benchmark configurations with a shared run function.

    Usage::

        suite = BenchmarkSuite(run_fn=my_run_function)
        suite.add(BenchmarkConfig("baseline", seeds=[42, 43, 44]))
        results = suite.run_all()
    """

    def __init__(self, run_fn: RunFn) -> None:
        self._run_fn = run_fn
        self._configs: list[BenchmarkConfig] = []

    def add(self, config: BenchmarkConfig) -> None:
        self._configs.append(config)

    def run_all(self, verbose: bool = False) -> dict[str, AggregatedResult]:
        results: dict[str, AggregatedResult] = {}
        for config in self._configs:
            runs: list[BenchmarkRun] = []
            for seed in config.seeds:
                run = BenchmarkRun(config=config, seed=seed)
                try:
                    run.report = self._run_fn(config, seed)
                except Exception as exc:  # noqa: BLE001
                    run.error = str(exc)
                runs.append(run)
                if verbose:
                    status = (
                        "PASS" if run.succeeded and run.report and run.report.passed else "FAIL"
                    )
                    print(f"  [{status}] {config.name} seed={seed}")
            results[config.name] = aggregate_reports(config.name, runs)
        return results
