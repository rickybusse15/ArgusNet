"""Scenario comparisons over paired seed sets (F4).

A comparison runs two or more *variants* of one scenario (different
configurations or policies) on the **same** seeds, so each seed is a pair and
the seed's own luck cancels out of the difference. This module is the
statistics and the report; ``scripts/compare_scenarios.py`` runs the seeds
through ``argusnet-rs`` and calls it.

The report states, for every metric:

- what the metric means and which direction is better (:data:`METRICS`);
- per variant: sample size, mean, median, sample standard deviation, min, max;
- against the baseline (the first variant): the paired difference, its
  standard error and a 95% interval (Student's t on the paired differences),
  and how many seeds improved, worsened or tied;

and, per variant, how many runs failed (did not finish, or were unsupported)
and how many finished outside the scenario's pass/fail envelope, plus the
runtime cost. A failed run is counted and shown, never dropped from the
denominator silently.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

# Metric id -> (better direction, definition). "higher" / "lower" / None (no
# preferred direction: reported, never scored).
METRICS: dict[str, tuple[str | None, str]] = {
    "localisation_rmse_m": (
        "lower",
        "Root-mean-square distance in metres between each fused track and the truth it "
        "was first paired with, over every frame the track exists.",
    ),
    "track_continuity_mean": (
        "higher",
        "Mean over targets of the fraction of steps in which the target has at least one "
        "accepted bearing observation.",
    ),
    "covariance_reduction_mean": (
        "higher",
        "Mean over tracks of (trace(P_first) - trace(P_last update)) / trace(P_first), the "
        "3x3 position block; tracks qualify only with enough updates (D-1).",
    ),
    "time_to_reacquire_mean_s": (
        "lower",
        "Mean duration in seconds of track-loss events (stale steps > 0 until 0 again); an "
        "event never recovered counts to the end of the run.",
    ),
    "mission_completion_rate": (
        "higher",
        "Required objectives met over required objectives (1.0 when none are required).",
    ),
    "energy_reserve_min": (
        "higher",
        "Lowest battery fraction left on any drone at the end of the run.",
    ),
    "battery_remaining_min": (
        "higher",
        "Lowest battery fraction any drone reached, as the battery ledger recorded it. "
        "Unlike energy_reserve_min it follows the energy profile, hover included (D-3).",
    ),
    "false_handoff_rate": ("lower", "Fraction of track handoffs that went to the wrong target."),
    "safety_override_count": (
        None,
        "Times the safety monitor overrode a commanded motion. Context-dependent: fewer "
        "can mean a safer plan or a less exercised monitor.",
    ),
}

# Two-sided 97.5% quantiles of Student's t, by degrees of freedom.
_T975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
    9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074,
    23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045,
    30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980,
}  # fmt: skip


# Relative size below which a difference is rounding, not a change.
_ROUNDING = 1e-9


def t975(df: int) -> float:
    """Student's t 97.5% quantile, conservatively rounded down to a tabulated df."""
    if df < 1:
        raise ValueError("t needs at least one degree of freedom")
    if df > 120:
        return 1.960
    return _T975[max(k for k in _T975 if k <= df)]


@dataclass
class RunResult:
    """One variant on one seed."""

    variant: str
    seed: int
    metrics: dict[str, float | None] = field(default_factory=dict)
    # Did not finish: the reason. Unsupported configurations count here too.
    error: str | None = None
    passed_envelope: bool | None = None
    wall_s: float | None = None
    tick_mean_ms: float | None = None
    config_sha256: str | None = None
    # The world the run happened in (environment id, terrain source).
    world: dict[str, Any] | None = None


def dispersion(values: Sequence[float]) -> dict[str, float | int | None]:
    n = len(values)
    return {
        "n": n,
        "mean": statistics.fmean(values) if n else None,
        "median": statistics.median(values) if n else None,
        "sd": statistics.stdev(values) if n > 1 else None,
        "min": min(values) if n else None,
        "max": max(values) if n else None,
    }


def paired(
    baseline: Mapping[int, float | None],
    variant: Mapping[int, float | None],
    direction: str | None,
) -> dict[str, Any]:
    """The paired difference ``variant - baseline`` over seeds both measured."""
    seeds = sorted(s for s in baseline if s in variant)
    pairs = [
        (baseline[s], variant[s])
        for s in seeds
        if baseline[s] is not None
        and variant[s] is not None
        and math.isfinite(baseline[s])  # type: ignore[arg-type]
        and math.isfinite(variant[s])  # type: ignore[arg-type]
    ]
    # A difference within rounding of the values is a tie: summing the same
    # quantities in another order must not read as a change.
    diffs = [
        0.0 if abs(v - b) <= _ROUNDING * max(1.0, abs(b), abs(v)) else v - b  # type: ignore[operator]
        for b, v in pairs
    ]
    n = len(diffs)
    out: dict[str, Any] = {"n_pairs": n, "mean_diff": None, "sd_diff": None, "se": None}
    out["ci95"] = None
    if n:
        out["mean_diff"] = statistics.fmean(diffs)
    if n > 1:
        sd = statistics.stdev(diffs)
        se = sd / math.sqrt(n)
        half = t975(n - 1) * se
        out.update(sd_diff=sd, se=se, ci95=[out["mean_diff"] - half, out["mean_diff"] + half])
    if direction in ("higher", "lower") and n:
        sign = 1 if direction == "higher" else -1
        out["improved"] = sum(1 for d in diffs if sign * d > 0)
        out["worsened"] = sum(1 for d in diffs if sign * d < 0)
        out["tied"] = sum(1 for d in diffs if d == 0)
        ci = out["ci95"]
        if ci is None:
            out["verdict"] = "too few pairs"
        else:
            # The interval in "improvement" units: positive is better.
            low, high = sorted((sign * ci[0], sign * ci[1]))
            if low > 0:
                out["verdict"] = "better"
            elif high < 0:
                out["verdict"] = "worse"
            else:
                out["verdict"] = "no detectable difference"
    return out


def compare(
    runs: Sequence[RunResult],
    variants: Sequence[str],
    metrics: Sequence[str] = tuple(METRICS),
) -> dict[str, Any]:
    """The comparison report body: per-variant summaries and paired
    differences against ``variants[0]``."""
    if not variants:
        raise ValueError("a comparison needs at least one variant")
    by_variant: dict[str, list[RunResult]] = {v: [] for v in variants}
    for run in runs:
        if run.variant not in by_variant:
            raise ValueError(f"run for unknown variant {run.variant!r}")
        by_variant[run.variant].append(run)
    seeds = sorted({r.seed for r in runs})

    summary: dict[str, Any] = {}
    for name, its_runs in by_variant.items():
        finished = [r for r in its_runs if r.error is None]
        walls = [r.wall_s for r in finished if r.wall_s is not None]
        ticks = [r.tick_mean_ms for r in finished if r.tick_mean_ms is not None]
        summary[name] = {
            "runs": len(its_runs),
            "failed": len(its_runs) - len(finished),
            "failures": {r.seed: r.error for r in its_runs if r.error is not None},
            "outside_envelope": sum(1 for r in finished if r.passed_envelope is False),
            "runtime": {"wall_s": dispersion(walls), "tick_mean_ms": dispersion(ticks)},
            "metrics": {
                m: dispersion(
                    [
                        v
                        for r in finished
                        if (v := r.metrics.get(m)) is not None and math.isfinite(v)
                    ]
                )
                for m in metrics
            },
        }

    def values(name: str, metric: str) -> dict[int, float | None]:
        return {r.seed: r.metrics.get(metric) for r in by_variant[name] if r.error is None}

    base = variants[0]
    against: dict[str, Any] = {}
    for name in variants[1:]:
        against[name] = {
            m: paired(values(base, m), values(name, m), METRICS.get(m, (None, ""))[0])
            for m in metrics
        }
    return {
        "baseline": base,
        "variants": list(variants),
        "seeds": seeds,
        "sample_size": len(seeds),
        "metric_definitions": {
            m: {
                "better": METRICS.get(m, (None, ""))[0],
                "definition": METRICS.get(m, (None, ""))[1],
            }
            for m in metrics
        },
        "summary": summary,
        "paired_against_baseline": against,
    }


def _fmt(value: float | None, digits: int = 4) -> str:
    """Significant figures, so a small but real difference never prints as 0."""
    return "–" if value is None else f"{value:.{digits}g}"


def _world(worlds: Sequence[Mapping[str, Any]] | None) -> str:
    if not worlds:
        return "?"
    parts = []
    for w in worlds:
        t = w.get("terrain") or {}
        parts.append(
            f"{t.get('source', '?')} terrain, preset {t.get('preset', '?')}, seed "
            f"{t.get('seed', '?')} ({t.get('generator_version', '?')}); "
            f"environment {w.get('environment_id', '?')}"
        )
    return "; ".join(parts)


def markdown(report: Mapping[str, Any]) -> str:
    """A readable rendering of :func:`compare`'s report (plus identities)."""
    lines = [f"# Scenario comparison: {report.get('scenario', '?')}", ""]
    ident = report.get("identity") or {}
    if ident:
        lines += [
            f"- Seeds: {report['sample_size']} paired ({report.get('seed_set') or 'explicit'})",
            f"- Code: `{ident.get('git_commit', '?')}`"
            + (" (uncommitted changes)" if ident.get("git_dirty") else ""),
            f"- Runtime binary sha256: `{str(ident.get('binary_sha256', '?'))[:16]}…`",
            f"- World: {_world(ident.get('world'))}",
            "",
        ]
    lines += [
        "## Variants",
        "",
        "| Variant | Arguments | Runs | Failed | Outside envelope "
        "| Wall s (median) | Tick ms (mean) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    args = report.get("variant_args") or {}
    for name in report["variants"]:
        s = report["summary"][name]
        lines.append(
            f"| {name} | `{' '.join(args.get(name, [])) or '(scenario defaults)'}` | {s['runs']} | "
            f"{s['failed']} | {s['outside_envelope']} | "
            f"{_fmt(s['runtime']['wall_s']['median'], 3)} | "
            f"{_fmt(s['runtime']['tick_mean_ms']['mean'], 3)} |"
        )
    lines += ["", "## Metrics", ""]
    for metric, info in report["metric_definitions"].items():
        better = info["better"]
        lines += [
            f"### `{metric}`" + (f" ({better} is better)" if better else ""),
            "",
            info["definition"],
            "",
            "| Variant | n | Mean | Median | SD | Min | Max |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for name in report["variants"]:
            d = report["summary"][name]["metrics"][metric]
            lines.append(
                f"| {name} | {d['n']} | {_fmt(d['mean'])} | {_fmt(d['median'])} | "
                f"{_fmt(d['sd'])} | {_fmt(d['min'])} | {_fmt(d['max'])} |"
            )
        rows = report["paired_against_baseline"]
        if rows:
            lines += [
                "",
                f"Paired against **{report['baseline']}** (variant − baseline):",
                "",
                "| Variant | Pairs | Mean diff | 95% interval "
                "| Improved / worsened / tied | Verdict |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
            for name, per_metric in rows.items():
                p = per_metric[metric]
                ci = p.get("ci95")
                ci_text = "–" if ci is None else f"{_fmt(ci[0])} to {_fmt(ci[1])}"
                counts = (
                    f"{p['improved']} / {p['worsened']} / {p['tied']}" if "improved" in p else "–"
                )
                lines.append(
                    f"| {name} | {p['n_pairs']} | {_fmt(p['mean_diff'])} | {ci_text} | {counts} | "
                    f"{p.get('verdict', 'not scored')} |"
                )
        lines.append("")
    failures = {n: s["failures"] for n, s in report["summary"].items() if s["failures"]}
    if failures:
        lines += ["## Failed runs", ""]
        for name, per_seed in failures.items():
            for seed, error in per_seed.items():
                lines.append(f"- {name}, seed {seed}: {error}")
        lines.append("")
    return "\n".join(lines)
