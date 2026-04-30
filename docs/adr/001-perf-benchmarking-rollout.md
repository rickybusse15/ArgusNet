# ADR 001: Performance Benchmarking Rollout

## Status

Accepted.

## Context

`docs/PERFORMANCE_AND_BENCHMARKING.md` defines ArgusNet benchmark levels, regression thresholds,
caching policy, language-boundary guidance, and the five-phase performance roadmap. The standard
needed an operational runner, CI gates, accepted baselines, and Rust/Python microbenchmark entry
points.

## Decision

Ship the first operational standard in one reviewable PR:

- Python `argusnet benchmark` runs canonical fast and slow scenario suites.
- Fast PR CI runs Python smoke benchmarks and performance regression checks.
- Nightly CI runs the slow multi-seed scenario sweep and full Rust Criterion benchmarks.
- Fast-suite accepted baselines live under `tests/golden/performance/`.
- Regression gates follow the standard thresholds: green within 5%, yellow at 5-20%, red above
  20%, with `ARGUSNET_PERF_OVERRIDE=1` as an explicit manual override for red failures.

## Consequences

The initial PR is intentionally broad, but it creates one shared measurement system before deeper
Rust migrations or engine-level rewrites. Slow-suite goldens and automated issue filing can be
added after nightly measurements stabilize.
