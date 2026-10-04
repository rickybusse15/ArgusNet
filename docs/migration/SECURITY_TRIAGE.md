# G3: scanner triage

Guardrail **G3** asks that every finding of pip-audit, bandit and cargo-audit
be resolved or documented with an owner and an expiry, that pins be kept, and
that the scanners become required checks **one at a time**, never by blanket
ignores. First pass: 2026-10-04, on `main` at `0144874`.

## State of each scanner

| Scanner | Gate | Result after this pass |
| --- | --- | --- |
| bandit (`bandit -q -r src/ -x src/argusnet/v1`) | **blocking** in CI and `ci-local.sh` | clean: 19 low-severity findings, each fixed or carrying a reasoned `# nosec <id>` on the line, reason on the line above |
| pip-audit | advisory | clean for the project's dependencies; the runner's own pip and setuptools are upgraded before the audit |
| cargo-audit | advisory | no vulnerabilities; 2 unmaintained-crate warnings, documented below |
| gitleaks | advisory (CI only) | not run locally |

pip-audit and cargo-audit become blocking next, after their triage has held
for a release: advisories arrive daily, and a gate that turns red on a
dependency nobody changed needs an owner ready to act on it.

## bandit (19, all low severity)

| Finding | Where | Decision |
| --- | --- | --- |
| B404 `import subprocess` (4) | `adapters/argusnet_grpc.py`, `cli/main.py`, `evaluation/benchmarks.py`, `evaluation/perf_summary.py` | `# nosec B404`: every call passes an argument list, never a shell |
| B603 subprocess call (7) | the daemon and its cargo build (`argusnet_grpc.py`); daemon, simulation (`sys.executable -m argusnet`) and viewer launched by `argusnet live` (`cli/main.py`); fixed `git`/`rustc`/`cargo` metadata commands (`benchmarks.py`, `perf_summary.py`) | `# nosec B603` with the reason: project-built binaries or the running interpreter, argument lists, no shell; arguments are the operator's own CLI input |
| B405 `xml.etree` | `evaluation/export.py` | `# nosec B405`: builds KML and GPX, never parses XML |
| B110/B112 `try`/`except`/`pass` or `continue` (5) | `planning/inspection.py` mission validity checks | **fixed**: a node or target whose state cannot be read is still skipped, but the skip is logged with the reason instead of passing silently |
| B101 `assert` (2) | `planning/inspection.py`, `sensing/ingestion/frame_stream.py` | **fixed**: explicit `RuntimeError`s, since `python -O` strips asserts |

## pip-audit

| Finding | Decision |
| --- | --- |
| pip 24.0 (PYSEC-2026-1795, -1796, -196, -2875, -2876, -3721) | environment tooling, not a project dependency: CI upgrades pip before auditing |
| setuptools 79.0.1 (PYSEC-2026-3447) | the build floor is raised to `setuptools>=83` (`pyproject.toml`); CI upgrades it before auditing |
| `argusnet` itself | not on PyPI, so not auditable by pip-audit (expected) |

## cargo-audit

| Finding | Crate path | Decision |
| --- | --- | --- |
| RUSTSEC-2026-0194, -0195 quick-xml 0.39.2 (high) | wayland-scanner → winit → bevy | **fixed**: `Cargo.lock` to quick-xml 0.41.0 via wayland-scanner 0.31.11 |
| RUSTSEC-2026-0257 webbrowser 1.2.0 | bevy_egui | **fixed**: webbrowser 1.2.4 |
| RUSTSEC-2026-0221 event-listener 5.4.1 (unsound) | bevy_asset | **fixed**: 5.4.2 |
| RUSTSEC-2026-0186 memmap2 0.9.10 (unsound) | fontique → bevy_text | **fixed**: 0.9.11 |
| RUSTSEC-2024-0436 paste (unmaintained) | simba → nalgebra | **documented**: a proc-macro with no runtime surface; owner: runtime maintainers; expiry: the next nalgebra upgrade, or 2027-04-01 |
| RUSTSEC-2026-0192 ttf-parser (unmaintained) | ab_glyph → sctk-adwaita → winit | **documented**: parses only the system decoration font on Wayland; owner: viewer maintainers; expiry: the next Bevy upgrade, or 2027-04-01 |

The lock updates are semver-compatible; the workspace builds and the viewer's
tests, GPU probes and windowed smoke pass with them.
