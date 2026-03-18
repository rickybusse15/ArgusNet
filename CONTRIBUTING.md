# Contributing Guide

## Architecture Governance

All changes must follow the architecture update plan. See `docs/` for subsystem documentation.

### Before making changes

1. Read `CLAUDE.md` for project conventions
2. Read `AGENT_TEAM.md` for multi-agent execution protocol
3. Read `docs/architecture.md` for current module map
4. Check `docs/adr/` for architecture decisions that affect your area

### Code Completion Standards (from architecture update Section 14)

#### Functional
- Compiles cleanly
- Passes tests (`cargo test` and `python3 -m pytest tests/ -q`)
- Integrates through existing interfaces unless an ADR approves a change
- Includes error handling
- Documents new config or runtime dependencies
- Logs important state transitions

#### Quality
- No hidden global state
- Preserves deterministic behavior where seeded
- Explicit units in types or documentation
- No duplicate planner or state-transition logic
- No unresolved placeholders in merged code
- Clear trait and ownership boundaries

#### Performance
- No unbounded allocations in hot loops
- Document per-frame computational cost
- Async/background for large terrain or asset loading
- Document cache strategy for expensive queries

#### Safety & Integrity
- Constraints cannot be silently bypassed
- No unsafe Rust unless justified and documented
- Validate state transitions
- Age out stale data explicitly
- Define fallback for missing terrain, sensor, or comms data

### Definition of Done

A feature is not complete unless it includes:
- [ ] Implementation
- [ ] Tests
- [ ] Config updates (if relevant)
- [ ] Documentation update
- [ ] Replay/UI visibility (if relevant)
- [ ] Metrics integration (if relevant)

### Architecture Decision Records

Any nontrivial architecture change requires an ADR in `docs/adr/`. Use `docs/adr/000-template.md` as the template.

### Conflict Resolution

When new work conflicts with existing code:
1. Do not silently patch around it
2. Identify the old assumption and new requirement
3. Explain why the old structure fails
4. Propose the smallest viable fix
5. Define migration steps
6. Identify tests needed
7. Write an ADR if nontrivial

Priority order: safety > architectural consistency > determinism > performance > convenience

## Build & Test

```bash
pip install -e .                    # Python deps
cargo test                          # Rust tests
python3 -m pytest tests/ -q         # Python tests
smart-tracker sim --duration-s 60   # Run simulation
```

## Crate / Module Map

See `docs/architecture.md` for the current map. The planned evolution (docs/STATE_OWNERSHIP.md) adds:
- `terrain-engine` — analytic terrain queries
- `fusion-engine` — fused track authority
- `trajectory-engine` — feasible path generation
- `safety-engine` — constraint enforcement
- `mission-gen` — procedural scenario generation
- `planner-engine` — cooperative role-based planning
- `eval-suite` — metrics and benchmarking
