# Contributing Guide

## Current and target runtime

Start with [state ownership](docs/STATE_OWNERSHIP.md) for implemented behavior and
[the Rust runtime target](docs/RUST_RUNTIME_ARCHITECTURE.md) for the end state.
[ADR-009](docs/adr/009-independently-scheduled-runtime.md) introduces independently
scheduled execution after the compatibility port; do not silently change legacy
causal order while moving code. Python remains supported offline tooling.

Each runtime PR names its P package, state owner, input/output timestamps,
execution model, resource/loss policy and tests. Preserve acquisition time and
component freshness; do not derive domain dt from render frames. Never hold
shared runtime locks across I/O or let worker completion order alter deterministic
results. Profile before adding threads or replacing data structures.

Keep implementation, local verification, pushed status, remote checks and release
status separate. Update runnable usage only for implemented interfaces. New
sensor/actuation adapters require [hardware qualification](docs/HARDWARE_INTEGRATION.md),
not just a desktop test. Documentation-only changes need link/consistency checks;
code changes run the relevant checks in [local CI](docs/LOCAL_CI.md).

## Architecture Governance

All changes must follow the architecture update plan. See `docs/` for subsystem documentation.

### Before making changes

1. Read `CLAUDE.md` for project conventions
2. Read `docs/architecture.md` for current module map
3. Check `docs/adr/` for architecture decisions that affect your area

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
argusnet sim --duration-s 60        # Run simulation
```

## Crate / Module Map

See [architecture](docs/architecture.md) for current components and
[the target ownership table](docs/RUST_RUNTIME_ARCHITECTURE.md#2-runtime-modules-and-owned-state)
for their final roles. Reuse `argusnet-world`, `argusnet-core`,
`argusnet-localization`, `argusnet-sensing`, `argusnet-mapping`,
`argusnet-planning`, `safety-engine`, `argusnet-sim` and `argusnet-replay`.
`argusnet-viewer` owns presentation; `argusnet-server`/`argusnet-proto` own external
transport. `argusnet-native` supports transitional/offline Python consumers.
Do not create duplicate `*-engine` crates just because an older roadmap used a
placeholder name. New crate extraction needs a demonstrated ownership boundary.
