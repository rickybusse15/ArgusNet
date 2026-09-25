# Running CI without GitHub-hosted runners

GitHub-hosted Actions minutes are a finite monthly allowance. When they run
out, every job in a run is created and then **fails within seconds** — no
runner is ever assigned, so the job records carry no `runner_id`, no
`runner_name` and no `steps`, and the log endpoint returns HTTP 404. The
failures name your code, but nothing of yours ever executed. Worth recognising,
because it looks exactly like a repository-wide breakage.

Two ways to keep working:

1. **`scripts/ci-local.sh`** — run the same checks on the machine in front of
   you. Nothing to install or register.
2. **A self-hosted runner** — register your own machine with GitHub so pushes
   and pull requests get real checks again. Self-hosted minutes are not
   metered.

They are not exclusive; the local script is the faster inner loop either way.

---

## Runtime verification additions (planned)

Existing jobs below remain the runnable local checks. P4.10 adds independent-rate
and rational-deadline tests, worker-order invariance, acquisition/delivery timing,
full reset/restore with queued jobs, bounded recorder/command overload and
freshness/epoch viewer fixtures. Do not label those jobs implemented until they
are actually wired into the script/workflow.

Use [verification and rollout](migration/VERIFICATION_AND_ROLLOUT.md) for acceptance
and [performance tracking](PERFORMANCE_AND_BENCHMARKING.md) for named-hardware
benchmarks. Software rendering smoke is not a GPU frame-budget pass. Local tests
are not remote CI, and neither is [SIL/HIL qualification](HARDWARE_INTEGRATION.md).
For documentation-only updates, inspect the diff, check local links/anchors and
confirm example commands against current parsers; do not report runtime tests
as passed merely because no source code changed.

## 1. `scripts/ci-local.sh`

```bash
scripts/ci-local.sh              # every job
scripts/ci-local.sh --quick      # python, rust-fast, rust-lint
scripts/ci-local.sh rust-viewer  # one or more jobs by name
scripts/ci-local.sh --list
```

Each job mirrors one in `.github/workflows/ci.yml`:

| job | what it runs |
|---|---|
| `python` | ruff, mypy, the proto-regeneration and field-disposition checks, pytest, the benchmark smoke, the perf gate |
| `rust-fast` | `cargo test --workspace --exclude argusnet-viewer` |
| `rust-lint` | `cargo fmt --check`, `clippy --workspace --all-targets -D warnings` |
| `rust-viewer` | the viewer's tests, the viewer binary, and the headless render smoke test |
| `packaging` | builds the wheel, installs it into a clean venv, runs the installed-package contract from `/tmp` |
| `security` | pip-audit, bandit, cargo-audit — all advisory, all skipped when not installed |

A failing step does not stop the job: one run reports every problem rather than
only the first. A job whose tools are all absent reports **skip**, not pass —
a green tick for checks that did not run is worse than no tick.

Measured in this repository's dev container (4 cores), warm build:

| job | time |
|---|---|
| `python` | 4m 31s |
| `rust-viewer` | 2m 32s |
| `rust-lint` | 1m 21s |
| `rust-fast` | 1m 17s |
| `packaging` | 54s |
| `security` | skipped (scanners not installed) |

The exit status is non-zero if any job failed, so it works as a pre-push hook.
`--quick` is still four and a half minutes because of pytest, so for a hook on
every push the two Rust jobs alone are the better trade:

```bash
printf '#!/bin/sh\nexec scripts/ci-local.sh rust-fast rust-lint\n' > .git/hooks/pre-push
chmod +x .git/hooks/pre-push
```

### Three ways it is not identical to CI

Stated plainly, because a local runner that quietly differs is worse than none:

- **One Python version, not three.** CI matrixes 3.10, 3.11 and 3.12; this runs
  whichever `python3` is on `PATH`. A version-specific failure will still only
  appear in CI.
- **No gitleaks.** CI uses a container action for it, and there is no local
  equivalent to invoke without asking you to install one.
- **Different render output.** The smoke test asserts a *floor* — more than 50
  distinct colours, more than 30 green pixels — never an equality, because
  lavapipe on Linux and Metal on macOS legitimately produce different counts.
  This container gives 10,703 colours; the hosted runner gave 7,946; a local
  Mesa build gave 6,493.

### Prerequisites

- Rust via `rustup` (the pinned toolchain in `rust-toolchain.toml` installs on
  first use), and `pip install -e ".[dev]"`.
- **Linux only:** `sudo apt-get install -y pkg-config libasound2-dev libudev-dev`
  to link anything Bevy, plus `mesa-vulkan-drivers` for the render smoke test —
  a hosted runner has no GPU and wgpu has no software fallback of its own. The
  script checks for both and tells you what is missing rather than installing
  anything itself.
- **macOS:** nothing. Bevy links CoreAudio and IOKit, and wgpu uses Metal.

---

## 2. A self-hosted runner on a MacBook

This gets pushes and pull requests checked again, with the results on the PR
where reviewers look.

### Register the runner

1. In the repository: **Settings → Actions → Runners → New self-hosted runner**,
   and pick **macOS** with the right architecture (`arm64` for Apple silicon,
   `x64` for Intel).
2. Run the commands GitHub shows you. They download the runner, then:

   ```bash
   ./config.sh --url https://github.com/rickybusse15/ArgusNet --token <TOKEN>
   ```

   The token is short-lived and shown on that page — do not commit it or paste
   it anywhere else.
3. Start it. For a foreground run while you watch it work:

   ```bash
   ./run.sh
   ```

   To have it start at login and keep running:

   ```bash
   ./svc.sh install
   ./svc.sh start
   ```

### Point the workflow at it

`.github/workflows/ci.yml` reads a repository variable:

```yaml
runs-on: ${{ vars.CI_RUNNER || 'ubuntu-latest' }}
```

Set it under **Settings → Secrets and variables → Actions → Variables**:

| name | value |
|---|---|
| `CI_RUNNER` | `self-hosted` |

Deleting the variable moves everything back to hosted runners; nothing in the
workflow needs editing either way.

### What the machine needs

- **Xcode command line tools:** `xcode-select --install`
- **Rust:** `rustup` — `rust-toolchain.toml` pins the version and installs it
  on first use.

  If you run the runner as a **service** (`./svc.sh install && ./svc.sh start`
  on macOS — no sudo, it is a launchd *user* agent), note that launchd gives it
  a minimal `PATH` (`/usr/bin:/bin:/usr/sbin:/sbin`) and never sources
  `~/.zprofile`. Nothing installed by rustup, Homebrew or conda is on that
  path, so every cargo step fails with `rustup: command not found` while
  rustup works perfectly in your terminal. Running the runner from a terminal
  instead hides the problem entirely, because it inherits your shell's `PATH`.

  The workflow handles that itself: every toolchain step sources
  `scripts/ci-rust-path.sh`, which searches the usual install locations —
  rustup's own `~/.cargo/bin`, `$CARGO_HOME/bin`, Homebrew's
  `/opt/homebrew/bin` and `/usr/local/bin`, and the conda prefixes.

  **But `rustup: command not found` is not always a PATH problem, and
  assuming it is wastes CI rounds.** On 2026-09-11 this runner hit it with
  `~/.cargo/bin` already on `PATH`, both `~/.cargo` and `~/.rustup` present,
  and both toolchains — including the pinned 1.98.0 — intact. What had gone
  missing was the **`rustup` binary itself**, leaving every proxy symlink in
  `~/.cargo/bin` (`cargo`, `rustc`, `rustfmt`, …, all of which point at
  `rustup`) dangling.

  That case is easy to misread, and it was misread twice here: `find -type f`
  skips symlinks, so `~/.cargo/bin` reads as empty and the machine looks like
  it never had Rust. The step now names the diagnosis outright — dangling
  proxies, toolchains present but rustup unrunnable, or nothing installed —
  rather than leaving it to be inferred from two directory listings.

  The repair for all three is the same, and it keeps `rust-toolchain.toml`
  authoritative in a way that pointing `PATH` at a toolchain directory would
  not:

  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  ```

  Adding `PATH=<your rust bin>:/usr/bin:/bin:/usr/sbin:/sbin` to
  `actions-runner/.env` is an equivalent fix if you prefer it on the machine.
- **A Python 3.10 or newer.** Whichever one `python3` resolves to is the one
  CI will use; nothing else is needed.

  **Correction (2026-09-11).** This section previously said
  `actions/setup-python` would download an interpreter if one was missing. It
  will not, on a self-hosted Mac, for two independent reasons — and the first
  attempt at a fix (`AGENT_TOOLSDIRECTORY` in the runner's `.env`) addresses
  neither:

  1. The macOS interpreters it fetches are **`.pkg` installers**. The `setup.sh`
     inside the tarball runs `sudo installer -pkg … -target /` to write into
     `/Library/Frameworks`. A personal Mac has no passwordless sudo, and giving
     CI any would be a worse idea than the problem it solves.
  2. The runner's default tool cache on macOS is
     **`/Users/runner/hostedtoolcache`** — a path under a user that exists only
     on GitHub's images — so the script dies at its first `mkdir` with
     `Permission denied` well before it reaches the `sudo`.

  So `actions/setup-python` is now **skipped on a self-hosted runner**, and the
  `python-tests` matrix collapses to a single `system` entry there rather than
  running the same interpreter three times (on one runner that would serialise
  into three identical jobs). The three-version matrix still runs on GitHub's
  images, so the gate is unchanged where it can be met.

  **CI never installs into your interpreter.** Every job that needs Python
  builds a virtualenv under `RUNNER_TEMP` first and works from that, so
  `pip install -e ".[dev]"` cannot disturb a conda base or a system Python.
- Nothing else: the apt steps are guarded on `runner.os == 'Linux'` and are
  skipped.

### Things worth knowing before you rely on it

- **A self-hosted runner executes whatever a workflow tells it to, on your
  machine, with your files.** That is fine for a repository only you push to.
  It is *not* fine for a public repository accepting pull requests from
  strangers: a fork's PR could run arbitrary code on your laptop. GitHub says
  the same, more forcefully. If this repository ever takes outside
  contributions, require approval for fork PRs (**Settings → Actions → General
  → Fork pull request workflows**) before leaving the runner enabled.
- **The workspace is not clean between runs.** A hosted runner starts from a
  fresh image every time; yours keeps `target/` and the checkout. That makes
  builds much faster and means a stale artifact can mask a broken cold build.
  Worth an occasional `cargo clean` — and note `target/` reached 24 GB in this
  repository before the debuginfo change, so keep an eye on disk.
- **The render smoke test uses Metal**, not lavapipe. This was written as an
  unverified prediction and is now **confirmed**: the first `rust-viewer` run on
  the macOS runner passed every step, the headless render included, in 1m21s.
  It is the one claim in this document that could not be checked from the Linux
  container, and it held.
- **Timing baselines do not transfer.** The perf gate's blocking half compares
  deterministic work counters, which are machine-independent and fine. The
  wall-clock half stays opt-in behind `ARGUSNET_RUN_PERF_TIMING` precisely
  because absolute timings do not survive a change of machine — do not enable
  it on the laptop and expect the committed baselines to mean anything.
- **`gitleaks` will be skipped.** It is a container action and macOS runners
  cannot host containers; the step is guarded on `runner.os == 'Linux'`. Run
  `brew install gitleaks && gitleaks detect` by hand if you want it.

---

## Which to use

Use `scripts/ci-local.sh` while iterating — it is the same checks with none of
the queueing, and it is the only one available when nothing has been pushed
yet. Register the self-hosted runner when you want the results attached to a
pull request, or when several branches are in flight and you would rather not
run each one by hand.

## When `rustup` goes missing from the runner

This has now happened twice on the same machine, and on 2026-09-15 it took
**every one of the six checks** down rather than only the three Rust jobs —
`packaging-contract` and `python-tests` build the `argusnet-native` cdylib
during `pip install`, and `security` runs `cargo-audit`, so all of them need a
working `cargo`.

The signature is distinctive and worth recognising:

```
##[error]rustup is not on PATH and was not found in any known location.
DIAGNOSIS: rustup's proxy symlinks in ~/.cargo/bin are intact but the
rustup binary they all point to is GONE, so every one of them dangles.
```

`~/.cargo/bin` is full of symlinks that all point at the `rustup` binary
beside them. If that one binary disappears, every proxy — `cargo`, `rustc`,
`cargo-clippy`, `rustfmt` — dangles at once, while the toolchains under
`~/.rustup/toolchains` remain untouched. Note that `find -type f` skips
symlinks, so the directory reads as *empty* and the machine looks like it
never had Rust installed. That is the wrong conclusion and it cost two CI
rounds the first time.

**The repair is one command**, and it is still the right one:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### The degraded fallback, and what it does not give up

`scripts/ci-rust-path.sh` no longer stops there. When `rustup` is absent but
the pinned toolchain is installed, it puts that toolchain's `bin` directory on
PATH directly and CI continues in a **degraded mode**, announced with a
`::warning::` on every run.

Pointing PATH at a toolchain directory is normally the wrong answer, because it
bypasses `rust-toolchain.toml` and lets the pinned version drift silently —
which is the exact failure the pin exists to prevent (`main` went red twice on
toolchain drift before it was added). The fallback answers that objection
rather than ignoring it:

- The directory is **chosen by** the pin. The channel is read out of
  `rust-toolchain.toml`; it is never hardcoded and never guessed.
- The result is **verified against** the pin. `rustc --version` must report the
  pinned channel, so a mismatch fails *louder* than before, not quieter.
- A channel the machine does not have **fails outright**. The fallback cannot
  silently substitute a different toolchain, which is the whole risk.

What is genuinely lost is rustup's component management. `cargo clippy` and
`cargo fmt` need `cargo-clippy`, `clippy-driver` and `rustfmt` to be present in
that same directory; the script checks for each and names any that are missing,
so a `rust-lint` failure reads as a missing component rather than as a lint
error.

Treat a run with that warning as a run to fix, not a new normal.
