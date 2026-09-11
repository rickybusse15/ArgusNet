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
- **Python 3.10, 3.11 and 3.12.** The `python-tests` job matrixes over all
  three, and `actions/setup-python` on a self-hosted macOS runner will download
  a build if it isn't already there. If you would rather run one version, drop
  the matrix down — but say so in the PR, since it narrows the gate.
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
- **The render smoke test will use Metal**, not lavapipe. It should pass, and
  it will be far faster. **This has not been verified on macOS from here** —
  this container is Linux — so treat the first `rust-viewer` run as the check
  of that claim rather than an assumption.
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
