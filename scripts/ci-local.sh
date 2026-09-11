#!/usr/bin/env bash
# Run the CI suite locally, on Linux or macOS.
#
# Every check here mirrors a job in .github/workflows/ci.yml. When that file
# changes, change this one -- a local runner that drifts from CI is worse than
# none, because it reports green for a suite that no longer matches.
#
# Written for bash 3.2, which is what macOS ships: no associative arrays, no
# `${var,,}`, no `mapfile`.
#
#   scripts/ci-local.sh                 # every job
#   scripts/ci-local.sh rust-fast       # one or more jobs
#   scripts/ci-local.sh --list          # what is available
#   scripts/ci-local.sh --quick         # skip the slow jobs (viewer, packaging, security)
#
# Exit status is non-zero if any job failed, so it works as a pre-push hook.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

OS="$(uname -s)"
ALL_JOBS="python rust-fast rust-lint rust-viewer packaging security"
QUICK_JOBS="python rust-fast rust-lint"

# Job results, recorded as parallel arrays (bash 3.2 has no associative arrays).
RESULT_NAMES=()
RESULT_STATUS=()
RESULT_SECONDS=()

# Set by a job that did nothing because its tools are absent. Reporting "pass"
# for a job that ran no checks is a green tick for work that did not happen.
JOB_SKIPPED=""

bold()  { printf '\033[1m%s\033[0m\n' "$1"; }
green() { printf '\033[32m%s\033[0m\n' "$1"; }
red()   { printf '\033[31m%s\033[0m\n' "$1"; }
yellow(){ printf '\033[33m%s\033[0m\n' "$1"; }

FAILED_STEPS=()

# Run one step. A failure is recorded and the job continues, so a single run
# reports every problem rather than only the first.
step() {
  local label="$1"; shift
  printf '\n  \033[1m> %s\033[0m\n' "$label"
  if "$@"; then
    return 0
  fi
  red "  FAILED: $label"
  FAILED_STEPS+=("$label")
  return 1
}

# Non-blocking step: mirrors CI's `continue-on-error: true` scanners.
soft_step() {
  local label="$1"; shift
  printf '\n  \033[1m> %s\033[0m (advisory)\n' "$label"
  if ! "$@"; then
    yellow "  advisory step reported findings: $label"
  fi
  return 0
}

run_job() {
  local name="$1"
  local started ended
  started=$(date +%s)
  FAILED_STEPS=()
  JOB_SKIPPED=""
  bold ""
  bold "=== $name ==="
  "job_${name//-/_}"
  ended=$(date +%s)
  RESULT_NAMES+=("$name")
  RESULT_SECONDS+=($((ended - started)))
  if [ ${#FAILED_STEPS[@]} -eq 0 ] && [ -n "$JOB_SKIPPED" ]; then
    RESULT_STATUS+=("skip")
    yellow "=== $name: SKIPPED ($JOB_SKIPPED) ==="
  elif [ ${#FAILED_STEPS[@]} -eq 0 ]; then
    RESULT_STATUS+=("pass")
    green "=== $name: pass ==="
  else
    RESULT_STATUS+=("FAIL")
    red "=== $name: FAIL (${FAILED_STEPS[*]}) ==="
  fi
}

# --- system dependencies -----------------------------------------------------
#
# CI apt-installs these on Linux. Here they are only CHECKED: installing system
# packages behind a contributor's back is not this script's job, and on macOS
# Bevy links against CoreAudio and IOKit instead, so nothing is needed at all.
check_system_deps() {
  if [ "$OS" != "Linux" ]; then
    return 0
  fi
  local missing=""
  pkg-config --exists alsa 2>/dev/null || missing="$missing libasound2-dev"
  pkg-config --exists libudev 2>/dev/null || missing="$missing libudev-dev"
  if [ -n "$missing" ]; then
    yellow "  missing Bevy build dependencies:$missing"
    yellow "  install with: sudo apt-get install -y pkg-config libasound2-dev libudev-dev"
    return 1
  fi
  return 0
}

# --- jobs --------------------------------------------------------------------

job_python() {
  step "ruff check" ruff check src/ tests/
  step "ruff format --check" ruff format --check src/ tests/
  step "mypy" mypy src/argusnet/ --ignore-missing-imports --follow-imports=skip
  step "protobuf bindings up to date" python3 scripts/regenerate_proto.py --check
  step "proto fields have a disposition" python3 scripts/check_field_parity.py
  step "pytest" python3 -m pytest tests/ -q --timeout=120
  step "fast benchmark smoke" python3 -m pytest tests/benchmarks/ -q -m benchmark_fast \
    --benchmark-only --benchmark-json=/tmp/argusnet-bench.json
  step "performance regression" python3 -m pytest tests/test_performance_regression.py -q
}

job_rust_fast() {
  # --exclude rather than naming crates, matching CI: a new workspace member is
  # covered by default instead of silently going untested.
  step "cargo test --workspace --exclude argusnet-viewer" \
    cargo test --workspace --exclude argusnet-viewer
}

job_rust_lint() {
  check_system_deps || FAILED_STEPS+=("system dependencies")
  # Formatting first: it compiles nothing, so a misformatted file fails in
  # seconds instead of behind a full clippy pass.
  step "cargo fmt --check" cargo fmt --all -- --check
  step "clippy" cargo clippy --workspace --all-targets -- -D warnings
}

job_rust_viewer() {
  check_system_deps || FAILED_STEPS+=("system dependencies")
  step "cargo test -p argusnet-viewer" cargo test -p argusnet-viewer
  step "build the viewer binary" cargo build -p argusnet-viewer --bin argusnet-viewer
  step "headless render smoke test" render_smoke
}

# The only check in the repository that reaches a shader. On Linux it needs
# lavapipe (mesa-vulkan-drivers) because there is no GPU; on macOS wgpu uses
# Metal directly and needs nothing installed.
render_smoke() {
  if [ "$OS" = "Linux" ] && [ -z "${ARGUSNET_SKIP_VULKAN_CHECK:-}" ]; then
    if ! ls /usr/share/vulkan/icd.d/*.json >/dev/null 2>&1; then
      yellow "  no Vulkan ICD found; install mesa-vulkan-drivers for lavapipe"
      return 1
    fi
  fi
  python3 -m pip install --quiet pillow || return 1
  python3 scripts/prepare_viewer_smoke_scene.py /tmp/argusnet-render-scene || return 1
  # 650 wide, not 640: wgpu pads each readback row to a 256-byte boundary, and
  # a width divisible by 64 needs no padding at all. Rendering at a round width
  # leaves the padded path untested.
  ./target/debug/argusnet-viewer \
    --scene /tmp/argusnet-render-scene \
    --headless \
    --output /tmp/argusnet-headless.png \
    --width 650 --height 480 --camera isometric || return 1
  python3 - <<'PY'
import sys
from PIL import Image

image = Image.open("/tmp/argusnet-headless.png").convert("RGB")
colors = image.getcolors(maxcolors=1 << 20) or []
top = sorted(colors, reverse=True)[:3]
if len(colors) <= 50:
    sys.exit(f"headless render looks empty: {top}")
green_pixels = sum(
    count for count, (r, g, b) in colors if g > 100 and g > 1.4 * r and g > 1.4 * b
)
if green_pixels <= 30:
    sys.exit(f"holographic track missing: {green_pixels} green pixels")
# CI asserts a floor, not an equality: a different Mesa build, or Metal instead
# of lavapipe, legitimately produces a different count.
print(f"headless render OK: {len(colors)} distinct colours, {green_pixels} green pixels")
PY
}

job_packaging() {
  local venv=/tmp/argusnet-contract-venv
  rm -rf "$venv" dist/
  step "build the wheel" sh -c 'python3 -m pip install --quiet --upgrade build && python3 -m build --wheel --outdir dist/ .'
  step "install into a clean environment" sh -c "python3 -m venv $venv && $venv/bin/pip install --quiet dist/*.whl"
  # Run from /tmp: the script refuses to run inside the source tree, since a
  # checkout on the path would mask the failures it exists to catch.
  step "installed package honours its contracts" \
    sh -c "cd /tmp && $venv/bin/python '$REPO_ROOT/scripts/check_installed_contract.py'"
}

job_security() {
  # All advisory, matching CI's continue-on-error. gitleaks is not run here:
  # CI uses a container action for it, and there is no local equivalent to
  # invoke without asking the contributor to install one.
  local ran=0
  local absent=""
  if command -v pip-audit >/dev/null 2>&1; then
    soft_step "pip-audit" pip-audit; ran=$((ran + 1))
  else
    absent="$absent pip-audit"
  fi
  if command -v bandit >/dev/null 2>&1; then
    soft_step "bandit" bandit -q -r src/ -x src/argusnet/v1; ran=$((ran + 1))
  else
    absent="$absent bandit"
  fi
  if command -v cargo-audit >/dev/null 2>&1; then
    soft_step "cargo audit" cargo audit; ran=$((ran + 1))
  else
    absent="$absent cargo-audit"
  fi
  if [ -n "$absent" ]; then
    yellow "  not installed:$absent"
    yellow "  install with: pip install pip-audit bandit && cargo install cargo-audit --locked"
  fi
  # A job that ran nothing must not report pass.
  if [ $ran -eq 0 ]; then
    JOB_SKIPPED="no scanners installed"
  fi
}

# --- entry point -------------------------------------------------------------

usage() {
  cat <<USAGE
Run the CI suite locally. Mirrors .github/workflows/ci.yml.

Usage: scripts/ci-local.sh [--quick] [--list] [job ...]

Jobs: $ALL_JOBS
  --quick   only: $QUICK_JOBS
  --list    print the job list and exit

Notes:
  * The three Python versions CI matrixes over become one here: whichever
    python3 is on PATH. A version-specific failure will still only show in CI.
  * Rendering needs mesa-vulkan-drivers on Linux; on macOS it uses Metal.
  * The security scanners are advisory and are skipped when not installed.
USAGE
}

JOBS=""
for arg in "$@"; do
  case "$arg" in
    --list) echo "$ALL_JOBS"; exit 0 ;;
    --quick) JOBS="$QUICK_JOBS" ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "unknown option: $arg" >&2; usage >&2; exit 2 ;;
    *) JOBS="$JOBS $arg" ;;
  esac
done
[ -z "${JOBS// /}" ] && JOBS="$ALL_JOBS"

for job in $JOBS; do
  case " $ALL_JOBS " in
    *" $job "*) ;;
    *) echo "unknown job: $job (have: $ALL_JOBS)" >&2; exit 2 ;;
  esac
done

bold "ArgusNet local CI  --  $OS, $(python3 --version 2>&1), $(cargo --version 2>&1)"
START=$(date +%s)
for job in $JOBS; do
  run_job "$job"
done
TOTAL=$(( $(date +%s) - START ))

bold ""
bold "=== summary (${TOTAL}s) ==="
EXIT=0
PASSED=0
SKIPPED=0
i=0
while [ $i -lt ${#RESULT_NAMES[@]} ]; do
  printf '  %-14s %-5s %4ss\n' "${RESULT_NAMES[$i]}" "${RESULT_STATUS[$i]}" "${RESULT_SECONDS[$i]}"
  case "${RESULT_STATUS[$i]}" in
    FAIL) EXIT=1 ;;
    skip) SKIPPED=$((SKIPPED + 1)) ;;
    pass) PASSED=$((PASSED + 1)) ;;
  esac
  i=$((i + 1))
done
if [ $EXIT -ne 0 ]; then
  red "at least one job failed"
elif [ $PASSED -eq 0 ]; then
  yellow "nothing ran: every selected job was skipped"
elif [ $SKIPPED -gt 0 ]; then
  yellow "$PASSED job(s) passed, $SKIPPED skipped -- a skipped job checked nothing"
else
  green "all selected jobs passed"
fi
exit $EXIT
