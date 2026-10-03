#!/usr/bin/env bash
# Run the windowed viewer on a virtual X display for a while and fail if it
# exits or panics. The headless render path leaves egui out entirely, so this
# is the only check that draws the operator shell: a UI that panics on its
# first frame passes every other test.
#
#   scripts/viewer_window_smoke.sh [SCENE_DIR] [SECONDS]
#
# Needs xvfb-run and libxkbcommon-x11 (Debian: xvfb libxkbcommon-x11-0), and a
# Vulkan driver (lavapipe: mesa-vulkan-drivers).
set -u
SCENE=${1:-/tmp/argusnet-render-scene}
SECONDS_TO_RUN=${2:-20}
BINARY=${ARGUSNET_VIEWER:-./target/debug/argusnet-viewer}
LOG=$(mktemp)
if ! command -v xvfb-run >/dev/null 2>&1; then
  echo "xvfb-run not found: install xvfb (and libxkbcommon-x11-0)"
  exit 1
fi
timeout "$SECONDS_TO_RUN" xvfb-run -a -s "-screen 0 1440x920x24" \
  "$BINARY" --scene "$SCENE" >"$LOG" 2>&1
status=$?
if grep -q "panicked" "$LOG"; then
  grep -A3 "panicked" "$LOG" | head -20
  echo "windowed viewer panicked"
  exit 1
fi
# 124: still running when the time ran out, which is the pass.
if [ "$status" -ne 124 ]; then
  tail -20 "$LOG"
  echo "windowed viewer exited early with status $status"
  exit 1
fi
echo "windowed viewer ran ${SECONDS_TO_RUN} s without exiting or panicking"
