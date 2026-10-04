#!/usr/bin/env bash
# Run the windowed viewer on a virtual X display for a while and fail if it
# exits or panics, or if the operator shell is not on screen. The headless
# render path leaves egui out entirely, so this is the only check that draws
# the shell: a UI that panics on its first frame, or is drawn every frame but
# shown nowhere, passes every other test.
#
#   scripts/viewer_window_smoke.sh [SCENE_DIR] [SECONDS]
#
# Needs xvfb-run and libxkbcommon-x11 (Debian: xvfb libxkbcommon-x11-0), a
# Vulkan driver (lavapipe: mesa-vulkan-drivers), and for the on-screen check
# ImageMagick's `import` and Python with Pillow.
set -u
SCENE=${1:-/tmp/argusnet-render-scene}
SECONDS_TO_RUN=${2:-20}
BINARY=${ARGUSNET_VIEWER:-./target/debug/argusnet-viewer}
LOG=$(mktemp)
SHOT=$(mktemp --suffix=.png)
if ! command -v xvfb-run >/dev/null 2>&1; then
  echo "xvfb-run not found: install xvfb (and libxkbcommon-x11-0)"
  exit 1
fi
# Inside the virtual display: start the viewer, let it run, photograph the
# screen, then stop it. Exit 124 means it was still running: the pass.
xvfb-run -a -s "-screen 0 1440x920x24" bash -c '
  "$0" --scene "$1" >"$2" 2>&1 &
  viewer=$!
  sleep "$3"
  if ! kill -0 "$viewer" 2>/dev/null; then
    wait "$viewer"
    exit $?
  fi
  if command -v import >/dev/null 2>&1; then
    import -window root "$4" 2>/dev/null
  fi
  kill "$viewer"
  wait "$viewer" 2>/dev/null
  exit 124
' "$BINARY" "$SCENE" "$LOG" "$SECONDS_TO_RUN" "$SHOT"
status=$?
if grep -q "panicked" "$LOG"; then
  grep -A3 "panicked" "$LOG" | head -20
  echo "windowed viewer panicked"
  exit 1
fi
if [ "$status" -ne 124 ]; then
  tail -20 "$LOG"
  echo "windowed viewer exited early with status $status"
  exit 1
fi
if [ -s "$SHOT" ]; then
  # The header (top 30 px, left 400 px) holds the application name, the
  # source indicator and the scene id. Drawn text gives hundreds of
  # anti-aliased shades; an empty strip gives a few dozen.
  python3 - "$SHOT" <<'PY' || exit 1
import sys
from PIL import Image

header = Image.open(sys.argv[1]).convert("RGB").crop((0, 0, 400, 30))
shades = len(header.getcolors(1 << 20) or [])
if shades < 150:
    sys.exit(f"the operator shell is not on screen: {shades} shades in the header")
print(f"operator shell on screen: {shades} shades in the header")
PY
else
  echo "no screenshot (ImageMagick's import missing): on-screen check skipped"
fi
echo "windowed viewer ran ${SECONDS_TO_RUN} s without exiting or panicking"
