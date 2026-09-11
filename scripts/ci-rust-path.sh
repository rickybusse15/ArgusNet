# Put the Rust toolchain on PATH. SOURCE this, do not execute it.
#
#   source scripts/ci-rust-path.sh
#
# WHY THIS EXISTS
#
# A self-hosted runner does not necessarily see the PATH you see in a
# terminal. Started as a launchd user agent (macOS `svc.sh install`) it can
# inherit a minimal PATH and never source the login shell's profile, which
# puts anything installed by rustup, Homebrew or conda out of reach. This
# searches the usual locations so a cargo step does not depend on how the
# runner happened to be launched.
#
# It is SOURCED rather than executed because a child process cannot change its
# parent's PATH: writing GITHUB_PATH alone fixes the FOLLOWING steps and
# leaves the step that needs rustup still broken.
#
# NOTE, from a real failure on 2026-09-11: a missing rustup is NOT always a
# PATH problem, and assuming it was cost two CI rounds. On that runner
# ~/.cargo/bin was already on PATH and ~/.cargo and ~/.rustup both existed --
# the rustup and cargo binaries themselves were simply gone from the machine.
# No workflow change can fix that, which is why the failure path below prints
# enough to tell the two cases apart instead of just retrying.

if ! command -v rustup >/dev/null 2>&1; then
  for _argusnet_candidate in \
    "$HOME/.cargo/bin" \
    "$CARGO_HOME/bin" \
    /opt/homebrew/bin \
    /usr/local/bin \
    "$HOME/.rustup/bin" \
    "$HOME/miniforge3/bin" \
    "$HOME/mambaforge/bin" \
    "$HOME/anaconda3/bin" \
    /opt/anaconda3/bin \
    "$HOME/miniconda3/bin"
  do
    if [ -n "$_argusnet_candidate" ] && [ -x "$_argusnet_candidate/rustup" ]; then
      export PATH="$_argusnet_candidate:$PATH"
      # For every step after this one in the same job.
      [ -n "$GITHUB_PATH" ] && echo "$_argusnet_candidate" >> "$GITHUB_PATH"
      echo "found the Rust toolchain in $_argusnet_candidate"
      break
    fi
  done
  unset _argusnet_candidate
fi

# Fail LOUDLY and with evidence. The point of the detail below is to
# distinguish "installed somewhere this script does not look" -- a workflow
# fix -- from "not installed on this machine at all", which is not.
if ! command -v rustup >/dev/null 2>&1; then
  echo "::error::rustup is not on PATH and was not found in any known location."
  echo "PATH=$PATH"
  echo "HOME=$HOME  CARGO_HOME=${CARGO_HOME:-unset}  RUSTUP_HOME=${RUSTUP_HOME:-unset}"
  echo "--- ~/.cargo/bin (where rustup keeps its proxies) ---"
  ls -la "$HOME/.cargo/bin" 2>&1 | head -20
  echo "--- ~/.rustup/toolchains (where the compilers live) ---"
  ls -la "$HOME/.rustup/toolchains" 2>&1 | head -10
  # Depth 6, not 4: ~/.rustup/toolchains/<name>/bin/cargo is five levels down
  # and a shallower search misses an intact toolchain behind missing proxies.
  echo "--- any rustup/cargo/rustc executable under \$HOME (depth 6) ---"
  find "$HOME" -maxdepth 6 \( -name rustup -o -name cargo -o -name rustc \) \
    -type f -perm -u+x 2>/dev/null | head -20
  echo
  echo "If the two listings above are empty, the toolchain is not installed on"
  echo "this machine and no workflow change can help. Reinstall it on the"
  echo "runner:  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
  return 1 2>/dev/null || exit 1
fi
