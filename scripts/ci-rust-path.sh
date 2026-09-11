# Put the Rust toolchain on PATH. SOURCE this, do not execute it.
#
#   source scripts/ci-rust-path.sh
#
# WHY THIS EXISTS
#
# A self-hosted runner started as a launchd user agent -- which is what macOS
# `svc.sh install` sets up, and what LOCAL_CI.md recommends, because a user
# agent has the window-server session Metal needs for the headless render --
# inherits launchd's minimal PATH (/usr/bin:/bin:/usr/sbin:/sbin) and never
# sources the login shell's profile. Anything installed by rustup, Homebrew or
# conda lives outside that PATH, so every cargo step dies with
#
#     rustup: command not found
#
# while rustup works perfectly in a terminal on the same machine. Started from
# a terminal the runner inherited the shell's PATH and the problem does not
# appear at all, which is what makes it confusing.
#
# Hosted runners already have the toolchain on PATH, so the search below finds
# it on the first `command -v` and does nothing.
#
# It is sourced rather than executed because a child process cannot change its
# parent's PATH: writing GITHUB_PATH alone would fix the FOLLOWING steps and
# leave the step that needs rustup still broken.

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

# Fail LOUDLY and with evidence rather than leaving the next command to say
# "command not found" and nothing else. The first attempt at this fix assumed
# ~/.cargo/bin and was wrong; a second wrong guess should cost a reading of
# this output, not another CI round.
if ! command -v rustup >/dev/null 2>&1; then
  echo "::error::rustup is not on PATH and was not found in any known location."
  echo "PATH=$PATH"
  echo "HOME=$HOME  CARGO_HOME=${CARGO_HOME:-unset}  RUSTUP_HOME=${RUSTUP_HOME:-unset}"
  echo "--- executables named rustup or cargo under \$HOME (depth 4) ---"
  find "$HOME" -maxdepth 4 \( -name rustup -o -name cargo \) -type f 2>/dev/null | head -20
  echo "--- common bin directories ---"
  for _d in "$HOME/.cargo" "$HOME/.rustup" /opt/homebrew/bin /usr/local/bin; do
    [ -e "$_d" ] && echo "exists: $_d" || echo "absent: $_d"
  done
  unset _d
  return 1 2>/dev/null || exit 1
fi
