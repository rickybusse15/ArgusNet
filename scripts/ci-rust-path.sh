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

# A LAST RESORT, when rustup itself is gone but its toolchains are not.
#
# Added 2026-09-15, after the 2026-09-11 failure above recurred and took the
# ENTIRE CI graph down -- not only the three Rust jobs. `packaging-contract`
# and `python-tests` build the `argusnet-native` cdylib during `pip install`,
# and `security` runs `cargo-audit`, so all six checks die at this step. Four
# pull requests merged or opened with no check ever running on them.
#
# This was deliberately NOT done twice before, on the grounds that pointing
# PATH at a toolchain directory bypasses rust-toolchain.toml and lets the
# pinned version drift silently -- which is the exact failure the pin was
# added to prevent. That objection is answered here rather than overruled:
#
#   * The directory is CHOSEN BY the pin. The channel is read out of
#     rust-toolchain.toml, never hardcoded and never guessed.
#   * The result is VERIFIED AGAINST the pin. `rustc --version` must report
#     the pinned channel or this fails, so a mismatch is louder than before,
#     not quieter.
#   * A channel the machine does not have fails outright. The fallback cannot
#     silently substitute a different toolchain, which is the whole risk.
#
# So the pin stays authoritative. What is lost is rustup's component
# management, which is why the missing components are named below: without
# them `cargo clippy` and `cargo fmt` fail, and a reader should not have to
# work out why.
#
# This is a degraded mode and says so on every run. The real repair is still
# to reinstall rustup on the machine.
if ! command -v rustup >/dev/null 2>&1 && [ -f rust-toolchain.toml ]; then
  _argusnet_channel="$(
    sed -n 's/^[[:space:]]*channel[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' \
      rust-toolchain.toml | head -1
  )"
  if [ -n "$_argusnet_channel" ]; then
    for _argusnet_bin in "$HOME/.rustup/toolchains/$_argusnet_channel"-*/bin; do
      [ -x "$_argusnet_bin/cargo" ] && [ -x "$_argusnet_bin/rustc" ] || continue
      # Confirm the binary really is the pinned channel before trusting it.
      # A directory name is a claim; `rustc --version` is the evidence.
      _argusnet_reported="$("$_argusnet_bin/rustc" --version 2>/dev/null)"
      case "$_argusnet_reported" in
        *"$_argusnet_channel"*) ;;
        *)
          echo "::error::$_argusnet_bin/rustc reports '$_argusnet_reported'," \
               "which is not the pinned channel $_argusnet_channel."
          continue
          ;;
      esac

      export PATH="$_argusnet_bin:$PATH"
      [ -n "$GITHUB_PATH" ] && echo "$_argusnet_bin" >> "$GITHUB_PATH"
      echo "::warning::rustup is missing; falling back to the pinned toolchain" \
           "at $_argusnet_bin. CI is running in a DEGRADED mode -- reinstall" \
           "rustup on this runner: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
      echo "pinned channel from rust-toolchain.toml: $_argusnet_channel"
      echo "rustc reports: $_argusnet_reported"

      # Name what rustup would have provided and this cannot. `cargo clippy`
      # and `cargo fmt` need these on PATH; without the note, a rust-lint
      # failure here looks like a lint error rather than a missing component.
      for _argusnet_component in cargo-clippy clippy-driver rustfmt cargo-fmt; do
        [ -x "$_argusnet_bin/$_argusnet_component" ] ||
          echo "::warning::component $_argusnet_component is not in" \
               "$_argusnet_bin; any step needing it will fail"
      done
      unset _argusnet_component
      break
    done
    unset _argusnet_bin _argusnet_reported
  fi
  unset _argusnet_channel
fi

# Report what will actually be used, in whichever mode we ended up in.
#
# This replaces the `rustup show` that each of the six call sites used to run
# after sourcing. That line cannot work in the fallback above, and deciding
# what to print is this script's job anyway -- it is the only thing that knows
# which mode is live.
_argusnet_report() {
  echo "cargo: $(command -v cargo 2>/dev/null) -- $(cargo --version 2>/dev/null)"
  echo "rustc: $(command -v rustc 2>/dev/null) -- $(rustc --version 2>/dev/null)"
  if command -v rustup >/dev/null 2>&1; then
    rustup show
  else
    echo "(no rustup: running on the pinned toolchain directly, see the warning above)"
  fi
}

# With the fallback engaged there is no rustup, and there does not need to be:
# every step downstream calls cargo, not rustup. A usable cargo is success, so
# the hard failure below stays reserved for a machine with no toolchain at all.
if ! command -v rustup >/dev/null 2>&1 && command -v cargo >/dev/null 2>&1; then
  _argusnet_report
  return 0 2>/dev/null || exit 0
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
  # Name the case rather than leaving it to be read off two listings. The
  # commonest one is invisible to `find -type f`: rustup fills ~/.cargo/bin
  # with symlinks that all point at the `rustup` binary beside them, and if
  # that one binary goes missing the whole farm dangles. A `-type f` search
  # skips symlinks, so the directory reads as empty and the machine looks like
  # it never had Rust -- which is exactly the wrong conclusion.
  if [ -L "$HOME/.cargo/bin/cargo" ] && [ ! -e "$HOME/.cargo/bin/rustup" ]; then
    echo "DIAGNOSIS: rustup's proxy symlinks in ~/.cargo/bin are intact but the"
    echo "rustup binary they all point to is GONE, so every one of them dangles."
    echo "The toolchains under ~/.rustup are untouched; only rustup is missing."
  elif [ -d "$HOME/.rustup/toolchains" ] && [ -n "$(ls -A "$HOME/.rustup/toolchains" 2>/dev/null)" ]; then
    echo "DIAGNOSIS: toolchains exist under ~/.rustup but rustup is not runnable."
  else
    echo "DIAGNOSIS: no Rust toolchain on this machine."
  fi
  echo "Reinstalling rustup repairs all three, and keeps rust-toolchain.toml"
  echo "authoritative -- which pointing PATH at a toolchain directory would not:"
  echo "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
  return 1 2>/dev/null || exit 1
fi

_argusnet_report
