#!/usr/bin/env bash
# Dev Launch-Services hygiene: unregister stale dev/build OpenBird.app bundles so
# they stop shadowing the real install by bundle id. Run after a dev build/test
# cycle. See docs/design/cleanup-tooling.md.
#
# SAFE BY DESIGN: only unregisters
#   - ghost registrations (the bundle no longer exists on disk), and
#   - dev/build bundles under */dist/OpenBird.app or */dist/dmg-stage/OpenBird.app
#     whose CFBundleIdentifier is exactly ai.openbird.OpenBird.
# /Applications/OpenBird.app is kept by default and unregistered ONLY with --all
# (after CFBundleIdentifier validation). The Homebrew prefix (Cellar/opt/libexec),
# /opt/homebrew, and /usr/local are NEVER touched — even with --all.
set -euo pipefail

APP_NAME="OpenBird"
BUNDLE_ID="ai.openbird.OpenBird"
LSREGISTER="/System/Library/Frameworks/CoreServices.framework/Versions/A/Frameworks/LaunchServices.framework/Versions/A/Support/lsregister"

INCLUDE_APPLICATIONS=0  # --all: also unregister a validated /Applications copy
REMOVE_DIST=0           # --dist: also rm -rf dist/OpenBird.app build artifacts

usage() {
  echo "usage: $0 [--all] [--dist]" >&2
  echo "  --all   also unregister a validated /Applications/$APP_NAME.app" >&2
  echo "  --dist  also rm -rf dist/$APP_NAME.app build artifacts under the repo" >&2
}

for arg in "$@"; do
  case "$arg" in
    --all) INCLUDE_APPLICATIONS=1 ;;
    --dist) REMOVE_DIST=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $arg" >&2; usage; exit 2 ;;
  esac
done

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "dev_cleanup: not macOS — nothing to do." >&2
  exit 0
fi
if [[ ! -x "$LSREGISTER" ]]; then
  echo "dev_cleanup: lsregister not found — nothing to do." >&2
  exit 0
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Homebrew prefix (so we never unregister a brew-managed libexec/OpenBird.app).
BREW_PREFIX=""
if command -v brew >/dev/null 2>&1; then
  BREW_PREFIX="$(brew --prefix 2>/dev/null || true)"
fi

# Best-effort quit a running instance so we aren't unregistering a live app.
osascript -e "quit app \"$APP_NAME\"" >/dev/null 2>&1 || true

# Is this path a protected real-install root we must never unregister?
# Returns 0 (protected) for /Applications (unless --all), any Homebrew prefix, and
# any Cellar/opt/libexec path — even as a ghost (Codex constraint).
is_protected() {
  local p="$1"
  case "$p" in
    /Applications/*) [[ "$INCLUDE_APPLICATIONS" -eq 1 ]] && return 1 || return 0 ;;
    /opt/homebrew/*|/usr/local/*) return 0 ;;
    */Cellar/*|*/libexec/*) return 0 ;;
  esac
  if [[ -n "$BREW_PREFIX" && "$p" == "$BREW_PREFIX"/* ]]; then
    return 0
  fi
  return 1
}

# Does this path look like a dev/build artifact we may unregister?
is_dev_build() {
  case "$1" in
    */dist/"$APP_NAME".app|*/dist/dmg-stage/"$APP_NAME".app) return 0 ;;
    *) return 1 ;;
  esac
}

bundle_id_of() {
  /usr/libexec/PlistBuddy -c "Print :CFBundleIdentifier" \
    "$1/Contents/Info.plist" 2>/dev/null || true
}

count_registrations() {
  "$LSREGISTER" -dump 2>/dev/null | grep -c "$APP_NAME.app" || true
}

before="$(count_registrations)"
echo "dev_cleanup: $before $APP_NAME.app registration line(s) before."

# Unregister one path, tracking success/failure. Increments `removed` only on a
# real success so a failed `lsregister -u` cannot masquerade as cleaned.
removed=0
failures=0
unregister() {
  local why="$1" path="$2"
  echo "  unregister ($why): $path"
  if "$LSREGISTER" -u "$path" >/dev/null 2>&1; then
    removed=$((removed + 1))
  else
    echo "    ! lsregister -u failed for: $path" >&2
    failures=$((failures + 1))
  fi
}

# Bash 3.2 (macOS stock) has no `mapfile`; read the dump line-by-line. Running the
# while-loop via process substitution keeps counters in the current shell.
while IFS= read -r path; do
  [[ -n "$path" ]] || continue
  if is_protected "$path"; then
    echo "  keep (protected): $path"
    continue
  fi
  if [[ ! -e "$path" ]]; then
    unregister "ghost" "$path"
    continue
  fi
  # Existing bundle: validate the bundle id BEFORE deciding eligibility, so an
  # unrelated OpenBird.app is never unregistered (Codex #5).
  bid="$(bundle_id_of "$path")"
  if [[ "$bid" != "$BUNDLE_ID" ]]; then
    echo "  keep (id=$bid): $path"
    continue
  fi
  if is_dev_build "$path"; then
    unregister "dev build" "$path"
  elif [[ "$INCLUDE_APPLICATIONS" -eq 1 && "$path" == "/Applications/$APP_NAME.app" ]]; then
    unregister "--all /Applications" "$path"
  else
    echo "  keep (not a dev/build path): $path"
  fi
done < <(
  "$LSREGISTER" -dump 2>/dev/null \
    | sed -n 's/^[[:space:]]*path:[[:space:]]*\(.*'"$APP_NAME"'\.app\)\( (0x[0-9a-f]*)\)\{0,1\}$/\1/p' \
    | sort -u
)

if [[ "$REMOVE_DIST" -eq 1 ]]; then
  for d in "$ROOT_DIR/dist/$APP_NAME.app" "$ROOT_DIR/dist/dmg-stage/$APP_NAME.app"; do
    if [[ -e "$d" ]]; then
      echo "  rm -rf $d"
      rm -rf "$d"
    fi
  done
fi

after="$(count_registrations)"
echo "dev_cleanup: done — unregistered $removed; $after registration line(s) remain."
if [[ "$failures" -gt 0 ]]; then
  echo "dev_cleanup: $failures unregister(s) FAILED — stale registrations remain." >&2
  exit 1
fi
