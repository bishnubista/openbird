#!/usr/bin/env bash
# sign_local.sh — ad-hoc sign OpenBird.app + its helpers with STABLE identifiers.
#
# Why ad-hoc (not a self-signed cert):
#   macOS TCC (Accessibility / Screen Recording / Microphone) tracks an app by
#   its code signature. For an ad-hoc signature the designated requirement is the
#   binary's *cdhash* plus a stable *identifier* — so once the app is installed
#   and its bytes stop changing, granted permissions PERSIST across launches. We
#   set an explicit identifier per binary so the requirement is stable and not
#   derived from a build path.
#
#   A self-signed certificate would additionally survive rebuilds, but it
#   requires a password-protected keychain that (a) is invisible during
#   `brew install` because Homebrew redirects $HOME, and (b) triggers GUI
#   keychain-access prompts. Ad-hoc needs no keychain, so it works cleanly inside
#   brew's sandbox with zero prompts. The only trade-off: a `brew upgrade` that
#   rebuilds the app changes the cdhash, so you re-grant permissions once after
#   an upgrade.
#
# Inside-out: helpers and any nested executables are signed before the app, or
# codesign on the bundle fails with "code object is not signed at all".
#
# Usage:
#   script/sign_local.sh /path/to/OpenBird.app

set -euo pipefail

APP_BUNDLE="${1:-}"
if [[ -z "$APP_BUNDLE" || ! -d "$APP_BUNDLE" ]]; then
  echo "sign_local.sh: expected a path to OpenBird.app" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_MACOS="$APP_BUNDLE/Contents/MacOS"
APP_ENTITLEMENTS="$ROOT_DIR/mac-app/OpenBird.entitlements"
HELPER_ENTITLEMENTS="$ROOT_DIR/mac-app/Helper.entitlements"

log() { echo "sign_local: $*" >&2; }

if ! command -v codesign >/dev/null 2>&1; then
  log "codesign not found; leaving the bundle unsigned"
  exit 0
fi

# Ad-hoc sign one path with a stable identifier (+ optional entitlements).
# Use -e (not -f): the app bundle is a directory, helpers/wrappers are files.
sign() {
  local path="$1" identifier="$2" entitlements="${3:-}"
  [[ -e "$path" ]] || return 0
  codesign --force --options runtime --timestamp=none \
    --identifier "$identifier" \
    ${entitlements:+--entitlements "$entitlements"} \
    --sign - "$path"
}

ent_helper="$HELPER_ENTITLEMENTS"; [[ -f "$ent_helper" ]] || ent_helper=""
ent_app="$APP_ENTITLEMENTS";       [[ -f "$ent_app" ]] || ent_app=""

# Helpers first (inside-out). The audio-helper carries the mic entitlement.
sign "$APP_MACOS/capture-helper" "ai.openbird.OpenBird.capture-helper" "$ent_helper"
sign "$APP_MACOS/audio-helper"   "ai.openbird.OpenBird.audio-helper"   "$ent_helper"

# Any other nested executables/scripts in MacOS/ (e.g. the openbird-cli wrapper)
# must be sealed before the app.
for nested in "$APP_MACOS"/*; do
  [[ -f "$nested" ]] || continue
  case "$(basename "$nested")" in
    OpenBird|capture-helper|audio-helper) continue ;;
  esac
  codesign --force --timestamp=none \
    --identifier "ai.openbird.OpenBird.$(basename "$nested")" \
    --sign - "$nested"
done

# Finally the app bundle.
sign "$APP_BUNDLE" "ai.openbird.OpenBird" "$ent_app"

log "ad-hoc signed with stable identifiers"
if codesign --verify --deep --strict "$APP_BUNDLE" >/dev/null 2>&1; then
  log "signature verified"
else
  log "warning: codesign --verify reported issues (continuing)"
fi
