#!/usr/bin/env bash
# sign_local.sh — sign OpenBird.app + its helpers with a STABLE local identity.
#
# Why this exists:
#   macOS ties TCC grants (Accessibility / Screen Recording / Microphone) to a
#   binary's code-signing identity. An unsigned or ad-hoc-signed bundle gets a
#   fresh identity on every build, so any permission the user grants evaporates
#   on the next launch. OpenBird is not distributed through the Mac App Store and
#   (by user choice) does not use a paid Apple Developer ID, so this script mints
#   a *self-signed* code-signing certificate ONCE, stores it in a dedicated
#   keychain, and reuses it on every build. That gives the bundle a stable
#   designated requirement, so granted permissions persist across reinstalls.
#
# Fail-soft contract:
#   Signing must NEVER break `brew install`. If the self-signed identity cannot
#   be created or used (locked-down CI, missing tools, etc.) this falls back to
#   ad-hoc signing (codesign -s -) so the app still installs and launches — only
#   the *persistence* of TCC grants is lost in that degraded mode.
#
# Usage:
#   script/sign_local.sh /path/to/OpenBird.app
#
# Env overrides (all optional):
#   OPENBIRD_SIGN_IDENTITY     cert common name      (default: "OpenBird Local Signing")
#   OPENBIRD_SIGN_KEYCHAIN     keychain file name    (default: openbird-codesign.keychain-db)
#   OPENBIRD_SIGN_KEYCHAIN_PW  keychain passphrase   (default: openbird-local)
#   OPENBIRD_SIGN_DISABLE=1    skip self-signing; ad-hoc only

set -euo pipefail

APP_BUNDLE="${1:-}"
if [[ -z "$APP_BUNDLE" || ! -d "$APP_BUNDLE" ]]; then
  echo "sign_local.sh: expected a path to OpenBird.app" >&2
  exit 2
fi

IDENTITY_CN="${OPENBIRD_SIGN_IDENTITY:-OpenBird Local Signing}"
KEYCHAIN_NAME="${OPENBIRD_SIGN_KEYCHAIN:-openbird-codesign.keychain-db}"
KEYCHAIN_PW="${OPENBIRD_SIGN_KEYCHAIN_PW:-openbird-local}"
KEYCHAIN_PATH="$HOME/Library/Keychains/$KEYCHAIN_NAME"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_CONTENTS="$APP_BUNDLE/Contents"
APP_MACOS="$APP_CONTENTS/MacOS"
APP_ENTITLEMENTS="$ROOT_DIR/mac-app/OpenBird.entitlements"
HELPER_ENTITLEMENTS="$ROOT_DIR/mac-app/Helper.entitlements"

log() { echo "sign_local: $*" >&2; }

adhoc_sign() {
  log "falling back to ad-hoc signing (TCC grants will NOT persist across rebuilds)"
  # Sign helpers first (inside-out), then the app.
  for helper in capture-helper audio-helper; do
    [[ -f "$APP_MACOS/$helper" ]] && codesign --force --sign - --timestamp=none \
      "$APP_MACOS/$helper" >/dev/null 2>&1 || true
  done
  codesign --force --deep --sign - --timestamp=none "$APP_BUNDLE" >/dev/null 2>&1 || true
  log "ad-hoc signing complete"
}

if [[ "${OPENBIRD_SIGN_DISABLE:-0}" == "1" ]]; then
  adhoc_sign
  exit 0
fi

# --------------------------------------------------------------------------- #
# 1. Ensure a dedicated keychain holding a stable self-signed codesign cert.   #
# --------------------------------------------------------------------------- #

# Echo the SHA-1 hash of the (first) signing identity matching IDENTITY_CN in our
# keychain, or nothing. NOTE: we deliberately do NOT pass `-v` — that filters to
# trust-valid identities, and an untrusted self-signed cert (which is exactly
# what we mint) would be hidden, causing a fresh duplicate every run.
identity_hash() {
  security find-identity -p codesigning "$KEYCHAIN_PATH" 2>/dev/null \
    | grep "$IDENTITY_CN" \
    | grep -oE '[0-9A-F]{40}' \
    | head -n1
}

ensure_identity() {
  # Already present? Reuse it (this is what makes the identity STABLE).
  [[ -n "$(identity_hash)" ]] && return 0

  log "minting self-signed code-signing identity '$IDENTITY_CN'"

  # Create the dedicated keychain if missing. A separate keychain with a known
  # passphrase avoids prompting for the login-keychain password during install.
  if [[ ! -f "$KEYCHAIN_PATH" ]]; then
    security create-keychain -p "$KEYCHAIN_PW" "$KEYCHAIN_PATH"
  fi
  security set-keychain-settings "$KEYCHAIN_PATH"  # no auto-lock timeout
  security unlock-keychain -p "$KEYCHAIN_PW" "$KEYCHAIN_PATH"

  local tmp
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' RETURN

  # Self-signed cert with a codeSigning EKU so it shows up under -p codesigning.
  openssl req -x509 -newkey rsa:2048 -nodes -days 3650 \
    -keyout "$tmp/key.pem" -out "$tmp/cert.pem" \
    -subj "/CN=$IDENTITY_CN" \
    -addext "basicConstraints=critical,CA:false" \
    -addext "keyUsage=critical,digitalSignature" \
    -addext "extendedKeyUsage=critical,codeSigning" >/dev/null 2>&1

  openssl pkcs12 -export -out "$tmp/bundle.p12" \
    -inkey "$tmp/key.pem" -in "$tmp/cert.pem" \
    -passout "pass:$KEYCHAIN_PW" >/dev/null 2>&1

  # Import key+cert and authorize codesign to use the key non-interactively.
  security import "$tmp/bundle.p12" -k "$KEYCHAIN_PATH" -P "$KEYCHAIN_PW" \
    -T /usr/bin/codesign -T /usr/bin/security >/dev/null 2>&1
  security set-key-partition-list -S apple-tool:,apple:,codesign: \
    -s -k "$KEYCHAIN_PW" "$KEYCHAIN_PATH" >/dev/null 2>&1

  # Add the keychain to the user search list so codesign can find the identity.
  local existing
  existing="$(security list-keychains -d user | sed -e 's/[[:space:]]*"//' -e 's/"[[:space:]]*$//')"
  if ! printf '%s\n' "$existing" | grep -qF "$KEYCHAIN_PATH"; then
    # shellcheck disable=SC2086
    security list-keychains -d user -s "$KEYCHAIN_PATH" $existing
  fi
}

sign_with_identity() {
  # Resolve the identity by SHA-1 hash so signing is unambiguous even if the
  # keychain somehow holds more than one cert with the same common name.
  local hash entitlements
  hash="$(identity_hash)"
  [[ -n "$hash" ]] || return 1

  # Helpers (inside-out first) with the mic/audio-input entitlement.
  for helper in capture-helper audio-helper; do
    [[ -f "$APP_MACOS/$helper" ]] || continue
    entitlements="$HELPER_ENTITLEMENTS"
    [[ -f "$entitlements" ]] || entitlements=""
    codesign --force --options runtime --timestamp=none \
      ${entitlements:+--entitlements "$entitlements"} \
      --keychain "$KEYCHAIN_PATH" \
      --sign "$hash" "$APP_MACOS/$helper"
  done

  # Any other nested executables/scripts in MacOS/ (e.g. the openbird-cli wrapper
  # shell script) must also be sealed before the app, or codesign on the bundle
  # fails with "code object is not signed at all" for the subcomponent.
  for nested in "$APP_MACOS"/*; do
    [[ -f "$nested" ]] || continue
    case "$(basename "$nested")" in
      OpenBird|capture-helper|audio-helper) continue ;;  # main binary + helpers handled
    esac
    codesign --force --timestamp=none \
      --keychain "$KEYCHAIN_PATH" \
      --sign "$hash" "$nested"
  done

  entitlements="$APP_ENTITLEMENTS"
  [[ -f "$entitlements" ]] || entitlements=""
  codesign --force --options runtime --timestamp=none \
    ${entitlements:+--entitlements "$entitlements"} \
    --keychain "$KEYCHAIN_PATH" \
    --sign "$hash" "$APP_BUNDLE"
}

# Try the stable self-signed path; fall back to ad-hoc on any failure so the
# install never breaks.
if ensure_identity && sign_with_identity; then
  log "signed with stable identity '$IDENTITY_CN'"
  codesign --verify --verbose=2 "$APP_BUNDLE" >&2 2>&1 || \
    log "warning: codesign --verify reported issues (continuing)"
else
  adhoc_sign
fi
