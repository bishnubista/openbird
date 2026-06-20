#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-run}"
APP_NAME="OpenBird"
BUNDLE_ID="ai.openbird.OpenBird"
MIN_SYSTEM_VERSION="13.0"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="$ROOT_DIR/dist"
APP_BUNDLE="$DIST_DIR/$APP_NAME.app"
APP_CONTENTS="$APP_BUNDLE/Contents"
APP_MACOS="$APP_CONTENTS/MacOS"
APP_RESOURCES="$APP_CONTENTS/Resources"
APP_BINARY="$APP_MACOS/$APP_NAME"
INFO_PLIST="$APP_CONTENTS/Info.plist"

usage() {
  echo "usage: $0 [run|--verify|--debug|--logs|--telemetry|--no-launch]" >&2
}

build_swiftpm_product() {
  local package_dir="$1"
  local product="$2"
  (cd "$package_dir" && swift build ${swift_build_args[@]+"${swift_build_args[@]}"} >&2)
  local bin_path
  bin_path="$(cd "$package_dir" && swift build ${swift_build_args[@]+"${swift_build_args[@]}"} --show-bin-path)"
  printf '%s/%s\n' "$bin_path" "$product"
}

pkill -x "$APP_NAME" >/dev/null 2>&1 || true

swift_build_args=()
if [[ "${OPENBIRD_SWIFTPM_DISABLE_SANDBOX:-}" == "1" ]]; then
  swift_build_args+=(--disable-sandbox)
fi

(cd "$ROOT_DIR/mac-app" && swift build ${swift_build_args[@]+"${swift_build_args[@]}"} >&2)
app_build_dir="$(cd "$ROOT_DIR/mac-app" && swift build ${swift_build_args[@]+"${swift_build_args[@]}"} --show-bin-path)"
capture_helper="$(build_swiftpm_product "$ROOT_DIR/capture-helper" CaptureHelper)"
audio_helper="$(build_swiftpm_product "$ROOT_DIR/audio-helper" AudioHelper)"
uv_bin="$(command -v uv || true)"
if [[ -z "$uv_bin" ]]; then
  for candidate in /opt/homebrew/bin/uv /usr/local/bin/uv; do
    if [[ -x "$candidate" ]]; then
      uv_bin="$candidate"
      break
    fi
  done
fi
uv_bin="${uv_bin:-uv}"
uv_bin_escaped="${uv_bin//\'/\'\\\'\'}"

rm -rf "$APP_BUNDLE"
mkdir -p "$APP_MACOS" "$APP_RESOURCES"
cp "$app_build_dir/$APP_NAME" "$APP_BINARY"
cp "$capture_helper" "$APP_MACOS/capture-helper"
cp "$audio_helper" "$APP_MACOS/audio-helper"
chmod +x "$APP_BINARY" "$APP_MACOS/capture-helper" "$APP_MACOS/audio-helper"

cat >"$APP_MACOS/openbird-cli" <<WRAPPER
#!/usr/bin/env bash
set -euo pipefail
UV_BIN='$uv_bin_escaped'
BIN_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="\$(cd "\$BIN_DIR/../../../.." && pwd)"
if [[ ! -f "\$REPO_ROOT/pyproject.toml" ]]; then
  echo "OpenBird dev bundle cannot find the source checkout at \$REPO_ROOT" >&2
  exit 127
fi
# Point the daemon at THIS bundle's signed helper (matches OPENBIRD_CAPTURE_HELPER
# contract in openbird/capture/daemon.py). Without this the daemon would look under
# /Applications/OpenBird.app and fail-closed / probe the wrong path for a dist build.
export OPENBIRD_CAPTURE_HELPER="\$BIN_DIR/capture-helper"
exec "\$UV_BIN" --directory "\$REPO_ROOT" run openbird "\$@"
WRAPPER
chmod +x "$APP_MACOS/openbird-cli"

cat >"$INFO_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleExecutable</key>
  <string>$APP_NAME</string>
  <key>CFBundleIdentifier</key>
  <string>$BUNDLE_ID</string>
  <key>CFBundleName</key>
  <string>$APP_NAME</string>
  <key>CFBundleDisplayName</key>
  <string>$APP_NAME</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>LSMinimumSystemVersion</key>
  <string>$MIN_SYSTEM_VERSION</string>
  <key>NSPrincipalClass</key>
  <string>NSApplication</string>
</dict>
</plist>
PLIST

# ---- Code signing (Tier 1: local signed dev bundle) ----
# Sign inside-out (helpers first, then the .app) so the app signature seals all
# nested content — the helpers, the openbird-cli shim, and Info.plist. Anything
# edited after this point breaks the seal, so signing is the LAST build step.
#
# Identity comes from OPENBIRD_SIGN_IDENTITY. Default is ad-hoc ("-"), which is
# fine for CI/test builds that never request macOS TCC. For the responsible-process
# probe and any real capture run, export a stable Developer ID, e.g.:
#   export OPENBIRD_SIGN_IDENTITY="Developer ID Application: bishnu bista (SB26BAMXJM)"
# Only a stable signing identity makes TCC grants persist across rebuilds.
SIGN_IDENTITY="${OPENBIRD_SIGN_IDENTITY:--}"

# A secure Apple timestamp is required for Developer ID *distribution* (notarization),
# but not for local TCC testing — and depending on Apple's TS server would make local
# builds fail offline. Default off; opt in with OPENBIRD_SIGN_TIMESTAMP=1.
if [[ "${OPENBIRD_SIGN_TIMESTAMP:-0}" == "1" ]]; then
  ts_flag=(--timestamp)
else
  ts_flag=(--timestamp=none)
fi

sign_macho() {
  local path="$1"; shift
  codesign --force --options runtime "${ts_flag[@]}" --sign "$SIGN_IDENTITY" "$@" "$path" >&2
}

echo "Signing bundle (identity: $SIGN_IDENTITY)" >&2
# Bare helper executables have no Info.plist, so give each an explicit identifier.
sign_macho "$APP_MACOS/capture-helper" --identifier dev.openbird.capture-helper
sign_macho "$APP_MACOS/audio-helper"   --identifier dev.openbird.audio-helper
# The openbird-cli shim is a shell script, not a Mach-O, but it lives in
# Contents/MacOS so `codesign --verify --strict` requires it to carry its own
# signature (stored in an extended attribute) before the app seal. Sign it WITHOUT
# --options runtime, which is a Mach-O-only concept.
codesign --force "${ts_flag[@]}" --identifier dev.openbird.cli \
  --sign "$SIGN_IDENTITY" "$APP_MACOS/openbird-cli" >&2
# App last: signs the main executable and seals the whole bundle.
sign_macho "$APP_BUNDLE"

echo "Verifying signature..." >&2
codesign --verify --strict --verbose=2 "$APP_BUNDLE" >&2
codesign -dvv "$APP_BUNDLE" 2>&1 | sed 's/^/  /' >&2
if [[ "$SIGN_IDENTITY" != "-" ]]; then
  signed_team="$(codesign -dvv "$APP_BUNDLE" 2>&1 | sed -n 's/^TeamIdentifier=//p')"
  expected_team="${OPENBIRD_EXPECTED_TEAM_ID:-}"
  if [[ -n "$expected_team" && "$signed_team" != "$expected_team" ]]; then
    echo "  WARNING: TeamIdentifier '$signed_team' != expected '$expected_team'" >&2
  else
    echo "  TeamIdentifier: ${signed_team:-<none>}" >&2
  fi
fi
# Gatekeeper assessment is informational in Tier 1 — the dev bundle is intentionally
# NOT notarized yet, so a rejection here is expected and must not fail the build.
spctl -a -vv "$APP_BUNDLE" 2>&1 | sed 's/^/  /' >&2 || \
  echo "  spctl: not accepted (expected — Tier 1 dev bundle is not notarized)" >&2

open_app() {
  /usr/bin/open -n "$APP_BUNDLE"
}

case "$MODE" in
  run)
    open_app
    ;;
  --verify|verify)
    open_app
    for _ in {1..40}; do
      if pgrep -x "$APP_NAME" >/dev/null; then
        exit 0
      fi
      sleep 0.25
    done
    echo "$APP_NAME did not appear after launch" >&2
    exit 1
    ;;
  --debug|debug)
    lldb -- "$APP_BINARY"
    ;;
  --logs|logs)
    open_app
    /usr/bin/log stream --info --style compact --predicate "process == \"$APP_NAME\""
    ;;
  --telemetry|telemetry)
    open_app
    /usr/bin/log stream --info --style compact --predicate "subsystem == \"$BUNDLE_ID\""
    ;;
  --no-launch|no-launch)
    echo "$APP_BUNDLE"
    ;;
  *)
    usage
    exit 2
    ;;
esac
