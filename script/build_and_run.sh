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

(cd "$ROOT_DIR/mac-app" && swift build ${swift_build_args[@]+"${swift_build_args[@]}"} )
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
