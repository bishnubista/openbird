#!/usr/bin/env bash
#
# e2e_screenshots.sh — capture OpenBird's UI surfaces into e2e/screenshots/ for
# visual/E2E verification against the Liquid Glass design references.
#
# Why this exists: from a headless/CLI session you cannot click menu-bar surfaces
# by hand. This harness drives the running app via the Accessibility (AX) API and
# captures each window by its exact screen region (so the terminal never appears in
# the shot). It captures what is reliably automatable and reports — honestly — what
# is not, instead of silently producing a blank frame.
#
# Prerequisites (one-time, per machine):
#   1. The terminal/automation host (e.g. Ghostty, Terminal, iTerm) must have
#      Accessibility permission: System Settings ▸ Privacy & Security ▸ Accessibility.
#      Verify with:  osascript -e 'tell application "System Events" to count menu bars of (first application process whose frontmost is true)'
#      (returns >= 1 when granted; 0 when blocked).
#   2. Install or build the target app bundle first. By default the harness tests
#      dist/OpenBird.app. Use --app to force an installed bundle such as
#      /Applications/OpenBird.app. On an unsigned dev build, approve the one-time
#      Keychain ACL prompt with "Always Allow" (NOT Escape — Escape denies it and
#      dismisses sheets).
#   3. Install cliclick (third-party) for the menu-dropdown coordinate click:
#        brew install cliclick
#      Only the menu-dropdown capture needs it; the other surfaces don't.
#
# Usage:
#   ./script/e2e_screenshots.sh
#   ./script/e2e_screenshots.sh --app /Applications/OpenBird.app
#   OPENBIRD_E2E_APP_PATH=/Applications/OpenBird.app ./script/e2e_screenshots.sh
#
# Known limitation: the menu-bar dropdown is anchored to OpenBird's status item.
# If your menu bar is full (notch overflow), macOS parks the hidden item off-screen
# (AX position like -1/976) and the popover can't be shown/captured. Free menu-bar
# space (⌘-drag icons, or quit a few menu-bar apps) to capture it.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/e2e/screenshots"
APP_NAME="OpenBird"
BUNDLE_ID="ai.openbird.OpenBird"
LSREGISTER="/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister"
mkdir -p "$OUT"

log() { printf '  • %s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

usage() {
  cat >&2 <<EOF
usage: $0 [--app /path/to/OpenBird.app]

Options:
  --app PATH   Capture the UI from this exact app bundle.
               Defaults to OPENBIRD_E2E_APP_PATH, then dist/OpenBird.app.
EOF
}

APP_BUNDLE_INPUT="${OPENBIRD_E2E_APP_PATH:-}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --app)
      [[ $# -ge 2 ]] || die "--app requires a path"
      APP_BUNDLE_INPUT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

if [[ -z "$APP_BUNDLE_INPUT" ]]; then
  APP_BUNDLE_INPUT="$ROOT/dist/$APP_NAME.app"
fi

canonical_dir() {
  local path="$1"
  [[ -d "$path" ]] || return 1
  (cd "$path" && pwd -P)
}

APP_BUNDLE="$(canonical_dir "$APP_BUNDLE_INPUT")" || die "app bundle not found: $APP_BUNDLE_INPUT"
APP_EXECUTABLE="$APP_BUNDLE/Contents/MacOS/$APP_NAME"
INFO_PLIST="$APP_BUNDLE/Contents/Info.plist"
[[ -x "$APP_EXECUTABLE" ]] || die "expected executable at $APP_EXECUTABLE"
[[ -f "$INFO_PLIST" ]] || die "expected Info.plist at $INFO_PLIST"
actual_bundle_id="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleIdentifier' "$INFO_PLIST" 2>/dev/null || true)"
[[ "$actual_bundle_id" == "$BUNDLE_ID" ]] || die "expected bundle id $BUNDLE_ID, got '${actual_bundle_id:-missing}'"

if [[ "$APP_BUNDLE" == "/Applications/$APP_NAME.app" ]]; then
  log "WARNING: targeting installed app at $APP_BUNDLE"
  log "         This run force-relaunches OpenBird and temporarily toggles onboarding."
fi
if [[ "$APP_BUNDLE" == /Applications/* ]] \
  && grep -q "OpenBird dev bundle cannot find the source checkout" "$APP_BUNDLE/Contents/MacOS/openbird-cli" 2>/dev/null; then
  log "WARNING: selected app contains the non-relocatable dev openbird-cli shim."
  log "         UI screenshots may render, but CLI-backed setup/timeline features can fail."
  log "         Use Homebrew's openbird-app or a notarized DMG for release evidence."
fi

ONBOARDING_KEY="openbird.onboarding.completed"
ONBOARDING_WAS_SET=0
ONBOARDING_VALUE=""
if ONBOARDING_VALUE="$(defaults read "$BUNDLE_ID" "$ONBOARDING_KEY" 2>/dev/null)"; then
  ONBOARDING_WAS_SET=1
fi

kill_app_instances() {
  pkill -f "$APP_NAME.app/Contents/MacOS/$APP_NAME" >/dev/null 2>&1 || true
}

restore_onboarding_pref() {
  kill_app_instances
  local _
  for _ in $(seq 1 50); do
    pgrep -f "$APP_NAME.app/Contents/MacOS/$APP_NAME" >/dev/null || break
    /bin/sleep 0.1
  done
  if [[ "$ONBOARDING_WAS_SET" -eq 1 ]]; then
    case "$ONBOARDING_VALUE" in
      1) defaults write "$BUNDLE_ID" "$ONBOARDING_KEY" -bool true >/dev/null 2>&1 || true ;;
      0) defaults write "$BUNDLE_ID" "$ONBOARDING_KEY" -bool false >/dev/null 2>&1 || true ;;
      *) defaults write "$BUNDLE_ID" "$ONBOARDING_KEY" "$ONBOARDING_VALUE" >/dev/null 2>&1 || true ;;
    esac
  else
    defaults delete "$BUNDLE_ID" "$ONBOARDING_KEY" >/dev/null 2>&1 || true
  fi
}
trap restore_onboarding_pref EXIT

canonical_file_path() {
  local path="$1"
  if [[ -e "$path" ]]; then
    local dir base
    dir="$(cd "$(dirname "$path")" && pwd -P)" || return 1
    base="$(basename "$path")"
    printf '%s/%s\n' "$dir" "$base"
  else
    printf '%s\n' "$path"
  fi
}

process_has_executable_path() {
  local pid="$1" expected="$2" path canonical
  while IFS= read -r path; do
    [[ -n "$path" ]] || continue
    canonical="$(canonical_file_path "$path")" || continue
    [[ "$canonical" == "$expected" ]] && return 0
  done < <(lsof -a -p "$pid" -d txt -Fn 2>/dev/null | sed -n 's/^n//p' || true)
  return 1
}

assert_selected_app_running() {
  local pids pid count
  pids="$(pgrep -f "$APP_NAME.app/Contents/MacOS/$APP_NAME" || true)"
  count="$(printf '%s\n' "$pids" | grep -c . || true)"
  [[ "$count" -eq 1 ]] || die "expected exactly one $APP_NAME GUI process, found $count (${pids//$'\n'/ })"
  pid="$pids"
  process_has_executable_path "$pid" "$APP_EXECUTABLE" \
    || die "running $APP_NAME process $pid does not expose expected executable $APP_EXECUTABLE"
}

prepare_launch_services() {
  if [[ -x "$ROOT/script/dev_cleanup.sh" ]]; then
    "$ROOT/script/dev_cleanup.sh" >/dev/null 2>&1 || log "WARN: dev_cleanup.sh did not fully clean LaunchServices registrations."
  fi
  if [[ -x "$LSREGISTER" ]]; then
    "$LSREGISTER" -f "$APP_BUNDLE" >/dev/null 2>&1 || log "WARN: could not register $APP_BUNDLE with LaunchServices."
  fi
}

# Quit the app, wait for it to fully exit, then relaunch (retrying open, since
# LaunchServices returns -600 if you re-open while the prior instance is quitting).
relaunch() {
  kill_app_instances
  local _
  for _ in $(seq 1 50); do
    pgrep -f "$APP_NAME.app/Contents/MacOS/$APP_NAME" >/dev/null || break
    /bin/sleep 0.1
  done
  # --enable-e2e-deeplinks arms the openbird:// router, which is inert in normal use
  # (so a stray URL can't force-show private data). `open --args` only delivers args
  # on a cold launch — which this is, since we just killed the prior instance.
  for _ in 1 2 3; do
    if open "$APP_BUNDLE" --args --enable-e2e-deeplinks 2>/dev/null; then break; fi
    /bin/sleep 0.5
  done
  for _ in $(seq 1 50); do
    if pgrep -f "$APP_NAME.app/Contents/MacOS/$APP_NAME" >/dev/null; then
      /bin/sleep 1
      assert_selected_app_running
      return 0
    fi
    /bin/sleep 0.2
  done
  die "$APP_NAME did not launch from $APP_BUNDLE"
}

# Is AX usable by this host process? Retry a few times: the frontmost app can expose
# zero readable menu bars for a moment mid-launch/transition, which made a single probe
# wrongly report AX as ungranted and abort the whole run.
ax_ok() {
  local n
  for _ in 1 2 3 4; do
    n=$(osascript -e 'tell application "System Events" to count menu bars of (first application process whose frontmost is true)' 2>/dev/null || echo 0)
    [ "${n:-0}" -ge 1 ] && return 0
    /bin/sleep 0.5
  done
  return 1
}

# Crop a screen region "x,y,w,h" → outfile, or warn if bounds are invalid.
_capture_region() {  # $1=bounds  $2=outfile  $3=label
  local bounds="$1" outfile="$2" label="$3" x y w h
  if [[ "$bounds" =~ ^-?[0-9]+,-?[0-9]+,[0-9]+,[0-9]+$ ]]; then
    IFS=',' read -r x y w h <<<"$bounds"
    /bin/sleep 0.3   # let the just-raised window settle on top before capture
    screencapture -x -R"${x},${y},${w},${h}" "$outfile"
    log "captured ${label} → ${outfile##*/} (${w}x${h})"
    return 0
  fi
  log "WARN: ${label} not found via AX (bounds='${bounds:-}'); skipped"
  return 1
}

# Capture the region of an OpenBird AX window whose NAME contains $1 → $2.
capture_window() {
  local match="$1" outfile="$2" bounds
  osascript -e "tell application \"$APP_NAME\" to activate" >/dev/null 2>&1 || true
  /bin/sleep 0.4
  bounds=$(osascript 2>/dev/null <<OSA || true
tell application "System Events" to tell process "$APP_NAME"
  set frontmost to true
  set theWin to (first window whose name contains "$match")
  perform action "AXRaise" of theWin
  set p to position of theWin
  set s to size of theWin
  return (item 1 of p as integer as text) & "," & (item 2 of p as integer as text) & "," & (item 1 of s as integer as text) & "," & (item 2 of s as integer as text)
end tell
OSA
)
  _capture_region "$bounds" "$outfile" "window '${match}'"
}

# Capture the region of an OpenBird AX window by SUBROLE $1 (e.g. the Ask panel is
# an AXSystemDialog with an empty name, so it can't be matched by name) → $2.
capture_window_by_subrole() {
  local subrole="$1" outfile="$2" bounds
  bounds=$(osascript 2>/dev/null <<OSA || true
tell application "System Events" to tell process "$APP_NAME"
  set frontmost to true
  set theWin to (first window whose subrole is "$subrole")
  perform action "AXRaise" of theWin
  set p to position of theWin
  set s to size of theWin
  return (item 1 of p as integer as text) & "," & (item 2 of p as integer as text) & "," & (item 1 of s as integer as text) & "," & (item 2 of s as integer as text)
end tell
OSA
)
  _capture_region "$bounds" "$outfile" "subrole '${subrole}'"
}

echo "OpenBird E2E screenshots → $OUT"
log "target app: $APP_BUNDLE"

if ! ax_ok; then
  echo "ERROR: Accessibility is not granted to this terminal/host." >&2
  echo "Grant it in System Settings ▸ Privacy & Security ▸ Accessibility, then re-run." >&2
  echo "Opening the pane for you…" >&2
  open "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility" || true
  exit 2
fi

# Non-fatal: cliclick is only needed for the menu-dropdown capture. Warn up-front so
# a missing third-party tool doesn't read as a silent capture failure later.
if ! command -v cliclick >/dev/null 2>&1; then
  log "NOTE: cliclick not installed — the menu-dropdown capture will be skipped."
  log "      Install with: brew install cliclick"
  HAS_CLICLICK=0
else
  HAS_CLICLICK=1
fi

# 1) Onboarding sheet — reset the one-time flag and relaunch so it presents, then
#    capture the main window region (the sheet renders centered over it).
log "onboarding: resetting first-run flag + relaunching"
prepare_launch_services
defaults delete "$BUNDLE_ID" "$ONBOARDING_KEY" >/dev/null 2>&1 || true
relaunch
capture_window "$APP_NAME" "$OUT/06-setup-onboarding.png"

# 2) Main window (config / SetupView). Mark onboarding complete first so the sheet
#    isn't covering it, then re-show the main window.
defaults write "$BUNDLE_ID" "$ONBOARDING_KEY" -bool true >/dev/null 2>&1 || true
relaunch
capture_window "$APP_NAME" "$OUT/00-main-window.png"

# 3) Ask Spotlight panel — summon via the global ⌥Space hotkey, then capture.
log "ask: triggering ⌥Space"
osascript -e "tell application \"$APP_NAME\" to activate" >/dev/null 2>&1 || true
/bin/sleep 0.5
osascript -e 'tell application "System Events" to key code 49 using {option down}' >/dev/null 2>&1 || true
/bin/sleep 1.2
# The Ask panel is a floating NSPanel with an empty name; match it by subrole.
# `|| true` so a missed panel (e.g. the synthetic ⌥Space didn't fire the Carbon hotkey)
# warns and continues instead of aborting the rest of the run under `set -e`.
capture_window_by_subrole "AXSystemDialog" "$OUT/02-ask-spotlight.png" || true
osascript -e 'tell application "System Events" to key code 53' >/dev/null 2>&1 || true  # esc → close panel

# 4) Today window — opened via the openbird://today deep-link (no menu bar needed).
log "today: opening via openbird://today deep-link"
open "openbird://today" >/dev/null 2>&1 || true
/bin/sleep 2
assert_selected_app_running
capture_window "Today" "$OUT/05-today-dayview.png" || true

# 4b) Expanded Ask (the unified surface: chat + optional Sources/Timeline rails) —
#     opened via the gated openbird://ask-expanded deep-link. Poll until discoverable.
log "ask (expanded): opening via openbird://ask-expanded deep-link"
open "openbird://ask-expanded" >/dev/null 2>&1 || true
assert_selected_app_running
for _ in $(seq 1 20); do
  if capture_window "Ask" "$OUT/03-ask-expanded.png"; then break; fi
  /bin/sleep 0.2
done

# 5) Menu dropdown — reached only through the menu-bar status item. The SwiftUI
#    MenuBarExtra(.window) popover does NOT open via AX, and a full menu bar parks
#    the status item off-screen (AX pos like -1/976), so a coordinate click is
#    impossible too. Try a real click only if the item is genuinely on-screen;
#    otherwise SKIP cleanly rather than saving a misleading full-screen frame.
log "menu dropdown: checking status-item visibility"
if [[ "$HAS_CLICLICK" -ne 1 ]]; then
  log "SKIP menu dropdown: cliclick is not installed."
else
  ITEM_POS=$(osascript 2>/dev/null <<OSA || true
tell application "System Events" to tell process "$APP_NAME"
  set p to position of menu bar item 1 of menu bar 2
  return (item 1 of p as integer as text) & "," & (item 2 of p as integer as text)
end tell
OSA
)
  ITEM_X="${ITEM_POS%%,*}"; ITEM_Y="${ITEM_POS##*,}"
  if [[ "${ITEM_X:-x}" =~ ^[0-9]+$ ]] && [ "${ITEM_X}" -ge 0 ] && [[ "${ITEM_Y:-99}" =~ ^[0-9]+$ ]] && [ "${ITEM_Y}" -lt 60 ]; then
    cliclick "c:${ITEM_X},${ITEM_Y}" >/dev/null 2>&1 || true
    /bin/sleep 0.8
    capture_window_by_subrole "AXSystemDialog" "$OUT/01-menu-dropdown.png" || true
  else
    log "SKIP menu dropdown: status item hidden/off-screen (pos=${ITEM_POS:-?})."
    log "     The menu bar is full; ⌘-drag icons or quit a few menu-bar apps to reveal"
    log "     OpenBird's icon, then re-run. (The Today window no longer needs this —"
    log "     it now opens via the openbird://today deep-link above.)"
  fi
fi

echo "Done. Files in $OUT:"
shots=$(find "$OUT" -maxdepth 1 -name '*.png' -print 2>/dev/null | sort)
if [ -n "$shots" ]; then
  printf '%s\n' "$shots" | sed 's#^#  #'
else
  echo "  (none)"
fi
