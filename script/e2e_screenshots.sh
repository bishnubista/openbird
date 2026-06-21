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
#   2. Build + launch the app first:
#        OPENBIRD_SWIFTPM_DISABLE_SANDBOX=1 ./script/build_and_run.sh --no-launch
#        open dist/OpenBird.app
#      On an unsigned dev build, approve the one-time Keychain ACL prompt with
#      "Always Allow" (NOT Escape — Escape denies it and dismisses sheets).
#
# Usage:  ./script/e2e_screenshots.sh
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
mkdir -p "$OUT"

log() { printf '  • %s\n' "$*"; }

# Quit the app, wait for it to fully exit, then relaunch (retrying open, since
# LaunchServices returns -600 if you re-open while the prior instance is quitting).
relaunch() {
  pkill -f "$APP_NAME.app/Contents/MacOS/$APP_NAME" >/dev/null 2>&1 || true
  local _
  for _ in $(seq 1 50); do
    pgrep -f "$APP_NAME.app/Contents/MacOS/$APP_NAME" >/dev/null || break
    /bin/sleep 0.1
  done
  # --enable-e2e-deeplinks arms the openbird:// router, which is inert in normal use
  # (so a stray URL can't force-show private data). `open --args` only delivers args
  # on a cold launch — which this is, since we just killed the prior instance.
  for _ in 1 2 3; do
    if open "$ROOT/dist/$APP_NAME.app" --args --enable-e2e-deeplinks 2>/dev/null; then break; fi
    /bin/sleep 0.5
  done
  /bin/sleep 5
}

# Is AX usable by this host process?
ax_ok() {
  local n
  n=$(osascript -e 'tell application "System Events" to count menu bars of (first application process whose frontmost is true)' 2>/dev/null || echo 0)
  [ "${n:-0}" -ge 1 ]
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

if ! pgrep -f "$APP_NAME.app/Contents/MacOS/$APP_NAME" >/dev/null; then
  echo "ERROR: $APP_NAME is not running. Launch it first (see header)." >&2
  exit 1
fi

if ! ax_ok; then
  echo "ERROR: Accessibility is not granted to this terminal/host." >&2
  echo "Grant it in System Settings ▸ Privacy & Security ▸ Accessibility, then re-run." >&2
  echo "Opening the pane for you…" >&2
  open "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility" || true
  exit 2
fi

# 1) Onboarding sheet — reset the one-time flag and relaunch so it presents, then
#    capture the main window region (the sheet renders centered over it).
log "onboarding: resetting first-run flag + relaunching"
defaults delete "$BUNDLE_ID" openbird.onboarding.completed >/dev/null 2>&1 || true
relaunch
capture_window "$APP_NAME" "$OUT/06-setup-onboarding.png"

# 2) Main window (config / SetupView). Mark onboarding complete first so the sheet
#    isn't covering it, then re-show the main window.
defaults write "$BUNDLE_ID" openbird.onboarding.completed -bool true >/dev/null 2>&1 || true
relaunch
capture_window "$APP_NAME" "$OUT/00-main-window.png"

# 3) Ask Spotlight panel — summon via the global ⌥Space hotkey, then capture.
log "ask: triggering ⌥Space"
osascript -e "tell application \"$APP_NAME\" to activate" >/dev/null 2>&1 || true
/bin/sleep 0.5
osascript -e 'tell application "System Events" to key code 49 using {option down}' >/dev/null 2>&1 || true
/bin/sleep 1.2
# The Ask panel is a floating NSPanel with an empty name; match it by subrole.
capture_window_by_subrole "AXSystemDialog" "$OUT/02-ask-spotlight.png"
osascript -e 'tell application "System Events" to key code 53' >/dev/null 2>&1 || true  # esc → close panel

# 4) Today window — opened via the openbird://today deep-link (no menu bar needed).
log "today: opening via openbird://today deep-link"
open "openbird://today" >/dev/null 2>&1 || true
/bin/sleep 2
capture_window "Today" "$OUT/05-today-dayview.png" || true

# 5) Menu dropdown — reached only through the menu-bar status item. The SwiftUI
#    MenuBarExtra(.window) popover does NOT open via AX, and a full menu bar parks
#    the status item off-screen (AX pos like -1/976), so a coordinate click is
#    impossible too. Try a real click only if the item is genuinely on-screen;
#    otherwise SKIP cleanly rather than saving a misleading full-screen frame.
log "menu dropdown: checking status-item visibility"
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

echo "Done. Files in $OUT:"
find "$OUT" -maxdepth 1 -name '*.png' -print 2>/dev/null | sed 's#^#  #' || echo "  (none)"
