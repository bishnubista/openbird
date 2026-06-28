#!/usr/bin/env bash
# =============================================================================
# verify_ask_app.sh — autonomous GUI-app verification of "Ask OpenBird".
# =============================================================================
# Validates the REAL app's ask pipeline end-to-end. It launches the dist bundle
# in self-test mode (OPENBIRD_SELFTEST_ASK), which runs the query through the SAME
# OpenBirdService.askChat seam the Ask panel uses — spawning the real `openbird
# chat` engine — then prints a deterministic outcome and exits. We assert on that
# line.
#
# Why not drive the panel with keystrokes? System Events automation triggers an
# Accessibility/Automation permission prompt — a human gate that defeats unattended
# self-testing. Self-test mode runs the same code path headlessly, prompt-free.
#
# Default mode is for beta rehearsal against the encrypted real store: the app
# resolves its own DB key before spawning the CLI child. On a fresh or unsigned
# bundle that may require a Keychain ACL prompt; the outer timeout maps that to
# BLOCKED, never PASS. Use --plaintext-dev for the old prompt-free path against an
# explicitly plaintext dev DB.
#
# Scope: this exercises the app's ask PIPELINE (service → CLI → grounding/citation
# validation), not the SwiftUI rendering of the answer bubble. That's where the
# grounding fix lives, so it's the meaningful signal.
#
# Exit: 0 = PASS (grounded, >=1 display source) · 1 = FAIL (ran, ungrounded)
#       2 = BLOCKED (no outcome line — build/launch/CLI problem, not a fix verdict)
#
# Usage: script/verify_ask_app.sh ["Summarize my day"] [--app /path/to/OpenBird.app] [--build] [--plaintext-dev]
# =============================================================================
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_APP_BUNDLE="$ROOT_DIR/dist/OpenBird.app"
APP_BUNDLE="${OPENBIRD_VERIFY_APP_PATH:-$DEFAULT_APP_BUNDLE}"
OUT_DIR="${OPENBIRD_VERIFY_OUT:-$ROOT_DIR/.verify-out}"
QUERY="Summarize my day"
DO_BUILD=0
PLAINTEXT_DEV=0
WAIT_SECONDS="${OPENBIRD_VERIFY_WAIT:-120}"
APP_WAS_OVERRIDDEN=0

if [ -n "${OPENBIRD_VERIFY_APP_PATH:-}" ]; then
  APP_WAS_OVERRIDDEN=1
fi

while [ "$#" -gt 0 ]; do
  case "$1" in
    --build)
      DO_BUILD=1
      shift
      ;;
    --plaintext-dev)
      PLAINTEXT_DEV=1
      shift
      ;;
    --app)
      [ "$#" -ge 2 ] || { echo "--app requires a path" >&2; exit 2; }
      APP_BUNDLE="$2"
      APP_WAS_OVERRIDDEN=1
      shift 2
      ;;
    --app=*)
      APP_BUNDLE="${1#--app=}"
      APP_WAS_OVERRIDDEN=1
      shift
      ;;
    --*) echo "unknown flag: $1" >&2; exit 2 ;;
    *)
      QUERY="$1"
      shift
      ;;
  esac
done

log() { printf '[verify-ask] %s\n' "$1" >&2; }
mkdir -p "$OUT_DIR"

if [ "$DO_BUILD" -eq 1 ] && [ "$APP_WAS_OVERRIDDEN" -eq 1 ]; then
  log "BLOCKED: --build only applies to the default dist app target"
  echo "VERDICT: BLOCKED"; exit 2
fi

if [ "$DO_BUILD" -eq 1 ]; then
  log "building app bundle (slow)…"
  OPENBIRD_SWIFTPM_DISABLE_SANDBOX=1 "$ROOT_DIR/script/build_and_run.sh" --no-launch \
    > "$OUT_DIR/build.log" 2>&1 || { log "BLOCKED: build failed ($OUT_DIR/build.log)"; echo "VERDICT: BLOCKED"; exit 2; }
fi

APP_BUNDLE="$(cd "$APP_BUNDLE" 2>/dev/null && pwd -P || printf '%s' "$APP_BUNDLE")"
APP_BIN="$APP_BUNDLE/Contents/MacOS/OpenBird"
log "target app: $APP_BUNDLE"
if [ ! -x "$APP_BIN" ]; then
  log "BLOCKED: no app bundle at $APP_BIN (run with --build)"; echo "VERDICT: BLOCKED"; exit 2
fi

if [ "$APP_WAS_OVERRIDDEN" -eq 0 ]; then
  pkill -f "dist/OpenBird.app/Contents/MacOS/OpenBird" 2>/dev/null && sleep 1
fi

# Headless self-test: query via env. The app runs the ask and exits on its own;
# `timeout` bounds a hung Keychain/model/CLI path. The query is NOT echoed to
# output — it could be a content-bearing question; keep this tool to counts,
# booleans, and safe reason codes like the app's own signpost.
log "running self-test ask (headless, no UI automation)…"
RUN_LOG="$OUT_DIR/selftest_run.log"
if [ "$PLAINTEXT_DEV" -eq 1 ]; then
  log "mode: plaintext dev DB (OPENBIRD_DISABLE_KEYRING=1)"
  OPENBIRD_SELFTEST_ASK="$QUERY" OPENBIRD_DISABLE_KEYRING=1 \
    gtimeout --signal=KILL "$WAIT_SECONDS" "$APP_BIN" > "$RUN_LOG" 2>&1
else
  log "mode: encrypted real store (app-owned DB key)"
  OPENBIRD_SELFTEST_ASK="$QUERY" \
    gtimeout --signal=KILL "$WAIT_SECONDS" "$APP_BIN" > "$RUN_LOG" 2>&1
fi
RC=$?

OUTCOME="$(grep -oE 'SELFTEST ask\.outcome [a-z0-9=_ ]+' "$RUN_LOG" | tail -1)"
if [ -z "$OUTCOME" ]; then
  log "BLOCKED: no SELFTEST outcome (exit $RC). Tail of run log:"
  tail -5 "$RUN_LOG" >&2
  echo "VERDICT: BLOCKED"; exit 2
fi

log "signal: $OUTCOME"
# An engine/CLI error (the ask threw) is NOT a grounding verdict — the app emits
# `error=1` and exits 2. Map it to BLOCKED, not FAIL (which is reserved for a real
# ungrounded answer), matching this script's documented exit contract.
if printf '%s' "$OUTCOME" | grep -q 'error=1'; then
  KIND="$(printf '%s' "$OUTCOME" | grep -oE 'kind=[a-z0-9_]+' | cut -d= -f2)"
  case "${KIND:-unknown}" in
    db_key_unavailable)
      log "BLOCKED: DB key unavailable (encrypted store unverifiable, or plaintext store — use --plaintext-dev)"
      ;;
    *)
      log "BLOCKED: self-test ask errored (kind=${KIND:-unknown}, exit $RC) — not a fix verdict"
      ;;
  esac
  echo "VERDICT: BLOCKED"; exit 2
fi
GROUNDED="$(printf '%s' "$OUTCOME" | grep -oE 'grounded=[0-9]+' | cut -d= -f2)"
CITES="$(printf '%s' "$OUTCOME" | grep -oE 'citations=[0-9]+' | cut -d= -f2)"
DERIVED="$(printf '%s' "$OUTCOME" | grep -oE 'derived=[0-9]+' | cut -d= -f2)"
SOURCES="$(printf '%s' "$OUTCOME" | grep -oE 'sources=[0-9]+' | cut -d= -f2)"
SOURCE_SIGNAL="${SOURCES:-${CITES:-0}}"

if [ "${GROUNDED:-0}" = "1" ] && [ "${SOURCE_SIGNAL:-0}" -ge 1 ]; then
  echo "VERDICT: PASS — grounded=$GROUNDED citations=${CITES:-?} derived=${DERIVED:-?} sources=${SOURCES:-$SOURCE_SIGNAL}"
  exit 0
fi
echo "VERDICT: FAIL — ungrounded (grounded=${GROUNDED:-?} citations=${CITES:-?} derived=${DERIVED:-?} sources=${SOURCES:-$SOURCE_SIGNAL})"
exit 1
