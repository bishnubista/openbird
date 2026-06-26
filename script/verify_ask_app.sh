#!/usr/bin/env bash
# =============================================================================
# verify_ask_app.sh — autonomous GUI-app verification of "Ask OpenBird".
# =============================================================================
# Validates the REAL app's ask pipeline end-to-end with NO human and NO permission
# prompts. It launches the dist bundle in self-test mode (OPENBIRD_SELFTEST_ASK),
# which runs the query through the SAME OpenBirdService.askChat seam the Ask panel
# uses — spawning the real `openbird chat` engine — then prints a deterministic
# outcome and exits. We assert on that line.
#
# Why not drive the panel with keystrokes? System Events automation triggers an
# Accessibility/Automation permission prompt — a human gate that defeats unattended
# self-testing. Self-test mode runs the same code path headlessly, prompt-free.
#
# Scope: this exercises the app's ask PIPELINE (service → CLI → grounding/citation
# validation), not the SwiftUI rendering of the answer bubble. That's where the
# grounding fix lives, so it's the meaningful signal.
#
# Exit: 0 = PASS (grounded, >=1 citation) · 1 = FAIL (ran, ungrounded)
#       2 = BLOCKED (no outcome line — build/launch/CLI problem, not a fix verdict)
#
# Usage: script/verify_ask_app.sh ["Summarize my day"] [--build]
# =============================================================================
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_BIN="$ROOT_DIR/dist/OpenBird.app/Contents/MacOS/OpenBird"
OUT_DIR="${OPENBIRD_VERIFY_OUT:-$ROOT_DIR/.verify-out}"
QUERY="Summarize my day"
DO_BUILD=0
WAIT_SECONDS="${OPENBIRD_VERIFY_WAIT:-120}"

for arg in "$@"; do
  case "$arg" in
    --build) DO_BUILD=1 ;;
    --*) echo "unknown flag: $arg" >&2; exit 2 ;;
    *) QUERY="$arg" ;;
  esac
done

log() { printf '[verify-ask] %s\n' "$1" >&2; }
mkdir -p "$OUT_DIR"

if [ "$DO_BUILD" -eq 1 ]; then
  log "building app bundle (slow)…"
  OPENBIRD_SWIFTPM_DISABLE_SANDBOX=1 "$ROOT_DIR/script/build_and_run.sh" --no-launch \
    > "$OUT_DIR/build.log" 2>&1 || { log "BLOCKED: build failed ($OUT_DIR/build.log)"; echo "VERDICT: BLOCKED"; exit 2; }
fi
if [ ! -x "$APP_BIN" ]; then
  log "BLOCKED: no app bundle at $APP_BIN (run with --build)"; echo "VERDICT: BLOCKED"; exit 2
fi

pkill -f "dist/OpenBird.app/Contents/MacOS/OpenBird" 2>/dev/null && sleep 1

# Headless self-test: query via env, unencrypted DB (no Keychain prompt). The app
# runs the ask and exits on its own; `timeout` bounds a hung model/CLI. The query
# is NOT echoed to output — it could be a content-bearing question; keep this tool
# to counts/booleans like the app's own signpost.
log "running self-test ask (headless, no permission prompts)…"
RUN_LOG="$OUT_DIR/selftest_run.log"
OPENBIRD_SELFTEST_ASK="$QUERY" OPENBIRD_DISABLE_KEYRING=1 \
  gtimeout --signal=KILL "$WAIT_SECONDS" "$APP_BIN" > "$RUN_LOG" 2>&1
RC=$?

OUTCOME="$(grep -oE 'SELFTEST ask\.outcome [a-z0-9= ]+' "$RUN_LOG" | tail -1)"
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
  log "BLOCKED: self-test ask errored (engine/CLI problem, exit $RC) — not a fix verdict"
  echo "VERDICT: BLOCKED"; exit 2
fi
GROUNDED="$(printf '%s' "$OUTCOME" | grep -oE 'grounded=[0-9]+' | cut -d= -f2)"
CITES="$(printf '%s' "$OUTCOME" | grep -oE 'citations=[0-9]+' | cut -d= -f2)"

if [ "${GROUNDED:-0}" = "1" ] && [ "${CITES:-0}" -ge 1 ]; then
  echo "VERDICT: PASS — grounded=$GROUNDED citations=$CITES"
  exit 0
fi
echo "VERDICT: FAIL — ungrounded (grounded=${GROUNDED:-?} citations=${CITES:-?})"
exit 1
