#!/usr/bin/env bash
# Verify the pinned tunnel client accepts OpenBird's hardened launch contract.
set -euo pipefail

CLIENT="${1:-}"
[[ -x "$CLIENT" ]] || { echo "usage: $0 TUNNEL_CLIENT" >&2; exit 2; }

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/openbird-tunnel-smoke.XXXXXX")"
HEALTH_URL_FILE="$TMP_DIR/health.url"
STDERR_LOG="$TMP_DIR/stderr.log"
CLIENT_PID=""

# shellcheck disable=SC2329 # invoked indirectly by the traps below
cleanup() {
  if [[ -n "$CLIENT_PID" ]] && kill -0 "$CLIENT_PID" 2>/dev/null; then
    # Job control gives this owned process its own group; never scan or kill by name.
    kill -TERM -- "-$CLIENT_PID" 2>/dev/null || kill -TERM "$CLIENT_PID" 2>/dev/null || true
    wait "$CLIENT_PID" 2>/dev/null || true
  fi
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT INT TERM

# A local discard port keeps the contract check offline. The health/admin server
# starts independently and proves that the real binary accepted every launch flag.
set -m
CONTROL_PLANE_API_KEY="$(uuidgen)" "$CLIENT" run \
  --embedded-mcp-stub \
  --control-plane.tunnel-id tunnel_0123456789abcdef0123456789abcdef \
  --control-plane.api-key env:CONTROL_PLANE_API_KEY \
  --control-plane.base-url http://127.0.0.1:9 \
  --health.listen-addr 127.0.0.1:0 \
  --health.url-file "$HEALTH_URL_FILE" \
  --admin-ui.log-buffer-events 1 \
  --log.file /dev/null \
  </dev/null >/dev/null 2>"$STDERR_LOG" &
CLIENT_PID=$!
set +m

for _ in {1..50}; do
  if ! kill -0 "$CLIENT_PID" 2>/dev/null; then
    break
  fi
  if [[ -s "$HEALTH_URL_FILE" ]]; then
    HEALTH_BASE="$(tr -d '\r\n' < "$HEALTH_URL_FILE")"
    if [[ "$HEALTH_BASE" == http://127.0.0.1:* ]] && \
      curl --fail --silent --show-error --max-time 1 "$HEALTH_BASE/healthz" >/dev/null; then
      exit 0
    fi
  fi
  sleep 0.1
done

echo "tunnel-client did not satisfy OpenBird's offline launch contract" >&2
if [[ -s "$STDERR_LOG" ]]; then
  sed -n '1,40p' "$STDERR_LOG" >&2
fi
exit 1
