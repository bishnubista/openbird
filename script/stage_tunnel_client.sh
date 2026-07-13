#!/usr/bin/env bash
# Stage the pinned official OpenAI Secure MCP Tunnel client into an app bundle.
set -euo pipefail

DEST="${1:-}"
[[ -n "$DEST" ]] || { echo "usage: $0 DEST" >&2; exit 2; }

VERSION="0.0.10"
case "$(uname -m)" in
  arm64)
    ARCH="arm64"
    SHA256="288accc7fd20cfee1d495adb933773af9e19ebc0cdef3173f7fb544afa5065b2"
    ;;
  x86_64)
    ARCH="amd64"
    SHA256="1a48616e584484f8bef4c1128d515ac96cf44d0d9609c1462abccc1793f4b847"
    ;;
  *) echo "unsupported macOS architecture: $(uname -m)" >&2; exit 1 ;;
esac

verify_macho() {
  local candidate="$1"
  [[ -f "$candidate" && -x "$candidate" ]] || return 1
  file -b "$candidate" | grep -q 'Mach-O'
}

if [[ -n "${OPENBIRD_TUNNEL_CLIENT:-}" ]]; then
  verify_macho "$OPENBIRD_TUNNEL_CLIENT" || {
    echo "OPENBIRD_TUNNEL_CLIENT must be an executable Mach-O file" >&2
    exit 1
  }
  cp "$OPENBIRD_TUNNEL_CLIENT" "$DEST"
  chmod +x "$DEST"
  exit 0
fi

CACHE_DIR="${OPENBIRD_BUILD_CACHE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/dist/cache}"
ASSET="tunnel-client-v${VERSION}-darwin-${ARCH}.zip"
ARCHIVE="$CACHE_DIR/$ASSET"
URL="https://github.com/openai/tunnel-client/releases/download/v${VERSION}/$ASSET"
mkdir -p "$CACHE_DIR"

if [[ ! -f "$ARCHIVE" ]] || [[ "$(shasum -a 256 "$ARCHIVE" | awk '{print $1}')" != "$SHA256" ]]; then
  rm -f "$ARCHIVE"
  curl --fail --location --silent --show-error "$URL" --output "$ARCHIVE"
fi
[[ "$(shasum -a 256 "$ARCHIVE" | awk '{print $1}')" == "$SHA256" ]] || {
  rm -f "$ARCHIVE"
  echo "tunnel-client checksum mismatch" >&2
  exit 1
}

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT
ditto -x -k "$ARCHIVE" "$TMP_DIR"
verify_macho "$TMP_DIR/tunnel-client" || {
  echo "official tunnel-client archive did not contain a runnable Mach-O" >&2
  exit 1
}
cp "$TMP_DIR/tunnel-client" "$DEST"
chmod +x "$DEST"
