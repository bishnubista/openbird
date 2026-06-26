#!/usr/bin/env bash
# package_dmg.sh — build a SELF-CONTAINED, Developer ID-signed, notarized .dmg of
# OpenBird for beta testers. One download → drag to /Applications → runs with no
# repo checkout, no system uv/Python, no Swift toolchain, no Gatekeeper warning.
#
# This is a SEPARATE channel from the brew/dev path (build_and_run.sh +
# sign_local.sh, which stay ad-hoc). It never mutates the brew bundle in place:
# it builds with OPENBIRD_SKIP_SIGN=1 and works in its own staging dir.
#
# Design + rationale: docs/design/notarized-dmg-distribution.md
#
# Required:
#   OPENBIRD_SIGN_IDENTITY  e.g. "Developer ID Application: bishnu bista (SB26BAMXJM)"
# Optional:
#   OPENBIRD_NOTARY_PROFILE    notarytool keychain profile (default: openbird-notary)
#   OPENBIRD_DMG_PY_VERSION    embedded CPython (default: 3.13)
#   OPENBIRD_DMG_EXTRAS        pip extras (default: encryption,integrations) — NOT meetings
#   OPENBIRD_DMG_SKIP_NOTARIZE=1   stop after building+signing+relocation test (fast iteration)
set -euo pipefail

APP_NAME="OpenBird"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Optional, gitignored local overrides (signing identity / notary profile / etc.).
# Neither the identity nor the profile name is a secret — the identity is public
# (it's in every signed binary) and the profile is just a keychain label; the real
# signing/notary credentials live in the macOS keychain. So these are CONFIG.
# shellcheck source=/dev/null
[ -f "$ROOT_DIR/script/release.env" ] && . "$ROOT_DIR/script/release.env"
NOTARY_PROFILE="${OPENBIRD_NOTARY_PROFILE:-openbird-notary}"
PY_VERSION="${OPENBIRD_DMG_PY_VERSION:-3.13}"
EXTRAS="${OPENBIRD_DMG_EXTRAS:-encryption,integrations}"

STAGE_DIR="$ROOT_DIR/dist/dmg-stage"
APP="$STAGE_DIR/$APP_NAME.app"
RES="$APP/Contents/Resources"
MACOS="$APP/Contents/MacOS"
DMG_PATH="$ROOT_DIR/dist/$APP_NAME.dmg"

ENT_APP="$ROOT_DIR/mac-app/OpenBird.entitlements"
ENT_HELPER="$ROOT_DIR/mac-app/Helper.entitlements"
# capture-helper needs Apple Events (browser URL scripting); audio-helper does
# NOT, so they sign with separate profiles (least privilege).
ENT_CAPTURE_HELPER="$ROOT_DIR/mac-app/CaptureHelper.entitlements"
ENT_PYTHON="$ROOT_DIR/mac-app/Python.entitlements"

log() { echo "package_dmg: $*" >&2; }
die() { echo "package_dmg: ERROR: $*" >&2; exit 1; }

remove_python_bytecode() {
  local root="$1"
  find "$root" -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete
  find "$root" -type d -name '__pycache__' -empty -delete
}

python_bytecode_manifest() {
  local root="$1"
  (
    cd "$root"
    {
      find . -type d -name '__pycache__' -print | LC_ALL=C sort | sed 's/^/D /'
      find . -type f \( -name '*.pyc' -o -name '*.pyo' \) -print | LC_ALL=C sort | while IFS= read -r f; do
        shasum -a 256 "$f" | sed 's/^/F /'
      done
    }
  )
}

assert_python_bytecode_unchanged() {
  local root="$1" before="$2" label="$3" after
  after="$(mktemp)"
  python_bytecode_manifest "$root" >"$after"
  if diff -u "$before" "$after" >&2; then
    rm -f "$after"
    return 0
  fi
  rm -f "$after"
  log "  Python bytecode changed during $label"
  return 1
}

if [[ "${1:-}" == "--self-test-bytecode-guard" ]]; then
  tmp_self_test="$(mktemp -d)"
  trap 'rm -rf "$tmp_self_test"' EXIT
  mkdir -p "$tmp_self_test/App.app/Contents/Resources/python/pkg/__pycache__"
  printf 'old\n' >"$tmp_self_test/App.app/Contents/Resources/python/pkg/__pycache__/a.pyc"
  before_self_test="$tmp_self_test/before.manifest"
  python_bytecode_manifest "$tmp_self_test/App.app" >"$before_self_test"
  printf 'changed\n' >"$tmp_self_test/App.app/Contents/Resources/python/pkg/__pycache__/a.pyc"
  if assert_python_bytecode_unchanged "$tmp_self_test/App.app" "$before_self_test" "self-test mutation" >/dev/null 2>&1; then
    die "bytecode guard self-test failed to catch a modified .pyc"
  fi
  printf 'old\n' >"$tmp_self_test/App.app/Contents/Resources/python/pkg/__pycache__/a.pyc"
  python_bytecode_manifest "$tmp_self_test/App.app" >"$before_self_test"
  printf 'new\n' >"$tmp_self_test/App.app/Contents/Resources/python/pkg/__pycache__/b.pyc"
  if assert_python_bytecode_unchanged "$tmp_self_test/App.app" "$before_self_test" "self-test addition" >/dev/null 2>&1; then
    die "bytecode guard self-test failed to catch a new .pyc"
  fi
  log "bytecode guard self-test OK"
  exit 0
fi

# Resolve the Developer ID signing identity: an explicit OPENBIRD_SIGN_IDENTITY
# (env or script/release.env) wins; otherwise auto-derive the single
# "Developer ID Application" identity from the keychain.
DEVID="${OPENBIRD_SIGN_IDENTITY:-}"
if [ -z "$DEVID" ]; then
  # Count RAW identity lines (not deduped): two distinct certs can share the same
  # CN string (e.g. a renewed + old one), and `sort -u` would collapse them to one,
  # auto-selecting ambiguously. Counting lines makes any duplicate a hard error.
  _lines="$(security find-identity -v -p codesigning 2>/dev/null | grep "Developer ID Application" || true)"
  _n="$(printf '%s\n' "$_lines" | grep -c . || true)"
  if [ "$_n" -eq 1 ]; then
    DEVID="$(printf '%s\n' "$_lines" | sed -n 's/.*"\(Developer ID Application: [^"]*\)".*/\1/p')"
  elif [ "$_n" -eq 0 ]; then
    die "no 'Developer ID Application' identity in the keychain — create one, or set OPENBIRD_SIGN_IDENTITY (env or script/release.env)."
  else
    printf '%s\n' "$_lines" >&2
    die "multiple Developer ID Application identities found — set OPENBIRD_SIGN_IDENTITY to choose one."
  fi
fi
log "Signing identity: $DEVID"
log "Notary profile: $NOTARY_PROFILE"

# ---------------------------------------------------------------------------
log "[1/8] Build unsigned app bundle (brew path untouched; OPENBIRD_SKIP_SIGN=1)"
OPENBIRD_SKIP_SIGN=1 OPENBIRD_SWIFTPM_DISABLE_SANDBOX="${OPENBIRD_SWIFTPM_DISABLE_SANDBOX:-1}" \
  "$ROOT_DIR/script/build_and_run.sh" --no-launch >/dev/null
SRC_APP="$ROOT_DIR/dist/$APP_NAME.app"
[[ -d "$SRC_APP" ]] || die "expected $SRC_APP from build_and_run.sh"

log "[2/8] Stage a private copy (never mutate the brew bundle in place)"
rm -rf "$STAGE_DIR"; mkdir -p "$STAGE_DIR"
cp -R "$SRC_APP" "$APP"
# build_and_run.sh built $SRC_APP with OPENBIRD_SKIP_SIGN=1, leaving the shared
# dev/brew artifact unsigned. Restore its normal ad-hoc signature so this
# packaging run leaves dist/OpenBird.app exactly as the brew/dev path expects.
"$ROOT_DIR/script/sign_local.sh" "$SRC_APP" >&2 || log "warn: could not re-sign dev artifact $SRC_APP"

# ---------------------------------------------------------------------------
log "[3/8] Embed a relocatable standalone CPython ($PY_VERSION)"
uv python install "$PY_VERSION" >&2 || true
PYBIN="$(uv python find "$PY_VERSION")"
PYBIN="$(readlink -f "$PYBIN" 2>/dev/null || echo "$PYBIN")"
PYROOT="$(cd "$(dirname "$PYBIN")/.." && pwd -P)"
[[ -x "$PYROOT/bin/python3" ]] || die "could not locate standalone python root from $PYBIN"
log "  standalone python: $PYROOT"
rm -rf "$RES/python"
cp -R "$PYROOT" "$RES/python"
BPY="$RES/python/bin/python3"
chmod +x "$BPY"

log "[4/8] Install openbird[$EXTRAS] into the embedded interpreter"
# This is now OUR private bundled interpreter, so drop the PEP-668 marker that
# uv-managed standalone Pythons ship (otherwise pip refuses: externally-managed).
rm -f "$RES/python/lib/python"*/EXTERNALLY-MANAGED
"$BPY" -m ensurepip --upgrade >/dev/null 2>&1 || true
"$BPY" -m pip install --upgrade pip >/dev/null 2>&1 || true
"$BPY" -m pip install --no-warn-script-location "${ROOT_DIR}[${EXTRAS}]" >&2
# Keep ONLY the interpreter + its python/python3 symlinks. Remove every other
# entry — files AND symlinks (console scripts bake absolute shebangs; idle3/pydoc3
# and *-config dangle once their script targets are gone, which breaks
# `codesign --verify --strict`). We launch via `python -m openbird`, so nothing
# else in bin/ is needed at runtime.
# Resolve the real interpreter name from the python3 symlink (e.g. python3.13) so
# this is version-agnostic — hardcoding a version would delete the actual
# interpreter when OPENBIRD_DMG_PY_VERSION differs. Keep python/python3/pythonX.Y;
# delete everything else (console scripts AND pythonX.Y-config, which carries an
# absolute shebang).
real_py="$(basename "$(readlink "$RES/python/bin/python3" 2>/dev/null || true)")"
[ -n "$real_py" ] && [ "$real_py" != "." ] || real_py="python${PY_VERSION}"
find "$RES/python/bin" -mindepth 1 -maxdepth 1 \
  ! -name "$real_py" ! -name 'python3' ! -name 'python' -delete

# Normalize libpython's install id: uv-managed standalone Pythons bake an absolute
# LC_ID_DYLIB into the uv cache path. python loads libpython via @executable_path
# (so it works), but we must not SHIP an absolute build-machine path. Done before
# signing so signatures stay valid.
for libpy in "$RES"/python/lib/libpython*.dylib; do
  [ -f "$libpy" ] || continue
  install_name_tool -id "@executable_path/../lib/$(basename "$libpy")" "$libpy"
done

log "[5/8] Rewrite launcher shim to the embedded interpreter (no uv, no repo)"
cat >"$MACOS/openbird-cli" <<'WRAPPER'
#!/bin/sh
set -eu
DIR="$(cd "$(dirname "$0")" && pwd)"
PY="$DIR/../Resources/python/bin/python3"
# Sanitize inherited env so the bundled interpreter can't be hijacked to load a
# foreign stdlib/extension or inject a dylib (we ship a self-contained runtime).
unset PYTHONHOME PYTHONPATH DYLD_LIBRARY_PATH DYLD_INSERT_LIBRARIES DYLD_FRAMEWORK_PATH
export PYTHONDONTWRITEBYTECODE=1
export OPENBIRD_CAPTURE_HELPER="$DIR/capture-helper"
export OPENBIRD_AUDIO_HELPER="$DIR/audio-helper"
exec "$PY" -m openbird "$@"
WRAPPER
chmod +x "$MACOS/openbird-cli"

# ---------------------------------------------------------------------------
log "[6/8] Relocation audit (HARD FAIL on build-path leakage)"
# Nothing we depend on may reference the build tree, the uv cache, or $HOME paths.
leak_scan() {
  # scan text files we rely on (shebangs, pyvenv, sysconfig) for absolute leaks
  grep -RIl -e "$ROOT_DIR" -e "$HOME/.local/share/uv" -e "$PYROOT" \
    "$RES/python/bin" "$MACOS/openbird-cli" 2>/dev/null || true
}
leaks="$(leak_scan)"
if [[ -n "$leaks" ]]; then
  log "  build-path leaks found in:"; echo "$leaks" >&2
  die "relocation audit failed (absolute build paths embedded)"
fi
# sysconfig/prefix legitimately resolve relative to the interpreter, so under the
# staging dir they DO contain $ROOT_DIR — that is not a leak. The real failure is
# a reference to the uv CACHE (means the bundle still depends on ~/.local/share/uv).
# The authoritative relocation proof is the moved-copy run test below.
syscfg="$("$BPY" -c 'import sysconfig,sys; print(sysconfig.get_paths()["stdlib"]); print(sys.prefix)')"
if echo "$syscfg" | grep -q "/.local/share/uv"; then
  die "sysconfig/prefix still references the uv cache (not self-contained): $syscfg"
fi
# Rigorous Mach-O audit: no embedded Mach-O may reference an absolute path outside
# /System or /usr/lib (catches uv-cache / build-tree leaks in LC_ID_DYLIB and
# LC_LOAD_DYLIB that the text grep + local run-test can miss — they resolve on the
# build machine but would break on a tester's Mac).
macho_leaks=""
while IFS= read -r -d '' f; do
  if file -b "$f" 2>/dev/null | grep -q "Mach-O"; then
    # otool -L prints a col-0 header (the file's own path) then TAB-indented lines
    # for the install id + each dependency. Only the indented lines are real refs;
    # matching col-0 would false-positive on every file's own path.
    refs="$(otool -L "$f" 2>/dev/null | grep -E '^[[:space:]]+/' \
      | awk '{print $1}' | grep -vE '^(/System/|/usr/lib/)' || true)"
    [ -n "$refs" ] && macho_leaks="${macho_leaks}
${f}:
${refs}"
  fi
done < <(find "$APP" -type f -print0)
if [ -n "$macho_leaks" ]; then
  printf '%s\n' "$macho_leaks" >&2
  die "Mach-O load-command audit failed (absolute non-system paths embedded)"
fi
log "  relocation audit clean"

# pip/sysconfig may write timestamp bytecode while staging. Rebuild a clean,
# hash-based cache as the final staged Python execution before the relocated
# smoke test and signing. Runtime writes stay disabled in openbird-cli.
log "  rebuilding sealed Python bytecode cache"
remove_python_bytecode "$APP"
"$BPY" -m compileall -q -f --invalidation-mode unchecked-hash "$RES/python" >&2
STAGED_BYTECODE_MANIFEST="$(mktemp)"
python_bytecode_manifest "$APP" >"$STAGED_BYTECODE_MANIFEST"

# Prove relocatability: copy the staged app to a DIFFERENT path and run the CLI.
RELOC_TMP="$(mktemp -d)"
RELOC_APP="$RELOC_TMP/$APP_NAME.app"
cp -R "$APP" "$RELOC_APP"
RELOC_BYTECODE_MANIFEST="$RELOC_TMP/bytecode.before"
python_bytecode_manifest "$RELOC_APP" >"$RELOC_BYTECODE_MANIFEST"
if "$RELOC_APP/Contents/MacOS/openbird-cli" --help >/dev/null 2>"$RELOC_TMP/err"; then
  log "  relocation run OK (CLI runs from a moved copy)"
else
  log "  CLI failed from moved copy:"; cat "$RELOC_TMP/err" >&2
  rm -rf "$RELOC_TMP"; die "embedded interpreter is NOT relocatable"
fi
assert_python_bytecode_unchanged "$RELOC_APP" "$RELOC_BYTECODE_MANIFEST" "relocated CLI smoke test" \
  || { rm -rf "$RELOC_TMP"; die "relocated CLI mutated Python bytecode inside the app bundle"; }
rm -rf "$RELOC_TMP"
assert_python_bytecode_unchanged "$APP" "$STAGED_BYTECODE_MANIFEST" "pre-sign staging" \
  || die "staged app Python bytecode changed before signing"
rm -f "$STAGED_BYTECODE_MANIFEST"

# ---------------------------------------------------------------------------
log "[7/8] Developer ID sign every Mach-O, inside-out (no --deep)"
sign_one() {  # path [entitlements]
  local path="$1" ent="${2:-}"
  codesign --force --options runtime --timestamp \
    ${ent:+--entitlements "$ent"} --sign "$DEVID" "$path"
}
# 7a. Every nested Mach-O detected by magic (libs, .so, embedded binaries),
#     EXCEPT the python interpreters (signed separately with Python.entitlements)
#     and the helpers/app (handled below). Sign these first.
while IFS= read -r -d '' f; do
  case "$f" in
    "$MACOS/$APP_NAME"|"$MACOS/capture-helper"|"$MACOS/audio-helper") continue ;;
    "$RES/python/bin/python"*) continue ;;
  esac
  if file -b "$f" 2>/dev/null | grep -q "Mach-O"; then
    sign_one "$f"
  fi
done < <(find "$APP" -type f -print0)
# 7b. Embedded Python interpreters → Python.entitlements (the dlopen-ers)
while IFS= read -r -d '' f; do
  if file -b "$f" 2>/dev/null | grep -q "Mach-O"; then
    sign_one "$f" "$ENT_PYTHON"
  fi
done < <(find "$RES/python/bin" -maxdepth 1 -type f -name 'python*' -print0)
# 7c. Helpers, launcher shim, then the app LAST (seals everything)
sign_one "$MACOS/capture-helper" "$ENT_CAPTURE_HELPER"
sign_one "$MACOS/audio-helper"   "$ENT_HELPER"
codesign --force --timestamp --sign "$DEVID" "$MACOS/openbird-cli"
sign_one "$APP" "$ENT_APP"

log "  verifying signature"
codesign --verify --strict --verbose=2 "$APP" >&2
codesign -dvv "$APP" 2>&1 | grep -E "Authority=Developer ID|TeamIdentifier|flags" | sed 's/^/  /' >&2

if [[ "${OPENBIRD_DMG_SKIP_NOTARIZE:-0}" == "1" ]]; then
  log "[8/8] SKIP_NOTARIZE set — stopping after sign. Signed app: $APP"
  exit 0
fi

# ---------------------------------------------------------------------------
log "[8/8] Notarize + staple (app, then dmg)"
ZIP="$ROOT_DIR/dist/$APP_NAME-notarize.zip"
rm -f "$ZIP"
/usr/bin/ditto -c -k --keepParent "$APP" "$ZIP"
log "  submitting app to notary ($NOTARY_PROFILE)…"
if ! xcrun notarytool submit "$ZIP" --keychain-profile "$NOTARY_PROFILE" --wait >&2; then
  die "app notarization failed (run: xcrun notarytool log <id> --keychain-profile $NOTARY_PROFILE)"
fi
xcrun stapler staple "$APP" >&2

log "  building dmg"
rm -f "$DMG_PATH"
DMG_SRC="$(mktemp -d)"; cp -R "$APP" "$DMG_SRC/"; ln -s /Applications "$DMG_SRC/Applications"
hdiutil create -volname "$APP_NAME" -srcfolder "$DMG_SRC" -ov -format UDZO "$DMG_PATH" >&2
rm -rf "$DMG_SRC"
log "  signing + notarizing the dmg"
codesign --force --timestamp --sign "$DEVID" "$DMG_PATH"
if ! xcrun notarytool submit "$DMG_PATH" --keychain-profile "$NOTARY_PROFILE" --wait >&2; then
  die "dmg notarization failed (run: xcrun notarytool log <id> --keychain-profile $NOTARY_PROFILE)"
fi
xcrun stapler staple "$DMG_PATH" >&2

log "  verifying Gatekeeper acceptance"
spctl --assess --type execute -vv "$APP" >&2 || die "app not accepted by Gatekeeper"
spctl --assess --type open --context context:primary-signature -vv "$DMG_PATH" >&2 \
  || die "dmg not accepted by Gatekeeper"
xcrun stapler validate "$DMG_PATH" >&2

echo "$DMG_PATH"
log "DONE — notarized, stapled: $DMG_PATH"
