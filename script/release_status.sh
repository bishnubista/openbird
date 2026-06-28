#!/usr/bin/env bash
#
# Report version alignment across the OpenBird release artifacts.
#
# OpenBird ships through two channels that SHARE one version (pyproject.toml is the
# single source of truth):
#   - the notarized beta .dmg  (GitHub release `beta-dmg-<x.y.z>`, pinned by the cask)
#   - Homebrew                 (cask -> the dmg; formula -> the `v<x.y.z>` source tarball)
#
# Because the release is a multi-PR pipeline (bump -> dmg -> cask -> formula), `main`
# can sit half-finished: pyproject advanced, the dmg published, but the cask/formula
# left pointing at the previous version. CI's `Version consistency` job only guards
# pyproject <-> uv.lock, so nothing else notices. This script does — and it does so
# WITHOUT false-flagging the legitimate transient window mid-release.
#
# The distinction it draws:
#   - pyproject ahead of the cask, but NO `beta-dmg-<pyproject>` release exists yet
#       => the dmg simply has not been built. That is the expected mid-release state.
#         Reported as "pending", exit 0.
#   - pyproject ahead of the cask, AND `beta-dmg-<pyproject>` IS published
#       => the dmg shipped but the cask bump was forgotten. That is real DRIFT.
#         Reported as "DRIFT", exit 1.
# The same logic applies to the formula vs. the `v<pyproject>` source release.
#
# GitHub queries (release existence) need an authenticated `gh`. When `gh` is
# unavailable the channel-completion checks are skipped (and clearly marked so),
# the local pyproject <-> uv.lock check still runs, and the script does NOT report
# drift it could not verify.
#
# Usage:
#   script/release_status.sh [--beta-rehearsal]
#
# Exit codes:
#   0  aligned, or a release legitimately in progress (nothing published yet to lag)
#   1  drift: a published artifact's packaging file was left on an older version
#   2  usage / internal error
#
set -euo pipefail

usage() {
  echo "usage: $(basename "$0") [--beta-rehearsal]" >&2
  exit 2
}

beta_rehearsal=0
case "$#" in
  0) ;;
  1)
    case "$1" in
      --beta-rehearsal) beta_rehearsal=1 ;;
      *) usage ;;
    esac
    ;;
  *) usage ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# --- read the local artifacts ------------------------------------------------

# pyproject.toml: first `version = "..."` line is the [project] version.
pyproject_v="$(awk -F'"' '/^version = / {print $2; exit}' pyproject.toml)"
# uv.lock: the project's own editable entry pins its version a line below its name.
# The trailing `|| true` matters: under `set -euo pipefail` a no-match `grep` exits
# non-zero, which would kill the script on the assignment BEFORE the empty-check
# below can report a clean "could not read version" / exit 2. `|| true` lets the
# substitution yield empty so the handler downstream owns the error path. (The
# awk-only extractions below exit 0 even on no match, so they need no guard.)
lock_v="$(grep -A1 '^name = "openbird"$' uv.lock \
  | awk -F'"' '/^version = / {print $2; exit}' || true)"
# Casks/openbird.rb: `version "<x.y.z>"`.
cask_v="$(awk -F'"' '/version "/ {print $2; exit}' Casks/openbird.rb)"
# Formula/openbird.rb: version lives in the source-tarball URL, .../download/v<x.y.z>/...
formula_v="$(grep -Eo 'releases/download/v[0-9]+\.[0-9]+\.[0-9]+/' Formula/openbird.rb \
  | head -n1 | grep -Eo '[0-9]+\.[0-9]+\.[0-9]+' || true)"

for pair in "pyproject:$pyproject_v" "uv.lock:$lock_v" "cask:$cask_v" "formula:$formula_v"; do
  if [ -z "${pair#*:}" ]; then
    echo "error: could not read version for '${pair%%:*}'" >&2
    exit 2
  fi
done

# --- probe the published GitHub releases (best-effort) -----------------------

# Releases live on the canonical repo (the cask URL is pinned to it too). Query it
# explicitly with --repo: an inferred repo would, on a fork, report an
# already-published upstream release as missing and mislabel real drift as pending.
canonical_repo="bishnubista/openbird"
dmg_tag="beta-dmg-$pyproject_v"
src_tag="v$pyproject_v"
dmg_url="https://github.com/$canonical_repo/releases/download/$dmg_tag/OpenBird.dmg"
src_url="https://github.com/$canonical_repo/releases/download/$src_tag/openbird-$pyproject_v.tar.gz"
gh_ok=0
dmg_published="unknown"
src_published="unknown"
repo_visibility="unknown"

if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
  gh_ok=1
  if gh release view "$dmg_tag" --repo "$canonical_repo" >/dev/null 2>&1; then dmg_published="yes"; else dmg_published="no"; fi
  if gh release view "$src_tag" --repo "$canonical_repo" >/dev/null 2>&1; then src_published="yes"; else src_published="no"; fi
  repo_visibility="$(gh repo view "$canonical_repo" --json visibility --jq '.visibility' 2>/dev/null || true)"
  [ -n "$repo_visibility" ] || repo_visibility="unknown"
fi

# --- evaluate each row -------------------------------------------------------
#
# Classify each packaging file against pyproject, given whether the artifact it
# pins has been published. This runs in the PARENT shell (no function whose output
# is captured with `$(...)`), so the drift/pending flags actually stick — and it
# stays portable to stock macOS bash 3.2 (no namerefs). Each `classify` echoes the
# row's status and, as a side effect, may set `drift=1` / `pending=1`.

drift=0
pending=0
unverified=0

# classify <file_version> <published:yes|no|unknown> -> echoes status string.
# A status line is informational only; the drift/pending vars are set in-line
# below where the comparison happens, so the flags survive into the verdict.
status_for() {
  local file_v="$1" published="$2"
  if [ "$file_v" = "$pyproject_v" ]; then
    echo "ok"; return
  fi
  case "$published" in
    yes)     echo "DRIFT (published $pyproject_v but pins $file_v)" ;;
    no)      echo "pending ($pyproject_v not built yet)" ;;
    *)       echo "unverified (gh unavailable)" ;;
  esac
}

# flags are set here, in the parent shell, so they reach the final verdict.
set_flags() {
  local file_v="$1" published="$2"
  [ "$file_v" = "$pyproject_v" ] && return
  case "$published" in
    yes) drift=1 ;;
    no)  pending=1 ;;
    *)   unverified=1 ;;  # gh unavailable AND this file differs from pyproject:
                          # cannot confirm whether a published release was left behind.
  esac
}

lock_status="ok"
if [ "$lock_v" != "$pyproject_v" ]; then
  lock_status="DRIFT (uv.lock disagrees with pyproject)"
  drift=1
fi
cask_status="$(status_for "$cask_v" "$dmg_published")"
set_flags "$cask_v" "$dmg_published"
formula_status="$(status_for "$formula_v" "$src_published")"
set_flags "$formula_v" "$src_published"

# --- optional beta rehearsal probes -----------------------------------------

http_status_code() {
  local url="$1" code rc
  if ! command -v curl >/dev/null 2>&1; then
    echo "unknown:no-curl"
    return
  fi
  rc=0
  code="$(curl -sS -L -o /dev/null -w '%{http_code}' --head --max-time 8 "$url" 2>/dev/null)" || rc=$?
  if [ "$rc" -ne 0 ] || [ -z "$code" ] || [ "$code" = "000" ]; then
    echo "unknown:http-000"
    return
  fi
  echo "$code"
}

is_http_success() {
  case "$1" in
    2*|3*) return 0 ;;
    *) return 1 ;;
  esac
}

is_http_client_failure() {
  case "$1" in
    4*) return 0 ;;
    *) return 1 ;;
  esac
}

public_download_status() {
  local published="$1" code="$2"
  if [ "$published" != "yes" ]; then
    echo "unknown (release not published)"
    return
  fi
  case "$code" in
    unknown:*) echo "unknown (${code#unknown:})"; return ;;
  esac
  if is_http_success "$code"; then
    echo "ok (HTTP $code)"
    return
  fi
  if [ "$repo_visibility" = "PRIVATE" ] && is_http_client_failure "$code"; then
    echo "private_expected (HTTP $code; token-gated release)"
    return
  fi
  if [ "$repo_visibility" = "PUBLIC" ] && is_http_client_failure "$code"; then
    echo "blocked (HTTP $code)"
    return
  fi
  echo "unknown (HTTP $code)"
}

resolve_link() {
  local path="$1" target dir
  while [ -L "$path" ]; do
    target="$(readlink "$path" 2>/dev/null || true)"
    [ -n "$target" ] || break
    case "$target" in
      /*) path="$target" ;;
      *)
        dir="$(cd "$(dirname "$path")" 2>/dev/null && pwd -P || true)"
        [ -n "$dir" ] || break
        path="$dir/$target"
        ;;
    esac
  done
  echo "$path"
}

brew_version() {
  local kind="$1"
  if ! command -v brew >/dev/null 2>&1; then
    echo "unavailable"
    return
  fi
  case "$kind" in
    cask) brew list --cask --versions openbird 2>/dev/null | awk '{print $2; exit}' ;;
    formula) brew list --versions openbird 2>/dev/null | awk '{print $2; exit}' ;;
  esac
}

beta_blocker=0
beta_blocker_reason=""

add_beta_blocker() {
  local reason="$1"
  beta_blocker=1
  if [ -n "$beta_blocker_reason" ]; then
    beta_blocker_reason="$beta_blocker_reason; $reason"
  else
    beta_blocker_reason="$reason"
  fi
}

bundle_signing_metadata() {
  local bundle="$1" out rc stapler_rc team signature notarization reason
  rc=0
  team="unknown"
  signature="unknown"
  notarization="unknown"
  reason="unknown signing status"

  if [ ! -d "$bundle" ]; then
    echo "missing|not-set|unknown|missing app bundle"
    return
  fi
  if ! command -v codesign >/dev/null 2>&1; then
    echo "unknown|unknown|unknown|codesign unavailable"
    return
  fi

  out="$(codesign -dv --verbose=4 "$bundle" 2>&1)" || rc=$?
  team="$(printf '%s\n' "$out" | awk -F= '/^TeamIdentifier=/ {print $2; exit}')"
  [ -n "$team" ] || team="not-set"

  if [ "$rc" -ne 0 ]; then
    if printf '%s\n' "$out" | grep -qi 'not signed'; then
      signature="unsigned"
      reason="unsigned app bundle"
    else
      signature="unknown"
      reason="codesign failed"
    fi
  elif printf '%s\n' "$out" | grep -q '^Signature=adhoc'; then
    signature="adhoc"
    reason="adhoc identity"
  elif printf '%s\n' "$out" | grep -q '^Authority=Developer ID Application'; then
    signature="developer_id"
    reason="ok"
  elif printf '%s\n' "$out" | grep -q '^Authority='; then
    signature="other_signed"
    reason="non-Developer ID signature"
  else
    signature="unknown"
    reason="no signing authority"
  fi

  if [ "$signature" = "developer_id" ]; then
    if command -v xcrun >/dev/null 2>&1; then
      stapler_rc=0
      xcrun stapler validate "$bundle" >/dev/null 2>&1 || stapler_rc=$?
      if [ "$stapler_rc" -eq 0 ]; then
        notarization="stapled"
      else
        notarization="absent"
        reason="notarization not stapled"
      fi
    else
      notarization="unknown"
      reason="stapler unavailable"
    fi
  fi

  echo "$signature|$team|$notarization|$reason"
}

if [ "$beta_rehearsal" -eq 1 ]; then
  dmg_http_code="$(http_status_code "$dmg_url")"
  src_http_code="$(http_status_code "$src_url")"
  dmg_public_status="$(public_download_status "$dmg_published" "$dmg_http_code")"
  src_public_status="$(public_download_status "$src_published" "$src_http_code")"
  case "$dmg_public_status" in blocked*) add_beta_blocker "DMG public download $dmg_public_status" ;; esac
  case "$src_public_status" in blocked*) add_beta_blocker "source public download $src_public_status" ;; esac

  beta_app_bundle="${OPENBIRD_RELEASE_STATUS_APP_BUNDLE:-/Applications/OpenBird.app}"
  installed_app_v="missing"
  if [ -f "$beta_app_bundle/Contents/Info.plist" ]; then
    installed_app_v="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$beta_app_bundle/Contents/Info.plist" 2>/dev/null || echo "unknown")"
  fi
  signing_metadata="$(bundle_signing_metadata "$beta_app_bundle")"
  app_signature="${signing_metadata%%|*}"
  signing_metadata_rest="${signing_metadata#*|}"
  app_team_id="${signing_metadata_rest%%|*}"
  signing_metadata_rest="${signing_metadata_rest#*|}"
  app_notarization="${signing_metadata_rest%%|*}"
  app_signing_reason="${signing_metadata_rest#*|}"
  if [ "$app_signature" != "developer_id" ]; then
    add_beta_blocker "installed app signing blocker: $beta_app_bundle uses $app_signature ($app_signing_reason)"
  elif [ "$app_notarization" != "stapled" ]; then
    add_beta_blocker "installed app signing blocker: $beta_app_bundle uses $app_signature ($app_signing_reason)"
  fi
  if [ "$installed_app_v" != "missing" ] && [ "$installed_app_v" != "$pyproject_v" ] && [ "$dmg_published" = "yes" ]; then
    add_beta_blocker "installed app version blocker: $beta_app_bundle is $installed_app_v but published target is $pyproject_v"
  fi
  brew_cask_v="$(brew_version cask || true)"
  [ -n "$brew_cask_v" ] || brew_cask_v="not-installed"
  brew_formula_v="$(brew_version formula || true)"
  [ -n "$brew_formula_v" ] || brew_formula_v="not-installed"
  path_openbird="$(command -v openbird 2>/dev/null || true)"
  [ -n "$path_openbird" ] || path_openbird="missing"
  path_target="-"
  path_status="missing"
  if [ "$path_openbird" != "missing" ]; then
    path_target="$(resolve_link "$path_openbird")"
    if [ "$path_target" = "$beta_app_bundle/Contents/MacOS/openbird-cli" ]; then
      path_status="bundled app CLI"
    else
      path_status="other target"
    fi
  fi
  homebrew_token_status="absent"
  if [ -n "${HOMEBREW_GITHUB_API_TOKEN:-}" ]; then
    homebrew_token_status="present"
  fi
fi

# --- report ------------------------------------------------------------------

printf '\nOpenBird release alignment (source of truth: pyproject.toml = %s)\n\n' "$pyproject_v"
printf '  %-22s %-8s %s\n' "artifact" "version" "status"
printf '  %-22s %-8s %s\n' "----------------------" "--------" "------"
printf '  %-22s %-8s %s\n' "pyproject.toml"          "$pyproject_v" "(source of truth)"
printf '  %-22s %-8s %s\n' "uv.lock"                 "$lock_v"      "$lock_status"
printf '  %-22s %-8s %s\n' "Casks/openbird.rb"       "$cask_v"      "$cask_status"
printf '  %-22s %-8s %s\n' "Formula/openbird.rb"     "$formula_v"   "$formula_status"
if [ "$gh_ok" -eq 1 ]; then
  printf '  %-22s %-8s %s\n' "release $dmg_tag" "-" "published=$dmg_published"
  printf '  %-22s %-8s %s\n' "release $src_tag" "-" "published=$src_published"
else
  printf '\n  (gh unavailable or unauthenticated: channel-completion checks skipped)\n'
fi

if [ "$beta_rehearsal" -eq 1 ]; then
  printf '\nBeta rehearsal distribution checks (advisory unless marked blocked)\n\n'
  printf '  %-26s %s\n' "repo visibility" "$repo_visibility"
  printf '  %-26s %s\n' "Homebrew GitHub token" "$homebrew_token_status"
  printf '  %-26s %s\n' "DMG public download" "$dmg_public_status"
  printf '  %-26s %s\n' "source public download" "$src_public_status"
  printf '\nBeta rehearsal local install checks (blocking where marked; this machine only)\n\n'
  printf '  %-26s %s\n' "installed app" "$installed_app_v ($beta_app_bundle)"
  printf '  %-26s %s\n' "app signature" "$app_signature ($app_signing_reason)"
  printf '  %-26s %s\n' "app team id" "$app_team_id"
  printf '  %-26s %s\n' "app notarization" "$app_notarization"
  printf '  %-26s %s\n' "brew cask receipt" "$brew_cask_v"
  printf '  %-26s %s\n' "brew formula receipt" "$brew_formula_v"
  printf '  %-26s %s\n' "PATH openbird" "$path_openbird"
  printf '  %-26s %s\n' "PATH target" "$path_target"
  printf '  %-26s %s\n' "PATH status" "$path_status"
fi

echo
if [ "$drift" -eq 1 ]; then
  echo "=> DRIFT: a published release shipped but its packaging file was left behind."
  echo "   Finish the release (see .claude/skills/release/SKILL.md) so the cask/formula"
  echo "   point at the published $pyproject_v artifacts."
  exit 1
fi
if [ "$beta_rehearsal" -eq 1 ] && [ "$beta_blocker" -eq 1 ]; then
  echo "=> beta blocker: $beta_blocker_reason"
  echo "   Fix the release/install evidence before treating this as public-beta ready."
  exit 1
fi
if [ "$pending" -eq 1 ]; then
  echo "=> in progress: pyproject is ahead, but the lagging artifact is not published yet."
  echo "   This is the expected mid-release state; no drift."
  exit 0
fi
if [ "$unverified" -eq 1 ]; then
  echo "=> unverified: a packaging file differs from pyproject, but gh is unavailable so"
  echo "   whether the $pyproject_v release was published (drift) or not (pending) is unknown."
  echo "   Re-run with an authenticated gh to resolve. NOT asserting aligned."
  exit 0
fi
echo "=> aligned: every channel is on $pyproject_v."
exit 0
