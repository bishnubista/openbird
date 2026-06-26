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
#   script/release_status.sh
#
# Exit codes:
#   0  aligned, or a release legitimately in progress (nothing published yet to lag)
#   1  drift: a published artifact's packaging file was left on an older version
#   2  usage / internal error
#
set -euo pipefail

usage() {
  echo "usage: $(basename "$0")" >&2
  exit 2
}

[ "$#" -eq 0 ] || usage

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
gh_ok=0
dmg_published="unknown"
src_published="unknown"

if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
  gh_ok=1
  if gh release view "$dmg_tag" --repo "$canonical_repo" >/dev/null 2>&1; then dmg_published="yes"; else dmg_published="no"; fi
  if gh release view "$src_tag" --repo "$canonical_repo" >/dev/null 2>&1; then src_published="yes"; else src_published="no"; fi
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

echo
if [ "$drift" -eq 1 ]; then
  echo "=> DRIFT: a published release shipped but its packaging file was left behind."
  echo "   Finish the release (see .claude/skills/release/SKILL.md) so the cask/formula"
  echo "   point at the published $pyproject_v artifacts."
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
