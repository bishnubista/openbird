#!/usr/bin/env bash
#
# Atomically bump the OpenBird version.
#
# pyproject.toml is the single source of truth for the version (see the README
# "Versioning" section), but uv.lock independently pins the project's OWN version
# as an editable-package entry. A bump that edits only pyproject.toml leaves the
# lock's self-version stale — exactly the inconsistency that slipped onto main in
# PR #112 and had to be cleaned up in a follow-up (#113). This script moves both
# in one shot so the bump is complete by construction.
#
# It deliberately does NOT touch Casks/openbird.rb or Formula/openbird.rb: those
# pin published release artifacts (a dmg / a source tarball) by URL + sha256 and
# are bumped by their own release flows once the artifact exists.
#
# Usage:
#   script/bump_version.sh <x.y.z>
#
set -euo pipefail

usage() {
  echo "usage: $(basename "$0") <x.y.z>" >&2
  exit 2
}

[ "$#" -eq 1 ] || usage
new_version="$1"

# Strict x.y.z only. pyproject.toml must be a PEP 440 version (uv rejects a
# semver-style `-suffix`), and every OpenBird release to date is plain x.y.z, so
# we keep the surface small rather than mirror the looser v-tag grammar.
if ! printf '%s' "$new_version" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+$'; then
  echo "error: '$new_version' is not a valid x.y.z version" >&2
  exit 1
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# Require pyproject.toml and uv.lock to be clean before we touch them. This keeps
# the bump isolated (no unrelated edits ride along in the commit) AND makes the
# restore-on-error trap below safe: it reverts to the committed state, so it can
# never discard pre-existing unstaged work in these files.
if ! git diff --quiet -- pyproject.toml uv.lock \
  || ! git diff --cached --quiet -- pyproject.toml uv.lock; then
  echo "error: pyproject.toml / uv.lock have uncommitted changes." >&2
  echo "       commit or stash them before bumping the version." >&2
  exit 1
fi

current_version="$(awk -F'"' '/^version = / {print $2; exit}' pyproject.toml)"
if [ -z "$current_version" ]; then
  echo "error: could not read current version from pyproject.toml" >&2
  exit 1
fi
if [ "$current_version" = "$new_version" ]; then
  echo "pyproject.toml is already at $new_version; nothing to do."
  exit 0
fi

echo "bumping $current_version -> $new_version"

# If anything below fails after we start editing, restore the two files we own so
# the working tree is never left half-bumped (e.g. uv lock fails on a network
# blip). The clean-state check above guarantees this reverts to the committed
# version — it cannot discard unstaged work.
restore_on_error() {
  echo "error: bump failed; restoring pyproject.toml and uv.lock" >&2
  git checkout -- pyproject.toml uv.lock 2>/dev/null || true
}
trap restore_on_error ERR

# 1) pyproject.toml — replace the first `version = "..."` line (the [project]
#    version). The `done` guard keeps us from touching any later version field.
tmp_file="$(mktemp)"
awk -v v="$new_version" '
  !done && /^version = "/ { sub(/"[^"]*"/, "\"" v "\""); done = 1 }
  { print }
' pyproject.toml > "$tmp_file"
mv "$tmp_file" pyproject.toml

# 2) uv.lock — re-lock so the openbird self-entry tracks the new version. `uv lock`
#    keeps every existing dependency pin, so the only change is the self-version
#    (plus any genuinely required re-resolution, which the diff below surfaces).
uv lock --quiet

# The version-bearing files are now consistently updated; from here on a
# non-zero exit is informational, not a reason to roll back.
trap - ERR

echo
echo "changed files:"
git --no-pager diff --stat -- pyproject.toml uv.lock || true

# 3) Surface any stray references to the OLD version that this script did not
#    rewrite, so the caller can decide whether they are intentional. Cask/Formula
#    and the lockfiles are excluded on purpose (handled by release flows).
echo
echo "remaining references to '$current_version' (review — cask/formula excluded by design):"
if git grep -n -F "$current_version" -- . \
    ':(exclude)uv.lock' ':(exclude)pyproject.toml' \
    ':(exclude)Casks/*' ':(exclude)Formula/*'; then
  echo "  ^ verify the matches above are not release-version references that also need bumping"
else
  echo "  (none)"
fi

echo
echo "done. Next: review the diff, commit pyproject.toml + uv.lock together, open a PR."
