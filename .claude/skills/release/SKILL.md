---
name: release
description: Orchestrate a FULL OpenBird release of a new version x.y.z across both distribution channels — the notarized beta .dmg AND the Homebrew formula + cask. Use when asked to "cut a release", "ship version x.y.z", "do a full release", or "release everything". For ONLY the notarized app, use release-dmg instead.
---

<!-- Keep this release skill mirrored across .claude/skills and .agents/skills. -->

# Cut a full OpenBird release (x.y.z)

OpenBird ships through **two channels that share one version** (`pyproject.toml` is the
single source of truth):

- **Notarized beta `.dmg`** — Developer ID-signed + notarized app for testers. Tag
  `beta-dmg-<x.y.z>`. This is the **Latest** GitHub release. Driven by the
  **`release-dmg`** skill.
- **Homebrew** — the `openbird` CLI **formula** (source tarball, tag `v<x.y.z>`) and the
  `openbird` **cask** (points at the notarized dmg).

This skill is the **ordered runbook** that drives all of it. The ordering is not
cosmetic — each step depends on an artifact the previous one produced. Run it
autonomously end-to-end per the repo working agreement (plan → Codex → implement → PR →
CodeRabbit → merge for each sub-change); only stop for a genuine blocker.

## Why the order is fixed (read before starting)

1. **Bump must land on `main` first.** The dmg builds *from `main`*, and CI fails any
   `beta-dmg-*` / `v*` tag whose commit's `pyproject.toml` disagrees with the tag. So the
   version bump PR must be merged before building or tagging.
2. **The cask needs the dmg's sha256**, which only exists after the dmg is built and
   published. So cask bump comes after the dmg.
3. **The formula tarball is built by the `v*` tag**, so tagging is what produces it; tag
   only after the bump is on `main`.

## Preconditions (verify, don't assume)

- On `main`, synced, clean: `git fetch origin && git checkout main && git pull --ff-only`.
- `gh auth status` authenticated for `bishnubista/openbird`.
- Signing identity + notary profile present — the `release-dmg` skill checks these; defer
  to it rather than re-deriving here.
- Pick the target `x.y.z` and confirm it is **not already released**:
  `git ls-remote --tags origin 'v<x.y.z>' 'beta-dmg-<x.y.z>'` and `gh release list` must
  not show it. If they do, the version is taken — bump to the next number.

## The pipeline

### 1. Atomic version bump → PR → merge
```bash
script/bump_version.sh <x.y.z>        # edits pyproject.toml AND re-locks uv.lock
```
Commit **both** files, open a PR, and merge once green. The `Version consistency` CI job
asserts `pyproject.toml` and `uv.lock` agree — do not bump them by hand and skip the
script, or you reintroduce the #112/#113 split. Re-sync `main` after merge.

### 2. Build + publish the notarized dmg
Invoke the **`release-dmg`** skill. It builds from `main`, notarizes + staples, publishes
`beta-dmg-<x.y.z>`, and marks it **Latest**. Record the published dmg's `sha256` (the skill
prints it; or `shasum -a 256` the asset downloaded from the release).

### 3. Bump the cask → PR → merge
Pin the cask to the **published** dmg (not the local build — hash what `brew` will fetch):
```bash
gh release download beta-dmg-<x.y.z> -R bishnubista/openbird -p OpenBird.dmg -D /tmp --clobber
shasum -a 256 /tmp/OpenBird.dmg      # this is the cask sha256
```
Edit `Casks/openbird.rb`: set `version "<x.y.z>"` and `sha256 "<hash>"` (the `url`
interpolates `version`, so it resolves to `beta-dmg-<x.y.z>/OpenBird.dmg`). Validate with
`brew style Casks/openbird.rb`, open a PR, merge once green (the `Homebrew style` job gates
it). Do NOT touch `Formula/openbird.rb` here.

### 4. Cut the source release / formula (tag-driven)
```bash
git tag -a v<x.y.z> -m "OpenBird v<x.y.z> (Homebrew source release)" <main-HEAD>
git push origin v<x.y.z>
```
This triggers `homebrew-release.yml`, which: builds `openbird-<x.y.z>.tar.gz`, creates the
`v<x.y.z>` release with **auto-generated notes** (`scripts/gen_changelog.py`, `--latest=false`
so it never displaces the dmg), validates the formula edit inline, and opens a **formula-bump
PR**. Watch the run (`gh run watch`), then drive that PR to merge like any other.

### 5. Verify the release is complete
```bash
gh release list --limit 5      # beta-dmg-<x.y.z> == Latest; v<x.y.z> present, NOT latest
grep -E '^\s*version' Casks/openbird.rb
grep -E '^\s*url' Formula/openbird.rb           # both pin <x.y.z>
```
Confirm `pyproject.toml`, `uv.lock`, the cask, and the formula URL all read `<x.y.z>`, and
that `beta-dmg-<x.y.z>` is Latest.

## Notes for tester-facing release notes
The dmg release notes are written by the `release-dmg` skill. For a richer changelog on
either release, generate it from history:
```bash
scripts/gen_changelog.py --from <prev-tag> --to <this-tag>   # or ..HEAD
```
(`--from beta-dmg-<prev>` for the dmg channel, `--from v<prev>` for the source channel.)

## Running gates safely
All long gates (Codex review, `swift build`, `pytest`, notarization, `gh run/pr watch`) run
as bounded background commands with explicit per-command deadlines and logfiles — never
`cmd | tail`. See the "Running gates safely" section of the repo `CLAUDE.md`.

## Real blockers (escalate, don't retry)
Missing signing identity / notary credentials, a notarization rejection (fix + rebuild —
never hand-edit a signed bundle), branch protection / required human review, or a version
already released. Everything else: drive to merge.
