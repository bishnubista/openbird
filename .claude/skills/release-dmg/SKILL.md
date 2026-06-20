---
name: release-dmg
description: Build, notarize, and publish an updated OpenBird beta .dmg to GitHub Releases. Use when asked to cut/release a new dmg, ship a beta build, update the notarized macOS app for testers, or "re-release the dmg".
---

# Release the OpenBird beta .dmg

Produces a self-contained, Developer ID-signed, **notarized + stapled** `OpenBird.dmg`
and publishes it to GitHub Releases so beta testers can download → drag to
/Applications → run with no Gatekeeper warning. This is a SEPARATE channel from the
Homebrew (`v0.1.x`) path and must not disturb it.

The whole pipeline lives in `script/package_dmg.sh`; this skill just drives it safely
and publishes the result.

## Prerequisites (verify, don't assume)

- On `main`, synced and clean: `git fetch origin && git checkout main && git pull --ff-only`.
  Releasing builds from `main`, so all merged features are included.
- Exactly one Developer ID identity present (the build auto-derives it):
  `security find-identity -v -p codesigning | grep "Developer ID Application"`
  should print a single `Developer ID Application: <Name> (<TEAMID>)` line. If there
  are several, set `OPENBIRD_SIGN_IDENTITY` (env or `script/release.env`).
- notarytool profile present (default `openbird-notary`, override via
  `OPENBIRD_NOTARY_PROFILE`). If missing, recreate with `xcrun notarytool
  store-credentials <profile>` (key material in `~/openbird-codesign/`,
  iCloud-backed). See `memory/openbird-signing.md`.
- `gh auth status` is authenticated for `bishnubista/openbird`.

## 1. Pick the version

Releases are tagged `beta-dmg-<x.y.z>`. Bump from the current latest:
`gh release list | grep beta-dmg`. Don't reuse a tag; don't silently replace an
asset — a new tag gives testers a version + changelog anchor.

## 2. Build + notarize (from `main`)

```bash
export OPENBIRD_SWIFTPM_DISABLE_SANDBOX=1
unset OPENBIRD_DMG_SKIP_NOTARIZE
./script/package_dmg.sh
```
Run this so it doesn't block the session — it takes several minutes, mostly Apple
notarization — then poll for completion (e.g. start it as a background task).
The signing identity is **auto-derived** from the single "Developer ID Application"
identity in the keychain, and the notary profile defaults to `openbird-notary`.
Override only if needed via env or a gitignored `script/release.env` (see
`script/release.env.example`) — e.g. when multiple Developer ID identities exist.
The script: embeds a relocatable Python, installs `openbird[encryption,integrations]`
(excludes `meetings`), prunes to interpreters, runs the relocation audit + run-from-
moved-copy test, Developer ID-signs every nested Mach-O inside-out, then notarizes +
staples the app AND the dmg, and verifies with `spctl`. It prints the dmg path on
success (`dist/OpenBird.dmg`). First notarization for a brand-new app can take ~40 min;
subsequent ones are minutes.

If it fails: the script surfaces `xcrun notarytool log <id> --keychain-profile
"<your profile>"` (the configured profile — default `openbird-notary`) to pinpoint a
rejected binary. Fix, rebuild — do NOT hand-edit the signed bundle.

## 3. Verify the artifact

```bash
xcrun stapler validate dist/OpenBird.dmg
spctl --assess --type open --context context:primary-signature -vv dist/OpenBird.dmg
# expect: accepted, source=Notarized Developer ID
shasum -a 256 dist/OpenBird.dmg
```

## 4. Publish

Write a short changelog of what merged since the previous tag
(`git log --oneline <prev-tag-or-commit>..main`). Then:

```bash
gh release create beta-dmg-<x.y.z> \
  --target main \
  --title "OpenBird Beta <x.y.z> — notarized .dmg" \
  --notes "<install steps + What's new + 'updating keeps your Accessibility grant' + sha256>" \
  dist/OpenBird.dmg
gh release edit beta-dmg-<x.y.z> --latest    # only if it should be the headline download
```

Notes to include for testers: download `OpenBird.dmg` → open → drag OpenBird to
Applications → launch (no Gatekeeper warning); first launch grant Accessibility, set
the capture allowlist in Setup, have a local Ollama running; updating from a prior
build keeps the Accessibility grant (stable signing identity) — just drag the new app
over the old; the `shasum -a 256` value for integrity.

## Notes / gotchas

- The signing identity is stable, so TCC grants persist across updates — testers
  don't re-grant. Never rotate the Developer ID cert casually.
- Use a non-`v` tag (`beta-dmg-*`) so the `v*` Homebrew release workflow is NOT
  triggered.
- The Homebrew formula pulls a specific tagged asset by URL, so marking a dmg release
  "Latest" does not affect `brew install`.
- `dist/` is gitignored — the 179 MB dmg never gets committed.
- Do not commit secrets; key material stays in `~/openbird-codesign/` (and iCloud).
