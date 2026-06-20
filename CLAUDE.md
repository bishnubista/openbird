# OpenBird — working agreement for Claude (and contributors)

Claude Code auto-loads this file for every session in this repo. It is the portable
source of truth for the review pipeline and conventions this project requires.
(Per-developer fast recall may also live in Claude memory, but THIS committed file is
authoritative.)

## Per-feature pipeline (REQUIRED for every non-trivial change)

Never jump straight to code. For each feature/fix:

1. **Research-first** — pull current best practices (Context7 for libraries/APIs; web for
   macOS TCC / code-signing / notarization). Don't reason from memory on things that drift.
2. **Codex adversarial review → consensus** — draft the plan/design, then review with Codex
   as a cross-family second opinion before coding:
   ```bash
   codex exec --model gpt-5.5 -c model_reasoning_effort=high --sandbox read-only \
     "<blocking-only review prompt: lead with 'VERDICT: approve|revise'>"
   ```
   Iterate plan ↔ Codex to consensus, THEN implement. After implementing, run Codex again
   on the diff (adversarial — find real bugs only).
3. **Implement** on a `feat/...` or `fix/...` branch. Get full local CI green:
   `uv run python -m pytest -q`, `shellcheck` for shell scripts, `swift build` for `mac-app`.
4. **Open a PR** (`gh pr create`) — body = what changed + how tested.
5. **CodeRabbit** reviews the PR — fix **ALL** concerns (one fix per commit, validate after
   each). Re-trigger with `@coderabbitai review` if it doesn't auto-run.
6. **Merge** only when BOTH gates are clean (Codex consensus + CodeRabbit pass) and the PR is
   mergeable: `gh pr merge <n> --squash --delete-branch`.

The two gates are different lenses — Codex hunts correctness/logic; CodeRabbit catches
conventions, edge cases, maintainability. Passing both beats either alone. Real examples
this caught: a relocation bug notarization couldn't see, chat text leaking into argv, an
ambiguous signing-identity selection, a leaked-reader deadlock, a weak test.

## Hard rules

- **NEVER push directly to `main`.** Always a reviewed PR. (A pre-commit hook enforces being
  on a branch.) Merging a reviewed PR is allowed once it passes the pipeline.
- **Sync before planning:** `git fetch origin && git log --oneline origin/main -20`, and branch
  off `origin/main` — the local checkout can be stale (this once caused a whole duplicated PR).
- **Stage files by name**, not `git add -A`. The secret-scan pre-commit hook must pass; use
  placeholders in docs.
- **Observability / privacy:** privacy-safe structured logging only — reason codes, metadata,
  counts; **never** captured text, window titles, or URLs.
- **Secrets** stay in the macOS Keychain / `~/openbird-codesign/` (not the repo). The signing
  identity and notary profile NAME are config, not secrets.

## Release (beta .dmg)

Invoke the **`/release-dmg`** skill — it drives `script/package_dmg.sh` (self-contained
embedded Python → Developer ID sign every nested Mach-O → notarize → staple → publish to
GitHub Releases). The signing identity auto-derives from the keychain; override via a
gitignored `script/release.env`. See `.claude/skills/release-dmg/SKILL.md`.

## Handy commands

- Tests: `uv run python -m pytest -q`
- Build the app bundle: `OPENBIRD_SWIFTPM_DISABLE_SANDBOX=1 ./script/build_and_run.sh --no-launch`
- Diagnostics: `openbird doctor` · DB check: `openbird data integrity`
- Dev runs without the recurring Keychain prompt: prefix with `OPENBIRD_DISABLE_KEYRING=1`
  (each `uv run` is a different unsigned interpreter, so the DB-key ACL never matches; the
  signed `.app` is unaffected).
