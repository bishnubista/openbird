# OpenBird - working agreement for Codex agents

This file is the Codex/agent-facing counterpart to `CLAUDE.md`. Keep the two files
aligned when changing repo workflow rules.

## Per-feature pipeline (REQUIRED for every non-trivial change)

Run this pipeline **autonomously end-to-end**, looping until the work is merged:
plan -> Claude adversarial review to consensus -> implement -> Claude adversarial
review of the diff to consensus -> create PR -> wait for CodeRabbit -> resolve
every review comment -> merge -> move to the next queued task. Do not pause for
confirmation between gates; **only stop to ask a human when genuinely stuck**.
This autonomy does not override branch protection, required human review,
credential prompts, sandbox/permission prompts, or safety constraints. Never jump
straight to code. For each feature/fix:

1. **Research-first** - pull current best practices when the topic drifts
   (libraries/APIs, macOS TCC, code-signing, notarization). Do not reason from
   memory on unstable facts.
2. **Claude adversarial review -> consensus** - draft the plan/design, then review
   with Claude as a cross-family, blocking-only second opinion before coding:
   ```bash
   claude -p --model opus --effort high \
     "<blocking-only review prompt: lead with 'VERDICT: approve|revise'; review only, do not implement or run the repo pipeline>" \
     < /dev/null > review.log 2>&1   # stdin + logfile are mandatory - see "Running gates safely"
   ```
   Iterate plan <-> Claude until the verdict is `approve` and no
   high/medium/blocking findings remain. Document any accepted low-risk tradeoff
   in the plan or PR body. THEN implement. After implementing, run Claude again on
   the diff (adversarial - find real bugs only) with the same stop rule.
3. **Implement** on a `feat/...` or `fix/...` branch. Get full local CI green:
   `uv run python -m pytest -q`, `shellcheck` for shell scripts, `swift build` for
   `mac-app`.
4. **Open a PR** (`gh pr create`) - body = what changed + how tested.
5. **CodeRabbit** reviews the PR - handle every comment: fix all actionable
   concerns (one fix per commit, validate after each), reply with a concrete
   rationale for anything non-actionable, duplicate, incorrect, or intentionally
   deferred, and re-trigger with `@coderabbitai review` if it does not auto-run.
   - **"Review limit reached" is a transient wait, NOT a blocker.** If CodeRabbit
     posts a rate-limit warning instead of a review ("we couldn't start this
     review because you've reached your PR review rate limit. More reviews will
     be available in N minutes"), do not escalate to a human or abandon the
     merge. Wait out the stated window (add a small buffer), then re-trigger with
     `@coderabbitai review`. Repeat if it is still limited. Only the *clean*
     review counts as the gate passing.
6. **Merge** only when BOTH gates are clean (Claude consensus + CodeRabbit pass) and
   the PR is mergeable: `gh pr merge <n> --squash --delete-branch`.

If a queue of tasks was provided, continue with the next queued task after the
merge. If there is no explicit next task source, stop after reporting the merge.
A real blocker means missing permissions, required human review, unavailable
external service, conflicting requirements, absent task source, or an unsafe or
destructive action that requires approval.

## Running gates safely (background review/build/test commands)

The gates (Claude review, `swift build`, `pytest`) run as long background commands.
Standing guardrails so they cannot silently hang the pipeline:

- **`claude -p` and `codex exec` MUST redirect stdin from `/dev/null`** in any
  background / non-TTY context - otherwise interactive input reads can block
  forever.
- **Redirect output to a logfile; never `cmd | tail`** - `tail` withholds all
  output until EOF, so a hung run looks identical to a slow one.
- **Bound each command with an explicit, per-command wall-clock deadline** - Claude
  review, Swift build, and pytest get different values, not one magic number. A
  deadline beats a poller: it is scoped to the command and self-terminates a hang.
  Use `timeout` / `gtimeout` from GNU coreutils (`brew install coreutils` - already
  installed in this dev env) and prefer `--signal=KILL` after a grace period so it
  reaps the whole process group, not just the parent PID.
- On a killed/timed-out gate, **cap retries and log why each retry happened** -
  never retry endlessly.
- **Distinguish hangs from blockers:** credential/permission prompts, merge
  conflicts, and required human review are BLOCKERS to escalate - never
  kill-and-retry them.
- CPU-time delta + log growth are **advisory diagnostics** (extend vs. retry), not
  a liveness policy - an API-bound Codex wait is healthy with low CPU and no log
  growth.

**Optional active supervisor - long unattended runs only.** For a long, unattended
autonomous run you MAY start an active supervisor (e.g. a short-interval loop) to
watch the gates. Scope it tightly or do not run it: only for a NAMED unattended
run, started and stopped by that run, with a visible PID/log, a max lifetime, and
a max retry count. It supervises ONLY the commands it launched or was explicitly
handed - it must NEVER scan-and-kill arbitrary processes on the host - and it has
NO authority to merge or perform destructive cleanup. Do not make it mandatory
for routine changes: harness task-completion notifications plus per-command
deadlines already cover those.

## Hard rules

- **NEVER push directly to `main`.** Always a reviewed PR; merging a reviewed PR is
  allowed once it passes the pipeline.
- **Sync before planning:** `git fetch origin && git log --oneline origin/main -20`,
  and branch off `origin/main`.
- **Stage files by name**, not `git add -A`. **Never commit secrets** - use
  placeholders in docs.
- **Observability / privacy:** privacy-safe structured logging only - reason codes,
  metadata, counts; **never** captured text, window titles, or URLs.
- **Secrets** stay in the macOS Keychain / `~/openbird-codesign/` (not the repo).
  The signing identity and notary profile name are config, not secrets.

## Release

A full version release spans **both channels** (notarized dmg + Homebrew formula +
cask) sharing one version. The ordered runbook lives in
`.claude/skills/release/SKILL.md` (mirrored at `.agents/skills/release/SKILL.md`) —
it sequences bump → dmg → cask → `v*` tag → verify and documents *why* the order is
fixed. The pieces:

- **Version bump** — `script/bump_version.sh <x.y.z>` edits `pyproject.toml` AND
  re-locks `uv.lock` in one step (`pyproject.toml` is the single source of truth; the
  lock pins the project's own version too). It refuses a dirty tree. Never bump the two
  by hand separately — the CI **`version-consistency`** job fails if they disagree.
- **Notarized .dmg** — see `.claude/skills/release-dmg/SKILL.md`.
  `script/package_dmg.sh` builds the self-contained bundle, Developer ID-signs every
  nested Mach-O, notarizes, staples, and produces `dist/OpenBird.dmg` (it does not
  publish); publishing to GitHub Releases (tag `beta-dmg-<x.y.z>`, pinned Latest) is a
  separate step. The signing identity auto-derives from the keychain; override via a
  gitignored `script/release.env`.
- **Homebrew** — the cask (`Casks/openbird.rb`, notarized app) is bumped to the
  published dmg's sha256; the formula (`Formula/openbird.rb`, CLI source) is bumped by
  pushing a `v<x.y.z>` tag, which triggers
  `.github/workflows/homebrew-release.yml` to build the source tarball, auto-generate
  release notes (`scripts/gen_changelog.py`), validate the formula edit inline, and open
  a formula-bump PR. The CI **`homebrew-style`** job lints both packaging files.

## Handy commands

- Tests: `uv run python -m pytest -q`
- Release: bump a version with `script/bump_version.sh <x.y.z>` (atomic pyproject +
  uv.lock); full flow in `.claude/skills/release/SKILL.md`.
- Changelog from history: `scripts/gen_changelog.py --from <prev-tag> --to <tag-or-HEAD>`
  (`--from beta-dmg-<prev>` for the dmg channel, `--from v<prev>` for the source channel).
- Build the app bundle:
  `OPENBIRD_SWIFTPM_DISABLE_SANDBOX=1 ./script/build_and_run.sh --no-launch`
- Diagnostics: `openbird doctor`
- DB check: `openbird data integrity`
- Dev runs without the recurring Keychain prompt: prefix with
  `OPENBIRD_DISABLE_KEYRING=1`
