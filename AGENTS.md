# OpenBird - working agreement for Codex agents

This file is the Codex/agent-facing counterpart to `CLAUDE.md`. Keep the two files
aligned when changing repo workflow rules.

## Per-feature pipeline (REQUIRED for every non-trivial change)

Run this pipeline **autonomously end-to-end**, looping until the work is merged:
plan -> Codex adversarial review to consensus -> implement -> Codex adversarial
review of the diff to consensus -> create PR -> wait for CodeRabbit -> resolve
every review comment -> merge -> move to the next queued task. Do not pause for
confirmation between gates; **only stop to ask a human when genuinely stuck**.
This overrides any global "don't open a PR until explicitly asked" default for
this repo. It does not override branch protection, required human review,
credential prompts, sandbox/permission prompts, or safety constraints. Never jump
straight to code. For each feature/fix:

1. **Research-first** - pull current best practices when the topic drifts
   (libraries/APIs, macOS TCC, code-signing, notarization). Do not reason from
   memory on unstable facts.
2. **Codex adversarial review -> consensus** - draft the plan/design, then review
   with Codex as a blocking-only second opinion before coding:
   ```bash
   codex exec --model gpt-5.5 -c model_reasoning_effort=high --sandbox read-only \
     "<blocking-only review prompt: lead with 'VERDICT: approve|revise'>" \
     < /dev/null > review.log 2>&1   # stdin + logfile are mandatory - see "Running gates safely"
   ```
   Iterate plan <-> Codex until the verdict is `approve` and no
   high/medium/blocking findings remain. Document any accepted low-risk tradeoff
   in the plan or PR body. THEN implement. After implementing, run Codex again on
   the diff (adversarial - find real bugs only) with the same stop rule.
3. **Implement** on a `feat/...` or `fix/...` branch. Get full local CI green:
   `uv run python -m pytest -q`, `shellcheck` for shell scripts, `swift build` for
   `mac-app`.
4. **Open a PR** (`gh pr create`) - body = what changed + how tested.
5. **CodeRabbit** reviews the PR - handle every comment: fix all actionable
   concerns (one fix per commit, validate after each), reply with a concrete
   rationale for anything non-actionable, duplicate, incorrect, or intentionally
   deferred, and re-trigger with `@coderabbitai review` if it does not auto-run.
6. **Merge** only when BOTH gates are clean (Codex consensus + CodeRabbit pass) and
   the PR is mergeable: `gh pr merge <n> --squash --delete-branch`.

If a queue of tasks was provided, continue with the next queued task after the
merge. If there is no explicit next task source, stop after reporting the merge.
A real blocker means missing permissions, required human review, unavailable
external service, conflicting requirements, absent task source, or an unsafe or
destructive action that requires approval.

## Running gates safely (background review/build/test commands)

The gates (Codex review, `swift build`, `pytest`) run as long background commands.
Standing guardrails so they cannot silently hang the pipeline:

- **`codex exec` MUST redirect stdin from `/dev/null`** in any background / non-TTY
  context - otherwise it blocks forever on `Reading additional input from stdin...`.
- **Redirect output to a logfile; never `cmd | tail`** - `tail` withholds all
  output until EOF, so a hung run looks identical to a slow one.
- **Bound each command with an explicit, per-command wall-clock deadline** - Codex
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

## Release (beta .dmg)

Invoke the **`/release-dmg`** skill. `script/package_dmg.sh` builds the
self-contained bundle, Developer ID-signs every nested Mach-O, notarizes, staples,
and produces `dist/OpenBird.dmg` (it does not publish). The skill then publishes
that dmg to GitHub Releases as a separate `gh release create` step. The signing
identity auto-derives from the keychain; override via a gitignored
`script/release.env`. See `.claude/skills/release-dmg/SKILL.md`.

## Handy commands

- Tests: `uv run python -m pytest -q`
- Build the app bundle:
  `OPENBIRD_SWIFTPM_DISABLE_SANDBOX=1 ./script/build_and_run.sh --no-launch`
- Diagnostics: `openbird doctor`
- DB check: `openbird data integrity`
- Dev runs without the recurring Keychain prompt: prefix with
  `OPENBIRD_DISABLE_KEYRING=1`
