# OpenBird — working agreement for Claude (and contributors)

Claude Code auto-loads this file for every session in this repo. It is the portable
source of truth for the review pipeline and conventions this project requires.
(Per-developer fast recall may also live in Claude memory, but THIS committed file is
authoritative.)

## Per-feature pipeline (REQUIRED for every non-trivial change)

Run this pipeline **autonomously end-to-end**, looping until the work is merged:
plan → Codex adversarial review to consensus → implement → Codex adversarial
review of the diff to consensus → create PR → wait for CodeRabbit → resolve every
review comment → merge → move to the next queued task. Do not pause for
confirmation between gates; **only stop to ask a human when genuinely stuck**.
This overrides any global "don't open a PR until explicitly asked" default for
this repo. It does not override branch protection, required human review,
credential prompts, sandbox/permission prompts, or safety constraints. Never jump
straight to code. For each feature/fix:

1. **Research-first** — pull current best practices (Context7 for libraries/APIs; web for
   macOS TCC / code-signing / notarization). Don't reason from memory on things that drift.
2. **Codex adversarial review → consensus** — draft the plan/design, then review with Codex
   as a cross-family second opinion before coding:
   ```bash
   codex exec --model gpt-5.5 -c model_reasoning_effort=high --sandbox read-only \
     "<blocking-only review prompt: lead with 'VERDICT: approve|revise'>" \
     < /dev/null > review.log 2>&1   # stdin + logfile are mandatory — see "Running gates safely"
   ```
   Iterate plan ↔ Codex until the verdict is `approve` and no high/medium/blocking
   findings remain. Document any accepted low-risk tradeoff in the plan or PR body.
   THEN implement. After implementing, run Codex again on the diff (adversarial —
   find real bugs only) with the same stop rule.
3. **Implement** on a `feat/...` or `fix/...` branch. Get full local CI green:
   `uv run python -m pytest -q`, `shellcheck` for shell scripts, `swift build` for `mac-app`.
4. **Open a PR** (`gh pr create`) — body = what changed + how tested.
5. **CodeRabbit** reviews the PR — handle every comment: fix all actionable concerns
   (one fix per commit, validate after each), reply with a concrete rationale for
   anything non-actionable, duplicate, incorrect, or intentionally deferred, and
   re-trigger with `@coderabbitai review` if it doesn't auto-run.
6. **Merge** only when BOTH gates are clean (Codex consensus + CodeRabbit pass) and the PR is
   mergeable: `gh pr merge <n> --squash --delete-branch`.

If a queue of tasks was provided, continue with the next queued task after the
merge. If there is no explicit next task source, stop after reporting the merge.
A real blocker means missing permissions, required human review, unavailable
external service, conflicting requirements, absent task source, or an unsafe or
destructive action that requires approval.

The two gates are different lenses — Codex hunts correctness/logic; CodeRabbit catches
conventions, edge cases, maintainability. Passing both beats either alone. Real examples
this caught: a relocation bug notarization couldn't see, chat text leaking into argv, an
ambiguous signing-identity selection, a leaked-reader deadlock, a weak test.

## Running gates safely (background review/build/test commands)

The gates (Codex review, `swift build`, `pytest`) run as long background commands. Standing
guardrails so they can't silently hang the pipeline:

- **`codex exec` MUST redirect stdin from `/dev/null`** in any background / non-TTY context —
  otherwise it blocks forever on `Reading additional input from stdin...`.
- **Redirect output to a logfile; never `cmd | tail`** — `tail` withholds all output until
  EOF, so a hung run looks identical to a slow one.
- **Bound each command with an explicit, per-command wall-clock deadline** — Codex review,
  Swift build, and pytest get *different* values, not one magic number. A deadline beats a
  poller: it is scoped to the command and self-terminates a hang. Use `timeout` / `gtimeout`
  from GNU coreutils (`brew install coreutils` — already installed in this dev env) and prefer
  `--signal=KILL` after a grace period so it reaps the whole **process group**, not just the
  parent PID.
- On a killed/timed-out gate, **cap retries and log why each retry happened** — never retry
  endlessly.
- **Distinguish hangs from blockers:** credential/permission prompts, merge conflicts, and
  required human review are BLOCKERS to escalate — never kill-and-retry them.
- CPU-time delta + log growth are **advisory diagnostics** (extend vs. retry), not a liveness
  policy — an API-bound Codex wait is healthy with low CPU and no log growth.

**Optional active supervisor — long unattended runs only.** For a long, unattended autonomous
run you MAY start an active supervisor (e.g. a short-interval loop) to watch the gates. Scope
it tightly or don't run it: only for a NAMED unattended run, started and stopped by that run,
with a visible PID/log, a max lifetime, and a max retry count. It supervises ONLY the commands
it launched or was explicitly handed — it must NEVER scan-and-kill arbitrary processes on the
host — and it has NO authority to merge or perform destructive cleanup. Do not make it
mandatory for routine changes: harness task-completion notifications plus per-command deadlines
already cover those.

## Hard rules

- **NEVER push directly to `main`.** Always a reviewed PR; merging a reviewed PR is allowed
  once it passes the pipeline. (Some local dev setups add a branch-guard git hook, but the
  repo does not ship one — treat this as a rule, not something enforced for you.)
- **Sync before planning:** `git fetch origin && git log --oneline origin/main -20`, and branch
  off `origin/main` — the local checkout can be stale (this once caused a whole duplicated PR).
- **Stage files by name**, not `git add -A`. **Never commit secrets** — use placeholders in
  docs (a local secret-scan hook may help, but don't rely on the repo having one).
- **Observability / privacy:** privacy-safe structured logging only — reason codes, metadata,
  counts; **never** captured text, window titles, or URLs.
- **Secrets** stay in the macOS Keychain / `~/openbird-codesign/` (not the repo). The signing
  identity and notary profile NAME are config, not secrets.

## Release (beta .dmg)

Invoke the **`/release-dmg`** skill. `script/package_dmg.sh` builds the self-contained bundle
→ Developer ID-signs every nested Mach-O → notarizes → staples → and produces
`dist/OpenBird.dmg` (it does NOT publish). The skill then **publishes** that dmg to GitHub
Releases as a separate `gh release create` step. The signing identity auto-derives from the
keychain; override via a gitignored `script/release.env`. See
`.claude/skills/release-dmg/SKILL.md`.

## Handy commands

- Tests: `uv run python -m pytest -q`
- Build the app bundle: `OPENBIRD_SWIFTPM_DISABLE_SANDBOX=1 ./script/build_and_run.sh --no-launch`
- Diagnostics: `openbird doctor` · DB check: `openbird data integrity`
- Dev runs without the recurring Keychain prompt: prefix with `OPENBIRD_DISABLE_KEYRING=1`
  (each `uv run` is a different unsigned interpreter, so the DB-key ACL never matches; the
  signed `.app` is unaffected).
- Uninstall / system-state cleanup: `openbird uninstall --dry-run` (preview), then
  `openbird uninstall` (LaunchAgent + Launch Services + Keychain key; data preserved) or
  `openbird uninstall --purge-data` (also deletes `~/.openbird`). The Keychain key is
  retained when an encrypted DB still depends on it.
- Dev Launch-Services hygiene (purge stale `dist/OpenBird.app` ghosts that shadow the real
  install by bundle id): `./script/dev_cleanup.sh` (add `--dist` to also delete build
  artifacts). By default only ghost + `*/dist/` dev builds are touched; `/Applications` is
  kept unless `--all`; Homebrew/Cellar/libexec installs are never unregistered.
