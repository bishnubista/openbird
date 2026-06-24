# Contributing to OpenBird

Thanks for your interest in OpenBird — a local-first, on-device personal memory
for macOS. This guide covers how to set up a dev environment, the quality gates a
change must pass, and the conventions we follow.

By contributing, you agree that your contributions are licensed under the
project's [Apache License 2.0](LICENSE) on an inbound=outbound basis (see
Section 5 of the license). No separate CLA is required.

## Project layout

OpenBird is five subsystems over one shared on-device text memory:

| Path             | What it is |
| ---------------- | ---------- |
| `openbird/`      | Python core — CLI, memory store (SQLite + FTS5 + sqlite-vec), search, RAG, routines |
| `capture-helper/`| Swift helper — active-window text via Accessibility (AX) |
| `audio-helper/`  | Swift helper — meeting capture via ScreenCaptureKit |
| `mac-app/`       | Native SwiftUI trust-controller app |
| `tests/`         | Python unit + integration tests (fake providers, no network) |
| `script/`        | Build, package, and release shell scripts |

## Development setup

Requirements: macOS, [`uv`](https://docs.astral.sh/uv/), Swift 6+ (Xcode CLT),
and [Ollama](https://ollama.com/) for end-to-end model runs.

```bash
uv sync --extra dev          # Python core + test deps
# add extras as your change needs them:
#   --extra encryption   SQLCipher storage gate + Keychain
#   --extra meetings     faster-whisper (portable ASR)
#   --extra meetings-mlx parakeet-mlx (Apple Silicon ASR)
#   --extra rerank       cross-encoder reranker
#   --extra integrations MCP clients
```

## Local CI — run the gates before you open a PR

Match what reviewers run. A PR is expected to be green on all gates that apply to
the code it touches:

```bash
# Python (always)
uv run python -m pytest -q                                   # unit + fake-provider E2E
uv run --extra encryption python scripts/encryption_gate.py  # if you touched storage/crypto

# Shell scripts
shellcheck script/*.sh

# Swift helpers / app (if you touched them)
( cd capture-helper && swift build )
( cd audio-helper   && swift build )
OPENBIRD_SWIFTPM_DISABLE_SANDBOX=1 ./script/build_and_run.sh --no-launch
```

The CI-safe E2E path uses fake capture/model providers, so the test suite runs
without macOS TCC permissions, a signed helper, Ollama, or network access. Tests
that genuinely need Ollama auto-skip when it's absent.

## Branching, commits, and PRs

- **Never push to `main`.** Branch off the latest `origin/main`:
  ```bash
  git fetch origin && git checkout -b feat/short-description origin/main
  ```
- **Branch names:** `feat/...`, `fix/...`, `docs/...`, `chore/...`.
- **Commits:** Conventional Commits (`feat(search): ...`, `fix(memory): ...`).
  Stage files by name; never `git add -A`.
- **One concern per PR.** Don't mix unrelated fixes. Smaller PRs review faster.
- **PR body:** what changed and how you tested it. Link the issue it closes.
- PRs are reviewed by maintainers (and an automated reviewer). Address every
  actionable comment — fix it, or reply with a concrete rationale if you're
  intentionally deferring.

## Privacy rules (non-negotiable)

OpenBird's whole premise is that your data stays on your device. Contributions
must preserve that:

- **Never log captured content.** No captured text, window titles, or URLs in
  logs, exceptions, or argv. Privacy-safe structured logging only — reason codes,
  metadata, and counts.
- **No new off-device network calls** without an explicit user opt-in
  (`OPENBIRD_ALLOW_CLOUD=1`) and a visible CLOUD ACTIVE indicator. A non-loopback
  host for any model/embed/rerank route is a cloud route and must be gated.
- **Capture is allowlist-first.** Don't widen what's captured by default.
- **No secrets in the repo.** Use placeholders in docs; real IDs and keys stay in
  the Keychain / local config, never committed.

## Reporting bugs and security issues

- Functional bugs and feature ideas: open a GitHub issue with repro steps and your
  environment (`openbird doctor` prints a content-safe, home-redacted version +
  system report; macOS version).
- Security vulnerabilities: **do not** open a public issue — follow
  [SECURITY.md](SECURITY.md).

## Code of conduct

Be respectful and constructive. Harassment or abuse won't be tolerated; report
conduct concerns to the maintainers through the repository's private channels.
