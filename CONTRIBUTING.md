# Contributing to OpenBird

Thanks for your interest in OpenBird — an open-source, local-first personal
memory for macOS. This guide covers the local setup and the checks CI enforces.

## Project scope & platform

OpenBird targets **macOS** (Accessibility, ScreenCaptureKit, launchd, Keychain).
The Python core and test suite are developed and verified on macOS; CI runs the
tests on `macos-latest`. Pure-logic changes may work on other platforms, but
macOS is the supported target.

## Development setup

Requirements: macOS, [`uv`](https://docs.astral.sh/uv/), Swift 6+ (Xcode CLT),
and [Ollama](https://ollama.com/) for the model-backed paths.

```bash
# Core + dev tooling. Add extras as your change needs them.
uv sync --extra dev --extra encryption --extra integrations
# Heavy extras (pull PyTorch) — only for meetings/rerank work:
#   uv sync --extra meetings --extra rerank
```

## Checks CI enforces

Run these before opening a PR — they are exactly what the `lint` and `test`
jobs run:

```bash
uv run ruff check .          # lint (see pyproject.toml for enabled rules)
uv run ruff format --check . # formatting (apply with: uv run ruff format .)
uv run pytest -q             # full suite (integration tests auto-skip)
```

`uv run mypy openbird` is also available; type checking is advisory (non-blocking
in CI) while annotations are added incrementally.

Integration tests that need external services (e.g. a live Ollama) skip
automatically when those services are unavailable.

## Pull requests

- Keep changes surgical: touch only what your change requires; don't reformat or
  refactor unrelated code in the same PR.
- Separate mechanical churn (e.g. a formatter sweep) into its own commit so the
  substantive diff stays reviewable.
- Match the surrounding code's style, naming, and comment density.
- Respect the privacy posture: capture/error paths deliberately avoid logging
  exception messages or tracebacks, because those can embed captured content.
  Don't add logging that leaks content.

## License

By contributing, you agree that your contributions are licensed under the
project's [Apache License 2.0](LICENSE).
