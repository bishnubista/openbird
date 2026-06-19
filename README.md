# OpenBird

**Open-source, local-first personal memory for your Mac.**

OpenBird is an always-on macOS work assistant that reads the **text** of your active window (never
screenshots), transcribes meetings from system audio, unifies everything into a **searchable personal
memory that stays on your machine**, lets you chat and draft grounded in that memory (with citations),
and runs scheduled "Routines." OpenBird is **local-first**: your data never leaves your device by
default, the model layer is **BYO-model** (Ollama out of the box, cloud opt-in), and the system is
fully auditable.

> Status: **early but working.** Python core + Swift capture/audio helpers + a native macOS
> trust-controller app build and pass the test suite. Functional screen/audio capture requires a signed bundle + macOS permissions (see
> [Release gates](#release-gates)).

## Why local-first?

- **Private by default:** memory lives in on-device SQLite, with encryption gated by preflight.
- **Text-first capture:** active-window text and UI metadata are stored, not screenshots or video.
- **Model choice:** Ollama is the default, and cloud models are explicit opt-in through LiteLLM.
- **Auditable behavior:** capture allowlists, redaction policy, source metadata, and deletion commands
  are part of the product surface.
- **Extensible integrations:** the MVP starts with a filesystem MCP connector and keeps write
  actions behind explicit confirmation.

## Architecture

Five subsystems over one shared, local text memory:

```
[capture-helper: Swift/AX] ─┐
[audio-helper: Swift/SCK]  ─┼─▶ SQLite + FTS5 + sqlite-vec   ◀── chat   (RAG + LLM, cited)
[integrations: MCP]        ─┘     hybrid search · RRF · MMR   ◀── routines (durable → RAG → deliver)
                                  ▲ privacy: allowlist, redaction, capture indicator, deletion controls
```

The memory model separates **observations** (one row per timestamped occurrence — never deduped) from
**content blobs/chunks** (deduped, embedded once). This keeps timeline queries ("what did I do
yesterday") correct while avoiding redundant storage. Citations resolve back to the specific
observation (app/window/time).

## Quickstart

Requirements: macOS, [`uv`](https://docs.astral.sh/uv/), Swift 6+ (Xcode CLT), and
[Ollama](https://ollama.com/).

```bash
# 1. Install deps
uv sync --extra encryption   # core + SQLCipher gate; add --extra meetings or integrations as needed

# 2. Local models
ollama pull llama3.2
ollama pull nomic-embed-text

# 3. Verify the environment
uv run openbird preflight    # reports ollama, sqlite-vec/FTS5, encryption, permissions
uv run --extra encryption python scripts/encryption_gate.py

# 4. Try it (no screen capture needed)
echo "Decision: store screen text in SQLite, not screenshots." > note.txt
uv run openbird ingest note.txt
uv run openbird chat "what did we decide about storage?"

# 5. Build the native helpers
( cd capture-helper && swift build )
( cd audio-helper   && swift build )

# 6. Build and launch the macOS app bundle
./script/build_and_run.sh --verify
```

### Homebrew install

After a tagged release is published, install the CLI and app bundle with:

```bash
brew tap bishnubista/openbird https://github.com/bishnubista/openbird.git
brew install bishnubista/openbird/openbird
openbird --help
openbird-app
```

The formula downloads the source archive attached to the matching GitHub tag
release. On every `v*` tag, `.github/workflows/homebrew-release.yml` builds a
deterministic `openbird-<version>.tar.gz` source archive, uploads it to the
GitHub release, updates `Formula/openbird.rb` with the new source URL and sha256,
and **opens a pull request** with that formula bump. The PR is reviewed and
merged through the normal protected-`main` flow rather than pushed directly, so
branch protection on the formula is preserved.

To cut the first Homebrew release:

```bash
git tag v0.1.0
git push origin v0.1.0
```

After the formula-bump pull request is merged, users can upgrade through the
normal Homebrew path:

```bash
brew update
brew upgrade openbird
```

The formula installs `openbird` with the SQLCipher encryption extra as a
virtualenv-backed CLI and stages `OpenBird.app` under Homebrew's prefix.
`openbird-app` launches that installed bundle.

> **Install requires network, and is not vendored/offline.** `brew install`
> builds from source: it runs `uv pip install ".[encryption]"`, which resolves
> and downloads Python dependencies from PyPI at install time. The dependency
> set is not vendored into the archive and is not pinned for reproducible
> offline installs. "Local-first" describes where your data lives at runtime
> (on your device), not the install path.
>
> **The Homebrew app bundle is unsigned and is not a supported capture/audio
> artifact.** `script/build_and_run.sh` does no code-signing or notarization, so
> the `OpenBird.app` staged by the formula cannot obtain macOS Screen Recording
> or Accessibility (TCC) permissions — screen and audio capture will not work
> from a brew install. Functional capture requires a signed bundle with a stable
> identity plus manually granted macOS permissions (see
> [Release gates](#release-gates)). The CLI memory features
> (`openbird ingest` / `chat` / `routine`) work fully from the brew install.

### CLI

```
openbird preflight              # environment + readiness report
openbird ingest <path>          # ingest a file or directory into memory
openbird chat "<question>"      # grounded, cited answer over your memory
openbird capture [--loop]       # run the capture daemon over the helper
openbird routine list|run <name>  # scheduled routines (daily-briefing, yesterday's-work, weekly-summary)
openbird meeting                # meeting capture (gated on signed audio helper)
openbird data stats             # row counts + active embedding cohort
openbird data purge --since <when> | --all   # cascade-delete stored memory
openbird data prune --older-than <span>      # retention prune (e.g. 90d, 24h)
openbird data vacuum            # reclaim freed disk space (VACUUM + WAL checkpoint)
```

### macOS app

`./script/build_and_run.sh` builds `dist/OpenBird.app` from the SwiftPM app target,
stages the capture/audio helpers into `Contents/MacOS`, adds an `openbird-cli`
shim for local preflight checks, and launches the bundle. The app exposes an
early trust surface: packaged-helper status, preflight summary, pause/resume
ingestion, stop helpers, and quick access to the local data folder.
Pause is an ingestion gate: helper processes may still read and emit text until
you use **Stop Helpers** or quit the helper process.

```bash
./script/build_and_run.sh           # build and launch dist/OpenBird.app
./script/build_and_run.sh --verify  # launch and confirm the app process exists
```

## Configuration

Environment overrides (all optional):

| Var | Default | Purpose |
|---|---|---|
| `OPENBIRD_DATA_DIR` | `~/.openbird` | where the SQLite memory lives |
| `OPENBIRD_LLM_BACKEND` | `litellm` | provider backend selector; `mlx` is reserved pending experiment promotion |
| `OPENBIRD_LLM_MODEL` | `ollama/llama3.2` | any LiteLLM model string (e.g. `claude-...`, `gpt-...`) |
| `OPENBIRD_EMBED_MODEL` | `ollama/nomic-embed-text` | embedding model (dim pinned per cohort) |
| `OPENBIRD_REQUIRE_ENCRYPTION` | `0` | when `1`, opening the DB **fails closed** (raises) instead of falling back to a plaintext file if SQLCipher cannot be verified |
| `OPENBIRD_RETENTION_DAYS` | `0` (keep forever) | default cutoff for `openbird data prune` when `--older-than` is omitted |

## Storage growth & retention

Memory is stored in on-device SQLite (WAL mode, `busy_timeout` so concurrent
readers/writers queue rather than fail). Each observation is one row referencing
a deduped content blob; identical chunks are embedded and indexed exactly once,
so storage grows with *unique* captured content, not raw capture volume. Vectors
dominate size (one `float32[embed_dim]` blob per unique chunk — ~3 KB/chunk at
the default 768 dims) alongside the FTS5 index.

To bound growth:

- `openbird data prune --older-than 90d` cascade-deletes observations older than
  the cutoff (and any blobs/chunks/index rows they orphaned), atomically.
- `openbird data vacuum` reclaims the freed pages back to the OS (deletes only
  mark pages free; the file shrinks after `VACUUM`). The WAL sidecar is bounded
  by an explicit `wal_autocheckpoint` and truncated on vacuum.

## Privacy

- **Allowlist-first:** the capture daemon only records apps you explicitly allow; terminals, editors,
  and browsers are off until enabled.
- **Redaction is defense-in-depth, not a guarantee.** Allowlists and blocklists are the primary
  protection; secret scrubbing is a second layer.
- **Whole-data-path:** captured text is never written to logs, exceptions, or argv; source metadata is
  scrubbed before storage.
- **Encryption** at rest activates when the SQLCipher gate passes; otherwise the DB is `0600` and
  `preflight` says `plaintext-0600` so the claim is never overstated. The gate command verifies
  SQLCipher + `sqlite-vec`, WAL, encrypted backup behavior, disabled extension loading, private
  permissions, and a write/read performance smoke in a temporary workspace.

## Release gates

Functional capture/audio require steps a dev build can't satisfy:
1. **Signed bundle / TCC** — helpers packaged as a signed `.app`/LaunchAgent with stable bundle IDs.
2. **System-audio** — 30-min ScreenCaptureKit system + mic capture proven (separate synchronized tracks).
3. **AX-compatibility** — measured coverage across the app matrix.
4. **Encryption** — SQLCipher validated against `sqlite-vec`, WAL, encrypted backup/export behavior,
   disabled extension loading, private file modes, and write/read performance.

## Development

```bash
uv sync --extra dev
uv run python -m pytest -q   # unit + fake-provider E2E; Ollama tests auto-skip if absent
uv run --extra encryption python scripts/encryption_gate.py
uv run python -m pytest tests/integration/test_e2e.py -q
./script/build_and_run.sh --no-launch
```

The CI-safe E2E path uses fake capture/model providers, so it runs without macOS
TCC permissions, a signed helper, Ollama, or network access.

## License

[Apache License 2.0](LICENSE) — permissive, with an explicit patent grant.
