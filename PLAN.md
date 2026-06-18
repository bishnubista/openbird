# Plan: OpenBird — open-source local-first personal memory

> **Revision 5** — Codex round-4 corrections: indexing is **chunk-level** (FTS over chunks, vectors per
> chunk) to match chunk-level dedup + occurrence citations (fixes an internal contradiction); crash/log
> leakage is handled by **prevention, not after-the-fact scrubbing**. Remaining round-4 items are
> calibration of inherent always-on-screen-reader risk and are logged as **Accepted residual risks**
> below — Codex's adversarial role now moves to reviewing the actual code.
>
> **Revision 4** — adds Codex round-3 findings: **whole-data-path privacy** (encrypt/scrub raw AX
> text, OCR images, PCM/WAV, subprocess stdout, temp files, crash logs, IPC — not just the DB);
> **prompt-injection defense** (retrieved captured content is untrusted); **occurrence-level
> citations** (resolve blob hits back to specific observations); **chunk-level dedup/normalization**
> (not full-window hashing); **mic↔SCK clock sync**; **OCR region masking**; and **meeting-consent UX**.
>
> **Revision 3** — incorporates Codex adversarial review rounds 1 & 2. **[R3]** changes: the memory
> data model now separates **observations (every timestamped occurrence)** from **deduped content**
> so dedup never destroys timeline semantics; a set of **Phase-0 spike gates** (signed-bundle/TCC,
> system-audio, AX-compatibility, SQLCipher) must pass *before* the parallel build; product **claims
> are scoped to what is verified** (encryption is a release gate or we say "local-only, not
> app-encrypted"; structured output is best-effort; diarization is experimental); parallel agents run
> in **isolated worktrees behind contract tests**; and a minimal **menu-bar trust controller** ships
> early. (R2 changes — ScreenCaptureKit audio, SQLCipher, gated local-LLM, frozen contracts — retained.)

## Context

OpenBird is an always-on macOS assistant that reads **active-window text** (not screenshots),
transcribes meetings from system audio, unifies everything into a searchable personal memory, lets
you chat/draft against it, and runs scheduled "Routines." The product is **local-first**: data
stays on-device by default, encryption is release-gated, the LLM is **Ollama by default** (cloud
opt-in via LiteLLM), and the code is auditable.

**Decisions (locked with user):** Build **all 5 subsystems**; **Python core (uv) + Swift helpers**;
**local-first Ollama**, pluggable via LiteLLM.

**Build method (user-requested):** a **multi-agent harness (Workflow)** builds the subsystems, and
**Codex (`codex-cli 0.137.0`) runs adversarial review** on the plan (this doc) and on each subsystem;
findings feed a fix pass before moving on.

## Honesty about scope (what "MVP" delivers vs not) [R2]

To avoid the overpromising Codex flagged, the first build is explicitly bounded:
- **Capture:** best-effort active-window text via AX API, with hard timeouts and an app-compatibility
  matrix. Apps with poor/virtualized AX trees (some Electron/browser/canvas apps) are **known-degraded**;
  OCR fallback is *in scope* but ships behind a flag.
- **Meetings:** **manual start/stop recording** first (not auto-record). System audio via
  ScreenCaptureKit; mic mixed in. Batch transcription with a sliding-window design — not true
  low-latency streaming in v1.
- **LLM (local default):** scoped to **summarize / search / draft**. Tool-calling write-actions are
  **disabled in local mode** unless the user opts into cloud or a strong tool-capable local model,
  and always require human confirmation.
- **Integrations:** **local filesystem MCP only** in MVP (clearly labeled). One real OAuth read
  connector is a fast-follow, not v1. All write actions disabled until auth+confirm+audit exist.
- **Privacy:** redaction is **defense-in-depth, not a guarantee**. **[R3] First run is allowlist-ONLY**
  — terminals, code editors, and browsers are **excluded until the user explicitly enables them**.
  Per-node AX role/subrole policy is applied before any text is aggregated. The audit log is encrypted.
  Ship a visible capture indicator + kill switch from day one.
- **Claims are scoped to verification [R3]:** we only call data "encrypted at rest" if the SQLCipher
  gate passes (else the UI/README says **"local-only, not yet app-encrypted"**). Structured/JSON output
  in local mode is **best-effort** (validate+retry, not constrained decoding). Speaker labeling /
  "me vs others" is **experimental**.

## Phase-0 spike gates — MUST pass before the parallel build [R3]

Codex flagged that the riskiest parts (TCC, system audio, AX coverage, encryption) were assumed rather
than proven. Each is now a **gate** with an explicit pass/fail, run before Phase 2:

1. **Signed-bundle / TCC gate.** Package the capture + audio helpers as a **signed minimal `.app` /
   LaunchAgent with stable bundle IDs** (not `swift run`). Prove Accessibility + Screen-Recording +
   mic grants persist across rebuilds. Preflight tests the **packaged artifact**, not the dev binary.
2. **System-audio gate.** Prove **30-minute** capture of system output **+ mic as separate
   synchronized tracks** via ScreenCaptureKit, including a **Bluetooth device switch** and at least one
   of Zoom/Meet/Teams. BlackHole/aggregate-device fallback documented. Fail → meetings stays
   manual/experimental and says so.
3. **AX-compatibility gate.** Run a capture spike against the named matrix (Safari, Chrome, Slack,
   Teams, Zoom, VS Code, Notion, Terminal); record expected fields, missing cases, latency, failure
   modes. App adapters are treated as **product-critical**, not nice-to-have.
4. **Encryption gate.** Validate `sqlite-vec` loads under `sqlcipher3`. **Pass → SQLCipher is the
   release-gated default.** Fail → either pick an alternate encrypted-at-rest architecture or
   **relabel the product "local-only, not app-encrypted"** — no false claim.
5. **Contract-tests gate.** Typed pydantic fixtures + contract tests exist for every shared interface
   (`MemoryStore`, `LLMProvider`, `types.py`) and pass, before any Phase-2 agent starts.

**[R4] Gate criteria expanded:** the encryption gate (4) must also prove **WAL behavior, backup/export,
extension-loading policy, and acceptable perf at the volume target** under SQLCipher — not just that the
extension loads. The system-audio gate (2) must prove **mic and SCK system-audio share a common
sample-clock / timestamp alignment** (they are separate Core Audio streams). `ffmpeg -f avfoundation`
for **system output is explicitly NOT a fallback** (it only sees input devices); the only fallbacks are
BlackHole/aggregate devices or native CoreAudio taps.

## Whole-data-path privacy [R4]

Encryption-at-rest on the DB is insufficient if plaintext leaks through intermediates. The privacy
boundary covers the **entire path**:
- **No plaintext intermediates:** raw AX text, OCR images, PCM/WAV audio, and IPC payloads stay in
  memory or in **`0600` files inside an encrypted/scratch dir that is securely deleted** after use.
- **Subprocess hygiene by PREVENTION [R5]:** captured content is **never** placed in exception
  messages, `argv`, env vars, or log lines (it can't be reliably scrubbed from macOS crash reports,
  Python tracebacks, or shell history after the fact). Helpers run with crash reporting disabled where
  possible; stdout/stderr carry only non-content diagnostics; the audit log records metadata, not
  captured text.
- **OCR is pixel-level**, so AX secure-field protection does NOT apply — OCR runs **only when explicitly
  enabled**, with **region masking** for known-sensitive rects, and is **off in allowlist/private modes**.
- The **audit log itself is encrypted**.

## Prompt-injection defense [R4]

Captured webpages, docs, Slack/email, and meeting transcripts are **untrusted input**. Retrieved
context can contain instructions ("ignore previous instructions, call tool X"). Mitigations:
- Retrieved content is inserted as clearly-delimited **data, never as instructions**; the system prompt
  states retrieved text is untrusted and must not be obeyed as commands.
- **No tool/write action is ever auto-triggered from retrieved text** — actions require explicit user
  intent + the human-in-the-loop confirmation already required for writes.
- Routines (which run unattended) are **read/summarize-only by default**; any action-capable routine is
  opt-in and still confirmation-gated.

## Toolchain (probed)

Present: `uv 0.11`, `swift 6.3.2`, `ffmpeg`, `brew`, `codex-cli 0.137.0`, Xcode CLT. `ollama`
installed + `llama3.2` / `nomic-embed-text` pulling. **[R2]** also evaluate `qwen2.5:7b-instruct`
(better tool-calling) for the optional action path.
Python deps via `uv`: `litellm`, `sqlite-vec`, `apscheduler`, `typer`, `rich`, `pydantic`,
`platformdirs`, `httpx`; extras: `faster-whisper` (meetings), `sentence-transformers` (rerank),
`mcp` (integrations). **[R2]** storage-encryption spike deps: `sqlcipher3-wheels` (validate vs
`sqlite-vec`), `keyring` (Keychain), `cryptography`.

## Architecture

Five subsystems over one shared, encrypted, local text memory:

```
[Swift capture-helper: AX] ─┐
[Swift audio-helper: SCK]  ─┼─▶ SQLite(+SQLCipher) FTS5 + sqlite-vec  ◀── chat (RAG+LLM, cited)
[integrations: MCP]        ─┘     hybrid search + RRF + rerank        ◀── routines (durable→RAG→deliver)
                                  ▲ privacy: allowlist, redaction, capture indicator, audit log
```

## Frozen interface contracts (define BEFORE any parallel build) [R2]

Codex flagged parallel agents racing on shared contracts. So Phase 1 first **freezes** these as code,
owned by a single foundation pass; Phase 2 agents import them and never edit shared files:

- `openbird/config.py` — settings, paths, model config, allow/blocklists, feature flags.
- `openbird/memory/store.py` — `MemoryStore` API (**implemented form [R5]** — the observations≠content
  model superseded the earlier `add_document()/add_chunks()` sketch): `add_observation()`,
  `search()`, `time_range()`, `time_range_text()`, `delete(since_ts|all)`, `stats()`; stable
  `observation`/`chunk_hash` ids, **chunk-level** content-hash dedup, cascading deletes across
  `content_blobs`/`observations`/`chunks`/`blob_chunks`/`fts`/`vec`.
- `openbird/llm/provider.py` — `LLMProvider`: `embed(texts)->vectors`, `complete(messages, *, json_schema=None)`;
  records `embedding_model`/`dim`/`normalized` metadata.
- `openbird/types.py` **[R2]** — pydantic event/record schemas (`CaptureEvent`, `Chunk`, `SearchHit`,
  `Citation`, `RoutineRun`) shared by all subsystems.
- `openbird/memory/schema.sql` — DB migrations; **one owner**.

Phase 2 agents each own a disjoint directory (`capture/`, `meetings/`, `chat/`, `routines/`,
`integrations/`) and are forbidden from editing shared files; CLI wiring is serialized in the
integration phase.

## Repo layout (greenfield)

```
openbird/
├─ pyproject.toml · README.md · PLAN.md · scripts/setup.sh
├─ Formula/openbird.rb    # Homebrew formula for CLI + app bundle install
├─ mac-app/               # SwiftPM: native macOS trust-controller app
├─ script/build_and_run.sh # builds/stages/launches dist/OpenBird.app
├─ capture-helper/        # Swift SPM: AXUIElement frontmost-window text → JSON (timeouts, dedup)
├─ audio-helper/          # [R2] Swift SPM: ScreenCaptureKit system audio + mic → WAV/PCM frames
├─ openbird/
│  ├─ config.py · types.py
│  ├─ memory/{store.py,schema.sql,ingest.py,search.py}
│  ├─ capture/{daemon.py,redact.py,adapters.py}   # [R2] adapters.py = per-app AX strategies
│  ├─ meetings/{audio.py,transcribe.py,pipeline.py}  # [R2] pipeline.py = VAD/window/stitch
│  ├─ llm/provider.py
│  ├─ chat/rag.py
│  ├─ routines/{scheduler.py,templates.py,store.py}  # [R2] durable routine_runs
│  ├─ integrations/mcp.py
│  ├─ storage/crypto.py   # [R2] SQLCipher/Keychain key mgmt
│  ├─ preflight.py        # [R2] reports TCC, ollama models, audio cap, encryption, allowlist
│  └─ cli.py              # capture / chat / ingest / routine / meeting / preflight / data purge
└─ tests/{unit,integration,manual_smoke}/   # [R2] tiered tests + fake adapters
```

## Subsystem build details (with R2 hardening)

1. **Memory (foundation). [R3] observations ≠ content.** The data model separates *what was seen* from
   *when/where it was seen*, so dedup never collapses the timeline:
   - `content_blobs`(content_hash PK, text) — **deduped** canonical text; embedded **once**.
   - `observations`(id, content_hash→blob, ts, app, window, url/title, session_id) — **one row per
     occurrence**, never deduped. The same window seen 50× = 50 observations, 1 blob.
   - `chunks`(id, content_hash, span, text) · `fts`(FTS5 **over chunks**) · `vec_chunks`(sqlite-vec,
     **fixed dim**, **embedded per chunk**). **[R5]** indexing/ranking is **chunk-level** (not per-blob)
     so retrieval, dedup, and occurrence citations are consistent; each chunk still maps via
     `content_hash`→blob→`observations`.
   **[R4] Dedup is chunk-level + normalized, not full-window:** hashing happens on **normalized chunks**
   (whitespace/boilerplate-stripped) so a one-character edit doesn't fork the whole window and static
   chrome/boilerplate doesn't dominate BM25/embeddings. Repeated boilerplate chunks are down-weighted.
   `search.py`: vector + BM25 → **RRF** → **MMR/source-session dedup at retrieval** (cap near-dupes) →
   optional cross-encoder rerank; **separate time-range path** (range-scan over `observations`, not
   semantic) for "what did I do yesterday" / routines / activity summaries. **[R4]** a blob/chunk hit
   **resolves back to its `observations`** (app/window/url/time) so ranking and citations are
   occurrence-aware, not blob-anonymous. Cascading deletes across blobs/observations/chunks/fts/vec with
   integrity checks; **load test at 100k+ observations** with a volume target + retention default, and a
   benchmark gate for when to migrate off sqlite-vec.
2. **Storage encryption [R2].** `storage/crypto.py`: target **SQLCipher** (whole-DB, so FTS index is
   also encrypted) with key in **macOS Keychain** via `keyring`. **First task = a spike** validating
   `sqlite-vec` loads under `sqlcipher3`. If incompatible: fall back to documented interim
   (DB file `0600` + rely on FileVault) with an explicit limitation note — **no false encryption claim**.
3. **LLM provider.** LiteLLM wrapper; default `ollama/llama3.2` + `ollama/nomic-embed-text`. **[R3]**
   persist exact **provider, model tag/digest, dimension, normalization flag, embedding timestamp per
   vector**; **refuse search across incompatible embedding cohorts** (query each cohort separately or
   reindex); reindex tool on model change; test embeddings *through LiteLLM* (not just direct Ollama).
   `complete(json_schema=)` is **best-effort structured generation** (validate+retry, NOT constrained
   decoding) — with **per-model acceptance tests** and default local models chosen by *measured* task
   performance. Citation/JSON/routine validators are hard gates regardless of model.
4. **Capture.** Swift `capture-helper`: `AXIsProcessTrustedWithOptions` prompt, **depth/node/time
   limits**, node-level dedup, AX observer lifecycle, stable bundle id/signing note (TCC is per signed
   path). `adapters.py`: per-app strategies + **compatibility matrix** (Safari/Chrome/Slack/Teams/
   Zoom/VS Code/Notion). OCR fallback behind a flag. `redact.py`: allowlist-first, regex+blocklist as
   defense-in-depth, skip incognito, block password managers + finance/health apps by default.
5. **Meetings [R3].** `audio-helper` (Swift, **ScreenCaptureKit** `SCStream` system audio **+ mic kept
   as SEPARATE synchronized tracks** through transcription — merged at the transcript-segment level,
   not the audio level, so "me vs others" stays meaningful). `pipeline.py`: VAD, chunk
   duration/overlap, transcript stitching, timestamp correction, **device-switch recovery**.
   `transcribe.py`: faster-whisper batch over sliding windows. **Speaker labeling is experimental**
   and presented as such. **[R4] mic and SCK system-audio are separate Core Audio streams** → align on
   a common sample-clock/timestamp before per-track transcription; flag drift. **[R4] consent UX:**
   botless capture is invisible to other participants — show a clear recording indicator, require
   explicit manual start, and surface a consent/legal note (one-/two-party-consent varies by region).
   **Manual record** in v1 (auto-record later). BlackHole/aggregate-device fallback documented. Gated by
   the Phase-0 system-audio spike.
6. **Chat.** `rag.py`: hybrid retrieve → **dedupe by document/session before context assembly** →
   grounded prompt → answer. **[R2] citation validation**: only chunk IDs present in the final
   context set may be cited; reject/repair hallucinated citations. **[R4]** citations resolve to a
   specific **observation** (app/window/time), not just a deduped blob, so "where did you get that?"
   names the real occurrence. Retrieved text is treated as untrusted data (see Prompt-injection
   defense). Tests with near-identical chunks.
7. **Routines [R2].** `routines/store.py`: **durable** `routine_runs` with idempotency keys; on
   startup, run missed jobs. `scheduler.py` APScheduler (best-effort while daemon runs; launchd noted).
   `templates.py`: daily-briefing / yesterday's-work / weekly-summary → time-range retrieval → deliver
   (stdout/notification) → store as entry.
8. **Integrations [R2].** `mcp.py`: MCP registry; **MVP = local filesystem MCP only (labeled)**. Write
   actions disabled until OAuth scopes + confirmation UX + audit log exist. One OAuth read connector is
   a fast-follow.
9. **Privacy & trust surface [R3].** Ship a **minimal menu-bar controller early** (even while the rest
   is CLI): shows live capture/audio state + the currently-captured app, exposes **pause + kill
   switch**, and can **stop the helper processes**. The current SwiftPM macOS app stages the helpers
   into `dist/OpenBird.app`, reports packaged-helper/preflight status, writes `capture.paused` to
   pause/resume ingestion, and stops helper processes by name. Pause is explicitly an ingestion gate;
   users must stop helpers to halt live helper reads. Backed by an **encrypted local audit log**.
   `openbird data purge --since` with verified cascade deletes. (A persistent notification is a
   supplement, not the primary trust surface — notifications need their own permission and can be
   silenced.)

## Build & adversarial-review workflow (the agent harness)

- **Phase 0 — Setup:** brew/ollama done; `uv sync`; scaffold pyproject/types/config; **storage
  encryption spike** (gate: SQLCipher+sqlite-vec works or fallback chosen).
- **Phase 1 — Freeze foundation:** `config.py`, `types.py`, `memory/*`, `llm/provider.py`,
  `storage/crypto.py` — single owner, contracts frozen, unit-tested.
- **Phase 2 — Subsystems (parallel, ISOLATED WORKTREES [R3]):** `capture`, `meetings`, `chat`,
  `routines`, `integrations` each build in their **own git worktree** behind the frozen contracts +
  **contract tests/typed fixtures**; they import shared interfaces and never edit shared files. A
  **single integration owner** merges; shared-contract changes are allowed only through a serialized
  foundation pass, never inside a subsystem worktree.
- **Per subsystem — Codex adversarial review** (`codex:codex-rescue`): attack correctness, races,
  injection, resource leaks, privacy leaks, citation provenance. Collect findings → fix pass →
  **integration check after each subsystem**, not only at the end.
- **Phase 3 — Integration:** serialized CLI wiring, `preflight.py`, README; E2E smoke.
- **Phase 4 — Final Codex review** of the assembled repo; iterate to consensus.

## Verification [R2 tiered]

- **Unit (CI-able, fake adapters):** memory upsert/dedup/cascade-delete, RRF, dim-check, redaction,
  citation validation, routine idempotency, JSON validate+retry.
- **Local integration (needs Ollama):** embed/search/chat round-trip with cited answers; reindex on
  model change.
- **Manual macOS smoke [R3 matrix]:** a repeatable test matrix recording **OS version, hardware,
  signed bundle hash, permissions state, target apps, expected outputs, and failure logs/screenshots**
  for the hard runtime risks (TCC prompts, AX traversal, ScreenCaptureKit audio, notifications, device
  switching). `openbird preflight` reports TCC/Accessibility, Ollama models, audio (mic+system)
  capability, **DB encryption status**, allow/blocklist — run against the **packaged signed artifact**.
- **Load:** ingest thousands of repeated screen captures; confirm dedup keeps recall clean.
- Codex final verdict = `approve`.

## Accepted residual risks [R5]

These were raised by Codex and are **inherent to an always-on screen reader** rather than fixable in a
plan; we accept and disclose them, and constrain scope accordingly:
- **AX coverage is uneven** across Chrome/Safari/Electron/canvas apps. The compatibility matrix defines
  *measured* coverage per app; uncovered apps are reported as degraded, OCR is the (privacy-gated) escape
  hatch. No universal-text guarantee is claimed.
- **Redaction is best-effort, not a guarantee.** Allowlist-only first run is the real protection; regex
  + blocklist is defense-in-depth. Documented plainly; users opt in to risky apps knowingly.
- **TCC/permissions are environment-sensitive** (bundle identity, helper exec path, OS version). The
  signed-bundle gate + packaged-artifact preflight mitigate but cannot eliminate this.
- **Local small models** (llama3.2) may produce weak JSON/citations; validators are hard gates and
  cloud opt-in exists for quality-critical tasks.
- **Always-on scale** beyond the 100k load test is validated by the benchmark gate + retention defaults,
  with a documented migration path off sqlite-vec.
- **Botless meeting capture consent** is a legal/ethical constraint surfaced in UI, not solved by code.
- **Shared-chunk occurrence resolution [code-review]:** a chunk deduped across multiple observations
  is cited to the **most-recent** occurrence (there is no per-occurrence signal at query time on a
  deduplicated store). This is an explicit, tested policy; a future occurrence-aware ranking could
  refine it. Locked by `test_shared_chunk_resolves_to_most_recent_occurrence`.
- **Mic↔system audio clock alignment [code-review]:** system frames are stamped with the
  ScreenCaptureKit buffer's presentation timestamp and mic frames with the AVAudioEngine tap's host
  time. Whether these share one clock domain in practice can only be confirmed on real audio hardware —
  it is exactly what the **Phase-0 system-audio gate** validates, and the Python `ClockSync` detects
  cross-track drift at runtime as a safety net. Unverifiable in CI; flagged for the hardware spike.
- **Embedding cohort vs mutable model tags [code-review]:** the cohort key pins provider/model-name/
  dimension/normalization (catching model swaps and dim changes), but a mutable Ollama tag pointing at
  a new build under the *same* name is not detected (the backend exposes no stable digest via
  LiteLLM). Mitigated by the reindex tool; flagged for users who repin tags.

## Out of scope (first pass)

Full SwiftUI menu-bar GUI (ship a CLI/notification trust shim instead), real OAuth for all 90
integrations (filesystem MCP + one read connector fast-follow), true low-latency streaming
transcription, auto-record meetings, Windows support.
