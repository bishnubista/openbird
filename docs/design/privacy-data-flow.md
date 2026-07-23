# OpenBird Privacy Data Flow

This document is the source of truth for OpenBird privacy claims. Keep it in sync
with `docs/privacy-routes.yaml`, preflight output, app copy, and tests.

OpenBird's privacy posture is local-first prevention, not server-side custody:
capture is allowlist-first, model routes are local by default, and any route that
sends captured memory off the device must be explicit and visible.

## Route Classes

- `local`: data stays on this Mac at runtime.
- `self-hosted-remote`: data leaves this Mac but goes to infrastructure the user
  controls, such as a non-loopback Ollama endpoint. This is still egress.
- `third-party-cloud`: data leaves this Mac for a provider controlled by someone
  else, such as OpenAI, Anthropic, or a hosted parsing/model endpoint.
- `unknown`: OpenBird cannot prove the route. Unknown is never release-green.

Provider labels are not enough. `ollama/*` is `local` only when the resolved host
is loopback. A non-loopback Ollama host must ride the same `CLOUD ACTIVE` warning
path as any other off-device model route.

## Data Flow Matrix

| Route | Source | Captured Fields | Storage | Runtime Egress | Retention/Delete | User Truth Surface |
|---|---|---|---|---|---|---|
| `capture.active_window` | Swift Accessibility helper | bundle id, sanitized window title, active-window text, timestamp, incognito flag | SQLite `observations`, `content_blobs`, `chunks`, FTS, sqlite-vec | none unless downstream chat/routine uses a remote model | `openbird data purge`, `prune`, and uninstall purge cascade through derived indexes | app capture state, allowlist/blocklist, preflight |
| `capture.ocr_window` | Swift helper, per-app opt-in `SCScreenshotManager` window still + on-device Vision OCR | bundle id, sanitized window title, recognized text, timestamp, OCR provenance flag | same SQLite memory path as AX text: `observations`, `content_blobs`, `chunks`, FTS, sqlite-vec; the transient `CGImage` is never stored | none unless downstream chat/routine uses a remote model | `openbird data purge`, `prune`, and uninstall purge cascade through derived indexes; deleting OCR text is identical to deleting AX text | app deep-capture section, capture health, preflight, privacy route manifest |
| `capture.pause` | pause sidecar in data dir | no content should be read while source-pause is active | no new rows | none | pause file removed by resume/uninstall cleanup | app/menu bar state must say source-paused only when helper enforces it |
| `ingest.files` | user-selected file or folder | file text/metadata extracted by local parser path | SQLite memory store and derived indexes | none unless model route is remote during embedding/chat | purge/prune cascade; future export is separate egress | CLI/app ingestion status |
| `chat.local` | question plus retrieved memory | question, retrieved chunks, citations | chat output is not persisted by the core CLI today | local Ollama loopback only | underlying memory deletion removes future retrievability | no cloud banner |
| `chat.day_facts` | deterministic day-fact answer branch | question used only for classification, local day-memory productivity facts, derived citation source ids, memory-context counts | SQLite `day_memories`; chat output is not persisted | explicit day-scoped branch returns before `_provider`; query-inferred branch may construct a provider before RAG, but returns before `provider.complete`; neither branch issues completion or embedding requests | purge/prune removes source observations and derived day memory, removing future retrievability | `reasoning_route=local_deterministic` / `Local only` label |
| `chat.day_memory` | deterministic broad day-memory answer branch | question used only for classification, distilled day-memory metrics, workstreams, sessions, open loops, domains/repos, derived citation source ids, memory-context counts | SQLite `day_memories`; chat output is not persisted | explicit day-scoped branch returns before `_provider` using only the non-embedding maintenance stub; query-inferred branch may construct a provider before RAG, but returns before `provider.complete`; neither branch issues completion or embedding requests | purge/prune removes source observations and derived day memory, removing future retrievability | `reasoning_route=local_deterministic` / `Local only` label |
| `chat.day_memory_cached_summary` | deterministic day answer composed with PRECOMPUTED block-summary narrative | distilled day-memory facts plus stored block-summary prose (local-model text generated EARLIER by the routines worker under the battery/idle gate), typed `block_summary` derived citations | SQLite `day_memories` + `block_summaries`; chat output is not persisted | none at answer time — composition is a local read, no provider call, no embedding; the generation-time route is the local Ollama model under `CloudOptInRequired` gating | deleting a cited span/observation trigger-deletes the block summary, and (recursive triggers) any day memory citing it; the composed narrative vanishes on the next answer | `reasoning_route=local_cached_model_summary` / `Local summary (cached model prose)` label — the deterministic label must NOT be shown once model prose is present |
| `chat.remote` | question plus retrieved memory | question, retrieved chunks, citations | local DB plus provider request body | `third-party-cloud` or `self-hosted-remote`, depending on provider destination | local deletion cannot recall payloads already sent to remote providers | `CLOUD ACTIVE` banner and per-answer/provider disclosure |
| `assistant.mcp_read` | assistant-invoked bounded local read | exclusion-filtered capture excerpt, app id, timestamp, source, observation id, host label; no URL or window title | no new OpenBird storage; assistant provider request body after tool use | `third-party-cloud` when Claude/ChatGPT invokes a content tool | local deletion removes future retrievability but cannot recall excerpts already sent | install warning, MCP tool descriptions/results, `openbird assistant status` |
| `assistant.activity_summary` | assistant-invoked bounded activity-span aggregation (no content read) | app bundle ids, per-app foreground/meeting durations and span counts, AFK seconds, context-switch count, focus-block timestamps, folded-tail app count and seconds, per-app redacted durations with closed-enum reason codes plus an unattributed-redacted total, resolved query window echo with its IANA timezone, host label; excluded apps stay an unnamed total; no captured text, window title, url host, or observation-derived statistic | no new OpenBird storage; assistant provider request body after tool use | `third-party-cloud` when Claude/ChatGPT invokes the summary tool | local deletion removes future retrievability but cannot recall behavioral metadata already sent | install warning + app connect confirmations, MCP tool description, per-response activity egress notice |
| `assistant.capture_status` | assistant-invoked metadata-only store status read | store-lifetime observation total, encryption state, exclusion-configuration counts, host label; no content, spans, or app identifiers | no new OpenBird storage; assistant provider request body after tool use | `third-party-cloud` when Claude/ChatGPT invokes the status tool | local deletion removes future retrievability but cannot recall counts already sent | install warning + app connect confirmations, MCP tool description, per-response status egress notice |
| `deep_brain.ask` / `briefing.model` / `productivity.coach` | user-triggered packet reasoning | distilled packet content sent to the active model route; local ledger stores only packet hash, byte/count metadata, route metadata, exclusion reason counts, outcome, and error kind | provider request body plus SQLite `reasoning_send_ledger`; ledger inherits the memory DB's at-rest protection and stores no raw question, answer, packet JSON, snippets, observation IDs, citation IDs, configured exclusion names, source app names, URL metadata, or window-title metadata | `third-party-cloud` or `self-hosted-remote` only when active model route is remote; local model routes do not write ledger rows | full purge removes ledger rows; selective source deletion leaves redacted remote-send audit metadata because remote payloads cannot be recalled | `Cloud reasoning active` label and local `data reasoning-ledger` audit surface |
| `routines.summary` | time-window retrieval | captured memory inside selected time window | routine output store where enabled | inherits active model route | underlying memory deletion removes future grounding | routine run output should inherit cloud banner |
| `meetings.audio` | manual signed ScreenCaptureKit/audio helper | transient system + mic PCM; relative-timestamped transcript and me/others track label | raw PCM stays in private pipes/queues; transcript checkpoints + observations live in the verified-SQLCipher DB | local ASR; saved/retrieved meeting text inherits configured remote embedding, chat, and scheduled-routine routes | purge/prune removes pending rows plus observation/blob/chunk/FTS/vector artifacts | preflight meeting readiness, persistent red app state, metadata-only CLI result |
| `meetings.model_download` | explicit first-use model preparation | public Parakeet model id + normal HTTP metadata; never audio/transcript | public weights in `~/.openbird/models/huggingface` | Hugging Face after approximately 2.51 GB consent; record path is forced offline | remove the model cache or uninstall/zap | app consent/progress/cancel and CLI JSONL progress |
| `data.export` | local memory DB | decrypted observations, text blobs, metadata, and citations selected by user | user-chosen destination | destination path may be synced by iCloud/Dropbox/etc.; encrypted export is planned but not implemented | exported files are outside OpenBird retention control | explicit export warning, 0600 output file, and route label |
| `diagnostics.logs` | runtime diagnostics | reason codes, counts, safe paths/status | local log/terminal only | none | normal terminal/log retention | must never include captured text, raw URLs, or raw window titles |

## Hard Invariants

1. Captured text must never be written to stderr, logs, argv, or environment.
2. Capture allowlists and dangerous-app blocks must run before reading AX text.
3. Pause must fail closed at the source-capture boundary. If the helper cannot
   determine pause state, it must not read AX text.
4. Remote model routes must be enforced at provider construction and disclosed in
   UI/CLI surfaces. UI toggles are not the enforcement boundary.
5. Deletion must remove every derived store: observations, blobs, chunks, FTS,
   vectors, and temporary/export scratch files owned by OpenBird.
6. Release readiness requires positive proof. `release_gate_ok=true` requires
   SQLCipher encryption to be verified and, on macOS, signed-helper/TCC proof.
7. OCR pixels are transient sensor input only. A window-scoped `CGImage` may exist
   inside the helper for an opted-in OCR attempt, but it must never be written to
   disk, logs, stderr, argv, environment, exports, or SQLite; only scrubbed
   recognized text may enter the normal capture pipe.
8. Assistant reads must use bounded local SQL/FTS paths, apply outbound exclusions
   before serialization, and never construct or call an embedding, reranking, or
   completion provider. URLs, window titles, and unknown-app legacy rows stay local.

## Product Decisions

- Public beta release gate: verified SQLCipher is required. Plaintext-0600 is an
  honest local development fallback, not release-green.
- GUI cloud opt-in source of truth: until a durable UI setting is implemented,
  cloud opt-in is per-process/environment only. Any future persistent setting
  must be surfaced as privacy state and tested against the actual subprocess env.
- Deep Brain model strategy: see `docs/design/deep-brain-model-strategy.md`.
  Local deterministic memory remains the privacy boundary and source of truth;
  cloud-capable reasoning triggers must resolve their route from the actual model
  destination, send only exclusion-filtered packets, and disclose the route.
- Reasoning send ledger scope: the first ledger slice records redacted metadata
  only for `deep_brain.ask`, `briefing.model`, and `productivity.coach` remote
  packet send attempts. It does not yet cover generic chat/RAG, meetings, or
  routine cloud sends, and it must not store configured app/source names, source
  app names, raw URLs, or window-title metadata. Users can inspect the redacted
  rows locally with `openbird data reasoning-ledger`; the command is a maintenance
  read and must not construct a model provider or send memory off-device.
- Deep capture / OCR scope: OCR is a deliberate exception to the text-only sensor
  mechanism, not to the text-only storage contract. It is off by default, per-app
  only after the allowlist gate, uses `CGPreflightScreenCaptureAccess` without
  prompting, suppresses while the mic is hot, and stores only recognized text over
  the same scrub/normalize/dedup path as Accessibility text.
- Connector/agent permissions: the read-only assistant MCP surface inherits the
  existing Deep Brain outbound exclusions and hard payload caps. SurfSense-style
  per-action ask/allow/deny rules remain deferred until OpenBird adds write actions;
  every future write must introduce its own explicit confirmation boundary.
