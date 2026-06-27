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
| `capture.pause` | pause sidecar in data dir | no content should be read while source-pause is active | no new rows | none | pause file removed by resume/uninstall cleanup | app/menu bar state must say source-paused only when helper enforces it |
| `ingest.files` | user-selected file or folder | file text/metadata extracted by local parser path | SQLite memory store and derived indexes | none unless model route is remote during embedding/chat | purge/prune cascade; future export is separate egress | CLI/app ingestion status |
| `chat.local` | question plus retrieved memory | question, retrieved chunks, citations | chat output is not persisted by the core CLI today | local Ollama loopback only | underlying memory deletion removes future retrievability | no cloud banner |
| `chat.day_facts` | deterministic day-fact answer branch | question used only for classification, local day-memory productivity facts, derived citation source ids, memory-context counts | SQLite `day_memories`; chat output is not persisted | explicit day-scoped branch returns before `_provider`; query-inferred branch may construct a provider before RAG, but returns before `provider.complete`; neither branch issues completion or embedding requests | purge/prune removes source observations and derived day memory, removing future retrievability | `reasoning_route=local_deterministic` / `Local only` label |
| `chat.day_memory` | deterministic broad day-memory answer branch | question used only for classification, distilled day-memory metrics, workstreams, sessions, open loops, domains/repos, derived citation source ids, memory-context counts | SQLite `day_memories`; chat output is not persisted | explicit day-scoped branch returns before `_provider` using only the non-embedding maintenance stub; query-inferred branch may construct a provider before RAG, but returns before `provider.complete`; neither branch issues completion or embedding requests | purge/prune removes source observations and derived day memory, removing future retrievability | `reasoning_route=local_deterministic` / `Local only` label |
| `chat.remote` | question plus retrieved memory | question, retrieved chunks, citations | local DB plus provider request body | `third-party-cloud` or `self-hosted-remote`, depending on provider destination | local deletion cannot recall payloads already sent to remote providers | `CLOUD ACTIVE` banner and per-answer/provider disclosure |
| `routines.summary` | time-window retrieval | captured memory inside selected time window | routine output store where enabled | inherits active model route | underlying memory deletion removes future grounding | routine run output should inherit cloud banner |
| `meetings.audio` | ScreenCaptureKit/audio helper | audio/transcript once enabled | local DB/transcript path | none for local ASR; remote if STT provider is remote | purge/prune/delete must remove transcript-derived memory | preflight TCC/audio readiness and provider route |
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
- Connector/agent permissions: SurfSense-style ask/allow/deny rules are deferred
  until OpenBird adds connector write actions or agent tools. Adding the rules
  without the scope would expand architecture without improving current privacy.
