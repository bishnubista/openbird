# Founder Context Recap

Status: implementation plan (2026-07-29)

## Product boundary

The first founder-context moment is an explicit Ask invocation:

> Bring me back up to speed on what I was working on.

OpenBird remains invisible until invoked. This is not a dashboard, notification,
daily digest, or general autonomous agent. The answer should reconstruct the
most likely current work thread from recent local evidence and make uncertainty
visible.

The response has four evidence-backed parts:

1. likely active focus;
2. recent activity;
3. decisions or concrete progress;
4. open loops.

Each rendered claim must carry at least one source id that was actually present
in the bounded prompt. The likely-focus claim requires at least two in-context
source ids. This is a provenance-membership guarantee, not an entailment proof:
OpenBird can verify that the cited occurrence was in the model context, but a
small deterministic checker cannot prove that a paraphrase is semantically
entailed. A field with no valid in-context source id is omitted rather than
rendered. Existing occurrence citation validation remains the final grounding
gate.

## Retrieval and answer design

- Add the exact founder-context question as the first empty-state Ask suggestion.
  There is no unsolicited surface.
- Founder-context routing runs before the existing generic synthesis intent, but
  only for a narrow phrase set: "bring me back up to speed [on what I was
  working on]," "where did I leave off," "catch me back up [on my work]," and
  "get me back up to speed [on my work]." Existing "catch me up," "recap my
  day/week," and "what did I work on" phrases retain their current today/three-day
  generic synthesis behavior. Topic-specific questions continue through the
  semantic or day-scoped paths.
- An explicit temporal phrase in the question (`yesterday`, `today`, `last
  week`, and the existing temporal vocabulary) disqualifies the founder-default
  branch and retains the existing temporal resolver. The five-day default never
  overrides a user-stated time scope.
- Resolve the default founder-context window to the trailing five days. Explicit
  caller-supplied day scopes continue to win and use the existing day path.
- The founder-context branch explicitly bypasses multi-day summary-first
  retrieval. Cached block/week prose never silently replaces the raw-occurrence
  contract, and all recap citations remain occurrence-level.
- Read each covered local day independently with a newest-first keyset-paginated
  reader across capture, meeting, ingest, and MCP observations. Each page is
  deduplicated immediately by `(session_id, content_hash)`; scanning stops at 240
  distinct rows or 1,200 scanned rows per day. Across five days the hard maximum
  is 1,200 distinct / 6,000 scanned rows, so a duplicate-heavy newest app cannot
  consume the whole multi-day budget and a high-volume current day cannot erase
  weekend/older-day coverage.
- Add `MemoryStore.founder_context_page(start_ts, end_ts, limit, before)`.
  It performs one bounded statement over the four admitted source types:
  `source IN ('capture','meeting','ingest','mcp')`, with keyset predicate
  `(ts < ? OR (ts = ? AND id < ?))`, ordered `ts DESC, id DESC`, and SQL
  `LIMIT ?`. Add schema/migration index
  `idx_observations_ts_id(ts DESC, id DESC)` so the reader never materializes an
  unbounded day before Python applies its caps.
- Apply `deep_brain_excluded_apps`, `deep_brain_excluded_sources`, and
  `deep_brain_excluded_observation_ids` at founder-context read time. Malformed
  app exclusion patterns fail closed. This is stricter than generic Ask and
  prevents the wider window from resurfacing rows the user explicitly excluded.
- Self-capture is always removed. Same-session duplicate content is removed
  inside the pagination loop before selection.
- Select at most eight prompt sources. Selection is deterministic and
  recency-weighted. At most three of the eight slots may be cue-reserved (one
  each for decision, progress, and open-loop cues); a reserved candidate must
  also clear the normal signal/recency floor, and the reserved set admits at most
  one row per `(app, source)` origin. Cue words inside untrusted content therefore
  cannot guarantee or dominate prompt inclusion. The selector does not run a
  model.
- Extend `provider.complete(...)` with explicit per-call `max_attempts` and
  `timeout` overrides, then invoke it once with a founder-specific budget of at
  most two structured attempts and 20 configured seconds per attempt. The
  provider's five-second guard makes the hard model ceiling
  `2 × (20 + 5) = 50` seconds, leaving 40 seconds of the macOS app's existing
  90-second child deadline for interpreter/import startup, Keychain/store open,
  the bounded five-day scan, selection, and JSON rendering. The recap does not
  add another completion. `LLMTimeoutError` aborts the current
  `provider.complete` call; the founder branch catches it and returns a
  grounded=false answer stating that the local recap timed out and can be
  retried, so the app does not kill the process first or show a generic child
  timeout. Use the configured OpenBird model route and existing cloud opt-in
  enforcement. A defensive parser type-checks every claim object, text value,
  and citation list even if optional `jsonschema` validation is unavailable,
  caps every section, and drops malformed/uncited fields independently. If no
  structured claim survives, return the normal explicit ungrounded message
  rather than an empty recap or a second model call.
- Keep the existing injection-resistant context fence, prompt overrides, citation
  objects, source chips, and click-through navigation.

The five-day span does not increase model context: the prompt remains capped at
eight sources with the existing per-source text cap. It only broadens a bounded
local selection scan, so it avoids the documented multi-day model-timeout
failure while retaining enough evidence to reconstruct a work thread after a
weekend or short gap. No screenshot or audio capture is introduced.

## Scheduled capture-quality evaluation

Use a separate per-user LaunchAgent, `ai.openbird.founder-context-eval`, instead
of cron or the long-running model-backed routines daemon.

- The agent runs a short-lived command at load and then at most every six hours,
  best-effort. launchd coalesces sleep-time intervals into one run after wake.
- The scheduled command performs aggregate SQL and filesystem reads only. It
  never calls an embedding, reranker, or completion model.
- The scheduled command opens an existing store only through
  `_store_maintenance()` / `_MaintenanceProvider`, which cannot embed and ignores
  configured cloud roles and embedding-cohort drift. Every store-open exception
  is mapped to a closed reason code and still writes a file/storage-only
  `not_ready` snapshot. Failures after a successful open are reported separately
  as `evaluation_failed`, so a SQL/schema problem is never mislabeled as a
  Keychain problem. If the snapshot filesystem itself is unavailable, the
  command instead returns a content-free `snapshot_write_failed` reason without
  leaking a path or operating-system error; no process can guarantee a file on
  an unwritable filesystem.
- launchd runs it with background process type, low-priority I/O, a positive
  nice value, decimal umask `63` (`0o077`), no `KeepAlive`, and stderr directed
  to `/dev/null`; the bounded snapshot is the diagnostic surface.
- It writes one atomic, mode-0600 snapshot under
  `<data_dir>/logs/founder-context-eval.json`. The prior snapshot is used only
  for count/byte deltas and is replaced, so evaluation storage is bounded. The
  aggregate read and replacement share a short SQLite `BEGIN IMMEDIATE` lock
  with deletion: either evaluation finishes first and deletion removes its
  snapshot, or deletion finishes first and evaluation reads only surviving rows.
- Interactive stderr is metadata-only: state, reason codes, counts, and elapsed
  time. The scheduled job emits no stdout/stderr. Raw captured text, snippets,
  window titles, URLs, prompts, and answer prose are forbidden.
- The optional manual answer probe runs only after the recorded snapshot
  transaction releases its lock. Probe fields are returned transiently and are
  never persisted in the scheduled metadata snapshot.
- Installation/loading remains explicit through CLI commands. `openbird
  uninstall` removes both the loaded job and plist while preserving captured
  data unless the existing purge option is chosen.
- Scheduled mode sets a read-only-Keychain guard before any store open. It may
  fetch an existing DB key but may never create or update a key. If the
  encrypted store cannot be opened under the scheduled code identity, the
  command writes a file/storage-only `not_ready` snapshot with reason
  `encrypted_store_unavailable` and exits cleanly; it never falls back to
  creating/opening a plaintext DB over encrypted bytes.
- Install is the only interactive Keychain preflight. When the DB header is not
  the SQLite magic, `eval founder-context install` temporarily allows a longer
  read-only key lookup from the exact absolute CLI executable that the plist
  will run. Installation is refused unless that executable can open the
  encrypted store and complete the metadata evaluation while the user is present
  to grant access. That child honors the installer's bounded 30-second lookup.
  Scheduled runs use the same executable, a forced two-second read-only lookup,
  and never write a key; a missing or revoked grant produces
  `encrypted_store_unavailable`.
- `db_is_plaintext_or_absent()` is the scheduled discriminator. A genuine
  nonempty plaintext store opens normally without SQLCipher. An absent or
  zero-length DB produces a `not_ready` snapshot with reason `store_absent` and
  is never opened or created. A nonempty file without the SQLite magic is
  treated as encrypted: failure to obtain and verify its existing key produces
  the unavailable snapshot, never a plaintext open attempt.

The snapshot measures:

- observed source buckets and app bundle ids derived only from stored
  observations that survived capture policy (never from rejected/private
  capture-attempt bundle ids);
- recent observation, distinct-context, session, and source counts;
- aggregate richness and decision/progress/open-loop cue counts computed inside
  SQLite without returning text;
- last-capture freshness and daemon liveness;
- database/WAL/SHM bytes, reclaimable pages, recent text bytes, and deltas;
- bounded capture-attempt counts, emitted bytes, average/max extraction time,
  and failure/budget reason counts, with no attempt bundle ids;
- evaluator elapsed time and peak resident memory;
- a `ready`, `partial`, or `not_ready` founder-context assessment with closed
  reason codes.

All observation/app/source and capture-attempt aggregates are restricted to the
five-day evaluation window and served by timestamp indexes. The evaluator does
not run the existing all-history capture-health app aggregation; daemon and pause
health come from bounded metadata sidecars instead.

An optional manual real-store answer probe may invoke the founder-context answer,
but its output is restricted to pass/fail, grounding mode, occurrence/derived
citation counts, citation app ids, timestamps, and elapsed time. It is never part
of the LaunchAgent command.

## Privacy and lifecycle

The answer reuses the existing Ask route. Captured rows stay local when the
configured model is local; a remote model is still blocked without explicit
cloud opt-in and retains the existing visible route label. All source text is
fenced as untrusted data.

The evaluator adds no captured-content store. Its snapshot contains metadata
only and lives inside the existing data directory, so `--purge-data` removes it.
Every selective/full observation delete and retention prune invalidates the
snapshot before the database transaction commits. If cache removal fails, the
source deletion rolls back rather than claiming success while deleted app/source
metadata persists. Recorded evaluation holds the same short database writer lock,
so it cannot reintroduce a pre-delete snapshot after that invalidation. If the
later commit fails, an absent derived snapshot is the privacy-safe direction. The
next scheduled run rebuilds it from surviving rows.
Existing allowlist, blocklist, private-window, dangerous-app, self-capture,
redaction, retention, deletion, and derived-citation invalidation contracts
remain unchanged.

## Acceptance and validation

1. A deterministic fixture contains a recent active thread, progress, a decision,
   an open loop, older competing work, repeated content, cue-injection noise, and
   self-capture. Per-day pagination/selection must include the useful roles
   within the source cap without letting one cue origin dominate.
2. Structured-response tests prove malformed fields, uncited fields,
   hallucinated citation ids, and a one-source "likely focus" are dropped.
3. Routing tests prove the exact prompt gets the five-day path, explicit day
   scope still wins, each founder phrase carrying an explicit temporal word
   retains that temporal scope, nearby generic-synthesis phrases retain current
   behavior, topic questions remain semantic, and stored block summaries cannot
   divert the founder path to summary-first.
4. Evaluation tests prove payloads contain only the approved metadata schema,
   readiness reason codes are deterministic, snapshots are atomic/0600/bounded,
   prune/purge invalidates snapshots, absent scheduled mode cannot create a DB,
   encrypted scheduled mode cannot mint a key, every store-open error records a
   closed `not_ready` snapshot, and the scheduled plist has no
   keep-alive/model command.
5. Provider/app-boundary tests pin the two-attempt, 20-second founder budget and
   the explicit grounded=false timeout response. The asserted ceiling is
   `attempts × (timeout + 5-second provider guard) = 50 seconds`, leaving a
   40-second startup/store/scan allowance below the 90-second Swift process
   deadline.
6. CLI tests cover run/record/install/uninstall behavior without touching the
   real launchd domain.
7. Swift tests pin the lead suggestion and existing stdin/citation flow.
8. Run focused Python and Swift tests, then full pytest, shellcheck for changed
   shell files (if any), `swift build`, and the applicable Swift test suites.
9. Run a privacy-safe real-store metadata evaluation. Run the optional answer
   probe only if the local provider is available; report only approved metadata.
10. Obtain blocking-only Claude approval for this plan and for the final diff.
