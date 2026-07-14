# Assistant MCP v2 — pagination, dedup, and trusted-metadata rollups

Status: approved (Codex gpt-5.6 adversarial review, 8 rounds to consensus, 2026-07-13)
Scope: `openbird/assistant.py`, `openbird/cli.py` (install warning), `openbird/memory/store.py`,
`openbird/memory/schema.sql` + `migrations.py` (v8, index-only),
`mac-app/Sources/OpenBirdApp/Views/SettingsView.swift` (consent copy), tests,
`docs/assistant-connectors.md`, `docs/privacy-routes.yaml`, `docs/design/privacy-data-flow.md`
Depends on: shipped connector (#259), activity spans (capture-efficiency redesign, v4 schema)

## Why

Field feedback from a connected Claude Desktop session identified five gaps, ranked by
leverage. Investigation shows **most of the requested data already exists in the local
schema and is simply not exposed by the MCP layer**:

| Feedback | Local reality |
|---|---|
| 1. `minutes=1440` returned only ~5 min, `truncated: true`, no way to page | `recent_capture_text` is newest-first `LIMIT n` with no cursor; `minutes` only bounds, never paginates |
| 2. Same page body returned verbatim 4× in one call | Storage already dedups content (`content_blobs` by `content_hash`); the serializer re-inflates every observation row with the full blob text |
| 3. No dwell/attention signal, only snapshot counts | `activity_spans` (v4) stores ground-truth "app frontmost from t1 to t2" rows with afk/meeting flags; `MemoryStore.active_seconds` already computes gap-capped active time |
| 4. Excerpt fuses title/URL/chrome into one blob | `observations.window` / `observations.url` are already separate columns; they are withheld by a **deliberate, documented privacy invariant** ("URLs and window titles stay local") |
| 5. No aggregate tool; day-level analysis forces raw-text egress | Span rollups can answer "what did I focus on" from **trusted metadata only** — lower injection surface than paging excerpts |

The architectural principle (from the same feedback, and consistent with the connector's
threat model): **fatten the trusted metadata layer, not the untrusted excerpt pipe.**
Every excerpt is `captured_content_is_untrusted: true`; rollups computed locally from
span metadata raise analytical power while *lowering* raw-text egress.

## In scope (this PR)

### 1. Keyset cursor pagination for `openbird_recent_capture`

New optional `cursor: str` tool argument; new `next_cursor: str | null` response field.

- The cursor is a **server-side handle**: an opaque random token
  (`secrets.token_urlsafe(24)`) that carries **no data at all**. The actual state —
  keyset consumption boundary `(ts, id)`, the window `[start_ts, end_ts]` frozen at
  first-call time, and the mint timestamp — lives in a bounded in-memory table on
  the service instance (never persisted). This is deliberately stronger than a
  signed self-describing token: an HMAC provides integrity but not confidentiality,
  and the boundary row can be an **excluded** observation, whose exact id and
  timestamp must not be readable out of a base64 payload. A random handle leaks
  nothing, and forgery is impossible without a table hit.
- **Bounds on the handle table**: capacity-capped LRU (64 entries — one walk needs
  exactly one live handle; concurrent assistants a handful), 15-minute TTL from the
  first page's mint time (`issued_at` carries across hops, so the TTL bounds the
  whole walk), entries dropped on use-after-expiry, guarded by a lock for safety
  under concurrent tool calls. Unknown, expired, or evicted handles fail closed
  with a bounded tool error — the assistant re-issues a fresh first call. The stdio
  server lives for the assistant session, so pagination works within a
  conversation; a restart empties the table and page walks simply restart.
- Window semantics enforced server-side at mint time (the only place state is
  created): `end_ts - start_ts <= MAX_MINUTES * 60` always holds because the window
  is computed from a validated `minutes` argument — a cursor can never widen any
  window beyond what a legitimate first call could request. Tool-argument
  validation on the handle itself: string, length cap (128), no other structure.
- First call: window is `[now - minutes*60, now]` exactly as today; response includes
  `window_start_ts` / `window_end_ts` and `next_cursor` when more rows remain.
- Subsequent calls: `cursor` supersedes `minutes`; the embedded `start_ts` keeps the
  window stable across pages (a moving `now - minutes` floor would silently drop the
  tail between pages). Keyset descent `(o.ts, o.id) < (cursor.ts, cursor.id)` makes
  concurrent capture writes benign — later pages are strictly older.
- `next_cursor: null` means the window is exhausted; `truncated` keeps meaning "this
  page hit a row/char cap", which is now recoverable instead of terminal.
- A malformed, unknown, expired, or evicted cursor raises `ValueError` (assistant-visible
  tool error), never a stack trace. The cursor grants nothing the tool cannot already
  read: every page re-applies the same exclusion filters, caps, and source gate.

Store change: `recent_capture_text` gains optional `before: tuple[ts, id]` applied as
`AND (o.ts < ? OR (o.ts = ? AND o.id < ?))`.

**Index (schema v8):** the existing `idx_observations_ts` indexes `ts` alone — the
`ORDER BY ts DESC, id DESC` keyset plus the `source = 'capture'` filter would force a
temporary B-tree re-sort of equal-timestamp groups on every page (pathologically
quadratic when many rows share one timestamp). Add migration v8 creating
`idx_observations_source_ts_id ON observations(source, ts, id)` via the established
append-only migration ladder (`SCHEMA_VERSION` 7 → 8); `schema.sql` gains the same
`CREATE INDEX IF NOT EXISTS` for fresh DBs. Index-only additive migration, no data
rewrite. Tests must cover a page walk where more than `SCAN_CAP` rows share a single
timestamp (keyset tie-break correctness, not just performance).

### 2. Blob-level dedup in content serialization

`_serialize_rows` currently emits one result per observation, so a window seen 50
times burns 50 of the 20-row budget on one blob. Change `openbird_recent_capture`
(only — see the search note below) to group by `content_hash` **within a page**:

- Each result becomes one **distinct content group**:
  `{observation_id, timestamp, app, source, excerpt, seen_count, first_ts, last_ts}`
  where `observation_id`/`timestamp` are the newest occurrence (citation anchor
  preserved) and `seen_count`/`first_ts`/`last_ts` are aggregates over the
  **exclusion-passed occurrences consumed by this page** — computed in Python from
  the rows already scanned, no extra query, and never counting occurrences an
  exclusion filter removed (a window-wide SQL aggregate would leak excluded-app
  occurrence counts, since glob/regex app exclusions cannot be pushed into SQL).
- **Consumption-boundary pagination** (no group ever lost, no wedge): fetch up to
  `SCAN_CAP` (200) rows below the cursor, walk newest → oldest. Each row either
  joins its hash's existing group (updating `seen_count`/`first_ts`) or starts a new
  group. Stop only at a row that would start group `limit + 1` (or whose excerpt
  cannot fit the remaining char budget). `next_cursor` is the keyset position of the
  **last consumed row**: everything above it is represented in emitted groups (or was
  excluded), everything below it — including the row that triggered the stop — is
  untouched and appears on the next page. A page whose rows all join existing groups
  still consumes them, so the cursor always advances while rows remain.
- Group semantics are **per page**: a blob recurring across page boundaries appears
  once per page with region-scoped stats (the assistant sums if it cares). This is
  the honest, bounded trade — it fully collapses the observed failure (four verbatim
  copies inside one page) without cross-page state, window-wide aggregate queries, or
  emission rules that can skip data.
- `apps` may differ across occurrences of one hash (same text seen in two apps). The
  group reports the newest occurrence's app; exclusion filtering stays
  **per-observation before grouping**, so an excluded app's occurrences never
  contribute to visible groups, counts, or timestamps.
- `openbird_search_capture` is **unchanged**: `lexical_capture_text` already
  collapses to one newest occurrence per matched chunk (`occurrence_rank = 1`), has
  no time window for aggregates to range over, and keeps its bounded top-k contract.

This multiplies the useful signal per call without raising any cap: same 20-row, same
12k-char budget, now spent on 20 *distinct* excerpts.

### 3. New tool: `openbird_activity_summary` (metadata only)

The "what did I focus on today" tool. Reads `activity_spans` only; **returns zero
captured text, zero window titles, zero URLs** — bundle ids, durations, and counts
only. Same `minutes` bound (1–1440), no cursor needed (output is a fixed-size
rollup, not a row stream).

Response shape:

```json
{
  "ok": true,
  "content_returned": false,
  "window_start_ts": 0.0, "window_end_ts": 0.0,
  "foreground_seconds": 0.0,
  "afk_seconds": 0.0,
  "meeting_seconds": 0.0,
  "redacted_seconds": 0.0,
  "excluded_seconds": 0.0,
  "context_switches": 0,
  "longest_focus": {"bundle_id": "…", "start_ts": 0.0, "end_ts": 0.0, "seconds": 0.0},
  "apps": [
    {"bundle_id": "…", "foreground_seconds": 0.0, "span_count": 0, "meeting_seconds": 0.0}
  ],
  "other_apps_seconds": 0.0,
  "other_apps_count": 0
}
```

- **Bounded span read**: the tool does not reuse the unbounded
  `spans_in_range` (`SELECT *`, no `LIMIT`). A new store read returns spans
  overlapping the window ordered by `start_ts` with `LIMIT MAX_SUMMARY_SPANS + 1`
  (`MAX_SUMMARY_SPANS = 5000` — a 24h day at the daemon's merge cadence produces
  well under that). If the sentinel row appears, the tool **fails closed** with a
  bounded tool error telling the assistant to request a narrower window — never a
  partial rollup that silently reads as complete.
- Spans overlapping the window are clipped to it before aggregation; spans whose
  clipped duration is not `> 0` are dropped entirely (they contribute no seconds, no
  `span_count`, no switches — matching the existing day-memory span-metric rule).
- **Total bucket precedence** — a span can satisfy several bucket predicates at once
  (a tier-0 span can be AFK and carry an exclusion-matched bundle; schema permits
  every combination), so classification is a strict first-match order and each
  clipped second lands in exactly one bucket:
  1. `excluded_seconds` — bundle matches an outbound exclusion (any tier, any afk
     state): the user's explicit "never egress this app" outranks everything;
  2. `redacted_seconds` — tier-0 span or `NULL` bundle (paused);
  3. `afk_seconds` — visible-app span flagged afk;
  4. visible foreground — everything else (feeds `apps[]`, `other_apps`,
     `foreground_seconds`).
  Tests cover the overlap combinations explicitly: tier0+afk (→ redacted),
  excluded+afk (→ excluded), tier0+excluded (→ excluded), meeting on tier-0 /
  excluded spans (→ moves no returned meeting total).
- **Meeting is an orthogonal overlay, not a bucket.** Capture semantics
  deliberately keep `meeting = 1` through AFK (sitting in a Zoom call without
  touching input is still a meeting — `spans.py` preserves it and the day-memory
  metric test requires it to count), so meeting time is aggregated across steps
  3–4: a **visible** span (not excluded, not tier-0) flagged meeting contributes
  its clipped seconds to its app's `meeting_seconds` whether or not it is afk.
  The duration partition (excluded/redacted/afk/foreground) is unaffected — a
  second can be both `afk_seconds` and meeting-labeled; the doc labels
  `meeting_seconds` as an overlay, not a partition member. So that afk-only
  meetings don't vanish from the response, `apps[]` ranks by
  `foreground_seconds + meeting_seconds` (an hour-long fully-afk Zoom call still
  surfaces with `foreground_seconds: 0, meeting_seconds: 3600`). Top-level
  `meeting_seconds` remains the sum over returned `apps[]` entries only.
  Hidden-span meeting time (tier-0 / excluded) still moves no returned total.
- **Privacy tiers respected by construction**: tier-0 spans (dangerous/blocklisted/
  private/paused apps) carry no window/url/identity by CHECK constraint; the summary
  aggregates their clipped durations into `redacted_seconds` without naming bundle or
  reason breakdown beyond what `capture_status` already reports. Tier-1 spans expose
  only `bundle_id` — already an egressed field on every content result.
- **Existing app exclusions apply with identical semantics**: span `bundle_id`
  matching reuses the exact matcher content egress uses
  (`openbird.capture.redact._bundle_matches_any`, the helper behind
  `filter_rows_for_deep_brain`), so exact, `glob:`, and `re:` exclusion entries all
  hold. Matching spans aggregate into `excluded_seconds` and are never named.
  `NULL` bundle ids (paused) fold into `redacted_seconds`.
- **Malformed exclusion patterns fail the tool closed.** `_bundle_matches` returns
  `False` on an unparseable `re:` pattern — correct for an allowlist gate, but
  fail-*open* when the entry is an exclusion (the span/observation would egress as
  if unexcluded). Before serializing anything, every assistant tool (content and
  summary) validates the configured `deep_brain_excluded_apps` `re:` entries with
  `re.compile`; if any is malformed the call returns a bounded, reason-code-only
  tool error ("invalid exclusion pattern configured — fix deep_brain_excluded_apps")
  with zero rows/spans. Tests cover a malformed `re:` entry on both the content and
  summary paths.
- `apps` is capped (top 30 by `foreground_seconds + meeting_seconds`, so afk-only
  meetings still rank); visible apps beyond the cap fold
  into `other_apps_seconds` / `other_apps_count` **without naming bundles**, so the
  partition identity survives truncation:
  `sum(apps[].foreground_seconds) + other_apps_seconds == foreground_seconds`, and
  `foreground + afk + redacted + excluded` sums to clipped span time. AFK time is
  excluded from per-app foreground totals and from `context_switches` /
  `longest_focus` (an afk span is not attention). **`context_switches` and
  `longest_focus` are computed over the visible sequence only** — after dropping
  afk, zero-clipped-duration, tier-0/redacted, and exclusion-matched spans — so a
  hidden bundle's transitions can never move a returned number (the same no-leak
  rule as the duration fields, applied to counts). A switch is a transition between
  consecutive spans of that filtered sequence with different bundle ids. A
  visible-app AFK span additionally **ends the current focus run** (a nap must not
  fuse two focus blocks; AFK is a disclosed bucket, so the break leaks nothing) —
  hidden excluded/redacted spans deliberately do not break a run, since the break
  itself would reveal that a hidden app intervened.
  Top-level `meeting_seconds` remains the sum over **returned** `apps[]` entries
  only; meeting time in the `other_apps` tail is deliberately not totaled (a
  nameless meeting total would invite subtraction games against future fields —
  same reasoning as the excluded/redacted rule).
- **No observation-derived stats.** `MemoryStore.active_seconds` was considered and
  rejected for this tool: it aggregates over *all* observations with no app/source/id
  exclusion path, so a top-level number would leak excluded-app activity that the
  span aggregation carefully hides in `excluded_seconds`. Every duration in the
  response is derived from the same exclusion-filtered span partition:
  `foreground_seconds` is the total non-afk visible-app time
  (`sum(apps[].foreground_seconds) + other_apps_seconds`), and
  visible + `afk` + `redacted` + `excluded` partitions the clipped span time, so
  nothing about excluded apps is recoverable by arithmetic. The same rule applies to
  every secondary total: top-level `meeting_seconds` sums **only the visible
  `apps[].meeting_seconds`** (tier-0 spans keep `meeting=1` in storage — a coarse
  non-allowlisted Zoom span is still a meeting — so an all-span meeting total would
  let subtraction recover redacted/excluded meeting time; visible-only makes the
  identity `meeting_seconds == sum(apps[].meeting_seconds)` hold trivially and
  leaks nothing).

This directly answers feedback items 3 and 5 with a **lower**-egress surface than any
excerpt paging, and it is the SAFE-MCP-aligned move: analysis on trusted metadata,
excerpt dips only when a specific question needs text.

### 4. Privacy truth surface + docs

Behavioral rollups are a **new category of egress** — bundle ids, per-app durations,
AFK/meeting metrics, switch counts, and focus timestamps leave the local boundary
even though `content_returned` is false. Every surface that states what leaves must
say so:

- `openbird_activity_summary` responses carry their own egress notice field
  (sibling of `ASSISTANT_EGRESS_NOTICE`, worded for behavioral metadata: app
  identifiers, durations, and activity patterns leave the local boundary; no
  captured text). The tool description says the same.
- `docs/assistant-connectors.md`: tool-table row for the new tool; the privacy
  boundary section extends beyond "content tools" to name behavioral-metadata
  egress explicitly; cursor contract and dedup group fields documented; re-assert
  the unchanged invariant that URLs and window titles never cross the boundary.
- `docs/privacy-routes.yaml` and `docs/design/privacy-data-flow.md`: add the
  assistant summary route (activity_spans → span aggregation → assistant boundary,
  metadata-only) so the privacy inventory matches the code.
- **Consent surfaces** (the warnings a user reads before connecting) must name
  behavioral metadata too, or consent is obtained under a false description:
  - the CLI install confirmation (`openbird assistant install-claude`), which
    currently prints the excerpt-only `ASSISTANT_EGRESS_NOTICE`;
  - the macOS app's Claude connect confirmation and ChatGPT setup copy in
    `mac-app/Sources/OpenBirdApp/Views/SettingsView.swift`, which promise only
    bounded excerpts / app ids / timestamps.
  All three add "app usage durations and activity patterns" to the disclosed
  egress. Swift copy change gates on `swift build`; CLI warning covered by the
  existing install-flow tests.

## Out of scope (deliberate, documented)

- **Structured `window_title` / `url` fields on content results** (feedback #4).
  Blocked on the shipped privacy promise "URLs and window titles stay local." If ever
  offered it must be a separate consented opt-in (config default-off plus an explicit
  egress warning in the settings UI), reviewed on its own — not smuggled into a
  serializer change. `url_host`-only exposure on tier-1 data is the likely shape.
  Follow-up design, separate PR.
- **Capture-side diff/chrome reduction** (feedback #2's second half: menu labels and
  nav chrome inside the AX text itself). That is capture-pipeline territory
  (`docs/design/capture-efficiency-redesign.md`), not MCP serialization. Blob dedup
  removes the *repetition*; chrome inside a single blob remains until the capture
  redesign addresses it.
- **`search_capture` pagination and dedup.** BM25 rank order has no stable keyset;
  offset pagination over shifting corpora lies. Search keeps its current top-k
  contract and its current response shape, entirely unchanged — its worst redundancy
  is already collapsed in SQL (`occurrence_rank = 1` keeps only the newest
  occurrence per matched chunk), and it has no time window for group aggregates to
  range over.
- **Seeding default exclusions.** The feedback's standing flag (`excluded_* all 0`)
  is user configuration, not a code defect; tier-0 structural redaction already
  protects dangerous apps at capture time, before anything reaches the DB. A
  recommended-exclusions onboarding nudge is a product decision — tracked separately.

## Invariants preserved (review checklist)

1. Read-only: no new write path; new store methods are SELECT-only maintenance reads.
2. No model calls: summary and pagination use direct SQLite; no embed/rerank/completion.
3. Bounds: every new argument validated with the existing `_bounded_*` idiom; page
   scan, aggregate query, and `apps` list all hard-capped; response size bounded.
4. Exclusions before serialization, on every page, for both content and span reads.
5. `source = 'capture'` gate unchanged; legacy no-app rows still fail closed.
6. Untrusted-content labeling unchanged on content tools; the summary tool sets
   `content_returned: false` like `capture_status`.
7. stdio-only transport, unchanged tool count discipline (3 → 4, all read-only).
8. Cursors are opaque server-side handles (random token, no decodable payload —
   excluded-row boundaries leak nothing) with server-held, semantically bounded
   state; no tool argument can widen any window beyond the shipped 24-hour maximum.

## Test plan

- Cursor: walk a 3-page window to exhaustion; page stability under concurrent newer
  inserts; malformed/oversized handle rejection; **unknown handle fails closed**
  (guessing yields nothing); **TTL expiry and LRU eviction fail closed**; **a page
  whose rows are all app/source/id-excluded returns a `next_cursor` that reveals
  nothing about the excluded rows** (token is random, no decodable payload);
  frozen `start_ts`/`end_ts` honored;
  `next_cursor` progress on an all-duplicates page; **correct keyset tie-break when
  more than `SCAN_CAP` rows share one timestamp**; migration v8 index present on
  both fresh and migrated DBs.
- Dedup: 50 observations of one blob = 1 group with `seen_count=50`, correct
  first/last; **no group lost when a page scans more distinct hashes than `limit`**
  (the Codex blocker scenario: every hash eventually emitted on a later page);
  excluded-app occurrences contribute nothing to counts; newest-occurrence citation
  anchor; `search_capture` response shape unchanged.
- Summary: clipping at window edges; **zero-clipped-duration spans contribute
  nothing** (no seconds, counts, or switches); **hidden-span transitions (excluded,
  tier-0, afk) never move `context_switches`/`longest_focus`**; **span-count
  overflow (> MAX_SUMMARY_SPANS) fails closed** with no partial rollup; afk
  excluded from foreground/switches/focus;
  tier-0 → `redacted_seconds` with no bundle leak; excluded app (exact, `glob:`,
  `re:`) → `excluded_seconds` with no bundle leak; **no observation-derived stat in
  the response** (excluded-app capture activity unrecoverable); duration partitions
  sum to clipped span time; **excluded/redacted meeting spans move no returned
  meeting total** (top-level `meeting_seconds` equals the sum of visible
  `apps[].meeting_seconds`); top-30 cap; empty window.
- Regression: existing exclusion/caps/no-model tests stay green; MCP server exposes
  exactly the four read-only tools.
