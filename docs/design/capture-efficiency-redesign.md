# Capture efficiency & detail redesign — research synthesis and proposal

Status: reviewed to consensus (Codex adversarial review, 3 rounds → approve, 2026-07-01)
Date: 2026-07-01
Accepted low-risk tradeoffs (from review): helper/Python policy parity needs tests
during implementation; Phase B's migration picks the concrete storage shape for
tier-1 identity keys. (Resolved in Phase B: a nullable `identity_key TEXT` column,
part of the merge identity tuple and the tier-0 CHECK; EXTRACTION of file/repo/
document identities is explicitly deferred to a later phase — the column is always
NULL in Phase B.)
Goal: help the user be productive and give **detailed, trustworthy analysis of how time
on the MacBook was spent** — while making capture cheaper, not heavier.

This doc synthesizes four parallel research passes: (1) a map of OpenBird's current
capture pipeline, (2) a deep-dive on screenpipe's engineering history, (3) a survey of
the personal-screen-memory / time-tracking landscape (Rewind, littlebird, ActivityWatch,
Dayflow, Timing/Rize/RescueTime, Microsoft Recall), and (4) current macOS (Sequoia
15 / Tahoe 26) capture-API best practice.

---

## 1. Where the industry converged (and why it matters to us)

Every surviving product in this category, approaching from opposite ends, landed on the
same hybrid stack:

| Layer | Winning pattern | Who proved it |
|---|---|---|
| Time accounting | **Cheap always-on metadata event stream** (app/window/URL/AFK), heartbeat-merged: store *state changes*, not samples | ActivityWatch (10+ yrs), Timing |
| Content | **Change-triggered text capture, accessibility-first**, OCR only when AX is empty | screenpipe v2 (Feb 2026), littlebird |
| Pixels | Ephemeral at most (short-retention buffer feeding an LLM) or none at all | Dayflow (3-day buffer), littlebird (none) |
| AI context | **Progressive summarization**: events → 15–30 min blocks → day narrative → queryable briefings | Dayflow, screenpipe pipes, littlebird routines |

The cautionary tales are equally consistent:

- **Rewind.ai** (dead, Dec 2025): continuous 0.5 Hz screenshots + video encode + OCR of
  every frame = ~210 MB/hr, ~20% CPU, 20–40% battery hit. Best recall fidelity in the
  category; unsustainable product.
- **screenpipe v1 → v2**: spent 18 months and thousands of issues discovering that
  continuous FPS capture with synchronous OCR was the original sin. Their Feb-2026
  rewrite (`EVENT_DRIVEN_CAPTURE_SPEC.md`) is essentially OpenBird's architecture:
  event-triggered, AX-text-first, OCR fallback, no video. Their budgets after the
  rewrite: **<0.5% CPU idle, <5% active, ~300 MB per 8-h day, AX walk hard-capped at
  200 ms, first data <5 s after start.**
- **Microsoft Recall**: content-classifier redaction demonstrably leaks (credit-card
  forms without "checkout" context were captured); *structural* exclusions (app,
  private-window, secure-field level) are the reliable layer. OpenBird already does
  this correctly (allowlist + blocklist + dangerous-app backstop + AXSecureTextField
  skip).
- **littlebird.ai** (the direct inspiration): active-window **text only, never pixels**
  — but cloud-stored on AWS. Local-first is OpenBird's wedge; their product framing
  (end-of-day narrative + routines/briefings, *no* productivity score) is worth
  copying; their storage locus is not.

**Validation**: OpenBird's baseline choices — AX text-first capture, no persisted
screenshots/images, allowlist-first privacy, observations vs. deduped content blobs
— are the converged industry answer. Phase C2 below adds the explicit exception:
opt-in, per-app, transient window stills for OCR when AX text is empty; only
recognized text is stored.
The gaps are (a) *when* we capture (2-s blind polling), (b) *time-accounting ground
truth* (we have observations, not measured durations), and (c) *context shaping for
the LLM* (raw chunks, no intermediate summarization layer).

## 2. Current-state audit (what's inefficient / what's missing)

Current pipeline (see `openbird/capture/daemon.py`, `capture-helper/`):

1. **One-shot helper spawned every 2 s.** Each poll pays process launch + AX warm-up +
   full frontmost-window tree walk (bounded: 5 K nodes / depth 40 / 2 s), even when
   nothing changed. Dedup happens *after* the walk (signature hash → coalesce), so the
   dominant cost — the AX walk — is paid unconditionally ~43,200×/day.
2. **No idle signal.** A locked/abandoned machine with a static window keeps a session
   "alive" (coalesced heartbeats advance the session clock). Time analysis built on
   sessions will overcount.
3. **No measured durations.** `observations` records *that* content appeared at ts, and
   Layer-4 sessions segment on app-change / 5-min gap — but nothing records "app X was
   frontmost from t1 to t2" as ground truth. Day-memory infers time from observation
   timestamps, which coalescing makes lossy.
4. **Degraded-coverage apps are blind spots**: Teams, Zoom, VS Code, terminals
   (blocklisted by design), Electron apps with empty AX trees. No OCR fallback shipped.
5. **Context to the model is two-tier only**: deterministic day-memory facts, then raw
   4 KB chunks via RAG. No intermediate "what happened in this 20-minute block"
   narrative layer, so synthesis queries over a full day lean on retrieval luck.

## 3. Proposal

### Phase A — Event-driven capture core (efficiency: the big win)

Replace "spawn helper every 2 s" with a **persistent helper process** in supervised
long-run mode (same signed binary — TCC grants are path-bound, so keeping one binary
matters), speaking newline-delimited JSON (NDJSON) events over the existing
private pipe. (Decided during Phase A plan review: the pipeline is already
NDJSON end-to-end and JSON string escaping guarantees framing; a typed line
protocol with `type`-less lines as capture frames keeps old binaries
compatible. Length-prefixing was considered and rejected — it would break
back-compat and buy nothing.)

Trigger table (adapted from screenpipe's spec + macOS API report):

| Trigger | Source | Debounce |
|---|---|---|
| App activated | `NSWorkspace.didActivateApplicationNotification` | 300 ms settle |
| Window/tab focus or title change | `AXObserver`: `kAXFocusedWindowChanged`, `kAXTitleChanged`, `kAXFocusedUIElementChanged` | 300 ms |
| Typing pause | input-activity heartbeat (see below) | 500 ms after last input |
| Idle fallback | timer | every 5–10 s while active |
| Hard floor / ceiling | — | ≥1 s between captures; force capture at ≤60 s gap |

Rules learned from the research (treat as requirements, not suggestions):

- **AX notifications are hints, not ground truth** — many apps (Chromium, Electron)
  under-emit. Keep the slow poll as backstop (the idle-fallback tick *is* the poll).
- **Force-capture ceiling is mandatory**: screenpipe's change-detector once got "stuck
  at 31 frames forever"; any diff/skip logic needs a max-gap safety net.
- **Idle detection**: `CGEventSourceSecondsSinceLastEventType(kCGEventSourceStateHIDSystemState, kCGAnyInputEventType)`
  — permission-free, no event tap, no new TCC. Poll it on the idle tick; emit
  `afk`/`not-afk` transitions (threshold ~2–3 min, instant return on input).
  Do NOT use a CGEventTap for this (needs Input Monitoring TCC; silent-disable race;
  screenpipe's tap caused system-wide mouse lag, issue #3489).
- **AX walk budget**: keep our existing 5 K-node bound but add a per-element messaging
  timeout (`AXUIElementSetMessagingTimeout`, ~0.25–1 s) and batch attribute reads
  (`AXUIElementCopyMultipleAttributeValues`). Accept partial text on timeout.
- Helper lifecycle: supervised by the Python daemon exactly like today (circuit
  breaker, fail-closed on TCC loss); one-shot mode stays for `openbird doctor` and
  as a degraded fallback.
- Wrap active work in `ProcessInfo.beginActivity(.background)`; run walks off-main at
  `.utility` QoS; use timer tolerance so wakeups coalesce.

Expected effect: AX walks drop from ~43 K/day to roughly the number of *real context
changes* (a few thousand), CPU floor drops to near-zero when the screen is static,
and capture *latency* on an app switch improves (300 ms settle vs. up to 2 s poll
delay) — more efficient AND more detailed at the boundaries, which is exactly where
time analysis needs precision.

### Phase B — Activity span layer (detail: measured time ground truth)

New lightweight signal, separate from text observations — the ActivityWatch insight.
Every trigger event (plus AFK transitions) updates a heartbeat-merged **activity span**.
Spans are **two-tier**. The tier is decided by a **text-independent policy
classification** — a new `classify_policy()` that evaluates ONLY the structural
states (paused / self-capture / allowlist / blocklist / dangerous / private-incognito)
and returns (tier, reason). It must NOT reuse `redact.decide()` as-is: `decide()`
rejects `no_text` *before* the structural checks (redact.py:347), so an allowlisted
app with an empty AX tree would wrongly classify as coarse, and a private-window
empty frame would lose its correct reason. Span classification and the
should-store-an-observation decision are separate questions evaluated from the same
structural inputs:

```sql
activity_spans(
  span_id TEXT PK,
  start_ts REAL, end_ts REAL,          -- merged extent (wall clock, storage only)
  bundle_id TEXT, app TEXT,            -- coarse identity, both tiers
  detail_tier INTEGER,                 -- 0 = coarse, 1 = full
  window TEXT,                         -- tier 1 ONLY; scrubbed via scrub_metadata()
  url_host TEXT,                       -- tier 1 ONLY; host only; URL capture stays opt-in
  afk INTEGER,                         -- 0/1
  meeting INTEGER DEFAULT 0,           -- Phase C
  reason TEXT                          -- tier 0 ONLY: why coarse; closed enum:
                                       --   not_allowlisted | blocklisted |
                                       --   dangerous | private | paused |
                                       --   self_capture. NULL for tier 1.
)
```

**Canonical identity contract** (one representation, no mixing): the reason enum is
the snake_case closed set above — prose names like "self-capture" always mean the
enum value `self_capture`. Field presence is tier-determined and NULL-sentineled:
tier 0 ⇒ `window = NULL`, `url_host = NULL`, `reason` set; tier 1 ⇒ `reason = NULL`,
`window`/`url_host` set (url_host NULL unless URL capture is enabled). The merge
identity tuple below compares these fields with NULL-equals-NULL semantics, so the
always-NULL fields of a tier can never split that tier's merges.

- **Tier 1 (full)** — only when `classify_policy()` returns full (allowlisted, not
  private/incognito, not paused, not dangerous): scrubbed window title, URL host, and
  per-app identity keys (file path / git repo / document, extracted by the existing
  adapter layer). Tier 1 with empty AX text is normal (span recorded, no observation).
- **Tier 0 (coarse)** — for everything else: time, bundle ID/app name, category, AFK,
  and a reason code. **No window title, no URL host, no AX-derived identity keys.**
  Enforced in the helper *before* any AX/AppleEvents call (the helper already emits
  `window: nil, url: nil, text: ""` for blocked apps — tier 0 reuses that exact path)
  and enforced again in Python at ingest. Window titles carry content (message
  subjects, document names, URLs); scrubbing is defense-in-depth, never the basis for
  collecting titles from apps the user blocked. Users can additionally exclude apps
  from spans entirely (then that time appears as "untracked").

This closes today's biggest analysis blind spot — blocklisted terminals/editors
currently vanish from the day entirely — while collecting strictly less than the
title-bearing design a naive spans table would imply: "45 min in Ghostty
(blocklisted)" is one coarse row.

**Merge rule** (ActivityWatch's algorithm, hardened for real timekeeping):

- Identity tuple = (bundle_id, detail_tier, **reason**, window, url_host, afk) —
  reason is part of identity so a pause/blocklist/private-state change for the same
  app can never be flattened into the prior coarse row with a stale reason. If a
  heartbeat's tuple equals the open span's AND it arrives within `pulsetime` on the
  **monotonic clock** (poll/idle-tick interval + jitter margin), extend `end_ts`; else
  close and open a new span. A 2-hour unchanged window = **one row** with an exact
  duration.
- **Monotonic time governs merge deadlines; wall clock is stored but never compared** —
  a wall-clock jump (NTP, timezone) can't create a multi-hour false span or a negative
  gap.
- **Sleep, screen lock, AFK-threshold transitions, capture pause, and policy/settings
  changes (allowlist/blocklist/URL-capture edits) force-close the open span**
  (`NSWorkspace.willSleepNotification`, `screensDidLockNotification` via the
  distributed-notification equivalent, the idle detector, and the existing pause-file
  / settings watchers). Wake/unlock opens a fresh span; no span may extend across a
  sleep boundary or a policy-state boundary.
- Span extension is additionally capped by the force-capture ceiling: no heartbeat for
  > ceiling ⇒ span closes at last-heartbeat time, not at "now".
- **Restart boundary**: monotonic deadlines are valid only within one process
  lifetime. Helper or daemon start/restart first force-closes any span left open —
  at its last-heartbeat wall time, never at "now" — before any merging resumes
  (equivalently: each process run has a restart epoch, and heartbeats never merge
  across epochs). A supervisor restart can therefore never mis-extend a stale span.
  (Amended during Phase B plan review: the boundary applies to every process whose
  MONOTONIC CLOCK OR SCHEDULER STATE feeds span merging. In stream mode that is
  both the daemon and the persistent helper — each `run_persistent` cycle is a new
  epoch. In the poll/one-shot fallback the helper is a stateless sampler carrying
  no clock or cadence state between 2-second spawns; the daemon's continuous
  monotonic clock is the only merge clock, so the DAEMON PROCESS is the epoch unit
  there — per-spawn epochs would forbid all merging and make the fallback useless.)
- Span boundaries debounce **independently** of content capture: a 300 ms app-switch
  settle delays the *text* capture, but the span boundary is recorded at the switch
  event itself, so fast A→B→A switching inside the debounce window still yields
  correct (small) spans.
- Semantics: spans measure **frontmost-app time on the active display** — not
  multi-display visibility, not audio/background activity. Document this in the UI.

- **`app_changed` boundary markers** (added in Phase B plan review): the helper
  emits `{type: "app_changed", ts, app}` the instant the frontmost app changes,
  BEFORE the capture debounce settles — span boundaries stay exact (the span
  start is backdated to the marker) even when a fast A→B→A switch coalesces B's
  capture frame away entirely (the middle app gets a real small span). Bundle id
  only — no AX call, no titles; old daemons parse the unknown type as a capture
  frame and reject it as `no_text` (harmless).
- **Prune rule** (tightened in Phase B plan review): retention (`before_ts`)
  deletes any span that BEGAN before the cutoff ENTIRELY — never truncated,
  never retained — so no span data from the pruned window survives and the
  span-delete invalidation trigger always fires; purge-of-recent (`since_ts`)
  deletes any span touching the purged window.

**Join model**: spans are first-class. Observations gain a nullable `span_id` column,
and assignment is **event-scoped, not lookup-scoped**: the daemon resolves the span ID
for the trigger event *first* (create-or-extend), then passes that immutable `span_id`
into `add_observation()`, which inserts it with the observation row inside the same
write transaction. The observation must never query "current open span" at insert
time — embedding happens outside the write transaction (store.py:300), so the open
span could drift between capture and insert. The existing Layer-4 `session_id` is NOT
the join key — sessions advance only on policy-accepted content frames, so blocked-app
spans would either dangle or mis-attribute to adjacent accepted content. Day-memory
time facts read from spans; sessions remain a content-grouping concept.

### Phase C — Detail enrichers (each independently shippable)

1. **Meeting detection**: CoreAudio `kAudioDevicePropertyDeviceIsRunningSomewhere`
   property listener (mic hot somewhere) + frontmost-app ∈ {Zoom, Teams, Meet tab,
   FaceTime, Webex, Discord} ⇒ `meeting=1` on the span. No TCC needed for the
   run-state signal. This both labels time ("3 meetings, 2.1 h") and lets any future
   heavy local inference defer while a call is live (screenpipe's Whisper-vs-Zoom GPU
   contention lesson).
2. **OCR fallback for AX-empty apps** (flag-gated, off by default): on capture where
   AX text is empty AND app is allowlisted, one `SCScreenshotManager` window-scoped
   still → Vision `RecognizeDocumentsRequest` (macOS 26; returns structured
   paragraphs/tables/entities — no manual line grouping) → discard pixels immediately,
   ingest text through the *existing* scrub/normalize/dedup pipeline. Costs the
   Screen-Recording TCC prompt + monthly re-auth nag + permanent orange menu-bar
   indicator, so it must stay opt-in per-app ("enable deep capture for Teams?").
   Terminals stay blocklisted regardless (screenpipe added an `app_prefers_ocr` list
   for terminals; our stance — don't capture them — is simpler and safer).
3. **Browser URL robustness**: keep AppleEvents for Chromium (existing); document that
   a browser extension (ActivityWatch model: `tabs.onActivated`, incognito flag native)
   is the correct long-term path for Safari/Firefox rather than AX-address-bar hacks.
   Never set `AXEnhancedUserInterface` on Chrome — it breaks the user's window
   management (long-standing Chromium side effect); `AXManualAccessibility` is
   Electron-only and version-fragile (fixed only after electron#38102) — set, tolerate
   failure, fall back.

#### Phase C1 — decisions as implemented (meeting flag; `MicMonitor.swift`, `spans.py`)

- **Signal plumbing**: the helper emits `{type:"system", kind:"mic_started"|"mic_stopped"}`
  edge events (the `system` vocabulary gained exactly these two — closed set; an OLD
  daemon coerces unknown kinds to `None` and `on_system(None, ts)` is a no-op, so
  back-compat is free). Edges, not heartbeat booleans: heartbeats quantize to ~5s and a
  mismatched-identity heartbeat closes without reopening; an edge splits at the exact
  instant, mirroring `afk_transition`. Named `mic_*`, not `meeting_*`: the helper reports
  the raw hardware signal; the meeting JUDGMENT is daemon-side policy.
- **CoreAudio mechanics**: `MicMonitor` (CaptureHelperCore) is a pure state machine over
  an injectable `MicHAL` protocol so the stateful listener lifecycle is unit-tested;
  `CoreAudioMicHAL` (executable target) is the mechanism-only adapter, listener blocks on
  the MAIN dispatch queue. Run state is read on the **input-scope**
  `DeviceIsRunningSomewhere` address (global scope goes true for output playback on
  duplex devices — AirPods music must not read as mic-hot), with a per-device
  global-scope fallback and a one-time `mic_scope_fallback` diag when the input-scope
  query errors. All input-capable devices are watched (Zoom can use a non-default mic);
  per-device listeners are removed BEFORE every re-enumeration (leak guard); emission is
  aggregate-flip-only plus an initial `mic_started` when already hot at startup. No TCC:
  run-state is metadata. Stream mode only (one-shot has no process to host listeners,
  same as AFK).
- **Meeting definition** (conjunction, computed in `SpanTracker`):
  `meeting = mic_hot AND (bundle_id in MEETING_BUNDLES OR url_host in MEETING_HOSTS)`.
  Mic-alone flags dictation/Siri; app-alone flags idling in Discord. The sets live in
  `spans.py` as frozensets storing CASEFOLDED constants; comparisons casefold and storage
  stays raw. Python-only — no Swift copy, no parity test (unlike `dangerous_apps.json`,
  which is a privacy boundary; a stale meeting entry mislabels time, never leaks
  content). Tier-0 spans match on bundle only (URL stripped by design): a coarse browser
  span never flags via a Meet tab (acceptable — needs the browser allowlisted +
  `capture_urls`), while a non-allowlisted Zoom still flags. User override deferred.
- **Span mechanics**: `meeting` joins `_SpanIdentity` — a mid-span mic edge SPLITS the
  span (truncate-close at the edge, reopen flipped), like `afk`; the AFK mirror copies
  `meeting` as-is (on a call you don't type — meeting spans routinely go AFK). Paused
  pseudo-spans never flag. Pending app-boundary markers record the mic state at marker
  time and backdate ONLY when the marker-time meeting bit equals the frame-time bit;
  otherwise the new span opens at the mic-transition ts and the pre-edge stretch is
  materialized with the marker bit (switch-to-Zoom at t=100 + mic at t=101 yields
  non-meeting [100,101) + meeting [101,...), never a meeting span backdated to 100).
  `begin_epoch` resets mic state — the fresh helper re-emits `mic_started` when already
  hot, so a live call is never lost, but a dead helper's stale hot bit cannot survive.
  `store.open_span` gained `meeting=` (the column existed since Phase B; NO migration).
- **Day-memory (v9 bump)**: `_span_metrics` adds `meeting_seconds` (AFK meeting spans
  COUNT — accumulated before the AFK early-continue) and `meeting_count` (intervals
  merged into runs across gaps <= 120s: split spans, brief mutes, and mid-call app
  switches stay one meeting). The bump to `day-memory-v9` is REQUIRED: the reuse gate
  compares `extractor_version` and span-extent hashes only, so a metrics-shape change
  alone would never invalidate cached days.
- **LLM deferral**: the liveness sidecar gains a `meeting` bit — the tracker latch
  `_mic_hot AND _meeting_seen` (set when a meeting span opens/extends; cleared on
  mic-off/epoch), so deferral holds while glancing at Notes mid-call and never fires for
  dictation. `should_run_background_llm` checks fresh+meeting FIRST and defers with
  `meeting_live` even on AC power (Zoom GPU/CPU contention exists on AC too — the
  screenpipe lesson). A STALE sidecar never defers on meeting alone (fail-open on
  meeting ONLY — a dead daemon must not block summaries forever; the 30s freshness bound
  self-heals a lost `mic_stopped`); the battery path stays fail-closed unchanged.
- **Block-summary interaction (documented tradeoff)**: a meeting split changes block
  span membership -> a NEW block key. In routine mode the 15-min settle rule means a
  live span being split can never belong to an already-summarized settled block, so no
  stale overlap arises. Only `--force` on-demand builds can summarize an unsettled block
  that a subsequent split re-keys, leaving an overlapping stale row until its sources
  change — documented in the CLI `--force` help text, not silently ignored.
- **Privacy**: no new content-bearing fields anywhere — the mic signal, the span bit,
  and the sidecar field are metadata bits; all logging is reason codes.

#### Phase C2 — decisions as implemented (OCR fallback; `OcrGate.swift`, `OCRCapture.swift`, `OcrBridge.swift`)

- **Stream-mode only** (decisive, not stylistic): the per-app OCR throttle needs
  process-lifetime "last attempt at" state, and a one-shot helper is a fresh process
  every ~2s — helper-local throttle state cannot exist there. Same precedent as
  MicMonitor/AFK. The daemon passes `--ocr-apps a,b --ocr-min-interval <s>` only when
  `capture_ocr_apps` is non-empty AND the spawn is `--stream`; one-shot spawns never
  see the flags, keeping `openbird doctor` / `--once` byte-identical.
- **Gate order** (`OcrGate`, pure state machine in CaptureHelperCore, injected
  monotonic clock, closed reason codes): axTextEmpty → optedIn → tccGranted →
  !micHot → per-app min-interval. Mic-hot suppression is the screenpipe
  Vision-vs-Zoom GPU-contention lesson; the throttle slot is consumed at DECISION
  time so a timed-out attempt cannot retry-storm. `Scheduler` is untouched — OCR is
  a per-capture fallback, not a cadence concern (AFK suppression already happens
  upstream in `Scheduler.tick`).
- **Self-capture gate grew a Swift copy**: `isSelfCapture` (exact
  `ai.openbird.openbird` root or dotted-child, NEVER substring) now runs in
  `captureFrontmost` BEFORE the allowlist gate / AX element creation / SCK — with
  screenshots in play, the Python-side backstop alone would have permitted a still of
  OpenBird's own UI. A two-copy parity test pins the Swift literal to
  `redact._SELF_BUNDLE_ROOT`.
- **macOS-14 floor, `.v13` package**: `SCScreenshotManager` requires macOS 14, so the
  real `ScreenCaptureKitOcrHAL` sits behind `#available(macOS 14.0, *)` and a
  macOS 13 stub reports `ocr_unavailable`. Window resolution is
  `SCShareableContent.excludingDesktopWindows` filtered by pid + layer 0 (exact
  AX-title match preferred, else largest area) — no private `_AXUIElementGetWindow`.
  Vision ships on `VNRecognizeTextRequest` (`.accurate`, auto language) ONLY; the
  macOS-26 `RecognizeDocumentsRequest` branch is deferred until the toolchain gate
  has CI coverage.
- **Two-phase semaphore bridge with late-completion ownership** (`OcrBridge`,
  unit-tested with fake HALs): the HAL is split at the PIXEL boundary. The async
  ACQUIRE phase (SCK window lookup + still) runs in a retained cancellable `Task`
  raced against the OCR share of a COMBINED AX+OCR ≤ 4s per-capture budget (OCR gets
  `min(2.5s, remaining)`); on timeout the task is cancelled and a per-call generation
  box is closed, so a late-acquired image is dropped UNREAD — never recognized,
  never emitted, never bleeding into a later capture. The synchronous RECOGNIZE
  phase (Vision) is NEVER abandoned: cancellation cannot interrupt
  `VNImageRequestHandler.perform`, so the bridge waits for the pass to finish and
  the image scope to close before returning (scope-bound pixel invariant; worst-case
  walk-queue occupancy is `deadline + one sub-second-typical .accurate pass`,
  documented in OcrBridge.swift). `CGPreflightScreenCaptureAccess` only — the helper
  never prompts; the grant flow lives in the mac-app (deep-capture section, stating
  the three costs: Screen Recording permission, monthly re-auth nag, permanent
  orange indicator).
- **Wire protocol**: capture frames gain `ocr: true` (only on fallback frames) via an
  explicit `encode(to:)` with `encodeIfPresent` for `type`/`trigger`/`ocr` — the
  omitted-when-nil contract is now stated rather than synthesized, pinned by
  raw-JSON shape tests. The `system` vocabulary gained exactly
  `ocr_available` / `ocr_unavailable` (startup preflight reading + observed flips);
  the daemon mirrors them into the liveness sidecar `ocr_state`, capture-health
  renders opted-in rows as available/unavailable/unknown, and stats grew an
  `ocr_captures` counter. Old daemons ignore the unknown key/kinds — back-compat is
  free, same argument as `mic_*`.
- **Pixels + text**: the `CGImage` never leaves the HAL function's scope (file-header
  invariant — never disk, never stderr/logs); recognized text flows over the same
  private stdout pipe and through the EXACT existing ingest path
  (`redact.apply → normalize → volatility → scrub → truncate → scrub_metadata`) —
  the no-bypass guarantee is asserted by test, not by new code. `preflight.py` now
  routes `screen_recording` to the CAPTURE helper (it owns SCK); mic/system-audio
  stay on the audio helper.

### Phase D — Hierarchical context for the AI (what the model should see)

The research consensus: raw capture dumps don't answer "how did I spend my day" —
progressive compaction does. Three layers, each grounded in the one below:

1. **Layer 0 — deterministic facts (exists: `day_memory.py`, extend with spans).**
   Exact durations by app/category/hour, focus blocks (long spans, low switch rate),
   context-switch counts (Rize's fragmentation signal), meeting time, AFK time, top
   repos/domains/documents. Computed from `activity_spans` — no LLM, no hallucination,
   citations to span IDs.
2. **Layer 1 — block summaries (new).** Lazily (idle-time / on-demand, never in the
   capture path), summarize each session/focus-block into 2–4 sentences: inputs =
   that block's spans + observation chunks; output stored like day_memories with
   citations. Local model first (existing Ollama qwen3-tier; Apple Foundation Models
   is a candidate on macOS 26 — no iOS-style rate limits, but its 4096-token context
   forces exactly this map-reduce shape anyway). Battery guard: only on AC power or
   when idle, batch-deferred (screenpipe's BATCH_TRANSCRIPTION lesson).
3. **Layer 2 — day narrative (exists as synthesis route, re-point it).** "Summarize my
   day" composes Layer 0 facts + Layer 1 block summaries instead of raw retrieved
   chunks; raw chunks (existing RAG) remain the detail/recall path for specific
   questions ("what was that error message?").

Prompt-context format stays the fenced untrusted-context pattern; each block summary
carries `[source_id]`s that resolve through existing citation validation.

**Categorization taxonomy** (for "productive vs not"): ship a small default rule set
over bundle-ID/URL-host (the RescueTime lesson: defaults ARE the product), five levels
(Focus Work / Other Work / Neutral / Personal / Distracting), user-overridable in
config; unmatched activity ≥2 min falls back to LLM classification with the block
summary as input (the Rize pattern), cached per identity key so each app/site is
classified once. Frame outputs like littlebird (narrative + briefings), not like a
surveillance scorecard.

#### Phase D — decisions as implemented (v5 schema, `openbird/summaries.py`, `openbird/taxonomy.py`)

Recording the adversarially-reviewed decisions so future phases build on what
actually shipped, not the sketch above:

- **Where summarization runs**: the ROUTINES daemon (`block-summaries` builtin,
  hourly) plus the on-demand `openbird summaries build [--date] [--force]`. NOT
  the capture daemon (a multi-second Ollama call would stall the single
  event-processing thread feeding SpanTracker) and NOT `ensure_day_memory`
  (which runs inside a `BEGIN IMMEDIATE` retry loop on the chat path — a model
  call there would hold the writer lock). Chat NEVER generates summaries
  synchronously; a day question composes whatever exists and degrades to the
  deterministic answer when none do.
- **Single block-boundary definition**: `summaries.compute_span_blocks` is the
  focus-block run-builder lifted out of `day_memory._span_metrics`, which now
  calls it — the summarizer and day-memory metrics can never disagree about
  what a block is. Blocks are keyed by `sha256(sorted span_ids)` and
  fingerprinted over `(span_id, real end_ts)` pairs: regeneration happens only
  on fingerprint drift (a member span extended), never on a version bump alone
  (battery). Only blocks settled ≥ 15 min (`block_summaries_settle_seconds`)
  are summarizable; each runner pass is batch-capped.
- **Battery/idle gate** (`should_run_background_llm`): AC power (pmset;
  subprocess failure treated as BATTERY, fail-closed) always allows. On battery
  a run requires the capture-liveness sidecar to be FRESH (finite
  `0 <= now - updated_at <=` the shared 30 s staleness bound,
  `capture.health.DAEMON_STALE_AFTER_SECONDS` — one definition, imported, never
  duplicated) AND `afk == true`. A stale/absent/malformed/future sidecar
  DEFERS: a stopped or wedged capture daemon must never make an ACTIVE user
  look idle (the earlier `last_active_span_end` proxy had exactly that failure
  mode and was dropped).
- **Catch-up coalescing**: the `block-summaries` routine sets
  `coalesce_catchup`; N missed hours produce exactly ONE bounded run. Ordering
  is retry-safe: the representative (newest missed) occurrence runs through the
  normal claim lifecycle FIRST; only after it succeeds are the remaining missed
  occurrences marked coalesced/done (`RoutineStore.mark_coalesced`,
  metadata-only). Failure leaves them unclaimed — never stranded `running`
  rows, never occurrences marked terminal before the work happened.
- **Grounding discipline**: prompt context items are S-labeled spans (metadata
  lines) + policy-accepted observation excerpts, fenced as untrusted; the model
  must cite. Hallucinated ids drop; ZERO valid citations ⇒ nothing is stored
  (reason code `block_summary_ungrounded`). DB `BEFORE INSERT` triggers
  back-stop ref integrity.
- **Deletion cascade** (v5): `block_summary_source_refs` mirrors
  `day_memory_source_refs`; deleting a cited observation/span trigger-deletes
  the summary, and — with `PRAGMA recursive_triggers = ON`, set per connection,
  READ BACK, and refused on a silent no-op — deleting a summary invalidates any
  day memory citing it (`source_kind='summary'`, dormant until E1). Full purge
  also wipes `category_assignments` (LLM-derived from captured content).
- **Route truthfulness (Layer 2 re-point)**: the deterministic day answer and
  the default briefing compose stored block summaries at ANSWER time (shared
  `compose_day_narrative`; no provider call, no payload embedding). When (and
  only when) summaries are composed, `reasoning_route` flips to
  `local_cached_model_summary` — the facts stay deterministic but the narrative
  sentences are precomputed local-model prose, and the label must say so. No
  summaries ⇒ byte-identical `local_deterministic`. See
  `docs/privacy-routes.yaml` (`chat.day_memory_cached_summary`,
  `summaries.block`).
- **Privacy**: `summary_text` is derived-sensitive — encrypted DB only, never
  logged, never in routine output (the routine line and the runner result are
  counts + reason codes). The taxonomy cache stores identity key + level only.
- **Cloud gating**: the shared `LLMProvider`'s `CloudOptInRequired` enforcement
  is the gate — no new consent flag; a cloud-opted user has opted this path in.
  No `reasoning_send_ledger` rows (that ledger is deep-brain-packet-scoped).
- **Taxonomy axes stay separate**: `taxonomy.py` owns the five-level JUDGMENT
  axis; `day_memory.classify_observation` keeps its DESCRIPTIVE categories.
  Nothing maps one onto the other (mixing description with judgment is the
  surveillance-scorecard failure mode). Day memories (v8) carry measured
  `span_time_by_level`, the `uncategorized_identity_seconds` fallback queue,
  and a `taxonomy_fingerprint` so an edited `taxonomy.json` or a newly cached
  LLM level rebuilds the cached day.
- **Explicitly deferred to E1**: block summaries in MODEL context for
  multi-day/`_answer_temporal` synthesis, and summary embedding/retrieval
  (citing summaries through `_validate_citations` — E1's "index the
  summaries"). Raw chunks remain the detail path untouched. Week rollups,
  entity ledger (E2), OCR (C2), and any productivity-score surface stay
  non-goals here.

### Data lifecycle — migrations, citations, prune/purge (applies to B, D, E)

The current durable-invalidation model is observation-only (`day_memory_sources`
references observations; purge deletes observations + day memories + ledger). Each
phase that adds derived data must extend that model, not bypass it:

- **Migration ladder**: every new table lands as a forward migration with a
  `SCHEMA_VERSION` bump in `openbird/memory/migrations.py` (one bump per phase:
  spans; block/week summaries; entities+evidence), with FKs and indexes defined in the
  migration, matching the existing v2/v3 pattern.
- **Typed citation sources**: day facts, block/week summaries, and entity evidence
  cite `(source_kind, source_id)` where source_kind ∈ {observation, span, summary}.
  Layer-0 facts computed from spans cite span IDs; the citation-validation gate
  resolves each kind through its own table.
- **`day_memory_sources` migration lands IN the spans phase (Phase B), not later**:
  the existing table is observation-only (schema.sql:89) with an observation-delete
  trigger only, so day-memory facts cannot cite spans until it's replaced. The Phase-B
  migration adds typed `day_memory_source_refs(day_memory_id, source_kind, source_id)`,
  migrates existing observation refs into it, updates
  `save_day_memory`/`ensure_day_memory`, adds a span-delete invalidation trigger (or
  prune-path invalidation), and ships span-delete invalidation tests.
- **Derived-artifact source tables**: block summaries and week memories get source
  tables mirroring `day_memory_sources` (summary_id → source_kind/source_id), so
  invalidation is queryable.
- **Prune/purge cascades**: `data prune --days N` and purge must delete spans in the
  window AND delete-or-mark-stale every derived artifact whose source set intersects
  the deleted rows (block summaries, day/week memories, entity evidence; entities
  whose evidence set becomes empty are marked dormant, not silently kept). A derived
  row must never outlive — or cite — a deleted source.

### Non-goals (explicit, from the failure catalog)

- No continuous screenshots, no video encoding, no audio-by-default (Rewind's death).
- No keystroke content capture; input activity is a *timestamp*, never text.
- No content-classifier-only privacy (Recall's failure) — structural exclusions stay
  the primary layer; scrubbing remains defense-in-depth.
- No cloud storage of captures (littlebird differentiation).

## 4. Budgets & health (adopt as acceptance criteria)

| Metric | Target (source: screenpipe v2 measured) |
|---|---|
| CPU idle (static screen) | <0.5% |
| CPU active | <5% |
| App-switch → observation in DB | <500 ms |
| AX walk wall-clock | ≤2 s total, ≤1 s per element messaging timeout |
| Storage | text-only; spans ≈ KB/day; unchanged blob/chunk dedup |
| First data after start | <5 s |
| Capture health | goes `stale` after N× expected event gap with no frame — never report `ok` off a stale timestamp (we already started this: effective-capture-health) |

Failure modes to design against on day one (screenpipe's scar tissue): silent write
drops (store-with-null-and-backfill, never skip), lying health endpoints, diff
thresholds without force-capture ceilings, supervisor restart storms (debounce), text
desynced from its trigger event (capture identity tuple + text atomically).

### Phase E — Second-brain queries (episodic + entity memory)

The product must answer two distinct query classes; they need different machinery:

**Episodic — "what was I doing last week?"** Served by Phases B+D with two additions:

1. **Week rollup**: compose day narratives into a week digest (same map-reduce shape,
   one more level). Store like day_memories (`week_memories` or a `scope` column).
2. **Index the summaries**: embed block/day summaries into the existing chunk index
   (tagged `source=summary`) so retrieval over long ranges hits compact grounded
   narratives instead of fishing through raw capture chunks. Temporal routing already
   parses "last week"; widen it to route long windows to summary-first retrieval.

#### Phase E1 — decisions as implemented (v6 schema, week rollup + summary indexing)

Recording the adversarially-reviewed decisions so E2 builds on what actually
shipped, not the sketch above:

- **Week storage — NO `week_memories` table**: one row per ISO week in the
  existing `day_memories` table with `local_date` = the week's MONDAY and
  `source_scope='week'`. `UNIQUE(local_date, source_scope)` gives per-week
  uniqueness for free, and the dormant v5
  `trg_day_memory_source_summary_delete` trigger gives deletion-cascade
  invalidation (block summary regenerated/deleted → citing week row dies,
  recursively through the verified span→summary chain). Week refs are
  SUMMARY-KIND ONLY; `save_week_memory` refuses zero refs in Python (mirroring
  `save_block_summary`) and the v5 insert-integrity trigger backstops unknown
  ids. The shared day-memory reader now returns `summary_ids` + a full typed
  `source_refs` list (legacy `source_ids`/`span_ids` untouched).
- **Freshness — fingerprint the stable substrate**: `member_fingerprint =
  sha256(sorted (block_key, block_fingerprint) pairs)`, never day-memory ids
  (day rows churn). Removals/regenerations self-heal via the delete trigger;
  ADDITIONS are caught by fingerprint drift at the next routine pass. Past
  weeks regenerate on drift only; the CURRENT week additionally waits
  `week_rollup_min_interval_seconds` (default 21600) since the last generation
  — a live week must not burn a model call every hour. A version bump alone
  never regenerates (battery rule); an unknown/incompatible stored version
  lazily upgrades once. `WEEK_EXTRACTOR_VERSION = "week-memory-v1"`.
- **Generation — model reduce in the SAME routines pass**: `_summarize_week`
  runs inside `run_block_summaries` (same battery/idle/meeting gate, same
  catch-up coalescing, routine name unchanged). The prompt mirrors the block
  prompt exactly (same FenceSpec, S-labels, JSON schema; persona key
  `week_summary`); context = per-day METADATA header lines + one fenced line
  per member block summary (capped at 60, longest preferred). Labels map ONLY
  to block-summary ids; ZERO valid citations ⇒ nothing stored
  (`week_memory_ungrounded`). Routine output stays counts-only
  (`weeks= week_ungrounded= indexed=`).
- **Summary indexing — parallel index tables, NOT synthetic observations**
  (the riskiest decision, deliberate): `summary_index_entries` +
  `fts_summaries` + `vec_summaries` (v6, `SCHEMA_VERSION = 6`; the vec table
  is created in `_apply_schema` at `Settings.embed_dim`, like `vec_chunks`).
  Rejected: (a) synthetic observations corrupt the occurrence model everywhere
  `source` is unfiltered — summary prose would double-count as activity in
  every temporal answer; (c) a `source_kind` on chunks collides with
  content-addressed global dedup and yields uncitable hits. Day-facts prose is
  deliberately NOT indexed (regenerated deterministically, churns on every
  fingerprint drift); block summaries are the compact grounded narrative
  layer.
- **Deletion is an API-ENFORCED contract**: SQLite triggers clean only the
  plain entries table (virtual tables cannot be written from triggers), so
  every production deletion path — `delete()`/`prune()` (pre-selected doomed
  ids), `save_block_summary` regeneration, `save_week_memory` regeneration —
  sweeps the fts/vec rows for doomed `('block', id)` AND dependent
  `('week', week_row_id)` rows in the SAME transaction, BEFORE the source
  delete. Raw SQL deletes against `block_summaries`/week rows outside these
  APIs are FORBIDDEN (documented in the store module + schema); `openbird
  data integrity` carries an orphan-count probe as the safety net. The
  embedding cohort counts BOTH vec tables (adoption rebuilds both); `openbird
  reindex` re-embeds summary entries in the same transaction. `stats()` keeps
  `day_memories` as the DAY count (verification scripts parse it) and adds
  `week_memories` + `summary_index_entries`.
- **Retrieval — summary-first for multi-day questions**: `_ContextItem` is an
  explicit occurrence/summary UNION; `_validate_citations` returns
  `(citations, derived)` and derived-only grounding passes the gate. The
  terminal cached week answer (synthesis + multi-day window) composes stored
  digests + per-day narratives + deterministic totals with NO provider call
  (route `local_cached_model_summary`; shared `compose_week_answer`, also the
  `openbird briefing --week` renderer). Below it, multi-day
  `_answer_temporal`/`_answer_scoped_specific` build MODEL context from block
  summaries + week digests (raw-row fallback under 2 items; raw chunks stay
  the detail path); fresh completions keep their truthful `local_model`/
  `cloud_reasoning_active` route — what changes is what egresses
  (`chat.summary_grounded`). The semantic (no-window) path RRF-merges
  `search_summaries` hits capped at 2 of the context slots. No regex changes.
- **Privacy**: digest/summary text lives ONLY in the encrypted DB (derived
  sensitive), never in logs or routine output; `docs/privacy-routes.yaml`
  gains `summaries.week`, `chat.week_memory_cached_summary`, and
  `chat.summary_grounded`, and `summaries.block`'s deletion contract now
  includes the index rows. Cloud gating remains `CloudOptInRequired` at
  provider construction — no new consent flag. Surfacing: `openbird briefing
  --week`, `openbird summaries list --weeks` (digest bodies interactive-only),
  and `summaries build` covers weeks + indexing automatically via the shared
  runner.

**Entity/state — "have I completed project X?"** A state judgment, not a time query.
Requires an **entity ledger** built on Phase B's stable identity keys:

```sql
entities(
  id TEXT PK, kind TEXT,       -- repo | domain | document | topic
  name TEXT, aliases TEXT,     -- "openbird" = repo key + Slack channel + doc titles
  first_ts REAL, last_ts REAL, -- activity extent (maintained from spans)
  status TEXT                  -- active | dormant | user-marked-done
)
entity_evidence(entity_id, ts, kind, source_kind, source_id)
  -- kind: pr_merged | ticket_closed | shipped_language | open_loop | open_loop_resolved
  -- source_kind: observation | span | summary  (typed source, so evidence can cite
  --   a span or a block summary, not only observations)
```

- **Aggregation**: nightly (idle-time) pass links spans/observations to entities by
  identity key + alias match; day_memory's existing *open loops* get promoted here and
  checked against later days for resolution instead of expiring day-scoped.
- **Completion signals** mined from captured text we already have: merge/close language
  in GitHub/Linear/Jira windows, "shipped/released/done" in summaries — stored as
  *evidence rows with citations*, never as bare inference.
- **Grounding rule (hard)**: completion answers must cite evidence + last-activity
  recency, or explicitly answer "no completion signal observed; last activity was
  &lt;date&gt;: &lt;cited event&gt;". An LLM guessing "looks done" from vibes is the failure mode;
  the existing citation-validation gate applies to evidence IDs.
- Later (out of scope here): user affordance to mark an entity done, which also makes
  the ledger a lightweight review surface ("dormant projects with unresolved loops").

#### Phase E2 — decisions as implemented (v7 schema, entity ledger + completion evidence)

Recording the adversarially-reviewed decisions so later phases build on what
actually shipped, not the sketch above:

- **Schema v7, additive-only** (`SCHEMA_VERSION = 7`, no rebuilds): `entities`
  with a deterministic `sha256("kind:casefold(name)")` id (idempotent upserts;
  identity is EXACT casefolded name — no fuzzy merging), `UNIQUE(kind, name)`,
  a status CHECK including `user_marked_done` (schema-ready; NO mark-done UI in
  E2 — the aggregation and dormancy passes never touch that status), and a
  TYPED `last_seen_source_kind`/`last_seen_source_id` pair (span-derived domain
  entities have no observation to cite) validated by BEFORE INSERT/UPDATE
  triggers and NULLed by per-kind BEFORE DELETE triggers when the source dies.
  `entity_evidence` has its own id PK plus a `detail` column (`NOT NULL
  DEFAULT ''` so the UNIQUE five-tuple dedupes re-runs — `detail` is
  load-bearing: open-loop → resolution matching needs the normalized item key,
  and the honest answer needs the matched cue). Per-kind BEFORE INSERT
  existence triggers and BEFORE DELETE cascade triggers mirror the
  day-memory/block-summary pairs; the recursive span → block-summary →
  evidence chain rides the verified `recursive_triggers` pragma. **NO fts/vec
  indexing of entities or evidence** — the ledger is answered
  deterministically by exact casefolded match; indexing would extend the
  API-enforced sweep contract for zero recall benefit and risk the
  double-counting E1 rejected. Consequence: `delete()`/`prune()` need no new
  pre-select sweep — plain-table triggers fully cover evidence.
- **Aggregation is STRUCTURALLY LLM-free**: `run_entity_aggregation(store, *,
  now, settings)` takes no provider argument at all — every evidence row is a
  regex match over a stored, citable source (an LLM judge cannot be re-verified
  against the citation; E3+ territory if ever). Runs in the SAME gated
  routines pass, right after summary indexing. Kinds populated: `repo` and
  `domain` only (`document`/`topic` are the fuzzy-noise failure mode; the
  CHECK admits them, no code mints them). `identity_key` stays reserved —
  lighting it up at capture time would change span merge identity, and spans
  lack URL paths anyway.
- **Per-source-class cursors** in the `embedding_meta` KV
  (`entity_aggregation.*`; wiped on full purge): observations advance a
  composite `(ts, id)` activity cursor through the last actually-processed row
  via the row-capped `observations_text_page` (`ORDER BY ts, id LIMIT` — the
  REAL batching bound; `time_range_text`'s max_chars caps text per row, not
  row count). The 48h overlap is a SEPARATE, equally-capped idempotent re-scan
  that never moves the cursor (no livelock when the overlap window outgrows
  the cap). Spans advance their own independent `(start_ts, span_id)` cursor
  (`spans_page` + `entity_aggregation.span_ts`/`.span_id`) with the same
  forward/overlap semantics, so a dense span history pages forward run over
  run. Summaries are mined by a `(generated_at, id)` cursor because
  regeneration replaces a historical row in place with a fresh `generated_at`
  — a regenerated old block is always re-mined (regression-tested).
- **Mining is item-anchored, never blob-global**: URL-is-item allows the
  status word anywhere in window+text; otherwise the word must sit within
  ±120 chars of THAT `_GITHUB_ITEM_RE` match (mixed-list pages mint evidence
  only for the merged item). Linear/Jira is a host+key+word conjunction on a
  domain entity. `shipped_language` scans block-summary prose for an entity
  name/alias within ±160 chars, citing the summary id. The day-memory
  extractors are EXPORTED (`REPO_RE`, `GITHUB_ITEM_RE`, `domain_from_url`),
  never duplicated. Aliasing is conservative: the bare repo name only while
  globally unique, removed on collision.
- **Open loops are REHYDRATED, never payload-trusted**: promotion re-reads the
  loop's cited observations (payload `source_ids` are sorted/capped — never
  chronological), requires a single stable item id across the rows, and cites
  the true earliest row; generic `cue` loops stay day-scoped. Resolution is
  the exact rule: same entity + identical `detail` key + later ts, inserting
  `open_loop_resolved` citing the resolving source. Dormancy runs at
  aggregation time AND synchronously inside the `delete()`/`prune()`
  transaction (evidence-less actives flip dormant; `user_marked_done` immune
  on every path).
- **Answering (`chat.entity_ledger`)**: the completion intent is checked AFTER
  window resolution; windowed questions filter evidence + last-activity to the
  window with window-scoped honest wording and never cite out-of-window rows;
  zero entity matches always FALL THROUGH (the ledger never eats content
  queries); ambiguity returns a deterministic candidate list, never a guess.
  The terminal answer is provider-free (route `local_deterministic`, grounding
  `derived`): evidence lines newest-first with `[n]` markers
  (`DerivedCitation` type `entity_evidence`, construction-side validated),
  unresolved open loops, and a last-activity line citing the TYPED last-seen
  ref — degrading to "the underlying event was deleted" when the trigger
  NULLed the pair.
- **Privacy/surfacing**: names/aliases/details are DERIVED SENSITIVE
  (encrypted DB only; logs and routine output carry counts + reason codes —
  `format_counts_line` gains `entities= evidence= loops_promoted=
  loops_resolved=`). `docs/privacy-routes.yaml` gains `entities.aggregation`
  and `chat.entity_ledger` (both class local, egress none). `openbird
  entities list/show` prints derived-sensitive bodies on an interactive
  terminal only (the `summaries list` guard); `stats()` gains
  `entities`/`entity_evidence`; `openbird data integrity` gains the
  entity-evidence orphan probe (belt-and-braces — triggers should keep it at
  zero always).

## 5. Suggested sequencing

1. **A** (persistent event-driven helper + idle detection) — pure efficiency, no schema
   change, biggest CPU win, improves boundary precision.
2. **B** (activity spans + heartbeat merge + day-memory duration facts) — unlocks the
   actual product goal: exact "how I spent my time".
3. **D1/D0** (block summaries + taxonomy) — turns spans into narrative and analysis.
4. **C1** (meeting flag) — cheap, high analysis value.
5. **E1** (week rollup + summary indexing) — completes episodic "last week" queries.
6. **E2** (entity ledger + completion evidence) — unlocks project-state queries.
7. **C2** (OCR fallback) — last; permission-cost gated, per-app opt-in.

Each phase is independently valuable and reviewable (per-feature pipeline applies:
plan → Codex consensus → implement → Codex diff review → PR → CodeRabbit → merge).

## Sources

Full agent reports (screenpipe deep-dive incl. their internal specs; landscape survey
incl. Rewind teardown, ActivityWatch heartbeat algorithm, Recall filtering; macOS API
report incl. Sequoia/Tahoe permission changes) — key primary sources:

- screenpipe `EVENT_DRIVEN_CAPTURE_SPEC.md`, `VISION_PIPELINE_SPEC.md`,
  `BATCH_TRANSCRIPTION_SPEC.md` — github.com/screenpipe/screenpipe/tree/main/docs
- Kevin Chen, "Rewind.ai app teardown" — kevinchen.co/blog/rewind-ai-app-teardown/
- ActivityWatch data model & heartbeat merging — docs.activitywatch.net
- Dayflow — github.com/JerryZLiu/Dayflow
- littlebird.ai product/FAQ; TechCrunch 2026-03-23 funding piece
- Microsoft Recall sensitive-info filtering — learn.microsoft.com (+ BetaNews/Borncity
  filter-failure tests)
- Apple: SCScreenshotManager, RecognizeDocumentsRequest (WWDC25), TN3193 (Foundation
  Models context), Sequoia 15.1 screen-recording prompt policy
- Timing/Rize/RescueTime categorization docs
