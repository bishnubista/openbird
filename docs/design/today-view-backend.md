# Design: Today/day-view backend (timeline + briefing)

**Status:** REVISED v2 — addresses 3 Codex findings (shared day window for
timeline+briefing; NULL-session grouping; gap-capped active time). Pending re-review.
**Branch:** `feat/today-timeline-backend` (off origin/main).
**Goal:** Expose the data the Claude Design "Today/day view" (handoff §3) needs —
a day's session timeline + stat chips + a daily-briefing prose card — as cheap,
local, resource-light CLI JSON the Swift app (PR E) renders. **No new subsystem,
no daemon, no new storage tables.**

## Why this is cheap (resource-conscious, the explicit requirement)

- `observations` already has an **indexed `session_id`** (PR #43 episodic sessions)
  and an **indexed `ts`**. The timeline is therefore a single indexed `GROUP BY`
  — no aggregation engine, no background job, no extra storage.
- The briefing **reuses the existing `yesterday`/`daily-briefing` routine
  template** (a pure function over a time window that calls the local LLM). It is
  generated **on demand** (only when the app asks) — **one** LLM call — and the
  Swift app caches it per day (PR E), so opening Today repeatedly costs nothing.
- The timeline path uses `_store_maintenance()` (already in cli.py): no cloud gate,
  a non-embedding provider stub → **no LLM/network is constructed at all** for the
  pure-SQL read.

## Changes

### `openbird/memory/store.py`
- `SessionSummary` dataclass: `session_id: str | None` (real id; NULL for legacy
  rows), `app: str | None`, `start_ts: float`, `end_ts: float`, `count: int`.
- `MemoryStore.day_sessions(start_ts, end_ts) -> list[SessionSummary]` (Codex #2 —
  NULL session_ids must NOT collapse into one false session; SQLite groups NULLs
  together). Group by a coalesced key so each legacy (NULL-session) row buckets on
  its own row id:
  ```sql
  SELECT session_id, app,
         MIN(ts) AS start_ts, MAX(ts) AS end_ts, COUNT(*) AS cnt
  FROM observations WHERE ts >= ? AND ts <= ?
  GROUP BY (CASE WHEN session_id IS NULL THEN id ELSE session_id END), app
  ORDER BY start_ts ASC
  ```
  `session_id` in the result is the real (nullable) value. Pure read; no embedding.
- `MemoryStore.active_seconds(start_ts, end_ts, gap_seconds) -> float` (Codex #3 —
  `sum(end-start)` over groups undercounts singletons and can overlap). Compute a
  gap-capped active time from consecutive observation deltas:
  ```sql
  WITH ordered AS (
    SELECT ts, LAG(ts) OVER (ORDER BY ts) AS prev
    FROM observations WHERE ts >= ? AND ts <= ?
  )
  SELECT COALESCE(SUM(MIN(ts - prev, ?)), 0) FROM ordered WHERE prev IS NOT NULL
  ```
  (SQLite scalar `MIN(a,b)`; window functions need SQLite ≥3.25 — macOS ships far
  newer.) Singleton/idle gaps contribute 0, which is the correct "engaged time".

### `openbird/cli.py`
- `_day_window(day_offset) -> (start, end)` (Codex #1) — the **single** source of a
  day's bounds, `_day_bounds(anchor, offset_days=-day_offset)` (local midnight, DST-
  safe). BOTH `timeline` and `briefing` consume these exact inclusive bounds, so the
  prose card and the session list always describe the same span.
- `openbird timeline [--day N] [--json]` (N: 0=today, 1=yesterday, …):
  - `(start, end) = _day_window(N)`; opens `_store_maintenance()` → **no
    provider/LLM, no cloud gate**.
  - emits `{day_offset, start, end, total_observations, distinct_apps,
    active_seconds, sessions: [{session_id, app, start, end, count}]}`.
    `total_observations`/`distinct_apps` derived from the sessions in Python;
    `active_seconds` from `store.active_seconds(start, end, gap)`.
- `openbird briefing [--day N] [--json]`:
  - `(start, end) = _day_window(N)`; opens `_store(provider=_provider())` and calls
    a NEW `RoutineTemplate.run_window(store, provider, start, end)` on the
    `yesterday` template — reusing its prompt/rendering but with EXPLICIT bounds
    (NOT its own `window(now)`), so it matches `timeline --day N`. Emits
    `{day_offset, start, end, text}`.
  - Empty window → the template's deterministic "no activity" line (no LLM call),
    so an empty day is free.
  - On-demand only; **the app caches per day** (PR E). The cron scheduler path is
    unchanged (still `RoutineTemplate.run` + RoutineStore idempotency).

### `openbird/routines/templates.py`
- Extract `RoutineTemplate.run_window(store, provider, start, end) -> str` holding
  the current body of `run` (get_text/render/empty-check/LLM). `run` becomes
  `run_window(store, provider, *self.window(now))` — behavior-preserving for the
  scheduler, and reusable by `briefing` with explicit bounds.

## Privacy / observability
- `timeline` returns only metadata (app names, timestamps, counts) — never blob
  text, window titles, or URLs. (Consistent with the content-free logging rule.)
- `briefing` returns LLM-summarized prose grounded in captured text; it is shown
  to the user in-app, never logged. Captured text is fenced as data by the
  existing template (`render_context_text`).

## Tests (`tests/unit/`)
- `day_sessions`: insert observations across two real sessions/apps in a window
  (and one outside it); assert grouping, ordering, per-session count, start/end,
  out-of-window exclusion, AND that two legacy NULL-session rows for the same app
  do **not** collapse into one session (Codex #2).
- `active_seconds`: deltas under the gap cap sum; a delta over the cap is clipped;
  a single observation → 0 (Codex #3).
- `run_window` (templates): same prompt/rendering as `run` but over explicit
  bounds; empty window → deterministic no-activity line, provider NOT called.
- `timeline` CLI (`--json`): shape + `distinct_apps` + `active_seconds` + empty
  day → zeros.
- `briefing` CLI (`--json`) with a stub provider: non-empty window → provider
  text over the SAME bounds as `timeline --day N`; empty window → no-activity line,
  provider NOT called.

## Definition of done
Codex consensus → implement → `uv run --extra dev python -m pytest -q` green →
Codex diff review → PR → CodeRabbit → merge. Then PR E renders this in the
Today/day-view UI.
