"""Unit tests for RAG synthesis-intent routing + signal-based selection.

Covers the `feat/second-brain-ask-fix` diff in ``openbird/chat/rag.py``:

  * ``RAG._intent_window`` — routes synthesis/meta queries ("Summarize my day",
    "what did I work on", "what should I follow up on?") to a time window;
    explicit temporal words ("yesterday", "this week") keep precedence; genuine
    content lookups stay semantic (return ``None``).
  * ``RAG._sample_even`` — even spread vs. earliest-N.
  * ``RAG._signal_score`` — informative window/body outranks generic + tracking.
  * ``RAG._answer_temporal`` — selection grounds on the rich LATE row, not just
    the earliest occurrences; grounding gate intact; synthesis routing never
    touches ``store.search``.

All deterministic: a fake store + a completer that cites whatever source ids it
is handed (so we assert SELECTION, not model behavior), and an injected
``RAG._now`` clock. No live LLM, no Ollama, no network.
"""

from __future__ import annotations

import datetime as _dt

from openbird.chat.rag import _DAY, _MULTIDAY_WINDOW_DAYS, RAG
from openbird.types import Observation

# A fixed, deterministic "now" used everywhere a clock matters.
NOW = 1_700_000_000.0


# -- fakes ---------------------------------------------------------------------


class CiteAllCompleter:
    """Records the messages it received and cites every source_id it can see.

    Mirrors the production prompt's ``[source_id: S1]`` header so a citation maps
    1:1 to a context item — letting tests assert *which rows were selected* rather
    than any model judgement.
    """

    def __init__(self):
        self.last_messages: list[dict] | None = None
        self.calls = 0

    def complete(self, messages, *, json_schema=None):
        self.last_messages = messages
        self.calls += 1
        user = messages[-1]["content"]
        ids: list[str] = []
        marker = "[source_id: "
        for line in user.splitlines():
            if marker in line:
                rest = line.split(marker, 1)[1]
                ids.append(rest.split("]", 1)[0])
        return {"answer": "Here is your day.", "citations": ids}


class FakeTemporalStore:
    """A store exposing BOTH ``time_range_text`` and ``search``.

    Having ``search`` present is deliberate: a synthesis query that routes through
    the time-range scan must NOT touch it. We record call counts to prove routing.
    """

    def __init__(self, rows):
        self._rows = rows
        self.search_calls = 0
        self.range_calls = 0
        self.last_window: tuple[float, float] | None = None

    def time_range_text(self, start, end, *, max_chars=2000):
        self.range_calls += 1
        self.last_window = (start, end)
        return list(self._rows)

    def search(self, query, k=10, *, semantic=True):
        self.search_calls += 1
        return []


def _obs(id_, *, ts, window=None, ch=None, app="App", session="s"):
    return Observation(
        id=id_,
        content_hash=ch or ("h_" + id_),
        ts=ts,
        app=app,
        window=window,
        url=None,
        session_id=session,
        source="capture",
    )


def _row(id_, *, ts, window=None, text="", ch=None):
    return (_obs(id_, ts=ts, window=window, ch=ch), text)


def _rag(rows=None):
    chatter = RAG(FakeTemporalStore(rows or []), CiteAllCompleter())
    chatter._now = lambda: NOW
    return chatter


def _today_start() -> float:
    return (
        _dt.datetime.fromtimestamp(NOW)
        .replace(hour=0, minute=0, second=0, microsecond=0)
        .timestamp()
    )


# -- _intent_window: synthesis intents resolve to a window --------------------


def test_summarize_my_day_routes_to_today_window():
    win = _rag()._intent_window("Summarize my day")
    assert win is not None
    assert win == (_today_start(), NOW)


def test_what_did_i_work_on_routes_to_today_window():
    win = _rag()._intent_window("What did I work on")
    assert win is not None
    assert win == (_today_start(), NOW)


def test_follow_up_routes_to_trailing_three_day_window():
    win = _rag()._intent_window("What should I follow up on?")
    assert win is not None
    start, end = win
    assert end == NOW
    # Multi-day "follow up" intent → trailing 3 days (NOT today, NOT a week).
    assert end - start == _MULTIDAY_WINDOW_DAYS * _DAY
    assert end - start == 3 * 86_400


def test_been_doing_routes_to_trailing_three_day_window():
    win = _rag()._intent_window("what have I been doing")
    assert win is not None
    start, end = win
    assert end - start == 3 * 86_400


# -- _intent_window: explicit temporal words keep precedence ------------------


def test_explicit_yesterday_keeps_precedence_over_synthesis_default():
    chatter = _rag()
    win = chatter._intent_window("what did I do yesterday")
    assert win is not None
    # Explicit temporal phrase wins: matches the dedicated temporal resolver, and
    # spans ~a single day (NOT today-start..now and NOT the 3-day default).
    assert win == chatter._temporal_window("what did I do yesterday")
    start, end = win
    assert abs((end - start) - _DAY) < 1.0  # ~one day (end is today-start - 1µs)
    assert start < _today_start()  # strictly before today => not the today default


def test_explicit_this_week_keeps_precedence_as_seven_days():
    win = _rag()._intent_window("what did I do this week")
    assert win is not None
    # Explicit "this week" => trailing 7 days, overriding any 3-day synthesis default.
    assert win == (NOW - 7 * _DAY, NOW)


# -- _intent_window: genuine content lookups stay semantic (None) -------------


def test_content_lookup_bugs_filed_recently_stays_semantic():
    # "recently" alone must NOT trigger synthesis routing.
    assert _rag()._intent_window("what login bugs were filed recently") is None


def test_content_lookup_decide_about_embeddings_stays_semantic():
    assert _rag()._intent_window("what did I decide about embeddings") is None


def test_content_lookup_summarize_named_doc_stays_semantic():
    assert _rag()._intent_window("summarize the auth design doc") is None


def test_tightened_triggers_do_not_capture_content_lookups():
    # "recap"/"follow up on"/"what's been happening" must require first-person /
    # avoid a topic object, so content lookups stay on semantic search.
    r = _rag()
    assert r._intent_window("recap the design doc") is None
    assert r._intent_window("follow up on the Stripe ticket") is None
    assert r._intent_window("what's been happening with the deploy") is None
    # The intended first-person forms still route.
    assert r._intent_window("recap my day") is not None
    assert r._intent_window("What should I follow up on?") is not None
    assert r._intent_window("what's been happening") is not None


def test_hyphenated_my_day_to_day_stays_semantic():
    # Regression: "my day-to-day workflow" must NOT match the synthesis "my day"
    # alternative — the hyphen is a regex word boundary, so without a `(?![\w-])`
    # guard it mis-routed a genuine content query to the chronological scan.
    assert _rag()._intent_window("my day-to-day workflow") is None
    assert _rag()._intent_window("review my day-to-day tasks doc") is None
    # The real synthesis phrases must still resolve to a window.
    assert _rag()._intent_window("Summarize my day") is not None
    assert _rag()._intent_window("my week") is not None


# -- _sample_even -------------------------------------------------------------


def test_sample_even_returns_all_when_within_n():
    assert RAG._sample_even([1, 2, 3], 5) == [1, 2, 3]


def test_sample_even_returns_all_when_exactly_n():
    assert RAG._sample_even([1, 2, 3], 3) == [1, 2, 3]


def test_sample_even_spreads_across_the_list_not_just_the_head():
    items = list(range(10))
    chosen = RAG._sample_even(items, 4)
    assert len(chosen) == 4
    assert chosen[0] == items[0]  # head always represented
    # Spread, not the contiguous earliest-N slice.
    assert chosen != items[:4]
    # Reaches well into the tail (late rows are reachable, not just the head).
    assert max(chosen) >= 7


def test_sample_even_zero_or_negative_n_is_empty():
    assert RAG._sample_even([1, 2, 3], 0) == []
    assert RAG._sample_even([1, 2, 3], -1) == []


# -- _signal_score ------------------------------------------------------------


def test_signal_score_rich_pr_title_outranks_generic_and_tracking():
    rich = _obs("rich", ts=1.0, window="Fix flaky deploy race in pipeline (#42)")
    rich_text = (
        "Reworked the retry/backoff loop, added a regression test, and verified "
        "the capture pipeline no longer drops frames under contention."
    )
    generic = _obs("gen", ts=1.0, window="Codex")  # bare app name => penalized
    tracking = _obs("trk", ts=1.0, window="Browser tab")
    tracking_text = "_id=abc123&uaid=xyz789 cross-site tracking fragment, no signal"

    s_rich = RAG._signal_score(rich, rich_text)
    s_generic = RAG._signal_score(generic, "x")
    s_tracking = RAG._signal_score(tracking, tracking_text)

    assert s_rich > s_generic
    assert s_rich > s_tracking


# -- _answer_temporal: selection grounds on the rich LATE row -----------------


def test_temporal_selection_includes_rich_late_row_not_only_earliest():
    """Earliest rows are low-signal junk; a rich PR row sits late in the day.

    The selection must rank by signal then spread across time, so the chosen
    (cited) context is NOT exclusively the earliest occurrences — the rich late
    row is grounded on.
    """
    # 6 earliest junk rows (generic 'Codex' windows, trivial bodies).
    junk = [
        _row(f"junk{i}", ts=float(i + 1), window="Codex", text="hi")
        for i in range(6)
    ]
    # 6 rich rows LATER in the day (informative window titles + substantial body).
    rich = [
        _row(
            f"rich{i}",
            ts=1000.0 + i,
            window=f"Land deploy race fix in capture pipeline (#4{i})",
            text=(
                f"Detailed work-log entry {i}: refactored the retry path, added a "
                "regression test, and confirmed the fix end to end across sessions."
            ),
        )
        for i in range(6)
    ]
    store = FakeTemporalStore(junk + rich)
    chatter = RAG(store, CiteAllCompleter())
    chatter._now = lambda: NOW

    result = chatter.answer("Summarize my day")

    assert result.grounded
    assert result.citations
    cited_ids = {c.observation_id for c in result.citations}
    cited_ts = [c.ts for c in result.citations]
    # At least one rich late row was selected...
    assert any(cid.startswith("rich") for cid in cited_ids)
    # ...and the selection reaches the late part of the day (not earliest-only).
    assert max(cited_ts) >= 1000.0
    assert max(cited_ts) > max(obs.ts for obs, _text in junk)
    # Routing proven: synthesis used the time-range scan, never semantic search.
    assert store.search_calls == 0
    assert store.range_calls >= 1


# -- grounding gate intact + routing proven -----------------------------------


def test_empty_window_returns_no_activity_and_skips_search():
    store = FakeTemporalStore([])  # no rows in the window
    chatter = RAG(store, CiteAllCompleter())
    chatter._now = lambda: NOW

    result = chatter.answer("Summarize my day")

    assert not result.grounded
    assert result.citations == []
    assert "any recorded activity" in result.answer.lower()
    # Synthesis routed to the time-range scan; semantic search untouched.
    assert store.search_calls == 0
    assert store.range_calls == 1


def test_no_activity_completer_never_invoked():
    completer = CiteAllCompleter()
    chatter = RAG(FakeTemporalStore([]), completer)
    chatter._now = lambda: NOW
    chatter.answer("what should I follow up on?")
    # Empty window short-circuits before any model call.
    assert completer.calls == 0


# -- self-capture never grounds an answer (parity with the briefing path) ------


def test_temporal_path_excludes_self_capture_rows():
    rows = [
        (_obs("self", ts=NOW - 100, app="ai.openbird.OpenBird",
              window="Ask about your work..."), "openbird ui body"),
        (_obs("work", ts=NOW - 50, app="com.google.Chrome",
              window="fix(rag): ground citations"), "real work body"),
    ]
    completer = CiteAllCompleter()
    chatter = RAG(FakeTemporalStore(rows), completer)
    chatter._now = lambda: NOW

    result = chatter.answer("Summarize my day")

    cited_apps = {c.app for c in result.citations}
    assert "ai.openbird.OpenBird" not in cited_apps  # self-capture never cited
    # The self-capture body must not even reach the model prompt.
    assert "openbird ui body" not in completer.last_messages[-1]["content"]
    assert any(c.app == "com.google.Chrome" for c in result.citations)


def test_temporal_path_all_self_capture_returns_no_activity():
    rows = [
        (_obs("s1", ts=NOW - 100, app="ai.openbird.OpenBird", window="UI"), "x"),
        (_obs("s2", ts=NOW - 50, app="ai.openbird.OpenBird.capture-helper",
              window="helper"), "y"),
    ]
    completer = CiteAllCompleter()
    chatter = RAG(FakeTemporalStore(rows), completer)
    chatter._now = lambda: NOW

    result = chatter.answer("Summarize my day")

    assert not result.grounded
    assert "any recorded activity" in result.answer.lower()
    assert completer.calls == 0  # all rows filtered → model never called
