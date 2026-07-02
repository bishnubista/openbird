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


def test_eval_specific_day_query_stays_out_of_synthesis_router():
    # Mirrors quality_eval.DAY_SCOPED_ASK_QUERIES' specific-path guard. If this
    # starts matching synthesis, eval quality no longer exercises
    # explicit_window_specific with a real model.
    assert RAG._is_synthesis_query("What about OpenBird?") is False


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


def test_explicit_window_specific_question_uses_query_relevance():
    """A day-scoped specific question must not use summary-only selection.

    Regression for the real "Ask about this day" failure: a low-signal Mail row
    containing the asked-about name was present in the selected day, but the
    broad summary selector filled all six source slots with richer engineering
    rows and the model had no way to answer.
    """
    rich_rows = [
        _row(
            f"rich{i}",
            ts=float(i * 1000),
            window=f"Land release pipeline improvement #{100 + i}",
            text=(
                "Detailed engineering work on notarization, packaging, tests, "
                "release notes, and CI cleanup. "
            ) * 3,
        )
        for i in range(24)
    ]
    alice = _row(
        "alice",
        ts=12_345.0,
        window="Mail",
        text="Asked Alice to confirm the renewal deadline and invoice owner.",
    )
    store = FakeTemporalStore(rich_rows + [alice])
    chatter = RAG(store, CiteAllCompleter())

    result = chatter.answer("what did I ask Alice about?", window=(0.0, 99_999.0))

    assert result.grounded
    assert any(c.observation_id == "alice" for c in result.citations)
    assert store.search_calls == 0  # hard-scoped range scan, not global search
    assert store.range_calls == 1


def test_explicit_window_specific_question_abstains_on_zero_overlap():
    rows = [
        _row(
            "deploy",
            ts=100.0,
            window="Deploy checklist",
            text="Reviewed release packaging and notarization steps.",
        )
    ]
    completer = CiteAllCompleter()
    chatter = RAG(FakeTemporalStore(rows), completer)

    result = chatter.answer("what about Alice?", window=(0.0, 999.0))

    assert not result.grounded
    assert result.citations == []
    assert result.used_hits == []
    assert "memory" in result.answer.lower()
    assert completer.calls == 0


def test_explicit_window_specific_question_keeps_short_numeric_ids():
    rows = [
        _row(
            "issue42",
            ts=100.0,
            window="Fix capture race #42",
            text="Regression notes for issue #42 and its verification steps.",
        ),
        _row(
            "other",
            ts=200.0,
            window="Release notes",
            text="General packaging cleanup with no issue reference.",
        ),
    ]
    chatter = RAG(FakeTemporalStore(rows), CiteAllCompleter())

    result = chatter.answer("what about #42?", window=(0.0, 999.0))

    assert result.grounded
    assert {c.observation_id for c in result.citations} == {"issue42"}


def test_temporal_selection_drops_low_signal_rows_when_enough_signal_exists():
    """Noisy days should not spend source slots on low-signal rows."""
    noise = [
        _row(
            f"noise{i}",
            ts=float(i * 100),
            window="Codex" if i % 2 else "New Tab",
            text="_id=abc&uaid=xyz transient browser tracking fragment",
        )
        for i in range(60)
    ]
    rich = [
        _row(
            f"event{i}",
            ts=float(ts),
            window=f"Land user-visible OpenBird feature slice #{200 + i}",
            text=(
                f"Important milestone {i}: completed focused work with tests, "
                "review, and handoff notes. "
            ) * 3,
        )
        for i, ts in enumerate([500, 8500, 17000, 26000, 35000, 47000, 59000, 71000])
    ]
    store = FakeTemporalStore(sorted(noise + rich, key=lambda p: p[0].ts))
    chatter = RAG(store, CiteAllCompleter())

    result = chatter.answer("Summarize my day")

    assert result.grounded
    assert len(result.used_hits) == 6
    cited_ids = {c.observation_id for c in result.citations}
    assert all(cid.startswith("event") for cid in cited_ids)
    assert not any(cid.startswith("noise") for cid in cited_ids)


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


# -- synthesis answering persona (PR1: fixes the whole-day-summary refusal) -----


def test_synthesis_and_strict_personas_differ():
    """The synthesis variant drops the strict single-source abstention clause but
    keeps the identical security scaffold."""
    from openbird.chat.rag import _SYNTHESIS_SYSTEM_PROMPT, _SYSTEM_PROMPT

    assert _SYSTEM_PROMPT != _SYNTHESIS_SYSTEM_PROMPT
    # The strict QA persona tells the model to abstain when a source doesn't
    # literally contain the answer — exactly what made it refuse a day summary.
    assert "say you don't have that in memory" in _SYSTEM_PROMPT
    assert "say you don't have that in memory" not in _SYNTHESIS_SYSTEM_PROMPT
    # The synthesis persona reframes the context as the user's own activity.
    assert "Synthesize" in _SYNTHESIS_SYSTEM_PROMPT
    # Both retain the framework security scaffold (fence tokens survive render).
    for prompt in (_SYSTEM_PROMPT, _SYNTHESIS_SYSTEM_PROMPT):
        assert "OPENBIRD_UNTRUSTED_CONTEXT" in prompt
        # Grounding contract (cite only listed sources) is preserved in both.
        assert "cite" in prompt.lower()


def test_synthesis_intent_sends_synthesis_persona_to_model():
    """A whole-day summary routes through the time-range scan with the synthesis
    system prompt — not the strict QA persona that triggers refusal."""
    rows = [
        _row("a", ts=NOW - 3600, window="github PR #1 title here", text="pr body"),
        _row("b", ts=NOW - 60, window="feat: implement the thing", text="commit"),
    ]
    completer = CiteAllCompleter()
    chatter = RAG(FakeTemporalStore(rows), completer)
    chatter._now = lambda: NOW

    chatter.answer("Summarize my day")

    assert completer.last_messages is not None
    assert completer.last_messages[0]["content"] == chatter._synthesis_system_prompt
    assert "Synthesize" in completer.last_messages[0]["content"]


def test_scoped_day_pointed_qa_keeps_strict_persona():
    """An explicit --day question that is NOT a summary keeps the strict QA
    persona (hard-scoped pointed lookups answer, not summarize) — Codex finding #1."""
    rows = [
        _row("a", ts=NOW - 3600, window="Rocket design doc", text="the rocket uses X"),
    ]
    completer = CiteAllCompleter()
    chatter = RAG(FakeTemporalStore(rows), completer)
    chatter._now = lambda: NOW

    # window=... mirrors `openbird chat "..." --day 0` (explicit hard scope).
    chatter.answer("what about the rocket?", window=(NOW - _DAY, NOW))

    assert completer.last_messages is not None
    assert completer.last_messages[0]["content"] == chatter._system_prompt
    assert completer.last_messages[0]["content"] != chatter._synthesis_system_prompt


def test_scoped_day_summary_uses_synthesis_persona():
    """An explicit --day summary DOES synthesize — intent gating is on the query."""
    rows = [_row("a", ts=NOW - 3600, window="github PR #1", text="pr body")]
    completer = CiteAllCompleter()
    chatter = RAG(FakeTemporalStore(rows), completer)
    chatter._now = lambda: NOW

    chatter.answer("summarize my day", window=(NOW - _DAY, NOW))

    assert completer.last_messages is not None
    assert completer.last_messages[0]["content"] == chatter._synthesis_system_prompt


def test_synthesis_persona_honors_user_override(monkeypatch):
    """A user OPENBIRD_PROMPT_RAG_SYNTHESIS override reaches temporal answers —
    Codex finding #2 (override semantics preserved for the new prompt). The env
    var carries inline persona text (never a path)."""
    from openbird.config import reset_settings_cache

    monkeypatch.setenv("OPENBIRD_PROMPT_RAG_SYNTHESIS", "CUSTOM SYNTHESIS PERSONA LINE.")
    reset_settings_cache()
    try:
        rows = [_row("a", ts=NOW - 60, window="github PR #1", text="pr body")]
        completer = CiteAllCompleter()
        chatter = RAG(FakeTemporalStore(rows), completer)
        chatter._now = lambda: NOW

        chatter.answer("Summarize my day")

        sys_prompt = completer.last_messages[0]["content"]
        assert "CUSTOM SYNTHESIS PERSONA LINE." in sys_prompt
        # Security scaffold survives the override.
        assert "OPENBIRD_UNTRUSTED_CONTEXT" in sys_prompt
    finally:
        reset_settings_cache()


def test_semantic_path_keeps_strict_persona():
    """Content lookups (non-synthesis) still use the strict single-source persona."""
    from openbird.types import SearchHit

    class _SearchStore:
        def time_range_text(self, start, end, *, max_chars=2000):
            return []

        def search(self, query, k=10, *, semantic=True):
            obs = _obs("d1", ts=NOW - 10, window="Auth design doc", app="Notes")
            return [SearchHit(
                chunk_id="c1", content_hash=obs.content_hash, text="auth uses JWT",
                score=1.0, observation=obs,
            )]

    completer = CiteAllCompleter()
    chatter = RAG(_SearchStore(), completer)
    chatter._now = lambda: NOW

    chatter.answer("what does the auth doc say")

    assert completer.last_messages is not None
    assert completer.last_messages[0]["content"] == chatter._system_prompt


def test_pointed_temporal_phrasing_matching_synthesis_re_uses_synthesis():
    """A pointed temporal query that MATCHES _SYNTHESIS_RE ("what did I do at 3pm")
    intent-routes through the time-range scan and uses the synthesis persona BY
    DESIGN — it already gets the broad activity sample, so synthesizing it beats
    abstaining. Pins the decision behind Codex's routing-vs-persona finding so the
    comment and behavior cannot silently drift apart again."""
    from openbird.chat.rag import _SYNTHESIS_RE

    query = "what did I do at 3pm"
    assert _SYNTHESIS_RE.search(query) is not None  # the regex match Codex flagged

    rows = [_row("a", ts=NOW - 3600, window="github PR #1 title", text="pr body")]
    completer = CiteAllCompleter()
    chatter = RAG(FakeTemporalStore(rows), completer)
    chatter._now = lambda: NOW

    chatter.answer(query)  # no explicit window → intent-routed, not hard-scoped

    assert completer.last_messages is not None
    assert completer.last_messages[0]["content"] == chatter._synthesis_system_prompt


def test_content_query_with_do_prefix_word_stays_semantic():
    """A content query whose word merely STARTS with a synthesis verb — "what did
    I document/download about X" (the 'do' in 'document') — must NOT route to the
    synthesis scan. Codex round-2 finding: _SYNTHESIS_RE needs a word boundary
    after the verb so the bare verb can't swallow a longer word."""
    from openbird.chat.rag import _SYNTHESIS_RE

    for content in (
        "what did I document about security",
        "what did I download the report",
        "what have I workshopped this week",
    ):
        assert _SYNTHESIS_RE.search(content) is None, content
    # ...the genuine synthesis phrasings still match.
    for synth in ("what did I do", "what did I do at 3pm", "what did I work on"):
        assert _SYNTHESIS_RE.search(synth) is not None, synth


# -- Phase E1: cached week answer + summary-first multi-day context ----------------


class RaisingCompleter:
    """Proves a path is provider-free: any completion call fails the test."""

    def complete(self, messages, *, json_schema=None):  # pragma: no cover
        raise AssertionError("provider must not be called on the cached path")


def _block(i, *, ts, text, date, key=None):
    return {
        "id": f"bs{i}",
        "local_date": date,
        "block_key": key or f"k{i}",
        "block_fingerprint": f"f{i}",
        "start_ts": ts,
        "end_ts": ts + 900.0,
        "dominant_bundle": "b",
        "level": None,
        "summary_text": text,
        "model": "m",
        "extractor_version": "block-summary-v1",
        "generated_at": ts,
        "source_count": 1,
        "source_refs": [{"source_kind": "observation", "source_id": f"obs{i}"}],
    }


def _week_row(monday, digest, *, summary_ids=("bs1",), start=None, end=None):
    return {
        "id": f"wk-{monday}",
        "local_date": monday,
        "source_scope": "week",
        "extractor_version": "week-memory-v1",
        "generated_at": NOW,
        "source_count": len(summary_ids),
        "source_ids": [],
        "span_ids": [],
        "summary_ids": list(summary_ids),
        "source_refs": [
            {"source_kind": "summary", "source_id": sid} for sid in summary_ids
        ],
        "payload": {
            "week_start_date": monday,
            "digest_text": digest,
            "member_fingerprint": "mf",
            "window": {"start": start, "end": end},
        },
    }


class SummaryStore(FakeTemporalStore):
    """FakeTemporalStore plus the Phase E1 summary readers."""

    def __init__(self, rows=None, *, weeks=None, blocks=None, day_memories=None,
                 summary_search=None):
        super().__init__(rows or [])
        self.weeks = list(weeks or [])
        self.blocks = list(blocks or [])
        self.day_memories = dict(day_memories or {})
        self.summary_search = list(summary_search or [])
        self.search_summaries_calls = 0

    def week_memories_overlapping(self, start, end):
        return list(self.weeks)

    def block_summaries_for_range(self, start, end):
        return [
            b for b in self.blocks
            if b["start_ts"] <= end and b["end_ts"] >= start
        ]

    def block_summaries_for_date(self, local_date):
        return [b for b in self.blocks if b["local_date"] == local_date]

    def get_day_memory(self, *, local_date, source_scope="capture"):
        return self.day_memories.get(local_date)

    def search_summaries(self, query, k=6, *, semantic=True):
        self.search_summaries_calls += 1
        return list(self.summary_search)


def _date_of(ts: float) -> str:
    return _dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")


def _multiday_window() -> tuple[float, float]:
    return NOW - 3 * _DAY, NOW


def _week_window() -> tuple[float, float]:
    # Week-shaped (>= 6 days): the cached week digest is gated to explicit
    # week-recap intents with windows of at least this span.
    return NOW - 7 * _DAY, NOW


def _day_memory_for(ts: float, *, active_seconds=1800.0):
    return {
        "id": f"dm-{_date_of(ts)}",
        "local_date": _date_of(ts),
        "payload": {
            "local_date": _date_of(ts),
            "coverage": {"observations": 5, "sessions": 1, "apps": 1},
            "metrics": {"active_seconds": active_seconds},
        },
    }


def test_cached_week_answer_is_terminal_and_provider_free():
    """Digest present -> composed cached answer; ZERO provider calls (raising
    stub), route local_cached_model_summary, typed week_memory + block_summary
    derived citations with full provenance."""
    window = _week_window()
    b1 = _block(1, ts=NOW - 2 * _DAY, text="Refactored the capture daemon.",
                date=_date_of(NOW - 2 * _DAY))
    store = SummaryStore(
        weeks=[_week_row("2026-06-22", "A week of capture work.",
                         start=window[0], end=window[1])],
        blocks=[b1],
        day_memories={_date_of(NOW - 2 * _DAY): _day_memory_for(NOW - 2 * _DAY)},
    )
    rag = RAG(store, RaisingCompleter())
    rag._now = lambda: NOW
    result = rag.answer("summarize my week", window=window)
    assert result.reasoning_route == "local_cached_model_summary"
    assert result.grounding == "derived"
    assert result.grounded is True
    assert "A week of capture work." in result.answer
    assert "Refactored the capture daemon." in result.answer
    assert "Deterministic totals" in result.answer
    types = [c.type for c in result.derived_citations]
    assert "week_memory" in types and "block_summary" in types
    week_cite = next(c for c in result.derived_citations if c.type == "week_memory")
    assert week_cite.source_id == "wk-2026-06-22"
    assert week_cite.derived_from_refs == [
        {"source_kind": "summary", "source_id": "bs1"}
    ]
    assert store.range_calls == 0 and store.search_calls == 0


def test_intent_week_query_reaches_cached_week_answer():
    """The no-explicit-window path ('my week' synthesis intent) also lands on
    the cached week answer when a digest exists."""
    window = _multiday_window()
    store = SummaryStore(
        weeks=[_week_row("2026-06-22", "Digest via intent window.",
                         start=window[0], end=window[1])],
        blocks=[_block(1, ts=NOW - 2 * _DAY, text="Block prose.",
                       date=_date_of(NOW - 2 * _DAY))],
    )
    rag = RAG(store, RaisingCompleter())
    rag._now = lambda: NOW
    result = rag.answer("recap my week")
    assert result.reasoning_route == "local_cached_model_summary"
    assert "Digest via intent window." in result.answer


def test_no_digest_but_day_summaries_still_cached_terminal():
    """Fallback ladder: no digest but per-day block summaries exist -> STILL the
    provider-free cached composition (narrative lines + totals), mirroring the
    cached day path."""
    window = _week_window()
    blocks = [
        _block(1, ts=NOW - 2.5 * _DAY, text="Worked on schema migrations.",
               date=_date_of(NOW - 2.5 * _DAY)),
    ]
    store = SummaryStore(blocks=blocks)
    rag = RAG(store, RaisingCompleter())
    rag._now = lambda: NOW
    result = rag.answer("summarize my week", window=window)
    assert result.reasoning_route == "local_cached_model_summary"
    assert "Worked on schema migrations." in result.answer
    assert {c.type for c in result.derived_citations} == {"block_summary"}


class RangeOnlySummaryStore(SummaryStore):
    """Blocks visible to the RANGE reader only (no per-day narrative source):
    exercises the summary-first MODEL branch behind the cached week path."""

    def block_summaries_for_date(self, local_date):
        return []


def test_no_cached_prose_falls_through_to_summary_first_completion():
    """Fallback ladder step 2: no digest and no per-day narrative -> FRESH
    provider completion over summary context (derived-only grounding passes the
    gate); raw rows untouched."""
    window = _multiday_window()
    blocks = [
        _block(1, ts=NOW - 2.5 * _DAY, text="Worked on schema migrations.",
               date=_date_of(NOW - 2.5 * _DAY)),
        _block(2, ts=NOW - 1.2 * _DAY, text="Wrote the retrieval tests.",
               date=_date_of(NOW - 1.2 * _DAY)),
    ]
    completer = CiteAllCompleter()
    store = RangeOnlySummaryStore(blocks=blocks)
    rag = RAG(store, completer)
    rag._now = lambda: NOW
    result = rag.answer("summarize my week", window=window)
    assert completer.calls == 1
    content = completer.last_messages[-1]["content"]
    assert "(block summary " in content
    assert "Worked on schema migrations." in content
    # Fresh completion keeps its truthful completion route (None for a stub
    # with no llm_model), NEVER the cached-summary label.
    assert result.reasoning_route != "local_cached_model_summary"
    assert result.grounded is True
    assert result.grounding == "derived"
    assert result.citations == []
    assert {c.type for c in result.derived_citations} == {"block_summary"}
    assert [c.source_id for c in result.derived_citations] == ["bs1", "bs2"]
    assert result.derived_citations[0].derived_from_refs == [
        {"source_kind": "observation", "source_id": "obs1"}
    ]
    assert store.range_calls == 0  # summary-first: no raw-row scan


def test_fewer_than_two_summaries_falls_back_to_raw_rows():
    """Fallback ladder step 3: <2 summary items -> the raw-row path,
    byte-identical legacy behavior (occurrence citations)."""
    window = _multiday_window()
    rows = [_row("a", ts=NOW - 2 * _DAY,
                 window="github PR #42 long informative title",
                 text="pull request body with plenty of detail " * 3)]
    store = RangeOnlySummaryStore(
        rows,
        blocks=[_block(1, ts=NOW - 2 * _DAY, text="only one block",
                       date=_date_of(NOW - 2 * _DAY))],
    )
    completer = CiteAllCompleter()
    rag = RAG(store, completer)
    rag._now = lambda: NOW
    result = rag.answer("summarize my week", window=window)
    assert store.range_calls == 1
    assert result.citations and result.citations[0].observation_id == "a"
    assert result.derived_citations == []
    assert result.grounding == "occurrence"


def test_week_memory_ungated_when_model_cites_nothing():
    """The grounding gate still replaces an uncited summary-context answer."""

    class NoCiteCompleter:
        def complete(self, messages, *, json_schema=None):
            return {"answer": "vague uncited prose", "citations": []}

    window = _multiday_window()
    blocks = [
        _block(1, ts=NOW - 2.5 * _DAY, text="alpha", date=_date_of(NOW - 2.5 * _DAY)),
        _block(2, ts=NOW - 1.5 * _DAY, text="beta", date=_date_of(NOW - 1.5 * _DAY)),
    ]
    rag = RAG(RangeOnlySummaryStore(blocks=blocks), NoCiteCompleter())
    rag._now = lambda: NOW
    result = rag.answer("summarize my week", window=window)
    assert result.grounding == "ungrounded"
    assert "vague uncited prose" not in result.answer


def test_single_day_synthesis_keeps_raw_path():
    """A single-day window must NOT take the summary-first path (raw chunks
    stay the detail path for day questions)."""
    today = _today_start()
    rows = [_row("a", ts=today + 3600.0,
                 window="feat: informative commit message title",
                 text="commit body text with details " * 4)]
    store = SummaryStore(
        rows,
        blocks=[
            _block(1, ts=today + 3600.0, text="block one", date=_date_of(today)),
            _block(2, ts=today + 7200.0, text="block two", date=_date_of(today)),
        ],
    )
    completer = CiteAllCompleter()
    rag = RAG(store, completer)
    rag._now = lambda: NOW
    result = rag.answer("summarize my day", window=(today, NOW))
    assert store.range_calls == 1  # raw scan, not summary context
    assert result.derived_citations == []


def test_specific_multiday_question_ranks_summaries_lexically():
    """Specific questions over a multi-day window use summary context ranked by
    query overlap (+ search_summaries), with the strict QA persona."""
    window = _multiday_window()
    blocks = [
        _block(1, ts=NOW - 2.5 * _DAY, text="Debugged the sqlcipher keychain flow.",
               date=_date_of(NOW - 2.5 * _DAY)),
        _block(2, ts=NOW - 1.5 * _DAY, text="Wrote homebrew cask release notes.",
               date=_date_of(NOW - 1.5 * _DAY)),
        _block(3, ts=NOW - 1.0 * _DAY, text="More keychain access debugging.",
               date=_date_of(NOW - 1.0 * _DAY)),
    ]
    completer = CiteAllCompleter()
    store = SummaryStore(blocks=blocks)
    rag = RAG(store, completer)
    rag._now = lambda: NOW
    result = rag.answer("what did I decide about the keychain?", window=window)
    assert completer.calls == 1
    content = completer.last_messages[-1]["content"]
    assert "keychain" in content
    assert "homebrew cask" not in content  # zero-overlap summary dropped
    cited = {c.source_id for c in result.derived_citations}
    assert cited == {"bs1", "bs3"}
    # Strict QA persona (not synthesis) for specific questions.
    assert "ACROSS" not in completer.last_messages[0]["content"]
    assert store.range_calls == 0


def test_specific_multiday_zero_overlap_falls_back_to_raw():
    window = _multiday_window()
    rows = [_row("a", ts=NOW - 2 * _DAY, window="quarterly plan document",
                 text="quarterly plan contents")]
    store = SummaryStore(
        rows,
        blocks=[
            _block(1, ts=NOW - 2.5 * _DAY, text="alpha work",
                   date=_date_of(NOW - 2.5 * _DAY)),
            _block(2, ts=NOW - 1.5 * _DAY, text="beta work",
                   date=_date_of(NOW - 1.5 * _DAY)),
        ],
    )
    completer = CiteAllCompleter()
    rag = RAG(store, completer)
    rag._now = lambda: NOW
    rag.answer("what did I write in the quarterly plan?", window=window)
    assert store.range_calls == 1  # summaries matched nothing -> raw detail path


def test_semantic_path_merges_summaries_capped_at_two_slots():
    """No-window semantic retrieval admits AT MOST 2 summary items of the 6
    context slots, RRF-merged with occurrence hits."""
    from openbird.types import SearchHit

    hits = [
        SearchHit(
            chunk_id=f"c{i}", content_hash=f"h{i}",
            text=f"occurrence chunk number {i} about the widget",
            score=1.0 - i * 0.05,
            observation=_obs(f"o{i}", ts=1000.0 + i),
        )
        for i in range(8)
    ]
    summary_results = [
        {
            "summary_kind": "block", "summary_id": f"bs{i}",
            "text": f"summary narrative {i} about the widget",
            "score": 0.9, "start_ts": 1000.0, "end_ts": 2000.0,
            "local_date": "2026-06-22",
            "source_refs": [{"source_kind": "observation", "source_id": f"o{i}"}],
        }
        for i in range(3)
    ]

    class SemanticStore(SummaryStore):
        def search(self, query, k=10, *, semantic=True):
            self.search_calls += 1
            return list(hits)

    store = SemanticStore(summary_search=summary_results)
    completer = CiteAllCompleter()
    rag = RAG(store, completer)
    rag._now = lambda: NOW
    result = rag.answer("tell me about the widget")
    assert store.search_calls == 1
    assert store.search_summaries_calls == 1
    content = completer.last_messages[-1]["content"]
    assert content.count("[source_id: ") == 6  # capped total context
    assert content.count("(block summary ") == 2  # summary slots capped at 2
    assert len(result.derived_citations) == 2
    assert len(result.citations) == 4
    assert result.grounding == "mixed"


def test_semantic_path_without_summary_index_is_unchanged():
    """A store without search_summaries keeps the historical occurrence-only
    behavior (hasattr-guarded merge)."""
    from openbird.types import SearchHit

    hits = [
        SearchHit(
            chunk_id="c1", content_hash="h1", text="the only chunk",
            score=1.0, observation=_obs("o1", ts=1000.0),
        )
    ]

    class OccStore(FakeTemporalStore):
        def search(self, query, k=10, *, semantic=True):
            self.search_calls += 1
            return list(hits)

    completer = CiteAllCompleter()
    rag = RAG(OccStore([]), completer)
    result = rag.answer("tell me about the widget")
    assert result.grounding == "occurrence"
    assert [c.observation_id for c in result.citations] == ["o1"]
    assert result.derived_citations == []
