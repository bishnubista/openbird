"""Unit tests for the opt-in RAG grounding diagnostics (``openbird/chat/rag_debug.py``).

Focus areas mirror the Codex review of the design:

  * tier parsing (``OPENBIRD_DEBUG_RAG`` -> None / meta / full);
  * the ``meta`` tier is a strict no-op when the flag is off;
  * PRIVACY: the default ``meta`` tier never echoes captured content — not the
    model's raw citation tokens (which can carry a copied URL / window title),
    not window titles, not snippets;
  * the format-drift probe and signal aggregation are correct and crash-free on
    empty / semantic-path inputs;
  * the ``full`` tier prints captured content but only behind a one-time banner;
  * the wiring: ``RAG._answer_temporal`` actually emits a trace under the flag.

Because the module isolates itself with ``propagate=False`` (so debug logging
can't un-gag other libraries), records are captured by attaching ``caplog``'s
handler directly to the module logger, and module globals are reset per test.
"""

from __future__ import annotations

import logging

import pytest

from openbird.chat import rag_debug as d
from openbird.chat.rag import RAG
from openbird.types import Observation, SearchHit

LOGGER_NAME = "openbird.chat.rag_debug"


class _CI:
    """Duck-typed stand-in for ``rag._ContextItem`` (source_id + hit)."""

    def __init__(self, source_id, hit):
        self.source_id = source_id
        self.hit = hit


def _ctx(source_id="S1", *, app="Ghostty", window="feat: add x", ts=1000.0, text="body"):
    obs = Observation(
        id=source_id, content_hash="h_" + source_id, ts=ts, app=app,
        window=window, url=None, session_id="s", source="capture",
    )
    hit = SearchHit(
        chunk_id=f"obs:{source_id}", content_hash="h_" + source_id,
        text=text, score=0.0, observation=obs,
    )
    return _CI(source_id, hit)


@pytest.fixture(autouse=True)
def _isolate(caplog):
    """Route module records into caplog despite ``propagate=False`` and reset state."""
    lg = logging.getLogger(LOGGER_NAME)
    saved = list(lg.handlers)
    for h in saved:
        lg.removeHandler(h)
    d._handler_attached = False
    d._full_banner_emitted = False
    lg.addHandler(caplog.handler)
    lg.setLevel(logging.INFO)
    yield
    for h in list(lg.handlers):
        lg.removeHandler(h)
    for h in saved:
        lg.addHandler(h)


def _text(caplog) -> str:
    return "\n".join(r.getMessage() for r in caplog.records)


# -- tier parsing --------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        ("0", None), ("off", None), ("false", None), ("no", None),
        ("garbage", None), ("", None),
        ("1", "meta"), ("on", "meta"), ("meta", "meta"), ("true", "meta"),
        ("YES", "meta"), (" Meta ", "meta"),
        ("2", "full"), ("full", "full"), ("FULL", "full"),
    ],
)
def test_debug_level_parsing(monkeypatch, value, expected):
    if value == "":
        monkeypatch.delenv("OPENBIRD_DEBUG_RAG", raising=False)
    else:
        monkeypatch.setenv("OPENBIRD_DEBUG_RAG", value)
    assert d.debug_level() == expected


def test_noop_when_flag_off(monkeypatch, caplog):
    monkeypatch.delenv("OPENBIRD_DEBUG_RAG", raising=False)
    d.emit_grounding_trace(
        route="semantic", raw={"answer": "x", "citations": ["S1"]},
        answer_text="x", claimed_ids=["S1"], citations=[], context=[_ctx()],
    )
    d.emit_retrieval_empty(route="semantic", reason="no_rows")
    assert caplog.records == []


# -- privacy: meta must never echo captured content ----------------------------


def test_meta_does_not_leak_citation_token_content(monkeypatch, caplog):
    monkeypatch.setenv("OPENBIRD_DEBUG_RAG", "meta")
    leak = "https://secret.example.com/p?token=abc123"
    d.emit_grounding_trace(
        route="semantic", raw={"answer": "x", "citations": [leak]},
        answer_text="x", claimed_ids=[leak], citations=[], context=[_ctx()],
    )
    text = _text(caplog)
    assert "secret.example.com" not in text
    assert "token=abc123" not in text
    # The token is still *counted* so the failure mode stays visible.
    assert "claimed_n=1" in text
    assert "unknown=1" in text


def test_meta_does_not_leak_window_or_snippet(monkeypatch, caplog):
    monkeypatch.setenv("OPENBIRD_DEBUG_RAG", "meta")
    ctx = [_ctx("S1", window="Secret PR title", text="Secret snippet body")]
    d.emit_grounding_trace(
        route="intent_synthesis", raw={"answer": "ok", "citations": ["S1"]},
        answer_text="ok", claimed_ids=["S1"], citations=["c"], context=ctx,
    )
    text = _text(caplog)
    assert "Secret PR title" not in text
    assert "Secret snippet body" not in text


# -- format-drift probe + counts ----------------------------------------------


def test_format_drift_counts_wrapped_ids(monkeypatch, caplog):
    monkeypatch.setenv("OPENBIRD_DEBUG_RAG", "meta")
    ctx = [_ctx("S1"), _ctx("S2", ts=2000.0)]
    # S1/S2 wrapped (drift), "Source 2" has no 'S<digit>' substring (uncatchable).
    claimed = ["[source_id: S1]", "(S2)", "Source 2"]
    d.emit_grounding_trace(
        route="intent_synthesis", raw={"answer": "s", "citations": claimed},
        answer_text="s", claimed_ids=claimed, citations=[], context=ctx,
    )
    text = _text(caplog)
    assert "unknown=3" in text
    assert "format_drift=2" in text
    assert "valid=0" in text
    assert "grounded=0" in text
    assert "replaced=1" in text


def test_valid_exact_ids_are_not_drift(monkeypatch, caplog):
    monkeypatch.setenv("OPENBIRD_DEBUG_RAG", "meta")
    ctx = [_ctx("S1"), _ctx("S2", ts=2000.0)]
    d.emit_grounding_trace(
        route="intent_synthesis", raw={"answer": "s", "citations": ["S1"]},
        answer_text="s", claimed_ids=["S1"], citations=["c"], context=ctx,
    )
    text = _text(caplog)
    assert "unknown=0" in text
    assert "format_drift=0" in text
    assert "grounded=1" in text


# -- refusal heuristic (separates H1 narrative-omission from H3 thin-day) -------


@pytest.mark.parametrize(
    "answer,expected",
    [
        ("You spent the morning on the PR.", "refusal=0"),
        ("I don't have that in memory.", "refusal=1"),
        ("I couldn't find anything notable.", "refusal=1"),
    ],
)
def test_refusal_heuristic(monkeypatch, caplog, answer, expected):
    monkeypatch.setenv("OPENBIRD_DEBUG_RAG", "meta")
    d.emit_grounding_trace(
        route="intent_synthesis", raw={"answer": answer, "citations": []},
        answer_text=answer, claimed_ids=[], citations=[], context=[_ctx()],
    )
    assert expected in _text(caplog)


# -- robustness on empty / semantic-path inputs --------------------------------


def test_empty_context_and_string_fallback_do_not_crash(monkeypatch, caplog):
    monkeypatch.setenv("OPENBIRD_DEBUG_RAG", "meta")
    d.emit_grounding_trace(
        route="semantic", raw="raw text fallback", answer_text="",
        claimed_ids=[], citations=[], context=[], retrieval={"hits": 0},
    )
    text = _text(caplog)
    assert "parse=string_fallback" in text
    assert "answer_empty=1" in text
    assert "signal_count=0" in text
    assert "chosen=0" in text


def test_signal_stats_emitted_when_scores_present(monkeypatch, caplog):
    monkeypatch.setenv("OPENBIRD_DEBUG_RAG", "meta")
    d.emit_grounding_trace(
        route="intent_synthesis", raw={"answer": "s", "citations": []},
        answer_text="s", claimed_ids=[], citations=[], context=[_ctx()],
        retrieval={"signal_scores": [-300, -300, 250, 400], "rows": 50, "deduped": 12},
    )
    text = _text(caplog)
    assert "signal_count=4" in text
    assert "lowsig=2" in text
    assert "signal_min=-300" in text
    assert "signal_max=400" in text
    assert "rows=50" in text
    assert "deduped=12" in text


# -- full tier: captured content, but behind a one-time banner -----------------


def test_full_tier_prints_content_behind_one_time_banner(monkeypatch, caplog):
    monkeypatch.setenv("OPENBIRD_DEBUG_RAG", "full")
    ctx = [_ctx("S1", window="Secret window", text="Secret body text")]
    for _ in range(2):
        d.emit_grounding_trace(
            route="intent_synthesis", raw={"answer": "real answer", "citations": ["S1"]},
            answer_text="real answer", claimed_ids=["S1"], citations=["c"], context=ctx,
        )
    text = _text(caplog)
    # Captured content IS allowed at the full tier.
    assert "Secret window" in text
    assert "Secret body text" in text
    assert "real answer" in text
    # ...but the banner fires exactly once across both calls.
    banners = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(banners) == 1


def test_emit_retrieval_empty(monkeypatch, caplog):
    monkeypatch.setenv("OPENBIRD_DEBUG_RAG", "meta")
    d.emit_retrieval_empty(
        route="intent_synthesis", reason="all_self_or_dupe",
        retrieval={"rows": 9, "dropped_self": 9},
    )
    text = _text(caplog)
    assert "route=intent_synthesis" in text
    assert "reason=all_self_or_dupe" in text
    assert "dropped_self=9" in text
    assert "grounded=0" in text


# -- wiring: the temporal path emits a trace under the flag ---------------------


class _Completer:
    def complete(self, messages, *, json_schema=None):
        # Cite whatever source ids the prompt lists, mirroring production headers.
        user = messages[-1]["content"]
        ids = []
        for line in user.splitlines():
            if "[source_id: " in line:
                ids.append(line.split("[source_id: ", 1)[1].split("]", 1)[0])
        return {"answer": "Here is your day.", "citations": ids}


class _Store:
    def __init__(self, rows):
        self._rows = rows

    def time_range_text(self, start, end, *, max_chars=2000):
        return list(self._rows)

    def search(self, query, k=10, *, semantic=True):  # present but unused here
        return []


def test_answer_temporal_emits_trace_under_flag(monkeypatch, caplog):
    monkeypatch.setenv("OPENBIRD_DEBUG_RAG", "meta")
    rows = [
        (Observation(id="a", content_hash="ha", ts=1000.0, app="Chrome",
                     window="github PR #1 title here", url=None, session_id="s",
                     source="capture"), "pr body one"),
        (Observation(id="b", content_hash="hb", ts=9000.0, app="Ghostty",
                     window="feat: implement the thing", url=None, session_id="s",
                     source="capture"), "commit work two"),
    ]
    rag = RAG(_Store(rows), _Completer())
    result = rag._answer_temporal("Summarize my day", (0.0, 1e12), route="intent_synthesis")
    assert result.grounded  # the cite-all completer grounds it
    grounding = [r for r in caplog.records if r.getMessage().startswith("rag.grounding ")]
    assert grounding, "expected a rag.grounding trace line"
    msg = grounding[-1].getMessage()
    assert "route=intent_synthesis" in msg
    assert "rows=2" in msg
    assert "grounded=1" in msg


def test_intent_specific_query_emits_distinct_route(monkeypatch, caplog):
    """A day-windowed SPECIFIC question (not synthesis phrasing) routes through
    _answer_scoped_specific and must be labelled ``route=intent_specific`` — distinct
    from the broad-synthesis ``intent_temporal``/``intent_synthesis`` labels — so the
    trace maps to the actual answer path. Regression: PR #157 added the
    scoped-specific path but it inherited the ``intent_temporal`` label, making the
    two day-window paths indistinguishable in diagnostics."""
    monkeypatch.setenv("OPENBIRD_DEBUG_RAG", "meta")
    rows = [
        (Observation(id="a", content_hash="ha", ts=1000.0, app="Mail",
                     window="Mail — renewal", url=None, session_id="s",
                     source="capture"),
         "Asked Alice to confirm the renewal deadline and invoice owner."),
    ]
    rag = RAG(_Store(rows), _Completer())
    rag._now = lambda: 1e9
    # "today" yields a window; "what did I ask Alice" is NOT synthesis phrasing.
    rag.answer("what did I ask Alice today?")
    grounding = [r for r in caplog.records if r.getMessage().startswith("rag.grounding ")]
    assert grounding, "expected a rag.grounding trace line"
    assert "route=intent_specific" in grounding[-1].getMessage()


def test_synthesis_query_keeps_intent_temporal_route(monkeypatch, caplog):
    """A broad synthesis query with a temporal word stays ``route=intent_temporal``
    (the _answer_temporal path) — the new specific-path label must not capture it."""
    monkeypatch.setenv("OPENBIRD_DEBUG_RAG", "meta")
    rows = [
        (Observation(id="a", content_hash="ha", ts=1000.0, app="Chrome",
                     window="github PR #1 title here", url=None, session_id="s",
                     source="capture"), "pr body one"),
    ]
    rag = RAG(_Store(rows), _Completer())
    rag._now = lambda: 1e9
    rag.answer("what did I work on yesterday?")
    grounding = [r for r in caplog.records if r.getMessage().startswith("rag.grounding ")]
    assert grounding
    assert "route=intent_temporal" in grounding[-1].getMessage()
