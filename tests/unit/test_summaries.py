"""Unit tests for the idle-time block summarizer (Phase D).

Covers: compute_span_blocks parity with day-memory span_focus_blocks, the
battery/idle gate truth table, settle/batch/skip/fingerprint-drift runner
behavior, citation validation (hallucinated dropped; zero-valid stores
nothing), the taxonomy LLM-fallback pass with caching, and the no-content
logging contract.
"""

from __future__ import annotations

import json
import logging
import re

import pytest

from openbird.config import Settings
from openbird.memory.store import MemoryStore
from openbird.summaries import (
    Block,
    block_fingerprint,
    block_key,
    build_block_summary_messages,
    compute_span_blocks,
    format_counts_line,
    run_block_summaries,
    should_run_background_llm,
)
from openbird import summaries as summaries_mod

from tests.unit.conftest import FakeProvider


def _span(sid, start, end, *, bundle="com.apple.mail", host=None, window=None,
          afk=0, reason=None):
    return {
        "span_id": sid,
        "start_ts": start,
        "end_ts": end,
        "bundle_id": bundle,
        "url_host": host,
        "window": window,
        "afk": afk,
        "reason": reason,
        "detail_tier": 0 if reason else 1,
    }


class _CiteAllProvider:
    """Completes with a summary citing every S-label present in the prompt."""

    llm_model = "stub-model"

    def __init__(self, summary="Worked on the memory schema in Mail."):
        self.summary = summary
        self.calls: list[list[dict]] = []
        self.schemas: list[dict | None] = []

    def complete(self, messages, *, json_schema=None):
        self.calls.append(messages)
        self.schemas.append(json_schema)
        labels = re.findall(r"\[source_id: (S\d+)\]", messages[-1]["content"])
        return {"summary": self.summary, "citation_ids": labels}


class _FixedProvider:
    llm_model = "stub-model"

    def __init__(self, response):
        self.response = response
        self.calls = 0

    def complete(self, messages, *, json_schema=None):
        self.calls += 1
        return self.response


@pytest.fixture
def store(mem_settings, fake_provider) -> MemoryStore:
    s = MemoryStore(db_path=":memory:", settings=mem_settings, provider=fake_provider)
    yield s
    s.close()


@pytest.fixture(autouse=True)
def _on_ac(monkeypatch):
    """Default the gate to AC power; gate tests re-patch _pmset_output."""
    monkeypatch.setattr(
        summaries_mod, "_pmset_output", lambda: "Now drawing from 'AC Power'"
    )


def _seed_block(store, *, start=1000.0, end=1900.0) -> tuple[str, str]:
    """Open two spans forming one >=10min block; return their ids."""
    s1 = store.open_span(
        epoch_id="e", start_ts=start, end_ts=start + 400.0,
        bundle_id="com.apple.mail", detail_tier=1,
    )
    s2 = store.open_span(
        epoch_id="e", start_ts=start + 430.0, end_ts=end,
        bundle_id="com.apple.Notes", detail_tier=1,
    )
    return s1, s2


# -- compute_span_blocks ---------------------------------------------------------


def test_compute_span_blocks_rules():
    spans = [
        _span("s1", 1000.0, 1400.0),
        _span("s2", 1430.0, 1900.0, bundle="com.apple.Notes"),
        # >60s gap breaks the run; the next span alone is too short.
        _span("s3", 2200.0, 2400.0),
        # AFK and paused spans never join a block.
        _span("s4", 2400.0, 3200.0, afk=1),
        _span("s5", 3200.0, 4000.0, bundle=None, reason="paused"),
    ]
    blocks = compute_span_blocks(spans)
    assert len(blocks) == 1
    assert blocks[0].span_ids == ("s1", "s2")
    assert blocks[0].start_ts == 1000.0 and blocks[0].end_ts == 1900.0
    # Dominance counts SPANS (the pre-lift _span_metrics semantics): a one-one
    # tie keeps insertion order, so the first bundle wins.
    assert blocks[0].dominant_bundle == "com.apple.mail"


def test_compute_span_blocks_parity_with_day_memory_span_focus_blocks():
    """The lifted extractor and _span_metrics report the SAME blocks."""
    from openbird.day_memory import build_day_memory

    spans = [
        _span("s1", 1000.0, 1400.0),
        _span("s2", 1430.0, 1900.0, bundle="com.apple.Notes"),
        _span("s3", 2200.0, 2400.0),
        _span("s4", 2500.0, 3300.0, bundle="com.apple.dt.Xcode"),
        _span("s5", 3300.0, 3400.0, afk=1),
    ]
    built = build_day_memory(
        [], start_ts=0.0, end_ts=100_000.0, day_offset=0, spans=spans
    )
    payload_blocks = built.payload["span_metrics"]["span_focus_blocks"]
    blocks = compute_span_blocks(spans, start_ts=0.0, end_ts=100_000.0)
    assert [
        {
            "start": b.start_ts,
            "end": b.end_ts,
            "seconds": round(b.end_ts - b.start_ts, 3),
            "dominant_bundle": b.dominant_bundle,
            "span_ids": list(b.span_ids),
        }
        for b in blocks
    ] == payload_blocks


def test_block_fingerprint_uses_real_span_ends():
    spans_a = [_span("s1", 1000.0, 1400.0), _span("s2", 1430.0, 1900.0)]
    spans_b = [_span("s1", 1000.0, 1400.0), _span("s2", 1430.0, 2000.0)]
    (block_a,) = compute_span_blocks(spans_a)
    (block_b,) = compute_span_blocks(spans_b)
    assert block_key(block_a) == block_key(block_b)  # same membership
    assert block_fingerprint(block_a) != block_fingerprint(block_b)  # extension


# -- gate truth table -------------------------------------------------------------


def _write_sidecar(settings, *, updated_at, afk=True, meeting=None, malformed=False):
    path = settings.data_dir / "capture.liveness.json"
    if malformed:
        path.write_text("{not json")
        return
    payload = {"updated_at": updated_at, "afk": afk}
    if meeting is not None:  # None = pre-C1 sidecar without the key
        payload["meeting"] = meeting
    path.write_text(json.dumps(payload))


def test_gate_ac_power_always_allows(monkeypatch, tmp_path):
    settings = Settings(data_dir=tmp_path)
    monkeypatch.setattr(summaries_mod, "_pmset_output", lambda: "Now drawing from 'AC Power'")
    allowed, reason = should_run_background_llm(None, settings, now=1000.0)
    assert (allowed, reason) == (True, "ac_power")


def test_gate_pmset_failure_is_battery_fail_closed(monkeypatch, tmp_path):
    settings = Settings(data_dir=tmp_path)
    monkeypatch.setattr(summaries_mod, "_pmset_output", lambda: None)
    # No sidecar at all: on (assumed) battery the run defers.
    allowed, reason = should_run_background_llm(None, settings, now=1000.0)
    assert (allowed, reason) == (False, "battery_liveness_stale")


def test_gate_battery_fresh_afk_allows(monkeypatch, tmp_path):
    settings = Settings(data_dir=tmp_path)
    monkeypatch.setattr(summaries_mod, "_pmset_output", lambda: "Battery Power")
    now = 10_000.0
    _write_sidecar(settings, updated_at=now - 5.0, afk=True)
    assert should_run_background_llm(None, settings, now=now) == (True, "battery_user_afk")
    # Boundary: exactly the shared 30s staleness bound is still fresh.
    _write_sidecar(settings, updated_at=now - 30.0, afk=True)
    assert should_run_background_llm(None, settings, now=now) == (True, "battery_user_afk")


def test_gate_battery_active_user_defers(monkeypatch, tmp_path):
    settings = Settings(data_dir=tmp_path)
    monkeypatch.setattr(summaries_mod, "_pmset_output", lambda: "Battery Power")
    now = 10_000.0
    _write_sidecar(settings, updated_at=now - 5.0, afk=False)
    assert should_run_background_llm(None, settings, now=now) == (
        False, "battery_user_active"
    )


def test_gate_battery_stale_future_malformed_absent_defer(monkeypatch, tmp_path):
    settings = Settings(data_dir=tmp_path)
    monkeypatch.setattr(summaries_mod, "_pmset_output", lambda: "Battery Power")
    now = 10_000.0
    expect = (False, "battery_liveness_stale")
    # Stale: older than the shared 30s bound (a stopped/wedged capture daemon
    # must never make an ACTIVE user look idle).
    _write_sidecar(settings, updated_at=now - 30.1, afk=True)
    assert should_run_background_llm(None, settings, now=now) == expect
    # Future timestamp: negative age is INVALID, not fresh.
    _write_sidecar(settings, updated_at=now + 60.0, afk=True)
    assert should_run_background_llm(None, settings, now=now) == expect
    # Non-finite timestamp.
    _write_sidecar(settings, updated_at="not-a-number", afk=True)
    assert should_run_background_llm(None, settings, now=now) == expect
    # Malformed sidecar.
    _write_sidecar(settings, updated_at=0, malformed=True)
    assert should_run_background_llm(None, settings, now=now) == expect
    # Absent sidecar.
    (settings.data_dir / "capture.liveness.json").unlink()
    assert should_run_background_llm(None, settings, now=now) == expect


def test_gate_fresh_meeting_defers_even_on_ac(monkeypatch, tmp_path):
    # Meeting deferral is checked FIRST: Zoom GPU/CPU contention exists on AC
    # power too (the screenpipe lesson).
    settings = Settings(data_dir=tmp_path)
    monkeypatch.setattr(
        summaries_mod, "_pmset_output", lambda: "Now drawing from 'AC Power'"
    )
    now = 10_000.0
    _write_sidecar(settings, updated_at=now - 5.0, afk=False, meeting=True)
    assert should_run_background_llm(None, settings, now=now) == (
        False, "meeting_live"
    )


def test_gate_battery_meeting_outranks_afk_allowance(monkeypatch, tmp_path):
    settings = Settings(data_dir=tmp_path)
    monkeypatch.setattr(summaries_mod, "_pmset_output", lambda: "Battery Power")
    now = 10_000.0
    _write_sidecar(settings, updated_at=now - 5.0, afk=True, meeting=True)
    # On a call you look AFK (not typing) — the meeting bit must still defer.
    assert should_run_background_llm(None, settings, now=now) == (
        False, "meeting_live"
    )


def test_gate_stale_sidecar_never_defers_on_meeting(monkeypatch, tmp_path):
    # Fail-open on meeting ONLY: a dead daemon (stale sidecar, lost
    # mic_stopped) must not block summaries forever. Battery staleness stays
    # fail-closed, unchanged.
    settings = Settings(data_dir=tmp_path)
    now = 10_000.0
    _write_sidecar(settings, updated_at=now - 31.0, afk=True, meeting=True)
    monkeypatch.setattr(
        summaries_mod, "_pmset_output", lambda: "Now drawing from 'AC Power'"
    )
    assert should_run_background_llm(None, settings, now=now) == (True, "ac_power")
    monkeypatch.setattr(summaries_mod, "_pmset_output", lambda: "Battery Power")
    assert should_run_background_llm(None, settings, now=now) == (
        False, "battery_liveness_stale"
    )


def test_gate_missing_meeting_key_means_no_meeting_deferral(monkeypatch, tmp_path):
    # A pre-C1 sidecar (no "meeting" key) stays valid: coerced to False.
    settings = Settings(data_dir=tmp_path)
    monkeypatch.setattr(
        summaries_mod, "_pmset_output", lambda: "Now drawing from 'AC Power'"
    )
    now = 10_000.0
    _write_sidecar(settings, updated_at=now - 5.0, afk=False)
    assert should_run_background_llm(None, settings, now=now) == (True, "ac_power")


def test_gate_uses_shared_staleness_constant():
    from openbird.capture.health import DAEMON_STALE_AFTER_SECONDS

    assert summaries_mod.DAEMON_STALE_AFTER_SECONDS is DAEMON_STALE_AFTER_SECONDS


# -- runner -----------------------------------------------------------------------


def test_runner_summarizes_settled_block_with_typed_refs(store, mem_settings):
    s1, s2 = _seed_block(store)
    obs = store.add_observation(
        "Drafting the memory schema doc", source="capture", ts=1200.0,
        app="com.apple.mail", span_id=s1,
    )
    provider = _CiteAllProvider()
    now = 1900.0 + 3600.0  # block settled
    counts = run_block_summaries(store, provider, now=now, settings=mem_settings)
    assert counts["summarized"] == 1
    assert counts["ungrounded"] == 0
    assert counts["deferred_reason"] is None
    rows = store.block_summaries_for_range(0.0, 1e12)
    assert len(rows) == 1
    row = rows[0]
    assert row["summary_text"] == "Worked on the memory schema in Mail."
    assert row["extractor_version"] == "block-summary-v1"
    assert row["model"] == "stub-model"
    # The dominant bundle (Mail, first on the one-span-each tie) has an
    # other_work DEFAULT_RULES entry -> the block level resolves from it.
    assert row["dominant_bundle"] == "com.apple.mail"
    assert row["level"] == "other_work"
    kinds = {(r["source_kind"], r["source_id"]) for r in row["source_refs"]}
    assert ("span", s1) in kinds and ("span", s2) in kinds
    assert ("observation", obs.id) in kinds

    # Re-run: fingerprint match -> skipped, zero provider calls.
    calls_before = len(provider.calls)
    counts = run_block_summaries(store, provider, now=now, settings=mem_settings)
    assert counts == {
        "summarized": 0, "skipped": 1, "ungrounded": 0, "classified": 0,
        "deferred_reason": None,
    }
    assert len(provider.calls) == calls_before


def test_runner_regenerates_on_fingerprint_drift(store, mem_settings):
    _s1, s2 = _seed_block(store)
    provider = _CiteAllProvider()
    now = 1900.0 + 3600.0
    run_block_summaries(store, provider, now=now, settings=mem_settings)
    first = store.block_summaries_for_range(0.0, 1e12)[0]

    store.extend_span(s2, 2100.0)  # member span extended -> fingerprint drifts
    counts = run_block_summaries(store, provider, now=now + 3600.0, settings=mem_settings)
    assert counts["summarized"] == 1
    rows = store.block_summaries_for_range(0.0, 1e12)
    assert len(rows) == 1  # regenerate replaces (same block_key)
    assert rows[0]["id"] != first["id"]
    assert rows[0]["block_fingerprint"] != first["block_fingerprint"]


def test_runner_settle_rule_defers_fresh_blocks_and_force_bypasses(store, mem_settings):
    _seed_block(store)
    provider = _CiteAllProvider()
    now = 1900.0 + 100.0  # block ended 100s ago < 900s settle
    counts = run_block_summaries(store, provider, now=now, settings=mem_settings)
    assert counts["summarized"] == 0 and counts["skipped"] == 0
    assert provider.calls == []
    counts = run_block_summaries(
        store, provider, now=now, settings=mem_settings, force=True
    )
    assert counts["summarized"] == 1


def test_runner_batch_limit_bounds_one_pass(store, tmp_path, fake_provider):
    settings = Settings(data_dir=tmp_path, embed_dim=768, block_summaries_batch_limit=2)
    for i in range(3):
        base = 1000.0 + i * 5000.0
        store.open_span(
            epoch_id="e", start_ts=base, end_ts=base + 900.0,
            bundle_id="com.apple.mail", detail_tier=1,
        )
    provider = _CiteAllProvider()
    counts = run_block_summaries(store, provider, now=50_000.0, settings=settings)
    assert counts["summarized"] == 2
    assert len(store.block_summary_keys()) == 2
    # The next pass picks up the remainder.
    counts = run_block_summaries(store, provider, now=50_000.0, settings=settings)
    assert counts["summarized"] == 1 and counts["skipped"] == 2


def test_runner_zero_valid_citations_stores_nothing(store, mem_settings, caplog):
    _seed_block(store)
    provider = _FixedProvider(
        {"summary": "totally made up", "citation_ids": ["S99", "nonsense"]}
    )
    with caplog.at_level(logging.INFO, logger="openbird.summaries"):
        counts = run_block_summaries(
            store, provider, now=10_000.0, settings=mem_settings
        )
    assert counts["summarized"] == 0
    assert counts["ungrounded"] == 1
    assert store.block_summary_keys() == {}
    assert any(
        "block_summary_ungrounded" in r.getMessage() for r in caplog.records
    )


def test_runner_drops_hallucinated_but_keeps_valid_citations(store, mem_settings):
    s1, _s2 = _seed_block(store)
    provider = _FixedProvider(
        {"summary": "grounded bit", "citation_ids": ["S1", "S42", "S1"]}
    )
    counts = run_block_summaries(store, provider, now=10_000.0, settings=mem_settings)
    assert counts["summarized"] == 1
    row = store.block_summaries_for_range(0.0, 1e12)[0]
    # S1 is the first span line; the hallucinated S42 and the duplicate drop.
    assert row["source_refs"] == [{"source_kind": "span", "source_id": s1}]


def test_runner_gate_defers_without_provider_calls(store, mem_settings, monkeypatch):
    _seed_block(store)
    monkeypatch.setattr(summaries_mod, "_on_ac_power", lambda: False)
    provider = _CiteAllProvider()
    counts = run_block_summaries(store, provider, now=10_000.0, settings=mem_settings)
    assert counts["deferred_reason"] == "battery_liveness_stale"
    assert provider.calls == []
    assert store.block_summary_keys() == {}


def test_runner_disabled_setting_defers(store, tmp_path):
    settings = Settings(data_dir=tmp_path, embed_dim=768, block_summaries_enabled=False)
    provider = _CiteAllProvider()
    counts = run_block_summaries(store, provider, now=10_000.0, settings=settings)
    assert counts["deferred_reason"] == "disabled"
    assert provider.calls == []


def test_runner_never_logs_summary_text(store, mem_settings, caplog):
    _seed_block(store)
    secret = "SECRET-SUMMARY-BODY-marker"
    provider = _CiteAllProvider(summary=secret)
    with caplog.at_level(logging.DEBUG):
        counts = run_block_summaries(
            store, provider, now=10_000.0, settings=mem_settings
        )
    assert counts["summarized"] == 1
    log_text = "\n".join(r.getMessage() for r in caplog.records)
    assert secret not in log_text
    assert secret not in format_counts_line(counts)


# -- taxonomy LLM fallback ---------------------------------------------------------


class _ClassifyingProvider(_CiteAllProvider):
    """Summarizes blocks AND answers taxonomy classification calls."""

    def __init__(self, level="personal"):
        super().__init__()
        self.level = level
        self.classify_calls = 0

    def complete(self, messages, *, json_schema=None):
        if json_schema and "level" in (json_schema.get("properties") or {}):
            self.classify_calls += 1
            return {"level": self.level}
        return super().complete(messages, json_schema=json_schema)


def test_runner_classifies_uncategorized_identity_and_caches(store, mem_settings):
    # An unknown-bundle block (>=2min active) with no rule/override/cache.
    store.open_span(
        epoch_id="e", start_ts=1000.0, end_ts=1900.0,
        bundle_id="com.unknown.app", detail_tier=1,
    )
    provider = _ClassifyingProvider(level="personal")
    counts = run_block_summaries(store, provider, now=10_000.0, settings=mem_settings)
    assert counts["classified"] == 1
    assert provider.classify_calls == 1
    assert store.get_category_assignments() == {"bundle:com.unknown.app": "personal"}

    # Second run: cached -> ZERO further classification calls.
    counts = run_block_summaries(store, provider, now=10_000.0, settings=mem_settings)
    assert counts["classified"] == 0
    assert provider.classify_calls == 1


def test_runner_taxonomy_batch_limit(store, tmp_path):
    settings = Settings(data_dir=tmp_path, embed_dim=768, taxonomy_llm_batch_limit=1)
    for i, bundle in enumerate(["com.unknown.one", "com.unknown.two"]):
        base = 1000.0 + i * 5000.0
        store.open_span(
            epoch_id="e", start_ts=base, end_ts=base + 300.0,
            bundle_id=bundle, detail_tier=1,
        )
    provider = _ClassifyingProvider()
    counts = run_block_summaries(store, provider, now=50_000.0, settings=settings)
    assert counts["classified"] == 1
    assert provider.classify_calls == 1


def test_runner_short_identities_below_threshold_never_classified(store, mem_settings):
    store.open_span(
        epoch_id="e", start_ts=1000.0, end_ts=1090.0,  # 90s < 120s threshold
        bundle_id="com.unknown.brief", detail_tier=1,
    )
    provider = _ClassifyingProvider()
    run_block_summaries(store, provider, now=10_000.0, settings=mem_settings)
    assert provider.classify_calls == 0


def test_runner_rejected_level_is_not_cached(store, mem_settings):
    store.open_span(
        epoch_id="e", start_ts=1000.0, end_ts=1900.0,
        bundle_id="com.unknown.app", detail_tier=1,
    )
    provider = _ClassifyingProvider(level="productive")  # off-enum
    counts = run_block_summaries(store, provider, now=10_000.0, settings=mem_settings)
    assert counts["classified"] == 0
    assert store.get_category_assignments() == {}


# -- prompt shape -------------------------------------------------------------------


def test_block_summary_messages_fence_and_label_map(store):
    s1, s2 = _seed_block(store)
    obs = store.add_observation(
        "body text", source="capture", ts=1200.0, span_id=s1,
        app="com.apple.mail",
    )
    (block,) = compute_span_blocks(store.spans_in_range(0.0, 1e12))
    rows = summaries_mod._block_observation_rows(store, block)
    messages, label_map = build_block_summary_messages(block, rows)
    fence = summaries_mod._FENCE
    user = messages[1]["content"]
    assert user.count(fence.open_token) == 1
    assert user.count(fence.close_token) == 1
    assert label_map == {
        "S1": ("span", s1),
        "S2": ("span", s2),
        "S3": ("observation", obs.id),
    }
    assert "minutes=" in user  # span metadata line present


def test_block_observation_rows_window_fallback_and_cap(store):
    _s1, _s2 = _seed_block(store)
    # NULL span_id but inside the window -> kept (fallback); outside -> dropped.
    inside = store.add_observation("inside window", source="capture", ts=1500.0)
    store.add_observation("outside window", source="capture", ts=5000.0)
    # Linked to a foreign span -> dropped.
    foreign = store.open_span(
        epoch_id="e", start_ts=4000.0, end_ts=4700.0,
        bundle_id="com.apple.mail", detail_tier=1,
    )
    store.add_observation("foreign span", source="capture", ts=1600.0, span_id=foreign)
    (block, *_rest) = compute_span_blocks(store.spans_in_range(0.0, 2000.0))
    rows = summaries_mod._block_observation_rows(store, block)
    assert [obs.id for obs, _ in rows] == [inside.id]


# -- Codex diff-review regressions (capture-only inputs; --date clipping) --------


def test_block_inputs_are_capture_rows_only(store, mem_settings):
    # A non-capture NULL-span observation inside the block window must be
    # excluded from summarizer grounding; a legacy capture NULL-span row kept.
    _seed_block(store)
    store.add_observation(
        "meeting transcript line that must not ground a block summary",
        source="meeting", ts=1100.0,
    )
    legacy = store.add_observation(
        "legacy capture text predating span linking with plenty of characters",
        source="capture", ts=1200.0,
    )
    provider = _CiteAllProvider()
    counts = run_block_summaries(store, provider, now=10_000.0, settings=mem_settings)
    assert counts["summarized"] == 1
    prompt = provider.calls[0][-1]["content"]
    assert "meeting transcript line" not in prompt
    assert "legacy capture text" in prompt
    saved = store.block_summaries_for_range(0.0, 10_000.0)[0]
    obs_refs = {
        r["source_id"]
        for r in saved["source_refs"]
        if r["source_kind"] == "observation"
    }
    assert legacy.id in obs_refs  # the legacy capture row grounds the summary


def test_date_run_clips_cross_midnight_block(store, mem_settings):
    # A block crossing midnight, built for --date D, is stored under D with a
    # clipped start — never under the previous local_date.
    import datetime as dt

    day_start = dt.datetime(2026, 1, 6, 0, 0, 0).timestamp()
    day_end = day_start + 86400.0
    sid = store.open_span(
        epoch_id="e", start_ts=day_start - 1800.0, end_ts=day_start + 1800.0,
        bundle_id="com.apple.mail", detail_tier=1,
    )
    store.add_observation(
        "work continuing across midnight with enough text to ground",
        source="capture", ts=day_start + 60.0, span_id=sid,
    )
    provider = _CiteAllProvider()
    counts = run_block_summaries(
        store, provider, now=day_end + 10_000.0, settings=mem_settings,
        window=(day_start, day_end), force=True,
    )
    assert counts["summarized"] == 1
    rows = store.block_summaries_for_date("2026-01-06")
    assert len(rows) == 1
    assert rows[0]["start_ts"] >= day_start
    assert store.block_summaries_for_date("2026-01-05") == []


def test_adjacent_date_builds_keep_both_clipped_summaries(store, mem_settings):
    # Codex round-2 regression: building Jan 5 then Jan 6 over the same
    # cross-midnight block must yield TWO rows (date-qualified keys), and
    # rebuilding Jan 5 must not steal Jan 6's row (or vice versa).
    import datetime as dt

    day6 = dt.datetime(2026, 1, 6, 0, 0, 0).timestamp()
    day5 = day6 - 86400.0
    sid = store.open_span(
        epoch_id="e", start_ts=day6 - 1800.0, end_ts=day6 + 1800.0,
        bundle_id="com.apple.mail", detail_tier=1,
    )
    store.add_observation(
        "cross midnight work grounding text long enough to select",
        source="capture", ts=day6 + 60.0, span_id=sid,
    )
    provider = _CiteAllProvider()
    now = day6 + 86400.0 + 10_000.0
    for start, end in [(day5, day6), (day6, day6 + 86400.0), (day5, day6)]:
        run_block_summaries(
            store, provider, now=now, settings=mem_settings,
            window=(start, end), force=True,
        )
    jan5 = store.block_summaries_for_date("2026-01-05")
    jan6 = store.block_summaries_for_date("2026-01-06")
    assert len(jan5) == 1 and len(jan6) == 1
    assert jan5[0]["end_ts"] <= day6 and jan6[0]["start_ts"] >= day6
