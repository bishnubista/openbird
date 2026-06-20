"""Tests for the Today/day-view backend: day_sessions, active_seconds, the
template run_window seam, and the `timeline` CLI command."""

from __future__ import annotations

import datetime as dt
import json

import pytest
from typer.testing import CliRunner

from openbird import cli
from openbird.config import reset_settings_cache
from openbird.memory.store import MemoryStore
from openbird.routines.templates import get_template
from openbird.types import Observation


def _store(mem_settings, fake_provider) -> MemoryStore:
    return MemoryStore(db_path=":memory:", settings=mem_settings, provider=fake_provider)


def _day(y: int, mo: int, d: int, h: int = 0, mi: int = 0) -> float:
    return dt.datetime(y, mo, d, h, mi, 0).timestamp()


def test_day_sessions_groups_by_session(mem_settings, fake_provider):
    store = _store(mem_settings, fake_provider)
    try:
        base = _day(2026, 6, 12, 9)
        store.add_observation("a1", source="capture", app="Code", session_id="s1", ts=base)
        store.add_observation("a2", source="capture", app="Code", session_id="s1", ts=base + 60)
        store.add_observation("b1", source="capture", app="Zoom", session_id="s2", ts=base + 600)
        # Outside the queried day — must be excluded.
        store.add_observation("c1", source="capture", app="Code", session_id="s3", ts=base + 5 * 86400)

        sessions = store.day_sessions(_day(2026, 6, 12), _day(2026, 6, 12, 23, 59))
        assert len(sessions) == 2
        by_id = {s.session_id: s for s in sessions}
        assert by_id["s1"].app == "Code" and by_id["s1"].count == 2
        assert by_id["s2"].app == "Zoom" and by_id["s2"].count == 1
        assert by_id["s1"].start_ts == base and by_id["s1"].end_ts == base + 60
        # Ordered chronologically by session start.
        assert sessions[0].start_ts <= sessions[1].start_ts
    finally:
        store.close()


def test_day_sessions_null_sessions_do_not_collapse(mem_settings, fake_provider):
    """Legacy rows with NULL session_id must NOT merge into one false session."""
    store = _store(mem_settings, fake_provider)
    try:
        base = _day(2026, 6, 12, 9)
        store.add_observation("n1", source="capture", app="Code", session_id=None, ts=base)
        store.add_observation("n2", source="capture", app="Code", session_id=None, ts=base + 60)

        sessions = store.day_sessions(_day(2026, 6, 12), _day(2026, 6, 12, 23, 59))
        assert len(sessions) == 2
        assert all(s.session_id is None and s.app == "Code" and s.count == 1 for s in sessions)
    finally:
        store.close()


def test_day_sessions_excludes_non_capture_sources(mem_settings, fake_provider):
    """The Today timeline is capture activity; ingest/mcp rows must not appear."""
    store = _store(mem_settings, fake_provider)
    try:
        base = _day(2026, 6, 12, 9)
        store.add_observation("cap", source="capture", app="Code", session_id="s1", ts=base)
        store.add_observation("note", source="ingest", ts=base + 60)

        sessions = store.day_sessions(_day(2026, 6, 12), _day(2026, 6, 12, 23, 59))
        assert len(sessions) == 1
        assert sessions[0].app == "Code" and sessions[0].session_id == "s1"
    finally:
        store.close()


def test_active_seconds_only_counts_capture(mem_settings, fake_provider):
    store = _store(mem_settings, fake_provider)
    try:
        base = _day(2026, 6, 12, 9)
        store.add_observation("c1", source="capture", app="Code", ts=base)
        store.add_observation("c2", source="capture", app="Code", ts=base + 30)
        store.add_observation("i1", source="ingest", ts=base + 45)  # must not count

        active = store.active_seconds(_day(2026, 6, 12), _day(2026, 6, 12, 23, 59), 300.0)
        assert active == pytest.approx(30.0)
    finally:
        store.close()


def test_active_seconds_caps_gaps(mem_settings, fake_provider):
    store = _store(mem_settings, fake_provider)
    try:
        base = _day(2026, 6, 12, 9)
        store.add_observation("x1", source="capture", app="Code", ts=base)
        store.add_observation("x2", source="capture", app="Code", ts=base + 30)          # +30 (< gap)
        store.add_observation("x3", source="capture", app="Code", ts=base + 30 + 10_000)  # big gap -> capped

        # 30 + min(10000, 300) = 330
        active = store.active_seconds(_day(2026, 6, 12), _day(2026, 6, 13, 23, 59), 300.0)
        assert active == pytest.approx(330.0)
    finally:
        store.close()


def test_active_seconds_single_observation_is_zero(mem_settings, fake_provider):
    store = _store(mem_settings, fake_provider)
    try:
        store.add_observation("only", source="capture", app="Code", ts=_day(2026, 6, 12, 9))
        active = store.active_seconds(_day(2026, 6, 12), _day(2026, 6, 12, 23, 59), 300.0)
        assert active == 0.0
    finally:
        store.close()


def test_run_window_empty_does_not_call_provider():
    template = get_template("yesterday")

    class Boom:
        def complete(self, messages):  # pragma: no cover - must not run
            raise AssertionError("LLM must not be called on an empty window")

    class EmptyStore:
        def time_range_text(self, start, end):
            return []

    out = template.run_window(EmptyStore(), Boom(), 0.0, 100.0)
    assert "No activity" in out


def test_run_window_uses_explicit_bounds_and_provider():
    template = get_template("yesterday")
    seen = {}

    class Store:
        def time_range_text(self, start, end):
            seen["bounds"] = (start, end)
            obs = Observation(
                id="1", content_hash="h", ts=50.0, app="Code",
                window=None, url=None, session_id="s", source="capture",
            )
            return [(obs, "did stuff")]

    class Provider:
        def complete(self, messages):
            return "SUMMARY"

    out = template.run_window(Store(), Provider(), 10.0, 90.0)
    assert out == "SUMMARY"
    assert seen["bounds"] == (10.0, 90.0)


def test_timeline_cli_json_empty(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENBIRD_DATA_DIR", str(tmp_path))
    reset_settings_cache()  # get_settings() is lru_cached; pick up the temp data dir
    res = CliRunner().invoke(cli.app, ["timeline", "--json", "--day", "0"])
    assert res.exit_code == 0, res.output
    payload = json.loads(res.stdout)
    assert payload["day_offset"] == 0
    assert payload["total_observations"] == 0
    assert payload["distinct_apps"] == 0
    assert payload["active_seconds"] == 0
    assert payload["sessions"] == []


def test_timeline_cli_rejects_negative_day(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENBIRD_DATA_DIR", str(tmp_path))
    reset_settings_cache()
    res = CliRunner().invoke(cli.app, ["timeline", "--day=-1", "--json"])
    assert res.exit_code == 2
