from __future__ import annotations

import datetime as dt
import json

import pytest
from typer.testing import CliRunner

from openbird import cli
from openbird.config import reset_settings_cache
from openbird.day_memory import build_day_memory, classify_observation
from openbird.types import Observation


@pytest.fixture(autouse=True)
def _clean_settings_cache():
    yield
    reset_settings_cache()


def _ts(y: int, mo: int, d: int, h: int, mi: int = 0) -> float:
    return dt.datetime(y, mo, d, h, mi).timestamp()


def _obs(
    oid: str,
    *,
    ts: float,
    app: str = "com.google.Chrome",
    window: str | None = None,
    url: str | None = None,
    session_id: str | None = None,
) -> Observation:
    return Observation(
        id=oid,
        content_hash=f"h-{oid}",
        ts=ts,
        app=app,
        window=window,
        url=url,
        session_id=session_id,
        source="capture",
    )


def test_classify_observation_uses_descriptive_categories():
    obs = _obs(
        "yt",
        ts=1.0,
        window="How to debug launchd - YouTube",
        url="https://www.youtube.com/watch?v=abc",
    )
    assert classify_observation(obs)[0] == "browser_media"

    code = _obs("code", ts=2.0, app="com.mitchellh.ghostty", window="openbird")
    assert classify_observation(code)[0] == "coding"


def test_build_day_memory_payload_has_metrics_sources_and_no_narrative_text():
    start = _ts(2026, 6, 12, 9)
    rows = [
        (
            _obs(
                "o1",
                ts=start,
                app="com.mitchellh.ghostty",
                window="openbird git status",
                session_id="s1",
            ),
            "ran tests in openbird",
        ),
        (
            _obs(
                "o2",
                ts=start + 60,
                app="com.google.Chrome",
                window="bishnubista/openbird: Local-first memory - Google Chrome",
                url="https://github.com/bishnubista/openbird",
                session_id="s2",
            ),
            "github repo",
        ),
        (
            _obs(
                "o3",
                ts=start + 360,
                app="com.google.Chrome",
                window="Debugging SwiftUI - YouTube",
                url="https://youtube.com/watch?v=debug",
                session_id="s3",
            ),
            "video title",
        ),
    ]

    built = build_day_memory(
        rows,
        start_ts=start,
        end_ts=start + 3600,
        day_offset=0,
        gap_seconds=300,
    )

    payload = built.payload
    assert built.source_ids == ["o1", "o2", "o3"]
    assert payload["narrative_status"] == "not_persisted"
    assert "narrative" not in payload
    assert payload["coverage"]["observations"] == 3
    assert payload["metrics"]["context_switch_count"] == 2
    assert payload["metrics"]["time_by_category"]["coding"] == 60
    assert payload["metrics"]["time_by_category"]["browser_research"] == 300
    assert any(d["value"] == "github.com" for d in payload["entities"]["domains"])
    assert any(d["value"] == "youtube.com" for d in payload["entities"]["domains"])


class _DayMemoryStoreStub:
    def __init__(self, rows):
        self.rows = rows
        self.saved = None

    def time_range_text(self, start, end, *, source=None):
        return [(o, t) for o, t in self.rows if start <= o.ts <= end]

    def save_day_memory(self, **kwargs):
        self.saved = kwargs
        return {
            "id": "dm1",
            "local_date": kwargs["local_date"],
            "source_scope": kwargs["source_scope"],
            "extractor_version": kwargs["extractor_version"],
            "generated_at": 123.0,
            "source_count": len(kwargs["source_ids"]),
            "source_ids": kwargs["source_ids"],
            "payload": kwargs["payload"],
        }

    def get_day_memory(self, *, local_date, source_scope="capture"):
        if self.saved is None:
            return None
        return self.save_day_memory(**self.saved)

    def close(self):
        pass


def test_day_memory_build_cli_is_no_model_and_json(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENBIRD_DATA_DIR", str(tmp_path))
    reset_settings_cache()
    today = dt.datetime.now().replace(hour=9, minute=0, second=0, microsecond=0).timestamp()
    rows = [
        (_obs("o1", ts=today, app="com.mitchellh.ghostty", window="openbird"), "coding"),
    ]
    stub = _DayMemoryStoreStub(rows)
    monkeypatch.setattr(cli, "_store_maintenance", lambda: stub)
    monkeypatch.setattr(
        cli,
        "_provider",
        lambda: (_ for _ in ()).throw(AssertionError("day-memory must not build provider")),
    )

    res = CliRunner().invoke(cli.app, ["day-memory", "build", "--day", "0", "--json"])
    assert res.exit_code == 0, res.output
    payload = json.loads(res.stdout)
    assert payload["built"] is True
    assert payload["day_memory"]["source_ids"] == ["o1"]
    assert payload["day_memory"]["payload"]["narrative_status"] == "not_persisted"
