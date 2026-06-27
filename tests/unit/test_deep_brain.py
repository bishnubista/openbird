from __future__ import annotations

import datetime as dt
import json
import logging

from typer.testing import CliRunner

from openbird import cli
from openbird.config import Settings, reset_settings_cache
from openbird.deep_brain import build_deep_brain_preview
from openbird.types import Observation


def _day(y: int, mo: int, d: int, h: int = 0, mi: int = 0) -> float:
    return dt.datetime(y, mo, d, h, mi, 0).timestamp()


def _obs(
    id_: str,
    *,
    h: str,
    ts: float,
    app: str | None = "Code",
    window: str | None = None,
    source: str = "capture",
) -> Observation:
    return Observation(
        id=id_,
        content_hash=h,
        ts=ts,
        app=app,
        window=window,
        url=None,
        session_id=id_,
        source=source,
    )


def _has_key(value, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_has_key(v, key) for v in value.values())
    if isinstance(value, list):
        return any(_has_key(v, key) for v in value)
    return False


def test_preview_builds_locally_with_cloud_off_and_no_source_ids(tmp_path):
    start = _day(2026, 6, 12, 9)
    rows = [(_obs("o1", h="h1", ts=start, window="openbird issue"), "fix issue")]
    settings = Settings(data_dir=tmp_path)

    packet = build_deep_brain_preview(
        rows,
        start_ts=start,
        end_ts=start + 3600,
        day_offset=0,
        source_scope="capture",
        settings=settings,
    )

    assert packet["route"] == "deep_brain.preview"
    assert packet["egress"] == "none_preview"
    assert packet["cloud_ready"] is False
    assert "OPENBIRD_ALLOW_CLOUD" in " ".join(packet["blocked_reasons"])
    assert "OPENBIRD_DEEP_BRAIN_ENABLED" in " ".join(packet["blocked_reasons"])
    assert _has_key(packet["memory_summary"], "source_ids") is False
    assert "source_fingerprint" not in packet["memory_summary"]


def test_preview_cloud_ready_requires_both_opt_ins(tmp_path):
    start = _day(2026, 6, 12, 9)
    rows = [(_obs("o1", h="h1", ts=start), "notes")]

    packet = build_deep_brain_preview(
        rows,
        start_ts=start,
        end_ts=start + 60,
        day_offset=0,
        source_scope="capture",
        settings=Settings(data_dir=tmp_path, allow_cloud=True, deep_brain_enabled=True),
    )

    assert packet["cloud_ready"] is True
    assert packet["blocked_reasons"] == []


def test_exclusions_apply_before_distillation_and_sources(tmp_path):
    start = _day(2026, 6, 12, 9)
    rows = [
        (
            _obs(
                "secret-terminal",
                h="h1",
                ts=start,
                app="com.mitchellh.ghostty",
                window="SECRET_TERMINAL_FOLLOWUP github.com/acme/private-repo issue #7",
            ),
            "todo SECRET_TERMINAL_FOLLOWUP github.com/acme/private-repo issue #7",
        ),
        (_obs("private-source", h="h2", ts=start + 30, source="private"), "PRIVATE_SOURCE_TOKEN"),
        (_obs("private-id", h="h3", ts=start + 40, window="PRIVATE_ID_WINDOW"), "PRIVATE_ID_TOKEN"),
        (_obs("kept", h="h4", ts=start + 60, window="public notes"), "public notes"),
    ]
    settings = Settings(
        data_dir=tmp_path,
        deep_brain_excluded_apps=["com.mitchellh.ghostty"],
        deep_brain_excluded_sources=["private"],
        deep_brain_excluded_observation_ids=["private-id"],
    )

    packet = build_deep_brain_preview(
        rows,
        start_ts=start,
        end_ts=start + 3600,
        day_offset=0,
        source_scope="capture",
        settings=settings,
    )
    rendered = json.dumps(packet["memory_summary"], sort_keys=True)
    rendered_sources = json.dumps(packet["selected_sources"], sort_keys=True)

    assert "SECRET_TERMINAL_FOLLOWUP" not in rendered
    assert "private-repo" not in rendered
    assert "PRIVATE_SOURCE_TOKEN" not in rendered
    assert "PRIVATE_ID_WINDOW" not in rendered
    assert "secret-terminal" not in rendered_sources
    assert "private-source" not in rendered_sources
    assert "private-id" not in rendered_sources
    assert packet["exclusions"]["excluded_by"] == {
        "app": 1,
        "observation_id": 1,
        "source": 1,
    }
    assert packet["exclusions"]["kept_observations"] == 1


def test_unknown_app_kept_is_counted_when_app_exclusion_configured(tmp_path):
    start = _day(2026, 6, 12, 9)
    rows = [(_obs("unknown", h="h1", ts=start, app=None), "unknown app row")]
    packet = build_deep_brain_preview(
        rows,
        start_ts=start,
        end_ts=start + 60,
        day_offset=0,
        source_scope="capture",
        settings=Settings(data_dir=tmp_path, deep_brain_excluded_apps=["Code"]),
    )

    assert packet["exclusions"]["unknown_app_kept"] == 1
    assert packet["exclusions"]["kept_observations"] == 1


def test_preview_builder_does_not_log_content(tmp_path, caplog):
    start = _day(2026, 6, 12, 9)
    rows = [(_obs("o1", h="h1", ts=start, window="PRIVATE_WINDOW"), "PRIVATE_TEXT")]
    with caplog.at_level(logging.INFO):
        build_deep_brain_preview(
            rows,
            start_ts=start,
            end_ts=start + 60,
            day_offset=0,
            source_scope="capture",
            settings=Settings(data_dir=tmp_path),
        )

    assert caplog.records == []


class _PreviewStore:
    def __init__(self, rows):
        self.rows = rows

    def time_range_text(self, start, end, source="capture"):
        return [(obs, text) for obs, text in self.rows if source is None or obs.source == source]

    def close(self):
        pass


def test_deep_brain_preview_cli_uses_maintenance_store_not_model_provider(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENBIRD_DATA_DIR", str(tmp_path))
    reset_settings_cache()
    start = _day(*dt.datetime.now().timetuple()[:3], 9)
    rows = [(_obs("o1", h="h1", ts=start, window="public notes"), "public notes")]
    used_maintenance_store = False

    def _fake_store_maintenance():
        nonlocal used_maintenance_store
        used_maintenance_store = True
        return _PreviewStore(rows)

    monkeypatch.setattr(cli, "_store_maintenance", _fake_store_maintenance)
    monkeypatch.setattr(
        cli,
        "_provider",
        lambda: (_ for _ in ()).throw(
            AssertionError("preview must not use configured model provider")
        ),
    )

    res = CliRunner().invoke(cli.app, ["deep-brain", "preview", "--day", "0", "--json"])

    assert res.exit_code == 0, res.output
    assert used_maintenance_store is True
    packet = json.loads(res.stdout)
    assert packet["route"] == "deep_brain.preview"
    assert packet["egress"] == "none_preview"
    assert packet["selected_sources"][0]["observation_id"] == "o1"
