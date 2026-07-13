"""Read-only desktop-assistant connector tests."""

from __future__ import annotations

import asyncio
import json
import stat
from pathlib import Path

import pytest
from typer.testing import CliRunner

from openbird import assistant, cli
from openbird.assistant import AssistantCaptureService
from openbird.config import Settings
from openbird.memory.store import MemoryStore
from openbird.types import Observation
from tests.unit.conftest import FakeProvider


def _observation(
    id_: str,
    *,
    app: str | None = "com.example.Editor",
    source: str = "capture",
    ts: float = 1000.0,
) -> Observation:
    return Observation(
        id=id_,
        content_hash=f"hash-{id_}",
        ts=ts,
        app=app,
        window="Sensitive document title",
        url="https://example.com/private?token=secret",
        source=source,
    )


class _FakeStore:
    def __init__(
        self,
        *,
        recent: list[tuple[Observation, str]] | None = None,
        search: list[tuple[Observation, str]] | None = None,
        stats: dict | None = None,
    ) -> None:
        self.recent = recent or []
        self.search = search or []
        self._stats = stats or {"observations": 0, "encryption_enabled": True}
        self.closed = False
        self.recent_args = None
        self.search_args = None

    def recent_capture_text(self, start_ts, end_ts, *, limit, max_chars):
        self.recent_args = (start_ts, end_ts, limit, max_chars)
        return self.recent

    def lexical_capture_text(self, query, *, limit, max_chars):
        self.search_args = (query, limit, max_chars)
        return self.search

    def stats(self):
        return self._stats

    def close(self):
        self.closed = True


def test_recent_capture_applies_all_exclusions_and_omits_ambient_metadata(tmp_path):
    rows = [
        (_observation("allowed"), "allowed excerpt"),
        (_observation("blocked-app", app="com.secret.App"), "app secret"),
        (_observation("blocked-source", source="meeting"), "source secret"),
        (_observation("blocked-id"), "id secret"),
        (_observation("unknown-app", app=None), "legacy secret"),
        (_observation("ingest", source="ingest"), "not capture"),
    ]
    store = _FakeStore(recent=rows)
    settings = Settings(
        data_dir=tmp_path,
        deep_brain_excluded_apps=["com.secret.App"],
        deep_brain_excluded_sources=["meeting"],
        deep_brain_excluded_observation_ids=["blocked-id"],
    )
    service = AssistantCaptureService(
        settings=settings, store_factory=lambda: store, clock=lambda: 1000.0
    )

    result = service.recent_capture(minutes=10, limit=10)

    assert [item["observation_id"] for item in result["results"]] == ["allowed"]
    assert result["excluded_observations"] == 3
    assert result["excluded_by"] == {
        "app": 1,
        "observation_id": 1,
        "unknown_app": 1,
    }
    assert "window" not in result["results"][0]
    assert "url" not in result["results"][0]
    assert "Sensitive document title" not in json.dumps(result)
    assert "token=secret" not in json.dumps(result)
    assert store.recent_args == (400.0, 1000.0, 50, 2000)
    assert store.closed is True


def test_capture_source_exclusion_blocks_all_content_tools(tmp_path):
    rows = [(_observation("allowed"), "must stay local")]
    recent_store = _FakeStore(recent=rows)
    search_store = _FakeStore(search=rows)
    stores = iter([recent_store, search_store])
    service = AssistantCaptureService(
        settings=Settings(
            data_dir=tmp_path,
            deep_brain_excluded_sources=["capture"],
        ),
        store_factory=lambda: next(stores),
        clock=lambda: 1000.0,
    )

    recent = service.recent_capture(minutes=10, limit=10)
    search = service.search_capture(query="local", limit=10)

    assert recent["results"] == []
    assert search["results"] == []
    assert recent["excluded_by"] == {"source": 1}
    assert search["excluded_by"] == {"source": 1}


def test_search_capture_enforces_excerpt_and_total_payload_caps(tmp_path):
    rows = [(_observation(str(i), ts=1000.0 - i), "x" * 4000) for i in range(20)]
    store = _FakeStore(search=rows)
    service = AssistantCaptureService(
        settings=Settings(data_dir=tmp_path), store_factory=lambda: store
    )

    result = service.search_capture(query="openbird", limit=20)

    assert sum(len(item["excerpt"]) for item in result["results"]) == 12_000
    assert all(len(item["excerpt"]) <= 2_000 for item in result["results"])
    assert len(result["results"]) == 6
    assert result["truncated"] is True
    assert store.search_args == ("openbird", 100, 2000)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"query": "", "limit": 1}, "query must not be blank"),
        ({"query": "x" * 501, "limit": 1}, "at most 500"),
        ({"query": "x", "limit": 0}, "between 1 and 20"),
    ],
)
def test_search_capture_rejects_invalid_bounds(tmp_path, kwargs, message):
    service = AssistantCaptureService(
        settings=Settings(data_dir=tmp_path), store_factory=_FakeStore
    )
    with pytest.raises(ValueError, match=message):
        service.search_capture(**kwargs)


def test_capture_status_is_metadata_only(tmp_path):
    store = _FakeStore(stats={"observations": 42, "encryption_enabled": True})
    service = AssistantCaptureService(
        settings=Settings(
            data_dir=tmp_path,
            deep_brain_excluded_apps=["com.secret.App"],
        ),
        store_factory=lambda: store,
    )

    result = service.capture_status()

    assert result == {
        "ok": True,
        "content_returned": False,
        "observations": 42,
        "encryption_enabled": True,
        "excluded_apps_configured": 1,
        "excluded_sources_configured": 0,
        "excluded_observations_configured": 0,
    }
    assert "excerpt" not in json.dumps(result)
    assert "com.secret.App" not in json.dumps(result)


def test_store_assistant_reads_never_call_model_methods(tmp_path):
    settings = Settings(data_dir=tmp_path, embed_dim=64)
    populated = MemoryStore(settings=settings, provider=FakeProvider(embed_dim=64))
    populated.add_observation(
        "OpenBird assistant connector work", app="com.example.Editor", source="capture", ts=50
    )
    populated.add_observation(
        "OpenBird imported notes", app="files", source="ingest", ts=60
    )
    populated.close()

    class NoModelProvider:
        embed_dim = 64

        def cohort_key(self):
            return "fake:fake-embed:64:deadbeef"

        def embed(self, _texts):
            raise AssertionError("assistant read attempted embedding")

        def complete(self, _messages, *, json_schema=None):
            raise AssertionError("assistant read attempted completion")

    class NoReranker:
        def rerank(self, _query, _docs):
            raise AssertionError("assistant read attempted reranking")

    store = MemoryStore(
        settings=settings, provider=NoModelProvider(), reranker=NoReranker()
    )
    try:
        recent = store.recent_capture_text(0, 100, limit=10, max_chars=2000)
        search = store.lexical_capture_text(
            "OpenBird assistant", limit=10, max_chars=2000
        )
    finally:
        store.close()

    assert [obs.source for obs, _ in recent] == ["capture"]
    assert [obs.source for obs, _ in search] == ["capture"]
    assert search[0][1] == "OpenBird assistant connector work"


def test_store_search_limits_after_deduplicating_repeated_capture(tmp_path):
    settings = Settings(data_dir=tmp_path, embed_dim=64)
    store = MemoryStore(settings=settings, provider=FakeProvider(embed_dim=64))
    repeated = "OpenBird assistant repeated capture"
    alternate = "OpenBird assistant alternate capture"
    try:
        for index in range(30):
            store.add_observation(
                repeated,
                app="com.example.Editor",
                source="capture",
                ts=float(index),
            )
        store.add_observation(
            alternate,
            app="com.example.Terminal",
            source="capture",
            ts=100.0,
        )

        results = store.lexical_capture_text(
            "OpenBird assistant", limit=2, max_chars=2000
        )
    finally:
        store.close()

    assert {text for _, text in results} == {repeated, alternate}
    assert next(obs.ts for obs, text in results if text == repeated) == 29.0


def test_mcp_server_exposes_only_bounded_read_tools(tmp_path):
    service = AssistantCaptureService(
        settings=Settings(data_dir=tmp_path), store_factory=_FakeStore
    )
    tools = asyncio.run(assistant.create_mcp_server(service).list_tools())
    dumped = {tool.name: tool.model_dump() for tool in tools}

    assert set(dumped) == {
        "openbird_recent_capture",
        "openbird_search_capture",
        "openbird_capture_status",
    }
    assert dumped["openbird_search_capture"]["inputSchema"]["required"] == ["query"]
    assert "untrusted" in dumped["openbird_search_capture"]["description"]
    assert "no capture text" in dumped["openbird_capture_status"]["description"]


def test_install_claude_config_preserves_other_content_and_writes_private_files(tmp_path):
    config_path = tmp_path / "Claude" / "claude_desktop_config.json"
    config_path.parent.mkdir()
    original = {
        "preferences": {"theme": "dark"},
        "mcpServers": {"other": {"command": "/usr/bin/other", "args": []}},
    }
    config_path.write_text(json.dumps(original), encoding="utf-8")
    config_path.chmod(0o644)
    executable = tmp_path / "openbird"
    executable.write_text("binary", encoding="utf-8")
    executable.chmod(0o700)

    result = assistant.install_claude_config(
        config_path=config_path, executable=executable
    )

    merged = json.loads(config_path.read_text(encoding="utf-8"))
    backup = Path(result["backup_path"])
    assert merged["preferences"] == original["preferences"]
    assert merged["mcpServers"]["other"] == original["mcpServers"]["other"]
    assert merged["mcpServers"]["openbird"] == {
        "command": str(executable.resolve()),
        "args": ["assistant", "serve"],
    }
    assert json.loads(backup.read_text(encoding="utf-8")) == original
    assert stat.S_IMODE(config_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(backup.stat().st_mode) == 0o600
    assert assistant.claude_config_status(config_path=config_path)["configured"] is True


def test_claude_config_status_rejects_missing_or_nonexecutable_command(tmp_path):
    config_path = tmp_path / "claude_desktop_config.json"
    executable = tmp_path / "openbird"
    executable.write_text("binary", encoding="utf-8")
    config_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "openbird": {
                        "command": str(executable),
                        "args": ["assistant", "serve"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    assert assistant.claude_config_status(config_path=config_path)["configured"] is False

    executable.chmod(0o700)
    assert assistant.claude_config_status(config_path=config_path)["configured"] is True

    executable.unlink()
    assert assistant.claude_config_status(config_path=config_path)["configured"] is False


def test_install_claude_config_refuses_malformed_input_without_mutation(tmp_path):
    config_path = tmp_path / "claude_desktop_config.json"
    config_path.write_text("{broken", encoding="utf-8")
    before = config_path.read_bytes()

    with pytest.raises(ValueError, match="invalid JSON"):
        assistant.install_claude_config(
            config_path=config_path, executable=tmp_path / "openbird"
        )

    assert config_path.read_bytes() == before
    assert not config_path.with_name(
        f"{config_path.name}.openbird-backup"
    ).exists()


def test_install_claude_config_refuses_concurrent_config_change(
    tmp_path, monkeypatch
):
    config_path = tmp_path / "claude_desktop_config.json"
    original = {"preferences": {"theme": "dark"}}
    concurrent = {"preferences": {"theme": "light"}}
    config_path.write_text(json.dumps(original), encoding="utf-8")
    real_atomic_write = assistant._atomic_private_write

    def change_before_replace(path, payload, *, expected_snapshot=None):
        if expected_snapshot is not None:
            config_path.write_text(json.dumps(concurrent), encoding="utf-8")
        return real_atomic_write(
            path, payload, expected_snapshot=expected_snapshot
        )

    monkeypatch.setattr(assistant, "_atomic_private_write", change_before_replace)

    with pytest.raises(assistant.ClaudeConfigConflictError, match="retry"):
        assistant.install_claude_config(
            config_path=config_path, executable=tmp_path / "openbird"
        )

    assert json.loads(config_path.read_text(encoding="utf-8")) == concurrent
    backup = config_path.with_name(f"{config_path.name}.openbird-backup")
    assert json.loads(backup.read_text(encoding="utf-8")) == original


def test_install_claude_config_creates_private_config_when_absent(tmp_path):
    config_path = tmp_path / "new" / "claude_desktop_config.json"
    executable = tmp_path / "openbird"
    executable.write_text("binary", encoding="utf-8")

    result = assistant.install_claude_config(
        config_path=config_path, executable=executable
    )

    assert result["backup_path"] is None
    assert stat.S_IMODE(config_path.stat().st_mode) == 0o600
    assert json.loads(config_path.read_text())["mcpServers"]["openbird"]["args"] == [
        "assistant",
        "serve",
    ]


def test_cli_install_claude_refuses_noninteractive_without_yes(monkeypatch):
    monkeypatch.setattr(
        assistant,
        "install_claude_config",
        lambda: pytest.fail("installer must not run without consent"),
    )

    result = CliRunner().invoke(cli.app, ["assistant", "install-claude"])

    assert result.exit_code == 1
    assert "ASSISTANT ACCESS" in result.output
    assert "Re-run with --yes" in result.output
