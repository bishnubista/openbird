"""Read-only desktop-assistant connector tests."""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
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
        spans: list[dict] | None = None,
        stats: dict | None = None,
    ) -> None:
        self.recent = recent or []
        self.search = search or []
        self.spans = spans or []
        self._stats = stats or {"observations": 0, "encryption_enabled": True}
        self.closed = False
        self.recent_args = None
        self.search_args = None
        self.spans_args = None

    def recent_capture_text(self, start_ts, end_ts, *, limit, max_chars, before=None):
        self.recent_args = (start_ts, end_ts, limit, max_chars, before)
        return self.recent

    def lexical_capture_text(self, query, *, limit, max_chars):
        self.search_args = (query, limit, max_chars)
        return self.search

    def capture_spans_overlapping(self, start_ts, end_ts, *, limit):
        self.spans_args = (start_ts, end_ts, limit)
        return self.spans[:limit]

    def stats(self):
        return self._stats

    def close(self):
        self.closed = True


def _span(
    span_id: str,
    *,
    start_ts: float,
    end_ts: float,
    bundle_id: str | None = "com.example.Editor",
    detail_tier: int = 1,
    afk: int = 0,
    meeting: int = 0,
    reason: str | None = None,
) -> dict:
    return {
        "span_id": span_id,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "bundle_id": bundle_id,
        "detail_tier": detail_tier,
        "afk": afk,
        "meeting": meeting,
        "reason": reason,
    }


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
    assert store.recent_args == (400.0, 1000.0, 201, 2000, None)
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
            assistant_host_label="test-host",
        ),
        store_factory=lambda: store,
    )

    result = service.capture_status()

    assert result == {
        "ok": True,
        "content_returned": False,
        "egress_notice": assistant.ASSISTANT_STATUS_EGRESS_NOTICE,
        "egress": {
            "scope": "status_metadata",
            "untrusted_content": False,
            "fields": sorted(assistant.STATUS_EGRESS_FIELDS),
        },
        "capture_host": "test-host",
        "observations_total": 42,
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
        "openbird_activity_summary",
        "openbird_capture_status",
    }
    assert dumped["openbird_search_capture"]["inputSchema"]["required"] == ["query"]
    assert "untrusted" in dumped["openbird_search_capture"]["description"]
    assert "no capture text" in dumped["openbird_capture_status"]["description"]
    assert "no capture text" in dumped["openbird_activity_summary"]["description"]
    assert "next_cursor" in dumped["openbird_recent_capture"]["description"]


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


def test_cli_install_claude_reports_config_conflict_without_traceback(monkeypatch):
    monkeypatch.setattr(
        assistant,
        "install_claude_config",
        lambda **_kwargs: (_ for _ in ()).throw(
            assistant.ClaudeConfigConflictError(
                "Claude Desktop config changed during installation; retry"
            )
        ),
    )

    result = CliRunner().invoke(
        cli.app, ["assistant", "install-claude", "--yes"]
    )

    assert result.exit_code == 1
    assert "Could not configure Claude Desktop" in result.output
    assert "retry" in result.output
    assert result.exception is not None


def test_configure_chatgpt_reconciles_profile_without_secret_in_argv(tmp_path):
    helper = tmp_path / "tunnel-client"
    helper.write_text("binary", encoding="utf-8")
    helper.chmod(0o700)
    executable = tmp_path / "bundle-b" / "openbird-cli"
    executable.parent.mkdir()
    executable.write_text("binary", encoding="utf-8")
    executable.chmod(0o700)
    calls = []
    profile_dir = tmp_path / "profiles"

    def runner(arguments, **kwargs):
        calls.append((arguments, kwargs))
        if arguments[1] == "init":
            (profile_dir / "openbird.yaml").write_text("profile", encoding="utf-8")
        return subprocess.CompletedProcess(arguments, 0)

    secret = "runtime-test-secret-never-in-argv"
    result = assistant.configure_chatgpt(
        "tunnel_abcdef123",
        executable=executable,
        tunnel_client=helper,
        profile_dir=profile_dir,
        environment={"CONTROL_PLANE_API_KEY": secret},
        runner=runner,
    )

    assert result == {"configured": True, "helper_available": True}
    init_args, init_kwargs = calls[0]
    assert init_args[1:3] == ["init", "--force"]
    assert "env:CONTROL_PLANE_API_KEY" in init_args
    assert f"{executable.resolve()} assistant serve" in init_args
    assert secret not in " ".join(init_args)
    assert init_kwargs["env"]["CONTROL_PLANE_API_KEY"] == secret
    assert calls[1][0][1:4] == ["doctor", "--profile", "openbird"]


def test_configure_chatgpt_reports_missing_profile_after_zero_exit(tmp_path):
    helper = tmp_path / "tunnel-client"
    helper.write_text("binary", encoding="utf-8")
    helper.chmod(0o700)
    executable = tmp_path / "openbird-cli"
    executable.write_text("binary", encoding="utf-8")
    executable.chmod(0o700)

    result = assistant.configure_chatgpt(
        "tunnel_abcdef123",
        executable=executable,
        tunnel_client=helper,
        profile_dir=tmp_path / "profiles",
        environment={"CONTROL_PLANE_API_KEY": "secret"},
        runner=lambda arguments, **_kwargs: subprocess.CompletedProcess(arguments, 0),
    )

    assert result == {"configured": False, "helper_available": True}


def test_configure_chatgpt_cli_rejects_unverified_profile(monkeypatch):
    monkeypatch.setattr(
        assistant,
        "configure_chatgpt",
        lambda *_args, **_kwargs: {"configured": False, "helper_available": True},
    )

    result = CliRunner().invoke(
        cli.app,
        [
            "assistant",
            "configure-chatgpt",
            "--yes",
            "--tunnel-id",
            "tunnel_abcdef123",
        ],
    )

    assert result.exit_code == 1
    assert "did not pass verification" in result.output


def test_configure_chatgpt_rebinds_existing_profile_to_relocated_bundle(tmp_path):
    helper = tmp_path / "tunnel-client"
    helper.write_text("binary", encoding="utf-8")
    helper.chmod(0o700)
    bundle_a = tmp_path / "Bundle A" / "openbird-cli"
    bundle_b = tmp_path / "Bundle B" / "openbird-cli"
    for executable in (bundle_a, bundle_b):
        executable.parent.mkdir()
        executable.write_text("binary", encoding="utf-8")
        executable.chmod(0o700)
    calls = []

    def runner(arguments, **_kwargs):
        calls.append(arguments)
        return subprocess.CompletedProcess(arguments, 0)

    for executable in (bundle_a, bundle_b):
        assistant.configure_chatgpt(
            "tunnel_abcdef123",
            executable=executable,
            tunnel_client=helper,
            profile_dir=tmp_path / "profiles",
            environment={"CONTROL_PLANE_API_KEY": "secret"},
            runner=runner,
        )

    init_a, init_b = calls[0], calls[2]
    assert init_a[1:3] == init_b[1:3] == ["init", "--force"]
    assert str(bundle_a.resolve()) in init_a[init_a.index("--mcp-command") + 1]
    assert str(bundle_b.resolve()) in init_b[init_b.index("--mcp-command") + 1]
    assert str(bundle_a.resolve()) not in init_b[init_b.index("--mcp-command") + 1]


def test_configure_chatgpt_rejects_bad_id_and_missing_key(tmp_path):
    with pytest.raises(ValueError, match="tunnel id"):
        assistant.configure_chatgpt("not-a-tunnel", environment={})
    with pytest.raises(ValueError, match="runtime key"):
        assistant.configure_chatgpt("tunnel_abcdef", environment={})


def test_chatgpt_run_arguments_are_privacy_hardened(tmp_path):
    helper = tmp_path / "tunnel-client"
    helper.write_text("binary", encoding="utf-8")
    helper.chmod(0o700)
    health = tmp_path / "runtime" / "health.url"

    arguments = assistant.chatgpt_run_arguments(
        tunnel_client=helper,
        profile_dir=tmp_path / "profiles",
        health_url_file=health,
    )

    joined = " ".join(arguments)
    assert "--health.listen-addr 127.0.0.1:0" in joined
    # v0.0.10 rejects zero with "log-buffer-events must be greater than zero".
    assert "--admin-ui.log-buffer-events 1" in joined
    assert "--log.file /dev/null" in joined
    assert "--open-web-ui" not in arguments
    assert "--allow-remote-ui" not in arguments
    assert stat.S_IMODE(health.stat().st_mode) == 0o600


def test_run_chatgpt_execs_owned_tunnel_without_intermediate_child(tmp_path, monkeypatch):
    helper = tmp_path / "tunnel-client"
    helper.write_text("binary", encoding="utf-8")
    helper.chmod(0o700)
    calls = []

    def fake_execve(path, arguments, environment):
        calls.append((path, arguments, environment))
        raise OSError("exec stopped by test")

    monkeypatch.setattr(assistant.os, "execve", fake_execve)
    with pytest.raises(OSError, match="stopped by test"):
        assistant.run_chatgpt_tunnel(
            tunnel_client=helper,
            profile_dir=tmp_path / "profiles",
            health_url_file=tmp_path / "health.url",
            environment={"CONTROL_PLANE_API_KEY": "secret"},
        )

    assert calls[0][0] == str(helper.resolve())
    assert calls[0][1][1] == "run"
    assert "secret" not in " ".join(calls[0][1])


def test_remove_chatgpt_deletes_only_owned_profile(tmp_path):
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    owned = profile_dir / "openbird.yaml"
    other = profile_dir / "other.yaml"
    owned.write_text("owned", encoding="utf-8")
    other.write_text("other", encoding="utf-8")

    assert assistant.remove_chatgpt_config(profile_dir=profile_dir)
    assert not owned.exists()
    assert other.read_text(encoding="utf-8") == "other"


def test_cli_remove_chatgpt_handles_filesystem_error(monkeypatch):
    monkeypatch.setattr(
        assistant,
        "remove_chatgpt_config",
        lambda: (_ for _ in ()).throw(PermissionError("profile is not writable")),
    )

    result = CliRunner().invoke(cli.app, ["assistant", "remove-chatgpt", "--yes"])

    assert result.exit_code == 1
    assert "Could not remove OpenBird's ChatGPT profile" in result.output
    assert "profile is not writable" in result.output


def test_tunnel_client_pins_match_homebrew_formula():
    root = Path(__file__).parents[2]
    script = (root / "script" / "stage_tunnel_client.sh").read_text()
    formula = (root / "Formula" / "openbird.rb").read_text()
    version = re.search(r'^VERSION="([^"]+)"$', script, re.MULTILINE)
    checksums = re.findall(r'^\s*SHA256="([0-9a-f]{64})"$', script, re.MULTILINE)

    assert version is not None
    assert len(checksums) == 2
    assert formula.count(f"tunnel-client-v{version.group(1)}-") == 2
    for checksum in checksums:
        assert formula.count(f'sha256 "{checksum}"') == 1
    assert script.count('"$ROOT_DIR/script/smoke_tunnel_client.sh" "$DEST"') == 2


# -- v2: cursor pagination, dedup groups, activity summary ---------------------


def _capture_obs(id_, *, ts, hash_=None, app="com.example.Editor"):
    return Observation(
        id=id_,
        content_hash=hash_ or f"hash-{id_}",
        ts=ts,
        app=app,
        source="capture",
    )


def test_recent_capture_groups_repeated_blobs_into_one_result(tmp_path):
    rows = [
        (_capture_obs(f"dup-{i}", ts=1000.0 - i, hash_="hash-shared"), "PR #264 body")
        for i in range(50)
    ] + [(_capture_obs("other", ts=100.0), "unrelated excerpt")]
    store = _FakeStore(recent=rows)
    service = AssistantCaptureService(
        settings=Settings(data_dir=tmp_path),
        store_factory=lambda: store,
        clock=lambda: 1000.0,
    )

    result = service.recent_capture(minutes=60, limit=10)

    shared = result["results"][0]
    assert shared["seen_count"] == 50
    assert shared["observation_id"] == "dup-0"  # newest occurrence is the anchor
    assert shared["timestamp"] == 1000.0
    assert shared["first_ts"] == 951.0
    assert shared["last_ts"] == 1000.0
    assert [item["excerpt"] for item in result["results"]] == [
        "PR #264 body",
        "unrelated excerpt",
    ]
    assert result["result_count"] == 2
    assert result["truncated"] is False
    assert result["next_cursor"] is None


def test_recent_capture_cursor_walks_window_without_losing_groups(tmp_path):
    settings = Settings(data_dir=tmp_path, embed_dim=64)
    store = MemoryStore(settings=settings, provider=FakeProvider(embed_dim=64))
    try:
        for i in range(30):
            store.add_observation(
                f"distinct excerpt {i}",
                app="com.example.Editor",
                source="capture",
                ts=1000.0 + i,
            )
    finally:
        store.close()

    def factory():
        return MemoryStore(settings=settings, provider=FakeProvider(embed_dim=64))

    service = AssistantCaptureService(
        settings=settings, store_factory=factory, clock=lambda: 1030.0
    )

    seen: list[str] = []
    cursor = None
    pages = 0
    while True:
        result = service.recent_capture(minutes=60, limit=12, cursor=cursor)
        seen.extend(item["excerpt"] for item in result["results"])
        pages += 1
        cursor = result["next_cursor"]
        if cursor is None:
            break
    assert pages == 3
    assert seen == [f"distinct excerpt {i}" for i in range(29, -1, -1)]
    assert len(set(seen)) == 30  # every group emitted exactly once


def test_recent_capture_cursor_window_is_frozen_against_new_inserts(tmp_path):
    settings = Settings(data_dir=tmp_path, embed_dim=64)
    store = MemoryStore(settings=settings, provider=FakeProvider(embed_dim=64))
    try:
        for i in range(25):
            store.add_observation(
                f"stable excerpt {i}",
                app="com.example.Editor",
                source="capture",
                ts=1000.0 + i,
            )
    finally:
        store.close()

    def factory():
        return MemoryStore(settings=settings, provider=FakeProvider(embed_dim=64))

    now = [1025.0]
    service = AssistantCaptureService(
        settings=settings, store_factory=factory, clock=lambda: now[0]
    )
    first = service.recent_capture(minutes=60, limit=20)
    assert first["next_cursor"] is not None

    concurrent = MemoryStore(settings=settings, provider=FakeProvider(embed_dim=64))
    try:
        concurrent.add_observation(
            "newer than the frozen window",
            app="com.example.Editor",
            source="capture",
            ts=1050.0,
        )
    finally:
        concurrent.close()

    # The wall clock advances between pages (within the cursor TTL); a
    # re-derived `[now - minutes, now]` window would now include ts=1050.0.
    # The frozen window ends at 1025.0 and must not.
    now[0] = 1100.0
    second = service.recent_capture(minutes=60, cursor=first["next_cursor"])
    texts = [item["excerpt"] for item in second["results"]]
    assert "newer than the frozen window" not in texts
    assert second["window_end_ts"] == first["window_end_ts"]
    assert second["next_cursor"] is None
    assert len(first["results"]) + len(texts) == 25


def test_recent_capture_rejects_bad_unknown_and_expired_cursors(tmp_path):
    now = [1000.0]
    service = AssistantCaptureService(
        settings=Settings(data_dir=tmp_path),
        store_factory=_FakeStore,
        clock=lambda: now[0],
    )
    with pytest.raises(ValueError, match="not a valid page token"):
        service.recent_capture(cursor="")
    with pytest.raises(ValueError, match="not a valid page token"):
        service.recent_capture(cursor="x" * 200)
    with pytest.raises(ValueError, match="unknown or expired"):
        service.recent_capture(cursor="forged-or-guessed-token")

    rows = [
        (_capture_obs(f"row-{i}", ts=1000.0 - i), f"text {i}") for i in range(10)
    ]
    paged = AssistantCaptureService(
        settings=Settings(data_dir=tmp_path),
        store_factory=lambda: _FakeStore(recent=rows),
        clock=lambda: now[0],
    )
    cursor = paged.recent_capture(minutes=60, limit=3)["next_cursor"]
    assert cursor is not None
    now[0] += 15 * 60 + 1
    with pytest.raises(ValueError, match="unknown or expired"):
        paged.recent_capture(cursor=cursor)


def test_recent_capture_all_excluded_page_advances_without_leaking(tmp_path):
    rows = [
        (_capture_obs(f"secret-{i}", ts=1000.0 - i, app="com.secret.App"), "hidden")
        for i in range(5)
    ]
    store = _FakeStore(recent=rows)
    service = AssistantCaptureService(
        settings=Settings(
            data_dir=tmp_path, deep_brain_excluded_apps=["com.secret.App"]
        ),
        store_factory=lambda: store,
        clock=lambda: 1000.0,
    )

    result = service.recent_capture(minutes=60, limit=5)

    assert result["results"] == []
    assert result["excluded_observations"] == 5
    assert result["next_cursor"] is None  # scan consumed the whole window
    dumped = json.dumps(result)
    assert "secret-" not in dumped  # no excluded id leaks anywhere, cursor included
    assert "hidden" not in dumped


def test_recent_capture_limit_stop_resumes_at_unconsumed_row(tmp_path):
    rows = [
        (_capture_obs(f"row-{i}", ts=1000.0 - i), f"text {i}") for i in range(6)
    ]
    store = _FakeStore(recent=rows)
    service = AssistantCaptureService(
        settings=Settings(data_dir=tmp_path),
        store_factory=lambda: store,
        clock=lambda: 1000.0,
    )

    first = service.recent_capture(minutes=60, limit=4)
    assert [item["excerpt"] for item in first["results"]] == [
        "text 0", "text 1", "text 2", "text 3",
    ]
    assert first["truncated"] is True
    assert first["next_cursor"] is not None
    # The page-2 keyset boundary is the last CONSUMED row (row-3), so the
    # stopping row (row-4) must be the first row requested next.
    service.recent_capture(cursor=first["next_cursor"])
    assert store.recent_args[4] == (997.0, "row-3")


def test_activity_summary_buckets_follow_strict_precedence(tmp_path):
    spans = [
        _span("visible", start_ts=0.0, end_ts=100.0),
        _span("afk", start_ts=100.0, end_ts=160.0, afk=1),
        _span("tier0", start_ts=160.0, end_ts=200.0, detail_tier=0),
        _span("tier0-afk", start_ts=200.0, end_ts=230.0, detail_tier=0, afk=1),
        _span("excl", start_ts=230.0, end_ts=280.0, bundle_id="com.secret.App"),
        _span(
            "excl-tier0-afk", start_ts=280.0, end_ts=300.0,
            bundle_id="com.secret.App", detail_tier=0, afk=1,
        ),
        _span("paused", start_ts=300.0, end_ts=320.0, bundle_id=None, detail_tier=0),
        _span("zero", start_ts=320.0, end_ts=320.0),
    ]
    service = AssistantCaptureService(
        settings=Settings(
            data_dir=tmp_path, deep_brain_excluded_apps=["com.secret.App"]
        ),
        store_factory=lambda: _FakeStore(spans=spans),
        clock=lambda: 400.0,
    )

    result = service.activity_summary(minutes=10)

    assert result["content_returned"] is False
    assert result["foreground_seconds"] == 100.0
    assert result["afk_seconds"] == 60.0
    assert result["redacted_seconds"] == 90.0   # tier0 + tier0-afk + paused
    assert result["excluded_seconds"] == 70.0   # both excluded spans, afk/tier0 or not
    assert result["apps"] == [
        {
            "bundle_id": "com.example.Editor",
            "foreground_seconds": 100.0,
            "span_count": 1,
            "meeting_seconds": 0.0,
        }
    ]
    dumped = json.dumps(result)
    assert "com.secret.App" not in dumped
    # Partition identity: visible + afk + redacted + excluded == clipped time.
    total = (
        result["foreground_seconds"] + result["afk_seconds"]
        + result["redacted_seconds"] + result["excluded_seconds"]
    )
    assert total == 320.0


def test_activity_summary_meeting_overlay_survives_afk(tmp_path):
    spans = [
        _span(
            "zoom", start_ts=0.0, end_ts=3600.0,
            bundle_id="us.zoom.xos", afk=1, meeting=1,
        ),
        _span("editor", start_ts=3600.0, end_ts=3660.0),
        _span(
            "hidden-meeting", start_ts=3660.0, end_ts=3700.0,
            detail_tier=0, meeting=1,
        ),
    ]
    service = AssistantCaptureService(
        settings=Settings(data_dir=tmp_path),
        store_factory=lambda: _FakeStore(spans=spans),
        clock=lambda: 3700.0,
    )

    result = service.activity_summary(minutes=100)

    zoom = next(a for a in result["apps"] if a["bundle_id"] == "us.zoom.xos")
    assert zoom["meeting_seconds"] == 3600.0
    assert zoom["foreground_seconds"] == 0.0  # fully afk, still surfaced
    assert result["meeting_seconds"] == 3600.0  # hidden tier-0 meeting moves nothing
    assert result["afk_seconds"] == 3600.0
    assert result["meeting_seconds"] == sum(
        a["meeting_seconds"] for a in result["apps"]
    )


def test_activity_summary_switches_and_focus_ignore_hidden_spans(tmp_path):
    spans = [
        _span("a1", start_ts=0.0, end_ts=100.0, bundle_id="com.a"),
        _span("hidden", start_ts=100.0, end_ts=150.0, detail_tier=0),
        _span("a2", start_ts=150.0, end_ts=260.0, bundle_id="com.a"),
        _span("b", start_ts=260.0, end_ts=300.0, bundle_id="com.b"),
        _span("afk", start_ts=300.0, end_ts=330.0, bundle_id="com.b", afk=1),
        _span("a3", start_ts=330.0, end_ts=350.0, bundle_id="com.a"),
    ]
    service = AssistantCaptureService(
        settings=Settings(data_dir=tmp_path),
        store_factory=lambda: _FakeStore(spans=spans),
        clock=lambda: 350.0,
    )

    result = service.activity_summary(minutes=10)

    # Visible non-afk sequence: a1, a2, b, a3 -> a->b, b->a = 2 switches (the
    # hidden tier-0 span between a1 and a2 does NOT split the run).
    assert result["context_switches"] == 2
    assert result["longest_focus"] == {
        "bundle_id": "com.a",
        "start_ts": 0.0,
        "end_ts": 260.0,
        "seconds": 210.0,
    }


def test_activity_summary_caps_apps_and_folds_tail(tmp_path):
    spans = [
        _span(f"s{i}", start_ts=i * 10.0, end_ts=i * 10.0 + 10.0 - i * 0.1,
              bundle_id=f"com.app.{i:02d}")
        for i in range(35)
    ]
    service = AssistantCaptureService(
        settings=Settings(data_dir=tmp_path),
        store_factory=lambda: _FakeStore(spans=spans),
        clock=lambda: 400.0,
    )

    result = service.activity_summary(minutes=10)

    assert len(result["apps"]) == 30
    assert result["other_apps_count"] == 5
    assert result["other_apps_seconds"] > 0
    named = sum(a["foreground_seconds"] for a in result["apps"])
    assert named + result["other_apps_seconds"] == pytest.approx(
        result["foreground_seconds"]
    )


def test_activity_summary_fails_closed_on_span_overflow(tmp_path):
    spans = [
        _span(f"s{i}", start_ts=float(i), end_ts=float(i) + 0.5)
        for i in range(assistant.MAX_SUMMARY_SPANS + 1)
    ]
    service = AssistantCaptureService(
        settings=Settings(data_dir=tmp_path),
        store_factory=lambda: _FakeStore(spans=spans),
        clock=lambda: float(assistant.MAX_SUMMARY_SPANS + 2),
    )
    with pytest.raises(ValueError, match="narrower window"):
        service.activity_summary(minutes=1440)


def test_malformed_regex_exclusion_fails_every_tool_closed(tmp_path):
    settings = Settings(
        data_dir=tmp_path, deep_brain_excluded_apps=["re:com.[unclosed"]
    )
    rows = [(_capture_obs("row", ts=1000.0), "text")]
    service = AssistantCaptureService(
        settings=settings,
        store_factory=lambda: _FakeStore(
            recent=rows, search=rows, spans=[_span("s", start_ts=0.0, end_ts=10.0)]
        ),
        clock=lambda: 1000.0,
    )
    for call in (
        lambda: service.recent_capture(minutes=10, limit=5),
        lambda: service.search_capture(query="text", limit=5),
        lambda: service.activity_summary(minutes=10),
    ):
        with pytest.raises(ValueError, match="deep_brain_excluded_apps"):
            call()


def test_store_recent_capture_text_keyset_before_boundary(tmp_path):
    settings = Settings(data_dir=tmp_path, embed_dim=64)
    store = MemoryStore(settings=settings, provider=FakeProvider(embed_dim=64))
    try:
        ids = [
            store.add_observation(
                f"tie {i}", app="com.example.Editor", source="capture", ts=500.0
            ).id
            for i in range(5)
        ]
        rows = store.recent_capture_text(0, 1000, limit=10, max_chars=100)
        assert len(rows) == 5
        boundary = rows[2][0]
        older = store.recent_capture_text(
            0, 1000, limit=10, max_chars=100, before=(boundary.ts, boundary.id)
        )
        assert [obs.id for obs, _ in older] == [obs.id for obs, _ in rows[3:]]
        assert set(ids) == {obs.id for obs, _ in rows}
    finally:
        store.close()


def test_store_capture_spans_overlapping_is_projected_and_bounded(tmp_path):
    settings = Settings(data_dir=tmp_path, embed_dim=64)
    store = MemoryStore(settings=settings, provider=FakeProvider(embed_dim=64))
    try:
        for i in range(4):
            store.open_span(
                epoch_id="epoch",
                start_ts=float(i * 10),
                end_ts=float(i * 10 + 5),
                bundle_id="com.example.Editor",
                detail_tier=1,
                window="Sensitive title",
                url_host="example.com",
            )
        store.open_span(
            epoch_id="epoch",
            start_ts=50.0,
            end_ts=60.0,
            bundle_id="com.blocked.App",
            detail_tier=0,
            reason="blocklisted",
        )
        spans = store.capture_spans_overlapping(0, 100, limit=3)
        assert len(spans) == 3
        assert set(spans[0]) == {
            "span_id", "start_ts", "end_ts", "bundle_id",
            "detail_tier", "afk", "meeting", "reason",
        }
        assert "Sensitive title" not in json.dumps(spans)
        # The tier-0 reason round-trips as the stored enum value, while the
        # forbidden columns stay out of the projection entirely.
        coarse = store.capture_spans_overlapping(50, 60, limit=10)
        tier0 = next(s for s in coarse if s["detail_tier"] == 0)
        assert tier0["bundle_id"] == "com.blocked.App"
        assert tier0["reason"] == "blocklisted"
        for forbidden in ("window", "url_host", "identity_key"):
            assert forbidden not in tier0
    finally:
        store.close()


def test_schema_v8_keyset_index_exists_on_fresh_and_migrated_dbs(tmp_path):
    from openbird.memory.migrations import SCHEMA_VERSION

    settings = Settings(data_dir=tmp_path, embed_dim=64)
    store = MemoryStore(settings=settings, provider=FakeProvider(embed_dim=64))

    def index_names(s):
        rows = s.conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index'"
        ).fetchall()
        return {next(iter(r.values())) if isinstance(r, dict) else r[0] for r in rows}

    try:
        assert "idx_observations_source_ts_id" in index_names(store)
        store.conn.execute("DROP INDEX idx_observations_source_ts_id")
        store.conn.execute("PRAGMA user_version = 7")
        store.conn.commit()
    finally:
        store.close()

    reopened = MemoryStore(settings=settings, provider=FakeProvider(embed_dim=64))
    try:
        assert "idx_observations_source_ts_id" in index_names(reopened)
        version = reopened.conn.execute("PRAGMA user_version").fetchone()
        value = next(iter(version.values())) if isinstance(version, dict) else version[0]
        assert int(value) == SCHEMA_VERSION >= 8
    finally:
        reopened.close()


def test_cursor_handles_are_single_use(tmp_path):
    rows = [
        (_capture_obs(f"row-{i}", ts=1000.0 - i), f"text {i}") for i in range(10)
    ]
    service = AssistantCaptureService(
        settings=Settings(data_dir=tmp_path),
        store_factory=lambda: _FakeStore(recent=rows),
        clock=lambda: 1000.0,
    )
    cursor = service.recent_capture(minutes=60, limit=3)["next_cursor"]
    assert cursor is not None

    service.recent_capture(cursor=cursor)
    with pytest.raises(ValueError, match="unknown or expired"):
        service.recent_capture(cursor=cursor)  # replay of a consumed handle


def test_recent_capture_scan_cap_page_of_duplicates_reports_truncated(tmp_path):
    scan_cap = assistant.SCAN_CAP
    rows = [
        (
            _capture_obs(f"dup-{i}", ts=500.0, hash_="hash-shared"),
            "same text every time",
        )
        for i in range(scan_cap + 1)
    ]
    service = AssistantCaptureService(
        settings=Settings(data_dir=tmp_path),
        store_factory=lambda: _FakeStore(recent=rows),
        clock=lambda: 1000.0,
    )

    result = service.recent_capture(minutes=60, limit=20)

    assert result["result_count"] == 1
    assert result["results"][0]["seen_count"] == scan_cap
    assert result["next_cursor"] is not None
    assert result["truncated"] is True  # the scan cap, not the group cap, hit


def test_cli_assistant_warnings_disclose_activity_egress(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)

    install = runner.invoke(cli.app, ["assistant", "install-claude"])
    chatgpt = runner.invoke(
        cli.app, ["assistant", "configure-chatgpt", "--tunnel-id", "tunnel_x"]
    )

    for result in (install, chatgpt):
        assert result.exit_code == 1
        # Rich wraps at the terminal width; normalize before phrase checks.
        flat = " ".join(result.output.split())
        assert "ASSISTANT ACCESS" in flat
        assert "ACTIVITY ACCESS" in flat
        assert "STATUS ACCESS" in flat
        assert "activity patterns" in flat
        # The consent copy must name every new egress category.
        assert "redacted" in flat
        assert "reason codes" in flat
        assert "host label" in flat
        assert "exclusion-configuration counts" in flat
        assert "timezone" in flat


def test_activity_summary_afk_gap_splits_focus_but_hidden_gap_does_not(tmp_path):
    spans = [
        _span("a1", start_ts=0.0, end_ts=100.0, bundle_id="com.a"),
        _span("nap", start_ts=100.0, end_ts=1900.0, bundle_id="com.a", afk=1),
        _span("a2", start_ts=1900.0, end_ts=1950.0, bundle_id="com.a"),
        _span("hidden", start_ts=1950.0, end_ts=2000.0, detail_tier=0),
        _span("a3", start_ts=2000.0, end_ts=2030.0, bundle_id="com.a"),
    ]
    service = AssistantCaptureService(
        settings=Settings(data_dir=tmp_path),
        store_factory=lambda: _FakeStore(spans=spans),
        clock=lambda: 2030.0,
    )

    result = service.activity_summary(minutes=60)

    # The AFK nap ends the first focus run (100s), so it cannot fuse with the
    # later spans into a 180s block. The hidden tier-0 span does NOT split the
    # second run (a2+a3 merge to 80s), and same-app resumption after AFK
    # counts no context switch. Longest focus is therefore the 100s run.
    assert result["context_switches"] == 0
    assert result["longest_focus"] == {
        "bundle_id": "com.a",
        "start_ts": 0.0,
        "end_ts": 100.0,
        "seconds": 100.0,
    }
    assert result["afk_seconds"] == 1800.0


# -- v3: redaction attribution, capture_host, machine-parseable egress ----------


def test_activity_summary_attributes_redacted_time_with_clipping(tmp_path):
    spans = [
        # Crosses the window START: only [340, 360] counts.
        _span("pre", start_ts=300.0, end_ts=360.0, detail_tier=0, reason="blocklisted"),
        _span(
            "mid", start_ts=360.0, end_ts=370.0,
            bundle_id="com.browser", detail_tier=0, reason="private",
        ),
        # Crosses the window END: only [390, 400] counts.
        _span("post", start_ts=390.0, end_ts=450.0, detail_tier=0, reason="blocklisted"),
        # NULL bundle crossing the start edge: unattributable, clipped to 10s.
        _span(
            "gap", start_ts=330.0, end_ts=350.0,
            bundle_id=None, detail_tier=0, reason="paused",
        ),
        # Excluded app crossing the end edge: excluded wins, never named.
        _span(
            "excl", start_ts=395.0, end_ts=460.0,
            bundle_id="com.secret.App", detail_tier=0, reason="blocklisted",
        ),
    ]
    service = AssistantCaptureService(
        settings=Settings(
            data_dir=tmp_path, deep_brain_excluded_apps=["com.secret.App"]
        ),
        store_factory=lambda: _FakeStore(spans=spans),
        clock=lambda: 400.0,
    )

    result = service.activity_summary(minutes=1)

    assert result["redacted_seconds"] == 50.0
    assert result["redacted_by_app"] == [
        {"bundle_id": "com.example.Editor", "reason": "blocklisted", "seconds": 30.0},
        {"bundle_id": "com.browser", "reason": "private", "seconds": 10.0},
    ]
    assert result["redacted_unattributed_seconds"] == 10.0
    assert result["redacted_other_seconds"] == 0.0
    assert result["excluded_seconds"] == 5.0
    assert "com.secret.App" not in json.dumps(result)
    # Breakdown invariant under clipping: attributed + tail + unattributed
    # must reconstruct the total exactly, from the same clipped durations.
    assert (
        sum(entry["seconds"] for entry in result["redacted_by_app"])
        + result["redacted_other_seconds"]
        + result["redacted_unattributed_seconds"]
        == result["redacted_seconds"]
    )


def test_activity_summary_caps_redacted_apps_and_folds_tail(tmp_path):
    spans = [
        _span(
            f"r{i}", start_ts=i * 10.0, end_ts=i * 10.0 + 10.0 - i * 0.1,
            bundle_id=f"com.red.{i:02d}", detail_tier=0, reason="not_allowlisted",
        )
        for i in range(35)
    ]
    service = AssistantCaptureService(
        settings=Settings(data_dir=tmp_path),
        store_factory=lambda: _FakeStore(spans=spans),
        clock=lambda: 400.0,
    )

    result = service.activity_summary(minutes=10)

    assert len(result["redacted_by_app"]) == assistant.MAX_SUMMARY_APPS
    assert result["redacted_other_seconds"] > 0
    assert (
        sum(entry["seconds"] for entry in result["redacted_by_app"])
        + result["redacted_other_seconds"]
        + result["redacted_unattributed_seconds"]
        == pytest.approx(result["redacted_seconds"])
    )


def test_activity_summary_maps_corrupt_null_reason_to_unknown(tmp_path):
    # Tier-0 reason is schema-guaranteed; a NULL here means a corrupted row,
    # and the egress-only "unknown" sentinel must absorb it (never crash,
    # never invent an enum member).
    spans = [_span("bad", start_ts=0.0, end_ts=10.0, detail_tier=0, reason=None)]
    service = AssistantCaptureService(
        settings=Settings(data_dir=tmp_path),
        store_factory=lambda: _FakeStore(spans=spans),
        clock=lambda: 400.0,
    )

    result = service.activity_summary(minutes=10)

    assert result["redacted_by_app"] == [
        {"bundle_id": "com.example.Editor", "reason": "unknown", "seconds": 10.0}
    ]


def test_capture_host_label_present_on_every_tool_response(tmp_path):
    settings = Settings(data_dir=tmp_path, assistant_host_label="studio-mini")
    rows = [(_capture_obs("row", ts=1000.0), "text")]
    spans = [_span("s", start_ts=990.0, end_ts=1000.0)]

    def service_for(store):
        return AssistantCaptureService(
            settings=settings, store_factory=lambda: store, clock=lambda: 1000.0
        )

    responses = [
        service_for(_FakeStore(recent=rows)).recent_capture(minutes=10, limit=5),
        service_for(_FakeStore(search=rows)).search_capture(query="text", limit=5),
        service_for(_FakeStore(spans=spans)).activity_summary(minutes=10),
        service_for(_FakeStore()).capture_status(),
    ]

    assert all(response["capture_host"] == "studio-mini" for response in responses)


def test_capture_host_defaults_to_hostname_then_sentinel(tmp_path, monkeypatch):
    monkeypatch.setattr(assistant.platform, "node", lambda: "real-host.local")
    service = AssistantCaptureService(
        settings=Settings(data_dir=tmp_path), store_factory=_FakeStore
    )
    assert service.capture_host == "real-host.local"

    monkeypatch.setattr(assistant.platform, "node", lambda: "")
    service = AssistantCaptureService(
        settings=Settings(data_dir=tmp_path), store_factory=_FakeStore
    )
    assert service.capture_host == "unknown-host"

    # A whitespace-only configured label falls back like an unset one.
    service = AssistantCaptureService(
        settings=Settings(data_dir=tmp_path, assistant_host_label="   "),
        store_factory=_FakeStore,
    )
    assert service.capture_host == "unknown-host"


def _walk_egress_paths(value, prefix=""):
    """Collect every data-bearing JSON path in a response.

    List items normalize to ``[]``; top-level bookkeeping keys are skipped;
    paths in EGRESS_MAP_FIELDS are terminal (their keys are data-dependent).
    An empty list or a None sub-object contributes nothing/its bare path, so
    the equality assertion below forces fixtures to be fully populated.
    """
    if isinstance(value, dict):
        paths = set()
        for key, sub in value.items():
            if not prefix and key in assistant.EGRESS_BOOKKEEPING_KEYS:
                continue
            path = f"{prefix}.{key}" if prefix else key
            if path in assistant.EGRESS_MAP_FIELDS:
                paths.add(path)
                continue
            paths |= _walk_egress_paths(sub, path)
        return paths
    if isinstance(value, list):
        paths = set()
        for item in value:
            paths |= _walk_egress_paths(item, f"{prefix}[]")
        return paths
    return {prefix}


def test_egress_declarations_match_emitted_paths_exactly(tmp_path):
    """Bidirectional pin: an emitted-but-undeclared field fails, and so does a
    declared-but-missing one. Fixtures must populate every declared path."""
    settings = Settings(
        data_dir=tmp_path,
        deep_brain_excluded_apps=["com.secret.App"],
        assistant_host_label="test-host",
    )
    rows = [
        (_capture_obs("kept", ts=1000.0), "kept excerpt"),
        (_capture_obs("gone", ts=999.0, app="com.secret.App"), "hidden"),
    ]
    spans = [
        _span("a", start_ts=0.0, end_ts=100.0, bundle_id="com.a"),
        _span("b", start_ts=100.0, end_ts=160.0, bundle_id="com.b"),
        _span("afk", start_ts=160.0, end_ts=200.0, bundle_id="com.a", afk=1),
        _span("red", start_ts=200.0, end_ts=230.0, detail_tier=0, reason="private"),
        _span(
            "gap", start_ts=230.0, end_ts=240.0,
            bundle_id=None, detail_tier=0, reason="paused",
        ),
        _span("excl", start_ts=240.0, end_ts=250.0, bundle_id="com.secret.App"),
    ]

    def service_for(store):
        return AssistantCaptureService(
            settings=settings, store_factory=lambda: store, clock=lambda: 1000.0
        )

    responses = {
        "recent": service_for(_FakeStore(recent=rows)).recent_capture(
            minutes=60, limit=5
        ),
        "search": service_for(_FakeStore(search=rows)).search_capture(
            query="kept", limit=5
        ),
        "activity": service_for(_FakeStore(spans=spans)).activity_summary(
            minutes=1440
        ),
        "status": service_for(
            _FakeStore(stats={"observations": 1, "encryption_enabled": True})
        ).capture_status(),
    }

    for name, response in responses.items():
        declared = set(response["egress"]["fields"])
        emitted = _walk_egress_paths(response)
        assert emitted == declared, (
            f"{name}: emitted-but-undeclared {sorted(emitted - declared)}; "
            f"declared-but-missing {sorted(declared - emitted)}"
        )
        assert response["egress"]["scope"] in {
            "capture_content", "activity_metadata", "status_metadata",
        }
        assert response["egress"]["untrusted_content"] is (
            response.get("captured_content_is_untrusted", False)
        )


def test_activity_summary_redacted_invariant_is_exact_with_fractional_spans(tmp_path):
    # Float addition is not associative: an independently accumulated total
    # can differ from the grouped-and-sorted component sum in the last ulp.
    # The total is derived from the emitted components, so equality is EXACT.
    spans = [
        _span(
            "f1", start_ts=0.0, end_ts=724.726879575788,
            bundle_id="com.a", detail_tier=0, reason="private",
        ),
        _span(
            "f2", start_ts=800.0, end_ts=800.0 + 541.5229705223803,
            bundle_id="com.b", detail_tier=0, reason="blocklisted",
        ),
        _span(
            "f3", start_ts=1400.0, end_ts=1400.0 + 774.7999935132281,
            bundle_id="com.a", detail_tier=0, reason="private",
        ),
        _span(
            "gap", start_ts=2200.0, end_ts=2233.3333333333335,
            bundle_id=None, detail_tier=0, reason="paused",
        ),
    ]
    service = AssistantCaptureService(
        settings=Settings(data_dir=tmp_path),
        store_factory=lambda: _FakeStore(spans=spans),
        clock=lambda: 2400.0,
    )

    result = service.activity_summary(minutes=40)

    assert result["redacted_seconds"] == (
        sum(entry["seconds"] for entry in result["redacted_by_app"])
        + result["redacted_other_seconds"]
        + result["redacted_unattributed_seconds"]
    )
    assert result["redacted_seconds"] > 0


# -- v3.2: activity_summary window modes ----------------------------------------


def _summary_service(tmp_path, *, spans=None, clock=1000.0, **settings_kwargs):
    return AssistantCaptureService(
        settings=Settings(data_dir=tmp_path, **settings_kwargs),
        store_factory=lambda: _FakeStore(spans=spans or []),
        clock=lambda: clock,
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"minutes": 10, "start_ts": 0.0, "end_ts": 60.0}, "only one of"),
        ({"minutes": 10, "local_day": "today"}, "only one of"),
        ({"start_ts": 0.0, "end_ts": 60.0, "local_day": "today"}, "only one of"),
        ({"start_ts": 0.0}, "provided together"),
        ({"end_ts": 60.0}, "provided together"),
        ({"timezone": "UTC"}, "only valid together with local_day"),
        ({"start_ts": True, "end_ts": 60.0}, "must be a number"),
        ({"start_ts": float("nan"), "end_ts": 60.0}, "must be finite"),
        ({"start_ts": 0.0, "end_ts": float("inf")}, "must be finite"),
        ({"start_ts": 60.0, "end_ts": 60.0}, "greater than start_ts"),
        ({"start_ts": 0.0, "end_ts": 90_000.0}, "at most 1440 minutes"),
        ({"local_day": "today", "timezone": "Not/AZone"}, "valid IANA"),
        ({"local_day": "today", "timezone": "  "}, "valid IANA"),
    ],
)
def test_activity_summary_rejects_invalid_window_modes(tmp_path, kwargs, message):
    service = _summary_service(tmp_path)
    with pytest.raises(ValueError, match=message):
        service.activity_summary(**kwargs)


@pytest.mark.parametrize(
    "bad_day",
    ["20260714", "2026-W29-2", "9999-12-31", "not-a-day", "2026-7-4", ""],
)
def test_activity_summary_local_day_enforces_strict_format(tmp_path, bad_day):
    service = _summary_service(tmp_path)
    with pytest.raises(ValueError, match="local_day must be"):
        service.activity_summary(local_day=bad_day, timezone="UTC")


def test_activity_summary_default_window_is_trailing_hour(tmp_path):
    service = _summary_service(tmp_path, clock=7200.0)

    result = service.activity_summary()

    assert result["window"] == {
        "mode": "minutes",
        "start_ts": 3600.0,
        "end_ts": 7200.0,
        "timezone": None,
        "local_day": None,
    }
    assert result["window_start_ts"] == 3600.0
    assert result["window_end_ts"] == 7200.0


def test_activity_summary_range_mode_is_half_open_with_echo(tmp_path):
    spans = [
        _span("inside", start_ts=150.0, end_ts=180.0),
        # Starts exactly at the exclusive end: clips to zero, contributes nothing.
        _span("at-end", start_ts=200.0, end_ts=260.0, bundle_id="com.late"),
    ]
    service = _summary_service(tmp_path, spans=spans, clock=10_000.0)

    result = service.activity_summary(start_ts=100.0, end_ts=200.0)

    assert result["window"] == {
        "mode": "range",
        "start_ts": 100.0,
        "end_ts": 200.0,
        "timezone": None,
        "local_day": None,
    }
    assert result["window_start_ts"] == 100.0
    assert result["window_end_ts"] == 200.0
    assert result["foreground_seconds"] == 30.0
    assert all(app["bundle_id"] != "com.late" for app in result["apps"])


def test_activity_summary_local_day_resolves_zoneinfo_midnights(tmp_path):
    from datetime import date, datetime, time as dt_time
    from zoneinfo import ZoneInfo

    tz = ZoneInfo("America/New_York")
    expected_start = datetime.combine(
        date(2026, 7, 14), dt_time(0, 0), tzinfo=tz
    ).timestamp()
    service = _summary_service(tmp_path, clock=10.0)

    result = service.activity_summary(local_day="2026-07-14", timezone="America/New_York")

    assert result["window"]["mode"] == "local_day"
    assert result["window"]["timezone"] == "America/New_York"
    assert result["window"]["local_day"] == "2026-07-14"
    assert result["window"]["start_ts"] == expected_start
    assert result["window"]["end_ts"] - result["window"]["start_ts"] == 86_400.0


@pytest.mark.parametrize(
    ("tz_name", "day", "expected_seconds"),
    [
        # 02:00 transitions.
        ("America/New_York", "2026-03-08", 82_800.0),
        ("America/New_York", "2026-11-01", 90_000.0),
        # Midnight transitions: nonexistent midnight (spring forward at 00:00)
        # and repeated midnight (fall back onto 00:00, fold=0 first occurrence).
        ("America/Havana", "2026-03-08", 82_800.0),
        ("America/Havana", "2026-11-01", 90_000.0),
    ],
)
def test_activity_summary_local_day_follows_dst_rules(
    tmp_path, tz_name, day, expected_seconds
):
    service = _summary_service(tmp_path, clock=10.0)

    result = service.activity_summary(local_day=day, timezone=tz_name)

    window = result["window"]
    assert window["end_ts"] - window["start_ts"] == expected_seconds
    # A 25h day deliberately exceeds the minutes-mode ceiling.
    if expected_seconds > 86_400.0:
        assert expected_seconds > assistant.MAX_MINUTES * 60


def test_activity_summary_skipped_local_day_fails_closed(tmp_path):
    # Samoa skipped 2011-12-30 crossing the dateline: both midnights resolve
    # to the same instant, and an empty window must never reach the store.
    service = _summary_service(tmp_path, clock=10.0)
    with pytest.raises(ValueError, match="does not exist in this timezone"):
        service.activity_summary(local_day="2011-12-30", timezone="Pacific/Apia")


def test_activity_summary_relative_days_resolve_in_requested_timezone(tmp_path):
    # 2026-07-15 03:00 UTC is still 2026-07-14 in Los Angeles.
    clock = 1784084400.0  # 2026-07-15T03:00:00Z
    from datetime import datetime, timezone as dt_timezone

    assert datetime.fromtimestamp(clock, dt_timezone.utc).strftime(
        "%Y-%m-%dT%H:%M"
    ) == "2026-07-15T03:00"
    service = _summary_service(tmp_path, clock=clock)

    today = service.activity_summary(local_day="today", timezone="America/Los_Angeles")
    yesterday = service.activity_summary(
        local_day="yesterday", timezone="America/Los_Angeles"
    )
    utc_today = service.activity_summary(local_day="today", timezone="UTC")

    assert today["window"]["local_day"] == "2026-07-14"
    assert yesterday["window"]["local_day"] == "2026-07-13"
    assert utc_today["window"]["local_day"] == "2026-07-15"


def test_activity_summary_uses_system_timezone_when_omitted(tmp_path, monkeypatch):
    monkeypatch.setattr(assistant, "_system_timezone_name", lambda: "Europe/Berlin")
    service = _summary_service(tmp_path, clock=10.0)

    result = service.activity_summary(local_day="2026-07-14")

    assert result["window"]["timezone"] == "Europe/Berlin"


def test_system_timezone_name_parses_symlink_and_fails_closed(monkeypatch):
    monkeypatch.setattr(
        assistant.os.path, "realpath",
        lambda _p: "/usr/share/zoneinfo/Europe/Berlin",
    )
    assert assistant._system_timezone_name() == "Europe/Berlin"

    monkeypatch.setattr(assistant.os.path, "realpath", lambda _p: "/nonsense")
    with pytest.raises(ValueError, match="could not resolve the system timezone"):
        assistant._system_timezone_name()

    monkeypatch.setattr(
        assistant.os.path, "realpath",
        lambda _p: "/usr/share/zoneinfo/Not/AZone",
    )
    with pytest.raises(ValueError, match="could not resolve the system timezone"):
        assistant._system_timezone_name()


def test_store_capture_spans_overlapping_is_half_open_and_skips_empty(tmp_path):
    settings = Settings(data_dir=tmp_path, embed_dim=64)
    store = MemoryStore(settings=settings, provider=FakeProvider(embed_dim=64))
    try:
        def span(start, end, bundle="com.example.Editor"):
            store.open_span(
                epoch_id="epoch", start_ts=start, end_ts=end,
                bundle_id=bundle, detail_tier=1,
            )

        span(120.0, 180.0)          # real overlap
        span(50.0, 100.0)           # ends exactly at window start: excluded
        span(200.0, 260.0)          # starts exactly at window end: excluded
        for i in range(10):         # zero-length rows: never fetched
            span(150.0 + i, 150.0 + i)

        # limit smaller than the junk-row count proves empty/boundary rows
        # no longer consume cap slots (the old closed predicate fetched them).
        rows = store.capture_spans_overlapping(100.0, 200.0, limit=4)
        assert [r["span_id"] for r in rows] and len(rows) == 1
        assert rows[0]["start_ts"] == 120.0
    finally:
        store.close()


def test_mcp_activity_summary_schema_and_window_modes_end_to_end(tmp_path):
    service = _summary_service(
        tmp_path,
        spans=[_span("s", start_ts=100.0, end_ts=160.0)],
        clock=1000.0,
    )
    server = assistant.create_mcp_server(service)

    tools = asyncio.run(server.list_tools())
    schema = next(
        t for t in tools if t.name == "openbird_activity_summary"
    ).inputSchema
    assert set(schema["properties"]) == {
        "minutes", "start_ts", "end_ts", "local_day", "timezone",
    }
    # FastMCP omits the `required` key entirely when every argument is optional.
    assert schema.get("required", []) == []

    _content, ranged = asyncio.run(
        server.call_tool(
            "openbird_activity_summary", {"start_ts": 50.0, "end_ts": 250.0}
        )
    )
    assert ranged["window"]["mode"] == "range"
    assert ranged["foreground_seconds"] == 60.0

    _content, day = asyncio.run(
        server.call_tool(
            "openbird_activity_summary",
            {"local_day": "1970-01-01", "timezone": "UTC"},
        )
    )
    assert day["window"]["mode"] == "local_day"
    assert day["window"]["timezone"] == "UTC"


def test_egress_declarations_hold_in_local_day_mode(tmp_path):
    spans = [
        _span("a", start_ts=100.0, end_ts=200.0, bundle_id="com.a"),
        _span("b", start_ts=200.0, end_ts=260.0, bundle_id="com.b"),
        _span("afk", start_ts=260.0, end_ts=300.0, bundle_id="com.a", afk=1),
        _span("red", start_ts=300.0, end_ts=330.0, detail_tier=0, reason="private"),
        _span(
            "gap", start_ts=330.0, end_ts=340.0,
            bundle_id=None, detail_tier=0, reason="paused",
        ),
    ]
    service = _summary_service(
        tmp_path, spans=spans, clock=1000.0, assistant_host_label="test-host"
    )

    response = service.activity_summary(local_day="1970-01-01", timezone="UTC")

    declared = set(response["egress"]["fields"])
    emitted = _walk_egress_paths(response)
    assert emitted == declared, (
        f"emitted-but-undeclared {sorted(emitted - declared)}; "
        f"declared-but-missing {sorted(declared - emitted)}"
    )
