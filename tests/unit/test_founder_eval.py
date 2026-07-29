"""Privacy, boundedness, and lifecycle tests for founder-context evaluation."""

from __future__ import annotations

import json
import logging
import os
import plistlib
import sqlite3
import stat
import sys
import threading
import time
import types
import uuid
from pathlib import Path

import pytest
from typer.testing import CliRunner

from openbird.capture.founder_eval import (
    MAX_SNAPSHOT_BYTES,
    evaluate_and_record_snapshot,
    evaluate_store,
    read_snapshot,
    snapshot_path,
    unavailable_report,
    write_snapshot,
)
from openbird.cli import app
from openbird.config import Settings, reset_settings_cache
from openbird.memory.store import MemoryStore
from openbird.routines.launchd import (
    FOUNDER_CONTEXT_EVAL_LABEL,
    build_founder_context_eval_plist,
)
from openbird.storage import crypto
from tests.unit.conftest import FakeProvider


def _store(tmp_path: Path) -> tuple[Settings, MemoryStore]:
    settings = Settings(data_dir=tmp_path, embed_dim=64)
    return settings, MemoryStore(
        settings=settings, provider=FakeProvider(embed_dim=64)
    )


def _write_fresh_liveness(settings: Settings, now: float) -> None:
    (settings.data_dir / "capture.liveness.json").write_text(
        json.dumps(
            {
                "updated_at": now,
                "instance_uuid": str(uuid.uuid4()),
                "pid": 123,
                "runtime_version": "0.14.0",
                "mode": "stream",
                "heartbeat_seq": 5,
            }
        )
    )


def _seed_rich_corpus(store: MemoryStore, *, now: float) -> None:
    for i in range(12):
        cue = (
            "Decided on the local-first architecture. "
            if i % 3 == 0
            else "Implemented and merged the bounded evaluator. "
            if i % 3 == 1
            else "Next step is follow up on validation. "
        )
        store.add_observation(
            cue
            + "Substantial founder work context with enough detail to support "
            + f"a grounded recap source {i}. SECRET_RAW_TEXT_{i}",
            source=("capture", "meeting", "ingest", "mcp")[i % 4],
            ts=now - i * 60,
            app=f"com.example.app{i % 3}",
            window=f"SECRET_WINDOW_TITLE_{i}",
            session_id=f"session-{i}",
        )
    store.conn.execute(
        """
        INSERT INTO capture_attempts(
            attempt_id, helper_epoch, trigger_seq, trigger_ts, started_ts,
            finished_ts, status, bundle_id, trigger, adapter_id,
            extractor_version, policy_tier, outcome, nodes_visited,
            bytes_emitted, elapsed_ms, completeness, reason_codes_json,
            coalesced_trigger_count
        ) VALUES (?, ?, 1, ?, ?, ?, 'finished', ?, 'typing_pause',
                  'generic_ax', 'generic_ax_v1', 1, 'captured_partial',
                  10, 500, 12, 'partial', ?, 0)
        """,
        (
            str(uuid.uuid4()),
            str(uuid.uuid4()),
            now - 30,
            now - 30,
            now - 29,
            "com.private.attempt.bundle",
            '["budget_exhausted"]',
        ),
    )
    store.conn.commit()


def test_eval_is_ready_and_payload_contains_metadata_only(monkeypatch, tmp_path):
    settings, store = _store(tmp_path)
    now = 1_800_000_000.0
    try:
        _seed_rich_corpus(store, now=now)
        _write_fresh_liveness(settings, now)
        monkeypatch.setattr(
            store,
            "capture_app_activity",
            lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("scheduled evaluation must not scan all history")
            ),
        )
        report = evaluate_store(store, settings=settings, now=now)
    finally:
        store.close()

    assert report["state"] == "ready"
    assert report["corpus"]["observations"] == 12
    assert report["corpus"]["source_counts"] == {
        "capture": 3,
        "meeting": 3,
        "ingest": 3,
        "mcp": 3,
    }
    assert report["capture"]["daemon"]["mode"] == "stream"
    attempts = report["capture"]["attempts"]
    assert attempts["completeness_counts"]["partial"] == 1
    assert attempts["outcome_counts"]["captured_partial"] == 1
    assert attempts["trigger_counts"]["typing_pause"] == 1
    assert attempts["adapter_counts"]["generic_ax"] == 1
    assert attempts["budget_exhausted"] == 1

    encoded = json.dumps(report, sort_keys=True)
    assert "com.private.attempt.bundle" not in encoded
    assert "SECRET_WINDOW_TITLE" not in encoded
    assert "SECRET_RAW_TEXT" not in encoded
    assert '"window"' not in encoded
    assert '"text"' not in encoded


def test_eval_stale_capture_is_not_ready_with_closed_reason(tmp_path):
    settings, store = _store(tmp_path)
    now = 1_800_000_000.0
    try:
        _seed_rich_corpus(store, now=now - 3 * 86_400)
        report = evaluate_store(store, settings=settings, now=now)
    finally:
        store.close()
    assert report["state"] == "not_ready"
    assert "capture_stale" in report["reason_codes"]


def test_snapshot_is_atomic_owner_only_and_prior_read_is_bounded(tmp_path):
    settings = Settings(data_dir=tmp_path)
    target = snapshot_path(settings)
    report = unavailable_report(settings=settings, reason="store_absent", now=1.0)
    write_snapshot(report, target)

    assert read_snapshot(target) == report
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert not list(target.parent.glob(f".{target.name}.*"))

    target.write_bytes(b"{" + b"x" * MAX_SNAPSHOT_BYTES + b"}")
    assert read_snapshot(target) is None


def test_selective_delete_invalidates_snapshot(tmp_path):
    settings, store = _store(tmp_path)
    try:
        obs = store.add_observation(
            "content that will be deleted",
            source="capture",
            ts=100.0,
            app="com.example.editor",
        )
        target = snapshot_path(settings)
        write_snapshot(
            unavailable_report(settings=settings, reason="store_absent"), target
        )
        assert target.exists()
        assert store.delete(since_ts=obs.ts) == 1
        assert not target.exists()
    finally:
        store.close()


def test_snapshot_unlink_failure_rolls_back_delete_and_logs_metadata_only(
    monkeypatch, caplog, tmp_path
):
    from openbird.capture import founder_eval

    _settings, store = _store(tmp_path)
    try:
        obs = store.add_observation(
            "PRIVATE CONTENT THAT MUST REMAIN ON FAILED DELETE",
            source="capture",
            ts=100.0,
            app="com.example.private",
        )

        def fail_invalidation(**_kwargs):
            raise PermissionError("PRIVATE FILESYSTEM DETAIL")

        monkeypatch.setattr(founder_eval, "invalidate_snapshot", fail_invalidation)
        with caplog.at_level(logging.WARNING, logger="openbird.memory"):
            try:
                store.delete(since_ts=obs.ts)
            except PermissionError:
                pass
            else:
                raise AssertionError("delete must fail when derived metadata persists")

        remaining = store.conn.execute(
            "SELECT COUNT(*) AS count FROM observations WHERE id = ?",
            (obs.id,),
        ).fetchone()["count"]
        assert remaining == 1
        assert (
            "founder_context_snapshot_invalidation_failed reason=unlink_error"
            in caplog.text
        )
        assert "PRIVATE FILESYSTEM DETAIL" not in caplog.text
        assert "PRIVATE CONTENT" not in caplog.text
        assert "com.example.private" not in caplog.text
    finally:
        store.close()


def test_recorded_eval_and_delete_have_a_strict_privacy_safe_order(
    monkeypatch, tmp_path
):
    from openbird.capture import founder_eval

    settings, seed = _store(tmp_path)
    try:
        obs = seed.add_observation(
            "recent founder work that must not survive in stale metadata",
            source="capture",
            ts=100.0,
            app="com.example.private",
        )
    finally:
        seed.close()

    target = snapshot_path(settings)
    real_write_snapshot = founder_eval.write_snapshot
    write_entered = threading.Event()
    release_write = threading.Event()
    delete_ready = threading.Event()
    start_delete = threading.Event()
    delete_attempted = threading.Event()
    errors: list[BaseException] = []
    deleted: list[int] = []

    def blocked_write(report, path):
        write_entered.set()
        if not release_write.wait(5):
            raise TimeoutError("test did not release snapshot write")
        real_write_snapshot(report, path)

    monkeypatch.setattr(founder_eval, "write_snapshot", blocked_write)

    def delete_worker():
        delete_store = MemoryStore(
            settings=settings,
            provider=FakeProvider(embed_dim=64),
        )
        original_begin = delete_store._begin

        def signaled_begin():
            delete_attempted.set()
            return original_begin()

        delete_store._begin = signaled_begin  # type: ignore[method-assign]
        delete_ready.set()
        try:
            if not start_delete.wait(5):
                raise TimeoutError("test did not start deletion")
            deleted.append(delete_store.delete(since_ts=obs.ts))
        except BaseException as exc:
            errors.append(exc)
        finally:
            delete_store.close()

    def eval_worker():
        eval_store = MemoryStore(
            settings=settings,
            provider=FakeProvider(embed_dim=64),
        )
        try:
            evaluate_and_record_snapshot(
                eval_store,
                settings=settings,
                path=target,
                now=200.0,
            )
        except BaseException as exc:
            errors.append(exc)
        finally:
            eval_store.close()

    delete_thread = threading.Thread(target=delete_worker, daemon=True)
    eval_thread = threading.Thread(target=eval_worker, daemon=True)
    delete_thread.start()
    assert delete_ready.wait(5)
    eval_thread.start()
    try:
        assert write_entered.wait(5)
        lock_probe = sqlite3.connect(str(settings.db_path), timeout=0.0)
        try:
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                lock_probe.execute("BEGIN IMMEDIATE")
        finally:
            lock_probe.close()
        start_delete.set()
        assert delete_attempted.wait(5)
    finally:
        release_write.set()
        start_delete.set()
    eval_thread.join(5)
    delete_thread.join(5)

    assert not eval_thread.is_alive()
    assert not delete_thread.is_alive()
    assert errors == []
    assert deleted == [1]
    assert not target.exists()
    verify = MemoryStore(settings=settings, provider=FakeProvider(embed_dim=64))
    try:
        remaining = verify.conn.execute(
            "SELECT COUNT(*) AS count FROM observations WHERE id = ?",
            (obs.id,),
        ).fetchone()["count"]
        assert remaining == 0
    finally:
        verify.close()


def test_periodic_plist_is_short_lived_silent_and_no_model():
    data = plistlib.loads(
        build_founder_context_eval_plist(
            program_args=[
                "/usr/local/bin/openbird",
                "eval",
                "founder-context",
                "run",
                "--record",
                "--scheduled",
                "--quiet",
            ]
        )
    )
    assert data["Label"] == FOUNDER_CONTEXT_EVAL_LABEL
    assert data["RunAtLoad"] is True
    assert data["StartInterval"] == 21_600
    assert "KeepAlive" not in data
    assert data["ProcessType"] == "Background"
    assert data["LowPriorityIO"] is True
    assert data["Nice"] > 0
    assert data["Umask"] == 63
    assert data["StandardOutPath"] == data["StandardErrorPath"] == "/dev/null"
    joined = " ".join(data["ProgramArguments"])
    assert "probe-answer" not in joined
    assert "chat" not in joined


def test_read_only_keyring_guard_never_mints_key(monkeypatch):
    calls = {"set": 0}
    fake = types.SimpleNamespace(
        get_password=lambda *_args: None,
        set_password=lambda *_args: calls.__setitem__("set", calls["set"] + 1),
    )
    monkeypatch.setitem(sys.modules, "keyring", fake)
    monkeypatch.delenv("OPENBIRD_DB_KEY", raising=False)
    monkeypatch.delenv("OPENBIRD_DISABLE_KEYRING", raising=False)
    monkeypatch.setenv("OPENBIRD_KEYRING_READ_ONLY", "1")

    assert crypto._get_or_create_key() is None
    assert calls["set"] == 0


def test_scheduled_absent_store_records_snapshot_without_creating_db(
    monkeypatch, tmp_path
):
    data_dir = tmp_path / "new-data"
    monkeypatch.setenv("OPENBIRD_DATA_DIR", str(data_dir))
    monkeypatch.setenv("OPENBIRD_DISABLE_KEYRING", "1")
    monkeypatch.delenv("OPENBIRD_DB_PATH", raising=False)
    reset_settings_cache()
    result = CliRunner().invoke(
        app,
        [
            "eval",
            "founder-context",
            "run",
            "--scheduled",
            "--record",
            "--quiet",
        ],
    )
    try:
        assert result.exit_code == 0, result.output
        assert not (data_dir / "openbird.db").exists()
        report = json.loads(
            (data_dir / "logs" / "founder-context-eval.json").read_text()
        )
        assert report["state"] == "not_ready"
        assert report["reason_codes"] == ["store_absent"]
    finally:
        reset_settings_cache()


def test_interactive_eval_honors_bounded_parent_keyring_timeout(
    monkeypatch, tmp_path
):
    import openbird.cli as cli_module

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    db_path = data_dir / "openbird.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE marker(value INTEGER)")
    conn.commit()
    conn.close()
    observed: list[str | None] = []

    def fail_open():
        observed.append(os.environ.get("OPENBIRD_KEYRING_TIMEOUT_SECONDS"))
        raise RuntimeError("open failed")

    monkeypatch.setattr(cli_module, "_store_maintenance", fail_open)
    monkeypatch.setenv("OPENBIRD_DATA_DIR", str(data_dir))
    monkeypatch.setenv("OPENBIRD_KEYRING_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("OPENBIRD_DISABLE_KEYRING", "1")
    reset_settings_cache()
    result = CliRunner().invoke(
        app,
        ["eval", "founder-context", "run", "--json"],
    )
    try:
        assert result.exit_code == 0, result.output
        assert observed == ["30.0"]
        assert json.loads(result.output)["reason_codes"] == ["store_open_failed"]
    finally:
        reset_settings_cache()


def test_scheduled_eval_forces_short_keyring_timeout(monkeypatch, tmp_path):
    import openbird.cli as cli_module

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    db_path = data_dir / "openbird.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE marker(value INTEGER)")
    conn.commit()
    conn.close()
    observed: list[str | None] = []

    def fail_open():
        observed.append(os.environ.get("OPENBIRD_KEYRING_TIMEOUT_SECONDS"))
        raise RuntimeError("open failed")

    monkeypatch.setattr(cli_module, "_store_maintenance", fail_open)
    monkeypatch.setenv("OPENBIRD_DATA_DIR", str(data_dir))
    monkeypatch.setenv("OPENBIRD_KEYRING_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("OPENBIRD_DISABLE_KEYRING", "1")
    reset_settings_cache()
    result = CliRunner().invoke(
        app,
        ["eval", "founder-context", "run", "--scheduled", "--quiet"],
    )
    try:
        assert result.exit_code == 0, result.output
        assert observed == ["2.0"]
    finally:
        reset_settings_cache()


def test_evaluation_failure_is_not_mislabeled_as_store_or_keychain_failure(
    monkeypatch, tmp_path
):
    import openbird.cli as cli_module
    from openbird.capture import founder_eval

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    db_path = data_dir / "openbird.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE marker(value INTEGER)")
    conn.commit()
    conn.close()
    settings = Settings(data_dir=data_dir)
    previous = unavailable_report(
        settings=settings,
        reason="store_absent",
        now=1.0,
    )
    previous["corpus"]["recent_text_bytes"] = 50
    write_snapshot(previous, snapshot_path(settings))

    fake_store = types.SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(cli_module, "_store_maintenance", lambda: fake_store)

    def fail_evaluation(*_args, **_kwargs):
        raise RuntimeError("PRIVATE DATABASE ERROR DETAIL")

    monkeypatch.setattr(founder_eval, "evaluate_store", fail_evaluation)
    monkeypatch.setenv("OPENBIRD_DATA_DIR", str(data_dir))
    monkeypatch.setenv("OPENBIRD_DISABLE_KEYRING", "1")
    reset_settings_cache()
    result = CliRunner().invoke(
        app,
        ["eval", "founder-context", "run", "--json"],
    )
    try:
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["reason_codes"] == ["evaluation_failed"]
        assert payload["storage"]["recent_text_bytes_delta"] == -50
        assert "PRIVATE DATABASE ERROR DETAIL" not in result.output
        assert "store_open_failed" not in result.output
        assert "encrypted_store_unavailable" not in result.output
    finally:
        reset_settings_cache()


def test_snapshot_write_failure_returns_closed_metadata_only_reason(
    monkeypatch, tmp_path
):
    from openbird.capture import founder_eval

    data_dir = tmp_path / "data"

    def fail_write(*_args, **_kwargs):
        raise OSError("PRIVATE FILESYSTEM DETAIL")

    monkeypatch.setattr(founder_eval, "write_snapshot", fail_write)
    monkeypatch.setenv("OPENBIRD_DATA_DIR", str(data_dir))
    monkeypatch.setenv("OPENBIRD_DISABLE_KEYRING", "1")
    reset_settings_cache()
    result = CliRunner().invoke(
        app,
        ["eval", "founder-context", "run", "--record", "--json"],
    )
    try:
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["reason_codes"] == ["snapshot_write_failed"]
        assert "PRIVATE FILESYSTEM DETAIL" not in result.output
        assert not snapshot_path(Settings(data_dir=data_dir)).exists()
    finally:
        reset_settings_cache()


def test_manual_answer_probe_is_transient_and_never_recorded(
    monkeypatch, tmp_path
):
    import openbird.cli as cli_module
    from openbird.config import get_settings

    data_dir = tmp_path / "data"
    monkeypatch.setenv("OPENBIRD_DATA_DIR", str(data_dir))
    monkeypatch.setenv("OPENBIRD_DISABLE_KEYRING", "1")
    reset_settings_cache()
    settings = get_settings()
    store = MemoryStore(
        settings=settings,
        provider=FakeProvider(embed_dim=settings.embed_dim),
    )
    store.add_observation(
        "recent founder work for a transient probe",
        source="capture",
        ts=time.time(),
        app="com.example.editor",
    )
    monkeypatch.setattr(cli_module, "_store_maintenance", lambda: store)
    monkeypatch.setattr(
        cli_module,
        "_founder_answer_probe",
        lambda _store: {
            "ok": True,
            "grounding": "occurrence",
            "occurrence_citation_count": 1,
            "derived_citation_count": 0,
            "citation_app_ids": ["com.example.editor"],
            "citation_timestamps": [100.0],
            "elapsed_ms": 1.0,
        },
    )
    result = CliRunner().invoke(
        app,
        [
            "eval",
            "founder-context",
            "run",
            "--record",
            "--probe-answer",
            "--json",
        ],
    )
    try:
        assert result.exit_code == 0, result.output
        output = json.loads(result.output)
        assert output["answer_probe"]["occurrence_citation_count"] == 1
        persisted = json.loads(snapshot_path(settings).read_text())
        assert "answer_probe" not in persisted
        assert "com.example.editor" in persisted["corpus"]["app_ids"]
    finally:
        reset_settings_cache()


def test_cli_install_and_uninstall_write_only_test_plist(monkeypatch, tmp_path):
    from openbird.routines import launchd

    target = tmp_path / "founder.plist"
    monkeypatch.setattr(
        launchd, "founder_context_eval_plist_path", lambda **_: target
    )
    monkeypatch.setenv("OPENBIRD_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("OPENBIRD_DISABLE_KEYRING", "1")
    reset_settings_cache()
    runner = CliRunner()
    installed = runner.invoke(
        app,
        [
            "eval",
            "founder-context",
            "install",
            "--executable",
            str(tmp_path / "openbird"),
        ],
    )
    assert installed.exit_code == 0, installed.output
    assert target.exists()
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    removed = runner.invoke(
        app, ["eval", "founder-context", "uninstall"]
    )
    try:
        assert removed.exit_code == 0, removed.output
        assert not target.exists()
    finally:
        reset_settings_cache()


def test_encrypted_install_preflights_exact_scheduled_executable(
    monkeypatch, tmp_path
):
    import subprocess

    from openbird.routines import launchd

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "openbird.db").write_bytes(b"\x01encrypted-header")
    target = tmp_path / "founder.plist"
    monkeypatch.setattr(
        launchd, "founder_context_eval_plist_path", lambda **_: target
    )
    calls: list[tuple[list[str], dict[str, str]]] = []
    original_run = subprocess.run

    def fake_run(command, **kwargs):
        if not command or command[0] != "/opt/test/openbird":
            return original_run(command, **kwargs)
        calls.append((list(command), dict(kwargs["env"])))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {
                    "state": "not_ready",
                    "reason_codes": ["capture_stale"],
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setenv("OPENBIRD_DATA_DIR", str(data_dir))
    monkeypatch.delenv("OPENBIRD_DISABLE_KEYRING", raising=False)
    reset_settings_cache()
    result = CliRunner().invoke(
        app,
        [
            "eval",
            "founder-context",
            "install",
            "--executable",
            "/opt/test/openbird",
        ],
    )
    try:
        assert result.exit_code == 0, result.output
        assert calls[0][0][:4] == [
            "/opt/test/openbird",
            "eval",
            "founder-context",
            "run",
        ]
        assert calls[0][1]["OPENBIRD_KEYRING_READ_ONLY"] == "1"
        assert target.exists()
    finally:
        reset_settings_cache()


def test_encrypted_install_reports_evaluation_failure_separately(
    monkeypatch, tmp_path
):
    import subprocess

    from openbird.routines import launchd

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "openbird.db").write_bytes(b"\x01encrypted-header")
    target = tmp_path / "founder.plist"
    monkeypatch.setattr(
        launchd, "founder_context_eval_plist_path", lambda **_: target
    )
    original_run = subprocess.run

    def fake_run(command, **kwargs):
        if not command or command[0] != "/opt/test/openbird":
            return original_run(command, **kwargs)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {
                    "state": "not_ready",
                    "reason_codes": ["evaluation_failed"],
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setenv("OPENBIRD_DATA_DIR", str(data_dir))
    monkeypatch.delenv("OPENBIRD_DISABLE_KEYRING", raising=False)
    reset_settings_cache()
    result = CliRunner().invoke(
        app,
        [
            "eval",
            "founder-context",
            "install",
            "--executable",
            "/opt/test/openbird",
        ],
    )
    try:
        assert result.exit_code == 1
        normalized = " ".join(result.output.split())
        assert "could not complete the metadata evaluation" in normalized
        assert "could not open and verify the encrypted store" not in normalized
        assert not target.exists()
    finally:
        reset_settings_cache()
