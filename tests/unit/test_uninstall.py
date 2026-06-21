"""Tests for cleanup tooling: non-creating path resolvers, the key-safety rule,
and the uninstall orchestration (no real launchctl / lsregister / Keychain)."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from openbird import config, uninstall
from openbird.storage import crypto

_MAGIC = b"SQLite format 3\x00"


# --------------------------------------------------------------------------- #
# config: non-creating resolvers (Codex #2)                                   #
# --------------------------------------------------------------------------- #


def test_data_dir_path_does_not_create(monkeypatch, tmp_path):
    target = tmp_path / "nope"
    monkeypatch.setenv("OPENBIRD_DATA_DIR", str(target))
    resolved = config.data_dir_path()
    assert resolved == target
    assert not target.exists()  # resolving must NOT materialize the dir


def test_db_file_path_custom_override_outside_data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENBIRD_DATA_DIR", str(tmp_path / "data"))
    custom = tmp_path / "elsewhere" / "custom.db"
    monkeypatch.setenv("OPENBIRD_DB_PATH", str(custom))
    assert config.db_file_path() == custom


def test_db_file_path_empty_override_is_unset(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENBIRD_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("OPENBIRD_DB_PATH", "")  # empty == "no override"
    assert config.db_file_path() == tmp_path / "data" / "openbird.db"


# --------------------------------------------------------------------------- #
# crypto: header check + delete_key                                           #
# --------------------------------------------------------------------------- #


def test_db_plaintext_or_absent_cases(tmp_path):
    missing = tmp_path / "missing.db"
    empty = tmp_path / "empty.db"
    empty.touch()
    plain = tmp_path / "plain.db"
    plain.write_bytes(_MAGIC + b"rest-of-file")
    enc = tmp_path / "enc.db"
    enc.write_bytes(b"\x00\x01encrypted-bytes-not-the-magic")

    assert crypto.db_is_plaintext_or_absent(missing) is True
    assert crypto.db_is_plaintext_or_absent(empty) is True
    assert crypto.db_is_plaintext_or_absent(plain) is True
    assert crypto.db_is_plaintext_or_absent(enc) is False  # encrypted -> retain key


class _FakeKeyring(types.SimpleNamespace):
    deleted: list[tuple[str, str]] = []

    @staticmethod
    def delete_password(service, user):
        _FakeKeyring.deleted.append((service, user))


def test_delete_key_invokes_keyring(monkeypatch):
    _FakeKeyring.deleted = []
    errors_mod = types.SimpleNamespace(PasswordDeleteError=RuntimeError)
    monkeypatch.setitem(sys.modules, "keyring", _FakeKeyring)
    monkeypatch.setitem(sys.modules, "keyring.errors", errors_mod)
    monkeypatch.delenv("OPENBIRD_DISABLE_KEYRING", raising=False)
    assert crypto.delete_key() is True
    assert _FakeKeyring.deleted == [(crypto._KEYRING_SERVICE, crypto._KEYRING_USER)]


def test_delete_key_respects_disable_env(monkeypatch):
    _FakeKeyring.deleted = []
    monkeypatch.setitem(sys.modules, "keyring", _FakeKeyring)
    monkeypatch.setenv("OPENBIRD_DISABLE_KEYRING", "1")
    assert crypto.delete_key() is False
    assert _FakeKeyring.deleted == []  # never touched the Keychain


# --------------------------------------------------------------------------- #
# uninstall orchestration: key-safety rule (Codex #1)                         #
# --------------------------------------------------------------------------- #


@pytest.fixture
def _no_system_side_effects(monkeypatch, tmp_path):
    """Neutralize launchctl/lsregister/plist so orchestration tests stay hermetic.

    Critically redirects the routines plist to a non-existent tmp path so the
    real `~/Library/LaunchAgents/ai.openbird.routines.plist` is never unlinked.
    """
    monkeypatch.setattr(uninstall, "_run", lambda cmd: (0, ""))
    monkeypatch.setattr(uninstall, "_registered_app_paths", lambda: [])
    monkeypatch.setattr(uninstall, "_is_macos", lambda: True)
    from openbird.routines import launchd

    monkeypatch.setattr(
        launchd, "agent_plist_path", lambda **_: tmp_path / "no-such.plist"
    )


def _patch_delete_key(monkeypatch):
    calls = {"n": 0}

    def fake_delete_key():
        calls["n"] += 1
        return True

    monkeypatch.setattr(crypto, "delete_key", fake_delete_key)
    return calls


def _status_of(results, action):
    return [r.status for r in results if r.action == action]


def test_dry_run_touches_nothing(monkeypatch, tmp_path, _no_system_side_effects):
    data_dir = tmp_path / "data"
    monkeypatch.setenv("OPENBIRD_DATA_DIR", str(data_dir))
    monkeypatch.delenv("OPENBIRD_DB_PATH", raising=False)
    calls = _patch_delete_key(monkeypatch)

    results = uninstall.run_uninstall(purge_data=True, dry_run=True)

    assert not data_dir.exists()  # never created or removed
    assert calls["n"] == 0  # key never deleted in dry-run
    assert "would" in _status_of(results, "keychain key")


def test_no_purge_plaintext_db_deletes_key(
    monkeypatch, tmp_path, _no_system_side_effects
):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "openbird.db").write_bytes(_MAGIC + b"x")
    monkeypatch.setenv("OPENBIRD_DATA_DIR", str(data_dir))
    monkeypatch.delenv("OPENBIRD_DB_PATH", raising=False)
    calls = _patch_delete_key(monkeypatch)

    results = uninstall.run_uninstall(purge_data=False, dry_run=False)

    assert calls["n"] == 1
    assert "done" in _status_of(results, "keychain key")
    assert data_dir.exists()  # data preserved when not purging


def test_no_purge_encrypted_db_retains_key(
    monkeypatch, tmp_path, _no_system_side_effects
):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "openbird.db").write_bytes(b"\x00encrypted")
    monkeypatch.setenv("OPENBIRD_DATA_DIR", str(data_dir))
    monkeypatch.delenv("OPENBIRD_DB_PATH", raising=False)
    calls = _patch_delete_key(monkeypatch)

    results = uninstall.run_uninstall(purge_data=False, dry_run=False)

    assert calls["n"] == 0  # encrypted DB present -> key retained
    assert "retained" in _status_of(results, "keychain key")


def test_purge_with_custom_external_encrypted_db_retains_key(
    monkeypatch, tmp_path, _no_system_side_effects
):
    """--purge-data removes the data dir, but a custom encrypted DB OUTSIDE it
    survives, so the key must be RETAINED (round-2 stranding fix)."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    external = tmp_path / "external" / "custom.db"
    external.parent.mkdir()
    external.write_bytes(b"\x00encrypted-external")
    monkeypatch.setenv("OPENBIRD_DATA_DIR", str(data_dir))
    monkeypatch.setenv("OPENBIRD_DB_PATH", str(external))
    calls = _patch_delete_key(monkeypatch)

    results = uninstall.run_uninstall(purge_data=True, dry_run=False)

    assert not data_dir.exists()  # purged
    assert external.exists()  # external custom DB untouched
    assert calls["n"] == 0  # key retained -> no stranding
    assert "retained" in _status_of(results, "keychain key")
    assert "done" in _status_of(results, "purge data")


def test_purge_default_db_deletes_key(monkeypatch, tmp_path, _no_system_side_effects):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "openbird.db").write_bytes(b"\x00encrypted")  # encrypted, but inside
    monkeypatch.setenv("OPENBIRD_DATA_DIR", str(data_dir))
    monkeypatch.delenv("OPENBIRD_DB_PATH", raising=False)
    calls = _patch_delete_key(monkeypatch)

    results = uninstall.run_uninstall(purge_data=True, dry_run=False)

    assert not data_dir.exists()  # default DB went with the data dir
    assert calls["n"] == 1  # DB now absent -> safe to delete key
    assert "done" in _status_of(results, "keychain key")


# --------------------------------------------------------------------------- #
# uninstall: Launch Services bundle-id validation (Codex #5)                   #
# --------------------------------------------------------------------------- #


def test_ls_unregister_validates_bundle_id(monkeypatch, tmp_path):
    monkeypatch.setattr(uninstall, "_is_macos", lambda: True)
    # A real sentinel so the `Path(_LSREGISTER).exists()` guard passes.
    sentinel = tmp_path / "lsregister"
    sentinel.touch()
    monkeypatch.setattr(uninstall, "_LSREGISTER", str(sentinel))

    # One real OpenBird bundle, one unrelated app sharing the name — both exist.
    ours = tmp_path / "dist" / "OpenBird.app"
    theirs = tmp_path / "other" / "OpenBird.app"
    ours.mkdir(parents=True)
    theirs.mkdir(parents=True)
    monkeypatch.setattr(uninstall, "_registered_app_paths", lambda: [ours, theirs])
    monkeypatch.setattr(
        uninstall,
        "_bundle_id_of",
        lambda p: uninstall.BUNDLE_ID if p == ours else "com.someone.else",
    )

    unregistered: list[str] = []

    def fake_run(cmd):
        if "-u" in cmd:
            unregistered.append(cmd[-1])
        return (0, "")

    monkeypatch.setattr(uninstall, "_run", fake_run)

    results = uninstall.unregister_launch_services(dry_run=False)

    assert str(ours) in unregistered
    assert str(theirs) not in unregistered  # unrelated app never touched
    assert any(r.status == "skip" and "com.someone.else" in r.detail for r in results)


# --------------------------------------------------------------------------- #
# uninstall: launchd job classification (Codex diff #2)                        #
# --------------------------------------------------------------------------- #


def test_bootout_success_is_done(monkeypatch):
    monkeypatch.setattr(uninstall, "_run", lambda cmd: (0, ""))
    r = uninstall._boot_out_routines("gui/501/ai.openbird.routines")
    assert r.status == "done"


def test_bootout_not_loaded_is_skip(monkeypatch):
    monkeypatch.setattr(
        uninstall, "_run", lambda cmd: (3, "Boot-out failed: 3: No such process")
    )
    r = uninstall._boot_out_routines("gui/501/ai.openbird.routines")
    assert r.status == "skip"  # benign — nothing was loaded


def test_bootout_real_failure_is_error(monkeypatch):
    # Both bootout and the legacy remove fail with a non-"not loaded" message.
    monkeypatch.setattr(uninstall, "_run", lambda cmd: (1, "Operation not permitted"))
    r = uninstall._boot_out_routines("gui/501/ai.openbird.routines")
    assert r.status == "error"  # job may still be loaded -> surfaced, not hidden
