"""Regression tests for the storage crypto opener."""

from __future__ import annotations

import os
import sqlite3
import stat
import sys
import threading
import types
import time
from pathlib import Path

from openbird.config import Settings
from openbird.storage import crypto


class _FakeCursor:
    def __init__(self, rows=()):
        self._rows = list(rows)

    def fetchone(self):
        return self._rows[0] if self._rows else None


class _FakeCipherConn:
    """File-backed sqlite wrapper that behaves like the SQLCipher probes need."""

    def __init__(self, path: str):
        self._conn = sqlite3.connect(path)

    def execute(self, sql, *args):
        norm = " ".join(sql.lower().split())
        if norm.startswith("pragma key"):
            return _FakeCursor()
        if norm.startswith("pragma cipher_version"):
            return _FakeCursor([("4.12.0 test",)])
        if norm.startswith("pragma journal_mode"):
            return _FakeCursor([("wal",)])
        if norm.startswith("select vec_version()"):
            return _FakeCursor([("v0.test",)])
        return self._conn.execute(sql, *args)

    def enable_load_extension(self, _enabled):
        return None

    def close(self):
        self._conn.close()


def _mode(path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


def test_sqlcipher_path_creates_private_db_file(monkeypatch, tmp_path):
    db_path = tmp_path / "nested" / "openbird.db"

    fake_sqlcipher = types.SimpleNamespace(
        connect=lambda path, *args, **kwargs: _FakeCipherConn(path)
    )
    monkeypatch.setitem(sys.modules, "sqlcipher3", fake_sqlcipher)
    monkeypatch.setattr(crypto, "_get_or_create_key", lambda: "a" * 64)
    monkeypatch.setattr(crypto, "_load_vec", lambda _conn: None)

    settings = Settings(data_dir=tmp_path, db_path=str(db_path))
    handle = crypto.open_db_verified(settings=settings)
    try:
        assert handle.encrypted is True
        assert handle.backend == "sqlcipher"
        assert handle.cipher_version == "4.12.0 test"
        assert settings.encryption_enabled is True
        assert _mode(db_path) == 0o600
        assert _mode(db_path.parent) == 0o700
    finally:
        handle.conn.close()


def test_plaintext_path_repairs_existing_db_permissions(monkeypatch, tmp_path):
    db_path = tmp_path / "openbird.db"
    seed = sqlite3.connect(db_path)
    seed.execute("PRAGMA journal_mode = WAL")
    seed.execute("CREATE TABLE t(x)")
    seed.execute("INSERT INTO t(x) VALUES (1)")
    seed.commit()
    os.chmod(db_path, 0o644)
    sidecars = (Path(f"{db_path}-wal"), Path(f"{db_path}-shm"))
    for sidecar in sidecars:
        if sidecar.exists():
            os.chmod(sidecar, 0o644)

    monkeypatch.setattr(crypto, "_get_or_create_key", lambda: None)

    settings = Settings(data_dir=tmp_path, db_path=str(db_path))
    handle = crypto.open_db_verified(settings=settings)
    try:
        assert handle.encrypted is False
        assert handle.backend == "sqlite3"
        assert settings.encryption_enabled is False
        assert _mode(db_path) == 0o600
        for sidecar in sidecars:
            if sidecar.exists():
                assert _mode(sidecar) == 0o600
    finally:
        handle.conn.close()
        seed.close()


def test_keyring_get_timeout_returns_none(monkeypatch):
    blocked = threading.Event()

    class BlockingKeyring:
        @staticmethod
        def get_password(_service, _user):
            blocked.wait(5)
            return "late"

        @staticmethod
        def set_password(_service, _user, _key):
            raise AssertionError("set_password should not run after get timeout")

    monkeypatch.setenv("OPENBIRD_KEYRING_TIMEOUT_SECONDS", "0.01")
    monkeypatch.setitem(sys.modules, "keyring", BlockingKeyring)

    started = time.perf_counter()
    try:
        assert crypto._get_or_create_key() is None
        assert time.perf_counter() - started < 0.5
    finally:
        blocked.set()


def test_keyring_set_timeout_returns_none(monkeypatch):
    blocked = threading.Event()

    class BlockingKeyring:
        @staticmethod
        def get_password(_service, _user):
            return None

        @staticmethod
        def set_password(_service, _user, _key):
            blocked.wait(5)

    monkeypatch.setenv("OPENBIRD_KEYRING_TIMEOUT_SECONDS", "0.01")
    monkeypatch.setitem(sys.modules, "keyring", BlockingKeyring)

    started = time.perf_counter()
    try:
        assert crypto._get_or_create_key() is None
        assert time.perf_counter() - started < 0.5
    finally:
        blocked.set()
