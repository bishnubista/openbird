"""Verify the OpenBird SQLCipher storage gate in a temporary workspace.

This is an install-time/developer gate, not a production migration. It proves
the real OpenBird DB opener can run SQLCipher with sqlite-vec, WAL, private file
permissions, encrypted backup behavior, disabled extension loading after setup,
and a small write/read performance smoke.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sqlite3
import stat
import tempfile
import time
from pathlib import Path
from typing import Any

from openbird.config import Settings
from openbird.storage import crypto


class GateFailure(RuntimeError):
    """Raised when an encryption-gate assertion fails."""


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return value


def _positive_float(raw: str) -> float:
    value = float(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return value


def _file_mode(path: Path) -> str:
    return oct(stat.S_IMODE(path.stat().st_mode))


def _assert_private(path: Path) -> None:
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode != 0o600:
        raise GateFailure(f"{path.name} mode is {oct(mode)}, expected 0o600")


def _assert_plain_sqlite_cannot_read(path: Path) -> None:
    try:
        conn = sqlite3.connect(path)
        try:
            conn.execute("SELECT name FROM sqlite_master").fetchall()
        finally:
            conn.close()
    except sqlite3.DatabaseError:
        return
    raise GateFailure(f"{path.name} was readable through plain sqlite3")


def _assert_extensions_disabled(conn: sqlite3.Connection, root: Path) -> None:
    missing_ext = root / "definitely_missing_extension"
    try:
        conn.load_extension(str(missing_ext))
    except Exception as exc:
        message = str(exc).lower()
        if "not authorized" in message or "disabled" in message:
            return
        raise GateFailure(
            f"extension loading is not cleanly disabled: {type(exc).__name__}: {exc}"
        ) from exc
    raise GateFailure("extension loading unexpectedly allowed a direct load")


def _open_sqlcipher(path: Path, key: str) -> sqlite3.Connection:
    try:
        import sqlcipher3  # type: ignore
    except ImportError as exc:
        raise GateFailure(
            "sqlcipher3 is missing; run with `uv run --extra encryption ...`"
        ) from exc

    conn = sqlcipher3.connect(str(path))  # type: ignore[attr-defined]
    conn.execute(f"PRAGMA key = \"x'{key}'\"")
    conn.execute("SELECT count(*) FROM sqlite_master")
    if crypto.cipher_version(conn) is None:
        conn.close()
        raise GateFailure("backup SQLCipher connection did not report cipher_version")
    return conn


def _verify_encrypted_backup(
    source: sqlite3.Connection, backup_path: Path, key: str
) -> dict[str, Any]:
    target = _open_sqlcipher(backup_path, key)
    try:
        source.backup(target)
        target.commit()
        target.execute("SELECT count(*) FROM gate_items").fetchone()
    finally:
        target.close()

    os.chmod(backup_path, 0o600)
    _assert_private(backup_path)
    _assert_plain_sqlite_cannot_read(backup_path)

    reopened = _open_sqlcipher(backup_path, key)
    try:
        (count,) = reopened.execute("SELECT count(*) FROM gate_items").fetchone()
        if count <= 0:
            raise GateFailure("encrypted backup did not preserve gate_items rows")
    finally:
        reopened.close()

    return {"path": str(backup_path), "mode": _file_mode(backup_path), "rows": count}


def run_gate(*, rows: int, max_seconds: float, keep: bool = False) -> dict[str, Any]:
    root_ctx = tempfile.TemporaryDirectory(prefix="openbird-encryption-gate-")
    root = Path(root_ctx.name)
    if keep:
        root_ctx.cleanup = lambda: None  # type: ignore[method-assign]

    key = secrets.token_hex(32)
    original_get_key = crypto._get_or_create_key
    crypto._get_or_create_key = lambda: key  # type: ignore[assignment]

    started = time.perf_counter()
    handle = None
    try:
        settings = Settings(data_dir=root)
        handle = crypto.open_db_verified(settings=settings)
        if not handle.encrypted or handle.backend != "sqlcipher" or not handle.cipher_version:
            raise GateFailure("OpenBird DB opener did not return a verified SQLCipher handle")

        db_path = Path(settings.db_path or "")
        _assert_private(db_path)
        _assert_plain_sqlite_cannot_read(db_path)
        _assert_extensions_disabled(handle.conn, root)

        journal_mode = handle.conn.execute("PRAGMA journal_mode").fetchone()[0]
        if str(journal_mode).lower() != "wal":
            raise GateFailure(f"journal mode is {journal_mode!r}, expected 'wal'")

        (vec_version,) = handle.conn.execute("SELECT vec_version()").fetchone()
        if not vec_version:
            raise GateFailure("sqlite-vec did not return a version under SQLCipher")

        handle.conn.execute("CREATE TABLE gate_items(id INTEGER PRIMARY KEY, body TEXT NOT NULL)")
        handle.conn.execute("CREATE VIRTUAL TABLE gate_fts USING fts5(body)")
        payloads = [(i, f"encrypted gate row {i}") for i in range(rows)]
        handle.conn.executemany("INSERT INTO gate_items(id, body) VALUES (?, ?)", payloads)
        handle.conn.executemany("INSERT INTO gate_fts(rowid, body) VALUES (?, ?)", payloads)
        handle.conn.commit()

        elapsed = time.perf_counter() - started
        if elapsed > max_seconds:
            raise GateFailure(f"gate workload took {elapsed:.3f}s, over {max_seconds:.3f}s")

        _assert_private(db_path)
        for suffix in ("-wal", "-shm"):
            sidecar = Path(f"{db_path}{suffix}")
            if sidecar.exists():
                _assert_private(sidecar)

        backup = _verify_encrypted_backup(handle.conn, root / "openbird-backup.db", key)
        search_hit = handle.conn.execute(
            "SELECT rowid FROM gate_fts WHERE gate_fts MATCH ? ORDER BY rowid LIMIT 1",
            ("row",),
        ).fetchone()
        if search_hit is None:
            raise GateFailure("FTS5 query returned no rows under SQLCipher")

        return {
            "ok": True,
            "workspace": str(root),
            "backend": handle.backend,
            "cipher_version": handle.cipher_version,
            "wal_enabled": True,
            "journal_mode": journal_mode,
            "sqlite_vec": vec_version,
            "db_mode": _file_mode(db_path),
            "rows": rows,
            "elapsed_seconds": round(elapsed, 4),
            "rows_per_second": round(rows / elapsed, 2) if elapsed else None,
            "backup": backup,
            "kept": keep,
        }
    finally:
        crypto._get_or_create_key = original_get_key  # type: ignore[assignment]
        if handle is not None:
            handle.conn.close()
        if not keep:
            root_ctx.cleanup()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the OpenBird SQLCipher encryption gate.")
    parser.add_argument(
        "--rows", type=_positive_int, default=1000, help="Rows to write through SQLCipher."
    )
    parser.add_argument(
        "--max-seconds",
        type=_positive_float,
        default=10.0,
        help="Maximum allowed gate runtime.",
    )
    parser.add_argument(
        "--keep", action="store_true", help="Keep the temporary workspace for inspection."
    )
    args = parser.parse_args()

    try:
        report = run_gate(rows=args.rows, max_seconds=args.max_seconds, keep=args.keep)
    except Exception as exc:
        print(
            json.dumps({"ok": False, "error": str(exc), "error_type": type(exc).__name__}, indent=2)
        )
        return 1

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
