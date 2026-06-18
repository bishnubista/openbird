"""Encrypted-at-rest DB connection with a documented plaintext fallback.

Target: whole-DB encryption via SQLCipher (``sqlcipher3``) with the key stored
in the macOS Keychain via ``keyring`` — so the FTS index and vectors are also
encrypted. If SQLCipher or sqlite-vec-under-SQLCipher is unavailable, we fall
back to a plain ``sqlite3`` database with ``0600`` permissions and set
``settings.encryption_enabled = False`` (the honest "local-only, not yet
app-encrypted" state). The chosen result is logged, never silently faked.
"""

from __future__ import annotations

import logging
import os
import queue
import secrets
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sqlite_vec

from openbird.config import Settings, get_settings

logger = logging.getLogger("openbird.storage.crypto")

_KEYRING_SERVICE = "openbird"
_KEYRING_USER = "db-encryption-key"
_PRIVATE_FILE_MODE = 0o600
_PRIVATE_DIR_MODE = 0o700
_DEFAULT_KEYRING_TIMEOUT_SECONDS = 2.0
_KEYRING_TIMEOUT = object()


@dataclass(frozen=True)
class DbHandle:
    """An open DB connection plus explicit, verified backend metadata.

    ``backend`` is ``"sqlcipher"`` only when a live ``PRAGMA cipher_version`` on
    this connection returned a non-empty value (i.e. the connection is genuinely
    encrypted), otherwise ``"sqlite3"``. ``cipher_version`` carries the proof
    string (or ``None``). ``encrypted`` is the single source of truth preflight
    keys off of — never a bare settings flag.
    """

    conn: sqlite3.Connection
    backend: str
    encrypted: bool
    cipher_version: str | None = None
    wal_enabled: bool = False


def cipher_version(conn: sqlite3.Connection) -> str | None:
    """Return ``PRAGMA cipher_version`` for ``conn``, or ``None`` if not SQLCipher.

    A plain ``sqlite3`` connection does not recognize this pragma and returns no
    row (or raises), so the result is the authoritative signal that the live
    connection is backed by SQLCipher. Never raises.
    """
    try:
        row = conn.execute("PRAGMA cipher_version").fetchone()
    except Exception:
        return None
    if not row:
        return None
    value = row[0]
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def mapping_row_factory(cursor: Any, row: tuple[Any, ...]) -> dict[str, Any]:
    """Return dict rows for both sqlite3 and sqlcipher3 cursors."""
    return {column[0]: row[index] for index, column in enumerate(cursor.description)}


def _load_vec(conn: sqlite3.Connection) -> None:
    """Load the sqlite-vec extension into ``conn``."""
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)


def _prepare_private_db_path(path: str) -> None:
    """Ensure the DB parent exists privately before SQLite creates files."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(p.parent, _PRIVATE_DIR_MODE)
    except OSError as exc:
        logger.warning("could not set private permissions on DB parent %s: %s", p.parent, exc)


def _chmod_private_db_files(path: str) -> None:
    """Tighten SQLite DB and WAL sidecar files to owner read/write only."""
    for candidate in (Path(path), Path(f"{path}-wal"), Path(f"{path}-shm")):
        if not candidate.exists():
            continue
        try:
            os.chmod(candidate, _PRIVATE_FILE_MODE)
        except OSError as exc:
            logger.warning("could not set private permissions on DB file %s: %s", candidate, exc)


def _keyring_timeout_seconds() -> float:
    raw = os.environ.get("OPENBIRD_KEYRING_TIMEOUT_SECONDS")
    if raw is None:
        return _DEFAULT_KEYRING_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError:
        logger.warning("invalid OPENBIRD_KEYRING_TIMEOUT_SECONDS=%r; using default", raw)
        return _DEFAULT_KEYRING_TIMEOUT_SECONDS
    if value <= 0:
        logger.warning("OPENBIRD_KEYRING_TIMEOUT_SECONDS must be positive; using default")
        return _DEFAULT_KEYRING_TIMEOUT_SECONDS
    return value


def _call_keyring(label: str, func: Any, *args: Any) -> Any:
    """Call a keyring function with a timeout so macOS prompts cannot hang CLI."""
    timeout = _keyring_timeout_seconds()
    result: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

    def worker() -> None:
        try:
            result.put((True, func(*args)))
        except Exception as exc:
            result.put((False, exc))

    thread = threading.Thread(
        target=worker,
        name=f"openbird-keyring-{label}",
        daemon=True,
    )
    thread.start()
    try:
        ok, value = result.get(timeout=timeout)
    except queue.Empty:
        logger.warning("keyring %s timed out after %.2fs; cannot encrypt", label, timeout)
        return _KEYRING_TIMEOUT
    if ok:
        return value
    raise value


def _get_or_create_key() -> str | None:
    """Fetch the DB key from the Keychain, creating one if needed.

    Returns ``None`` if ``keyring`` is unavailable, signalling that encryption
    cannot proceed.
    """
    try:
        import keyring
    except ImportError:
        return None

    try:
        key = _call_keyring("get_password", keyring.get_password, _KEYRING_SERVICE, _KEYRING_USER)
        if key is _KEYRING_TIMEOUT:
            return None
        if key is None:
            key = secrets.token_hex(32)
            stored = _call_keyring(
                "set_password",
                keyring.set_password,
                _KEYRING_SERVICE,
                _KEYRING_USER,
                key,
            )
            if stored is _KEYRING_TIMEOUT:
                return None
        return key
    except Exception as exc:  # keyring backend errors, locked keychain, etc.
        logger.warning("keyring unavailable for DB key (%s); cannot encrypt", type(exc).__name__)
        return None


def _try_sqlcipher(path: str, key: str) -> sqlite3.Connection | None:
    """Attempt to open an encrypted SQLCipher connection with sqlite-vec.

    Returns the connection on success, or ``None`` if SQLCipher is unavailable,
    sqlite-vec fails to load under it, or the connection cannot be *verified* as
    genuinely encrypted via ``PRAGMA cipher_version``. WAL must also engage
    under SQLCipher per the encryption gate.
    """
    try:
        import sqlcipher3  # type: ignore
    except ImportError:
        return None

    conn: sqlite3.Connection | None = None
    try:
        _prepare_private_db_path(path)
        conn = sqlcipher3.connect(path)  # type: ignore[attr-defined]
        _chmod_private_db_files(path)
        # PRAGMA key must run before any other operation on the DB.
        conn.execute(f"PRAGMA key = \"x'{key}'\"")
        conn.execute("SELECT count(*) FROM sqlite_master")  # force key check
        # Verify this is really SQLCipher on the LIVE connection, not a plain
        # sqlite3 that happened to swallow the key pragma.
        if cipher_version(conn) is None:
            logger.warning("opened DB reports no cipher_version; not SQLCipher")
            conn.close()
            return None
        _load_vec(conn)
        conn.execute("SELECT vec_version()")
        # Encryption gate also requires WAL behavior under SQLCipher.
        row = conn.execute("PRAGMA journal_mode = WAL").fetchone()
        mode = str(row[0]).lower() if row and row[0] is not None else ""
        if mode != "wal":
            raise RuntimeError(f"SQLCipher WAL mode unavailable: {mode or 'unknown'}")
        _chmod_private_db_files(path)
        return conn
    except Exception as exc:
        logger.warning("SQLCipher path unusable (%s); falling back to plaintext", type(exc).__name__)
        # Don't leak the connection if a later step (sqlite-vec, WAL) failed.
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        return None


def _open_plaintext(path: str) -> sqlite3.Connection:
    """Open a plain sqlite3 DB with 0600 file perms and sqlite-vec loaded."""
    _prepare_private_db_path(path)
    conn = sqlite3.connect(path)
    # Touch was implicit on connect; tighten perms on first creation and every
    # subsequent open so older DBs are repaired in place.
    _chmod_private_db_files(path)
    _load_vec(conn)
    return conn


def open_db_verified(
    path: str | None = None,
    *,
    settings: Settings | None = None,
) -> DbHandle:
    """Open the OpenBird DB and return a :class:`DbHandle` with verified metadata.

    Tries SQLCipher + sqlite-vec with a Keychain-stored key, *verifying*
    encryption on the live connection (``PRAGMA cipher_version`` + WAL). If that
    stack is unavailable or unverifiable, falls back to plain ``sqlite3`` with
    ``0600`` permissions. ``settings.encryption_enabled`` is set to match the
    *verified* outcome — never optimistically. The outcome is logged.

    Args:
        path: DB file path; defaults to ``settings.db_path``.
        settings: Settings to read paths from and record the encryption result
            into; defaults to :func:`get_settings`.

    Returns:
        A :class:`DbHandle` whose ``encrypted``/``backend`` reflect a live probe.
    """
    settings = settings or get_settings()
    db_path = path or settings.db_path
    assert db_path is not None

    key = _get_or_create_key()
    if key is not None:
        conn = _try_sqlcipher(db_path, key)
        if conn is not None:
            ver = cipher_version(conn)
            # Defensive: only call it encrypted if the live probe still confirms.
            if ver is not None:
                settings.encryption_enabled = True
                logger.info("DB opened with SQLCipher encryption at rest (%s)", ver)
                return DbHandle(
                    conn=conn,
                    backend="sqlcipher",
                    encrypted=True,
                    cipher_version=ver,
                    wal_enabled=True,
                )
            conn.close()

    conn = _open_plaintext(db_path)
    settings.encryption_enabled = False
    logger.info(
        "DB opened WITHOUT app-level encryption (local-only, 0600 perms; "
        "relies on FileVault). Install sqlcipher3 + keyring to enable SQLCipher."
    )
    return DbHandle(
        conn=conn,
        backend="sqlite3",
        encrypted=False,
        cipher_version=None,
        wal_enabled=False,
    )


def open_encrypted_db(
    path: str | None = None,
    *,
    settings: Settings | None = None,
) -> sqlite3.Connection:
    """Open the OpenBird database, returning just the connection.

    Thin wrapper over :func:`open_db_verified` for callers that only need the
    connection (e.g. :class:`~openbird.memory.store.MemoryStore`). Encryption is
    still verified on the live connection and recorded in ``settings``.

    Returns:
        An open connection with sqlite-vec loaded.
    """
    return open_db_verified(path, settings=settings).conn


__all__ = [
    "DbHandle",
    "cipher_version",
    "mapping_row_factory",
    "open_db_verified",
    "open_encrypted_db",
]
