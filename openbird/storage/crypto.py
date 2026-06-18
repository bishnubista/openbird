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
# Wait up to this long for a busy DB lock before raising "database is locked"
# (B3): a concurrent reader/writer must queue, not fail immediately, while the
# single WAL writer slot is briefly held by an INSERT-only transaction.
_BUSY_TIMEOUT_MS = 5000
# Auto-checkpoint the WAL back into the main DB roughly every this-many pages so
# the -wal sidecar cannot grow without bound (H10). 1000 pages is SQLite's
# default; we set it explicitly so the bound is documented and deterministic.
_WAL_AUTOCHECKPOINT_PAGES = 1000


class EncryptionUnavailableError(RuntimeError):
    """Raised when strict encryption is required but cannot be verified.

    Surfaced only when ``OPENBIRD_REQUIRE_ENCRYPTION`` is enabled (or
    ``settings.require_encryption`` is True) and the SQLCipher path could not be
    verified — instead of silently degrading to a plaintext DB (H2). The message
    distinguishes *why* encryption was unavailable (missing deps, key
    unavailable, or an unverifiable cipher) so the failure is actionable.
    """


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
    under SQLCipher per the encryption gate (PLAN.md:84).
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
        # Queue on a busy lock rather than failing immediately (B3).
        conn.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
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
        # Bound the WAL sidecar so it cannot grow without limit (H10).
        conn.execute(f"PRAGMA wal_autocheckpoint = {_WAL_AUTOCHECKPOINT_PAGES}")
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


def _open_plaintext(path: str) -> tuple[sqlite3.Connection, bool]:
    """Open a plain sqlite3 DB with 0600 file perms and sqlite-vec loaded.

    Returns the connection plus a live-probed ``wal_enabled`` flag. WAL is
    REQUIRED for the local-first concurrency contract (B3): a reader must not be
    blocked by — nor error against — the single short INSERT-only writer txn.
    ``busy_timeout`` is set so a contended lock queues instead of failing.
    """
    _prepare_private_db_path(path)
    conn = sqlite3.connect(path)
    # Touch was implicit on connect; tighten perms on first creation and every
    # subsequent open so older DBs are repaired in place.
    _chmod_private_db_files(path)
    _load_vec(conn)
    # Queue on a busy lock instead of erroring immediately (B3).
    conn.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
    # Enable WAL on the plaintext path too (previously hard-coded off): readers
    # and the writer can then proceed concurrently. Probe the LIVE mode rather
    # than assuming it engaged.
    conn.execute("PRAGMA journal_mode = WAL")
    row = conn.execute("PRAGMA journal_mode").fetchone()
    wal_enabled = bool(row) and str(row[0]).lower() == "wal"
    if wal_enabled:
        # Bound the WAL sidecar so it cannot grow without limit (H10).
        conn.execute(f"PRAGMA wal_autocheckpoint = {_WAL_AUTOCHECKPOINT_PAGES}")
    else:
        logger.warning(
            "plaintext DB could not enable WAL (journal_mode=%s); "
            "concurrent reads during writes may block",
            (row[0] if row else "unknown"),
        )
    # Re-tighten perms now that the -wal/-shm sidecars exist.
    _chmod_private_db_files(path)
    return conn, wal_enabled


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

    require_encryption = bool(getattr(settings, "require_encryption", False))

    # Track WHY encryption was unavailable so strict mode can raise an actionable
    # error distinguishing missing deps / locked-or-missing key / unverifiable
    # cipher (H2) — rather than a vague "encryption unavailable".
    reason = "no key available (keyring/Keychain unavailable, locked, or timed out)"
    try:
        import sqlcipher3  # type: ignore  # noqa: F401
    except ImportError:
        reason = "sqlcipher3 is not installed"

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
            reason = "SQLCipher opened but its encryption could not be verified"
        else:
            # _try_sqlcipher already logged the specific failure; the key existed
            # so deps/key were not the blocker.
            reason = "the SQLCipher backend was unusable (see logs for the cause)"

    if require_encryption:
        # H2 strict mode: refuse to silently create/open a PLAINTEXT database.
        raise EncryptionUnavailableError(
            "OPENBIRD_REQUIRE_ENCRYPTION is set but the database could not be "
            f"opened with verified SQLCipher encryption: {reason}. "
            "Refusing to fall back to a plaintext database. Install the "
            "'encryption' extra (sqlcipher3 + keyring) and ensure the Keychain "
            "is unlocked, or unset OPENBIRD_REQUIRE_ENCRYPTION to allow the "
            "0600 plaintext fallback."
        )

    conn, wal_enabled = _open_plaintext(db_path)
    settings.encryption_enabled = False
    logger.warning(
        "DB opened WITHOUT app-level encryption (local-only, 0600 perms, "
        "WAL=%s; relies on FileVault for at-rest protection). Cause: %s. "
        "Install the 'encryption' extra (sqlcipher3 + keyring) to enable "
        "SQLCipher, or set OPENBIRD_REQUIRE_ENCRYPTION=1 to refuse plaintext.",
        wal_enabled,
        reason,
    )
    return DbHandle(
        conn=conn,
        backend="sqlite3",
        encrypted=False,
        cipher_version=None,
        wal_enabled=wal_enabled,
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
    "EncryptionUnavailableError",
    "cipher_version",
    "mapping_row_factory",
    "open_db_verified",
    "open_encrypted_db",
]
