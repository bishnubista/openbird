"""Resolve a prompt's persona override from env or the on-disk prompts dir.

A user may override a prompt's *persona* (its tone / answering behavior) without
touching the framework-owned security scaffold (see :mod:`openbird.prompts.core`).
Two override channels, in precedence order:

1. ``OPENBIRD_PROMPT_<KEY>`` env var — ALWAYS inline persona text (never a path),
   so there is no "is this a file or a string?" ambiguity. An unset/empty var is
   "no override".
2. ``<prompts_dir>/<key>.txt`` — a file whose entire contents are the persona.

The prompts dir is user-writable, so the file read is **TOCTOU-safe**: a single
``os.open`` with ``O_NOFOLLOW`` (a symlink fails the open) followed by ``fstat``
on the *descriptor* — never a pre-check on the path that could be swapped before
the read. Size and regular-file checks are enforced against the open fd.

:func:`resolve_persona` returns a :class:`PersonaResolution` diagnostic, not a
bare string, so the ``prompts validate`` CLI can distinguish "no override" from
"override present but refused" (and *why*) without parsing logs. The runtime just
reads ``.persona`` (``None`` => use the spec default), so a bad override degrades
to the default prompt instead of breaking chat.
"""

from __future__ import annotations

import errno
import os
import stat
from dataclasses import dataclass
from pathlib import Path

# Max persona file size. Personas are short instruction blocks; a multi-MB file
# is a mistake or an attack, not a prompt. Enforced from the open descriptor.
MAX_PERSONA_BYTES = 64 * 1024


@dataclass(frozen=True)
class PersonaResolution:
    """The outcome of resolving one prompt's persona override.

    ``persona is None`` means "use the spec default" — either because no override
    exists (``ok=True``) or because an override was present but refused
    (``ok=False`` with a ``reason`` code). ``source`` is where the resolver
    looked: ``"env"``, ``"file"``, or ``"default"``.
    """

    key: str
    source: str
    persona: str | None
    path: Path | None
    ok: bool
    reason: str


def _env_key(key: str) -> str:
    """Return the inline-override env var name for a prompt key."""
    return f"OPENBIRD_PROMPT_{key.upper()}"


def resolve_persona(key: str, *, prompts_dir: Path | str) -> PersonaResolution:
    """Resolve ``key``'s persona override (env > file > default)."""
    # 1. Env override — always inline text. An unset/empty var is "no override".
    raw_env = os.environ.get(_env_key(key))
    if raw_env:
        if not raw_env.strip():
            return PersonaResolution(key, "env", None, None, False, "empty")
        return PersonaResolution(key, "env", raw_env, None, True, "")

    # 2. File override — TOCTOU-safe read of <prompts_dir>/<key>.txt.
    path = Path(prompts_dir) / f"{key}.txt"
    file_res = _read_persona_file(key, path)
    if file_res is not None:
        return file_res

    # 3. No override.
    return PersonaResolution(key, "default", None, None, True, "")


def _read_persona_file(key: str, path: Path) -> PersonaResolution | None:
    """Read a persona file safely; ``None`` means "no override file present".

    Returns a refused :class:`PersonaResolution` (``ok=False``) for a symlink,
    non-regular file, oversized file, invalid UTF-8, or empty content.
    """
    # O_NONBLOCK is critical: opening a FIFO O_RDONLY otherwise BLOCKS until a
    # writer appears, so a malicious/accidental named pipe at <key>.txt would hang
    # the chat process forever. For a regular file O_NONBLOCK is a no-op; for a
    # FIFO/device it returns immediately so fstat can reject it as not-regular.
    try:
        fd = os.open(
            path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
        )
    except FileNotFoundError:
        return None  # genuinely absent -> no override
    except OSError as exc:
        # O_NOFOLLOW makes a symlink fail with ELOOP; anything else is unreadable.
        reason = "symlink" if exc.errno == errno.ELOOP else "unreadable"
        return PersonaResolution(key, "file", None, path, False, reason)

    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            return PersonaResolution(key, "file", None, path, False, "not-regular")
        # Clear O_NONBLOCK now that we know it is a regular file, so reads behave
        # normally (a regular-file read never blocks regardless, but keep flags clean).
        os.set_blocking(fd, True)
        data = _read_capped(fd, MAX_PERSONA_BYTES + 1)
    finally:
        os.close(fd)

    if len(data) > MAX_PERSONA_BYTES:
        return PersonaResolution(key, "file", None, path, False, "too-large")
    try:
        text = data.decode("utf-8")  # strict
    except UnicodeDecodeError:
        return PersonaResolution(key, "file", None, path, False, "utf8")
    if not text.strip():
        return PersonaResolution(key, "file", None, path, False, "empty")
    return PersonaResolution(key, "file", text, path, True, "")


def _read_capped(fd: int, limit: int) -> bytes:
    """Read up to ``limit`` bytes from ``fd`` (os.read may short-read)."""
    chunks: list[bytes] = []
    remaining = limit
    while remaining > 0:
        chunk = os.read(fd, remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)
