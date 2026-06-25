"""Single-instance lock for the continuous capture daemon.

Only ONE ``openbird capture --loop`` daemon may run at a time; two would
double-capture the same screen events into the store (observed in the wild as
twin capture sessions with identical timestamps). We enforce single-instance with
an advisory :func:`fcntl.flock` on ``<data_dir>/capture.lock``.

Why ``flock`` and not a hand-rolled PID file: the kernel releases an advisory
lock automatically when the holding process dies by *any* means — clean exit,
SIGTERM, SIGKILL, or crash. That removes the stale-PID problem entirely (the exact
failure that orphaned a daemon and caused double-capture): a dead holder leaves no
live lock behind, only an empty sentinel file the next daemon simply re-locks.

The lock file is a PERMANENT sentinel and is **never unlinked**. Unlinking would
open an inode-split race: daemon A could still hold a lock on the now-unlinked
inode while daemon C creates and locks a *fresh* file at the same path — two
inodes, two "held" locks, two daemons capturing. Releasing the lock therefore
means closing the fd and nothing more.

The lock applies ONLY to the long-lived ``--loop`` daemon. Bounded ``--once``
passes are intentionally lock-free so the app can run diagnostics concurrently.

Filesystem assumption: ``data_dir`` is local (the default ``~/.openbird`` on
APFS). Advisory ``flock`` is unreliable on networked/exotic mounts; those surface
here as a raised ``OSError`` (fail loud) rather than a false "already running".
"""

from __future__ import annotations

import errno
import fcntl
import os
from pathlib import Path

LOCK_FILENAME = "capture.lock"


class CaptureLock:
    """A held single-instance lock. Release by closing the underlying fd.

    Obtain instances via :func:`acquire_capture_lock`; usable as a context
    manager so the lock is released on scope exit.
    """

    def __init__(self, fd: int, path: Path) -> None:
        self._fd = fd
        self.path = path

    def release(self) -> None:
        """Release the lock (idempotent). Closes the fd; never unlinks the file."""
        if self._fd < 0:
            return
        fd, self._fd = self._fd, -1
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def __enter__(self) -> "CaptureLock":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()


def acquire_capture_lock(data_dir: Path) -> CaptureLock | None:
    """Try to acquire the exclusive capture-daemon lock under ``data_dir``.

    Returns a held :class:`CaptureLock` on success, or ``None`` if another daemon
    already holds it (``EAGAIN``/``EWOULDBLOCK``).

    Any *other* ``OSError`` (permission denied, a filesystem that does not support
    advisory locks, a broken data dir) is re-raised: we must fail loudly rather
    than silently skip capture or mistake an unrelated error for "already running".
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / LOCK_FILENAME
    # O_CLOEXEC: an exec'd capture-helper subprocess must never inherit and hold
    # the lock. subprocess.Popen also defaults to close_fds=True on POSIX; this is
    # belt-and-suspenders. fchmod pins 0600 regardless of the process umask.
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
    try:
        os.fchmod(fd, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        os.close(fd)
        if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
            return None
        raise
    # Record our PID for human diagnostics only — correctness comes from the
    # flock, never these bytes. Best-effort: a write failure must not drop a
    # lock we successfully hold.
    try:
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode())
        os.fsync(fd)
    except OSError:
        pass
    return CaptureLock(fd, path)
