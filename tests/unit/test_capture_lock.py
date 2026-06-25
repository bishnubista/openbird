"""Unit tests for the capture single-instance lock (``openbird/capture/lock.py``).

The lock is the *authoritative* guard against two ``capture --loop`` daemons
double-capturing. These tests exercise the real kernel ``flock`` semantics:
because advisory ``flock`` attaches to the open file description (not the
process), two separate ``os.open`` calls in this one process genuinely contend,
so the single-instance behaviour is testable without spawning subprocesses.
"""

from __future__ import annotations

import errno
import fcntl
import os
import stat

import pytest

from openbird.capture import lock as lockmod
from openbird.capture.cli import CAPTURE_EXIT_ALREADY_RUNNING
from openbird.capture.lock import LOCK_FILENAME, acquire_capture_lock


def test_second_acquire_returns_none_while_held(tmp_path):
    first = acquire_capture_lock(tmp_path)
    assert first is not None
    try:
        # A second daemon must NOT get the lock — the whole point.
        assert acquire_capture_lock(tmp_path) is None
    finally:
        first.release()


def test_reacquire_after_release(tmp_path):
    first = acquire_capture_lock(tmp_path)
    assert first is not None
    first.release()
    # Once released, the next daemon can start.
    second = acquire_capture_lock(tmp_path)
    assert second is not None
    second.release()


def test_context_manager_releases(tmp_path):
    with acquire_capture_lock(tmp_path) as held:
        assert held is not None
        assert acquire_capture_lock(tmp_path) is None
    # Exiting the with-block released it.
    again = acquire_capture_lock(tmp_path)
    assert again is not None
    again.release()


def test_release_is_idempotent(tmp_path):
    held = acquire_capture_lock(tmp_path)
    assert held is not None
    held.release()
    held.release()  # must not raise / double-close


def test_lock_file_is_0600(tmp_path):
    held = acquire_capture_lock(tmp_path)
    try:
        mode = stat.S_IMODE(os.stat(tmp_path / LOCK_FILENAME).st_mode)
        assert mode == 0o600
    finally:
        held.release()


def test_lock_fd_is_cloexec(tmp_path):
    held = acquire_capture_lock(tmp_path)
    try:
        flags = fcntl.fcntl(held._fd, fcntl.F_GETFD)
        assert flags & fcntl.FD_CLOEXEC, "helper subprocess must not inherit the lock"
    finally:
        held.release()


def test_release_never_unlinks_sentinel(tmp_path):
    # Unlinking would reintroduce the inode-split race; the file must persist.
    held = acquire_capture_lock(tmp_path)
    held.release()
    assert (tmp_path / LOCK_FILENAME).exists()


def test_lock_records_pid_for_diagnostics(tmp_path):
    held = acquire_capture_lock(tmp_path)
    try:
        contents = (tmp_path / LOCK_FILENAME).read_text().strip()
        assert contents == str(os.getpid())
    finally:
        held.release()


def test_non_eagain_oserror_fails_loud(tmp_path, monkeypatch):
    # A permission/unsupported-FS error must NOT be silently treated as "held"
    # (which would make the daemon skip capture). It must propagate.
    def boom(fd, op):
        raise OSError(errno.EPERM, "operation not permitted")

    monkeypatch.setattr(lockmod.fcntl, "flock", boom)
    with pytest.raises(OSError) as exc:
        acquire_capture_lock(tmp_path)
    assert exc.value.errno == errno.EPERM


def test_eagain_maps_to_already_running_exit_code():
    # The benign "already running" path uses a dedicated, app-recognisable code.
    assert CAPTURE_EXIT_ALREADY_RUNNING == 7
