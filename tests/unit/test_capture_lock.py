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
import typer

import openbird.cli as cli_mod
from openbird.capture import lock as lockmod
from openbird.capture.cli import CAPTURE_EXIT_ALREADY_RUNNING, capture
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


def test_loop_acquires_lock_before_provider_and_store(tmp_path, monkeypatch):
    # Contract: a second `--loop` daemon must lose the lock and exit code 7
    # BEFORE building the provider (cloud-confirm prompt) or opening the store
    # (which could otherwise exit with the reindex code). So with the lock already
    # held, neither `_provider` nor `_store` may be called.
    monkeypatch.setenv("OPENBIRD_DATA_DIR", str(tmp_path))

    def _provider_must_not_run(*_a, **_k):
        raise AssertionError("provider must not be built while the lock is held")

    def _store_must_not_run(*_a, **_k):
        raise AssertionError("store must not be opened while the lock is held")

    monkeypatch.setattr(cli_mod, "_provider", _provider_must_not_run)
    monkeypatch.setattr(cli_mod, "_store", _store_must_not_run)

    held = acquire_capture_lock(tmp_path)
    try:
        with pytest.raises(typer.Exit) as exc:
            capture(
                once=False,
                max_events=50,
                poll_interval=2.0,
                helper=None,
                allow_unsigned=False,
            )
        assert exc.value.exit_code == CAPTURE_EXIT_ALREADY_RUNNING
    finally:
        held.release()


def test_once_pass_is_lock_free(tmp_path, monkeypatch):
    # A bounded --once pass must NOT take the lock (diagnostics run concurrently
    # with the daemon). Hold the lock, then assert --once still proceeds to build
    # the provider rather than exiting 7. We stop it right after that proof.
    monkeypatch.setenv("OPENBIRD_DATA_DIR", str(tmp_path))

    class _Reached(Exception):
        pass

    def _provider_reached(*_a, **_k):
        raise _Reached()

    monkeypatch.setattr(cli_mod, "_provider", _provider_reached)

    held = acquire_capture_lock(tmp_path)
    try:
        # --once ignores the held lock and proceeds far enough to build the
        # provider (our sentinel), proving it never consulted the lock.
        with pytest.raises(_Reached):
            capture(
                once=True,
                max_events=1,
                poll_interval=2.0,
                helper=None,
                allow_unsigned=False,
            )
    finally:
        held.release()
