"""Typer command wiring for the capture feature."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console

from openbird.config import get_settings

_console = Console()
_err_console = Console(stderr=True)

# Capture-daemon exit codes. 0 = clean. 3/4 are raised below for helper/supervisor
# failures. 5 means the on-disk index was built under a different embedding model
# (cohort mismatch): recoverable by `openbird reindex`. The mac app maps this code
# to an actionable "Reindex" affordance instead of the generic "stopped (exit 1)".
CAPTURE_EXIT_REINDEX_REQUIRED = 5

# Another `capture --loop` daemon already holds the single-instance lock. This is
# a BENIGN outcome, not a crash: the app may optimistically spawn a daemon that
# loses the race, and it should treat this code as "already capturing" rather than
# surfacing an error. (6 is reserved for _CAPTURE_NO_PROGRESS_EXIT below.)
CAPTURE_EXIT_ALREADY_RUNNING = 7


def register_capture_command(app: typer.Typer) -> None:
    """Register capture commands on the root OpenBird CLI."""
    app.command(name="capture")(capture)


def capture(
    once: bool = typer.Option(
        True, "--once/--loop", help="Run a single bounded pass (default) or loop."
    ),
    max_events: int = typer.Option(
        50, "--max-events", help="Stop after this many received events (--once)."
    ),
    poll_interval: float = typer.Option(
        2.0,
        "--poll-interval",
        help="Seconds between helper re-spawns in --loop mode.",
    ),
    helper: Optional[str] = typer.Option(
        None,
        "--helper",
        help="Path to a capture helper binary or emitter command. Defaults to the "
        "signed bundle path; a fake emitter can be supplied for local testing.",
    ),
    allow_unsigned: bool = typer.Option(
        False,
        "--allow-unsigned",
        help="Permit a non-bundle helper command (for testing). TCC grants are "
        "per signed path, so this is unsafe for real capture.",
    ),
    stream: bool = typer.Option(
        True,
        "--stream/--poll",
        help="--loop only: run the helper persistently (event-driven --stream, "
        "the default; auto-downgrades to polling for an old helper binary) or "
        "force the legacy one-shot polling mode.",
    ),
) -> None:
    """Run the capture daemon once (or in a loop) over the capture helper.

    The real screen reading is done by the signed Swift ``capture-helper`` that
    emits JSON capture events on stdout. By default this fails closed if the
    signed bundle is missing (TCC grants are bound to a signed path). For local
    testing you can point ``--helper`` at a fake emitter with ``--allow-unsigned``.
    """
    from openbird.capture.daemon import (
        CaptureDaemon,
        CaptureSupervisorError,
        HelperUnavailableError,
    )
    from openbird.capture.lock import acquire_capture_lock
    # Function-local imports (and routing store construction through openbird.cli._store)
    # avoid a module-level import cycle with the CLI package.
    from openbird.cli import _print_cohort_mismatch_hint, _provider, _store
    from openbird.memory.store import EmbeddingCohortMismatch

    import logging
    import signal
    import threading

    helper_cmd = _parse_helper_cmd(helper)
    settings = get_settings()
    # Single-instance guard for the long-lived --loop daemon, acquired BEFORE any
    # provider/store setup: a second daemon must lose fast and cleanly with the
    # benign "already running" code — never running the cloud-confirm prompt,
    # opening the store, or exiting with the reindex code first. --once is
    # lock-free so bounded diagnostic passes stay concurrent. The flock is
    # authoritative (kernel releases it on any death); the app's pre-spawn check
    # is only an advisory optimization.
    capture_lock = acquire_capture_lock(settings.data_dir) if not once else None
    if not once and capture_lock is None:
        _err_console.print(
            "[yellow]capture: another capture daemon is already running; "
            "exiting[/]"
        )
        raise typer.Exit(code=CAPTURE_EXIT_ALREADY_RUNNING)
    try:
        # Build the cloud-checked provider: capture sends screen text to the
        # embedding model, so it must go through the same opt-in confirm + CLOUD
        # ACTIVE banner path as every other store-opening command — never silently.
        provider = _provider()
        # `reraise_cohort_mismatch=True`: unlike one-shot commands, the daemon
        # exits with a DISTINCT code so the mac app can offer a one-click reindex
        # instead of showing a generic "stopped unexpectedly (exit 1)". Terminal
        # users still get the readable hint on stderr.
        try:
            store = _store(
                provider=provider, settings=settings, reraise_cohort_mismatch=True
            )
        except EmbeddingCohortMismatch as exc:
            _print_cohort_mismatch_hint(exc)
            raise typer.Exit(code=CAPTURE_EXIT_REINDEX_REQUIRED) from exc
        try:
            daemon = CaptureDaemon(
                store,
                settings=settings,
                helper_cmd=helper_cmd,
                require_signed_helper=not allow_unsigned,
            )
            if not stream:
                # --poll: force the legacy one-shot cadence for this run. The
                # default (--stream) keeps auto behavior: persistent unless the
                # env forces otherwise or the binary proves it can't stream.
                daemon.force_oneshot_mode()
            try:
                if once:
                    stats = daemon.run(max_events=max_events)
                else:
                    # Continuous capture: supervise the one-shot helper,
                    # re-spawning it on a cadence until SIGINT/SIGTERM requests a
                    # clean stop. The single-instance lock is already held above.
                    logging.basicConfig(
                        level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s",
                    )
                    stop = threading.Event()

                    def _handle(signum, _frame) -> None:
                        _err_console.print(
                            f"[yellow]capture: signal {signum}, shutting down[/]"
                        )
                        stop.set()

                    signal.signal(signal.SIGINT, _handle)
                    signal.signal(signal.SIGTERM, _handle)
                    _console.print(
                        "[green]capture daemon started[/] (Ctrl-C to stop)"
                    )
                    stats = daemon.run_forever(
                        poll_interval=poll_interval, stop_event=stop
                    )
            except HelperUnavailableError as exc:
                _err_console.print(f"[red]Capture helper unavailable:[/] {exc}")
                raise typer.Exit(code=3) from None
            except CaptureSupervisorError as exc:
                # Sustained helper failure tripped the circuit breaker — exit
                # nonzero so this is not mistaken for a clean session.
                _err_console.print(f"[red]Capture supervisor aborted:[/] {exc}")
                raise typer.Exit(code=4) from None
        finally:
            store.close()
    finally:
        # Release last — after the store is closed — so a waiting daemon cannot
        # grab the lock while our store is still open. Releasing only closes the
        # fd; the sentinel file is never unlinked (see capture/lock.py).
        if capture_lock is not None:
            capture_lock.release()

    _report_and_finish(stats, once=once)


# Exit code for a session that tried to capture but ingested nothing.
# Codes 3 and 4 are taken (HelperUnavailableError / CaptureSupervisorError), and
# 5 is reserved for CAPTURE_EXIT_REINDEX_REQUIRED (cohort-mismatch -> the mac app
# maps 5 to its one-click "reindex required" affordance; see AppModel
# .captureReindexExitCode). A no-progress failure is unrelated to reindex, so it
# uses 6 — otherwise the app would mis-route a broken session as "needs reindex".
_CAPTURE_NO_PROGRESS_EXIT = 6


def _report_and_finish(stats, *, once: bool) -> None:
    """Print the summary line, then exit nonzero on a clearly-failed session.

    Shared by both the ``--once`` and ``--loop`` paths so they enforce the same
    policy. The summary line is always printed (it is useful for diagnostics);
    then we raise a nonzero ``typer.Exit`` when the session tried to do work but
    produced nothing usable:

        received > 0 and ingested == 0 and errors > 0

    That is the "totally failing" signal — a daemon that saw events but ingested
    none of them while hitting errors — which previously printed "complete" and
    exited 0, so a fully broken capture session looked like success to scripts and
    launchd.

    Deliberately NOT treated as failure (these stay exit 0):
    - A clean stop that ingested fine, even if a few transient errors occurred —
      e.g. a long ``--loop`` Ctrl-C'd after ingesting (ingested > 0).
    - A quiet session that received nothing (received == 0): nothing to do.
    The signal is specifically "nothing ingested but errors happened", not
    "any error at all".
    """
    _console.print(
        f"[green]Capture {'pass' if once else 'session'} complete.[/] "
        f"received={stats.received} ingested={stats.ingested} "
        f"coalesced={stats.coalesced} rejected={stats.rejected} errors={stats.errors} "
        f"heartbeats={stats.heartbeats} afk_transitions={stats.afk_transitions} "
        f"attempts_started={stats.capture_attempts_started} "
        f"attempts_finished={stats.capture_attempts_finished}"
    )
    if stats.received > 0 and stats.ingested == 0 and stats.errors > 0:
        _err_console.print(
            "[red]Capture failed:[/] received events but ingested none "
            f"(errors={stats.errors})."
        )
        raise typer.Exit(code=_CAPTURE_NO_PROGRESS_EXIT)


def _parse_helper_cmd(helper: Optional[str]) -> Optional[tuple[str, ...]]:
    """Split a helper command string into argv, or None to use the default."""
    if helper is None:
        return None
    import shlex

    parts = shlex.split(helper)
    return tuple(parts) if parts else None
