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
    # Function-local imports (and routing store construction through openbird.cli._store)
    # avoid a module-level import cycle with the CLI package.
    from openbird.cli import _print_cohort_mismatch_hint, _provider, _store
    from openbird.memory.store import EmbeddingCohortMismatch

    import logging
    import signal
    import threading

    helper_cmd = _parse_helper_cmd(helper)
    settings = get_settings()
    # Build the cloud-checked provider: capture sends screen text to the
    # embedding model, so it must go through the same opt-in confirm + CLOUD
    # ACTIVE banner path as every other store-opening command — never silently.
    provider = _provider()
    # `reraise_cohort_mismatch=True`: unlike one-shot commands, the daemon exits
    # with a DISTINCT code so the mac app can offer a one-click reindex instead of
    # showing a generic "stopped unexpectedly (exit 1)". Terminal users still get
    # the readable hint on stderr.
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
        try:
            if once:
                stats = daemon.run(max_events=max_events)
            else:
                # Continuous capture: supervise the one-shot helper, re-spawning
                # it on a cadence until SIGINT/SIGTERM requests a clean stop.
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
                _console.print("[green]capture daemon started[/] (Ctrl-C to stop)")
                stats = daemon.run_forever(
                    poll_interval=poll_interval, stop_event=stop
                )
        except HelperUnavailableError as exc:
            _err_console.print(f"[red]Capture helper unavailable:[/] {exc}")
            raise typer.Exit(code=3) from None
        except CaptureSupervisorError as exc:
            # Sustained helper failure tripped the circuit breaker — exit nonzero
            # so this is not mistaken for a clean session.
            _err_console.print(f"[red]Capture supervisor aborted:[/] {exc}")
            raise typer.Exit(code=4) from None
    finally:
        store.close()

    _console.print(
        f"[green]Capture {'pass' if once else 'session'} complete.[/] "
        f"received={stats.received} ingested={stats.ingested} "
        f"coalesced={stats.coalesced} rejected={stats.rejected} errors={stats.errors}"
    )


def _parse_helper_cmd(helper: Optional[str]) -> Optional[tuple[str, ...]]:
    """Split a helper command string into argv, or None to use the default."""
    if helper is None:
        return None
    import shlex

    parts = shlex.split(helper)
    return tuple(parts) if parts else None
