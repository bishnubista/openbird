"""Typer command wiring for the capture feature."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console

from openbird.config import get_settings

_console = Console()
_err_console = Console(stderr=True)


def register_capture_command(app: typer.Typer) -> None:
    """Register capture commands on the root OpenBird CLI."""
    app.command(name="capture")(capture)


def capture(
    once: bool = typer.Option(
        True, "--once/--loop", help="Run a single bounded pass (default) or loop."
    ),
    max_events: int = typer.Option(
        50, "--max-events", help="Stop after this many received events."
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
        HelperUnavailableError,
    )
    from openbird.cli import _provider
    from openbird.memory.store import MemoryStore

    helper_cmd = _parse_helper_cmd(helper)
    settings = get_settings()
    # Build the cloud-checked provider [H3]: capture sends screen text to the
    # embedding model, so it must go through the same opt-in confirm + CLOUD
    # ACTIVE banner path as every other store-opening command — never silently.
    provider = _provider()
    store = MemoryStore(settings=settings, provider=provider)
    try:
        daemon = CaptureDaemon(
            store,
            settings=settings,
            helper_cmd=helper_cmd,
            require_signed_helper=not allow_unsigned,
        )
        try:
            stats = daemon.run(max_events=None if not once else max_events)
        except HelperUnavailableError as exc:
            _err_console.print(f"[red]Capture helper unavailable:[/] {exc}")
            raise typer.Exit(code=3)
    finally:
        store.close()

    _console.print(
        f"[green]Capture pass complete.[/] received={stats.received} "
        f"ingested={stats.ingested} rejected={stats.rejected} errors={stats.errors}"
    )


def _parse_helper_cmd(helper: Optional[str]) -> Optional[tuple[str, ...]]:
    """Split a helper command string into argv, or None to use the default."""
    if helper is None:
        return None
    import shlex

    parts = shlex.split(helper)
    return tuple(parts) if parts else None
