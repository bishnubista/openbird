"""OpenBird command-line interface (Typer).

Wires the frozen subsystems into a single ``openbird`` entrypoint:

    openbird preflight            # readiness report (TCC, ollama, sqlite, crypto)
    openbird ingest <path>        # ingest a file or directory into memory
    openbird chat "<question>"    # grounded RAG answer with occurrence citations
    openbird capture [--once]     # run the capture daemon over the signed helper
    openbird routine list         # list built-in routine templates
    openbird routine run <name>   # run one routine occurrence now
    openbird meeting              # meetings stub (manual-record, gated subsystem)
    openbird data purge --since   # cascade-delete observations since a timestamp

Design notes
------------
* The CLI is the serialized integration point (PLAN Phase 3): it imports the
  shared contracts and never reaches inside subsystem internals.
* Captured content is treated carefully: ``chat`` prints answers and citation
  *metadata* (app/window/time + a short snippet) but the command surface never
  dumps raw blobs to logs.
* Heavy/optional pieces (the LLM provider, transcription) are imported lazily so
  ``preflight`` and ``--help`` work even when Ollama or the meetings extra are
  unavailable.
"""

from __future__ import annotations

import datetime as _dt
import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from openbird.capture.cli import register_capture_command
from openbird.config import get_settings

app = typer.Typer(
    name="openbird",
    help="OpenBird — local-first, open-source personal memory for macOS.",
    no_args_is_help=True,
    add_completion=False,
)

routine_app = typer.Typer(help="Manage and run scheduled routines.", no_args_is_help=True)
app.add_typer(routine_app, name="routine")
register_capture_command(app)

_console = Console()
_err_console = Console(stderr=True)


# --------------------------------------------------------------------------- #
# Lazily-constructed shared services                                          #
# --------------------------------------------------------------------------- #


def _provider():
    """Construct the configured LLM provider (imported lazily)."""
    from openbird.llm.provider import create_llm_provider

    return create_llm_provider(get_settings())


def _store(*, provider=None):
    """Open the on-disk :class:`MemoryStore` (imported lazily)."""
    from openbird.memory.store import MemoryStore

    return MemoryStore(settings=get_settings(), provider=provider)


# --------------------------------------------------------------------------- #
# preflight                                                                   #
# --------------------------------------------------------------------------- #


@app.command()
def preflight(
    json_out: bool = typer.Option(
        False, "--json", help="Emit the raw report as JSON instead of a table."
    ),
    probe_embedding: bool = typer.Option(
        False, "--probe-embedding", help="Request a real embedding to confirm dim."
    ),
    no_ollama: bool = typer.Option(
        False, "--no-ollama", help="Skip the Ollama network probe."
    ),
) -> None:
    """Report whether this environment can actually run OpenBird.

    Aggregates Ollama reachability + required models, the embedding dimension,
    sqlite-vec/FTS5 availability, DB encryption status, the active allow/block
    lists, and (on macOS) TCC/audio capability gates. Never raises: every check
    degrades to ``unknown`` rather than crashing.
    """
    from openbird.preflight import run_preflight

    report = run_preflight(
        get_settings(),
        probe_ollama=not no_ollama,
        probe_embedding=probe_embedding,
    )

    if json_out:
        _console.print_json(json.dumps(report))
        raise typer.Exit(code=0 if report.get("runtime_ok") else 1)

    _render_preflight(report)
    raise typer.Exit(code=0 if report.get("runtime_ok") else 1)


def _render_preflight(report: dict) -> None:
    """Pretty-print a preflight report as a status table."""
    table = Table(title="OpenBird preflight", show_header=True, header_style="bold")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")

    ollama = report["ollama"]
    reachable = ollama.get("reachable")
    missing = ollama.get("missing_models")
    if reachable is True:
        o_status = "ok" if not missing else "degraded"
        o_detail = (
            "all required models present"
            if not missing
            else f"missing: {', '.join(missing)}"
        )
    elif reachable == "unknown":
        o_status, o_detail = "unknown", "probe skipped"
    else:
        o_status, o_detail = "down", f"unreachable ({ollama.get('error')})"
    table.add_row("ollama", o_status, f"{ollama.get('host')} · {o_detail}")

    emb = report["embedding"]
    emb_detail = f"{emb['model']} dim={emb['configured_dim']}"
    if emb.get("probed"):
        emb_detail += f" probed={emb.get('probed_dim')} ok={emb.get('dim_ok')}"
    table.add_row("embedding", "info", emb_detail)

    sq = report["sqlite"]
    sq_ok = sq.get("vec_available") and sq.get("fts5_available")
    table.add_row(
        "sqlite",
        "ok" if sq_ok else "fail",
        f"vec={sq.get('vec_available')} fts5={sq.get('fts5_available')} "
        f"(sqlite {sq.get('sqlite_version')})",
    )

    enc = report["encryption"]
    table.add_row(
        "encryption",
        str(enc.get("status")),
        f"backend={enc.get('backend')} verified={enc.get('verified')}",
    )

    priv = report["privacy"]
    table.add_row(
        "privacy",
        "info",
        f"allowlist={len(priv['allowlist'])} blocklist={len(priv['blocklist'])} "
        f"ocr={priv['ocr_enabled']}",
    )

    mac = report["macos"]
    if mac.get("is_macos"):
        mac_status = "ok" if mac.get("all_passed") else "unknown"
        mac_detail = (
            f"ax={mac['accessibility']} screen={mac['screen_recording']} "
            f"mic={mac['microphone']} sysaudio={mac['system_audio']}"
        )
    else:
        mac_status, mac_detail = "n/a", "not macOS"
    table.add_row("macos", mac_status, mac_detail)

    _console.print(table)
    summary = "READY" if report.get("runtime_ok") else "NOT READY"
    release = "release-gate OK" if report.get("release_gate_ok") else "release-gate not met"
    style = "bold green" if report.get("runtime_ok") else "bold yellow"
    _console.print(f"[{style}]{summary}[/] — runtime checks; {release}.")


# --------------------------------------------------------------------------- #
# ingest                                                                      #
# --------------------------------------------------------------------------- #


@app.command()
def ingest(
    path: Path = typer.Argument(..., help="A text file or directory to ingest."),
    glob: str = typer.Option(
        "*", "--glob", help="When PATH is a directory, only ingest files matching this glob."
    ),
    max_bytes: int = typer.Option(
        1_000_000, "--max-bytes", help="Skip files larger than this many bytes."
    ),
) -> None:
    """Ingest a file (or every matching file in a directory) into memory.

    Each file becomes one observation tagged ``source="ingest"`` with the file
    name as the window and a ``file://`` URL. Text is normalized + chunk-level
    deduped + embedded by the store.
    """
    path = path.expanduser()
    if not path.exists():
        _err_console.print(f"[red]No such path:[/] {path}")
        raise typer.Exit(code=2)

    files = _collect_files(path, glob=glob, max_bytes=max_bytes)
    if not files:
        _err_console.print("[yellow]No matching files to ingest.[/]")
        raise typer.Exit(code=1)

    store = _store()
    ingested = 0
    skipped = 0
    try:
        for fp in files:
            try:
                text = fp.read_text(encoding="utf-8", errors="replace")
            except OSError:
                skipped += 1
                continue
            if not text.strip():
                skipped += 1
                continue
            store.add_observation(
                text,
                app="ingest",
                window=fp.name,
                url=fp.resolve().as_uri(),
                source="ingest",
            )
            ingested += 1
    finally:
        store.close()

    _console.print(
        f"[green]Ingested[/] {ingested} file(s); skipped {skipped}."
    )


def _collect_files(path: Path, *, glob: str, max_bytes: int) -> list[Path]:
    """Resolve PATH to a sorted list of regular files under the size cap."""
    if path.is_file():
        candidates = [path]
    else:
        candidates = sorted(p for p in path.rglob(glob) if p.is_file())
    out: list[Path] = []
    for p in candidates:
        try:
            if p.stat().st_size > max_bytes:
                continue
        except OSError:
            continue
        out.append(p)
    return out


# --------------------------------------------------------------------------- #
# chat                                                                        #
# --------------------------------------------------------------------------- #


@app.command()
def chat(
    question: str = typer.Argument(..., help="A natural-language question."),
    k: int = typer.Option(10, "--k", help="Retrieval depth."),
    no_semantic: bool = typer.Option(
        False, "--no-semantic", help="BM25-only retrieval (skip the embedding call)."
    ),
) -> None:
    """Answer a question grounded in your captured memory, with citations.

    Runs hybrid retrieval over the store, builds an injection-resistant grounded
    prompt, asks the LLM, then prints the answer plus occurrence-level citations
    (app / window / time + a short snippet) that name where each fact came from.
    """
    from openbird.chat.rag import RAG

    provider = _provider()
    store = _store(provider=provider)
    try:
        rag = RAG(store, provider)
        result = rag.answer(question, k=k, semantic=not no_semantic)
    finally:
        store.close()

    if not result.grounded and result.answer:
        # Surface the grounding gate up front so an ungrounded answer is never
        # mistaken for verified memory.
        _console.print("[yellow]⚠ ungrounded — no verified source for this answer[/]")
    _console.print(result.answer or "[dim](no answer)[/]")
    if result.citations:
        _console.print("\n[bold]Sources[/]")
        for i, c in enumerate(result.citations, start=1):
            when = _fmt_ts(c.ts)
            where = " / ".join(p for p in (c.app, c.window) if p) or "unknown"
            _console.print(f"  [cyan][{i}][/] {where} · {when}")
            _console.print(f"      [dim]{c.snippet}[/]")
    else:
        _console.print("\n[dim]No citations (answer not grounded in a stored occurrence).[/]")


# --------------------------------------------------------------------------- #
# routine                                                                     #
# --------------------------------------------------------------------------- #


@routine_app.command("list")
def routine_list() -> None:
    """List the built-in read/summarize-only routine templates."""
    from openbird.routines.templates import BUILTIN_TEMPLATES

    table = Table(title="Built-in routines", show_header=True, header_style="bold")
    table.add_column("Name")
    table.add_column("Interval")
    table.add_column("Prompt")
    for name, tmpl in BUILTIN_TEMPLATES.items():
        table.add_row(name, _fmt_interval(tmpl.interval), tmpl.prompt)
    _console.print(table)


@routine_app.command("run")
def routine_run(
    name: str = typer.Argument(..., help="Routine template name (see `routine list`)."),
    show_output: bool = typer.Option(
        False,
        "--show-output",
        help="Print the generated summary to stdout (interactive use only).",
    ),
) -> None:
    """Run a single occurrence of a routine now and print its status.

    The routine range-scans the relevant time window of observations, asks the
    LLM for a grounded summary, and records a durable :class:`RoutineRun`. The
    summary body is only echoed when ``--show-output`` is given (interactive),
    keeping unattended runs from leaking captured content to logs.
    """
    from openbird.routines.scheduler import RoutineScheduler, stdout_deliverer
    from openbird.routines.templates import BUILTIN_TEMPLATES

    if name not in BUILTIN_TEMPLATES:
        _err_console.print(
            f"[red]Unknown routine:[/] {name}. Try `openbird routine list`."
        )
        raise typer.Exit(code=2)

    provider = _provider()
    store = _store(provider=provider)
    try:
        deliverer = stdout_deliverer if show_output else None
        scheduler = RoutineScheduler(
            memory_store=store, provider=provider, deliverer=deliverer
        )
        scheduler.register_template(BUILTIN_TEMPLATES[name])
        run = scheduler.fire(name)
    finally:
        store.close()

    if run is None:
        _console.print(
            f"[yellow]Routine {name!r} already ran for this occurrence (idempotent skip).[/]"
        )
        return
    _console.print(
        f"[green]Routine {name!r} finished[/] status={run.status} "
        f"scheduled={_fmt_ts(run.scheduled_ts)}"
    )


@routine_app.command("start")
def routine_start(
    lookback_days: float = typer.Option(
        7.0,
        "--lookback-days",
        help="Cap startup catch-up to this many days (0 = no catch-up). "
        "Prevents a startup 'LLM storm' after long downtime.",
    ),
    no_catch_up: bool = typer.Option(
        False, "--no-catch-up", help="Skip missed-occurrence catch-up entirely."
    ),
) -> None:
    """Run the routine scheduler as a foreground daemon until SIGINT/SIGTERM [B2].

    This is the always-on entrypoint a LaunchAgent (see `routine install`) execs.
    It registers every built-in routine, catches up missed occurrences (capped),
    then fires each on its interval. Output is delivered via the content-safe
    no-op sink, so no summary bodies reach stdout/stderr — only metadata logs.
    """
    import logging
    import signal
    import threading

    from openbird.routines.scheduler import RoutineScheduler
    from openbird.routines.templates import BUILTIN_TEMPLATES

    # Metadata-only logs to stderr (the LaunchAgent routes these to a 0600 file).
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )

    stop = threading.Event()

    def _handle(signum, _frame) -> None:
        _err_console.print(f"[yellow]routine daemon: signal {signum}, shutting down[/]")
        stop.set()

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    # `--lookback-days 0` (or negative) means "no catch-up", NOT unbounded
    # catch-up: lookback=None disables the *cap*, so it must pair with
    # catch_up=False to actually skip catch-up.
    catch_up = not no_catch_up and lookback_days > 0
    lookback = lookback_days * 86400.0 if catch_up else None

    store = None
    scheduler = None
    try:
        provider = _provider()
        store = _store(provider=provider)
        scheduler = RoutineScheduler(memory_store=store, provider=provider)
        for template in BUILTIN_TEMPLATES.values():
            scheduler.register_template(template)
        scheduler.start(catch_up=catch_up, lookback=lookback)
        _console.print(
            f"[green]routine daemon started[/] routines={len(scheduler.routines)} "
            f"(Ctrl-C to stop)"
        )
        stop.wait()
    except Exception as exc:  # noqa: BLE001 - daemon must not dump content to logs
        # Content-safe fatal handling: log the exception CLASS only (never the
        # message/traceback, which can embed captured content), then exit nonzero
        # so launchd restarts us (throttled).
        logging.getLogger("openbird.routines").error(
            "routine daemon fatal: error_class=%s", type(exc).__name__, exc_info=False
        )
        raise typer.Exit(code=1) from None
    finally:
        # Run cleanup on every path. Each step is attempted independently and its
        # failure is logged content-safe (class only, no traceback) and swallowed,
        # so a raising shutdown can't skip store.close() and no cleanup traceback
        # escapes to the daemon log.
        _log = logging.getLogger("openbird.routines")
        if scheduler is not None:
            try:
                scheduler.shutdown(wait=True)
            except Exception as exc:  # noqa: BLE001 - content-safe cleanup
                _log.error(
                    "scheduler shutdown failed: error_class=%s",
                    type(exc).__name__,
                    exc_info=False,
                )
        if store is not None:
            try:
                store.close()
            except Exception as exc:  # noqa: BLE001 - content-safe cleanup
                _log.error(
                    "store close failed: error_class=%s",
                    type(exc).__name__,
                    exc_info=False,
                )


@routine_app.command("install")
def routine_install(
    load: bool = typer.Option(
        False, "--load", help="Also `launchctl load` the agent now (starts it)."
    ),
) -> None:
    """Write the per-user LaunchAgent so routines run at login [B2].

    Writes ~/Library/LaunchAgents/ai.openbird.routines.plist pointing at the
    resolved `openbird` executable. By default it only writes the file and
    prints the load command; pass --load to register it with launchd now.
    """
    import shutil
    import subprocess

    from openbird.routines.launchd import agent_plist_path, build_agent_plist

    openbird_exe = shutil.which("openbird") or sys.argv[0]
    if not Path(openbird_exe).is_absolute():
        _err_console.print(
            "[red]Could not resolve an absolute path to the `openbird` executable.[/] "
            "Install the CLI (e.g. `uv tool install .`) so it is on PATH, then retry."
        )
        raise typer.Exit(code=1)

    settings = get_settings()
    log_dir = settings.data_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stderr_path = str(log_dir / "routines.err.log")

    plist_bytes = build_agent_plist(
        program_args=[openbird_exe, "routine", "start"],
        stderr_path=stderr_path,
    )
    path = agent_plist_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(plist_bytes)
    _console.print(f"[green]Wrote LaunchAgent:[/] {path}")

    if load:
        try:
            subprocess.run(["launchctl", "load", str(path)], check=True)
            _console.print("[green]Loaded into launchd (running at next interval).[/]")
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            _err_console.print(f"[red]launchctl load failed:[/] {type(exc).__name__}")
            raise typer.Exit(code=1) from exc
    else:
        _console.print(f"To start it now:  launchctl load {path}")


@routine_app.command("uninstall")
def routine_uninstall(
    unload: bool = typer.Option(
        False, "--unload", help="Also `launchctl unload` the agent before removing."
    ),
) -> None:
    """Remove the per-user LaunchAgent for the routine daemon [B2]."""
    import subprocess

    from openbird.routines.launchd import agent_plist_path

    path = agent_plist_path()
    if unload and path.exists():
        try:
            subprocess.run(["launchctl", "unload", str(path)], check=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            _err_console.print(f"[red]launchctl unload failed:[/] {type(exc).__name__}")
    if path.exists():
        path.unlink()
        _console.print(f"[green]Removed LaunchAgent:[/] {path}")
    else:
        _console.print("[yellow]No LaunchAgent installed.[/]")


# --------------------------------------------------------------------------- #
# meeting (stub)                                                              #
# --------------------------------------------------------------------------- #


@app.command()
def meeting() -> None:
    """Meetings stub (manual-record subsystem, gated on a signed audio helper).

    Meeting capture uses the Swift ``audio-helper`` (ScreenCaptureKit system
    audio + mic as separate synchronized tracks) and is **manual-record** in v1.
    It requires the packaged signed helper plus Screen-Recording/Microphone TCC,
    so the full pipeline cannot be driven from this CLI yet. This command reports
    the meetings readiness (the transcription extra) and exits.
    """
    from openbird.meetings.transcribe import whisper_available

    _console.print("[bold]OpenBird meetings[/] (manual-record, experimental)")
    _console.print(
        "- Audio capture requires the signed ScreenCaptureKit `audio-helper` "
        "with Screen-Recording + Microphone TCC."
    )
    available = whisper_available()
    state = "installed" if available else "not installed"
    _console.print(f"- faster-whisper transcription extra: {state}.")
    _console.print(
        "- Speaker labeling ('me vs others') is experimental; consent indicator "
        "and manual start are required by design."
    )
    if not available:
        _console.print(
            "[dim]Install with `uv sync --extra meetings` to enable transcription.[/]"
        )


# --------------------------------------------------------------------------- #
# data                                                                        #
# --------------------------------------------------------------------------- #

data_app = typer.Typer(help="Manage stored data (purge, stats).", no_args_is_help=True)
app.add_typer(data_app, name="data")


@data_app.command("purge")
def data_purge(
    since: Optional[str] = typer.Option(
        None,
        "--since",
        help="Delete observations at/after this time. Accepts a unix timestamp, "
        "an ISO date/datetime, or a relative span like '7d', '24h', '30m'.",
    ),
    all_: bool = typer.Option(
        False, "--all", help="Delete ALL observations and content (irreversible)."
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
) -> None:
    """Cascade-delete observations (and orphaned blobs/chunks/index entries).

    Exactly one of ``--since`` or ``--all`` is required. Deletion cascades across
    observations, content blobs, chunks, the FTS index, and the vector table, so
    purged content is removed from every index (a verified trust-surface action).
    """
    if all_ == bool(since):
        _err_console.print("[red]Provide exactly one of --since or --all.[/]")
        raise typer.Exit(code=2)

    since_ts = _parse_since(since) if since else None

    if not yes:
        target = "ALL data" if all_ else f"data since {_fmt_ts(since_ts)}"
        confirm = typer.confirm(f"Permanently delete {target}?", default=False)
        if not confirm:
            _console.print("[yellow]Aborted.[/]")
            raise typer.Exit(code=1)

    store = _store()
    try:
        deleted = store.delete(all=all_, since_ts=since_ts)
    finally:
        store.close()
    _console.print(f"[green]Deleted[/] {deleted} observation(s) (cascade complete).")


@data_app.command("stats")
def data_stats() -> None:
    """Print memory-store row counts and the active embedding cohort."""
    store = _store()
    try:
        stats = store.stats()
    finally:
        store.close()
    _console.print_json(json.dumps(stats))


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #


def _fmt_ts(ts: float | None) -> str:
    """Format a unix timestamp as a local datetime (or a dash for None)."""
    if ts is None:
        return "-"
    return _dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _fmt_interval(seconds: float) -> str:
    """Render a routine interval in days/hours/minutes."""
    if seconds % 86400 == 0:
        return f"{int(seconds // 86400)}d"
    if seconds % 3600 == 0:
        return f"{int(seconds // 3600)}h"
    return f"{int(seconds // 60)}m"


def _parse_since(value: str) -> float:
    """Parse a --since value into a unix timestamp.

    Accepts a bare unix timestamp, a relative span (``7d``/``24h``/``30m``/
    ``45s``), or an ISO 8601 date/datetime. Relative spans are subtracted from
    *now*.
    """
    import time

    value = value.strip()

    # Relative span like "7d", "24h", "30m", "45s".
    units = {"d": 86400, "h": 3600, "m": 60, "s": 1}
    if value and value[-1] in units and value[:-1].replace(".", "", 1).isdigit():
        return time.time() - float(value[:-1]) * units[value[-1]]

    # Bare unix timestamp.
    try:
        return float(value)
    except ValueError:
        pass

    # ISO date / datetime.
    try:
        dt = _dt.datetime.fromisoformat(value)
        return dt.timestamp()
    except ValueError as exc:
        raise typer.BadParameter(
            f"could not parse --since {value!r}; use a unix ts, ISO date, or span like '7d'."
        ) from exc


def main() -> None:
    """Console-script entrypoint."""
    app()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())
