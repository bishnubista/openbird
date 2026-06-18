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


def _resolve_cloud_opt_in(remote_models: dict[str, str]) -> bool:
    """Decide whether to proceed with a REMOTE model route [H3].

    Called when the provider factory refused because a cloud (or remote-Ollama)
    model is configured without opt-in. On an interactive TTY we ask for an
    explicit confirmation after showing exactly which model(s) would send
    captured memory off the machine; non-interactively we refuse (the safe,
    automation-friendly default). Returns True only if the user opts in here.
    """
    names = ", ".join(f"{role}={model}" for role, model in remote_models.items())
    _err_console.print(
        "[bold red]CLOUD MODEL CONFIGURED[/] — the following model(s) would send "
        f"your captured memory to a third party: [bold]{names}[/]."
    )
    if not sys.stdin.isatty():
        _err_console.print(
            "[red]Refusing[/] (non-interactive). Set [bold]OPENBIRD_ALLOW_CLOUD=1[/] "
            "to opt in, or use a local ollama/* model on a loopback host."
        )
        return False
    return typer.confirm(
        "Send captured memory to this remote model?", default=False
    )


def _provider():
    """Construct the configured LLM provider, enforcing cloud opt-in [H3].

    The provider factory refuses a remote model unless cloud is opted into. Here
    we surface that refusal as an interactive confirm (TTY) or a clean exit
    (non-interactive), then print the CLOUD ACTIVE banner whenever a remote model
    is in use so it is never silent.
    """
    from openbird.llm.provider import (
        CloudOptInRequired,
        cloud_banner,
        create_llm_provider,
    )

    settings = get_settings()
    try:
        provider = create_llm_provider(settings)
    except CloudOptInRequired as exc:
        if not _resolve_cloud_opt_in(exc.remote_models):
            raise typer.Exit(code=2) from exc
        provider = create_llm_provider(settings, allow_cloud=True)

    banner = cloud_banner(settings)
    if banner:
        _err_console.print(f"[bold yellow]⚠ {banner}[/]")
    return provider


def _store(*, provider=None):
    """Open the on-disk :class:`MemoryStore` with the cloud-checked provider.

    Always builds the provider through :func:`_provider` (unless one is passed
    in) so the cloud opt-in policy + banner apply on every store-opening command
    (ingest, data stats, reindex), not just chat/routine. MemoryStore would
    otherwise construct a provider internally and bypass the CLI's confirm/banner.
    """
    from openbird.memory.store import MemoryStore

    if provider is None:
        provider = _provider()
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

    cloud = report.get("cloud", {})
    if cloud.get("active"):
        if cloud.get("blocked"):
            c_status, c_detail = (
                "blocked",
                f"remote {cloud.get('remote_models')} — set OPENBIRD_ALLOW_CLOUD=1 to opt in",
            )
        else:
            c_status, c_detail = (
                "CLOUD ACTIVE",
                f"remote {cloud.get('remote_models')} (memory leaves this machine)",
            )
    else:
        c_status, c_detail = "local", "all models local"
    table.add_row("cloud", c_status, c_detail)

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
# reindex                                                                     #
# --------------------------------------------------------------------------- #


def _serialize_f32(vector: list[float]) -> bytes:
    """Pack a float vector into sqlite-vec's little-endian float32 blob format.

    Re-declared locally (it is a one-line ``struct.pack``) so reindex does not
    import a private symbol from :mod:`openbird.memory.store`.
    """
    import struct

    return struct.pack(f"<{len(vector)}f", *vector)


@app.command()
def reindex(
    batch_size: int = typer.Option(
        64, "--batch-size", help="How many chunks to embed per provider call."
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
    force: bool = typer.Option(
        False,
        "--force",
        help="Re-embed even when the stored cohort already matches the provider.",
    ),
) -> None:
    """Re-embed every stored chunk under the current embedding model [M2].

    Switching ``OPENBIRD_EMBED_MODEL`` (or its dimension) leaves the store in a
    cohort mismatch — old vectors are incompatible with new queries. This rebuilds
    ``vec_chunks`` at the new dimension, re-embeds all chunk text with the current
    provider, and adopts the new cohort key. The whole rebuild runs in a single
    transaction: if any embedding call fails (e.g. Ollama down) it rolls back, so
    the old, searchable vectors and cohort survive intact.

    The DB is opened directly (not via ``MemoryStore``) because ``MemoryStore``
    refuses to construct on a populated cohort mismatch — which is exactly the
    state ``reindex`` exists to repair.
    """
    from pathlib import Path

    from openbird.storage.crypto import mapping_row_factory, open_encrypted_db

    settings = get_settings()
    provider = _provider()  # cloud opt-in + banner enforced here too.
    new_cohort = provider.cohort_key()
    new_dim = provider.embed_dim

    conn = open_encrypted_db(settings.db_path, settings=settings)
    conn.row_factory = mapping_row_factory
    conn.execute("PRAGMA foreign_keys = ON")
    # open_encrypted_db does NOT apply the schema (only MemoryStore does), so on a
    # fresh/uninitialized DB the embedding_meta/chunks tables would not exist yet.
    # Apply the base schema so reindex degrades to a clean "0 chunk(s)" rather than
    # raising "no such table". The vec table is (re)created below at the new dim.
    _schema = (Path(__file__).resolve().parent / "memory" / "schema.sql").read_text(
        encoding="utf-8"
    )
    conn.executescript(_schema)
    conn.commit()
    try:
        cohort_row = conn.execute(
            "SELECT value FROM embedding_meta WHERE key = 'cohort_key'"
        ).fetchone()
        current_cohort = cohort_row["value"] if cohort_row else None

        chunk_rows = conn.execute(
            "SELECT rowid_int, text FROM chunks WHERE rowid_int IS NOT NULL "
            "ORDER BY rowid_int"
        ).fetchall()
        total = len(chunk_rows)

        if current_cohort == new_cohort and not force:
            _console.print(
                f"[green]Already on cohort[/] {new_cohort} "
                f"({total} chunk(s)); nothing to do. Use --force to rebuild anyway."
            )
            return

        _console.print(
            f"Reindex: [cyan]{current_cohort or '(none)'}[/] -> [cyan]{new_cohort}[/] "
            f"· {total} chunk(s) · dim={new_dim}"
        )
        if not yes:
            if not sys.stdin.isatty():
                _err_console.print(
                    "[red]Refusing[/] (non-interactive). Re-run with --yes to reindex."
                )
                raise typer.Exit(code=1)
            if not typer.confirm("Re-embed all chunks under the new cohort?", default=False):
                _console.print("[yellow]Aborted.[/]")
                raise typer.Exit(code=1)

        try:
            # Wrap the whole rebuild in ONE explicit transaction. Python's sqlite3
            # does NOT auto-begin a transaction for DDL (only DML), so without an
            # explicit BEGIN the DROP/CREATE would auto-commit and a later failure
            # could not roll back — destroying the old vectors. With BEGIN, a mid-
            # reindex embed failure rolls back the drop+rebuild atomically.
            conn.execute("BEGIN")
            # Rebuild the vector table at the (possibly new) dimension. CREATE ...
            # IF NOT EXISTS would keep the stale dim, so drop first.
            conn.execute("DROP TABLE IF EXISTS vec_chunks")
            conn.execute(
                f"CREATE VIRTUAL TABLE vec_chunks USING vec0("
                f"chunk_rowid INTEGER PRIMARY KEY, embedding FLOAT[{new_dim}])"
            )

            done = 0
            with _progress_columns() as progress:
                task = progress.add_task("Embedding chunks", total=total) if total else None
                for start in range(0, total, max(1, batch_size)):
                    batch = chunk_rows[start : start + max(1, batch_size)]
                    vectors = provider.embed([r["text"] for r in batch])
                    for row, vec in zip(batch, vectors):
                        conn.execute(
                            "INSERT INTO vec_chunks(chunk_rowid, embedding) VALUES (?, ?)",
                            (int(row["rowid_int"]), _serialize_f32(vec)),
                        )
                    done += len(batch)
                    if task is not None:
                        progress.update(task, completed=done)

            # Adopt the new cohort only after every vector is in place.
            if current_cohort is None:
                conn.execute(
                    "INSERT INTO embedding_meta(key, value) VALUES ('cohort_key', ?)",
                    (new_cohort,),
                )
            else:
                conn.execute(
                    "UPDATE embedding_meta SET value = ? WHERE key = 'cohort_key'",
                    (new_cohort,),
                )
            conn.commit()
        except Exception as exc:
            conn.rollback()
            _err_console.print(
                f"[red]Reindex failed[/] ({type(exc).__name__}: {exc}); rolled back. "
                "Your existing vectors and cohort are unchanged."
            )
            raise typer.Exit(code=1) from exc
    finally:
        conn.close()

    _console.print(
        f"[green]Reindexed[/] {total} chunk(s) under cohort {new_cohort} (dim={new_dim})."
    )


def _progress_columns():
    """A rich progress bar for long reindex runs (spinner + bar + count)."""
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
    )

    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=_console,
        transient=True,
    )


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
