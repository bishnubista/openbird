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
* The CLI is the serialized integration point: it imports the
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
import logging
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.markup import escape
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
summaries_app = typer.Typer(
    help="Build and inspect idle-time block summaries.", no_args_is_help=True
)
eval_app = typer.Typer(help="Run local eval harnesses.", no_args_is_help=True)
day_memory_app = typer.Typer(
    help="Build and inspect deterministic daily memory artifacts.",
    no_args_is_help=True,
)
deep_brain_app = typer.Typer(
    help="Preview the local packet for opt-in Deep Brain reasoning.",
    no_args_is_help=True,
)
entities_app = typer.Typer(
    help="Inspect the derived entity ledger (projects, domains, completion evidence).",
    no_args_is_help=True,
)
assistant_app = typer.Typer(
    help="Connect read-only OpenBird capture to desktop assistants.",
    no_args_is_help=True,
)
app.add_typer(routine_app, name="routine")
app.add_typer(summaries_app, name="summaries")
app.add_typer(entities_app, name="entities")
app.add_typer(eval_app, name="eval")
app.add_typer(day_memory_app, name="day-memory")
app.add_typer(deep_brain_app, name="deep-brain")
app.add_typer(assistant_app, name="assistant")
register_capture_command(app)

from openbird.prompts.cli import prompts_app  # noqa: E402 - after app is defined

app.add_typer(prompts_app, name="prompts")

_console = Console()
_err_console = Console(stderr=True)
_log = logging.getLogger("openbird.cli")


@assistant_app.command("serve", hidden=True)
def assistant_serve() -> None:
    """Run the local read-only MCP server over stdio."""
    from openbird.assistant import run_mcp_server

    try:
        run_mcp_server()
    except RuntimeError as exc:
        _err_console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(code=1) from exc


@assistant_app.command("install-claude")
def assistant_install_claude(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
    executable: Optional[Path] = typer.Option(
        None,
        "--executable",
        hidden=True,
        help="Pin Claude to the OpenBird CLI that launched this installer.",
    ),
) -> None:
    """Connect OpenBird capture to Claude Desktop on this Mac."""
    from openbird.assistant import (
        ASSISTANT_EGRESS_NOTICE,
        ClaudeConfigConflictError,
        install_claude_config,
    )

    _err_console.print(f"[bold yellow]ASSISTANT ACCESS[/] — {ASSISTANT_EGRESS_NOTICE}")
    if not yes:
        if not sys.stdin.isatty():
            _err_console.print(
                "[red]Refusing[/] (non-interactive). Re-run with --yes to connect Claude."
            )
            raise typer.Exit(code=1)
        if not typer.confirm("Connect read-only OpenBird capture to Claude Desktop?", default=False):
            _console.print("[yellow]Aborted.[/]")
            raise typer.Exit(code=1)
    try:
        result = install_claude_config(executable=executable)
    except (ClaudeConfigConflictError, FileNotFoundError, OSError, ValueError) as exc:
        _err_console.print(f"[red]Could not configure Claude Desktop:[/] {escape(str(exc))}")
        raise typer.Exit(code=1) from exc
    _console.print(f"[green]Connected[/] Claude Desktop via {result['config_path']}.")
    _console.print("Restart Claude Desktop, then ask it to use OpenBird.")


@assistant_app.command("status")
def assistant_status(json_out: bool = typer.Option(False, "--json")) -> None:
    """Show whether Claude Desktop has the OpenBird connector configured."""
    from openbird.assistant import claude_config_status

    try:
        result = claude_config_status()
    except ValueError as exc:
        _err_console.print(f"[red]Claude Desktop config error:[/] {escape(str(exc))}")
        raise typer.Exit(code=1) from exc
    if json_out:
        _console.print_json(json.dumps(result))
        return
    if result["configured"]:
        _console.print(f"[green]Connected[/] Claude Desktop · {result['command']}")
    else:
        _console.print("[yellow]Not connected[/] · run openbird assistant install-claude")


@assistant_app.command("chatgpt-status")
def assistant_chatgpt_status(
    json_out: bool = typer.Option(False, "--json"),
    executable: Optional[Path] = typer.Option(None, "--executable", hidden=True),
) -> None:
    """Show metadata-only ChatGPT tunnel setup readiness."""
    from openbird.assistant import chatgpt_status

    result = chatgpt_status(executable=executable)
    if json_out:
        _console.print_json(json.dumps(result))
    elif result["configured"] and result["helper_available"]:
        _console.print("[green]Configured[/] ChatGPT Secure MCP Tunnel")
    else:
        _console.print("[yellow]Setup needed[/] · use OpenBird Settings")


@assistant_app.command("configure-chatgpt")
def assistant_configure_chatgpt(
    tunnel_id: str = typer.Option(..., "--tunnel-id"),
    yes: bool = typer.Option(False, "--yes", "-y"),
    executable: Optional[Path] = typer.Option(None, "--executable", hidden=True),
) -> None:
    """Configure OpenBird's official OpenAI Secure MCP Tunnel profile."""
    from openbird.assistant import ASSISTANT_EGRESS_NOTICE, configure_chatgpt

    _err_console.print(f"[bold yellow]ASSISTANT ACCESS[/] — {ASSISTANT_EGRESS_NOTICE}")
    if not yes:
        _err_console.print("[red]Refusing[/]. Re-run from OpenBird Settings or with --yes.")
        raise typer.Exit(code=1)
    try:
        configure_chatgpt(tunnel_id, executable=executable)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        _err_console.print(f"[red]Could not configure ChatGPT:[/] {escape(str(exc))}")
        raise typer.Exit(code=1) from exc
    _console.print("[green]Configured[/] ChatGPT Secure MCP Tunnel.")


@assistant_app.command("run-chatgpt", hidden=True)
def assistant_run_chatgpt(
    executable: Optional[Path] = typer.Option(None, "--executable", hidden=True),
) -> None:
    """Run OpenBird's configured outbound ChatGPT tunnel."""
    from openbird.assistant import run_chatgpt_tunnel

    try:
        run_chatgpt_tunnel(executable=executable)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        _err_console.print(f"[red]ChatGPT tunnel could not start:[/] {escape(str(exc))}")
        raise typer.Exit(code=1) from exc


@assistant_app.command("remove-chatgpt", hidden=True)
def assistant_remove_chatgpt(yes: bool = typer.Option(False, "--yes", "-y")) -> None:
    """Remove only OpenBird's local ChatGPT tunnel profile."""
    from openbird.assistant import remove_chatgpt_config

    if not yes:
        _err_console.print("[red]Refusing[/]. Re-run from OpenBird Settings or with --yes.")
        raise typer.Exit(code=1)
    if not remove_chatgpt_config():
        _err_console.print("[red]Could not remove OpenBird's ChatGPT profile.[/]")
        raise typer.Exit(code=1)
    _console.print("[green]Removed[/] OpenBird's ChatGPT tunnel profile.")


# --------------------------------------------------------------------------- #
# Lazily-constructed shared services                                          #
# --------------------------------------------------------------------------- #


def _resolve_cloud_opt_in(remote_models: dict[str, str]) -> bool:
    """Decide whether to proceed with a REMOTE model route.

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
    """Construct the configured LLM provider, enforcing cloud opt-in.

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


def _completion_provider(*, packet_label: str = "Deep Brain packet"):
    """Construct a provider for completion-only routes, gating only the LLM role."""
    from openbird.llm.provider import (
        CloudOptInRequired,
        classify_models,
        create_llm_provider,
    )

    settings = get_settings()
    try:
        provider = create_llm_provider(settings, cloud_roles=("llm",))
    except CloudOptInRequired as exc:
        names = ", ".join(f"{role}={model}" for role, model in exc.remote_models.items())
        _err_console.print(
            "[red]Cloud opt-in required[/] for completion model(s): "
            f"[bold]{names}[/]."
        )
        raise typer.Exit(code=2) from exc

    remote = classify_models(settings)
    if remote.get("llm"):
        _err_console.print(
            "[bold yellow]⚠ CLOUD ACTIVE — remote llm="
            f"{remote['llm']} ({packet_label} leaves this machine)[/]"
        )
    return provider


def _print_cohort_mismatch_hint(exc) -> None:
    """Render a content-free recovery hint for an embedding-cohort mismatch.

    The cohort keys are ``provider:model:dim:hash`` strings (no captured text), so
    this is privacy-safe to print. Printed to the error console BEFORE the caller
    raises ``typer.Exit`` — some callers (e.g. ``routine start``) catch a broad
    ``Exception`` around the run and would otherwise swallow the Exit and log only
    its class, leaving the user without the actionable next step.
    """
    _err_console.print(
        f"[red]Embedding model changed[/] ({exc.stored} → {exc.current}).\n"
        "Run [bold]openbird reindex[/] to rebuild the vector index under the new "
        "model, then retry.\n"
        "[dim](Or set OPENBIRD_EMBED_MODEL to the previous model to defer.)[/]"
    )


def _store(*, provider=None, settings=None, reraise_cohort_mismatch=False):
    """Open the on-disk :class:`MemoryStore` with the cloud-checked provider.

    Always builds the provider through :func:`_provider` (unless one is passed
    in) so the cloud opt-in policy + banner apply on every command that may
    EMBED (ingest, chat, capture, reindex). MemoryStore would otherwise construct
    a provider internally and bypass the CLI's confirm/banner. For delete-only
    maintenance (purge/stats) use :func:`_store_maintenance`, which does not gate
    on cloud opt-in since no captured content is sent anywhere.

    Converts an :class:`EmbeddingCohortMismatch` (the user switched embed models on
    a populated store) into a friendly ``openbird reindex`` hint + exit rather than
    a raw traceback. This is the single shared seam every EMBED command — and the
    capture daemon CLI — routes through, so the recovery path is uniform.

    ``reraise_cohort_mismatch``: the long-running ``capture --loop`` daemon needs
    to translate the mismatch into its OWN distinct exit code (so the mac app can
    tell "needs reindex" apart from a generic crash), so it passes True to get the
    typed exception back instead of the generic ``typer.Exit(code=1)``.
    """
    from openbird.memory.store import EmbeddingCohortMismatch, MemoryStore

    if provider is None:
        provider = _provider()
    if settings is None:
        settings = get_settings()
    try:
        return MemoryStore(settings=settings, provider=provider)
    except EmbeddingCohortMismatch as exc:
        if reraise_cohort_mismatch:
            raise
        _print_cohort_mismatch_hint(exc)
        raise typer.Exit(code=1) from exc


class _MaintenanceProvider:
    """A non-embedding provider stub for delete-only maintenance (purge/stats).

    Reports the EXACT cohort already recorded in the store so
    ``MemoryStore._record_cohort`` sees a match and never raises — even after the
    user switched ``OPENBIRD_EMBED_MODEL``/dimension (the very state that would
    otherwise block the privacy/cleanup path). It never embeds: purge and stats
    do not call ``embed``, and if anything ever did it fails loudly rather than
    silently sending captured content anywhere.
    """

    def __init__(self, cohort: str | None, embed_dim: int) -> None:
        self._cohort = cohort
        self.embed_dim = embed_dim
        self.normalized = False

    def cohort_key(self) -> str:
        # No stored cohort yet (fresh DB) -> a sentinel; _record_cohort will just
        # insert it, and there are no vectors to mix.
        return self._cohort or "maintenance:none:0:0"

    def embed(self, texts):  # pragma: no cover - must never run on these paths
        raise RuntimeError("maintenance provider must not embed")

    def complete(self, messages, *, json_schema=None):  # pragma: no cover
        raise RuntimeError("maintenance provider must not complete")


def _peek_cohort(settings) -> str | None:
    """Read the recorded cohort_key from the DB without constructing MemoryStore.

    Opens the raw (cloud-gate-free) connection so we can mirror the stored cohort
    into :class:`_MaintenanceProvider`, sidestepping the cohort-mismatch guard for
    delete-only ops. Returns None on a missing table / fresh DB.
    """
    from openbird.storage.crypto import open_encrypted_db

    conn = open_encrypted_db(settings.db_path, settings=settings)
    try:
        row = conn.execute(
            "SELECT value FROM embedding_meta WHERE key = 'cohort_key'"
        ).fetchone()
        if row is None:
            return None
        # row may be a tuple or a mapping depending on row_factory (default tuple).
        return row[0] if not hasattr(row, "keys") else row["value"]
    except Exception:
        return None
    finally:
        conn.close()


def _store_maintenance():
    """Open the store for local maintenance ops WITHOUT a cloud gate.

    Purge/prune/vacuum/stats never embed or send captured content to a model, so:
      * they must NOT require ``OPENBIRD_ALLOW_CLOUD`` (deleting local data on a
        privacy tool can't depend on cloud opt-in), AND
      * they must NOT be blocked by an embedding-cohort mismatch after the user
        switched embed models (the cleanup path has to keep working).
    Both are achieved with :class:`_MaintenanceProvider`, which reports the
    already-stored cohort (so ``_record_cohort`` matches) and never embeds.
    """
    from openbird.memory.store import MemoryStore

    settings = get_settings()
    provider = _MaintenanceProvider(_peek_cohort(settings), settings.embed_dim)
    return MemoryStore(settings=settings, provider=provider)


def _render_chat_result(result, *, json_out: bool) -> None:
    """Render a chat result through the CLI's existing JSON/human contract."""
    if json_out:
        _console.print_json(json.dumps(result.to_public_dict()))
        raise typer.Exit(code=0)

    if result.grounding == "ungrounded" and result.answer:
        # Surface the grounding gate up front so an ungrounded answer is never
        # mistaken for verified memory.
        _console.print("[yellow]⚠ ungrounded — no verified source for this answer[/]")
    if result.answer:
        _console.print(escape(result.answer))
    else:
        _console.print("[dim](no answer)[/]")
    if result.citations:
        _console.print("\n[bold]Sources[/]")
        for i, c in enumerate(result.citations, start=1):
            when = _fmt_ts(c.ts)
            where = " / ".join(p for p in (c.app, c.window) if p) or "unknown"
            _console.print(f"  [cyan][{i}][/] {escape(where)} · {when}")
            _console.print(f"      [dim]{escape(c.snippet)}[/]")
    if result.derived_citations:
        _console.print("\n[bold]Derived sources[/]")
        for c in result.derived_citations:
            _console.print(
                f"  [cyan][{c.index}][/] {escape(c.label)} · "
                f"{c.derived_from_total} source observation(s)"
            )
            _console.print(f"      [dim]{escape(c.snippet)}[/]")
    else:
        if not result.citations:
            _console.print(
                "\n[dim]No citations (answer not grounded in a stored occurrence).[/]"
            )


# --------------------------------------------------------------------------- #
# Today / day view (timeline + briefing)                                      #
# --------------------------------------------------------------------------- #


def _day_window(day_offset: int) -> tuple[float, float]:
    """Inclusive ``[start, end]`` of the local calendar day ``day_offset`` days ago
    (0=today, 1=yesterday). The single source of day bounds shared by ``timeline``
    and ``briefing`` so the session list and the prose summary cover the same span.
    """
    from openbird.routines.templates import _day_bounds

    return _day_bounds(_dt.datetime.now().timestamp(), offset_days=-day_offset)


def _validate_deep_brain_days(days: int) -> None:
    from openbird.deep_brain import MAX_DEEP_BRAIN_DAYS

    if days < 1 or days > MAX_DEEP_BRAIN_DAYS:
        _err_console.print(
            f"[red]--days must be between 1 and {MAX_DEEP_BRAIN_DAYS}.[/]"
        )
        raise typer.Exit(code=2)


def _merge_cli_exclusions(configured: list[str], cli_values: list[str] | None) -> list[str]:
    """Append CLI exclusion values without mutating cached settings."""
    merged: list[str] = []
    seen: set[str] = set()
    for value in [*configured, *(cli_values or [])]:
        item = str(value).strip()
        if not item:
            continue
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged


def _settings_with_cli_cloud_exclusions(
    settings,
    *,
    exclude_app: list[str] | None = None,
    exclude_source: list[str] | None = None,
    exclude_observation_id: list[str] | None = None,
):
    """Return a settings copy with one-off cloud packet exclusions applied."""
    if not (exclude_app or exclude_source or exclude_observation_id):
        return settings
    return replace(
        settings,
        deep_brain_excluded_apps=_merge_cli_exclusions(
            settings.deep_brain_excluded_apps, exclude_app
        ),
        deep_brain_excluded_sources=_merge_cli_exclusions(
            settings.deep_brain_excluded_sources, exclude_source
        ),
        deep_brain_excluded_observation_ids=_merge_cli_exclusions(
            settings.deep_brain_excluded_observation_ids, exclude_observation_id
        ),
    )


def _has_cli_cloud_exclusions(
    exclude_app: list[str] | None,
    exclude_source: list[str] | None,
    exclude_observation_id: list[str] | None,
) -> bool:
    return bool(exclude_app or exclude_source or exclude_observation_id)


def _deep_brain_status_payload(settings) -> dict[str, Any]:
    """Build a local-only Deep Brain consent/status payload from settings.

    Privacy: this is settings-only. It must not open the memory DB, construct a
    provider, or probe the network. ``deep_brain_ask_blocked_reasons`` currently
    classifies model routes by configured strings/loopback host only; if that ever
    changes, this status route must stay no-egress.
    """
    from openbird.deep_brain import (
        deep_brain_ask_blocked_reasons,
        deep_brain_blocked_reasons,
    )

    cloud_blocked_reasons = deep_brain_blocked_reasons(settings)
    ask_blocked_reasons = deep_brain_ask_blocked_reasons(settings)
    cloud_gates_enabled = not cloud_blocked_reasons
    ask_available = not ask_blocked_reasons
    if not settings.deep_brain_enabled:
        route_label = "Deep Brain off"
    elif cloud_gates_enabled:
        route_label = "Cloud reasoning gates enabled"
    elif ask_available:
        route_label = "Deep Brain local ask available · no cloud"
    else:
        route_label = "Deep Brain blocked"

    return {
        "route": "deep_brain.status",
        "egress": "none",
        "route_label": route_label,
        "deep_brain_enabled": bool(settings.deep_brain_enabled),
        "cloud_opt_in": bool(settings.allow_cloud),
        "cloud_gates_enabled": cloud_gates_enabled,
        "cloud_blocked_reasons": cloud_blocked_reasons,
        "ask_available": ask_available,
        "ask_blocked_reasons": ask_blocked_reasons,
        "exclusions": {
            "excluded_apps_configured": list(settings.deep_brain_excluded_apps),
            "excluded_sources_configured": list(settings.deep_brain_excluded_sources),
            "excluded_observation_ids_configured": len(
                settings.deep_brain_excluded_observation_ids
            ),
        },
        "env_vars": {
            "deep_brain_enabled": "OPENBIRD_DEEP_BRAIN_ENABLED",
            "cloud_opt_in": "OPENBIRD_ALLOW_CLOUD",
            "excluded_apps": "OPENBIRD_DEEP_BRAIN_EXCLUDED_APPS",
            "excluded_sources": "OPENBIRD_DEEP_BRAIN_EXCLUDED_SOURCES",
            "excluded_observation_ids": "OPENBIRD_DEEP_BRAIN_EXCLUDED_OBSERVATION_IDS",
        },
    }


def _packet_audit_from_deep_brain_packet(packet: dict[str, Any]) -> dict[str, Any]:
    from openbird.deep_brain import packet_json_for_model
    from openbird.reasoning_ledger import packet_payload_audit

    return packet_payload_audit(
        packet_json_for_model(packet),
        selected_source_count=len(packet.get("selected_sources") or []),
        exclusions=packet.get("exclusions"),
    )


def _packet_audit_from_productivity_packet(packet: dict[str, Any]) -> dict[str, Any]:
    from openbird.day_memory import productivity_coach_packet_json_for_model
    from openbird.reasoning_ledger import packet_payload_audit

    return packet_payload_audit(
        productivity_coach_packet_json_for_model(packet),
        selected_source_count=0,
        exclusions=packet.get("exclusions"),
    )


def _safe_packet_audit_from_deep_brain_packet(packet: dict[str, Any]) -> dict[str, Any]:
    try:
        return _packet_audit_from_deep_brain_packet(packet)
    except Exception as exc:  # noqa: BLE001  # pragma: no cover - defensive audit path
        _log.info("reasoning_packet_audit_skipped reason=%s", type(exc).__name__)
        return {}


def _safe_packet_audit_from_productivity_packet(packet: dict[str, Any]) -> dict[str, Any]:
    try:
        return _packet_audit_from_productivity_packet(packet)
    except Exception as exc:  # noqa: BLE001  # pragma: no cover - defensive audit path
        _log.info("reasoning_packet_audit_skipped reason=%s", type(exc).__name__)
        return {}


def _record_reasoning_send_success(
    *, feature: str, settings, result: dict[str, Any]
) -> None:
    try:
        if result.get("egress") != "active_model_route":
            return
        _record_reasoning_send(
            feature=feature,
            settings=settings,
            packet_route=result.get("packet_route"),
            reasoning_route=result.get("reasoning_route"),
            egress="active_model_route",
            model=result.get("model"),
            packet_hash=result.get("packet_hash"),
            packet_bytes=result.get("packet_bytes"),
            selected_source_count=int(result.get("selected_source_count") or 0),
            citation_count=len(result.get("citations") or []),
            excluded_observations=int(result.get("excluded_observations") or 0),
            excluded_by=result.get("excluded_by") or {},
            outcome="success",
        )
    except Exception as exc:  # noqa: BLE001  # pragma: no cover - defensive best-effort audit
        _log.info("reasoning_send_ledger_skipped reason=%s", type(exc).__name__)


def _record_reasoning_send_error(
    *,
    feature: str,
    settings,
    packet_route: str | None,
    packet_audit: dict[str, Any] | None,
    error: BaseException,
) -> None:
    try:
        from openbird.llm.provider import classify_models

        remote_model = classify_models(settings).get("llm")
        if not remote_model:
            return
        audit = packet_audit or {}
        _record_reasoning_send(
            feature=feature,
            settings=settings,
            packet_route=packet_route,
            reasoning_route="cloud_reasoning_active",
            egress="active_model_route",
            model=remote_model,
            packet_hash=audit.get("packet_hash"),
            packet_bytes=audit.get("packet_bytes"),
            selected_source_count=int(audit.get("selected_source_count") or 0),
            citation_count=0,
            excluded_observations=int(audit.get("excluded_observations") or 0),
            excluded_by=audit.get("excluded_by") or {},
            outcome="error",
            error_kind=type(error).__name__,
        )
    except Exception as exc:  # noqa: BLE001  # pragma: no cover - defensive best-effort audit
        _log.info("reasoning_send_ledger_skipped reason=%s", type(exc).__name__)


def _record_reasoning_send(
    *,
    feature: str,
    settings,
    packet_route: str | None,
    reasoning_route: str | None,
    egress: str,
    model: str | None,
    packet_hash: str | None,
    packet_bytes: int | None,
    selected_source_count: int,
    citation_count: int,
    excluded_observations: int,
    excluded_by: dict[str, int],
    outcome: str,
    error_kind: str | None = None,
) -> None:
    """Best-effort redacted ledger write; never mask the command result."""
    try:
        from openbird.reasoning_ledger import advisory_route_class, provider_family

        store = _store_maintenance()
        try:
            store.record_reasoning_send(
                feature=feature,
                packet_route=packet_route,
                reasoning_route=reasoning_route,
                egress=egress,
                route_class=advisory_route_class(model, settings),
                provider_family=provider_family(model),
                model=model,
                packet_hash=packet_hash,
                packet_bytes=packet_bytes,
                selected_source_count=selected_source_count,
                citation_count=citation_count,
                excluded_observations=excluded_observations,
                excluded_by=excluded_by,
                outcome=outcome,
                error_kind=error_kind,
            )
        finally:
            store.close()
    except Exception as exc:  # noqa: BLE001  # pragma: no cover - defensive best-effort audit
        _log.info("reasoning_send_ledger_skipped reason=%s", type(exc).__name__)


def _deep_brain_day_windows(day: int, days: int) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    for offset in range(day + days - 1, day - 1, -1):
        start, end = _day_window(offset)
        windows.append(
            {
                "start": start,
                "end": end,
                "day_offset": offset,
                "local_date": _dt.datetime.fromtimestamp(start).date().isoformat(),
            }
        )
    return windows


def _group_rows_by_day_windows(rows, windows: list[dict[str, Any]]) -> list[list]:
    grouped: list[list] = [[] for _ in windows]
    for item in rows:
        obs, _text = item
        for idx, window in enumerate(windows):
            if float(window["start"]) <= obs.ts <= float(window["end"]):
                grouped[idx].append(item)
                break
    return grouped


def _build_deep_brain_period_packet(
    day: int, days: int, source_scope: str, settings
) -> dict[str, Any]:
    from openbird.deep_brain import build_deep_brain_period_preview

    windows = _deep_brain_day_windows(day, days)
    start = float(windows[0]["start"])
    end = float(windows[-1]["end"])
    store = _store_maintenance()
    try:
        rows = store.time_range_text(start, end, source=source_scope)
    finally:
        store.close()
    return build_deep_brain_period_preview(
        _group_rows_by_day_windows(rows, windows),
        day_windows=windows,
        day_offset=day,
        days=days,
        source_scope=source_scope,
        settings=settings,
    )


@app.command("timeline")
def timeline(
    day: int = typer.Option(0, "--day", help="Day offset: 0=today, 1=yesterday, ..."),
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Show a day's capture sessions (app, span, count) + active time.

    Pure local read: opens the store without the cloud gate or an LLM provider.
    """
    if day < 0:
        _err_console.print("[red]--day must be >= 0.[/]")
        raise typer.Exit(code=2)
    start, end = _day_window(day)
    store = _store_maintenance()
    try:
        sessions = store.day_sessions(start, end)
        active = store.active_seconds(start, end, get_settings().session_gap_seconds)
    finally:
        store.close()
    payload = {
        "day_offset": day,
        "start": start,
        "end": end,
        "total_observations": sum(s.count for s in sessions),
        "distinct_apps": len({s.app for s in sessions if s.app}),
        "active_seconds": active,
        "sessions": [
            {
                "session_id": s.session_id,
                "app": s.app,
                "start": s.start_ts,
                "end": s.end_ts,
                "count": s.count,
                "window": s.window,
            }
            for s in sessions
        ],
    }
    if as_json:
        _console.print_json(json.dumps(payload))
        return
    if not sessions:
        _console.print("[yellow]No capture sessions in the selected day.[/]")
        return
    table = Table(title=f"Timeline (day -{day})", show_header=True, header_style="bold")
    table.add_column("App")
    table.add_column("Start")
    table.add_column("End")
    table.add_column("Captures", justify="right")
    for s in sessions:
        table.add_row(
            s.app or "(unknown)", _fmt_ts(s.start_ts), _fmt_ts(s.end_ts), str(s.count)
        )
    _console.print(table)
    _console.print(
        f"{payload['total_observations']} captures · {payload['distinct_apps']} apps · "
        f"{active / 60:.0f} min active"
    )


@app.command("briefing")
def briefing(
    day: int = typer.Option(1, "--day", help="Day offset: 0=today, 1=yesterday, ..."),
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
    model: bool = typer.Option(
        False,
        "--model",
        help="Opt in to model-written prose (configured provider) instead of the "
        "default local deterministic day-memory summary.",
    ),
    exclude_app: list[str] = typer.Option(
        [],
        "--exclude-app",
        help="Only with --model: exclude an app/bundle id from the model packet.",
    ),
    exclude_source: list[str] = typer.Option(
        [],
        "--exclude-source",
        help="Only with --model: exclude an observation source from the model packet.",
    ),
    exclude_observation_id: list[str] = typer.Option(
        [],
        "--exclude-observation-id",
        help="Only with --model: exclude one observation id from the model packet.",
    ),
    signals: bool = typer.Option(
        False,
        "--signals",
        help="Use the experimental high-signal local classifier instead of broad prose.",
    ),
    week: bool = typer.Option(
        False,
        "--week",
        help="Render the WEEK overview (stored week digest + per-day narratives, "
        "local composition only) for the week containing the selected --day.",
    ),
) -> None:
    """Generate a grounded day briefing.

    Default (privacy-preserving): renders a deterministic, no-model, local-only
    summary distilled from the persisted day memory, with the same clickable source
    trail. No completion provider is constructed and no model is called.

    ``--model`` is the explicit opt-in escape hatch for model-written prose via the
    configured provider over an exclusion-filtered distilled day packet; a remote
    model still requires the cloud opt-in enforced at provider construction.
    ``--signals`` is the separate experimental local classifier path. ``--week``
    renders the week overview from STORED artifacts only (route
    ``local_cached_model_summary`` when cached prose is composed, else the
    deterministic aggregate). ``--model``, ``--signals``, and ``--week`` are
    mutually exclusive.
    """
    if day < 0:
        _err_console.print("[red]--day must be >= 0.[/]")
        raise typer.Exit(code=2)
    if sum(1 for flag in (model, signals, week) if flag) > 1:
        _err_console.print(
            "[red]--model, --signals, and --week are mutually exclusive.[/]"
        )
        raise typer.Exit(code=2)
    if not model and _has_cli_cloud_exclusions(
        exclude_app, exclude_source, exclude_observation_id
    ):
        _err_console.print(
            "[red]Cloud exclusion flags apply only to model briefings. "
            "Use --model, or omit the exclusion flags for the local briefing.[/]"
        )
        raise typer.Exit(code=2)

    if week:
        _briefing_week(day, as_json=as_json)
        return

    start, end = _day_window(day)
    if signals:
        _briefing_signals(day, start, end, as_json=as_json)
        return

    if model:
        _briefing_model(
            day,
            start,
            end,
            as_json=as_json,
            exclude_app=exclude_app,
            exclude_source=exclude_source,
            exclude_observation_id=exclude_observation_id,
        )
        return

    _briefing_local(day, start, end, as_json=as_json)


def _briefing_local(day: int, start: float, end: float, *, as_json: bool) -> None:
    """Default briefing: deterministic, no-model, local-only day-memory summary.

    When stored block summaries exist for the day, their chronological narrative
    is composed AFTER the facts prose via the SHARED
    :func:`openbird.day_memory.compose_day_narrative` helper (same one the chat
    day route uses, so the surfaces cannot drift) — and the reported
    ``reasoning_route`` truthfully flips to ``local_cached_model_summary``: the
    narrative is precomputed local-model prose (no provider call happens here).
    """
    from openbird.day_memory import (
        compose_day_narrative,
        day_memory_context,
        local_date_for_window,
        render_day_memory_prose,
    )
    from openbird.routines.templates import select_briefing_sources

    local_date = local_date_for_window(start)
    store = _store_maintenance()
    try:
        # Fetch the grounding rows for the source trail, and (re)build the persisted
        # day memory over the SAME [start, end] + source. Both reads use identical
        # bounds; on an OPEN day a row landing between the two reads could appear in
        # one but not the other: bounded and non-privacy (accepted tradeoff vs.
        # threading pre-fetched rows through the store API).
        rows = store.time_range_text(start, end, source="capture")
        saved = store.ensure_day_memory(
            local_date=local_date,
            start_ts=start,
            end_ts=end,
            day_offset=day,
            source_scope="capture",
        )
        # hasattr-guarded (parity with the chat route) so simpler store stubs
        # keep working; a missing reader simply composes no narrative.
        reader = getattr(store, "block_summaries_for_date", None)
        summaries = (reader(local_date) or []) if callable(reader) else []
        # Entity-ledger review nudge (Phase E2): a COUNT of dormant projects
        # still carrying unresolved open loops. Count only — entity names are
        # derived sensitive and never enter briefing text (which may be piped
        # or stored); `openbird entities list --status dormant` shows them
        # interactively.
        dormant_loops = _dormant_entities_with_unresolved_loops(store)
    finally:
        store.close()

    text = render_day_memory_prose(saved.get("payload", {}))
    reasoning_route = "local_deterministic"
    narrative, _summary_citations = compose_day_narrative(
        saved.get("payload", {}), summaries
    )
    if narrative:
        text = f"{text}\n\n{narrative}"
        reasoning_route = "local_cached_model_summary"
    if dormant_loops:
        text = (
            f"{text}\n\n{dormant_loops} dormant project(s) have unresolved "
            "open loops (see `openbird entities list --status dormant`)."
        )
    sources, total_sources = select_briefing_sources(rows)
    if as_json:
        _console.print_json(
            json.dumps(
                {
                    "day_offset": day,
                    "start": start,
                    "end": end,
                    "text": text,
                    # Route truthfulness: local_deterministic when no model prose
                    # is present; local_cached_model_summary when precomputed
                    # block-summary narrative was composed in (still no egress
                    # and no provider call at briefing time).
                    "reasoning_route": reasoning_route,
                    "memory_context": day_memory_context(saved),
                    "sources": sources,
                    # Full count of distinct grounding groups; > len(sources) means
                    # the trail was capped (UI shows "N of M"), never silent.
                    "sources_total": total_sources,
                }
            )
        )
        return
    _console.print(text)


def _briefing_week(day: int, *, as_json: bool) -> None:
    """Week briefing: STORED artifacts only, composed locally (never a model call).

    Renders the stored week digest + per-day block-summary narrative lines +
    deterministic totals via the SHARED :func:`openbird.day_memory.compose_week_answer`
    helper (same one the chat cached-week route uses, so the surfaces cannot
    drift). Route truthfulness: ``local_cached_model_summary`` when cached
    model prose was composed; otherwise a deterministic aggregate over stored
    day memories with ``local_deterministic``. Nothing is rebuilt and no
    provider is called.
    """
    from openbird.day_memory import compose_week_answer

    target = _dt.datetime.fromtimestamp(time.time() - day * 86_400.0).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    monday_dt = target - _dt.timedelta(days=target.weekday())
    monday = monday_dt.strftime("%Y-%m-%d")
    start = monday_dt.timestamp()
    end = (monday_dt + _dt.timedelta(days=7)).timestamp() - 1e-6

    store = _store_maintenance()
    try:
        week_row = store.get_week_memory(monday)
        day_entries = []
        for offset in range(7):
            local_date = (monday_dt + _dt.timedelta(days=offset)).strftime("%Y-%m-%d")
            saved = store.get_day_memory(local_date=local_date)
            summaries = store.block_summaries_for_date(local_date)
            day_entries.append((local_date, saved, summaries))
    finally:
        store.close()

    weeks = [week_row] if week_row else []
    text, citations, has_prose = compose_week_answer(weeks, day_entries)
    if has_prose:
        reasoning_route = "local_cached_model_summary"
    else:
        # Deterministic aggregate over whatever stored day memories exist.
        reasoning_route = "local_deterministic"
        total_seconds = 0.0
        days_with_data = 0
        for _local_date, saved, _summaries in day_entries:
            payload = (saved or {}).get("payload") or {}
            if int((payload.get("coverage") or {}).get("observations") or 0) > 0:
                days_with_data += 1
                total_seconds += float(
                    (payload.get("metrics") or {}).get("active_seconds") or 0.0
                )
        if days_with_data:
            text = (
                f"Week of {monday}: about {round(total_seconds / 60)} recorded "
                f"active minute(s) across {days_with_data} day(s) with stored "
                "day memories. No week digest or block summaries are stored "
                "yet; they are generated by the idle-time routines pass."
            )
        else:
            text = (
                f"No stored memories for the week of {monday} yet. Summaries "
                "are generated by the idle-time routines pass (or `openbird "
                "summaries build`)."
            )

    if as_json:
        _console.print_json(
            json.dumps(
                {
                    "week_start": monday,
                    "start": start,
                    "end": end,
                    "text": text,
                    "reasoning_route": reasoning_route,
                    "derived_citations": [
                        {
                            "index": c.index,
                            "source_id": c.source_id,
                            "type": c.type,
                            "label": c.label,
                            "snippet": c.snippet,
                            "derived_from": c.derived_from,
                            "derived_from_total": c.derived_from_total,
                            "derived_from_refs": c.derived_from_refs,
                        }
                        for c in citations
                    ],
                }
            )
        )
        return
    _console.print(text)


def _briefing_model(
    day: int,
    start: float,
    end: float,
    *,
    as_json: bool,
    exclude_app: list[str] | None = None,
    exclude_source: list[str] | None = None,
    exclude_observation_id: list[str] | None = None,
) -> None:
    """Opt-in briefing: model-written prose over a distilled day packet."""
    from openbird.deep_brain import (
        build_deep_brain_preview,
        complete_from_deep_brain_packet,
    )

    settings = _settings_with_cli_cloud_exclusions(
        get_settings(),
        exclude_app=exclude_app,
        exclude_source=exclude_source,
        exclude_observation_id=exclude_observation_id,
    )
    store = _store_maintenance()
    try:
        rows = store.time_range_text(start, end, source="capture")
    finally:
        store.close()

    packet = build_deep_brain_preview(
        rows,
        start_ts=start,
        end_ts=end,
        day_offset=day,
        source_scope="capture",
        settings=settings,
        blocked_reasons=[],
    )
    if not packet.get("selected_sources"):
        payload = {
            "day_offset": day,
            "start": start,
            "end": end,
            "text": "I do not have enough cited briefing evidence for that day.",
            "confidence": "insufficient_evidence",
            "grounded": False,
            "reasoning_route": "local_deterministic",
            "egress": "none",
            "packet_route": packet.get("route"),
            "packet_build_route": packet.get("packet_build_route"),
            "sources": [],
            "sources_total": packet.get("sources_total", 0),
            "exclusions": packet.get("exclusions", {}),
        }
        if as_json:
            _console.print_json(json.dumps(payload))
            return
        _console.print(payload["text"])
        return

    provider = _completion_provider(packet_label="day briefing packet")
    try:
        result = complete_from_deep_brain_packet(
            "Write a concise briefing for this day from the packet.",
            packet,
            provider,
            settings=settings,
            ungrounded_answer="I could not ground a model-written briefing in the day packet.",
        )
    except Exception as exc:  # noqa: BLE001 - provider/audit errors must preserve original failure
        _record_reasoning_send_error(
            feature="briefing.model",
            settings=settings,
            packet_route=packet.get("route"),
            packet_audit=_safe_packet_audit_from_deep_brain_packet(packet),
            error=exc,
        )
        raise
    _record_reasoning_send_success(
        feature="briefing.model", settings=settings, result=result
    )
    sources = _briefing_sources_from_packet_citations(result.get("citations", []))

    if as_json:
        _console.print_json(
            json.dumps(
                {
                    "day_offset": day,
                    "start": start,
                    "end": end,
                    "text": result["answer"],
                    "confidence": result.get("confidence"),
                    "grounded": result.get("grounded", False),
                    "reasoning_route": result.get("reasoning_route"),
                    "egress": result.get("egress"),
                    "packet_route": result.get("packet_route"),
                    "packet_build_route": result.get("packet_build_route"),
                    "sources": sources,
                    "sources_total": packet.get("sources_total", 0),
                    "exclusions": packet.get("exclusions", {}),
                }
            )
        )
        return
    _console.print(result["answer"])


def _briefing_sources_from_packet_citations(citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert packet citation metadata back to the briefing source JSON shape."""
    return [
        {
            "observation_id": source.get("observation_id"),
            "app": source.get("app"),
            "window": source.get("window_or_url"),
            "ts": source.get("ts"),
            "snippet": source.get("snippet"),
        }
        for source in citations
    ]


def _briefing_signals(day: int, start: float, end: float, *, as_json: bool) -> None:
    """Run the opt-in signal-first briefing path.

    The signal classifier is local-only in this experimental path. If the active
    model route is remote, or local model construction/call fails, it degrades to
    deterministic per-item fallback instead of silently using cloud.
    """
    from openbird.llm.provider import classify_models
    from openbird.signals import SignalClassifier, render_signal_brief

    store = _store_maintenance()
    try:
        rows = store.time_range_text(start, end, source="capture")
    finally:
        store.close()

    settings = get_settings()
    provider = None
    local_model_status = "not_needed"
    if rows:
        remote = classify_models(settings)
        if remote:
            local_model_status = "disabled_remote_route"
        else:
            try:
                provider = _provider()
                local_model_status = "available"
            except Exception:  # noqa: BLE001 - signal path degrades locally
                # Provider construction can fail because the local route is not
                # ready. The signal path is experimental/local-only, so it treats
                # that as deterministic fallback rather than using remote
                # completion.
                local_model_status = "unavailable"

    classifier = SignalClassifier(provider)
    result = classifier.classify_window(
        rows,
        start_ts=start,
        end_ts=end,
        local_model_status=local_model_status,
    )
    text = render_signal_brief(result)
    if as_json:
        _console.print_json(
            json.dumps(
                {
                    "day_offset": day,
                    "start": start,
                    "end": end,
                    "text": text,
                    "signals": [
                        {
                            "candidate_id": s.candidate_id,
                            "label": s.label,
                            "confidence": s.confidence,
                            "user_value": s.user_value,
                            "short_label": s.short_label,
                            "evidence_observation_ids": list(s.evidence_observation_ids),
                            "reason_codes": list(s.reason_codes),
                            "deterministic_fallback": s.deterministic_fallback,
                        }
                        for s in result.signals
                    ],
                    "hidden_count": result.hidden_count,
                    "grouped_duplicates_count": result.grouped_duplicates_count,
                    "low_confidence_count": result.low_confidence_count,
                    "deterministic_fallback_count": result.deterministic_fallback_count,
                    "sensitive_quarantine_count": result.sensitive_quarantine_count,
                    "local_model_status": result.local_model_status,
                }
            )
        )
        return
    _console.print(text)


# --------------------------------------------------------------------------- #
# Deep Brain preview                                                          #
# --------------------------------------------------------------------------- #


@deep_brain_app.command("status")
def deep_brain_status(
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Show Deep Brain opt-in and exclusion status without reading memory."""
    payload = _deep_brain_status_payload(get_settings())
    if as_json:
        _console.print_json(json.dumps(payload))
        return

    exclusions = payload["exclusions"]
    _console.print(f"Deep Brain: {payload['route_label']}")
    _console.print(
        f"Feature gate: {'on' if payload['deep_brain_enabled'] else 'off'} "
        f"({payload['env_vars']['deep_brain_enabled']})"
    )
    _console.print(
        f"Cloud opt-in: {'on' if payload['cloud_opt_in'] else 'off'} "
        f"({payload['env_vars']['cloud_opt_in']})"
    )
    if payload["cloud_blocked_reasons"]:
        _console.print("Cloud gates missing:")
        for reason in payload["cloud_blocked_reasons"]:
            _console.print(f"- {reason}")
    if payload["ask_blocked_reasons"]:
        _console.print("Ask gates missing:")
        for reason in payload["ask_blocked_reasons"]:
            _console.print(f"- {reason}")
    apps = exclusions["excluded_apps_configured"]
    sources = exclusions["excluded_sources_configured"]
    _console.print(
        "Exclusions: "
        f"{len(apps)} app(s), {len(sources)} source(s), "
        f"{exclusions['excluded_observation_ids_configured']} observation id(s)"
    )
    if apps:
        _console.print(f"Excluded apps: {', '.join(apps)}")
    if sources:
        _console.print(f"Excluded sources: {', '.join(sources)}")
    _console.print("No data was sent.")


@deep_brain_app.command("preview")
def deep_brain_preview(
    day: int = typer.Option(0, "--day", help="Day offset: 0=today, 1=yesterday, ..."),
    days: int = typer.Option(1, "--days", help="Trailing local days to include."),
    source_scope: str = typer.Option(
        "capture", "--source-scope", help="Observation source to preview."
    ),
    exclude_app: list[str] = typer.Option(
        [],
        "--exclude-app",
        help="Exclude an app/bundle id from the preview packet.",
    ),
    exclude_source: list[str] = typer.Option(
        [],
        "--exclude-source",
        help="Exclude an observation source from the preview packet.",
    ),
    exclude_observation_id: list[str] = typer.Option(
        [],
        "--exclude-observation-id",
        help="Exclude one observation id from the preview packet.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Build a local-only Deep Brain packet preview.

    The preview is a consent surface: it applies configured exclusions before
    distillation and shows what a future cloud reasoning route would be eligible
    to use. It never constructs the configured LLM/model provider and never
    sends data off this Mac. Store access uses the local maintenance provider
    stub, whose embed/complete methods fail closed if accidentally called.
    """
    if day < 0:
        _err_console.print("[red]--day must be >= 0.[/]")
        raise typer.Exit(code=2)
    _validate_deep_brain_days(days)
    settings = _settings_with_cli_cloud_exclusions(
        get_settings(),
        exclude_app=exclude_app,
        exclude_source=exclude_source,
        exclude_observation_id=exclude_observation_id,
    )
    packet = _build_deep_brain_period_packet(day, days, source_scope, settings)
    if as_json:
        _console.print_json(json.dumps(packet))
        return
    ready = "ready" if packet["cloud_ready"] else "not ready"
    label = packet["local_date"]
    if packet.get("period"):
        period = packet["period"]
        label = f"{period['start_local_date']}..{period['end_local_date']}"
    _console.print(
        f"Deep Brain preview for {label} ({ready}); "
        f"{packet['exclusions']['kept_observations']} kept, "
        f"{packet['exclusions']['excluded_observations']} excluded. "
        "No data was sent."
    )


@deep_brain_app.command("ask")
def deep_brain_ask(
    question: Optional[str] = typer.Argument(None, help="Question to answer from the day packet."),
    day: int = typer.Option(0, "--day", help="Day offset: 0=today, 1=yesterday, ..."),
    days: int = typer.Option(1, "--days", help="Trailing local days to include."),
    source_scope: str = typer.Option(
        "capture", "--source-scope", help="Observation source to distill."
    ),
    exclude_app: list[str] = typer.Option(
        [],
        "--exclude-app",
        help="Exclude an app/bundle id from the Deep Brain packet.",
    ),
    exclude_source: list[str] = typer.Option(
        [],
        "--exclude-source",
        help="Exclude an observation source from the Deep Brain packet.",
    ),
    exclude_observation_id: list[str] = typer.Option(
        [],
        "--exclude-observation-id",
        help="Exclude one observation id from the Deep Brain packet.",
    ),
    stdin: bool = typer.Option(False, "--stdin", help="Read the question from stdin."),
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Ask the configured model over the exact Deep Brain preview packet."""
    if day < 0:
        _err_console.print("[red]--day must be >= 0.[/]")
        raise typer.Exit(code=2)
    _validate_deep_brain_days(days)
    if stdin:
        question = sys.stdin.read().strip()
    if not question:
        _err_console.print("[red]Provide a question or pass --stdin.[/]")
        raise typer.Exit(code=2)

    from openbird.deep_brain import (
        answer_deep_brain,
        deep_brain_ask_blocked_reasons,
    )

    settings = _settings_with_cli_cloud_exclusions(
        get_settings(),
        exclude_app=exclude_app,
        exclude_source=exclude_source,
        exclude_observation_id=exclude_observation_id,
    )
    packet = _build_deep_brain_period_packet(day, days, source_scope, settings)
    blocked = deep_brain_ask_blocked_reasons(settings)
    if blocked:
        payload = {
            "ok": False,
            "answer": "Deep Brain ask is not enabled.",
            "blocked_reasons": blocked,
            "reasoning_route": "blocked",
            "egress": "none",
            "packet_route": packet.get("route"),
            "packet": {
                "local_date": packet.get("local_date"),
                "day_offset": packet.get("day_offset"),
                "source_scope": packet.get("source_scope"),
                "period": packet.get("period"),
                "coverage": packet.get("memory_summary", {}).get("coverage", {}),
                "sources_total": packet.get("sources_total", 0),
                "exclusions": packet.get("exclusions", {}),
            },
        }
        if as_json:
            _console.print_json(json.dumps(payload))
        else:
            _err_console.print("[red]Deep Brain ask refused:[/]")
            for reason in blocked:
                _err_console.print(f"- {reason}")
        raise typer.Exit(code=2)

    provider = _completion_provider()
    try:
        result = answer_deep_brain(question, packet, provider, settings=settings)
    except Exception as exc:  # noqa: BLE001 - provider/audit errors must preserve original failure
        _record_reasoning_send_error(
            feature="deep_brain.ask",
            settings=settings,
            packet_route=packet.get("route"),
            packet_audit=_safe_packet_audit_from_deep_brain_packet(packet),
            error=exc,
        )
        raise
    _record_reasoning_send_success(
        feature="deep_brain.ask", settings=settings, result=result
    )
    if as_json:
        _console.print_json(json.dumps(result))
        return
    _console.print(result["answer"])


# --------------------------------------------------------------------------- #
# deterministic day memory                                                    #
# --------------------------------------------------------------------------- #


@day_memory_app.command("build")
def day_memory_build(
    day: int = typer.Option(0, "--day", help="Day offset: 0=today, 1=yesterday, ..."),
    source_scope: str = typer.Option(
        "capture", "--source-scope", help="Observation source to distill."
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Build and persist a deterministic daily memory artifact.

    This path is local-only and no-model: it computes descriptive metrics and
    categories from captured observations, then stores them with source
    dependencies so purge deletes the derived artifact too. No narrative prose is
    persisted.
    """
    if day < 0:
        _err_console.print("[red]--day must be >= 0.[/]")
        raise typer.Exit(code=2)

    from openbird.day_memory import local_date_for_window

    start, end = _day_window(day)
    local_date = local_date_for_window(start)
    store = _store_maintenance()
    try:
        saved = store.ensure_day_memory(
            local_date=local_date,
            start_ts=start,
            end_ts=end,
            day_offset=day,
            source_scope=source_scope,
            force=True,
        )
    finally:
        store.close()

    payload = {"built": True, "day_memory": saved}
    if as_json:
        _console.print_json(data=payload)
        return
    coverage = saved["payload"]["coverage"]
    _console.print(
        f"[green]Built[/] day memory for {local_date}: "
        f"{coverage['observations']} observation(s), "
        f"{coverage['sessions']} session(s)."
    )


@day_memory_app.command("show")
def day_memory_show(
    day: int = typer.Option(0, "--day", help="Day offset: 0=today, 1=yesterday, ..."),
    source_scope: str = typer.Option(
        "capture", "--source-scope", help="Observation source to inspect."
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Show a fresh deterministic daily memory artifact, building if needed."""
    if day < 0:
        _err_console.print("[red]--day must be >= 0.[/]")
        raise typer.Exit(code=2)

    from openbird.day_memory import local_date_for_window, saved_day_memory_with_day_offset

    start, _end = _day_window(day)
    local_date = local_date_for_window(start)
    store = _store_maintenance()
    try:
        saved = store.ensure_day_memory(
            local_date=local_date,
            start_ts=start,
            end_ts=_end,
            day_offset=day,
            source_scope=source_scope,
            force=False,
        )
    finally:
        store.close()

    if saved is None:
        payload = {
            "built": False,
            "local_date": local_date,
            "source_scope": source_scope,
            "message": "day memory has not been built",
        }
        if as_json:
            _console.print_json(data=payload)
        else:
            _console.print(
                f"[yellow]No day memory built[/] for {local_date} ({source_scope}). "
                "Run `openbird day-memory build --day N` first."
            )
        raise typer.Exit(code=1)

    display_saved = saved_day_memory_with_day_offset(saved, day)
    payload = {"built": True, "day_memory": display_saved}
    if as_json:
        _console.print_json(data=payload)
        return
    metrics = display_saved["payload"]["metrics"]
    _console.print(f"[bold]{local_date}[/] · {display_saved['source_count']} source(s)")
    _console.print_json(data=metrics)


# --------------------------------------------------------------------------- #
# productivity facts                                                          #
# --------------------------------------------------------------------------- #


@app.command("productivity")
def productivity(
    day: int = typer.Option(0, "--day", help="Day offset: 0=today, 1=yesterday, ..."),
    source_scope: str = typer.Option(
        "capture", "--source-scope", help="Observation source to analyze."
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Show local-only productivity facts for a day.

    This is deterministic and provider-free: it ensures the day-memory artifact,
    then projects factual metrics and cited source ids from that local payload.
    """
    if day < 0:
        _err_console.print("[red]--day must be >= 0.[/]")
        raise typer.Exit(code=2)

    from openbird.day_memory import build_productivity_report, local_date_for_window

    start, end = _day_window(day)
    local_date = local_date_for_window(start)
    store = _store_maintenance()
    try:
        saved = store.ensure_day_memory(
            local_date=local_date,
            start_ts=start,
            end_ts=end,
            day_offset=day,
            source_scope=source_scope,
            force=False,
        )
    finally:
        store.close()

    report = build_productivity_report(saved, day_offset=day)
    if as_json:
        _console.print_json(data=report)
        return

    facts = report["productivity"]["facts"]
    top_category = facts.get("top_category")
    longest = facts.get("longest_focus_block")
    _console.print(
        f"[bold]{report['local_date']}[/] · "
        f"{facts['active_minutes']} active minute(s), "
        f"{facts['context_switch_count']} context switch(es), "
        f"{facts['context_switches_per_active_hour']} switch(es)/active hour."
    )
    if top_category:
        _console.print(
            "Top category: "
            f"{top_category['category']} ({top_category['minutes']}m, "
            f"{top_category['source_count']} source(s))."
        )
    if longest:
        _console.print(
            "Longest focus block: "
            f"{longest['category']} ({round(float(longest['seconds']) / 60.0, 1)}m, "
            f"{longest['session_count']} session(s))."
        )


@app.command("productivity-coach")
def productivity_coach(
    question: Optional[str] = typer.Argument(
        None, help="Coaching question to answer from local productivity facts."
    ),
    day: int = typer.Option(0, "--day", help="Day offset: 0=today, 1=yesterday, ..."),
    source_scope: str = typer.Option(
        "capture", "--source-scope", help="Observation source to analyze."
    ),
    exclude_app: list[str] = typer.Option(
        [],
        "--exclude-app",
        help="Exclude an app/bundle id from the coaching packet.",
    ),
    exclude_source: list[str] = typer.Option(
        [],
        "--exclude-source",
        help="Exclude an observation source from the coaching packet.",
    ),
    exclude_observation_id: list[str] = typer.Option(
        [],
        "--exclude-observation-id",
        help="Exclude one observation id from the coaching packet.",
    ),
    stdin: bool = typer.Option(False, "--stdin", help="Read the question from stdin."),
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Ask the configured model for cited coaching over local productivity facts."""
    if day < 0:
        _err_console.print("[red]--day must be >= 0.[/]")
        raise typer.Exit(code=2)
    if stdin:
        question = sys.stdin.read().strip()
    if not question:
        _err_console.print("[red]Provide a question or pass --stdin.[/]")
        raise typer.Exit(code=2)

    from openbird.day_memory import (
        answer_productivity_coach,
        build_productivity_coach_packet,
        build_productivity_coach_report,
        productivity_coach_blocked_reasons,
    )

    start, end = _day_window(day)
    settings = _settings_with_cli_cloud_exclusions(
        get_settings(),
        exclude_app=exclude_app,
        exclude_source=exclude_source,
        exclude_observation_id=exclude_observation_id,
    )
    store = _store_maintenance()
    try:
        rows = store.time_range_text(start, end, source=source_scope)
    finally:
        store.close()

    report = build_productivity_coach_report(
        rows,
        start_ts=start,
        end_ts=end,
        day_offset=day,
        source_scope=source_scope,
        settings=settings,
    )
    packet = build_productivity_coach_packet(report)
    blocked = productivity_coach_blocked_reasons(settings)
    if blocked:
        payload = {
            "ok": False,
            "answer": "Productivity coaching is not enabled.",
            "blocked_reasons": blocked,
            "reasoning_route": "blocked",
            "egress": "none",
            "packet_route": packet.get("route"),
            "packet": {
                "local_date": packet.get("local_date"),
                "day_offset": packet.get("day_offset"),
                "source_scope": packet.get("source_scope"),
                "citation_count": packet.get("citation_count", 0),
                "exclusions": packet.get("exclusions", {}),
            },
        }
        if as_json:
            _console.print_json(json.dumps(payload))
        else:
            _err_console.print("[red]Productivity coaching refused:[/]")
            for reason in blocked:
                _err_console.print(f"- {reason}")
        raise typer.Exit(code=2)

    if packet["citation_count"] == 0:
        payload = {
            "ok": True,
            "question": question,
            "answer": "I do not have enough cited productivity evidence to coach on that.",
            "confidence": "insufficient_evidence",
            "grounded": False,
            "reasoning_route": "local_deterministic",
            "egress": "none",
            "model": None,
            "packet_route": packet["route"],
            "citations": [],
            "local_date": report.get("local_date"),
            "source_scope": report.get("source_scope"),
            "exclusions": packet.get("exclusions", {}),
        }
        if as_json:
            _console.print_json(json.dumps(payload))
            return
        _console.print(payload["answer"])
        return

    provider = _completion_provider(packet_label="productivity coaching packet")
    try:
        result = answer_productivity_coach(
            question, report, provider, settings=settings, packet=packet
        )
    except Exception as exc:  # noqa: BLE001 - provider/audit errors must preserve original failure
        _record_reasoning_send_error(
            feature="productivity.coach",
            settings=settings,
            packet_route=packet.get("route"),
            packet_audit=_safe_packet_audit_from_productivity_packet(packet),
            error=exc,
        )
        raise
    _record_reasoning_send_success(
        feature="productivity.coach", settings=settings, result=result
    )
    if as_json:
        _console.print_json(json.dumps(result))
        return
    _console.print(result["answer"])


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
    elif reachable == "n/a":
        o_status, o_detail = "n/a", "not used by the active (cloud/mlx) route"
    else:
        o_status, o_detail = "down", f"unreachable ({ollama.get('error')})"
    table.add_row("ollama", o_status, f"{ollama.get('host')} · {o_detail}")

    emb = report["embedding"]
    emb_detail = f"{emb['model']} dim={emb['configured_dim']}"
    if emb.get("probed"):
        emb_detail += f" probed={emb.get('probed_dim')} ok={emb.get('dim_ok')}"
    table.add_row("embedding", "info", emb_detail)

    comp = report.get("completion", {})
    comp_detail = str(comp.get("model"))
    if comp.get("probed"):
        comp_detail += f" probe_ok={comp.get('ok')}"
    table.add_row("chat-model", "info", comp_detail)

    bk = report.get("backend", {})
    bk_supported = bk.get("supported", True)
    table.add_row(
        "backend",
        "ok" if bk_supported else "unsupported",
        f"{bk.get('name')}" + ("" if bk_supported else " (not wired; reserved)"),
    )

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
        f"ocr={priv['ocr_enabled']} ocr_apps={priv.get('ocr_apps', 0)}",
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
# doctor                                                                      #
# --------------------------------------------------------------------------- #


@app.command()
def doctor(
    json_out: bool = typer.Option(
        False, "--json", help="Emit the redacted diagnostic as JSON."
    ),
    no_ollama: bool = typer.Option(
        False, "--no-ollama", help="Skip the Ollama network probe."
    ),
) -> None:
    """Print a content-safe diagnostic to share when reporting an issue.

    Reuses preflight and adds code-signing identity, quarantine state, and
    allowlist status. Home paths are redacted, secrets scrubbed, and allow/block
    lists reduced to counts; the report never contains captured text. Never raises.
    """
    from openbird.doctor import build_doctor_report, render

    # Resolve settings INSIDE build_doctor_report's never-crash boundary (a bad
    # env or unwritable data dir must still yield a diagnostic, not a traceback).
    report = build_doctor_report(probe_ollama=not no_ollama)
    if json_out:
        _console.print_json(json.dumps(report))
    else:
        _console.print(render(report))
    raise typer.Exit(code=0 if report.get("runtime_ok") else 1)


# --------------------------------------------------------------------------- #
# uninstall                                                                   #
# --------------------------------------------------------------------------- #


@app.command()
def uninstall(
    purge_data: bool = typer.Option(
        False,
        "--purge-data",
        help="Also delete the data dir (~/.openbird). Irreversible — destroys "
        "captured observations.",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print exactly what would be removed; touch nothing."
    ),
) -> None:
    """Remove OpenBird's system state (LaunchAgent, Launch Services, Keychain key).

    Clears the leftovers that linger after the app is trashed. Captured data in
    ~/.openbird is PRESERVED unless --purge-data is given. The Keychain DB key is
    removed only when no encrypted DB depends on it (else it is retained so an
    encrypted DB is never stranded). Run `openbird uninstall --dry-run` first to
    preview. Reports only paths and reason codes — never captured content.
    """
    from openbird.uninstall import run_uninstall

    if not dry_run and not yes:
        target = (
            "system state AND ALL data (~/.openbird)"
            if purge_data
            else "system state (data preserved)"
        )
        if not typer.confirm(f"Remove OpenBird {target}?", default=False):
            _console.print("[yellow]Aborted.[/]")
            raise typer.Exit(code=1)

    results = run_uninstall(purge_data=purge_data, dry_run=dry_run)

    table = Table(title="OpenBird uninstall" + (" (dry-run)" if dry_run else ""))
    table.add_column("Step")
    table.add_column("Status")
    table.add_column("Detail")
    _styles = {
        "done": "green",
        "would": "cyan",
        "skip": "dim",
        "retained": "yellow",
        "error": "red",
    }
    for r in results:
        table.add_row(
            r.action, f"[{_styles.get(r.status, 'white')}]{r.status}[/]", r.detail
        )
    _console.print(table)

    if any(r.status == "retained" for r in results):
        from openbird.config import data_dir_path, db_file_path

        db = db_file_path().resolve()
        data_dir = data_dir_path().resolve()
        db_inside_data_dir = db == data_dir or data_dir in db.parents
        if db_inside_data_dir:
            remedy = "Re-run with --purge-data to remove both."
        else:
            # An external OPENBIRD_DB_PATH is NOT removed by --purge-data, so
            # suggesting it would be misleading (CodeRabbit).
            remedy = (
                f"The encrypted DB at {db} is outside the data dir; handle it "
                "separately (decrypt/move/delete), then re-run uninstall."
            )
        _console.print(
            "[yellow]Note:[/] the Keychain key was kept because an encrypted DB "
            f"still depends on it. {remedy}"
        )
    had_error = any(r.status == "error" for r in results)
    if had_error:
        _err_console.print("[red]Some steps failed — see 'error' rows above.[/]")
    raise typer.Exit(code=1 if had_error else 0)


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

    files, escaped = _collect_files(path, glob=glob, max_bytes=max_bytes)
    if not files:
        if escaped:
            # Everything that matched escaped the selected root via a symlink —
            # surface the refusal so this is not silently a no-op.
            _err_console.print(
                "[yellow]No matching files to ingest[/] "
                f"({escaped} skipped: symlink escapes the selected directory)."
            )
        else:
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
                # Use the literal selected path (containment already verified in
                # _collect_files), never a resolved path that could point
                # outside the selected root. ``absolute()`` makes a relative
                # input (e.g. ``openbird ingest notes.txt``) URI-expressible
                # without resolving symlinks back out of the root the way
                # ``resolve()`` would.
                url=fp.absolute().as_uri(),
                source="ingest",
            )
            ingested += 1
    finally:
        store.close()

    msg = f"[green]Ingested[/] {ingested} file(s); skipped {skipped}."
    if escaped:
        msg += f" Refused {escaped} symlink(s) that escape the selected directory."
    _console.print(msg)


def _is_within(candidate: Path, root: Path) -> bool:
    """True iff ``candidate``'s real path is contained within ``root``'s real path.

    Both sides are canonicalized with ``Path.resolve()`` (which follows symlinks),
    so a symlink whose target lives outside ``root`` is rejected. ``root`` itself
    counts as contained (a directory is its own ancestor for our purposes).
    """
    try:
        real_candidate = candidate.resolve()
        real_root = root.resolve()
    except OSError:
        return False
    return real_candidate == real_root or real_candidate.is_relative_to(real_root)


def _collect_files(
    path: Path, *, glob: str, max_bytes: int
) -> tuple[list[Path], int]:
    """Resolve PATH to a sorted list of in-root regular files under the size cap.

    Returns ``(files, escaped)`` where ``escaped`` counts candidates skipped
    because their real (symlink-resolved) path lies outside the selected root —
    a directory walk must never ingest content from outside the directory the
    user selected. ``rglob`` does not recurse *into* symlinked directories, and
    the containment check below additionally rejects symlinked files (and the
    symlinked-directory entries themselves) whose target escapes the root.
    """
    if path.is_file():
        # A single explicit file argument: honor it as long as it resolves
        # within its own parent directory (the intuitive root for one path). A
        # symlink to an arbitrary external location is refused.
        root = path.parent
        candidates = [path]
    else:
        root = path
        candidates = sorted(path.rglob(glob))

    out: list[Path] = []
    escaped = 0
    for p in candidates:
        if not _is_within(p, root):
            # Symlink (file or dir) whose real target escapes the selected root.
            escaped += 1
            continue
        try:
            if not p.is_file():
                continue
            if p.stat().st_size > max_bytes:
                continue
        except OSError:
            continue
        out.append(p)
    return out, escaped


# --------------------------------------------------------------------------- #
# chat                                                                        #
# --------------------------------------------------------------------------- #


@app.command()
def chat(
    question: Optional[str] = typer.Argument(None, help="A natural-language question."),
    k: int = typer.Option(10, "--k", help="Retrieval depth."),
    no_semantic: bool = typer.Option(
        False, "--no-semantic", help="BM25-only retrieval (skip the embedding call)."
    ),
    day: Optional[int] = typer.Option(
        None,
        "--day",
        help="Hard-scope the answer to one calendar day's observations "
        "(0=today, 1=yesterday, ...). Omit for unscoped retrieval.",
    ),
    json_out: bool = typer.Option(
        False, "--json", help="Emit the answer + citations as JSON (used by the app UI)."
    ),
    stdin: bool = typer.Option(
        False,
        "--stdin",
        help="Read the question from stdin so it never appears in argv (used by the app UI).",
    ),
) -> None:
    """Answer a question grounded in your captured memory, with citations.

    Runs hybrid retrieval over the store, builds an injection-resistant grounded
    prompt, asks the LLM, then prints the answer plus occurrence-level citations
    (app / window / time + a short snippet) that name where each fact came from.

    Pass ``--day N`` (0=today, 1=yesterday, ..., matching ``timeline``/``briefing``)
    to HARD-SCOPE the answer to that calendar day: retrieval and every citation are
    confined to that day's observations. Without ``--day`` retrieval is unscoped.

    Pass ``--stdin`` (or omit the argument) to read the question from stdin; the
    app UI uses this so chat text never lands in the process argument list.
    """
    import sys

    if day is not None and day < 0:
        _err_console.print("[red]--day must be >= 0.[/]")
        raise typer.Exit(code=2)

    if stdin or question is None:
        question = sys.stdin.read()
    question = (question or "").strip()
    if not question:
        _console.print("[red]No question provided.[/]")
        raise typer.Exit(code=2)

    from openbird.chat.rag import RAG

    # An explicit --day hard-scopes retrieval to that calendar day's window
    # (shared with `timeline`/`briefing` via `_day_window`), so the answer and
    # every citation are confined to that day. Omitted -> unscoped.
    window = _day_window(day) if day is not None else None

    if window is not None:
        local_store = _store_maintenance()
        try:
            local_rag = RAG(local_store, local_store.provider)
            local_result = local_rag.answer_deterministic_day_memory(question, window)
        finally:
            local_store.close()
        if local_result is not None:
            _render_chat_result(local_result, json_out=json_out)
            return

    provider = _provider()
    store = _store(provider=provider)
    try:
        rag = RAG(store, provider)
        result = rag.answer(question, k=k, semantic=not no_semantic, window=window)
    finally:
        store.close()

    _render_chat_result(result, json_out=json_out)


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
    """Run the routine scheduler as a foreground daemon until SIGINT/SIGTERM.

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
    """Write the per-user LaunchAgent so routines run at login.

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
    """Remove the per-user LaunchAgent for the routine daemon."""
    import subprocess

    from openbird.routines.launchd import agent_plist_path

    path = agent_plist_path()
    unload_failed = False
    if unload and path.exists():
        try:
            subprocess.run(["launchctl", "unload", str(path)], check=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            _err_console.print(f"[red]launchctl unload failed:[/] {type(exc).__name__}")
            unload_failed = True
    if path.exists():
        path.unlink()
        _console.print(f"[green]Removed LaunchAgent:[/] {path}")
    else:
        _console.print("[yellow]No LaunchAgent installed.[/]")
    if unload_failed:
        # The plist was removed, but launchd may still hold the (now orphaned)
        # job; surface that as a nonzero exit instead of pretending success.
        _err_console.print(
            "[yellow]Note:[/] the agent file was removed but `launchctl unload` "
            "failed; the running job may persist until logout or a manual unload."
        )
        raise typer.Exit(code=1)


# --------------------------------------------------------------------------- #
# summaries (Phase D block summaries)                                         #
# --------------------------------------------------------------------------- #


def _parse_local_date(value: str) -> tuple[float, float]:
    """Return the [start, end] bounds of a local ``YYYY-MM-DD`` day."""
    try:
        day = _dt.datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        _err_console.print(f"[red]Invalid --date:[/] expected YYYY-MM-DD, got {value!r}")
        raise typer.Exit(code=2) from exc
    start = day.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + _dt.timedelta(days=1) - _dt.timedelta(microseconds=1)
    return start.timestamp(), end.timestamp()


@summaries_app.command("build")
def summaries_build(
    date: Optional[str] = typer.Option(
        None, "--date", help="Local day to build (YYYY-MM-DD); default: trailing lookback."
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help=(
            "Bypass the battery/idle/meeting gate and the settle delay (NEVER "
            "the cloud gate). Tradeoff: forcing may summarize a still-active "
            "block whose membership later changes (e.g. a mid-block meeting "
            "split re-keys it), leaving a stale overlapping summary until its "
            "sources change; the settled routine path cannot hit this."
        ),
    ),
) -> None:
    """Build block summaries on demand (same bounded runner the routine uses).

    The shared runner also covers the week-rollup and summary-index steps
    automatically (Phase E1). Output is counts + reason codes only — summary
    and digest bodies stay in the encrypted memory DB (`openbird summaries
    list` / `list --weeks` shows them interactively).
    """
    from openbird.summaries import format_counts_line, run_block_summaries

    window = _parse_local_date(date) if date else None
    provider = _provider()
    store = _store(provider=provider)
    try:
        counts = run_block_summaries(
            store,
            provider,
            now=time.time(),
            settings=get_settings(),
            force=force,
            window=window,
        )
    finally:
        store.close()
    _console.print(f"[green]block summaries:[/] {format_counts_line(counts)}")


@summaries_app.command("list")
def summaries_list(
    date: Optional[str] = typer.Option(
        None, "--date", help="Local day to list (YYYY-MM-DD); default: today."
    ),
    weeks: bool = typer.Option(
        False,
        "--weeks",
        help="List stored WEEK digests (newest first) instead of a day's "
        "block summaries.",
    ),
) -> None:
    """List stored block summaries for one local day (or week digests).

    Summary/digest text (derived sensitive) is printed ONLY on an interactive
    terminal; a piped/captured invocation gets metadata (times, model, source
    counts) so summary bodies never land in logs or scrollback files.
    """
    if weeks:
        if date:
            _err_console.print("[red]--weeks lists all stored week digests; "
                               "--date applies only to block summaries.[/]")
            raise typer.Exit(code=2)
        _summaries_list_weeks()
        return
    local_date = date or _dt.date.today().isoformat()
    if date:
        _parse_local_date(date)  # validate format
    store = _store_maintenance()
    try:
        rows = store.block_summaries_for_date(local_date)
    finally:
        store.close()
    if not rows:
        _console.print(f"[yellow]No block summaries stored for {local_date}.[/]")
        return
    interactive = sys.stdout.isatty()
    for row in rows:
        header = (
            f"{_fmt_ts(row['start_ts'])} -> {_fmt_ts(row['end_ts'])} "
            f"bundle={row.get('dominant_bundle') or '-'} "
            f"level={row.get('level') or '-'} sources={row.get('source_count', 0)} "
            f"model={row.get('model')}"
        )
        _console.print(f"[bold]{escape(header)}[/]")
        if interactive:
            _console.print(f"  {escape(str(row.get('summary_text') or ''))}")
    if not interactive:
        _console.print(
            f"[dim]{len(rows)} summary bodies withheld (non-interactive output).[/]"
        )


def _summaries_list_weeks() -> None:
    """List stored week digests (same interactive-only body-printing rule)."""
    store = _store_maintenance()
    try:
        # Week rows are few (one per ISO week); a wide finite overlap window
        # returns them all, ordered by Monday date.
        rows = store.week_memories_overlapping(0.0, time.time() + 366 * 86_400.0)
    finally:
        store.close()
    if not rows:
        _console.print("[yellow]No week digests stored yet.[/]")
        return
    interactive = sys.stdout.isatty()
    for row in reversed(rows):  # newest first
        payload = row.get("payload") or {}
        header = (
            f"week of {row.get('local_date')} "
            f"members={len(row.get('summary_ids') or [])} "
            f"model={payload.get('model') or '-'} "
            f"generated={_fmt_ts(row.get('generated_at'))}"
        )
        _console.print(f"[bold]{escape(header)}[/]")
        if interactive:
            _console.print(f"  {escape(str(payload.get('digest_text') or ''))}")
    if not interactive:
        _console.print(
            f"[dim]{len(rows)} digest bodies withheld (non-interactive output).[/]"
        )


# --------------------------------------------------------------------------- #
# entities (Phase E2)                                                         #
# --------------------------------------------------------------------------- #


def _entity_evidence_counts(store) -> dict[str, int]:
    """Per-entity evidence row counts (metadata only)."""
    rows = store.conn.execute(
        "SELECT entity_id, COUNT(*) AS c FROM entity_evidence GROUP BY entity_id"
    ).fetchall()
    return {r["entity_id"]: int(r["c"]) for r in rows}


def _dormant_entities_with_unresolved_loops(store) -> int:
    """COUNT of dormant entities carrying an unresolved open loop.

    Metadata only (never names). hasattr-guarded so pre-v7 store stubs keep
    working. A loop counts as resolved ONLY when an ``open_loop_resolved``
    row with the same exact detail key has a ts LATER than THAT loop row's ts
    — a REOPENED loop (newer than the last resolution) counts as unresolved
    again until a later completion resolves it (per-loop timestamps, never
    bare detail membership).
    """
    lister = getattr(store, "list_entities", None)
    if not callable(lister):
        return 0
    count = 0
    for entity in lister(status="dormant"):
        rows = store.entity_evidence_for(entity["id"], limit=50)
        resolved_latest: dict[str, float] = {}
        for r in rows:
            if r["kind"] != "open_loop_resolved":
                continue
            detail = str(r["detail"])
            resolved_latest[detail] = max(
                resolved_latest.get(detail, float("-inf")), float(r["ts"])
            )
        if any(
            r["kind"] == "open_loop"
            and not resolved_latest.get(str(r["detail"]), float("-inf"))
            > float(r["ts"])
            for r in rows
        ):
            count += 1
    return count


@entities_app.command("list")
def entities_list(
    kind: Optional[str] = typer.Option(
        None, "--kind", help="Filter by kind: repo|domain|document|topic."
    ),
    status: Optional[str] = typer.Option(
        None, "--status", help="Filter by status: active|dormant|user_marked_done."
    ),
    as_json: bool = typer.Option(
        False, "--json",
        help="Machine-readable output (names included on an interactive "
        "terminal only).",
    ),
) -> None:
    """List ledger entities (kind, status, activity extent, evidence count).

    Entity names and aliases are DERIVED SENSITIVE (distilled from captured
    content): they are printed ONLY on an interactive terminal — a piped or
    captured invocation gets counts and metadata so names never land in logs
    or scrollback files (the `summaries list` body-guard rule).
    """
    store = _store_maintenance()
    try:
        rows = store.list_entities(kind=kind, status=status)
        evidence_counts = _entity_evidence_counts(store)
    finally:
        store.close()
    if not rows:
        _console.print("[yellow]No ledger entities stored yet.[/]")
        return
    interactive = sys.stdout.isatty()
    if as_json:
        items = []
        for row in rows:
            item = {
                "id": row["id"],
                "kind": row["kind"],
                "status": row["status"],
                "first_ts": row["first_ts"],
                "last_ts": row["last_ts"],
                "evidence_count": evidence_counts.get(row["id"], 0),
            }
            if interactive:  # derived-sensitive bodies: interactive only
                item["name"] = row["name"]
                item["aliases"] = row["aliases"]
            items.append(item)
        _console.print_json(json.dumps({"entities": items}))
        if not interactive:
            _err_console.print(
                f"[dim]{len(rows)} entity names withheld (non-interactive "
                "output).[/]"
            )
        return
    for row in rows:
        meta = (
            f"kind={row['kind']} status={row['status']} "
            f"first={_fmt_ts(row['first_ts'])} last={_fmt_ts(row['last_ts'])} "
            f"evidence={evidence_counts.get(row['id'], 0)}"
        )
        if interactive:
            aliases = ", ".join(row["aliases"]) if row["aliases"] else "-"
            _console.print(
                f"[bold]{escape(str(row['name']))}[/] "
                f"(aliases: {escape(aliases)}) {escape(meta)}"
            )
        else:
            _console.print(escape(meta))
    if not interactive:
        _console.print(
            f"[dim]{len(rows)} entity names withheld (non-interactive output).[/]"
        )


@entities_app.command("show")
def entities_show(
    name: str = typer.Argument(
        ..., help="Entity name or alias (casefolded; exact match preferred)."
    ),
) -> None:
    """Show one entity's evidence rows (kind, date, detail, source refs).

    Names and evidence details are DERIVED SENSITIVE — bodies print to an
    interactive terminal only; a piped invocation gets counts.
    """
    store = _store_maintenance()
    try:
        candidates = store.entities_matching(name)
        needle = name.casefold()
        exact = [
            e for e in candidates
            if str(e["name"]).casefold() == needle
            or needle in [str(a).casefold() for a in e["aliases"]]
        ]
        pool = exact or candidates
        if not pool:
            _console.print("[yellow]No ledger entity matches that name.[/]")
            raise typer.Exit(code=1)
        if len(pool) > 1:
            interactive = sys.stdout.isatty()
            if interactive:
                names = ", ".join(sorted(str(e["name"]) for e in pool))
                _err_console.print(
                    f"[red]Ambiguous:[/] matches {len(pool)} entities: "
                    f"{escape(names)}."
                )
            else:
                _err_console.print(
                    f"[red]Ambiguous:[/] matches {len(pool)} entities "
                    "(names withheld: non-interactive output)."
                )
            raise typer.Exit(code=2)
        entity = pool[0]
        evidence = store.entity_evidence_for(entity["id"], limit=50)
    finally:
        store.close()
    interactive = sys.stdout.isatty()
    header = (
        f"kind={entity['kind']} status={entity['status']} "
        f"first={_fmt_ts(entity['first_ts'])} last={_fmt_ts(entity['last_ts'])} "
        f"evidence={len(evidence)}"
    )
    if interactive:
        aliases = ", ".join(entity["aliases"]) if entity["aliases"] else "-"
        _console.print(
            f"[bold]{escape(str(entity['name']))}[/] (aliases: {escape(aliases)})"
        )
    _console.print(escape(header))
    for row in evidence:
        line = (
            f"{_fmt_ts(row['ts'])} {row['kind']} "
            f"source={row['source_kind']}:{row['source_id']}"
        )
        if interactive:
            detail = str(row.get("detail") or "")
            if detail:
                line += f" detail={detail}"
            _console.print(f"  {escape(line)}")
    if not interactive:
        _console.print(
            f"[dim]{len(evidence)} evidence rows withheld "
            "(non-interactive output).[/]"
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
    from openbird.meetings.transcribe import (
        meetings_backend_available,
        parakeet_available,
        whisper_available,
    )

    _console.print("[bold]OpenBird meetings[/] (manual-record, experimental)")
    _console.print(
        "- Audio capture requires the signed ScreenCaptureKit `audio-helper` "
        "with Screen-Recording + Microphone TCC."
    )
    parakeet = parakeet_available()
    whisper = whisper_available()
    # On a host with both, `auto` prefers parakeet-mlx (recommended on Apple Silicon).
    active = "parakeet-mlx" if parakeet else ("faster-whisper" if whisper else "none")
    _console.print(
        f"- transcription backends: parakeet-mlx "
        f"[{'installed' if parakeet else 'not installed'}], "
        f"faster-whisper [{'installed' if whisper else 'not installed'}] "
        f"→ active (auto): [bold]{active}[/]."
    )
    _console.print(
        "- Speaker labeling ('me vs others') is experimental; consent indicator "
        "and manual start are required by design."
    )
    if not meetings_backend_available():
        _console.print(
            "[dim]Install a backend: `uv sync --extra meetings-mlx` "
            "(parakeet-mlx, Apple Silicon, recommended) or "
            "`uv sync --extra meetings` (faster-whisper).[/]"
        )


# --------------------------------------------------------------------------- #
# eval                                                                        #
# --------------------------------------------------------------------------- #


@eval_app.command("signals")
def eval_signals(
    fixture: Path = typer.Argument(..., help="Signal eval JSONL fixture."),
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Run the deterministic signal classifier eval fixture."""
    from openbird.signals import (
        load_signal_eval_jsonl,
        run_signal_eval,
        signal_eval_report_payload,
    )

    try:
        cases = load_signal_eval_jsonl(fixture)
        report = run_signal_eval(cases)
    except (OSError, ValueError) as exc:
        _err_console.print(f"[red]Invalid signal eval fixture:[/] {exc}")
        raise typer.Exit(code=2) from exc

    payload = signal_eval_report_payload(report)
    if as_json:
        _console.print_json(data=payload)
    else:
        table = Table(title="Signal eval", show_header=True, header_style="bold")
        table.add_column("Metric")
        table.add_column("Value", justify="right")
        for key in (
            "passed",
            "total_cases",
            "precision_at_5",
            "must_surface_recall",
            "missed_important_count",
            "noise_rate",
            "sensitive_leak_count",
            "sensitive_quarantine_miss_count",
        ):
            table.add_row(key, str(payload[key]))
        _console.print(table)

    if not report.passed:
        raise typer.Exit(code=1)


@eval_app.command("quality")
def eval_quality(
    runs: int = typer.Option(3, "--runs", help="LLM runs per surface (majority must pass)."),
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Quality gate for briefing + ask over the real store with the configured model.

    Makes live LLM calls (slow, non-deterministic) — a manual pre-PR gate, not a
    CI test. Uses the same provider as ``chat``/``briefing`` (local by default; a
    remote model still requires the ``OPENBIRD_ALLOW_CLOUD`` opt-in via
    ``_provider()``, so it can never reach cloud silently). Checks each surface N
    times: ask answers must be grounded + cited with no self-capture; briefings
    must have zero ungrounded ``#N`` refs and no self-capture source. Exits
    non-zero if any surface fails its strict-majority gate.
    """
    from openbird.routines.quality_eval import quality_eval_payload, run_quality_eval

    provider = _provider()
    store = _store(provider=provider)
    try:
        report = run_quality_eval(store, provider, day_window=_day_window, runs=runs)
    finally:
        store.close()

    payload = quality_eval_payload(report)
    if as_json:
        _console.print_json(data=payload)
    else:
        table = Table(title="Quality eval", show_header=True, header_style="bold")
        table.add_column("Surface")
        table.add_column("Pass", justify="center")
        table.add_column("Runs (ok)")
        for c in report.checks:
            ok = sum(1 for r in c.runs if r.get("ok"))
            table.add_row(c.label, "✓" if c.passed else "✗", f"{ok}/{len(c.runs)}")
        _console.print(table)

    if not report.passed:
        raise typer.Exit(code=1)


# --------------------------------------------------------------------------- #
# data                                                                        #
# --------------------------------------------------------------------------- #

data_app = typer.Typer(help="Manage stored data (purge, stats).", no_args_is_help=True)
app.add_typer(data_app, name="data")


REASONING_LEDGER_FIELDS = (
    "created_at",
    "feature",
    "packet_route",
    "reasoning_route",
    "egress",
    "route_class",
    "provider_family",
    "model",
    "packet_hash",
    "packet_bytes",
    "selected_source_count",
    "citation_count",
    "excluded_observations",
    "excluded_by",
    "outcome",
    "error_kind",
    "deletion_caveat",
)


def _redacted_reasoning_ledger_row(row: dict[str, Any]) -> dict[str, Any]:
    return {field: row.get(field) for field in REASONING_LEDGER_FIELDS}


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

    store = _store_maintenance()
    try:
        deleted = store.delete(all=all_, since_ts=since_ts)
    finally:
        store.close()
    _console.print(f"[green]Deleted[/] {deleted} observation(s) (cascade complete).")


@data_app.command("prune")
def data_prune(
    older_than: str = typer.Option(
        ...,
        "--older-than",
        help="Delete observations OLDER than this. Accepts a relative span like "
        "'30d', '24h', a unix timestamp, or an ISO date/datetime.",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
) -> None:
    """Retention prune: cascade-delete observations older than a cutoff (H10).

    Removes observations with ``ts`` strictly before the cutoff, cascading to
    orphaned blobs/chunks/index entries (atomic, rollback-guarded). Run
    ``openbird data vacuum`` afterwards to reclaim the freed space on disk.
    """
    cutoff = _parse_since(older_than, option_name="--older-than")
    if not yes:
        confirm = typer.confirm(
            f"Permanently delete data older than {_fmt_ts(cutoff)}?", default=False
        )
        if not confirm:
            _console.print("[yellow]Aborted.[/]")
            raise typer.Exit(code=1)

    # Retention prune is a delete-only maintenance op (like purge): it must not
    # require cloud opt-in or be blocked by an embedding-cohort mismatch, so it
    # uses the maintenance store opener — consistent with `data purge`/`data stats`.
    store = _store_maintenance()
    try:
        deleted = store.prune(older_than_ts=cutoff)
    finally:
        store.close()
    _console.print(
        f"[green]Pruned[/] {deleted} observation(s) older than {_fmt_ts(cutoff)} "
        f"(cascade complete). Run `openbird data vacuum` to reclaim disk space."
    )


@data_app.command("vacuum")
def data_vacuum() -> None:
    """Reclaim disk space: checkpoint the WAL and VACUUM the database (H10).

    Deletes/prunes only mark pages free; the file shrinks only after VACUUM
    rewrites it. Prints the bytes reclaimed.
    """
    # Vacuum is local maintenance: it must keep working even if the configured
    # embed model is cloud-backed or no longer matches the stored cohort.
    store = _store_maintenance()
    try:
        result = store.vacuum()
    finally:
        store.close()
    reclaimed = result["bytes_reclaimed"]
    _console.print(
        f"[green]Vacuumed[/] — reclaimed {reclaimed} bytes "
        f"({result['bytes_before']} -> {result['bytes_after']})."
    )
    _console.print_json(json.dumps(result))


@data_app.command("stats")
def data_stats() -> None:
    """Print memory-store row counts and the active embedding cohort."""
    store = _store_maintenance()
    try:
        stats = store.stats()
    finally:
        store.close()
    _console.print_json(json.dumps(stats))


@data_app.command("capture-health")
def data_capture_health(
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
    recent_window_seconds: float = typer.Option(
        24 * 60 * 60,
        "--recent-window-seconds",
        min=0,
        help="Window for recent per-app capture counts.",
    ),
) -> None:
    """Print metadata-only effective capture state and per-app health."""
    from openbird.capture.health import build_capture_health

    settings = get_settings()
    now = time.time()
    recent_since = now - recent_window_seconds
    store = _store_maintenance()
    try:
        activity = store.capture_app_activity(recent_since_ts=recent_since)
    finally:
        store.close()
    payload = build_capture_health(
        settings=settings,
        activity_by_app=activity,
        generated_at=now,
        recent_window_seconds=recent_window_seconds,
    )
    if as_json:
        _console.print_json(json.dumps(payload))
        return

    table = Table(title="Capture health", show_header=True, header_style="bold")
    table.add_column("App")
    table.add_column("State")
    table.add_column("Quality")
    table.add_column("OCR")
    table.add_column("Recent", justify="right")
    table.add_column("Total", justify="right")
    # Daemon-level OCR availability (Phase C2): the helper's Screen Recording
    # preflight edges, via the liveness sidecar. Rendered ONLY off a FRESH
    # daemon (state == "ok") — a dead daemon's stale "available" must not
    # claim OCR is live. health.py already nulls ocr_state for a non-ok
    # daemon; this guard is defense-in-depth at the render boundary.
    daemon = payload.get("daemon", {})
    daemon_ocr_state = daemon.get("ocr_state") if daemon.get("state") == "ok" else None
    for row in payload["apps"]:
        if row.get("ocr") == "opted_in":
            # Opted-in row: combine with the fresh daemon state — available /
            # unavailable when reported, honest "unknown" otherwise.
            if daemon_ocr_state in ("available", "unavailable"):
                ocr_cell = f"ocr_{daemon_ocr_state}"
            else:
                ocr_cell = "unknown"
        else:
            ocr_cell = "-"
        table.add_row(
            row["bundle_id"],
            row["effective_state"],
            row["quality"],
            ocr_cell,
            str(row["recent_observations"]),
            str(row["total_observations"]),
        )
    _console.print(table)


@data_app.command("capture-audit")
def data_capture_audit(
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
    recent_window_seconds: float = typer.Option(
        24 * 60 * 60,
        "--recent-window-seconds",
        min=0,
        help="Window for capture richness aggregates.",
    ),
    minimum_samples: int = typer.Option(
        5,
        "--minimum-samples",
        min=1,
        help="Samples required before assigning a stable context-quality bucket.",
    ),
) -> None:
    """Evaluate whether recent capture contains useful context, without exposing it."""
    from openbird.capture.audit import build_capture_audit
    from openbird.capture.health import build_capture_health

    settings = get_settings()
    now = time.time()
    recent_since = now - recent_window_seconds
    store = _store_maintenance()
    try:
        activity = store.capture_app_activity(recent_since_ts=recent_since)
        quality = store.capture_content_quality(recent_since_ts=recent_since)
    finally:
        store.close()
    health = build_capture_health(
        settings=settings,
        activity_by_app=activity,
        generated_at=now,
        recent_window_seconds=recent_window_seconds,
    )
    payload = build_capture_audit(
        health=health,
        content_quality=quality,
        min_samples=minimum_samples,
    )
    if as_json:
        _console.print_json(json.dumps(payload))
        return

    table = Table(title="Capture context audit", show_header=True, header_style="bold")
    table.add_column("App")
    table.add_column("Context")
    table.add_column("Coverage")
    table.add_column("Samples", justify="right")
    table.add_column("Chars p50/p90", justify="right")
    table.add_column("Lines p50/p90", justify="right")
    for row in payload["apps"]:
        table.add_row(
            row["bundle_id"],
            row["context_quality"],
            row["coverage"],
            str(row["sample_count"]),
            f"{row['chars_p50']}/{row['chars_p90']}",
            f"{row['lines_p50']}/{row['lines_p90']}",
        )
    _console.print(table)
    _console.print(f"Overall: {payload['overall_state']}")


@data_app.command("reasoning-ledger")
def data_reasoning_ledger(
    limit: int = typer.Option(
        20,
        "--limit",
        min=1,
        max=200,
        help="Number of recent redacted reasoning-send rows to show.",
    ),
    json_out: bool = typer.Option(
        False,
        "--json",
        help="Emit exact redacted ledger rows as JSON.",
    ),
) -> None:
    """Inspect redacted local metadata for remote reasoning packet sends.

    This is a local maintenance read: it never constructs a model provider,
    embeds content, or sends captured memory off-device. The ledger stores only
    route/count/hash metadata; raw prompts, answers, packets, snippets, source
    IDs, app/source names, URLs, and window titles are intentionally absent.
    """
    store = _store_maintenance()
    try:
        rows = [
            _redacted_reasoning_ledger_row(row)
            for row in store.list_reasoning_send_ledger(limit=limit)
        ]
    finally:
        store.close()

    if json_out:
        _console.print_json(json.dumps({"rows": rows}))
        return

    table = Table(title="Reasoning Send Ledger", show_header=True, header_style="bold")
    table.add_column("Time", no_wrap=True)
    table.add_column("Feature", no_wrap=True)
    table.add_column("Route")
    table.add_column("Provider", no_wrap=True)
    table.add_column("Outcome", no_wrap=True)
    table.add_column("Packet", no_wrap=True)
    table.add_column("Sources", justify="right", no_wrap=True)
    table.add_column("Exclusions")
    for row in rows:
        packet_hash = str(row.get("packet_hash") or "")
        packet_label = packet_hash[:12] if packet_hash else "-"
        excluded_by = row.get("excluded_by") or {}
        if isinstance(excluded_by, dict) and excluded_by:
            exclusions = ", ".join(
                f"{escape(str(key))}:{int(value)}"
                for key, value in sorted(excluded_by.items())
            )
        else:
            exclusions = "-"
        table.add_row(
            _fmt_ts(float(row["created_at"])) if row.get("created_at") is not None else "-",
            escape(str(row.get("feature") or "-")),
            escape(str(row.get("route_class") or "-")),
            escape(str(row.get("provider_family") or "-")),
            escape(str(row.get("outcome") or "-")),
            packet_label,
            str(row.get("selected_source_count") or 0),
            exclusions,
        )
    _console.print(table)
    if not rows:
        _console.print("No remote reasoning sends recorded.")


@data_app.command("export")
def data_export(
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Write decrypted memory export as JSONL to this file.",
    ),
    since: Optional[str] = typer.Option(
        None,
        "--since",
        help="Export observations at/after this time. Accepts unix ts, ISO date, or 7d/24h spans.",
    ),
    until: Optional[str] = typer.Option(
        None,
        "--until",
        help="Export observations at/before this time. Accepts unix ts, ISO date, or 7d/24h spans.",
    ),
    source: Optional[str] = typer.Option(
        None,
        "--source",
        help="Restrict export to one observation source, e.g. capture or ingest.",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Replace an existing output file.",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
) -> None:
    """Export decrypted observations and text as JSONL.

    Export is explicit egress: the chosen destination may be cloud-synced outside
    OpenBird's control, and exported files are not affected by later purge/prune.
    """
    since_ts = _parse_since(since) if since else None
    until_ts = _parse_since(until, option_name="--until") if until else None
    if since_ts is not None and until_ts is not None and since_ts > until_ts:
        _err_console.print("[red]Refusing[/] --since must be before --until.")
        raise typer.Exit(code=2)

    if output.exists() and not overwrite:
        _err_console.print(
            f"[red]Refusing[/] output exists: {output}. Use --overwrite to replace it."
        )
        raise typer.Exit(code=2)
    if not output.parent.exists():
        _err_console.print(f"[red]Refusing[/] parent directory does not exist: {output.parent}")
        raise typer.Exit(code=2)

    warning = (
        "EXPORT WARNING: writing decrypted OpenBird memory to "
        f"{output}. The destination may be cloud-synced outside OpenBird control; "
        "later purge/prune will not delete this export."
    )
    _err_console.print(f"[bold yellow]{warning}[/]")
    if not yes:
        if not sys.stdin.isatty():
            _err_console.print("[red]Refusing[/] (non-interactive). Re-run with --yes to export.")
            raise typer.Exit(code=1)
        if not typer.confirm("Export decrypted memory to this destination?", default=False):
            _console.print("[yellow]Aborted.[/]")
            raise typer.Exit(code=1)

    count = 0
    store = _store_maintenance()
    try:
        flags = os.O_WRONLY | os.O_CREAT
        flags |= os.O_TRUNC if overwrite else os.O_EXCL
        fd = os.open(output, flags, 0o600)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fd = -1
                for row in store.export_observations(
                    since_ts=since_ts, until_ts=until_ts, source=source
                ):
                    fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    count += 1
            output.chmod(0o600)
        finally:
            if fd >= 0:
                os.close(fd)
    finally:
        store.close()

    _console.print(f"[green]Exported[/] {count} observation(s) to {output}.")


@data_app.command("integrity")
def data_integrity(
    quick: bool = typer.Option(
        False, "--quick", help="Faster quick_check (skips some cross-page/index checks)."
    ),
) -> None:
    """Verify the on-disk database is not corrupt (SQLite integrity check).

    Also probes the summary-index deletion contract (Phase E1) and the
    entity-evidence trigger contract (Phase E2): non-zero orphan counts mean a
    code path bypassed the sweep APIs / deletion triggers.
    """
    # Open the DB raw (not via MemoryStore) so a corrupt DB — exactly what this
    # command diagnoses — is reported rather than crashing in schema/migrations.
    from openbird.memory.store import (
        check_database_integrity,
        check_entity_evidence_orphans,
        check_summary_index_orphans,
    )

    settings = get_settings()
    result = check_database_integrity(settings.db_path, settings=settings, quick=quick)
    orphans = check_summary_index_orphans(settings.db_path, settings=settings)
    entity_orphans = check_entity_evidence_orphans(settings.db_path, settings=settings)
    ok = result["ok"] and orphans["ok"] and entity_orphans["ok"]
    if ok:
        _console.print("[green]integrity: ok[/]")
        counts = orphans.get("counts")
        if counts is not None:
            _console.print(
                "[green]summary index: ok[/] "
                f"(fts_orphans=0 vec_orphans=0 entry_orphans=0)"
            )
        if entity_orphans.get("counts") is not None:
            _console.print(
                "[green]entity ledger: ok[/] "
                "(observation_orphans=0 span_orphans=0 summary_orphans=0)"
            )
    else:
        _console.print("[red]integrity: PROBLEMS DETECTED[/]")
        for problem in (
            result["problems"] + orphans["problems"] + entity_orphans["problems"]
        ):
            _console.print(f"  - {problem}")
    raise typer.Exit(code=0 if ok else 1)


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
    """Re-embed every stored chunk under the current embedding model.

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
        # summary_rows are (re)selected INSIDE the write transaction below —
        # a pre-BEGIN snapshot could race a concurrent summary deletion and
        # re-insert vectors for dead entries (zero-orphan contract). This
        # pre-count exists only for the early-exit/progress sizing.
        summary_total = int(
            conn.execute(
                "SELECT COUNT(*) c FROM summary_index_entries"
            ).fetchone()["c"]
        )

        if current_cohort == new_cohort and not force:
            _console.print(
                f"[green]Already on cohort[/] {new_cohort} "
                f"({total} chunk(s)); nothing to do. Use --force to rebuild anyway."
            )
            return

        _console.print(
            f"Reindex: [cyan]{current_cohort or '(none)'}[/] -> [cyan]{new_cohort}[/] "
            f"· {total} chunk(s) · {summary_total} summary entr(ies) · dim={new_dim}"
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
            # Transactional snapshot (see comment above): entries read under
            # the write lock cannot be deleted before their vectors land.
            summary_rows = conn.execute(
                "SELECT entry_rowid, text FROM summary_index_entries "
                "ORDER BY entry_rowid"
            ).fetchall()
            summary_total = len(summary_rows)
            # Rebuild the vector table at the (possibly new) dimension. CREATE ...
            # IF NOT EXISTS would keep the stale dim, so drop first.
            conn.execute("DROP TABLE IF EXISTS vec_chunks")
            conn.execute(
                f"CREATE VIRTUAL TABLE vec_chunks USING vec0("
                f"chunk_rowid INTEGER PRIMARY KEY, embedding FLOAT[{new_dim}])"
            )

            # The summary index shares the embedding cohort — rebuild its vec
            # table at the new dimension too, in the SAME transaction, so both
            # vector tables always carry one cohort (or roll back together).
            conn.execute("DROP TABLE IF EXISTS vec_summaries")
            conn.execute(
                f"CREATE VIRTUAL TABLE vec_summaries USING vec0("
                f"entry_rowid INTEGER PRIMARY KEY, embedding FLOAT[{new_dim}])"
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

                stask = (
                    progress.add_task("Embedding summaries", total=summary_total)
                    if summary_total
                    else None
                )
                sdone = 0
                for start in range(0, summary_total, max(1, batch_size)):
                    batch = summary_rows[start : start + max(1, batch_size)]
                    vectors = provider.embed([r["text"] for r in batch])
                    for row, vec in zip(batch, vectors):
                        conn.execute(
                            "INSERT INTO vec_summaries(entry_rowid, embedding) "
                            "VALUES (?, ?)",
                            (int(row["entry_rowid"]), _serialize_f32(vec)),
                        )
                    sdone += len(batch)
                    if stask is not None:
                        progress.update(stask, completed=sdone)

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
        f"[green]Reindexed[/] {total} chunk(s) and {summary_total} summary "
        f"entr(ies) under cohort {new_cohort} (dim={new_dim})."
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


def _parse_since(value: str, *, option_name: str = "--since") -> float:
    """Parse a time-spec value into a unix timestamp.

    Accepts a bare unix timestamp, a relative span (``7d``/``24h``/``30m``/
    ``45s``), or an ISO 8601 date/datetime. Relative spans are subtracted from
    *now*. ``option_name`` names the originating CLI flag so parse errors point
    at the flag the user actually passed (e.g. ``--older-than`` for prune).
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
            f"could not parse {option_name} {value!r}; use a unix ts, ISO date, "
            "or span like '7d'."
        ) from exc


def main() -> None:
    """Console-script entrypoint."""
    app()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())
