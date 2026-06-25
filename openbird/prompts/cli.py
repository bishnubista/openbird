"""``openbird prompts`` — inspect and customize the swappable system prompts.

Commands operate on a prompt *persona* (its editable tone / answering rules); the
security scaffold (untrusted-data fence rules) is framework-owned and never
written into a persona file. See :mod:`openbird.prompts.core`.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from openbird.prompts import registry
from openbird.prompts.core import PromptSpec, render
from openbird.prompts.loader import PersonaResolution, resolve_persona

prompts_app = typer.Typer(
    help="Inspect and customize OpenBird's swappable system prompts.",
    no_args_is_help=True,
)
_console = Console()


def _prompts_dir() -> Path:
    """Return the configured persona-override directory."""
    from openbird.config import get_settings

    return Path(get_settings().prompts_dir or "")


def _spec_or_exit(key: str) -> PromptSpec:
    """Return the spec for ``key``, or exit 2 if the key is unknown."""
    registry.ensure_loaded()
    try:
        return registry.get(key)
    except KeyError:
        _console.print(
            f"[red]Unknown prompt key {key!r}.[/red] Known keys: "
            f"{', '.join(registry.keys()) or '(none)'}"
        )
        raise typer.Exit(code=2) from None


def _status(res: PersonaResolution) -> str:
    """Render a colored ok/refused status cell for the list table."""
    return "[green]ok[/green]" if res.ok else f"[red]refused: {res.reason}[/red]"


@prompts_app.command("list")
def prompts_list() -> None:
    """List prompts with their active override source and validity."""
    registry.ensure_loaded()
    prompts_dir = _prompts_dir()
    table = Table(title="System prompts", show_header=True, header_style="bold")
    table.add_column("Key")
    table.add_column("Source")
    table.add_column("Status")
    for key in registry.keys():
        res = resolve_persona(key, prompts_dir=prompts_dir)
        table.add_row(key, res.source, _status(res))
    _console.print(table)


@prompts_app.command("show")
def prompts_show(
    key: str = typer.Argument(..., help="Prompt key (e.g. 'rag')."),
    full: bool = typer.Option(
        False, "--full", help="Show the whole rendered prompt, not just the persona."
    ),
) -> None:
    """Print a prompt's active persona, or the full rendered prompt with --full."""
    spec = _spec_or_exit(key)
    res = resolve_persona(key, prompts_dir=_prompts_dir())
    if full:
        # render() applies the locked scaffold around the (resolved) persona.
        _console.print(render(spec, res.persona))
        return
    persona = res.persona if res.persona is not None else spec.default_persona
    label = res.source if res.ok else f"{res.source} (refused: {res.reason}; default)"
    _console.print(f"[bold]persona[/bold] (source: {label}):\n")
    _console.print(persona)


@prompts_app.command("edit")
def prompts_edit(
    key: str = typer.Argument(..., help="Prompt key (e.g. 'rag')."),
) -> None:
    """Scaffold and open a prompt's persona override file in $EDITOR.

    The file contains ONLY the persona (its full contents become model input).
    The locked security scaffold is printed here for context but never written
    into the file.
    """
    spec = _spec_or_exit(key)
    prompts_dir = _prompts_dir()
    prompts_dir.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(prompts_dir, 0o700)
    except OSError:
        pass
    path = prompts_dir / f"{key}.txt"
    # Consistency with the hardened runtime loader: never scaffold or edit through
    # a symlink or special file. os.path.lexists catches broken symlinks that
    # path.exists() (which follows links) would miss.
    if os.path.lexists(path) and (path.is_symlink() or not path.is_file()):
        _console.print(
            f"[red]{path} exists but is not a regular file (symlink/special). "
            f"Run 'openbird prompts reset {key}' to remove it first.[/red]"
        )
        raise typer.Exit(code=2)
    created = False
    if not path.exists():
        # Atomic create that refuses a symlink swapped in mid-operation (O_EXCL +
        # O_NOFOLLOW), mirroring the loader's read-side guarantees.
        fd = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
        )
        try:
            os.write(fd, (spec.default_persona.strip() + "\n").encode("utf-8"))
        finally:
            os.close(fd)
        created = True

    _console.print(
        f"[bold]Editing persona for {key!r}[/bold] "
        f"({'created from default' if created else 'existing'}): {path}\n"
    )
    _console.print(
        "[dim]The following security scaffold is LOCKED (framework-composed, not "
        "editable) and wraps your persona at runtime:[/dim]"
    )
    _console.print(f"[dim]--- preamble ---\n{spec.security_preamble}[/dim]")
    _console.print(f"[dim]--- epilogue ---\n{spec.security_epilogue}[/dim]")
    _console.print(
        f"[dim]Required fence tokens: {', '.join(spec.fence.required_tokens())}[/dim]\n"
    )

    editor = os.environ.get("EDITOR") or os.environ.get("VISUAL")
    if not editor:
        _console.print(f"Set $EDITOR to edit interactively, or edit the file: {path}")
        return
    try:
        returncode = subprocess.call([editor, str(path)])  # noqa: S603 - user's $EDITOR
    except OSError as exc:
        _console.print(f"[red]Could not launch $EDITOR ({editor}): {exc}[/red]")
        raise typer.Exit(code=1) from exc
    if returncode != 0:
        _console.print(f"[red]$EDITOR exited with status {returncode}.[/red]")
        raise typer.Exit(code=returncode or 1)


@prompts_app.command("reset")
def prompts_reset(
    key: str = typer.Argument(..., help="Prompt key (e.g. 'rag')."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation."),
) -> None:
    """Delete a prompt's persona override file, reverting to the default."""
    _spec_or_exit(key)
    path = _prompts_dir() / f"{key}.txt"
    # os.path.lexists (not path.exists) so a BROKEN symlink — which the runtime
    # loader still treats as a refused override — can be removed here too.
    if not os.path.lexists(path):
        _console.print(f"No override file for {key!r}; already using the default.")
        return
    if not yes and not typer.confirm(f"Delete {path} and revert {key!r} to default?"):
        raise typer.Exit(code=1)
    path.unlink()  # removes the symlink itself, never its target
    _console.print(f"Removed {path}; {key!r} now uses the default persona.")


@prompts_app.command("validate")
def prompts_validate(
    key: str = typer.Argument(None, help="Validate one key; omit to validate all."),
) -> None:
    """Validate persona overrides render cleanly; exit non-zero on any failure."""
    registry.ensure_loaded()
    targets = [key] if key else list(registry.keys())
    if key:
        _spec_or_exit(key)
    prompts_dir = _prompts_dir()
    failures = 0
    for target in targets:
        spec = registry.get(target)
        res = resolve_persona(target, prompts_dir=prompts_dir)
        if not res.ok:
            failures += 1
            _console.print(f"[red]✗[/red] {target}: refused ({res.reason})")
            continue
        try:
            render(spec, res.persona)
        except Exception as exc:  # PromptValidationError or unexpected
            failures += 1
            _console.print(f"[red]✗[/red] {target}: render failed ({exc})")
        else:
            _console.print(f"[green]✓[/green] {target}: ok (source: {res.source})")
    if failures:
        raise typer.Exit(code=2)
