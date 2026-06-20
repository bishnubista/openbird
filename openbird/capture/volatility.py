"""De-flicker volatile UI noise from captured text before hashing and storage.

The capture daemon stores one *blob* per poll and dedups blobs by the SHA-256 of
their normalized text (``ingest.content_hash``). A live terminal / TUI / progress
UI re-renders a tiny animated glyph every frame — a braille spinner (``⠂``→``⠐``),
a star "thinking" glyph (``✳``→``✶``), or a moving progress bar — while the rest
of a large (~100 KB+) scrollback is byte-identical. That one mutating glyph forks
the content hash every ~2 seconds, so the coalesce gate sees a "new" capture and a
fresh full-scrollback blob is stored each frame. That is the 300 MB/hour bloat.

:func:`normalize` collapses these high-churn, zero-information UI animations to a
stable form so flicker frames hash identically (coalescing at BOTH the daemon's
``_signature`` gate and the ``content_blobs`` layer). It is **deliberately narrow
and token-scoped**: every rule removes only a volatile *token* and preserves the
meaningful words around it on the same line. It must NEVER rewrite real content —
mirroring the conservative stance of :func:`adapters.normalize_for_app`, which runs
just before this and has already collapsed intra-line whitespace and stripped each
line. Pure regex; no app-specific logic.

Intentionally NOT handled here (deferred to the near-dup layer): clock ``HH:MM:SS``,
parenthesized elapsed ``(12s)``, and ``↑/↓`` byte-rate counters. Those carry
information and scrubbing them risks falsely merging genuinely distinct log/prose
lines (``10:15:30 build failed`` vs ``10:16:02 build failed``). Spinners and progress
bars are the dominant flicker source, and terminals — the main clock-ticking
offenders — are blocked by default, so any residual counter-bloat is confined to an
explicitly allowlisted terminal.
"""

from __future__ import annotations

import re

# ANSI / VT escape sequences: CSI (``ESC [ ... final``) plus the simple two-char
# ``ESC <char>`` forms. Stripped entirely — they are pure display control.
_ANSI_RE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

# Leading spinner glyphs at the START of a line: the full braille block
# (U+2800–U+28FF, used by the vast majority of CLI spinners incl. Claude Code) and
# a focused set of dingbat star/"thinking" glyphs (✳ ✶ ✻ ✽ ✺ ✦). Removed with any
# trailing spaces so the line's label survives (``✳ Building…`` -> ``Building…``).
# Anchored to line start because these glyphs never BEGIN meaningful prose.
_SPINNER_GLYPHS = (
    "⠀-⣿"  # braille patterns
    "✳✶✻✽✺✦"  # ✳ ✶ ✻ ✽ ✺ ✦
)
_LEADING_SPINNER_RE = re.compile(rf"^[{_SPINNER_GLYPHS}]+[ \t]*")

# ASCII spinners ``| / - \`` are removed ONLY when the whole (stripped) line is
# exactly one such char — stripping a LEADING ``-`` or ``/`` would destroy
# ``- bullet`` list items and ``/usr/bin`` paths.
_ASCII_SPINNER_LINE = frozenset("|/-\\")

# Block-drawing glyphs used to draw bars: full/partial blocks and light/medium/dark
# shades. A run of these is an unambiguous progress indicator.
_BLOCK = "█-▏░-▓"

# The characters a progress-bar *body* may contain (ASCII fill + block glyphs +
# spacing). Used to bound bracket/pipe bar matches without eating real text.
_BAR_BODY = rf"[#=>\-.{_BLOCK} ]"

# Strong bar signals — at least one must appear inside the brackets for it to count
# as a progress bar, so ``[TODO]`` / ``[1.2.3]`` / ``[ERROR]`` / ``[ ]`` are kept.
_BAR_SIGNAL = rf"[#=>{_BLOCK}]"

# Bracketed bar. A real progress bar is either WIDE (>=4 body chars, optional
# trailing percentage) or carries an explicit trailing percentage. Short bracket
# markers such as ``[#]``, ``[=>]``, ``[==>]``, ``[>]`` are NOT bars and survive —
# requiring width-or-percent (not just "contains a bar char") avoids eating
# meaningful bracketed tokens in prose/code.
_PROGRESS_BRACKET_RE = re.compile(
    rf"\[(?=[^\]]*{_BAR_SIGNAL})(?:{_BAR_BODY}){{4,}}\](?:[ \t]*\d{{1,3}}%)?"
    rf"|\[(?=[^\]]*{_BAR_SIGNAL})(?:{_BAR_BODY}){{1,3}}\][ \t]*\d{{1,3}}%"
)

# tqdm-style ``45%|████    |`` (percentage THEN piped bar). The leading ``\d%``
# guard means a bare ``| pipe |`` is never matched.
_PROGRESS_PIPE_RE = re.compile(rf"\d{{1,3}}%[ \t]*\|(?:{_BAR_BODY})*\|")

# A bare run of >=3 block-drawing chars: ``███████░░░``.
_BLOCK_BAR_RE = re.compile(rf"[{_BLOCK}]{{3,}}")

# Collapse the double spaces that token removal can leave mid-line. Safe: by the
# time we run, ``adapters.normalize_for_app`` has already single-spaced each line,
# so any 2+ run here was created by our own removals.
_MULTISPACE_RE = re.compile(r"[ \t]{2,}")


def normalize(text: str) -> str:
    """De-flicker volatile UI noise so animated frames hash identically.

    Operates line by line, removing only volatile tokens (ANSI control, leading
    spinner glyphs, standalone ASCII spinners, progress bars) while preserving the
    meaningful words on each line. Idempotent: ``normalize(normalize(x)) ==
    normalize(x)``.

    Args:
        text: Text already cleaned by :func:`adapters.normalize_for_app` (each line
            stripped, intra-line whitespace single-spaced).

    Returns:
        The de-flickered text. May be empty if every line was pure animation.
    """
    if not text:
        return ""

    out_lines: list[str] = []
    for raw_line in text.split("\n"):
        line = _ANSI_RE.sub("", raw_line)
        if line.strip() in _ASCII_SPINNER_LINE:
            # A whole-line ASCII spinner carries no information.
            out_lines.append("")
            continue
        line = _LEADING_SPINNER_RE.sub("", line)
        line = _PROGRESS_BRACKET_RE.sub("", line)
        line = _PROGRESS_PIPE_RE.sub("", line)
        line = _BLOCK_BAR_RE.sub("", line)
        line = _MULTISPACE_RE.sub(" ", line).rstrip()
        out_lines.append(line)
    return "\n".join(out_lines)


__all__ = ["normalize"]
