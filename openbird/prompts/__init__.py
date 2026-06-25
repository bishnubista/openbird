"""Modular, swappable LLM system prompts.

PR1 ships the core machinery (:class:`FenceSpec`, :class:`PromptSpec`,
:func:`render`) and wires the RAG prompt through it. On-disk persona overrides,
the ``openbird prompts`` CLI, the remaining feature prompts, and the injection
``test`` harness land in later PRs (see ``prompts-plan.md``).
"""

from __future__ import annotations

from openbird.prompts.core import (
    FenceSpec,
    PromptSpec,
    PromptValidationError,
    render,
)

__all__ = [
    "FenceSpec",
    "PromptSpec",
    "PromptValidationError",
    "render",
]
