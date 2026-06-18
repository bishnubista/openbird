"""Shared LLM provider contracts.

The concrete runtime provider lives in :mod:`openbird.llm.provider`; this module
holds only the structural contract used by memory, chat, routines, and future
backend implementations.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from openbird.config import Settings


@runtime_checkable
class LLMProviderProtocol(Protocol):
    """Runtime contract for OpenBird model providers."""

    settings: Settings
    embed_model: str
    llm_model: str
    embed_dim: int
    normalized: bool

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""

    def complete(
        self,
        messages: list[dict],
        *,
        json_schema: dict | None = None,
    ) -> str | dict:
        """Return raw text, or a parsed object when structured output is requested."""

    def cohort_key(self) -> str:
        """Return a stable identity for persisted embedding cohorts."""


__all__ = ["LLMProviderProtocol"]
