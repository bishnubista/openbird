"""Shared fixtures: a deterministic fake embedding provider (no Ollama/network)."""

from __future__ import annotations

import hashlib
import math

import pytest

from openbird.config import Settings


@pytest.fixture(autouse=True)
def _no_gui_allowlist(monkeypatch):
    """Neutralize the macOS GUI-allowlist bridge for ALL unit tests.

    ``_settings_from_env`` reads the real ``ai.openbird.OpenBird`` defaults domain
    when ``OPENBIRD_ALLOWLIST`` is unset. On a developer Mac that runs the app,
    that domain is populated, so any ``get_settings()`` test would silently pick up
    the real allowlist. Default it to "unreadable" everywhere; bridge tests opt
    back in by re-patching ``_read_gui_allowlist``.
    """
    monkeypatch.setattr("openbird.config._read_gui_allowlist", lambda: None)


class FakeProvider:
    """A deterministic stand-in for :class:`LLMProvider`.

    Produces stable ``embed_dim``-length vectors derived from a hash of the
    text, so identical text yields identical vectors (and similar text yields
    similar vectors) without any network call.
    """

    def __init__(self, embed_dim: int = 768) -> None:
        self.embed_dim = embed_dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def _vec(self, text: str) -> list[float]:
        # Seed a simple LCG from the text hash; fill + L2-normalize.
        h = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)
        state = h % (2**61 - 1) or 1
        vec: list[float] = []
        for _ in range(self.embed_dim):
            state = (state * 6364136223846793005 + 1442695040888963407) % (2**64)
            vec.append((state / 2**64) - 0.5)
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def cohort_key(self) -> str:
        return f"fake:fake-embed:{self.embed_dim}:deadbeef"


@pytest.fixture
def fake_provider() -> FakeProvider:
    return FakeProvider(embed_dim=768)


@pytest.fixture
def mem_settings(tmp_path) -> Settings:
    return Settings(data_dir=tmp_path, embed_dim=768)
