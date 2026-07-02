"""A small registry mapping prompt keys to their :class:`PromptSpec`.

Lets the ``openbird prompts`` CLI and the override loader enumerate prompts by
key. Feature modules self-register at import (e.g. ``openbird.chat.rag`` calls
:func:`register`); :func:`ensure_loaded` imports those modules so the registry is
populated without relying on incidental import order.
"""

from __future__ import annotations

from openbird.prompts.core import PromptSpec

_REGISTRY: dict[str, PromptSpec] = {}

# Feature modules whose import populates the registry. Imported lazily by
# ensure_loaded() to avoid an import cycle at module-load time. PR3 adds the
# routine/meeting/signal modules here.
_FEATURE_MODULES: tuple[str, ...] = (
    "openbird.chat.rag",
    "openbird.routines.templates",
    "openbird.meetings.transcribe",
    "openbird.signals.classifier",
    "openbird.taxonomy",
)
_loaded = False


def register(spec: PromptSpec) -> None:
    """Register (or replace) a prompt spec by its key. Idempotent."""
    _REGISTRY[spec.key] = spec


def get(key: str) -> PromptSpec:
    """Return the spec for ``key``; raises :class:`KeyError` if unknown."""
    return _REGISTRY[key]


def keys() -> tuple[str, ...]:
    """All registered prompt keys, sorted for stable CLI output."""
    return tuple(sorted(_REGISTRY))


def ensure_loaded() -> None:
    """Import the feature modules so every prompt is registered. Idempotent."""
    global _loaded
    if _loaded:
        return
    import importlib

    for module in _FEATURE_MODULES:
        importlib.import_module(module)
    _loaded = True
