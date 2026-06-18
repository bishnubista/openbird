"""Central settings, paths, model config, allow/blocklists, and feature flags.

Implemented as a plain dataclass (pydantic-settings is optional and not a hard
dependency). ``get_settings()`` returns a cached instance honoring ``OPENBIRD_*``
environment overrides.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from functools import lru_cache
from pathlib import Path

import platformdirs

# Apps excluded from capture until the user explicitly enables them [R3]:
# terminals, code editors, browsers, password managers, finance/health apps.
_DEFAULT_BLOCKLIST: list[str] = [
    "com.apple.Terminal",
    "com.googlecode.iterm2",
    "com.microsoft.VSCode",
    "com.1password.1password",
    "com.agilebits.onepassword7",
    "com.apple.keychainaccess",
]


def _default_data_dir() -> Path:
    """Resolve the OpenBird data directory (``~/.openbird`` by default)."""
    env = os.environ.get("OPENBIRD_DATA_DIR")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".openbird"


@dataclass
class Settings:
    """Runtime configuration for OpenBird.

    Field values are sourced (in precedence order) from explicit constructor
    arguments, then ``OPENBIRD_*`` environment variables, then defaults.
    """

    data_dir: Path = field(default_factory=_default_data_dir)
    db_path: str | None = None

    # Provider backend. "litellm" preserves today's runtime behavior, including
    # the default Ollama model strings and cloud opt-in via LiteLLM-compatible
    # model names. "mlx" is intentionally reserved until the isolated experiment
    # has been promoted into production code.
    llm_backend: str = "litellm"
    llm_model: str = "ollama/llama3.2"
    embed_model: str = "ollama/nomic-embed-text"
    embed_dim: int = 768

    # LLM resilience [B4/H4]: explicit timeouts and bounded retries so a
    # reachable-but-wedged Ollama (or a flaky cloud endpoint) can never hang the
    # process forever. Values are seconds; retries are passed to LiteLLM, which
    # backs off on connection errors / 5xx / rate-limit responses.
    llm_timeout: float = 60.0
    embed_timeout: float = 30.0
    llm_num_retries: int = 2

    # Cloud opt-in [H3]: OpenBird is local-first. Resolving a *remote* model
    # (anything not ollama/* or an mlx local backend) silently POSTs private
    # memory to a third party, so it MUST be explicitly opted into. Default off.
    allow_cloud: bool = False

    # Runtime Ollama host [M1]: the base URL threaded into LiteLLM as api_base
    # for ollama/* models, so the runtime provider talks to the SAME host that
    # preflight probes. None falls back to the OLLAMA_HOST / OPENBIRD_OLLAMA_HOST
    # env vars (see resolved_ollama_host) then the localhost default.
    ollama_host: str | None = None

    allowlist: list[str] = field(default_factory=list)
    blocklist: list[str] = field(default_factory=lambda: list(_DEFAULT_BLOCKLIST))

    ocr_enabled: bool = False
    # Set by storage.crypto.open_encrypted_db depending on whether an encrypted
    # backend (SQLCipher) is actually available. Default False = "local-only,
    # not yet app-encrypted".
    encryption_enabled: bool = False

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir).expanduser()
        if self.db_path is None:
            self.db_path = str(self.data_dir / "openbird.db")
        # Ensure the data directory exists with private (0700) permissions.
        self.data_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.data_dir, 0o700)
        except OSError:
            pass


def _coerce(name: str, raw: str, default: object) -> object:
    """Coerce an env-var string to the type implied by the field default."""
    if name in ("allowlist", "blocklist"):
        return [item.strip() for item in raw.split(",") if item.strip()]
    if isinstance(default, bool) or name in ("ocr_enabled", "encryption_enabled", "allow_cloud"):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(default, float) or name in ("llm_timeout", "embed_timeout"):
        return float(raw)
    if isinstance(default, int) and not isinstance(default, bool):
        return int(raw)
    return raw


_COERCE_DEFAULTS: dict[str, object] = {
    "allowlist": [],
    "blocklist": [],
    "ocr_enabled": False,
    "encryption_enabled": False,
    "allow_cloud": False,
    "embed_dim": 768,
    "llm_timeout": 60.0,
    "embed_timeout": 30.0,
    "llm_num_retries": 2,
}

# The localhost Ollama base URL used when nothing else is configured. Kept in
# sync with preflight._DEFAULT_OLLAMA_HOST via this single definition.
DEFAULT_OLLAMA_HOST = "http://localhost:11434"


def resolved_ollama_host(settings: "Settings | None" = None) -> str:
    """Resolve the Ollama base URL with a single precedence used everywhere.

    Precedence (highest first):
      1. ``OLLAMA_HOST`` env var (the LiteLLM / Ollama community convention).
      2. ``OPENBIRD_OLLAMA_HOST`` env var (== ``settings.ollama_host``).
      3. The localhost default.

    Both preflight and the runtime provider call this so a green preflight and
    the actual runtime call always target the same host [M1]. The result is
    always a full ``scheme://...`` URL: a bare ``host:port`` (the common
    ``OLLAMA_HOST`` form) is normalized to ``http://host:port`` so LiteLLM's
    ``api_base`` and preflight's ``urljoin`` both work [B+F round-3].
    """
    env_host = os.environ.get("OLLAMA_HOST")
    if env_host:
        return _normalize_host_url(env_host)
    if settings is not None and settings.ollama_host:
        return _normalize_host_url(settings.ollama_host)
    openbird_host = os.environ.get("OPENBIRD_OLLAMA_HOST")
    if openbird_host:
        return _normalize_host_url(openbird_host)
    return DEFAULT_OLLAMA_HOST


def _normalize_host_url(host: str) -> str:
    """Ensure ``host`` is a full URL, defaulting a missing scheme to ``http://``.

    Ollama's ``OLLAMA_HOST`` convention accepts a bare ``host:port`` (e.g.
    ``localhost:11434``); LiteLLM ``api_base`` and ``urljoin`` both need a scheme,
    so prepend ``http://`` when none is present. A value that already has a scheme
    (``http://`` / ``https://``) is returned unchanged.
    """
    text = host.strip()
    if not text:
        return DEFAULT_OLLAMA_HOST
    if "://" in text:
        return text
    return f"http://{text}"


# Loopback host names that keep traffic on this machine. A non-loopback Ollama
# host means captured chunks leave the device, so it is treated as remote even
# for an ollama/* model (route-based cloud classification [H3]).
_LOOPBACK_HOSTS: frozenset[str] = frozenset(
    {"localhost", "127.0.0.1", "::1", "0.0.0.0", "[::1]"}
)


def is_loopback_host(host_url: str) -> bool:
    """Return True if ``host_url`` resolves to this machine (loopback).

    Parses the URL host component and compares against known loopback names.
    A bare/blank host is treated as loopback (the localhost default). Anything
    that fails to parse is treated as NON-loopback (the safe, opt-in-required
    default), so a malformed host never silently counts as local.
    """
    from urllib.parse import urlsplit

    raw = (host_url or "").strip()
    if not raw:
        return True
    candidate = raw if "//" in raw else f"//{raw}"
    try:
        hostname = urlsplit(candidate).hostname
    except ValueError:
        return False
    if hostname is None:
        return False
    return hostname.lower() in _LOOPBACK_HOSTS


def _settings_from_env() -> Settings:
    """Build a :class:`Settings` instance applying ``OPENBIRD_*`` overrides."""
    overrides: dict[str, object] = {}

    for f in fields(Settings):
        env_key = f"OPENBIRD_{f.name.upper()}"
        raw = os.environ.get(env_key)
        if raw is None:
            continue
        # Determine a sensible default for type coercion without constructing
        # a full Settings (which would touch the filesystem).
        default_val = _COERCE_DEFAULTS.get(f.name, "")
        overrides[f.name] = _coerce(f.name, raw, default_val)

    return Settings(**overrides)  # type: ignore[arg-type]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a process-wide cached :class:`Settings` honoring env overrides."""
    return _settings_from_env()


def reset_settings_cache() -> None:
    """Clear the cached settings (primarily for tests)."""
    get_settings.cache_clear()


__all__ = [
    "Settings",
    "get_settings",
    "reset_settings_cache",
    "resolved_ollama_host",
    "is_loopback_host",
    "DEFAULT_OLLAMA_HOST",
]
