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

    allowlist: list[str] = field(default_factory=list)
    blocklist: list[str] = field(default_factory=lambda: list(_DEFAULT_BLOCKLIST))

    ocr_enabled: bool = False
    # Set by storage.crypto.open_encrypted_db depending on whether an encrypted
    # backend (SQLCipher) is actually available. Default False = "local-only,
    # not yet app-encrypted".
    encryption_enabled: bool = False
    # When True (env OPENBIRD_REQUIRE_ENCRYPTION=1), opening the DB RAISES rather
    # than silently degrading to a plaintext file when SQLCipher cannot be
    # verified (H2). Default False keeps the backward-compatible plaintext
    # fallback.
    require_encryption: bool = False
    # Retention window in days. When > 0, observations older than this many days
    # are eligible for `openbird data prune` / programmatic pruning (H10). 0 (the
    # default) disables automatic retention; data is kept until explicitly pruned.
    retention_days: int = 0

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
    if isinstance(default, bool) or name in (
        "ocr_enabled",
        "encryption_enabled",
        "require_encryption",
    ):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(default, int) and not isinstance(default, bool):
        return int(raw)
    return raw


_COERCE_DEFAULTS: dict[str, object] = {
    "allowlist": [],
    "blocklist": [],
    "ocr_enabled": False,
    "encryption_enabled": False,
    "require_encryption": False,
    "retention_days": 0,
    "embed_dim": 768,
}


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


__all__ = ["Settings", "get_settings", "reset_settings_cache"]
