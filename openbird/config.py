"""Central settings, paths, model config, allow/blocklists, and feature flags.

Implemented as a plain dataclass (pydantic-settings is optional and not a hard
dependency). ``get_settings()`` returns a cached instance honoring ``OPENBIRD_*``
environment overrides.
"""

from __future__ import annotations

import math
import os
import plistlib
import subprocess
import sys
from dataclasses import dataclass, field, fields
from functools import lru_cache
from pathlib import Path
from xml.parsers.expat import ExpatError

import platformdirs

# Apps excluded from capture until the user explicitly enables them:
# terminals, code editors, browsers, password managers, finance/health apps.
# NOTE: the blocklist is SUBTRACTIVE — `redact.decide` applies the allowlist first,
# then removes blocklisted apps. So allowlisting a terminal alone will NOT capture
# it; a user must also drop it from the blocklist (e.g. OPENBIRD_BLOCKLIST=... or a
# custom `blocklist`). Third-party terminals are included because their virtualized
# scrollback both leaks secrets and re-renders animated glyphs every frame (the
# capture-bloat source Layer 1 / `volatility` addresses).
_DEFAULT_BLOCKLIST: list[str] = [
    "com.apple.Terminal",
    "com.googlecode.iterm2",
    "com.mitchellh.ghostty",
    "net.kovidgoyal.kitty",
    "dev.warp.Warp-Stable",
    "co.zeit.hyper",
    "org.alacritty",
    "com.github.wez.wezterm",
    "com.microsoft.VSCode",
    "com.1password.1password",
    "com.agilebits.onepassword7",
    "com.apple.keychainaccess",
]

# The macOS menu-bar app persists the user's capture allowlist in its
# ``UserDefaults.standard`` domain (its bundle id). The CLI is a SEPARATE process
# with no shared defaults, so absent an ``OPENBIRD_ALLOWLIST`` override it reads
# this domain directly to stay in lockstep with what the app captures — otherwise
# ``openbird doctor`` reports EMPTY and a hand-started ``openbird capture --loop``
# records nothing even though the app is configured. Keep these in sync with
# mac-app/Sources/OpenBirdApp/Services/OpenBirdService.swift (allowlistKey).
_GUI_PREFS_DOMAIN = "ai.openbird.OpenBird"
_GUI_ALLOWLIST_KEY = "openbird.captureAllowlist"


def _default_data_dir() -> Path:
    """Resolve the OpenBird data directory (``~/.openbird`` by default)."""
    env = os.environ.get("OPENBIRD_DATA_DIR")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".openbird"


# Memory cutoff (bytes) separating the small-RAM (16 GB Mac) generation tier from
# the large-RAM (24/32 GB) tier. 18 GiB cleanly splits the two real Apple-Silicon
# SKUs: a 16 GB machine reports just under 16 GiB, a 24/32 GB machine well above
# 18 GiB. Any 16–18 GiB host falls into the conservative (lighter-model) tier.
_LLM_TIER_BYTES = 18 * 1024**3


def _total_memory_bytes() -> int:
    """Best-effort total physical memory in bytes; NEVER raises.

    macOS-first: ``sysctl -n hw.memsize`` is the reliable source on Apple Silicon
    (``os.sysconf("SC_PHYS_PAGES")`` is not dependable across macOS Python builds).
    Falls back to POSIX sysconf (Linux/CI), then to ``0`` — which classifies as the
    small (lighter-model) tier — if both are unavailable. Evaluated at call time so
    the tier reflects the running machine, not import-time state.
    """
    try:
        out = subprocess.run(
            ["/usr/sbin/sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
        return int(out.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        # POSIX sysconf may return -1 for an indeterminate value; only trust a
        # strictly-positive product so the non-negative "bytes" contract holds.
        if pages > 0 and page_size > 0:
            return pages * page_size
    except (OSError, ValueError, AttributeError):
        pass
    return 0


def _default_llm_model() -> str:
    """Pick the default local generation model by total memory (RAM-tiered).

    Small-memory machines (≈16 GB Macs, ``<= _LLM_TIER_BYTES``) default to the
    lighter ``ollama/qwen3:4b``; larger machines (24/32 GB) to ``ollama/qwen3:8b``.
    ``OPENBIRD_LLM_MODEL`` (or an explicit constructor arg) overrides this entirely.
    """
    if _total_memory_bytes() <= _LLM_TIER_BYTES:
        return "ollama/qwen3:4b"
    return "ollama/qwen3:8b"


def data_dir_path() -> Path:
    """Resolve the data dir WITHOUT creating it (side-effect-free).

    Mirrors :func:`_default_data_dir` precedence (``OPENBIRD_DATA_DIR`` →
    ``~/.openbird``) but, unlike :func:`get_settings`, never calls ``mkdir`` /
    ``chmod`` via :meth:`Settings.__post_init__`. Use this for read-only / destructive
    flows (e.g. ``uninstall``, ``--dry-run``) where merely resolving the path must
    not materialize ``~/.openbird`` on an otherwise-clean machine.
    """
    return _default_data_dir()


def db_file_path() -> Path:
    """Resolve the DB file WITHOUT creating anything (side-effect-free).

    Mirrors :meth:`Settings.__post_init__` precedence: a non-empty
    ``OPENBIRD_DB_PATH`` wins (it may point OUTSIDE the data dir); otherwise
    ``<data dir>/openbird.db``. An empty ``OPENBIRD_DB_PATH`` is treated as unset,
    matching :func:`_settings_from_env`.
    """
    override = os.environ.get("OPENBIRD_DB_PATH")
    if override:
        return Path(override).expanduser()
    return data_dir_path() / "openbird.db"


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
    # Generation model default is RAM-tiered (see _default_llm_model): qwen3:4b on
    # ~16 GB Macs, qwen3:8b on 24/32 GB. OPENBIRD_LLM_MODEL overrides it.
    llm_model: str = field(default_factory=_default_llm_model)
    # Default embedder: Google EmbeddingGemma (300M, native 768-dim, strongest
    # sub-500M retrieval). Switching from a prior embedder changes the embedding
    # cohort — existing stores must `openbird reindex` once (the CLI guides this;
    # see EmbeddingCohortMismatch). nomic-embed-text remains a documented fallback
    # for >2K-token chunks (EmbeddingGemma's context window is ~2K).
    embed_model: str = "ollama/embeddinggemma"
    embed_dim: int = 768

    # LLM resilience: explicit timeouts and bounded retries so a
    # reachable-but-wedged Ollama (or a flaky cloud endpoint) can never hang the
    # process forever. Values are seconds; retries are passed to LiteLLM, which
    # backs off on connection errors / 5xx / rate-limit responses.
    llm_timeout: float = 60.0
    embed_timeout: float = 30.0
    llm_num_retries: int = 2

    # Cloud opt-in: OpenBird is local-first. Resolving a *remote* model
    # (anything not ollama/* or an mlx local backend) silently POSTs private
    # memory to a third party, so it MUST be explicitly opted into. Default off.
    allow_cloud: bool = False
    # Deep Brain is a separate, future cloud-reasoning consent layer. The preview
    # command is local-only; this flag only marks whether a future sender may use
    # its packet once OPENBIRD_ALLOW_CLOUD is also enabled.
    deep_brain_enabled: bool = False
    deep_brain_excluded_apps: list[str] = field(default_factory=list)
    deep_brain_excluded_sources: list[str] = field(default_factory=list)
    deep_brain_excluded_observation_ids: list[str] = field(default_factory=list)

    # Runtime Ollama host: the base URL threaded into LiteLLM as api_base
    # for ollama/* models, so the runtime provider talks to the SAME host that
    # preflight probes. None falls back to the OLLAMA_HOST / OPENBIRD_OLLAMA_HOST
    # env vars (see resolved_ollama_host) then the localhost default.
    ollama_host: str | None = None

    # Directory holding user persona overrides for swappable system prompts
    # (``<prompts_dir>/<key>.txt``). None resolves to ``<data dir>/prompts`` in
    # __post_init__. OPENBIRD_PROMPTS_DIR overrides. The dir is created lazily by
    # ``openbird prompts edit``; runtime reads tolerate its absence.
    prompts_dir: str | None = None

    allowlist: list[str] = field(default_factory=list)
    blocklist: list[str] = field(default_factory=lambda: list(_DEFAULT_BLOCKLIST))

    ocr_enabled: bool = False
    # Opt-in: capture the active browser tab's URL via Apple Events. OFF by
    # default — enabling it makes the helper script browsers (Chrome/Safari/…),
    # which triggers a one-time macOS Automation consent prompt per browser. URLs
    # are scrubbed (query/fragment + tokens) before storage. OPENBIRD_CAPTURE_URLS=1.
    capture_urls: bool = False
    # Set by storage.crypto.open_encrypted_db depending on whether an encrypted
    # backend (SQLCipher) is actually available. Default False = "local-only,
    # not yet app-encrypted".
    encryption_enabled: bool = False
    # When True (env OPENBIRD_REQUIRE_ENCRYPTION=1), opening the DB RAISES rather
    # than silently degrading to a plaintext file when SQLCipher cannot be
    # verified. Default False keeps the backward-compatible plaintext
    # fallback.
    require_encryption: bool = False
    # Retention window in days. When > 0, observations older than this many days
    # are eligible for `openbird data prune` / programmatic pruning (H10). 0 (the
    # default) disables automatic retention; data is kept until explicitly pruned.
    retention_days: int = 0

    # Episodic-session gap (seconds). The capture daemon starts a NEW session id
    # when the foreground app changes or activity pauses longer than this, so
    # temporal recall ("what did I do today") can group contiguous activity. Kept
    # in sync with capture.daemon._DEFAULT_SESSION_GAP.
    session_gap_seconds: float = 300.0

    # Optional cross-encoder reranker (between RRF fusion and MMR). DISABLED by
    # default: an empty rerank_model is a no-op, so search behavior is unchanged.
    # When set (e.g. "bge-reranker-v2-m3"), search reorders the fused candidates by
    # a llama.cpp-compatible /v1/rerank endpoint at rerank_host, then falls back to
    # RRF order on any error. A NON-loopback rerank_host is a remote route (sends
    # query+chunk text off-device) and is cloud-gated exactly like the llm/embed
    # roles.
    rerank_model: str = ""
    rerank_host: str | None = None
    rerank_top_n: int = 0  # 0 = rerank all fused candidates
    rerank_timeout: float = 10.0

    def __post_init__(self) -> None:
        # session_gap_seconds feeds numeric session-boundary arithmetic in the
        # capture daemon (_session_for). A NaN/inf gap freezes segmentation (every
        # finite comparison is False) and a negative gap over-splits every frame,
        # both silently — so reject non-finite/negative values from env/config here
        # at the single source of truth rather than letting them propagate.
        self.session_gap_seconds = float(self.session_gap_seconds)
        if not math.isfinite(self.session_gap_seconds) or self.session_gap_seconds < 0:
            raise ValueError(
                "session_gap_seconds must be a finite, non-negative number"
            )
        # The reranker deadline is a HARD wall-clock bound on a synchronous call in
        # the search path; a NaN/inf/<=0 value would disable the deadline and let a
        # wedged rerank server hang every search. Reject it at the single source.
        self.rerank_timeout = float(self.rerank_timeout)
        if not math.isfinite(self.rerank_timeout) or self.rerank_timeout <= 0:
            raise ValueError("rerank_timeout must be a finite, positive number")
        # A negative rerank_top_n would silently fall through to the "rerank all"
        # path (HTTPReranker only emits top_n when > 0); reject it so the surface is
        # unambiguous (0 = rerank all candidates; >0 = cap how many to return).
        self.rerank_top_n = int(self.rerank_top_n)
        if self.rerank_top_n < 0:
            raise ValueError("rerank_top_n must be >= 0 (0 = rerank all candidates)")
        self.data_dir = Path(self.data_dir).expanduser()
        if self.db_path is None:
            self.db_path = str(self.data_dir / "openbird.db")
        if self.prompts_dir is None:
            self.prompts_dir = str(self.data_dir / "prompts")
        # Ensure the data directory exists with private (0700) permissions.
        self.data_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.data_dir, 0o700)
        except OSError:
            pass


def _coerce(name: str, raw: str, default: object) -> object:
    """Coerce an env-var string to the type implied by the field default."""
    if isinstance(default, list):
        return [item.strip() for item in raw.split(",") if item.strip()]
    if isinstance(default, bool) or name in (
        "ocr_enabled",
        "encryption_enabled",
        "allow_cloud",
        "deep_brain_enabled",
        "require_encryption",
        "capture_urls",
    ):
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
    "deep_brain_enabled": False,
    "deep_brain_excluded_apps": [],
    "deep_brain_excluded_sources": [],
    "deep_brain_excluded_observation_ids": [],
    "require_encryption": False,
    "capture_urls": False,
    "retention_days": 0,
    "session_gap_seconds": 300.0,
    "embed_dim": 768,
    "llm_timeout": 60.0,
    "embed_timeout": 30.0,
    "llm_num_retries": 2,
    "rerank_top_n": 0,
    "rerank_timeout": 10.0,
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
    the actual runtime call always target the same host. The result is
    always a full ``scheme://...`` URL: a bare ``host:port`` (the common
    ``OLLAMA_HOST`` form) is normalized to ``http://host:port`` so LiteLLM's
    ``api_base`` and preflight's ``urljoin`` both work.
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


# LiteLLM model-string prefixes that route to a local Ollama server. Both the
# generate (``ollama/``) and chat (``ollama_chat/``) prefixes are valid; missing
# the latter would misclassify a local chat model as cloud and skip api_base.
_OLLAMA_PREFIXES: tuple[str, ...] = ("ollama/", "ollama_chat/")


def is_ollama_model(model: str) -> bool:
    """True if ``model`` is a LiteLLM Ollama model string (generate or chat)."""
    name = (model or "").strip().lower()
    return name.startswith(_OLLAMA_PREFIXES)


def ollama_bare_model(model: str) -> str | None:
    """Return the bare Ollama model name (prefix stripped), or None if not Ollama.

    ``ollama/llama3.2:3b`` -> ``llama3.2:3b``; ``ollama_chat/llama3.2`` ->
    ``llama3.2``. The ``:tag`` is preserved.
    """
    name = (model or "").strip()
    for prefix in _OLLAMA_PREFIXES:
        if name.lower().startswith(prefix):
            return name[len(prefix):]
    return None


# Loopback host names that keep traffic on this machine. A non-loopback Ollama
# host means captured chunks leave the device, so it is treated as remote even
# for an ollama/* model (route-based cloud classification).
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


def _read_gui_allowlist() -> list[str] | None:
    """Read the menu-bar app's saved capture allowlist from macOS user defaults.

    Returns the normalized list, or ``None`` when it cannot be read (non-macOS,
    domain/key absent, malformed, or ``defaults`` unavailable/slow) so the caller
    falls back to the empty default. Read via ``defaults export`` (cfprefsd-backed,
    so it reflects the app's latest value even when the on-disk plist lags) rather
    than parsing the plist file directly. Best-effort and never raises: a failure
    here must degrade to "no allowlist", never crash the CLI.
    """
    if sys.platform != "darwin":
        return None
    try:
        # Absolute path (never a PATH-shadowed binary); capture_output keeps a
        # "domain does not exist" message off the CLI's own stderr; bounded so a
        # wedged cfprefsd cannot hang settings resolution.
        proc = subprocess.run(
            ["/usr/bin/defaults", "export", _GUI_PREFS_DOMAIN, "-"],
            capture_output=True,
            timeout=2.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0 or not proc.stdout:
        return None
    try:
        data = plistlib.loads(proc.stdout)
    except (plistlib.InvalidFileException, ValueError, ExpatError):
        return None
    raw = data.get(_GUI_ALLOWLIST_KEY) if isinstance(data, dict) else None
    if not isinstance(raw, list):
        return None
    # Normalize identically to the OPENBIRD_ALLOWLIST env path: str-only, trimmed,
    # de-duped, order-preserving. Return None (not []) when nothing usable remains
    # so an explicitly-empty saved list and "unreadable" both fall back to default.
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, str):
            continue
        bundle_id = item.strip()
        if bundle_id and bundle_id not in seen:
            seen.add(bundle_id)
            cleaned.append(bundle_id)
    return cleaned or None


def _settings_from_env() -> Settings:
    """Build a :class:`Settings` instance applying ``OPENBIRD_*`` overrides.

    Allowlist precedence (highest first): explicit ``Settings(allowlist=...)`` arg,
    ``OPENBIRD_ALLOWLIST`` (incl. ``""`` → explicit empty), the menu-bar app's
    saved macOS prefs (:func:`_read_gui_allowlist`), then the empty default.
    """
    overrides: dict[str, object] = {}

    for f in fields(Settings):
        env_key = f"OPENBIRD_{f.name.upper()}"
        raw = os.environ.get(env_key)
        if raw is None:
            continue
        # For SCALAR fields, an empty value means "no override": a stray
        # ``OPENBIRD_DB_PATH=""`` must fall back to ``<data dir>/openbird.db``
        # rather than becoming a degenerate empty path — and this keeps the
        # signed Swift app's DB-path resolution (which skips empty values) in
        # lockstep. For LIST fields (``_COERCE_DEFAULTS`` value is a list, e.g.
        # allowlist/blocklist), empty is MEANINGFUL: it explicitly clears to
        # ``[]`` (e.g. ``OPENBIRD_BLOCKLIST=""`` drops the default blocklist), so
        # it must NOT be skipped.
        if raw == "" and not isinstance(_COERCE_DEFAULTS.get(f.name), list):
            continue
        # Determine a sensible default for type coercion without constructing
        # a full Settings (which would touch the filesystem).
        default_val = _COERCE_DEFAULTS.get(f.name, "")
        overrides[f.name] = _coerce(f.name, raw, default_val)

    # Bridge the menu-bar app's saved allowlist into the CLI when the env var is
    # absent ("allowlist" in overrides iff OPENBIRD_ALLOWLIST was set above, incl.
    # the explicit-empty case). The app injects OPENBIRD_ALLOWLIST into the daemon
    # it spawns, so this only affects CLI invocations the app did not launch.
    if "allowlist" not in overrides:
        gui_allowlist = _read_gui_allowlist()
        if gui_allowlist is not None:
            overrides["allowlist"] = gui_allowlist

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
    "data_dir_path",
    "db_file_path",
    "resolved_ollama_host",
    "is_loopback_host",
    "is_ollama_model",
    "ollama_bare_model",
    "DEFAULT_OLLAMA_HOST",
]
