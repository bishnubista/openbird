"""Central settings, paths, model config, allow/blocklists, and feature flags.

Implemented as a plain dataclass (pydantic-settings is optional and not a hard
dependency). ``get_settings()`` returns a cached instance honoring ``OPENBIRD_*``
environment overrides.
"""

from __future__ import annotations

import logging
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

logger = logging.getLogger("openbird.config")


def _clamp_setting(
    name: str,
    value: object,
    *,
    default: float,
    lo: float,
    hi: float | None = None,
) -> float:
    """Clamp a numeric tuning knob into its legal range (never raise).

    Non-finite/unparseable values fall back to ``default`` first. Any
    adjustment logs a reason-code line (name + bounds only — settings values
    are operator-supplied numbers, safe to log).
    """
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        parsed = float("nan")
    if not math.isfinite(parsed):
        logger.warning("config: %s invalid; using default %.1f", name, default)
        parsed = default
    clamped = max(lo, parsed if hi is None else min(hi, parsed))
    if clamped != parsed:
        logger.warning(
            "config: %s=%.3f out of range; clamped to %.1f", name, parsed, clamped
        )
    return clamped

# High-context developer apps that may be explicitly enabled for detailed local
# capture. Keep this exact-ID set separate from the dangerous-app backstop:
# detailed capture may subtract one of these entries from the user blocklist, but
# it can never make a password manager / Keychain app eligible.
DETAILED_CAPTURE_ELIGIBLE_APPS: tuple[str, ...] = (
    "com.apple.Terminal",
    "com.googlecode.iterm2",
    "com.mitchellh.ghostty",
    "net.kovidgoyal.kitty",
    "dev.warp.Warp-Stable",
    "co.zeit.hyper",
    "org.alacritty",
    "com.github.wez.wezterm",
    "com.microsoft.VSCode",
)

# Apps excluded from capture until the user explicitly enables an eligible app:
# terminals, code editors, password managers, and Keychain.
# NOTE: the blocklist is SUBTRACTIVE — `redact.decide` applies the allowlist first,
# then removes blocklisted apps. So allowlisting a terminal alone will NOT capture
# it; a user must also drop it from the blocklist (e.g. OPENBIRD_BLOCKLIST=... or a
# custom `blocklist`). Third-party terminals are included because their virtualized
# scrollback both leaks secrets and re-renders animated glyphs every frame (the
# capture-bloat source Layer 1 / `volatility` addresses).
_DEFAULT_BLOCKLIST: list[str] = [
    *DETAILED_CAPTURE_ELIGIBLE_APPS,
    "com.1password.1password",
    "com.agilebits.onepassword7",
    "com.apple.keychainaccess",
]

# The macOS menu-bar app persists the user's capture allowlist (and the Phase
# C2 OCR opt-in list) in its ``UserDefaults.standard`` domain (its bundle id).
# The CLI is a SEPARATE process with no shared defaults, so absent an
# ``OPENBIRD_ALLOWLIST`` / ``OPENBIRD_CAPTURE_OCR_APPS`` override it reads this
# domain directly to stay in lockstep with what the app captures — otherwise
# ``openbird doctor`` reports EMPTY and a hand-started ``openbird capture --loop``
# records nothing even though the app is configured. Keep these in sync with
# mac-app/Sources/OpenBirdApp/Services/OpenBirdService.swift (allowlistKey /
# ocrAppsKey).
_GUI_PREFS_DOMAIN = "ai.openbird.OpenBird"
_GUI_ALLOWLIST_KEY = "openbird.captureAllowlist"
_GUI_OCR_APPS_KEY = "openbird.captureOcrApps"
_GUI_DETAILED_CAPTURE_APPS_KEY = "openbird.detailedCaptureApps"


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

    # Host label included in assistant MCP responses so a user running OpenBird
    # on several Macs can tell whose capture they are reading. None falls back
    # to platform.node(), then the "unknown-host" sentinel. Its egress is
    # disclosed in the assistant install warnings and per-response notices.
    assistant_host_label: str | None = None

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
    # Exact bundle ids from DETAILED_CAPTURE_ELIGIBLE_APPS that may bypass their
    # default terminal/editor blocklist entry. The allowlist, dangerous-app,
    # self-capture, and private-window gates remain independently fail-closed.
    # Non-empty detailed capture also forces require_encryption in __post_init__.
    detailed_capture_apps: list[str] = field(default_factory=list)

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

    # Stream-mode capture timing knobs (Phase A, event-driven capture). All four
    # are CLAMPED into their legal ranges in __post_init__ rather than rejected:
    # capture must never fail to start over a bad tuning knob, but an
    # out-of-range value (e.g. a 0.01s idle tick) would defeat the CPU/power
    # budget or the >=1s capture floor, so the clamp is the budget's enforcement
    # point. A clamped value logs a reason-code line. Env: OPENBIRD_CAPTURE_*.
    capture_afk_threshold_seconds: float = 150.0  # AFK after this HID-idle; >= 30
    capture_idle_tick_seconds: float = 5.0  # backstop poll cadence; [5, 10]
    capture_force_ceiling_seconds: float = 60.0  # max capture gap; >= max(gap, tick)
    capture_min_gap_seconds: float = 1.0  # hard floor between captures; >= 1
    # Span merge pulsetime (Phase B). 0 = derive from the idle tick
    # (max(2*tick+5, 15)); explicit values clamp to [5, ceiling].
    capture_span_pulsetime_seconds: float = 0.0

    # Opt-in OCR fallback (Phase C2). Bundle ids whose AX-EMPTY captures may
    # fall back to a window-scoped screenshot + on-device Vision OCR in the
    # stream helper. Default EMPTY = OCR fully off. Env:
    # OPENBIRD_CAPTURE_OCR_APPS (comma-separated); absent that, the menu-bar
    # app's saved ``openbird.captureOcrApps`` prefs key bridges in — exactly
    # mirroring the allowlist precedence. Subset-of-allowlist holds BY
    # CONSTRUCTION, not by validation: the helper's OCR branch sits after its
    # allowlist gate returned true, so an OCR entry that is not allowlisted is
    # simply inert — no cross-list validation code exists on purpose.
    capture_ocr_apps: list[str] = field(default_factory=list)
    # Per-app minimum seconds between OCR attempts (the helper's OcrGate
    # throttle). Clamped in __post_init__ (lo=10) — clamp-never-reject, same
    # budget-enforcement stance as the other capture_*_seconds knobs.
    capture_ocr_min_interval_seconds: float = 30.0

    # Idle-time block summaries (Phase D). Generated ONLY by the routines-daemon
    # worker / the on-demand `openbird summaries build` — never the capture or
    # chat paths. Env: OPENBIRD_BLOCK_SUMMARIES_*, OPENBIRD_TAXONOMY_LLM_BATCH_LIMIT.
    block_summaries_enabled: bool = True
    # Max blocks summarized per runner invocation (bounds each hourly firing AND
    # the single coalesced catch-up run after downtime).
    block_summaries_batch_limit: int = 8
    # Only blocks that ENDED at least this long ago are summarizable (a still-
    # growing block would churn regenerations).
    block_summaries_settle_seconds: float = 900.0
    # Trailing window the runner rescans for new/changed blocks.
    block_summaries_lookback_days: float = 3.0
    # Max uncategorized identities classified per runner invocation.
    taxonomy_llm_batch_limit: int = 5

    # Week rollups (Phase E1). Generated by the SAME routines pass (and
    # `openbird summaries build`) as block summaries — one model reduce over the
    # week's block-summary narratives. Env: OPENBIRD_WEEK_ROLLUP_*.
    week_rollup_enabled: bool = True
    # Trailing Monday-aligned weeks the runner rescans (current week included).
    week_rollup_lookback_weeks: int = 2
    # The CURRENT (live) week regenerates on member drift only after this long
    # since the last generation — a live week must not burn a model call every
    # hourly firing. Past weeks regenerate on drift alone.
    week_rollup_min_interval_seconds: float = 21600.0
    # Max stale/missing summaries (block + week) re-embedded into the summary
    # index per runner invocation. Env: OPENBIRD_SUMMARY_INDEX_BATCH_LIMIT.
    summary_index_batch_limit: int = 32

    # Entity ledger aggregation (Phase E2). A DETERMINISTIC (no-LLM) pass run
    # by the SAME gated routines invocation as block summaries; it mines repo/
    # domain entities and completion evidence from stored sources. Env:
    # OPENBIRD_ENTITY_*.
    entity_ledger_enabled: bool = True
    # First-run scan bound: days of history mined when no watermark exists yet.
    entity_aggregation_lookback_days: float = 14.0
    # Entities with no recorded activity for this long go dormant at
    # aggregation time (user_marked_done rows are immune).
    entity_dormant_after_days: float = 21.0
    # REAL row cap on observations scanned per aggregation run (the composite
    # (ts, id) cursor resumes where the capped run stopped).
    entity_evidence_batch_limit: int = 2000

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
        eligible = {bundle_id.lower(): bundle_id for bundle_id in DETAILED_CAPTURE_ELIGIBLE_APPS}
        normalized_detailed: list[str] = []
        seen_detailed: set[str] = set()
        for item in self.detailed_capture_apps:
            if not isinstance(item, str):
                continue
            canonical = eligible.get(item.strip().lower())
            if canonical is not None and canonical not in seen_detailed:
                seen_detailed.add(canonical)
                normalized_detailed.append(canonical)
        self.detailed_capture_apps = normalized_detailed
        # Detailed terminal/editor capture can include commands, output, and
        # copied secrets. It must never open or create a plaintext fallback DB,
        # even when OPENBIRD_REQUIRE_ENCRYPTION was omitted or explicitly false.
        if self.detailed_capture_apps:
            self.require_encryption = True
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
        # Clamp the stream-capture timing knobs into their legal ranges (never
        # reject: capture must not fail to start over a tuning knob, but the
        # ranges ARE the CPU/power budget, so out-of-range values can't stand).
        # NaN/inf fall back to the field default before clamping.
        self.capture_afk_threshold_seconds = _clamp_setting(
            "capture_afk_threshold_seconds",
            self.capture_afk_threshold_seconds,
            default=150.0,
            lo=30.0,
        )
        self.capture_idle_tick_seconds = _clamp_setting(
            "capture_idle_tick_seconds",
            self.capture_idle_tick_seconds,
            default=5.0,
            lo=5.0,
            hi=10.0,
        )
        self.capture_min_gap_seconds = _clamp_setting(
            "capture_min_gap_seconds",
            self.capture_min_gap_seconds,
            default=1.0,
            lo=1.0,
        )
        # OCR throttle floor: 10s per app is the budget's enforcement point
        # (the Swift OcrGate defensively re-clamps the argv value too).
        self.capture_ocr_min_interval_seconds = _clamp_setting(
            "capture_ocr_min_interval_seconds",
            self.capture_ocr_min_interval_seconds,
            default=30.0,
            lo=10.0,
        )
        # Cross-field: the force ceiling must not undercut the floor or the tick
        # (a ceiling below either would force captures faster than the budget).
        self.capture_force_ceiling_seconds = _clamp_setting(
            "capture_force_ceiling_seconds",
            self.capture_force_ceiling_seconds,
            default=60.0,
            lo=max(self.capture_min_gap_seconds, self.capture_idle_tick_seconds),
        )
        # Span pulsetime: 0 means "derive"; explicit values are clamped into
        # [5, ceiling] (a pulsetime above the ceiling would merge across gaps
        # the ceiling is required to split).
        try:
            pulse = float(self.capture_span_pulsetime_seconds)
        except (TypeError, ValueError):
            pulse = 0.0
        if not math.isfinite(pulse) or pulse <= 0:
            self.capture_span_pulsetime_seconds = 0.0
        else:
            self.capture_span_pulsetime_seconds = _clamp_setting(
                "capture_span_pulsetime_seconds",
                pulse,
                default=0.0,
                lo=5.0,
                hi=self.capture_force_ceiling_seconds,
            )
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
        "encryption_enabled",
        "allow_cloud",
        "deep_brain_enabled",
        "require_encryption",
        "capture_urls",
    ):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(default, float) or name in ("llm_timeout", "embed_timeout"):
        if name.startswith("capture_") and name.endswith("_seconds"):
            # Clamp-never-reject knobs: a typo'd env value must not prevent
            # settings construction (capture must not fail to start over a
            # tuning knob). NaN routes through _clamp_setting's invalid->
            # default path in __post_init__, which logs the reason code.
            try:
                return float(raw)
            except ValueError:
                return float("nan")
        return float(raw)
    if isinstance(default, int) and not isinstance(default, bool):
        return int(raw)
    return raw


_COERCE_DEFAULTS: dict[str, object] = {
    "allowlist": [],
    "blocklist": [],
    "detailed_capture_apps": [],
    "capture_ocr_apps": [],
    "capture_ocr_min_interval_seconds": 30.0,
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
    "capture_afk_threshold_seconds": 150.0,
    "capture_idle_tick_seconds": 5.0,
    "capture_force_ceiling_seconds": 60.0,
    "capture_min_gap_seconds": 1.0,
    "capture_span_pulsetime_seconds": 0.0,
    "block_summaries_enabled": True,
    "block_summaries_batch_limit": 8,
    "block_summaries_settle_seconds": 900.0,
    "block_summaries_lookback_days": 3.0,
    "taxonomy_llm_batch_limit": 5,
    "week_rollup_enabled": True,
    "week_rollup_lookback_weeks": 2,
    "week_rollup_min_interval_seconds": 21600.0,
    "summary_index_batch_limit": 32,
    "entity_ledger_enabled": True,
    "entity_aggregation_lookback_days": 14.0,
    "entity_dormant_after_days": 21.0,
    "entity_evidence_batch_limit": 2000,
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


def _read_gui_string_list(key: str) -> list[str] | None:
    """Read one saved string-list ``key`` from the menu-bar app's user defaults.

    Returns the normalized list, or ``None`` when it cannot be read (non-macOS,
    domain/key absent, malformed, or ``defaults`` unavailable/slow) so the caller
    falls back to the empty default. Read via ``defaults export`` (cfprefsd-backed,
    so it reflects the app's latest value even when the on-disk plist lags) rather
    than parsing the plist file directly. Best-effort and never raises: a failure
    here must degrade to "no list", never crash the CLI.
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
    raw = data.get(key) if isinstance(data, dict) else None
    if not isinstance(raw, list):
        return None
    # Normalize identically to the OPENBIRD_* env list path: str-only, trimmed,
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


def _read_gui_allowlist() -> list[str] | None:
    """The menu-bar app's saved capture allowlist (see _read_gui_string_list)."""
    return _read_gui_string_list(_GUI_ALLOWLIST_KEY)


def _read_gui_ocr_apps() -> list[str] | None:
    """The menu-bar app's saved OCR opt-in list (see _read_gui_string_list).

    Kept as a named wrapper (not an inline ``_read_gui_string_list`` call) so
    tests can neutralize/patch each GUI bridge independently, exactly like
    ``_read_gui_allowlist``.
    """
    return _read_gui_string_list(_GUI_OCR_APPS_KEY)


def _read_gui_detailed_capture_apps() -> list[str] | None:
    """The menu-bar app's saved per-app detailed-capture grants."""
    return _read_gui_string_list(_GUI_DETAILED_CAPTURE_APPS_KEY)


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

    # Same bridge for the Phase C2 OCR opt-in list (env wins, incl. explicit
    # empty; otherwise the app's saved ``openbird.captureOcrApps`` applies).
    if "capture_ocr_apps" not in overrides:
        gui_ocr_apps = _read_gui_ocr_apps()
        if gui_ocr_apps is not None:
            overrides["capture_ocr_apps"] = gui_ocr_apps

    # Same bridge for explicit terminal/editor detailed-capture grants. Env wins,
    # including an explicit empty value that turns every grant off.
    if "detailed_capture_apps" not in overrides:
        gui_detailed_apps = _read_gui_detailed_capture_apps()
        if gui_detailed_apps is not None:
            overrides["detailed_capture_apps"] = gui_detailed_apps

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
    "DETAILED_CAPTURE_ELIGIBLE_APPS",
]
