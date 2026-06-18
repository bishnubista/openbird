"""Preflight: a pure-Python readiness report for an OpenBird install.

``run_preflight()`` aggregates a dict describing whether the environment can
actually run OpenBird:

* Ollama reachable + required models present (best-effort HTTP probe).
* The configured embedding dimension and whether a probe confirmed it.
* ``sqlite-vec`` and FTS5 availability (loaded in-process, no network).
* DB encryption status (delegated to :mod:`openbird.storage.crypto`).
* The active allowlist / blocklist and the OCR feature flag.
* macOS-only capability stubs (TCC / Accessibility, audio capture) that report
  ``"unknown"`` off-mac or when they cannot be determined.

Every check is defensive: no probe is allowed to raise. When a check cannot be
performed the result is ``"unknown"`` (or ``False`` for hard capability flags),
never an exception. Dependencies (settings, the HTTP getter, the DB opener, the
embedding provider factory, and the platform identifier) are injectable so the
aggregation can be unit-tested with fakes and no live services.
"""

from __future__ import annotations

import json
import os
import platform
import sqlite3
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Protocol
from urllib.parse import urljoin

from openbird.config import (
    Settings,
    get_settings,
    is_ollama_model,
    ollama_bare_model,
    resolved_ollama_host,
)

# Models OpenBird depends on by default; preflight checks they are pulled. Used
# as a fallback when settings do not name ollama/* models (route-aware path
# below derives the real required set from settings.llm_model/embed_model).
_REQUIRED_MODELS: tuple[str, ...] = ("llama3.2", "nomic-embed-text")


# --------------------------------------------------------------------------- #
# Typing helpers                                                              #
# --------------------------------------------------------------------------- #


class _EmbedProvider(Protocol):
    """The slice of :class:`~openbird.llm.provider.LLMProvider` preflight uses."""

    embed_dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


# Inject points (all defaulted) so the aggregation is testable with fakes.
HttpGetter = Callable[[str, float], "tuple[int, bytes]"]
DbOpener = Callable[..., sqlite3.Connection]
ProviderFactory = Callable[[Settings], _EmbedProvider]


def _ollama_host(settings: Settings | None = None) -> str:
    """Resolve the Ollama base URL via the shared resolver.

    Delegates to :func:`openbird.config.resolved_ollama_host` so preflight and
    the runtime provider always agree on the host (same precedence:
    ``OLLAMA_HOST`` > ``settings.ollama_host`` / ``OPENBIRD_OLLAMA_HOST`` >
    default).
    """
    return resolved_ollama_host(settings)


def _ollama_required_models(settings: Settings) -> tuple[str, ...]:
    """Derive the local Ollama models the active route actually needs.

    Strips only the Ollama provider prefix, KEEPING any ``:tag`` (e.g.
    ``ollama/llama3.2:3b`` -> ``llama3.2:3b``) so preflight checks the exact model
    the runtime will request — otherwise a green preflight could accept a
    different tag (``llama3.2:latest``) than the provider pulls.

    For a route that uses NO Ollama models at all (cloud-only / mlx-only) this
    returns an EMPTY tuple — preflight must not report the default ``llama3.2`` /
    ``nomic-embed-text`` as required/missing for a route that never touches
    Ollama. The defaults are used only as the unconfigured-but-Ollama fallback.
    """
    wanted: list[str] = []
    for model in (settings.llm_model, settings.embed_model):
        bare = ollama_bare_model(model)
        if bare and bare not in wanted:
            wanted.append(bare)
    if wanted:
        return tuple(wanted)
    # No ollama models configured: only fall back to defaults if neither model is
    # set to something (preserves the bare-defaults case); a cloud/mlx route gets
    # an empty required set.
    any_model_configured = bool(
        (settings.llm_model or "").strip() or (settings.embed_model or "").strip()
    )
    return () if any_model_configured else _REQUIRED_MODELS


def _has_mlx_model(settings: Settings) -> bool:
    """True if either configured model is an ``mlx*`` string (unwired today).

    The MLX runtime is reserved/not wired, and litellm cannot serve ``mlx/*``
    model strings, so a route naming one is not runnable regardless of backend.
    """
    return any(
        (m or "").strip().lower().startswith("mlx")
        for m in (settings.llm_model, settings.embed_model)
    )


def _http_get(url: str, timeout: float) -> tuple[int, bytes]:
    """Perform a GET, returning ``(status_code, body)``.

    Raises on transport errors; callers wrap this so failures become
    ``reachable=False`` rather than propagating.
    """
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (localhost)
        return resp.getcode(), resp.read()


# --------------------------------------------------------------------------- #
# Individual checks                                                           #
# --------------------------------------------------------------------------- #


def check_ollama(
    *,
    host: str | None = None,
    required_models: tuple[str, ...] = _REQUIRED_MODELS,
    http_get: HttpGetter = _http_get,
    timeout: float = 2.0,
) -> dict[str, Any]:
    """Probe a local Ollama server for reachability and required models.

    Returns a dict with ``reachable`` (bool), ``host``, the list of
    ``models_present`` reported by the server, and per-required-model presence
    plus a ``missing_models`` list. Never raises: transport errors yield
    ``reachable=False`` with an ``error`` string.
    """
    base = host or _ollama_host()
    result: dict[str, Any] = {
        "reachable": False,
        "host": base,
        "models_present": [],
        "required_models": list(required_models),
        "models": {m: False for m in required_models},
        "missing_models": list(required_models),
        "error": None,
    }
    try:
        status, body = http_get(urljoin(base.rstrip("/") + "/", "api/tags"), timeout)
    except Exception as exc:  # connection refused, timeout, DNS, etc.
        result["error"] = type(exc).__name__
        return result

    if status != 200:
        result["error"] = f"http {status}"
        return result

    result["reachable"] = True
    try:
        payload = json.loads(body.decode("utf-8")) if body else {}
        names = [m.get("name", "") for m in payload.get("models", []) if isinstance(m, dict)]
    except (ValueError, AttributeError, TypeError) as exc:
        result["error"] = f"parse:{type(exc).__name__}"
        names = []

    result["models_present"] = names
    present = {}
    for req in required_models:
        # Match "llama3.2" against "llama3.2:latest", "llama3.2:3b", etc.
        present[req] = any(n == req or n.split(":", 1)[0] == req for n in names)
    result["models"] = present
    result["missing_models"] = [m for m, ok in present.items() if not ok]
    return result


def check_embedding(
    settings: Settings,
    *,
    provider_factory: ProviderFactory | None = None,
    probe: bool = False,
) -> dict[str, Any]:
    """Report the configured embedding model/dimension, optionally probing it.

    Without ``probe`` (the default, network-free) this reports the configured
    ``embed_model`` and ``embed_dim`` with ``probed=False``. With ``probe=True``
    a real embedding is requested through ``provider_factory`` and the returned
    dimension is compared to the configured one. Probe failures are captured in
    ``error`` and never raised.
    """
    result: dict[str, Any] = {
        "model": settings.embed_model,
        "configured_dim": settings.embed_dim,
        "probed": False,
        "probed_dim": None,
        "dim_ok": None,
        "error": None,
    }
    if not probe:
        return result

    try:
        factory = provider_factory or _default_provider_factory
        provider = factory(settings)
        vectors = provider.embed(["preflight"])
        dim = len(vectors[0]) if vectors else 0
        result["probed"] = True
        result["probed_dim"] = dim
        result["dim_ok"] = dim == settings.embed_dim
    except Exception as exc:
        result["error"] = type(exc).__name__
    return result


def check_completion(
    settings: Settings,
    *,
    provider_factory: ProviderFactory | None = None,
    probe: bool = False,
) -> dict[str, Any]:
    """Report the configured chat model, optionally probing a tiny completion.

    Without ``probe`` (the default, network-free) this only echoes the configured
    ``llm_model`` with ``probed=False`` / ``ok=None``. With ``probe=True`` a
    minimal completion is requested so a REMOTE chat model's endpoint/credentials
    are validated before preflight reports READY (a successful *embedding* probe
    does not prove the chat endpoint works). Failures are captured, never raised.
    """
    result: dict[str, Any] = {
        "model": settings.llm_model,
        "probed": False,
        "ok": None,
        "error": None,
    }
    if not probe:
        return result

    try:
        factory = provider_factory or _default_provider_factory
        provider = factory(settings)
        text = provider.complete([{"role": "user", "content": "ping"}])
        result["probed"] = True
        result["ok"] = isinstance(text, (str, dict))
    except Exception as exc:
        result["error"] = type(exc).__name__
    return result


def _default_provider_factory(settings: Settings) -> _EmbedProvider:
    """Construct the configured provider (imported lazily)."""
    from openbird.llm.provider import create_llm_provider

    return create_llm_provider(settings)


def check_sqlite_vec(*, connect: Callable[[], sqlite3.Connection] | None = None) -> dict[str, Any]:
    """Report whether ``sqlite-vec`` and FTS5 are usable in this process.

    Loads the sqlite-vec extension into an in-memory DB and queries
    ``vec_version()``; checks FTS5 by creating a throwaway virtual table. Both
    sub-checks are independent and never raise out of this function.
    """
    result: dict[str, Any] = {
        "sqlite_version": sqlite3.sqlite_version,
        "vec_available": False,
        "vec_version": None,
        "fts5_available": False,
        "error": None,
    }

    def _new_conn() -> sqlite3.Connection:
        return (connect or (lambda: sqlite3.connect(":memory:")))()

    # sqlite-vec
    try:
        conn = _new_conn()
        try:
            import sqlite_vec

            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            (version,) = conn.execute("SELECT vec_version()").fetchone()
            result["vec_available"] = True
            result["vec_version"] = version
        finally:
            conn.close()
    except Exception as exc:
        result["error"] = type(exc).__name__

    # FTS5
    try:
        conn = _new_conn()
        try:
            conn.execute("CREATE VIRTUAL TABLE _pf_fts USING fts5(x)")
            result["fts5_available"] = True
        finally:
            conn.close()
    except Exception:
        result["fts5_available"] = False

    return result


def _verify_cipher(conn: sqlite3.Connection) -> str | None:
    """Probe ``PRAGMA cipher_version`` on a *live* connection.

    A plain ``sqlite3`` connection does not implement this pragma and yields no
    row, so a non-empty result is the authoritative proof that the connection is
    actually SQLCipher-backed. Never raises.
    """
    try:
        from openbird.storage.crypto import cipher_version

        return cipher_version(conn)
    except Exception:
        # Fall back to an inline probe if crypto can't be imported.
        try:
            row = conn.execute("PRAGMA cipher_version").fetchone()
        except Exception:
            return None
        if not row or row[0] is None:
            return None
        text = str(row[0]).strip()
        return text or None


def check_encryption(
    settings: Settings,
    *,
    db_opener: DbOpener | None = None,
) -> dict[str, Any]:
    """Report DB encryption status by VERIFYING SQLCipher on the live connection.

    Opens the database via ``db_opener`` and inspects the connection directly:
    encryption is reported only when ``PRAGMA cipher_version`` returns a
    non-empty value on that live connection (and, if the opener exposes a
    verified :class:`~openbird.storage.crypto.DbHandle`, when it agrees). A plain
    ``sqlite3`` connection — even with ``settings.encryption_enabled`` flipped
    True — must NOT pass: the settings flag is never trusted here.
    Failures are reported as ``status="unknown"`` and never raised.
    """
    result: dict[str, Any] = {
        "enabled": None,
        "status": "unknown",
        "backend": "unknown",
        "cipher_version": None,
        "wal_enabled": None,
        "verified": False,
        "error": None,
    }
    try:
        opener = db_opener or _default_db_opener
        opened = opener(settings.db_path, settings=settings)

        # Support both a bare connection and a verified DbHandle.
        handle = None
        conn = opened
        if hasattr(opened, "conn") and hasattr(opened, "encrypted"):
            handle = opened
            conn = opened.conn

        try:
            live_cipher = _verify_cipher(conn)
            live_encrypted = live_cipher is not None
            # The handle's own claim must AGREE with the live probe; trust the
            # stricter of the two so a lying handle cannot fake encryption.
            if handle is not None:
                handle_encrypted = bool(getattr(handle, "encrypted", False))
                encrypted = live_encrypted and handle_encrypted
                backend = getattr(handle, "backend", None) or (
                    "sqlcipher" if encrypted else "sqlite3"
                )
                wal = bool(getattr(handle, "wal_enabled", False))
                cipher = getattr(handle, "cipher_version", None) or live_cipher
            else:
                encrypted = live_encrypted
                backend = "sqlcipher" if encrypted else "sqlite3"
                wal = bool(_wal_enabled(conn)) if encrypted else False
                cipher = live_cipher
        finally:
            try:
                conn.close()
            except Exception:
                pass

        result["enabled"] = encrypted
        result["verified"] = True
        result["cipher_version"] = cipher
        result["wal_enabled"] = wal
        result["status"] = "encrypted" if encrypted else "plaintext-0600"
        result["backend"] = backend if encrypted else "sqlite3"
    except Exception as exc:
        result["error"] = type(exc).__name__
    return result


def _wal_enabled(conn: sqlite3.Connection) -> bool:
    """Best-effort check that the connection's journal mode is WAL."""
    try:
        row = conn.execute("PRAGMA journal_mode").fetchone()
    except Exception:
        return False
    return bool(row) and str(row[0]).lower() == "wal"


def _default_db_opener(path: str | None, *, settings: Settings):
    """Open the DB via crypto (imported lazily), returning a verified handle.

    Returns a :class:`~openbird.storage.crypto.DbHandle` so :func:`check_encryption`
    can cross-check the backend's claim against a live ``cipher_version`` probe.
    """
    from openbird.storage.crypto import open_db_verified

    return open_db_verified(path, settings=settings)


# Grant states for a macOS TCC/audio capability.
GRANT_PASSED = "passed"
GRANT_FAILED = "failed"
GRANT_UNKNOWN = "unknown"

# The TCC/audio capabilities a packaged signed helper must report on.
_MACOS_GATES: tuple[str, ...] = (
    "accessibility",
    "screen_recording",
    "microphone",
    "system_audio",
)


class HelperProbe(Protocol):
    """A signed-helper probe: maps a capability name to its grant state.

    Implemented by the packaged signed helper / LaunchAgent. It
    returns one of :data:`GRANT_PASSED` / :data:`GRANT_FAILED` /
    :data:`GRANT_UNKNOWN` for a given capability. Pure-Python preflight cannot
    determine TCC grants itself, so without a helper every gate is ``unknown``.
    """

    def __call__(self, capability: str) -> str: ...


CAPTURE_HELPER_PATH_ENV = "OPENBIRD_CAPTURE_HELPER"
AUDIO_HELPER_PATH_ENV = "OPENBIRD_AUDIO_HELPER"
_HELPER_PROBE_ARG = "--preflight-grants"
_HELPER_PROBE_TIMEOUT_SECONDS = 3.0


def _normalize_grant(value: object) -> str:
    """Coerce a probe return value to a known grant state (default unknown)."""
    text = str(value).strip().lower()
    if text in (GRANT_PASSED, "granted", "authorized", "ok", "true", "yes"):
        return GRANT_PASSED
    if text in (GRANT_FAILED, "denied", "restricted", "false", "no"):
        return GRANT_FAILED
    return GRANT_UNKNOWN


def _helper_path_from_env(name: str) -> Path | None:
    """Return an executable helper path from ``name`` when the app wrapper set it."""
    raw = os.environ.get(name)
    if not raw:
        return None
    path = Path(raw).expanduser()
    if path.is_file() and os.access(path, os.X_OK):
        return path
    return None


def _run_helper_grant_probe(path: Path) -> dict[str, str]:
    """Run one packaged helper's non-capturing grant probe.

    Helpers return a flat JSON object such as ``{"accessibility": "passed"}``.
    Raises on execution/parsing failure so the final report can distinguish a
    broken probe from a helper that successfully reported an ungranted gate.
    """
    try:
        proc = subprocess.run(
            [str(path), _HELPER_PROBE_ARG],
            check=False,
            capture_output=True,
            text=True,
            timeout=_HELPER_PROBE_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        raise RuntimeError(type(exc).__name__) from exc
    if proc.returncode != 0:
        raise RuntimeError(f"exit:{proc.returncode}")
    try:
        payload = json.loads(proc.stdout)
    except Exception as exc:
        raise ValueError(type(exc).__name__) from exc
    if not isinstance(payload, dict):
        raise ValueError("non_object_json")
    return {str(k): _normalize_grant(v) for k, v in payload.items()}


def _packaged_helper_probe(
    *,
    capture_helper: Path | None = None,
    audio_helper: Path | None = None,
    runner: Callable[[Path], dict[str, str]] = _run_helper_grant_probe,
) -> HelperProbe | None:
    """Build a cached probe over the helpers bundled beside ``openbird-cli``.

    The app wrapper exports helper paths before invoking the Python CLI. Outside
    that packaged path there may be no stable signed helper, so returning
    ``None`` keeps preflight's report honestly ``unknown``.
    """
    capture_path = capture_helper or _helper_path_from_env(CAPTURE_HELPER_PATH_ENV)
    audio_path = audio_helper or _helper_path_from_env(AUDIO_HELPER_PATH_ENV)
    if capture_path is None and audio_path is None:
        return None

    cache: dict[str, dict[str, str]] = {}
    errors: dict[str, str] = {}

    def _probe(capability: str) -> str:
        if capability == "accessibility":
            if capture_path is None:
                raise FileNotFoundError(CAPTURE_HELPER_PATH_ENV)
            key, path = "capture", capture_path
        else:
            if audio_path is None:
                raise FileNotFoundError(AUDIO_HELPER_PATH_ENV)
            key, path = "audio", audio_path
        if key in errors:
            raise RuntimeError(errors[key])
        if key not in cache:
            try:
                cache[key] = runner(path)
            except Exception as exc:
                errors[key] = type(exc).__name__
                raise
        return _normalize_grant(cache[key].get(capability, GRANT_UNKNOWN))

    return _probe


def check_macos_capabilities(
    *,
    system: str | None = None,
    helper_probe: HelperProbe | None = None,
) -> dict[str, Any]:
    """macOS TCC / Accessibility + audio capability gate.

    Authoritative grant states come from a packaged **signed helper** via
    ``helper_probe`` — pure-Python preflight cannot read TCC. Each
    gate is reported as :data:`GRANT_PASSED` / :data:`GRANT_FAILED` /
    :data:`GRANT_UNKNOWN`. Off-mac everything is ``unknown``/unavailable. Without
    a helper, every gate stays ``unknown`` (the honest default) — preflight must
    NOT report green from a stub. ``all_passed`` is True only when every gate is
    ``passed`` (so ``unknown`` never counts as green).
    """
    plat = system or platform.system()
    is_mac = plat == "Darwin"

    grants: dict[str, str] = {g: GRANT_UNKNOWN for g in _MACOS_GATES}
    probe_error: str | None = None

    if is_mac and helper_probe is not None:
        for cap in _MACOS_GATES:
            try:
                grants[cap] = _normalize_grant(helper_probe(cap))
            except Exception as exc:
                grants[cap] = GRANT_UNKNOWN
                probe_error = type(exc).__name__

    all_passed = is_mac and all(grants[g] == GRANT_PASSED for g in _MACOS_GATES)
    any_failed = any(grants[g] == GRANT_FAILED for g in _MACOS_GATES)
    helper_present = helper_probe is not None

    if not is_mac:
        note = "Not macOS; capture/audio capabilities are unavailable."
    elif not helper_present:
        note = (
            "Authoritative TCC/Accessibility and audio-capture checks require the "
            "packaged signed helper; reported as 'unknown' here."
        )
    elif probe_error:
        note = (
            "Packaged helper probe was unavailable or failed; unverified gates "
            "are reported as 'unknown'."
        )
    elif all_passed:
        note = "All TCC/audio gates granted (verified via signed helper)."
    elif any_failed:
        note = "Signed helper reported some TCC/audio gates not passed."
    else:
        note = "Signed helper could not prove every TCC/audio gate."

    return {
        "platform": plat,
        "is_macos": is_mac,
        "helper_present": helper_present,
        "accessibility": grants["accessibility"],
        "screen_recording": grants["screen_recording"],
        "microphone": grants["microphone"],
        "system_audio": grants["system_audio"],
        "all_passed": all_passed,
        "any_failed": any_failed,
        "probe_error": probe_error,
        "note": note,
    }


# --------------------------------------------------------------------------- #
# Aggregation                                                                 #
# --------------------------------------------------------------------------- #


def run_preflight(
    settings: Settings | None = None,
    *,
    probe_ollama: bool = True,
    probe_embedding: bool = False,
    probe_encryption: bool = True,
    http_get: HttpGetter = _http_get,
    provider_factory: ProviderFactory | None = None,
    db_opener: DbOpener | None = None,
    system: str | None = None,
    helper_probe: "HelperProbe | None" = None,
    ollama_host: str | None = None,
    ollama_timeout: float = 2.0,
) -> dict[str, Any]:
    """Aggregate every preflight check into a single report dict.

    Args:
        settings: Configuration; defaults to :func:`get_settings`.
        probe_ollama: When ``True`` (default), probe the Ollama HTTP API. When
            ``False``, skip the network call and report ``reachable="unknown"``.
        probe_embedding: When ``True``, request a real embedding to confirm the
            dimension. Off by default (network-free).
        probe_encryption: When ``True`` (default), open/close the DB so the
            crypto layer can set the real encryption status. When ``False``,
            only the configured flag is reported without touching disk.
        http_get / provider_factory / db_opener / system / ollama_host /
            ollama_timeout: Injection points for testing with fakes.

    Returns:
        A JSON-serializable dict with a top-level ``ok`` summary and a nested
        result per check. No check raises; undeterminable values are
        ``"unknown"``.
    """
    settings = settings or get_settings()

    # Route-aware: classify the configured models and derive the local
    # models the active route actually needs, instead of hard-coding defaults.
    from openbird.llm.provider import classify_models

    # Resolve the host FIRST and classify against it, so an explicit (possibly
    # non-loopback) ``ollama_host`` override is reflected in cloud classification
    # — not just in the probe — preventing a remote host from looking local.
    resolved_host = ollama_host or _ollama_host(settings)
    remote_models = classify_models(settings, ollama_host=resolved_host)
    required_models = _ollama_required_models(settings)
    # Does the active route depend on local Ollama at all (either model ollama*)?
    uses_ollama = any(
        is_ollama_model(m) for m in (settings.llm_model, settings.embed_model)
    )

    if not uses_ollama:
        # Cloud-only / mlx-only route never touches local Ollama: skip the probe
        # (no point reporting localhost "down" or spending the network call for a
        # service the active route does not use). Reported as not-applicable.
        ollama = {
            "reachable": "n/a",
            "host": resolved_host,
            "models_present": [],
            "required_models": [],
            "models": {},
            "missing_models": [],
            "error": None,
        }
    elif probe_ollama:
        ollama = check_ollama(
            host=resolved_host,
            required_models=required_models,
            http_get=http_get,
            timeout=ollama_timeout,
        )
    else:
        ollama = {
            "reachable": "unknown",
            "host": resolved_host,
            "models_present": [],
            "required_models": list(required_models),
            "models": {m: "unknown" for m in required_models},
            "missing_models": "unknown",
            "error": None,
        }

    # Probe against the SAME resolved host the report shows. When an explicit
    # ollama_host override is given, carry it in the settings used to build the
    # probe provider so LiteLLM's api_base targets that host (not env/default) —
    # otherwise a probe could succeed on localhost while the checked host differs.
    probe_settings = settings
    if ollama_host is not None and provider_factory is None:
        import dataclasses

        probe_settings = dataclasses.replace(settings, ollama_host=resolved_host)

    embedding = check_embedding(
        probe_settings, provider_factory=provider_factory, probe=probe_embedding
    )
    # Probe the chat model under the SAME flag so a remote chat endpoint is
    # validated (an embedding probe does not exercise the completion endpoint).
    completion = check_completion(
        probe_settings, provider_factory=provider_factory, probe=probe_embedding
    )
    sqlite_vec_info = check_sqlite_vec()

    if probe_encryption:
        encryption = check_encryption(settings, db_opener=db_opener)
    else:
        # Without a live probe the settings flag is NOT trustworthy (a plain
        # sqlite3 with the flag flipped would lie). Report unknown/unverified.
        encryption = {
            "enabled": None,
            "status": "unknown",
            "backend": "unknown",
            "cipher_version": None,
            "wal_enabled": None,
            "verified": False,
            "error": None,
        }

    mac_helper_probe = helper_probe
    if mac_helper_probe is None and system is None and platform.system() == "Darwin":
        mac_helper_probe = _packaged_helper_probe()
    macos = check_macos_capabilities(system=system, helper_probe=mac_helper_probe)

    report: dict[str, Any] = {
        "ollama": ollama,
        "embedding": embedding,
        "completion": completion,
        "sqlite": sqlite_vec_info,
        "encryption": encryption,
        "privacy": {
            "allowlist": list(settings.allowlist),
            "blocklist": list(settings.blocklist),
            "ocr_enabled": bool(settings.ocr_enabled),
        },
        "macos": macos,
        # Provider backend readiness. Two unwired-MLX cases must NOT report READY:
        #   * the reserved ``mlx`` *backend* (factory raises NotImplementedError);
        #   * ``mlx/*`` *model strings* under the default litellm backend — litellm
        #     cannot serve them, so the first runtime call would fail.
        "backend": {
            "name": (settings.llm_backend or "").strip().lower(),
            "supported": (
                (settings.llm_backend or "").strip().lower() == "litellm"
                and not _has_mlx_model(settings)
            ),
        },
        # Cloud route status: which configured models are remote, whether
        # cloud is opted into, and whether captured memory would actually leave
        # this machine on the current config. "blocked" = remote model set but
        # no opt-in (the factory would refuse).
        "cloud": {
            "remote_models": remote_models,
            "active": bool(remote_models),
            "allow_cloud": bool(settings.allow_cloud),
            "blocked": bool(remote_models) and not settings.allow_cloud,
            "llm_model": settings.llm_model,
            "embed_model": settings.embed_model,
            "uses_local_ollama": uses_ollama,
        },
    }
    report["runtime_ok"] = _runtime_ok(report)
    report["release_gate_ok"] = _release_gate_ok(report)
    # Back-compat: ``ok`` is the runtime readiness summary (can OpenBird run?).
    report["ok"] = report["runtime_ok"]
    return report


def _runtime_ok(report: dict[str, Any]) -> bool:
    """Whether the *hard runtime* requirements are satisfied.

    Runtime-OK means the parts OpenBird cannot run without are present:
      * sqlite-vec + FTS5 usable, AND
      * EVERY active model role is usable, checked PER ROLE so any route
        (local-only, cloud-only, or mixed) validates each half:
          - local Ollama (if either role uses it): reachable with no missing
            required models;
          - a remote EMBED role: not blocked, and an embedding probe succeeded
            (``embedding.dim_ok``);
          - a remote CHAT role: not blocked, and a completion probe succeeded
            (``completion.ok``) — a successful embedding probe does NOT prove the
            chat endpoint/credentials work.
        For a remote role, "no probe run" means UNVERIFIED, which is NOT READY:
        sqlite/Ollama alone must never report READY when a missing/invalid cloud
        key would fail the first call. The configured provider BACKEND must also
        be wired (``backend.supported``) — the reserved ``mlx`` backend is never
        runtime-OK because the factory raises for it.

    It deliberately does NOT gate on encryption or macOS capture grants — the
    product runs without capture and runs plaintext-with-0600 if SQLCipher is
    absent. ``"unknown"`` Ollama (probe skipped) does not count as OK.
    """
    ollama = report["ollama"]
    sqlite_info = report["sqlite"]
    cloud = report.get("cloud", {})
    embedding = report.get("embedding", {})
    completion = report.get("completion", {})
    backend = report.get("backend", {})
    remote_models = cloud.get("remote_models", {}) or {}

    sqlite_ok = bool(sqlite_info.get("vec_available")) and bool(sqlite_info.get("fts5_available"))

    # The configured provider backend must be wired (only litellm today; mlx is
    # reserved and the factory raises). An unwired backend can never run.
    if not backend.get("supported", True):
        return False

    # A remote model configured without opt-in cannot run (factory refuses).
    if cloud.get("blocked"):
        return False

    # Local Ollama half of the route (if used) must be reachable with its models.
    if cloud.get("uses_local_ollama", True):
        if not (ollama.get("reachable") is True and not ollama.get("missing_models")):
            return False

    # Each remote role must be verified by ITS OWN probe (covers cloud-only,
    # cloud-embed-only, cloud-chat-only, and mixed routes).
    if "embed" in remote_models and embedding.get("dim_ok") is not True:
        return False
    if "llm" in remote_models and completion.get("ok") is not True:
        return False

    return bool(sqlite_ok)


def _release_gate_ok(report: dict[str, Any]) -> bool:
    """Whether the stricter release gate passes.

    The release gate requires everything :func:`_runtime_ok` requires PLUS:
      * verified at-rest encryption (live SQLCipher ``cipher_version``), and
      * on macOS, every TCC/audio gate reported ``passed`` by the signed helper.

    ``unknown`` (e.g. encryption not probed, or no signed helper) is NOT green:
    the gate only opens on positive proof. Off-mac, the TCC/audio gates are not
    applicable and do not block the release gate.
    """
    if not _runtime_ok(report):
        return False

    enc = report["encryption"]
    enc_ok = bool(enc.get("verified")) and bool(enc.get("enabled"))
    if not enc_ok:
        return False

    macos = report["macos"]
    if macos.get("is_macos"):
        if not macos.get("all_passed"):
            return False
    return True


__all__ = [
    "run_preflight",
    "check_ollama",
    "check_embedding",
    "check_completion",
    "check_sqlite_vec",
    "check_encryption",
    "check_macos_capabilities",
    "_packaged_helper_probe",
    "HelperProbe",
    "GRANT_PASSED",
    "GRANT_FAILED",
    "GRANT_UNKNOWN",
]
