"""LLM provider seam plus the current LiteLLM/Ollama-backed implementation.

Defaults to local Ollama (``ollama/llama3.2`` + ``ollama/nomic-embed-text``).
Embeddings are guarded to the configured dimension, and a stable ``cohort_key``
identifies the (provider, model, dim, normalized) tuple so the memory store can
refuse to mix incompatible embedding cohorts [R3].

The public factory, :func:`create_llm_provider`, is the insertion point for
future backends. The current production backend is still LiteLLM, which preserves
the existing Ollama-by-default path and cloud opt-in behavior.
"""

from __future__ import annotations

import hashlib
import json
import threading
from typing import Any, Callable

from openbird.config import (
    Settings,
    get_settings,
    is_loopback_host,
    is_ollama_model,
    resolved_ollama_host,
)
from openbird.llm.base import LLMProviderProtocol


class LLMTimeoutError(TimeoutError):
    """Raised when an LLM call exceeds its wall-clock deadline [B4].

    The LiteLLM ``timeout`` kwarg is honored for most provider paths, but at
    least the Ollama *embedding* path can ignore it against a reachable-but-
    wedged server (TCP accepted, never responds). A hard wall-clock guard
    around every call guarantees the call cannot hang the process regardless of
    LiteLLM-internal behavior; the worker thread is abandoned (daemon) so a stuck
    socket leaks at most one thread instead of blocking ingest/chat forever.
    """


class CloudOptInRequired(RuntimeError):
    """Raised when a *remote* model is configured without explicit cloud opt-in.

    OpenBird is local-first [H3]: resolving a model that would send captured
    memory off this machine (a cloud model, or an ``ollama/*`` model pointed at a
    non-loopback host) requires the user to opt in via ``OPENBIRD_ALLOW_CLOUD=1``
    (or an interactive confirm at the CLI). The factory refuses otherwise so no
    code path can silently exfiltrate private content.
    """

    def __init__(self, remote_models: dict[str, str]) -> None:
        self.remote_models = dict(remote_models)
        names = ", ".join(f"{role}={model!r}" for role, model in remote_models.items())
        super().__init__(
            "Cloud opt-in required: the configured model(s) would send captured "
            f"memory off this machine ({names}). OpenBird is local-first and will "
            "not do this silently. To proceed, set OPENBIRD_ALLOW_CLOUD=1 (or "
            "confirm interactively). To stay local, use an ollama/* model on a "
            "loopback host (the default)."
        )


# Model-name prefixes that run locally regardless of host. ``ollama/`` is local
# ONLY when its resolved host is loopback (see classify_route); these prefixes
# are unconditionally on-device.
_LOCAL_MODEL_PREFIXES: tuple[str, ...] = ("mlx/", "mlx-community/", "mlx_community/")


def is_local_model(model: str, *, ollama_host: str | None = None) -> bool:
    """Return True if ``model`` runs on this machine (no data leaves the device).

    A LiteLLM Ollama model (``ollama/*`` or ``ollama_chat/*``) is local only when
    ``ollama_host`` is loopback (a remote Ollama endpoint exfiltrates chunks just
    like a cloud API). ``mlx*`` is always local. Everything else (gpt-*, claude-*,
    text-embedding-3-*, openai/*, anthropic/*, gemini/*, …) is remote.
    """
    name = (model or "").strip().lower()
    if is_ollama_model(name):
        host = ollama_host if ollama_host is not None else resolved_ollama_host()
        return is_loopback_host(host)
    return any(name.startswith(p) for p in _LOCAL_MODEL_PREFIXES)


def classify_models(settings: Settings, *, ollama_host: str | None = None) -> dict[str, str]:
    """Return ``{role: model}`` for each model that is REMOTE under this config.

    Roles are ``"llm"`` and ``"embed"``. An empty dict means the whole route is
    local (the local-first default). Used by the factory (to enforce opt-in) and
    by preflight / the CLI banner (to surface CLOUD ACTIVE).

    ``ollama_host`` overrides the resolved host for classification — preflight
    passes the SAME host it probes so an explicit (possibly non-loopback) host
    override cannot be classified as local while the report shows a remote host.
    """
    host = ollama_host if ollama_host is not None else resolved_ollama_host(settings)
    remote: dict[str, str] = {}
    if not is_local_model(settings.llm_model, ollama_host=host):
        remote["llm"] = settings.llm_model
    if not is_local_model(settings.embed_model, ollama_host=host):
        remote["embed"] = settings.embed_model
    return remote


def cloud_active(settings: Settings | None = None) -> bool:
    """True if any resolved model is remote (whether or not opt-in is set)."""
    return bool(classify_models(settings or get_settings()))


def cloud_banner(settings: Settings | None = None) -> str | None:
    """Return a one-line 'CLOUD ACTIVE' description, or None if fully local."""
    resolved = settings or get_settings()
    remote = classify_models(resolved)
    if not remote:
        return None
    parts = ", ".join(f"{role}={model}" for role, model in remote.items())
    return f"CLOUD ACTIVE — remote model(s): {parts} (captured memory leaves this machine)"


class LiteLLMProvider:
    """Thin wrapper over LiteLLM for embeddings and chat completion."""

    backend_name = "litellm"

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        normalized: bool = False,
        allow_cloud: bool | None = None,
    ) -> None:
        """Create a provider.

        Args:
            settings: Configuration; defaults to :func:`get_settings`.
            normalized: Whether embeddings are L2-normalized before storage.
                Recorded in :meth:`cohort_key` so cohorts stay consistent.
            allow_cloud: Opt-in override. When ``None`` (default), taken from
                ``settings.allow_cloud`` / ``OPENBIRD_ALLOW_CLOUD``.

        Raises:
            CloudOptInRequired: if any resolved model is *remote* (a cloud API,
                or an ``ollama/*`` model on a non-loopback host) and cloud is not
                opted into. The guard lives HERE — not only in
                :func:`create_llm_provider` — so the security gate cannot be
                bypassed by constructing the concrete provider directly [H3].
        """
        self.settings = settings or get_settings()
        cloud_ok = self.settings.allow_cloud if allow_cloud is None else allow_cloud
        if not cloud_ok:
            remote = classify_models(self.settings)
            if remote:
                raise CloudOptInRequired(remote)
        self.embed_model = self.settings.embed_model
        self.llm_model = self.settings.llm_model
        self.embed_dim = self.settings.embed_dim
        self.normalized = normalized
        # B4/H4: explicit per-call timeouts + bounded transport retries.
        self.llm_timeout = self.settings.llm_timeout
        self.embed_timeout = self.settings.embed_timeout
        self.num_retries = self.settings.llm_num_retries
        # M1: the Ollama base URL to thread into LiteLLM as api_base for ollama/*
        # models so the runtime call targets the same host preflight probes.
        self._ollama_host = resolved_ollama_host(self.settings)

    def _call_with_timeout(self, fn: Callable[[], Any], *, timeout: float) -> Any:
        """Run ``fn`` with a hard wall-clock deadline [B4].

        LiteLLM's own ``timeout`` is passed through too (so a well-behaved
        backend cancels promptly and the worker thread exits), but this guard is
        the backstop for paths that ignore it: after ``timeout`` (+ a small grace
        margin so LiteLLM's own timeout fires first when it works) we raise
        :class:`LLMTimeoutError` and return control to the caller. The worker
        runs on a daemon thread, so a wedged socket leaks one thread rather than
        blocking ingest/chat forever.
        """
        # Grace margin: prefer LiteLLM's own (cleaner) cancellation when it works;
        # the wall-clock guard only fires if that fails to return in time.
        deadline = float(timeout) + 5.0
        box: dict[str, Any] = {}

        def _runner() -> None:
            try:
                box["result"] = fn()
            except BaseException as exc:  # noqa: BLE001 (re-raised on the caller thread)
                box["error"] = exc

        # A DAEMON thread is essential: if the call wedges, the worker keeps the
        # stuck socket but never blocks interpreter shutdown (concurrent.futures'
        # ThreadPoolExecutor uses non-daemon threads + an atexit join, which would
        # hang process exit — the very failure mode B4 is about).
        worker = threading.Thread(target=_runner, name="openbird-llm", daemon=True)
        worker.start()
        worker.join(deadline)
        if worker.is_alive():
            raise LLMTimeoutError(
                f"LLM call exceeded {deadline:.0f}s wall-clock deadline "
                f"(configured timeout {timeout:.0f}s). The backend may be "
                "reachable but wedged; aborting so the process does not hang."
            )
        if "error" in box:
            raise box["error"]
        return box.get("result")

    def _model_kwargs(self, model: str, *, timeout: float) -> dict[str, Any]:
        """Common LiteLLM kwargs: timeout, bounded retries, and (M1) api_base.

        ``api_base`` is only set for ``ollama/*`` models — a cloud model must NOT
        be pointed at the local Ollama host. ``num_retries`` lets LiteLLM retry
        connection errors / 5xx / rate-limit responses with backoff; total
        transport attempts per call = ``num_retries + 1``.
        """
        kwargs: dict[str, Any] = {"timeout": timeout, "num_retries": self.num_retries}
        if is_ollama_model(model):
            kwargs["api_base"] = self._ollama_host
        return kwargs

    # -- embeddings -----------------------------------------------------------

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed ``texts`` and assert each vector matches ``embed_dim``.

        Returns one vector per input text, in order. Raises ``ValueError`` if any
        returned vector's dimension differs from ``settings.embed_dim`` (a hard
        guard against silently mixing embedding models/cohorts).
        """
        if not texts:
            return []

        import litellm

        kwargs = self._model_kwargs(self.embed_model, timeout=self.embed_timeout)
        resp = self._call_with_timeout(
            lambda: litellm.embedding(model=self.embed_model, input=texts, **kwargs),
            timeout=self.embed_timeout,
        )
        vectors = [self._extract_embedding(item) for item in resp["data"]]

        # Count guard: a short/long response would otherwise silently zip with the
        # caller's chunk rowids, leaving some chunks unembedded (or misaligned).
        if len(vectors) != len(texts):
            raise ValueError(
                f"Embedding count mismatch: got {len(vectors)} vectors for "
                f"{len(texts)} inputs from model {self.embed_model!r}"
            )

        for vec in vectors:
            if len(vec) != self.embed_dim:
                raise ValueError(
                    f"Embedding dimension mismatch: got {len(vec)}, "
                    f"expected {self.embed_dim} for model {self.embed_model!r}"
                )
        return vectors

    @staticmethod
    def _extract_embedding(item: Any) -> list[float]:
        """Pull the float vector out of a LiteLLM embedding data item."""
        if isinstance(item, dict):
            return list(item["embedding"])
        return list(item.embedding)

    # -- completion -----------------------------------------------------------

    def complete(
        self,
        messages: list[dict],
        *,
        json_schema: dict | None = None,
    ) -> str | dict:
        """Generate a completion for ``messages``.

        If ``json_schema`` is provided, returns a parsed ``dict`` produced by
        best-effort structured generation (validate + retry; NOT constrained
        decoding). On repeated failure the last raw text is returned so callers
        can decide how to handle it. Otherwise returns the raw string content.
        """
        import litellm

        if json_schema is None:
            kwargs = self._model_kwargs(self.llm_model, timeout=self.llm_timeout)
            resp = self._call_with_timeout(
                lambda: litellm.completion(
                    model=self.llm_model, messages=messages, **kwargs
                ),
                timeout=self.llm_timeout,
            )
            return self._content(resp)

        attempts = 3
        schema_msg = {
            "role": "system",
            "content": (
                "Respond with a single valid JSON object that conforms to this "
                f"JSON schema. Output JSON only, no prose:\n{json.dumps(json_schema)}"
            ),
        }
        convo = [schema_msg, *messages]
        last_text = ""
        for _ in range(attempts):
            kwargs = self._model_kwargs(self.llm_model, timeout=self.llm_timeout)
            resp = self._call_with_timeout(
                lambda convo=convo, kwargs=kwargs: litellm.completion(
                    model=self.llm_model,
                    messages=convo,
                    response_format={"type": "json_object"},
                    **kwargs,
                ),
                timeout=self.llm_timeout,
            )
            last_text = self._content(resp)
            parsed = self._try_parse_json(last_text)
            if parsed is not None and self._validate(parsed, json_schema):
                return parsed
            # Retry WITHOUT echoing the invalid output back as an assistant turn:
            # that text may contain prompt-injection content the model copied out
            # of the fenced untrusted context, and replaying it as prior assistant
            # output would launder it past the fence. A metadata-only correction
            # keeps the retry grounded in the original (fenced) messages.
            convo = [
                schema_msg,
                *messages,
                {
                    "role": "user",
                    "content": (
                        "Your previous response was not valid JSON for the schema. "
                        "Respond again with a single valid JSON object only, no prose."
                    ),
                },
            ]
        # Best-effort: return whatever parsed last, else the raw text.
        parsed = self._try_parse_json(last_text)
        return parsed if parsed is not None else last_text

    @staticmethod
    def _content(resp: Any) -> str:
        """Extract assistant text content from a LiteLLM completion response."""
        choice = resp["choices"][0] if isinstance(resp, dict) else resp.choices[0]
        msg = choice["message"] if isinstance(choice, dict) else choice.message
        content = msg["content"] if isinstance(msg, dict) else msg.content
        return content or ""

    @staticmethod
    def _try_parse_json(text: str) -> dict | None:
        """Parse ``text`` as a JSON object, tolerating surrounding prose."""
        text = text.strip()
        try:
            obj = json.loads(text)
            return obj if isinstance(obj, dict) else None
        except (json.JSONDecodeError, ValueError):
            pass
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                obj = json.loads(text[start : end + 1])
                return obj if isinstance(obj, dict) else None
            except (json.JSONDecodeError, ValueError):
                return None
        return None

    @staticmethod
    def _validate(obj: dict, schema: dict) -> bool:
        """Best-effort schema validation (jsonschema if present, else top keys)."""
        try:
            import jsonschema

            jsonschema.validate(obj, schema)
            return True
        except ImportError:
            required = schema.get("required", [])
            return all(key in obj for key in required)
        except Exception:
            return False

    # -- cohort identity ------------------------------------------------------

    def cohort_key(self) -> str:
        """Return a stable id for this embedding cohort.

        The key hashes (embedding provider/model, dimension, normalization) so
        the memory store can refuse to search across incompatible cohorts.
        """
        provider = self.embed_model.split("/", 1)[0] if "/" in self.embed_model else "unknown"
        raw = f"{provider}|{self.embed_model}|{self.embed_dim}|{int(self.normalized)}"
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        return f"{provider}:{self.embed_model}:{self.embed_dim}:{digest}"


class LLMProvider(LiteLLMProvider):
    """Backward-compatible name for the default concrete provider.

    New code should prefer :func:`create_llm_provider` so backend selection stays
    centralized, but keeping this constructor preserves existing imports and
    runtime behavior.
    """


def create_llm_provider(
    settings: Settings | None = None,
    *,
    backend: str | None = None,
    normalized: bool = False,
    allow_cloud: bool | None = None,
) -> LLMProviderProtocol:
    """Build the configured model provider.

    ``litellm`` is the only production backend today and keeps the existing
    Ollama-by-default path unchanged. ``mlx`` is deliberately a reserved backend:
    the experiment under ``experiments/mlx-runtime/`` produced a "revisit after
    provider split" verdict, so this branch is the safe place to wire it after a
    fresh acceptance run.

    Cloud opt-in [H3]: if any resolved model is *remote* (a cloud API, or an
    ``ollama/*`` model on a non-loopback host) construction raises
    :class:`CloudOptInRequired` unless cloud is opted into. The guard is enforced
    in :meth:`LiteLLMProvider.__init__` (so direct construction cannot bypass it);
    here we just forward ``allow_cloud`` — taken from the explicit argument when
    given (the CLI passes ``True`` after an interactive confirm), otherwise from
    ``settings.allow_cloud`` / ``OPENBIRD_ALLOW_CLOUD``.
    """
    resolved_settings = settings or get_settings()
    selected = (backend or resolved_settings.llm_backend).strip().lower()
    if selected == "litellm":
        return LLMProvider(resolved_settings, normalized=normalized, allow_cloud=allow_cloud)
    if selected == "mlx":
        raise NotImplementedError(
            "The MLX backend is not wired into OpenBird runtime yet. "
            "Re-run experiments/mlx-runtime/ and promote it here only after "
            "its JSON, citation, latency, and setup gates pass."
        )
    raise ValueError(
        f"Unsupported LLM backend {selected!r}; expected 'litellm' "
        "or reserved backend 'mlx'."
    )


__all__ = [
    "LLMProvider",
    "LLMProviderProtocol",
    "LiteLLMProvider",
    "create_llm_provider",
    "CloudOptInRequired",
    "LLMTimeoutError",
    "is_local_model",
    "classify_models",
    "cloud_active",
    "cloud_banner",
]
