"""LLM provider seam plus the current LiteLLM/Ollama-backed implementation.

Defaults to local Ollama (``ollama/llama3.2`` + ``ollama/nomic-embed-text``).
Embeddings are guarded to the configured dimension, and a stable ``cohort_key``
identifies the (provider, model, dim, normalized) tuple so the memory store can
refuse to mix incompatible embedding cohorts.

The public factory, :func:`create_llm_provider`, is the insertion point for
future backends. The current production backend is still LiteLLM, which preserves
the existing Ollama-by-default path and cloud opt-in behavior.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from openbird.config import Settings, get_settings
from openbird.llm.base import LLMProviderProtocol


class LiteLLMProvider:
    """Thin wrapper over LiteLLM for embeddings and chat completion."""

    backend_name = "litellm"

    def __init__(self, settings: Settings | None = None, *, normalized: bool = False) -> None:
        """Create a provider.

        Args:
            settings: Configuration; defaults to :func:`get_settings`.
            normalized: Whether embeddings are L2-normalized before storage.
                Recorded in :meth:`cohort_key` so cohorts stay consistent.
        """
        self.settings = settings or get_settings()
        self.embed_model = self.settings.embed_model
        self.llm_model = self.settings.llm_model
        self.embed_dim = self.settings.embed_dim
        self.normalized = normalized

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

        resp = litellm.embedding(model=self.embed_model, input=texts)
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
            resp = litellm.completion(model=self.llm_model, messages=messages)
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
            resp = litellm.completion(
                model=self.llm_model,
                messages=convo,
                response_format={"type": "json_object"},
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
) -> LLMProviderProtocol:
    """Build the configured model provider.

    ``litellm`` is the only production backend today and keeps the existing
    Ollama-by-default path unchanged. ``mlx`` is deliberately a reserved backend:
    the experiment under ``experiments/mlx-runtime/`` produced a "revisit after
    provider split" verdict, so this branch is the safe place to wire it after a
    fresh acceptance run.
    """
    resolved_settings = settings or get_settings()
    selected = (backend or resolved_settings.llm_backend).strip().lower()
    if selected == "litellm":
        return LLMProvider(resolved_settings, normalized=normalized)
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
]
