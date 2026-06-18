"""Unit tests for LLMProvider with litellm mocked (no Ollama/network)."""

from __future__ import annotations

import pytest

from openbird.config import Settings, get_settings, reset_settings_cache
from openbird.llm.base import LLMProviderProtocol
from openbird.llm.provider import LLMProvider, LiteLLMProvider, create_llm_provider


class _FakeLiteLLM:
    """Minimal litellm stand-in capturing the embedding dim and JSON behavior."""

    def __init__(self, dim: int = 768, completion_text: str = ""):
        self.dim = dim
        self.completion_text = completion_text

    def embedding(self, *, model, input):
        return {"data": [{"embedding": [0.0] * self.dim} for _ in input]}

    def completion(self, *, model, messages, **kwargs):
        return {"choices": [{"message": {"content": self.completion_text}}]}


def _provider(dim: int = 768) -> LLMProvider:
    return LLMProvider(Settings(embed_dim=dim))


def test_embed_returns_vectors_of_expected_dim(monkeypatch):
    fake = _FakeLiteLLM(dim=768)
    monkeypatch.setitem(__import__("sys").modules, "litellm", fake)
    out = _provider(768).embed(["hello", "world"])
    assert len(out) == 2
    assert all(len(v) == 768 for v in out)


def test_embed_dim_guard_raises_on_mismatch(monkeypatch):
    fake = _FakeLiteLLM(dim=512)  # provider expects 768
    monkeypatch.setitem(__import__("sys").modules, "litellm", fake)
    with pytest.raises(ValueError, match="dimension mismatch"):
        _provider(768).embed(["oops"])


def test_embed_empty_input_short_circuits():
    # No litellm needed; empty input returns [] without calling the backend.
    assert _provider().embed([]) == []


class _ShortLiteLLM(_FakeLiteLLM):
    """Returns one fewer embedding than requested (provider under-responds)."""

    def embedding(self, *, model, input):
        items = list(input)[1:]  # drop one -> count mismatch
        return {"data": [{"embedding": [0.0] * self.dim} for _ in items]}


def test_embed_count_guard_raises_on_short_response(monkeypatch):
    # Regression: a short embedding response must NOT silently zip with the
    # caller's chunk rowids (which would leave some chunks unembedded).
    fake = _ShortLiteLLM(dim=768)
    monkeypatch.setitem(__import__("sys").modules, "litellm", fake)
    with pytest.raises(ValueError, match="count mismatch"):
        _provider(768).embed(["a", "b", "c"])


def test_complete_plain_text(monkeypatch):
    fake = _FakeLiteLLM(completion_text="just some prose")
    monkeypatch.setitem(__import__("sys").modules, "litellm", fake)
    out = _provider().complete([{"role": "user", "content": "hi"}])
    assert out == "just some prose"


def test_complete_json_schema_parses_object(monkeypatch):
    fake = _FakeLiteLLM(completion_text='{"answer": 42}')
    monkeypatch.setitem(__import__("sys").modules, "litellm", fake)
    schema = {"type": "object", "required": ["answer"]}
    out = _provider().complete([{"role": "user", "content": "q"}], json_schema=schema)
    assert out == {"answer": 42}


def test_cohort_key_stable_and_dim_sensitive():
    a = _provider(768).cohort_key()
    b = _provider(768).cohort_key()
    c = _provider(512).cohort_key()
    assert a == b
    assert a != c
    assert "nomic-embed-text" in a


def test_default_factory_preserves_litellm_provider():
    provider = create_llm_provider(Settings(embed_dim=768))
    assert isinstance(provider, LiteLLMProvider)
    assert isinstance(provider, LLMProviderProtocol)
    assert provider.llm_model == "ollama/llama3.2"
    assert provider.embed_model == "ollama/nomic-embed-text"


def test_factory_propagates_normalized_flag_to_cohort_key():
    provider = create_llm_provider(Settings(embed_dim=768), normalized=True)
    assert isinstance(provider, LiteLLMProvider)
    assert provider.normalized is True
    default_provider = create_llm_provider(Settings(embed_dim=768))
    assert provider.cohort_key() != default_provider.cohort_key()


def test_settings_reads_llm_backend_from_env(monkeypatch, tmp_path):
    reset_settings_cache()
    monkeypatch.setenv("OPENBIRD_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OPENBIRD_LLM_BACKEND", "mlx")
    try:
        assert get_settings().llm_backend == "mlx"
        with pytest.raises(NotImplementedError, match="experiments/mlx-runtime"):
            create_llm_provider()
    finally:
        reset_settings_cache()


def test_factory_backend_argument_overrides_settings():
    settings = Settings(embed_dim=768, llm_backend="mlx")
    provider = create_llm_provider(settings, backend="litellm")
    assert isinstance(provider, LiteLLMProvider)


def test_factory_reserves_mlx_backend_for_experiment_promotion():
    settings = Settings(embed_dim=768, llm_backend="mlx")
    with pytest.raises(NotImplementedError, match="experiments/mlx-runtime"):
        create_llm_provider(settings)


def test_factory_rejects_unknown_backend():
    settings = Settings(embed_dim=768, llm_backend="mystery")
    with pytest.raises(ValueError, match="Unsupported LLM backend"):
        create_llm_provider(settings)
