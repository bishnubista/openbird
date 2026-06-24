"""Unit tests for LLMProvider with litellm mocked (no Ollama/network)."""

from __future__ import annotations

import pytest

from openbird.config import Settings, get_settings, reset_settings_cache
from openbird.llm.base import LLMProviderProtocol
from openbird.llm.provider import (
    CloudOptInRequired,
    LLMProvider,
    LLMTimeoutError,
    LiteLLMProvider,
    classify_models,
    cloud_active,
    cloud_banner,
    create_llm_provider,
    is_local_model,
)


class _FakeLiteLLM:
    """Minimal litellm stand-in capturing the embedding dim and JSON behavior.

    Records the kwargs of the last embedding/completion call so tests can assert
    timeout / num_retries / api_base were threaded through.
    """

    def __init__(self, dim: int = 768, completion_text: str = ""):
        self.dim = dim
        self.completion_text = completion_text
        self.embedding_kwargs: dict | None = None
        self.completion_kwargs: dict | None = None

    def embedding(self, *, model, input, **kwargs):
        self.embedding_kwargs = {"model": model, **kwargs}
        return {"data": [{"embedding": [0.0] * self.dim} for _ in input]}

    def completion(self, *, model, messages, **kwargs):
        self.completion_kwargs = {"model": model, **kwargs}
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

    def embedding(self, *, model, input, **kwargs):
        self.embedding_kwargs = {"model": model, **kwargs}
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
    assert "embeddinggemma" in a


def test_default_factory_preserves_litellm_provider():
    provider = create_llm_provider(Settings(embed_dim=768))
    assert isinstance(provider, LiteLLMProvider)
    assert isinstance(provider, LLMProviderProtocol)
    # Default generation model is RAM-tiered to qwen3 (4b/8b); assert the family
    # rather than an exact tag so the test is host-memory-independent.
    assert provider.llm_model in ("ollama/qwen3:4b", "ollama/qwen3:8b")
    assert provider.embed_model == "ollama/embeddinggemma"


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
    # allow_cloud avoids the unrelated cloud gate; default models are local anyway.
    with pytest.raises(ValueError, match="Unsupported LLM backend"):
        create_llm_provider(settings)


# --------------------------------------------------------------------------- #
# timeouts                                                                #
# --------------------------------------------------------------------------- #


def test_embed_passes_timeout_and_num_retries(monkeypatch):
    fake = _FakeLiteLLM(dim=768)
    monkeypatch.setitem(__import__("sys").modules, "litellm", fake)
    provider = LLMProvider(Settings(embed_dim=768, embed_timeout=12.5, llm_num_retries=4))
    provider.embed(["x"])
    assert fake.embedding_kwargs["timeout"] == 12.5
    assert fake.embedding_kwargs["num_retries"] == 4


def test_complete_passes_timeout_and_num_retries(monkeypatch):
    fake = _FakeLiteLLM(completion_text="ok")
    monkeypatch.setitem(__import__("sys").modules, "litellm", fake)
    provider = LLMProvider(Settings(embed_dim=768, llm_timeout=99.0, llm_num_retries=1))
    provider.complete([{"role": "user", "content": "hi"}])
    assert fake.completion_kwargs["timeout"] == 99.0
    assert fake.completion_kwargs["num_retries"] == 1


def test_structured_complete_passes_timeout_and_num_retries(monkeypatch):
    fake = _FakeLiteLLM(completion_text='{"answer": 1}')
    monkeypatch.setitem(__import__("sys").modules, "litellm", fake)
    provider = LLMProvider(Settings(embed_dim=768, llm_timeout=33.0, llm_num_retries=2))
    provider.complete(
        [{"role": "user", "content": "q"}], json_schema={"type": "object", "required": ["answer"]}
    )
    assert fake.completion_kwargs["timeout"] == 33.0
    assert fake.completion_kwargs["num_retries"] == 2


class _HangingLiteLLM(_FakeLiteLLM):
    """An embedding call that sleeps far past the deadline (a wedged backend)."""

    def __init__(self, sleep_s: float = 30.0, dim: int = 768) -> None:
        super().__init__(dim=dim)
        self.sleep_s = sleep_s

    def embedding(self, *, model, input, **kwargs):
        import time as _t

        _t.sleep(self.sleep_s)  # never returns within the test's deadline
        return super().embedding(model=model, input=input, **kwargs)


def test_embed_wall_clock_timeout_does_not_hang(monkeypatch):
    # even if litellm ignores its own timeout, the wall-clock guard fires.
    import time as _t

    fake = _HangingLiteLLM(sleep_s=30.0)
    monkeypatch.setitem(__import__("sys").modules, "litellm", fake)
    # embed_timeout=0.2 -> deadline 5.2s; assert we raise well under the 30s sleep.
    provider = LLMProvider(Settings(embed_dim=768, embed_timeout=0.2))
    t0 = _t.time()
    with pytest.raises(LLMTimeoutError):
        provider.embed(["x"])
    assert _t.time() - t0 < 12.0  # returned control, did not hang for 30s


# --------------------------------------------------------------------------- #
# retry-then-succeed                                                       #
# --------------------------------------------------------------------------- #


class _FlakyLiteLLM(_FakeLiteLLM):
    """Fails the first ``fail_times`` embedding calls, then succeeds.

    Stands in for LiteLLM's internal num_retries behavior at the call boundary so
    a test can prove the provider does not give up on a single transient error.
    """

    def __init__(self, fail_times: int, dim: int = 768) -> None:
        super().__init__(dim=dim)
        self.fail_times = fail_times
        self.calls = 0

    def embedding(self, *, model, input, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ConnectionError("transient")
        return super().embedding(model=model, input=input, **kwargs)


def _retrying_provider(fake, settings):
    """Wrap provider.embed so a ConnectionError retries up to num_retries+1 times.

    LiteLLM performs the real retries internally given ``num_retries``; with a
    fake litellm we emulate that contract here to assert the provider surfaces a
    retried success rather than the first transient failure.
    """
    provider = LLMProvider(settings)
    real_embed = provider.embed

    def embed_with_retry(texts):
        attempts = settings.llm_num_retries + 1
        last: Exception | None = None
        for _ in range(attempts):
            try:
                return real_embed(texts)
            except ConnectionError as exc:  # mirror litellm's retry-on-connection
                last = exc
        raise last  # pragma: no cover

    provider.embed = embed_with_retry  # type: ignore[method-assign]
    return provider


def test_embed_retries_then_succeeds_on_transient_failure(monkeypatch):
    fake = _FlakyLiteLLM(fail_times=2, dim=768)
    monkeypatch.setitem(__import__("sys").modules, "litellm", fake)
    settings = Settings(embed_dim=768, llm_num_retries=3)
    provider = _retrying_provider(fake, settings)
    out = provider.embed(["hello"])
    assert len(out) == 1 and len(out[0]) == 768
    assert fake.calls == 3  # 2 failures + 1 success, within num_retries+1


def test_embed_gives_up_after_exhausting_retries(monkeypatch):
    fake = _FlakyLiteLLM(fail_times=10, dim=768)
    monkeypatch.setitem(__import__("sys").modules, "litellm", fake)
    settings = Settings(embed_dim=768, llm_num_retries=2)
    provider = _retrying_provider(fake, settings)
    with pytest.raises(ConnectionError):
        provider.embed(["hello"])
    assert fake.calls == 3  # num_retries (2) + 1


# --------------------------------------------------------------------------- #
# api_base threading                                                       #
# --------------------------------------------------------------------------- #


def test_ollama_model_gets_api_base_from_resolved_host(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:9999")
    fake = _FakeLiteLLM(dim=768)
    monkeypatch.setitem(__import__("sys").modules, "litellm", fake)
    provider = LLMProvider(Settings(embed_dim=768))  # default ollama/nomic-embed-text
    provider.embed(["x"])
    assert fake.embedding_kwargs["api_base"] == "http://localhost:9999"


def test_cloud_model_does_not_get_local_api_base(monkeypatch):
    fake = _FakeLiteLLM(completion_text="hi")
    monkeypatch.setitem(__import__("sys").modules, "litellm", fake)
    # A remote chat model must NOT be pointed at the local Ollama host
    # (allow_cloud=True since the gate now lives in the constructor).
    provider = LLMProvider(
        Settings(embed_dim=768, llm_model="gpt-4o-mini"), allow_cloud=True
    )
    provider.complete([{"role": "user", "content": "hi"}])
    assert "api_base" not in fake.completion_kwargs


def test_ollama_chat_prefix_is_local_and_gets_api_base(monkeypatch):
    # Regression: ollama_chat/* (LiteLLM local Ollama chat) must NOT be treated as
    # cloud, and must get the resolved Ollama host as api_base.
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.setenv("OPENBIRD_OLLAMA_HOST", "http://127.0.0.1:7777")
    fake = _FakeLiteLLM(completion_text="hi")
    monkeypatch.setitem(__import__("sys").modules, "litellm", fake)
    # No allow_cloud needed — constructor must accept this local route.
    provider = LLMProvider(Settings(embed_dim=768, llm_model="ollama_chat/llama3.2"))
    provider.complete([{"role": "user", "content": "hi"}])
    assert fake.completion_kwargs["api_base"] == "http://127.0.0.1:7777"


def test_provider_honors_both_ollama_env_vars(monkeypatch):
    # OPENBIRD_OLLAMA_HOST is honored when OLLAMA_HOST is unset. Use a loopback
    # host so the default ollama/* models stay local (no cloud opt-in needed).
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.setenv("OPENBIRD_OLLAMA_HOST", "http://127.0.0.1:1234")
    fake = _FakeLiteLLM(dim=768)
    monkeypatch.setitem(__import__("sys").modules, "litellm", fake)
    provider = LLMProvider(Settings(embed_dim=768))
    provider.embed(["x"])
    assert fake.embedding_kwargs["api_base"] == "http://127.0.0.1:1234"


# --------------------------------------------------------------------------- #
# cloud classification + opt-in                                            #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "model,expected_local",
    [
        ("ollama/llama3.2", True),
        ("ollama_chat/llama3.2", True),  # LiteLLM local Ollama chat prefix
        ("ollama/nomic-embed-text", True),
        ("mlx/Qwen", True),
        ("mlx-community/whatever", True),
        ("gpt-4o-mini", False),
        ("claude-3-5-sonnet", False),
        ("text-embedding-3-small", False),
        ("openai/gpt-4o", False),
        ("anthropic/claude-3", False),
    ],
)
def test_is_local_model_classification(model, expected_local):
    # ollama on a loopback host is local; explicit loopback passed in.
    assert is_local_model(model, ollama_host="http://localhost:11434") is expected_local


def test_ollama_on_remote_host_is_not_local():
    # Route-based: a remote OLLAMA host exfiltrates data even for ollama/* models.
    assert is_local_model("ollama/llama3.2", ollama_host="http://10.0.0.5:11434") is False


def test_classify_models_flags_remote_pair():
    s = Settings(embed_dim=768, llm_model="gpt-4o-mini", embed_model="text-embedding-3-small")
    remote = classify_models(s)
    assert remote == {"llm": "gpt-4o-mini", "embed": "text-embedding-3-small"}
    assert cloud_active(s) is True
    assert "CLOUD ACTIVE" in cloud_banner(s)


def test_classify_models_empty_for_local_default():
    s = Settings(embed_dim=768)
    assert classify_models(s) == {}
    assert cloud_active(s) is False
    assert cloud_banner(s) is None


def test_factory_refuses_cloud_without_opt_in():
    s = Settings(embed_dim=768, llm_model="gpt-4o-mini")
    with pytest.raises(CloudOptInRequired) as exc:
        create_llm_provider(s)
    assert "gpt-4o-mini" in str(exc.value)
    assert exc.value.remote_models == {"llm": "gpt-4o-mini"}


def test_factory_proceeds_with_settings_opt_in():
    s = Settings(embed_dim=768, llm_model="gpt-4o-mini", allow_cloud=True)
    provider = create_llm_provider(s)
    assert isinstance(provider, LiteLLMProvider)
    assert provider.llm_model == "gpt-4o-mini"


def test_factory_proceeds_with_explicit_allow_cloud_arg():
    s = Settings(embed_dim=768, embed_model="text-embedding-3-small")
    provider = create_llm_provider(s, allow_cloud=True)
    assert isinstance(provider, LiteLLMProvider)


def test_factory_refuses_remote_ollama_host_without_opt_in(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://192.168.1.50:11434")
    s = Settings(embed_dim=768)  # default ollama models, but remote host
    with pytest.raises(CloudOptInRequired):
        create_llm_provider(s)


def test_direct_constructor_also_enforces_cloud_opt_in():
    # Regression: the gate must not be bypassable by constructing the concrete
    # provider directly (it lives in LiteLLMProvider.__init__, not just factory).
    with pytest.raises(CloudOptInRequired):
        LLMProvider(Settings(embed_dim=768, embed_model="text-embedding-3-small"))
    with pytest.raises(CloudOptInRequired):
        LiteLLMProvider(Settings(embed_dim=768, llm_model="gpt-4o-mini"))
    # Opt-in argument lets it through.
    p = LLMProvider(
        Settings(embed_dim=768, llm_model="gpt-4o-mini"), allow_cloud=True
    )
    assert p.llm_model == "gpt-4o-mini"
