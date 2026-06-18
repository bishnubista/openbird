"""Unit tests for Settings env coercion + Ollama host resolution."""

from __future__ import annotations

import pytest

from openbird.config import (
    DEFAULT_OLLAMA_HOST,
    Settings,
    get_settings,
    is_loopback_host,
    is_ollama_model,
    ollama_bare_model,
    reset_settings_cache,
    resolved_ollama_host,
)


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    # Keep filesystem side effects in a temp dir and start from a clean cache.
    monkeypatch.setenv("OPENBIRD_DATA_DIR", str(tmp_path))
    reset_settings_cache()
    yield
    reset_settings_cache()


# --------------------------------------------------------------------------- #
# env coercion                                                                #
# --------------------------------------------------------------------------- #


def test_allow_cloud_coerced_from_env(monkeypatch):
    monkeypatch.setenv("OPENBIRD_ALLOW_CLOUD", "1")
    assert get_settings().allow_cloud is True
    reset_settings_cache()
    monkeypatch.setenv("OPENBIRD_ALLOW_CLOUD", "no")
    assert get_settings().allow_cloud is False


def test_timeouts_coerced_to_float(monkeypatch):
    monkeypatch.setenv("OPENBIRD_LLM_TIMEOUT", "45.5")
    monkeypatch.setenv("OPENBIRD_EMBED_TIMEOUT", "10")
    s = get_settings()
    assert s.llm_timeout == 45.5
    assert isinstance(s.llm_timeout, float)
    assert s.embed_timeout == 10.0


def test_num_retries_coerced_to_int(monkeypatch):
    monkeypatch.setenv("OPENBIRD_LLM_NUM_RETRIES", "5")
    assert get_settings().llm_num_retries == 5


def test_invalid_float_timeout_raises(monkeypatch):
    monkeypatch.setenv("OPENBIRD_LLM_TIMEOUT", "not-a-number")
    with pytest.raises(ValueError):
        get_settings()


def test_defaults_are_sane():
    s = Settings()
    assert s.llm_timeout == 60.0
    assert s.embed_timeout == 30.0
    assert s.llm_num_retries == 2
    assert s.allow_cloud is False
    assert s.ollama_host is None


# --------------------------------------------------------------------------- #
# host resolution                                                         #
# --------------------------------------------------------------------------- #


def test_resolved_host_defaults_to_localhost(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.delenv("OPENBIRD_OLLAMA_HOST", raising=False)
    assert resolved_ollama_host(Settings()) == DEFAULT_OLLAMA_HOST


def test_ollama_host_env_takes_precedence(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://a:1")
    monkeypatch.setenv("OPENBIRD_OLLAMA_HOST", "http://b:2")
    assert resolved_ollama_host(Settings(ollama_host="http://c:3")) == "http://a:1"


def test_settings_ollama_host_beats_openbird_env(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.setenv("OPENBIRD_OLLAMA_HOST", "http://b:2")
    assert resolved_ollama_host(Settings(ollama_host="http://c:3")) == "http://c:3"


def test_openbird_env_used_when_no_settings_or_ollama_host(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.setenv("OPENBIRD_OLLAMA_HOST", "http://b:2")
    assert resolved_ollama_host(Settings()) == "http://b:2"


def test_empty_ollama_host_setting_falls_through(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.delenv("OPENBIRD_OLLAMA_HOST", raising=False)
    # Empty string is not a usable host; resolver falls back to default.
    assert resolved_ollama_host(Settings(ollama_host="")) == DEFAULT_OLLAMA_HOST


def test_bare_host_port_is_normalized_to_http_url(monkeypatch):
    # OLLAMA_HOST commonly is a bare host:port; api_base / urljoin need a scheme.
    monkeypatch.setenv("OLLAMA_HOST", "localhost:11434")
    assert resolved_ollama_host(Settings()) == "http://localhost:11434"
    monkeypatch.setenv("OLLAMA_HOST", "10.0.0.5:11434")
    assert resolved_ollama_host(Settings()) == "http://10.0.0.5:11434"


def test_scheme_host_is_left_unchanged(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "https://secure-ollama:443")
    assert resolved_ollama_host(Settings()) == "https://secure-ollama:443"


# --------------------------------------------------------------------------- #
# loopback classification                                                      #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "model,is_ollama,bare",
    [
        ("ollama/llama3.2", True, "llama3.2"),
        ("ollama/llama3.2:3b", True, "llama3.2:3b"),
        ("ollama_chat/llama3.2", True, "llama3.2"),
        ("OLLAMA_CHAT/Mistral", True, "Mistral"),  # case-insensitive prefix
        ("gpt-4o-mini", False, None),
        ("mlx/Qwen", False, None),
        ("", False, None),
    ],
)
def test_ollama_predicate_and_bare(model, is_ollama, bare):
    assert is_ollama_model(model) is is_ollama
    assert ollama_bare_model(model) == bare


@pytest.mark.parametrize(
    "host,loopback",
    [
        ("http://localhost:11434", True),
        ("http://127.0.0.1:11434", True),
        ("http://[::1]:11434", True),
        ("localhost:11434", True),
        ("", True),
        ("http://10.0.0.5:11434", False),
        ("http://my-server.lan:11434", False),
        ("http://192.168.1.10", False),
    ],
)
def test_is_loopback_host(host, loopback):
    assert is_loopback_host(host) is loopback
