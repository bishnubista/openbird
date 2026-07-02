"""Unit tests for Settings env coercion + Ollama host resolution."""

from __future__ import annotations

import plistlib
import subprocess

import pytest

from openbird.config import (
    DEFAULT_OLLAMA_HOST,
    Settings,
    _read_gui_allowlist,
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


def test_deep_brain_settings_coerced_from_env(monkeypatch):
    monkeypatch.setenv("OPENBIRD_DEEP_BRAIN_ENABLED", "1")
    monkeypatch.setenv("OPENBIRD_DEEP_BRAIN_EXCLUDED_APPS", "com.a, glob:com.b.*")
    monkeypatch.setenv("OPENBIRD_DEEP_BRAIN_EXCLUDED_SOURCES", "capture,meeting")
    monkeypatch.setenv("OPENBIRD_DEEP_BRAIN_EXCLUDED_OBSERVATION_IDS", "obs-1, obs-2")

    s = get_settings()
    assert s.deep_brain_enabled is True
    assert s.deep_brain_excluded_apps == ["com.a", "glob:com.b.*"]
    assert s.deep_brain_excluded_sources == ["capture", "meeting"]
    assert s.deep_brain_excluded_observation_ids == ["obs-1", "obs-2"]


def test_capture_urls_coerced_to_bool_from_env(monkeypatch):
    # Default off (opt-in); truthy/falsey strings coerce to a real bool, never a
    # string (a stray "0" must not read as truthy).
    assert get_settings().capture_urls is False
    reset_settings_cache()
    monkeypatch.setenv("OPENBIRD_CAPTURE_URLS", "1")
    assert get_settings().capture_urls is True
    reset_settings_cache()
    monkeypatch.setenv("OPENBIRD_CAPTURE_URLS", "0")
    assert get_settings().capture_urls is False


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


def test_empty_db_path_env_falls_back_to_default(tmp_path, monkeypatch):
    # An empty OPENBIRD_DB_PATH must be treated as unset (not a degenerate empty
    # path), so it resolves to <data dir>/openbird.db. This keeps the signed
    # Swift app's DB-path resolution (which skips empty values) in lockstep so
    # both inspect the SAME file before the app mints/injects an encryption key.
    monkeypatch.setenv("OPENBIRD_DB_PATH", "")
    assert get_settings().db_path == str(tmp_path / "openbird.db")


def test_non_empty_db_path_env_is_honored(monkeypatch):
    monkeypatch.setenv("OPENBIRD_DB_PATH", "/tmp/custom-openbird.db")
    assert get_settings().db_path == "/tmp/custom-openbird.db"


def test_empty_blocklist_env_clears_to_empty(monkeypatch):
    # Empty is MEANINGFUL for list fields: OPENBIRD_BLOCKLIST="" must explicitly
    # clear the default blocklist to [] (NOT be skipped as "no override").
    monkeypatch.setenv("OPENBIRD_BLOCKLIST", "")
    assert get_settings().blocklist == []


def test_blocklist_defaults_when_env_unset(monkeypatch):
    # With no env var at all, the non-empty default blocklist stands.
    monkeypatch.delenv("OPENBIRD_BLOCKLIST", raising=False)
    assert get_settings().blocklist != []


# --------------------------------------------------------------------------- #
# allowlist: GUI-prefs bridge (precedence)                                    #
# --------------------------------------------------------------------------- #
# NOTE: the autouse `_no_gui_allowlist` fixture (conftest) patches
# `_read_gui_allowlist` -> None for hermeticity; these tests re-patch it.


def test_allowlist_from_gui_prefs_when_env_unset(monkeypatch):
    # No OPENBIRD_ALLOWLIST -> inherit the menu-bar app's saved allowlist.
    monkeypatch.delenv("OPENBIRD_ALLOWLIST", raising=False)
    monkeypatch.setattr(
        "openbird.config._read_gui_allowlist", lambda: ["com.apple.Safari"]
    )
    assert get_settings().allowlist == ["com.apple.Safari"]


def test_env_allowlist_overrides_gui_prefs(monkeypatch):
    # An explicit env var wins over the GUI prefs (and the reader isn't consulted).
    monkeypatch.setenv("OPENBIRD_ALLOWLIST", "com.google.Chrome")
    monkeypatch.setattr(
        "openbird.config._read_gui_allowlist", lambda: ["com.apple.Safari"]
    )
    assert get_settings().allowlist == ["com.google.Chrome"]


def test_empty_env_allowlist_beats_gui_prefs(monkeypatch):
    # OPENBIRD_ALLOWLIST="" is an explicit clear and must win over GUI prefs.
    monkeypatch.setenv("OPENBIRD_ALLOWLIST", "")
    monkeypatch.setattr(
        "openbird.config._read_gui_allowlist", lambda: ["com.apple.Safari"]
    )
    assert get_settings().allowlist == []


def test_allowlist_empty_when_no_env_and_no_prefs(monkeypatch):
    monkeypatch.delenv("OPENBIRD_ALLOWLIST", raising=False)
    monkeypatch.setattr("openbird.config._read_gui_allowlist", lambda: None)
    assert get_settings().allowlist == []


# --------------------------------------------------------------------------- #
# _read_gui_allowlist: parsing & failure modes                                #
# --------------------------------------------------------------------------- #


def _fake_defaults(monkeypatch, *, returncode=0, stdout=b"", raises=None):
    """Stub macOS so `_read_gui_allowlist` exercises its parse path on any host."""
    monkeypatch.setattr("openbird.config.sys.platform", "darwin")

    def fake_run(*_args, **_kwargs):
        if raises is not None:
            raise raises
        return subprocess.CompletedProcess(_args, returncode, stdout=stdout, stderr=b"")

    monkeypatch.setattr("openbird.config.subprocess.run", fake_run)


def test_read_gui_allowlist_parses_normalizes_and_dedupes(monkeypatch):
    payload = plistlib.dumps(
        {"openbird.captureAllowlist": [" com.a ", "com.a", "com.b", "", 5]}
    )
    _fake_defaults(monkeypatch, stdout=payload)
    # trimmed, de-duped, order-preserved, non-strings dropped.
    assert _read_gui_allowlist() == ["com.a", "com.b"]


def test_read_gui_allowlist_none_on_non_darwin(monkeypatch):
    monkeypatch.setattr("openbird.config.sys.platform", "linux")
    # subprocess must not even be consulted off-macOS.
    monkeypatch.setattr(
        "openbird.config.subprocess.run",
        lambda *a, **k: pytest.fail("defaults must not run off-macOS"),
    )
    assert _read_gui_allowlist() is None


def test_read_gui_allowlist_none_on_nonzero_returncode(monkeypatch):
    _fake_defaults(monkeypatch, returncode=1, stdout=b"")
    assert _read_gui_allowlist() is None


def test_read_gui_allowlist_none_on_malformed_xml(monkeypatch):
    _fake_defaults(monkeypatch, stdout=b"<not a plist>")
    assert _read_gui_allowlist() is None


def test_read_gui_allowlist_none_when_key_absent(monkeypatch):
    _fake_defaults(monkeypatch, stdout=plistlib.dumps({"some.other.key": ["x"]}))
    assert _read_gui_allowlist() is None


def test_read_gui_allowlist_none_when_defaults_missing(monkeypatch):
    _fake_defaults(monkeypatch, raises=FileNotFoundError("no defaults"))
    assert _read_gui_allowlist() is None


def test_read_gui_allowlist_none_when_only_invalid_entries(monkeypatch):
    _fake_defaults(monkeypatch, stdout=plistlib.dumps({"openbird.captureAllowlist": ["", 7]}))
    assert _read_gui_allowlist() is None


def test_session_gap_seconds_default_and_valid(tmp_path):
    assert Settings(data_dir=tmp_path).session_gap_seconds == 300.0
    assert Settings(data_dir=tmp_path, session_gap_seconds=0).session_gap_seconds == 0.0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), -1.0])
def test_session_gap_seconds_rejects_non_finite_or_negative(tmp_path, bad):
    # A bad gap silently freezes/over-splits session segmentation; reject at the
    # single source of truth (env coercion produces a float, so this is the gate).
    with pytest.raises(ValueError, match="session_gap_seconds"):
        Settings(data_dir=tmp_path, session_gap_seconds=bad)


def test_defaults_are_sane():
    s = Settings()
    assert s.llm_timeout == 60.0
    assert s.embed_timeout == 30.0
    assert s.llm_num_retries == 2
    assert s.allow_cloud is False
    assert s.deep_brain_enabled is False
    assert s.deep_brain_excluded_apps == []
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


# --------------------------------------------------------------------------- #
# RAM-tiered generation-model default                                         #
# --------------------------------------------------------------------------- #

_GIB = 1024**3


@pytest.mark.parametrize(
    "total_bytes, expected",
    [
        (16 * _GIB, "ollama/qwen3:4b"),       # 16 GB Mac -> small tier
        (18 * _GIB, "ollama/qwen3:4b"),       # exactly at cutoff (<=) -> small tier
        (18 * _GIB + 1, "ollama/qwen3:8b"),   # one byte over -> large tier
        (32 * _GIB, "ollama/qwen3:8b"),       # 32 GB Mac -> large tier
        (0, "ollama/qwen3:4b"),               # probe failure -> conservative tier
    ],
)
def test_llm_model_default_is_ram_tiered(monkeypatch, total_bytes, expected):
    # Patch the memory probe at its module seam so the tier is deterministic.
    monkeypatch.setattr("openbird.config._total_memory_bytes", lambda: total_bytes)
    reset_settings_cache()
    try:
        assert get_settings().llm_model == expected
    finally:
        # Clear the Settings cached from the monkeypatched probe so a later test
        # can't read this stale value once monkeypatch unwinds (order-independence).
        reset_settings_cache()


def test_llm_model_env_override_beats_ram_tier(monkeypatch):
    # An explicit OPENBIRD_LLM_MODEL must win over the hardware-derived default.
    monkeypatch.setattr("openbird.config._total_memory_bytes", lambda: 32 * _GIB)
    monkeypatch.setenv("OPENBIRD_LLM_MODEL", "ollama/llama3.2")
    reset_settings_cache()
    try:
        assert get_settings().llm_model == "ollama/llama3.2"
    finally:
        reset_settings_cache()


def test_total_memory_bytes_never_raises():
    # Contract: the probe degrades to a non-negative int, never an exception.
    from openbird.config import _total_memory_bytes

    assert isinstance(_total_memory_bytes(), int)
    assert _total_memory_bytes() >= 0


# ---------------------------------------------------------------------------
# Stream-capture timing knobs (Phase A) — clamps, not rejections
# ---------------------------------------------------------------------------


def test_capture_timing_defaults():
    s = Settings(data_dir="/tmp/openbird-test-config")
    assert s.capture_afk_threshold_seconds == 150.0
    assert s.capture_idle_tick_seconds == 5.0
    assert s.capture_force_ceiling_seconds == 60.0
    assert s.capture_min_gap_seconds == 1.0


def test_capture_idle_tick_clamped_to_legal_range():
    lo = Settings(data_dir="/tmp/openbird-test-config", capture_idle_tick_seconds=0.01)
    assert lo.capture_idle_tick_seconds == 5.0
    hi = Settings(data_dir="/tmp/openbird-test-config", capture_idle_tick_seconds=99.0)
    assert hi.capture_idle_tick_seconds == 10.0


def test_capture_min_gap_floor_is_one_second():
    s = Settings(data_dir="/tmp/openbird-test-config", capture_min_gap_seconds=0.01)
    assert s.capture_min_gap_seconds == 1.0


def test_capture_afk_threshold_floor():
    s = Settings(data_dir="/tmp/openbird-test-config", capture_afk_threshold_seconds=1.0)
    assert s.capture_afk_threshold_seconds == 30.0


def test_capture_ceiling_cross_field_floor():
    # Ceiling below both the floor and the tick must be raised to their max.
    s = Settings(
        data_dir="/tmp/openbird-test-config",
        capture_idle_tick_seconds=8.0,
        capture_force_ceiling_seconds=2.0,
    )
    assert s.capture_force_ceiling_seconds == 8.0


def test_capture_timing_nonfinite_falls_back_to_default():
    s = Settings(
        data_dir="/tmp/openbird-test-config",
        capture_idle_tick_seconds=float("nan"),
        capture_force_ceiling_seconds=float("inf"),
    )
    assert s.capture_idle_tick_seconds == 5.0
    assert s.capture_force_ceiling_seconds == 60.0


def test_capture_timing_env_override_clamped(monkeypatch):
    monkeypatch.setenv("OPENBIRD_CAPTURE_IDLE_TICK_SECONDS", "0.5")
    monkeypatch.setenv("OPENBIRD_CAPTURE_MIN_GAP_SECONDS", "0.1")
    reset_settings_cache()
    try:
        s = get_settings()
        assert s.capture_idle_tick_seconds == 5.0
        assert s.capture_min_gap_seconds == 1.0
    finally:
        reset_settings_cache()
