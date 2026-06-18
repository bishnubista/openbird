"""Unit tests for the preflight aggregation, using fakes (no live services)."""

from __future__ import annotations

import json
import sqlite3

import pytest

from openbird.preflight import (
    GRANT_FAILED,
    GRANT_PASSED,
    GRANT_UNKNOWN,
    check_embedding,
    check_encryption,
    check_macos_capabilities,
    check_ollama,
    check_sqlite_vec,
    run_preflight,
)
from openbird.config import Settings


# --------------------------------------------------------------------------- #
# Fakes                                                                       #
# --------------------------------------------------------------------------- #


def make_http_get(*, status=200, models=("llama3.2:latest", "nomic-embed-text:latest"), raise_exc=None):
    """Build a fake http_get(url, timeout) -> (status, body)."""

    def _get(url, timeout):
        if raise_exc is not None:
            raise raise_exc
        body = json.dumps({"models": [{"name": m} for m in models]}).encode("utf-8")
        return status, body

    return _get


class FakeEmbedProvider:
    def __init__(self, settings, *, dim=768):
        self.embed_dim = dim
        self._dim = dim

    def embed(self, texts):
        return [[0.0] * self._dim for _ in texts]


class _CipherConn:
    """A sqlite3 connection wrapper that answers ``PRAGMA cipher_version``.

    Simulates a genuine SQLCipher connection: the live ``PRAGMA cipher_version``
    probe returns a non-empty version string. A plain ``sqlite3.connect`` does
    NOT — which is exactly what distinguishes a real encrypted DB from a fake.
    """

    def __init__(self, *, cipher_version="4.5.6 community", journal_mode="wal"):
        self._conn = sqlite3.connect(":memory:")
        self._cipher_version = cipher_version
        self._journal_mode = journal_mode

    def execute(self, sql, *args):
        norm = " ".join(sql.lower().split())
        if norm.startswith("pragma cipher_version"):
            return _FakeCursor([(self._cipher_version,)] if self._cipher_version else [])
        if norm.startswith("pragma journal_mode"):
            return _FakeCursor([(self._journal_mode,)])
        return self._conn.execute(sql, *args)

    def close(self):
        self._conn.close()


class _FakeCursor:
    def __init__(self, rows):
        self._rows = list(rows)

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return list(self._rows)


class _FakeHandle:
    """Mimics crypto.DbHandle: a connection plus verified backend metadata."""

    def __init__(self, conn, *, backend, encrypted, cipher_version=None, wal_enabled=False):
        self.conn = conn
        self.backend = backend
        self.encrypted = encrypted
        self.cipher_version = cipher_version
        self.wal_enabled = wal_enabled


def encrypted_handle_opener():
    """Opener returning a verified, genuinely-encrypted DbHandle."""

    def _opener(path, *, settings):
        conn = _CipherConn(cipher_version="4.5.6 community", journal_mode="wal")
        return _FakeHandle(
            conn,
            backend="sqlcipher",
            encrypted=True,
            cipher_version="4.5.6 community",
            wal_enabled=True,
        )

    return _opener


def plaintext_handle_opener():
    """Opener returning a verified plaintext DbHandle (no cipher_version)."""

    def _opener(path, *, settings):
        conn = sqlite3.connect(":memory:")
        return _FakeHandle(conn, backend="sqlite3", encrypted=False)

    return _opener


def lying_flag_opener():
    """A plain sqlite3 conn whose opener flips settings.encryption_enabled True.

    This is the *attack*: a plaintext connection that merely sets the flag. The
    live ``PRAGMA cipher_version`` probe returns nothing, so it must NOT pass.
    """

    def _opener(path, *, settings):
        settings.encryption_enabled = True
        return sqlite3.connect(":memory:")

    return _opener


def lying_handle_opener():
    """A DbHandle claiming encrypted=True over a plaintext (no-cipher) conn.

    Even a lying handle cannot fake encryption: the live probe disagrees.
    """

    def _opener(path, *, settings):
        conn = sqlite3.connect(":memory:")
        return _FakeHandle(conn, backend="sqlcipher", encrypted=True, cipher_version="x")

    return _opener


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(data_dir=tmp_path, embed_dim=768)


def all_passed_probe(capability: str) -> str:
    return GRANT_PASSED


def mixed_probe(capability: str) -> str:
    return GRANT_PASSED if capability != "system_audio" else GRANT_FAILED


# --------------------------------------------------------------------------- #
# check_ollama                                                                #
# --------------------------------------------------------------------------- #


def test_ollama_reachable_with_all_models():
    res = check_ollama(host="http://x:11434", http_get=make_http_get())
    assert res["reachable"] is True
    assert res["models"] == {"llama3.2": True, "nomic-embed-text": True}
    assert res["missing_models"] == []


def test_ollama_missing_model_is_reported():
    res = check_ollama(http_get=make_http_get(models=("llama3.2:latest",)))
    assert res["reachable"] is True
    assert res["models"]["nomic-embed-text"] is False
    assert res["missing_models"] == ["nomic-embed-text"]


def test_ollama_unreachable_does_not_raise():
    res = check_ollama(http_get=make_http_get(raise_exc=ConnectionRefusedError()))
    assert res["reachable"] is False
    assert res["error"] == "ConnectionRefusedError"
    assert res["missing_models"] == ["llama3.2", "nomic-embed-text"]


def test_ollama_non_200_is_unreachable():
    res = check_ollama(http_get=make_http_get(status=500))
    assert res["reachable"] is False
    assert "500" in res["error"]


def test_ollama_matches_bare_model_name():
    res = check_ollama(http_get=make_http_get(models=("llama3.2", "nomic-embed-text")))
    assert res["missing_models"] == []


# --------------------------------------------------------------------------- #
# check_embedding                                                             #
# --------------------------------------------------------------------------- #


def test_embedding_no_probe_reports_config(settings):
    res = check_embedding(settings, probe=False)
    assert res["model"] == "ollama/nomic-embed-text"
    assert res["configured_dim"] == 768
    assert res["probed"] is False
    assert res["dim_ok"] is None


def test_embedding_probe_dim_ok(settings):
    res = check_embedding(
        settings,
        probe=True,
        provider_factory=lambda s: FakeEmbedProvider(s, dim=768),
    )
    assert res["probed"] is True
    assert res["probed_dim"] == 768
    assert res["dim_ok"] is True


def test_embedding_probe_dim_mismatch(settings):
    res = check_embedding(
        settings,
        probe=True,
        provider_factory=lambda s: FakeEmbedProvider(s, dim=512),
    )
    assert res["dim_ok"] is False
    assert res["probed_dim"] == 512


def test_embedding_probe_failure_captured(settings):
    def boom(_s):
        raise RuntimeError("no ollama")

    res = check_embedding(settings, probe=True, provider_factory=boom)
    assert res["probed"] is False
    assert res["error"] == "RuntimeError"


# --------------------------------------------------------------------------- #
# check_sqlite_vec                                                            #
# --------------------------------------------------------------------------- #


def test_sqlite_vec_and_fts5_available():
    res = check_sqlite_vec()
    assert res["vec_available"] is True
    assert res["vec_version"] is not None
    assert res["fts5_available"] is True


def test_sqlite_vec_load_failure_is_reported():
    class BadConn:
        def enable_load_extension(self, *_):
            raise sqlite3.OperationalError("extensions disabled")

        def execute(self, *_):  # FTS5 path still works on a real conn
            return sqlite3.connect(":memory:").execute("CREATE VIRTUAL TABLE t USING fts5(x)")

        def close(self):
            pass

    res = check_sqlite_vec(connect=lambda: BadConn())
    assert res["vec_available"] is False
    assert res["error"] is not None


# --------------------------------------------------------------------------- #
# check_encryption — verified on the LIVE connection                          #
# --------------------------------------------------------------------------- #


def test_encryption_plaintext(settings):
    res = check_encryption(settings, db_opener=plaintext_handle_opener())
    assert res["enabled"] is False
    assert res["status"] == "plaintext-0600"
    assert res["backend"] == "sqlite3"
    assert res["verified"] is True


def test_encryption_encrypted_requires_live_cipher_version(settings):
    res = check_encryption(settings, db_opener=encrypted_handle_opener())
    assert res["enabled"] is True
    assert res["status"] == "encrypted"
    assert res["backend"] == "sqlcipher"
    assert res["cipher_version"] == "4.5.6 community"
    assert res["wal_enabled"] is True
    assert res["verified"] is True


def test_encryption_plain_conn_with_flag_true_must_not_pass(settings):
    """A plain sqlite3 connection that merely flips the flag must NOT be 'encrypted'.

    This is the regression guard for the falsely-green-encryption finding: trust
    only a live ``PRAGMA cipher_version`` probe, never settings.encryption_enabled.
    """
    res = check_encryption(settings, db_opener=lying_flag_opener())
    assert settings.encryption_enabled is True  # the opener lied
    assert res["enabled"] is False  # but preflight refuses to believe it
    assert res["status"] == "plaintext-0600"
    assert res["backend"] == "sqlite3"


def test_encryption_lying_handle_cannot_fake_encryption(settings):
    """A DbHandle claiming encrypted=True over a plaintext conn must NOT pass."""
    res = check_encryption(settings, db_opener=lying_handle_opener())
    assert res["enabled"] is False
    assert res["status"] == "plaintext-0600"


def test_encryption_opener_failure_is_unknown(settings):
    def boom(_path, *, settings):
        raise OSError("disk gone")

    res = check_encryption(settings, db_opener=boom)
    assert res["status"] == "unknown"
    assert res["error"] == "OSError"
    assert res["verified"] is False


# --------------------------------------------------------------------------- #
# check_macos_capabilities — unknown / failed / passed                        #
# --------------------------------------------------------------------------- #


def test_macos_off_mac_is_unknown():
    res = check_macos_capabilities(system="Linux")
    assert res["is_macos"] is False
    assert res["accessibility"] == GRANT_UNKNOWN
    assert res["system_audio"] == GRANT_UNKNOWN
    assert res["all_passed"] is False


def test_macos_on_mac_without_helper_is_unknown_not_green():
    res = check_macos_capabilities(system="Darwin")
    assert res["is_macos"] is True
    assert res["helper_present"] is False
    assert res["accessibility"] == GRANT_UNKNOWN
    assert res["all_passed"] is False  # unknown is never green
    assert "signed helper" in res["note"]


def test_macos_signed_helper_all_passed():
    res = check_macos_capabilities(system="Darwin", helper_probe=all_passed_probe)
    assert res["helper_present"] is True
    assert res["accessibility"] == GRANT_PASSED
    assert res["system_audio"] == GRANT_PASSED
    assert res["all_passed"] is True


def test_macos_signed_helper_mixed_is_not_all_passed():
    res = check_macos_capabilities(system="Darwin", helper_probe=mixed_probe)
    assert res["system_audio"] == GRANT_FAILED
    assert res["any_failed"] is True
    assert res["all_passed"] is False


def test_macos_helper_probe_error_is_unknown():
    def boom(_cap):
        raise RuntimeError("helper crashed")

    res = check_macos_capabilities(system="Darwin", helper_probe=boom)
    assert res["accessibility"] == GRANT_UNKNOWN
    assert res["all_passed"] is False
    assert res["probe_error"] == "RuntimeError"


# --------------------------------------------------------------------------- #
# run_preflight aggregation                                                   #
# --------------------------------------------------------------------------- #


def test_run_preflight_runtime_ok_but_release_gate_blocked_when_grants_unknown(settings):
    """Runtime can be OK while the release gate stays blocked on unknown grants."""
    report = run_preflight(
        settings,
        http_get=make_http_get(),
        db_opener=encrypted_handle_opener(),
        system="Darwin",
        ollama_host="http://x:11434",
    )
    # Runtime readiness ignores capture grants / encryption.
    assert report["runtime_ok"] is True
    assert report["ok"] is True  # back-compat alias of runtime_ok
    # Encryption is verified-encrypted, but macOS grants are unknown (no helper).
    assert report["encryption"]["enabled"] is True
    assert report["macos"]["all_passed"] is False
    # Release gate must NOT be green while TCC/audio grants are unknown.
    assert report["release_gate_ok"] is False
    assert report["macos"]["is_macos"] is True
    json.dumps(report)  # whole report must be JSON-serializable


def test_run_preflight_release_gate_green_when_all_proven(settings):
    report = run_preflight(
        settings,
        http_get=make_http_get(),
        db_opener=encrypted_handle_opener(),
        system="Darwin",
        helper_probe=all_passed_probe,
        ollama_host="http://x:11434",
    )
    assert report["runtime_ok"] is True
    assert report["encryption"]["enabled"] is True
    assert report["macos"]["all_passed"] is True
    assert report["release_gate_ok"] is True


def test_run_preflight_release_gate_blocked_when_encryption_plaintext(settings):
    report = run_preflight(
        settings,
        http_get=make_http_get(),
        db_opener=plaintext_handle_opener(),
        system="Darwin",
        helper_probe=all_passed_probe,
        ollama_host="http://x:11434",
    )
    assert report["runtime_ok"] is True
    assert report["encryption"]["enabled"] is False
    # Even with all TCC grants passed, plaintext storage blocks the release gate.
    assert report["release_gate_ok"] is False


def test_run_preflight_release_gate_blocked_when_flag_lies(settings):
    """The falsely-green-encryption attack must not open the release gate."""
    report = run_preflight(
        settings,
        http_get=make_http_get(),
        db_opener=lying_flag_opener(),
        system="Darwin",
        helper_probe=all_passed_probe,
        ollama_host="http://x:11434",
    )
    assert report["encryption"]["enabled"] is False
    assert report["release_gate_ok"] is False


def test_run_preflight_off_mac_release_gate_ignores_tcc(settings):
    report = run_preflight(
        settings,
        http_get=make_http_get(),
        db_opener=encrypted_handle_opener(),
        system="Linux",
        ollama_host="http://x:11434",
    )
    assert report["macos"]["is_macos"] is False
    # Off-mac TCC/audio gates are N/A; encryption proven -> release gate green.
    assert report["release_gate_ok"] is True


def test_run_preflight_ollama_down_not_ok(settings):
    report = run_preflight(
        settings,
        http_get=make_http_get(raise_exc=ConnectionRefusedError()),
        db_opener=plaintext_handle_opener(),
    )
    assert report["ok"] is False
    assert report["runtime_ok"] is False
    assert report["release_gate_ok"] is False
    assert report["ollama"]["reachable"] is False


def test_run_preflight_missing_model_not_ok(settings):
    report = run_preflight(
        settings,
        http_get=make_http_get(models=("llama3.2:latest",)),
        db_opener=plaintext_handle_opener(),
    )
    assert report["ok"] is False
    assert report["ollama"]["missing_models"] == ["nomic-embed-text"]


def test_run_preflight_skip_probes_reports_unknown(settings):
    report = run_preflight(
        settings,
        probe_ollama=False,
        probe_encryption=False,
    )
    assert report["ollama"]["reachable"] == "unknown"
    assert report["ok"] is False  # unknown ollama is not OK
    # Skipped encryption probe is unknown/unverified, NOT trusted from the flag.
    assert report["encryption"]["status"] == "unknown"
    assert report["encryption"]["verified"] is False
    assert report["release_gate_ok"] is False


def test_run_preflight_reflects_privacy_config(tmp_path):
    s = Settings(
        data_dir=tmp_path,
        embed_dim=768,
        allowlist=["com.apple.Safari"],
        blocklist=["com.apple.Terminal"],
        ocr_enabled=True,
    )
    report = run_preflight(
        s,
        http_get=make_http_get(),
        db_opener=plaintext_handle_opener(),
    )
    assert report["privacy"]["allowlist"] == ["com.apple.Safari"]
    assert report["privacy"]["blocklist"] == ["com.apple.Terminal"]
    assert report["privacy"]["ocr_enabled"] is True


# --------------------------------------------------------------------------- #
# cloud section + route-aware runtime_ok (H3) / host agreement (M1)            #
# --------------------------------------------------------------------------- #


def test_cloud_section_local_default(settings):
    report = run_preflight(
        settings, http_get=make_http_get(), db_opener=plaintext_handle_opener()
    )
    assert report["cloud"]["active"] is False
    assert report["cloud"]["blocked"] is False
    assert report["cloud"]["remote_models"] == {}


def test_cloud_section_blocked_without_opt_in(tmp_path):
    s = Settings(data_dir=tmp_path, embed_dim=768, llm_model="gpt-4o-mini")
    report = run_preflight(
        s, http_get=make_http_get(), db_opener=plaintext_handle_opener()
    )
    assert report["cloud"]["active"] is True
    assert report["cloud"]["blocked"] is True
    assert report["cloud"]["remote_models"] == {"llm": "gpt-4o-mini"}
    # A remote model without opt-in cannot run -> not runtime-OK.
    assert report["runtime_ok"] is False


def test_cloud_route_with_opt_in_does_not_require_ollama(tmp_path):
    # Cloud chat + cloud embed, opted in: Ollama down should NOT block runtime_ok.
    s = Settings(
        data_dir=tmp_path,
        embed_dim=768,
        llm_model="gpt-4o-mini",
        embed_model="text-embedding-3-small",
        allow_cloud=True,
    )
    report = run_preflight(
        s,
        http_get=make_http_get(raise_exc=ConnectionRefusedError()),
        db_opener=plaintext_handle_opener(),
    )
    assert report["cloud"]["active"] is True
    assert report["cloud"]["blocked"] is False
    assert report["cloud"]["uses_local_ollama"] is False
    # sqlite is available in this env; Ollama is irrelevant for a cloud route.
    assert report["runtime_ok"] is True


def test_preflight_required_models_derived_from_settings(tmp_path):
    # Custom Ollama models -> preflight must check THOSE, not the hard defaults.
    s = Settings(
        data_dir=tmp_path,
        embed_dim=768,
        llm_model="ollama/mistral",
        embed_model="ollama/mxbai-embed-large",
    )
    report = run_preflight(
        s,
        http_get=make_http_get(models=("mistral:latest", "mxbai-embed-large:latest")),
        db_opener=plaintext_handle_opener(),
    )
    assert set(report["ollama"]["required_models"]) == {"mistral", "mxbai-embed-large"}
    assert report["ollama"]["missing_models"] == []
    assert report["runtime_ok"] is True


def test_preflight_and_provider_agree_on_ollama_host(monkeypatch, tmp_path):
    # M1 regression: preflight host == runtime provider api_base host.
    monkeypatch.setenv("OPENBIRD_OLLAMA_HOST", "http://customhost:4242")
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    s = Settings(data_dir=tmp_path, embed_dim=768)
    report = run_preflight(
        s, http_get=make_http_get(), db_opener=plaintext_handle_opener()
    )
    preflight_host = report["ollama"]["host"]

    # Build the runtime provider and capture the api_base it would use.
    from openbird.llm.provider import LLMProvider

    fake = type(
        "F", (), {
            "embedding_kwargs": None,
            "embedding": lambda self, *, model, input, **kw: (
                setattr(self, "embedding_kwargs", kw)
                or {"data": [{"embedding": [0.0] * 768} for _ in input]}
            ),
        },
    )()
    monkeypatch.setitem(__import__("sys").modules, "litellm", fake)
    LLMProvider(s).embed(["x"])
    assert preflight_host == "http://customhost:4242"
    assert fake.embedding_kwargs["api_base"] == preflight_host
