"""Unit tests for the preflight aggregation, using fakes (no live services)."""

from __future__ import annotations

import json
import sqlite3

import pytest

from openbird.preflight import (
    GRANT_FAILED,
    GRANT_PASSED,
    GRANT_UNKNOWN,
    _packaged_helper_probe,
    _run_helper_grant_probe,
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


def make_http_get(
    *,
    status=200,
    # Default serves BOTH qwen3 tiers (4b/8b) so the RAM-tiered default generation
    # route resolves "present" regardless of the host's memory tier, plus the
    # default embedder (embeddinggemma). Tests needing a gap pass explicit `models`.
    models=("qwen3:4b", "qwen3:8b", "embeddinggemma:latest"),
    raise_exc=None,
    version="0.11.10",  # meets the EmbeddingGemma minimum by default
):
    """Build a fake http_get(url, timeout) -> (status, body), URL-aware.

    Serves ``/api/tags`` (model list) and ``/api/version`` (version string) so the
    EmbeddingGemma Ollama-version gate can be exercised. ``version=None`` makes the
    version endpoint return an empty body (version unknown / advisory).
    """

    def _get(url, timeout):
        if raise_exc is not None:
            raise raise_exc
        if url.endswith("/api/version"):
            body = json.dumps({"version": version}).encode("utf-8") if version else b""
            return status, body
        body = json.dumps({"models": [{"name": m} for m in models]}).encode("utf-8")
        return status, body

    return _get


class FakeEmbedProvider:
    def __init__(self, settings, *, dim=768):
        self.embed_dim = dim
        self._dim = dim

    def embed(self, texts):
        return [[0.0] * self._dim for _ in texts]

    def complete(self, messages, *, json_schema=None):
        # Successful completion probe (the chat endpoint "works").
        return "pong"


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
    assert res["models"] == {"qwen3": True, "embeddinggemma": True}
    assert res["missing_models"] == []


def test_ollama_missing_model_is_reported():
    res = check_ollama(http_get=make_http_get(models=("qwen3:4b",)))
    assert res["reachable"] is True
    assert res["models"]["embeddinggemma"] is False
    assert res["missing_models"] == ["embeddinggemma"]


def test_ollama_unreachable_does_not_raise():
    res = check_ollama(http_get=make_http_get(raise_exc=ConnectionRefusedError()))
    assert res["reachable"] is False
    assert res["error"] == "ConnectionRefusedError"
    assert res["missing_models"] == ["qwen3", "embeddinggemma"]


def test_ollama_non_200_is_unreachable():
    res = check_ollama(http_get=make_http_get(status=500))
    assert res["reachable"] is False
    assert "500" in res["error"]


def test_ollama_matches_bare_model_name():
    res = check_ollama(http_get=make_http_get(models=("qwen3", "embeddinggemma")))
    assert res["missing_models"] == []


def test_embeddinggemma_version_gate_ok_on_new_ollama():
    # Default route requires embeddinggemma -> the Ollama-version gate fires and
    # passes when the server meets the 0.11.10 minimum.
    res = check_ollama(http_get=make_http_get(version="0.11.10"))
    assert res["version"] == "0.11.10"
    assert res["version_ok"] is True


def test_embeddinggemma_version_gate_fails_on_old_ollama():
    res = check_ollama(http_get=make_http_get(version="0.11.9"))
    assert res["version"] == "0.11.9"
    assert res["version_ok"] is False


def test_version_gate_unknown_when_unreadable_is_advisory():
    # version=None -> the version endpoint returns empty; gate is advisory (None).
    res = check_ollama(http_get=make_http_get(version=None))
    assert res["version_ok"] is None


def test_no_version_gate_when_embeddinggemma_not_required():
    # A non-embeddinggemma embedder route must NOT probe/gate the Ollama version.
    res = check_ollama(required_models=("qwen3", "nomic-embed-text"), http_get=make_http_get())
    assert "version_ok" not in res


# --------------------------------------------------------------------------- #
# check_embedding                                                             #
# --------------------------------------------------------------------------- #


def test_embedding_no_probe_reports_config(settings):
    res = check_embedding(settings, probe=False)
    assert res["model"] == "ollama/embeddinggemma"
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


def test_packaged_helper_probe_merges_capture_and_audio_helpers(tmp_path):
    capture = tmp_path / "capture-helper"
    audio = tmp_path / "audio-helper"
    calls = []

    def runner(path):
        calls.append(path.name)
        if path == capture:
            return {"accessibility": "passed"}
        return {
            "screen_recording": "passed",
            "microphone": "denied",
            "system_audio": "passed",
        }

    probe = _packaged_helper_probe(
        capture_helper=capture,
        audio_helper=audio,
        runner=runner,
    )
    assert probe is not None
    assert probe("accessibility") == GRANT_PASSED
    assert probe("screen_recording") == GRANT_PASSED
    assert probe("microphone") == GRANT_FAILED
    assert probe("system_audio") == GRANT_PASSED
    # Each helper is executed once and cached across capability lookups.
    assert calls == ["capture-helper", "audio-helper"]


def test_packaged_helper_probe_unavailable_without_helper_env(monkeypatch):
    monkeypatch.delenv("OPENBIRD_CAPTURE_HELPER", raising=False)
    monkeypatch.delenv("OPENBIRD_AUDIO_HELPER", raising=False)
    assert _packaged_helper_probe() is None


def test_packaged_helper_probe_bad_json_keeps_grants_unknown(tmp_path):
    capture = tmp_path / "capture-helper"
    audio = tmp_path / "audio-helper"
    probe = _packaged_helper_probe(
        capture_helper=capture,
        audio_helper=audio,
        runner=lambda _path: {},
    )
    assert probe is not None
    assert probe("accessibility") == GRANT_UNKNOWN
    assert probe("screen_recording") == GRANT_UNKNOWN


def test_helper_grant_subprocess_parses_json(tmp_path):
    helper = tmp_path / "helper"
    helper.write_text('#!/bin/sh\nprintf \'{"accessibility":"authorized"}\\n\'\n')
    helper.chmod(0o700)
    assert _run_helper_grant_probe(helper) == {"accessibility": GRANT_PASSED}


def test_helper_grant_subprocess_failure_surfaces_probe_error(tmp_path):
    helper = tmp_path / "helper"
    helper.write_text("#!/bin/sh\nexit 2\n")
    helper.chmod(0o700)
    audio = tmp_path / "audio-helper"
    audio.write_text(
        "#!/bin/sh\n"
        "printf '{\"screen_recording\":\"passed\","
        "\"microphone\":\"passed\",\"system_audio\":\"passed\"}\\n'\n"
    )
    audio.chmod(0o700)
    probe = _packaged_helper_probe(capture_helper=helper, audio_helper=audio)

    res = check_macos_capabilities(system="Darwin", helper_probe=probe)
    assert res["accessibility"] == GRANT_UNKNOWN
    assert res["probe_error"] == "RuntimeError"
    assert "probe was unavailable or failed" in res["note"]

    # Cached helper failures should re-raise the original exception type.
    assert probe("screen_recording") == GRANT_PASSED
    with pytest.raises(RuntimeError):
        probe("accessibility")


def test_packaged_helper_probe_partial_helper_reports_unavailable(tmp_path):
    probe = _packaged_helper_probe(capture_helper=tmp_path / "capture-helper")

    res = check_macos_capabilities(system="Darwin", helper_probe=probe)
    assert res["screen_recording"] == GRANT_UNKNOWN
    assert res["probe_error"] == "FileNotFoundError"


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
        ollama_host="http://localhost:11434",
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
        ollama_host="http://localhost:11434",
    )
    assert report["runtime_ok"] is True
    assert report["encryption"]["enabled"] is True
    assert report["macos"]["all_passed"] is True
    assert report["release_gate_ok"] is True


def test_run_preflight_uses_packaged_helper_probe_on_real_macos(monkeypatch, settings):
    from openbird import preflight as preflight_module

    monkeypatch.setattr(preflight_module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        preflight_module,
        "_packaged_helper_probe",
        lambda: all_passed_probe,
    )
    report = run_preflight(
        settings,
        http_get=make_http_get(),
        db_opener=encrypted_handle_opener(),
        ollama_host="http://localhost:11434",
    )
    assert report["macos"]["helper_present"] is True
    assert report["macos"]["all_passed"] is True
    assert report["release_gate_ok"] is True


def test_run_preflight_release_gate_blocked_when_encryption_plaintext(settings):
    report = run_preflight(
        settings,
        http_get=make_http_get(),
        db_opener=plaintext_handle_opener(),
        system="Darwin",
        helper_probe=all_passed_probe,
        ollama_host="http://localhost:11434",
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
        ollama_host="http://localhost:11434",
    )
    assert report["encryption"]["enabled"] is False
    assert report["release_gate_ok"] is False


def test_run_preflight_off_mac_release_gate_ignores_tcc(settings):
    report = run_preflight(
        settings,
        http_get=make_http_get(),
        db_opener=encrypted_handle_opener(),
        system="Linux",
        ollama_host="http://localhost:11434",
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
    # Serve the generation tiers but NOT the embedder, so exactly one model is
    # missing and preflight reports not-ok.
    report = run_preflight(
        settings,
        http_get=make_http_get(models=("qwen3:4b", "qwen3:8b")),
        db_opener=plaintext_handle_opener(),
    )
    assert report["ok"] is False
    assert report["ollama"]["missing_models"] == ["embeddinggemma"]


def test_runtime_not_ready_on_too_old_ollama_for_embeddinggemma(settings):
    # All required models present, but Ollama < 0.11.10 can't serve embeddinggemma,
    # so runtime must NOT report ready (the version gate is ENFORCED, not just
    # recorded).
    report = run_preflight(
        settings,
        http_get=make_http_get(version="0.11.9"),
        db_opener=plaintext_handle_opener(),
    )
    assert report["ollama"]["missing_models"] == []
    assert report["ollama"]["version_ok"] is False
    assert report["runtime_ok"] is False


def test_runtime_ready_on_new_enough_ollama_for_embeddinggemma(settings):
    report = run_preflight(
        settings,
        http_get=make_http_get(version="0.11.10"),
        db_opener=plaintext_handle_opener(),
    )
    assert report["ollama"]["version_ok"] is True
    assert report["runtime_ok"] is True


def test_runtime_unaffected_when_version_unreadable(settings):
    # version unreadable -> advisory (None), must NOT block runtime readiness.
    report = run_preflight(
        settings,
        http_get=make_http_get(version=None),
        db_opener=plaintext_handle_opener(),
    )
    assert report["ollama"]["version_ok"] is None
    assert report["runtime_ok"] is True


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
# cloud section + route-aware runtime_ok / host agreement            #
# --------------------------------------------------------------------------- #


def test_cloud_section_local_default(settings):
    report = run_preflight(
        settings, http_get=make_http_get(), db_opener=plaintext_handle_opener()
    )
    assert report["cloud"]["active"] is False
    assert report["cloud"]["blocked"] is False
    assert report["cloud"]["remote_models"] == {}
    assert report["ollama"]["auto_pull_allowed"] is True


def test_ollama_auto_pull_allowed_for_loopback_host(settings):
    report = run_preflight(
        settings,
        ollama_host="http://127.0.0.1:11434",
        http_get=make_http_get(),
        db_opener=plaintext_handle_opener(),
    )
    assert report["ollama"]["host"] == "http://127.0.0.1:11434"
    assert report["ollama"]["auto_pull_allowed"] is True


def test_ollama_auto_pull_blocked_for_remote_host(settings):
    report = run_preflight(
        settings,
        ollama_host="http://10.0.0.5:11434",
        http_get=make_http_get(),
        db_opener=plaintext_handle_opener(),
    )
    assert report["ollama"]["host"] == "http://10.0.0.5:11434"
    assert report["ollama"]["auto_pull_allowed"] is False


def test_ollama_auto_pull_blocked_for_malformed_host(settings):
    report = run_preflight(
        settings,
        ollama_host="http://[::1",
        http_get=make_http_get(),
        db_opener=plaintext_handle_opener(),
    )
    assert report["ollama"]["auto_pull_allowed"] is False


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


def test_cloud_route_with_opt_in_not_ready_without_probe(tmp_path):
    # Cloud route, opted in but NOT probed: Ollama is irrelevant, but the remote
    # endpoint is unverified, so runtime_ok must stay False (a missing API key
    # would fail the first call — sqlite alone must not report READY).
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
        probe_embedding=False,
    )
    assert report["cloud"]["active"] is True
    assert report["cloud"]["blocked"] is False
    assert report["cloud"]["uses_local_ollama"] is False
    assert report["runtime_ok"] is False


def test_cloud_route_ready_when_embedding_probe_succeeds(tmp_path):
    # Same cloud route, but a successful embedding probe confirms the endpoint
    # works -> runtime_ok True without any Ollama dependency.
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
        provider_factory=lambda settings: FakeEmbedProvider(settings, dim=768),
        probe_embedding=True,
    )
    assert report["embedding"]["dim_ok"] is True
    assert report["runtime_ok"] is True


def test_mixed_local_chat_cloud_embed_needs_embed_probe(tmp_path):
    # MIXED route: local Ollama chat + cloud embed, opted in. Ollama is reachable
    # with the chat model, but the remote embed role is unverified without a probe
    # -> runtime_ok must stay False (the embed call would fail on a bad API key).
    s = Settings(
        data_dir=tmp_path,
        embed_dim=768,
        llm_model="ollama/llama3.2",
        embed_model="text-embedding-3-small",
        allow_cloud=True,
    )
    report = run_preflight(
        s,
        http_get=make_http_get(models=("llama3.2:latest",)),
        db_opener=plaintext_handle_opener(),
        probe_embedding=False,
    )
    assert report["cloud"]["active"] is True
    assert report["cloud"]["uses_local_ollama"] is True
    assert report["ollama"]["reachable"] is True
    assert report["runtime_ok"] is False  # remote embed role not probe-verified


def test_probe_uses_overridden_ollama_host(monkeypatch, tmp_path):
    # Regression: with an ollama_host override + probe, the embedding/completion
    # provider must target that host (api_base), not env/default localhost.
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.delenv("OPENBIRD_OLLAMA_HOST", raising=False)

    class _FakeLiteLLM:
        def __init__(self):
            self.embedding_api_base = None

        def embedding(self, *, model, **kw):
            # `input` arrives via **kw to avoid shadowing the builtin (Ruff A006).
            self.embedding_api_base = kw.get("api_base")
            texts = kw.get("input", [])
            return {"data": [{"embedding": [0.0] * 768} for _ in texts]}

        def completion(self, *, model, messages, **kw):
            return {"choices": [{"message": {"content": "ok"}}]}

    fake = _FakeLiteLLM()
    monkeypatch.setitem(__import__("sys").modules, "litellm", fake)

    s = Settings(data_dir=tmp_path, embed_dim=768)  # default ollama models
    # 127.0.0.1 stays loopback (no cloud opt-in needed) but is an explicit override.
    run_preflight(
        s,
        ollama_host="http://127.0.0.1:9988",
        probe_ollama=False,
        probe_encryption=False,
        probe_embedding=True,  # uses the real default provider factory
    )
    assert fake.embedding_api_base == "http://127.0.0.1:9988"


def test_mixed_route_ready_when_both_ollama_and_embed_probe_ok(tmp_path):
    s = Settings(
        data_dir=tmp_path,
        embed_dim=768,
        llm_model="ollama/llama3.2",
        embed_model="text-embedding-3-small",
        allow_cloud=True,
    )
    report = run_preflight(
        s,
        http_get=make_http_get(models=("llama3.2:latest",)),
        db_opener=plaintext_handle_opener(),
        provider_factory=lambda settings: FakeEmbedProvider(settings, dim=768),
        probe_embedding=True,
    )
    assert report["ollama"]["reachable"] is True
    assert report["embedding"]["dim_ok"] is True
    assert report["runtime_ok"] is True


def test_cloud_chat_local_embed_needs_completion_probe(tmp_path):
    # Remote CHAT + local Ollama embed, opted in. A successful EMBED probe must
    # NOT satisfy the remote chat role; without a completion probe -> not READY.
    s = Settings(
        data_dir=tmp_path,
        embed_dim=768,
        llm_model="gpt-4o-mini",
        embed_model="ollama/nomic-embed-text",
        allow_cloud=True,
    )
    report = run_preflight(
        s,
        http_get=make_http_get(models=("nomic-embed-text:latest",)),
        db_opener=plaintext_handle_opener(),
        provider_factory=lambda settings: FakeEmbedProvider(settings, dim=768),
        probe_embedding=False,  # no probes run at all
    )
    assert report["cloud"]["remote_models"] == {"llm": "gpt-4o-mini"}
    assert report["runtime_ok"] is False


def test_cloud_chat_ready_when_completion_probe_ok(tmp_path):
    s = Settings(
        data_dir=tmp_path,
        embed_dim=768,
        llm_model="gpt-4o-mini",
        embed_model="ollama/nomic-embed-text",
        allow_cloud=True,
    )
    report = run_preflight(
        s,
        http_get=make_http_get(models=("nomic-embed-text:latest",)),
        db_opener=plaintext_handle_opener(),
        provider_factory=lambda settings: FakeEmbedProvider(settings, dim=768),
        probe_embedding=True,  # runs both embed + completion probes
    )
    assert report["completion"]["ok"] is True
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


def test_default_preflight_requires_exact_ram_tier_tag(monkeypatch, tmp_path):
    # Regression for the RAM-tiered default: on a large-memory host the default
    # generation route is the EXACT tag qwen3:8b. A server that only has qwen3:4b
    # pulled must report qwen3:8b MISSING (not satisfied by the other tier) so
    # preflight isn't green while the runtime would request an absent tag.
    monkeypatch.setattr("openbird.config._total_memory_bytes", lambda: 32 * 1024**3)
    s = Settings(data_dir=tmp_path, embed_dim=768)  # default route -> qwen3:8b
    assert s.llm_model == "ollama/qwen3:8b"
    report = run_preflight(
        s,
        http_get=make_http_get(models=("qwen3:4b", "embeddinggemma:latest")),
        db_opener=plaintext_handle_opener(),
    )
    assert "qwen3:8b" in report["ollama"]["required_models"]
    assert report["ollama"]["missing_models"] == ["qwen3:8b"]
    assert report["runtime_ok"] is False


def test_ollama_host_override_classified_as_remote(tmp_path):
    # Regression: an explicit non-loopback ollama_host override must be reflected
    # in cloud classification (not just the probe), or it could look local.
    s = Settings(data_dir=tmp_path, embed_dim=768)  # default ollama models
    report = run_preflight(
        s,
        ollama_host="http://10.0.0.5:11434",
        http_get=make_http_get(),
        db_opener=plaintext_handle_opener(),
    )
    assert report["ollama"]["host"] == "http://10.0.0.5:11434"
    assert report["cloud"]["active"] is True
    assert report["cloud"]["blocked"] is True  # no OPENBIRD_ALLOW_CLOUD
    assert report["runtime_ok"] is False


def test_mlx_backend_not_runtime_ready(tmp_path):
    # The mlx backend is reserved (factory raises NotImplementedError), so even a
    # local mlx route with sqlite available must NOT report READY.
    s = Settings(
        data_dir=tmp_path,
        embed_dim=768,
        llm_backend="mlx",
        llm_model="mlx/Qwen",
        embed_model="mlx/embed",
    )
    report = run_preflight(
        s,
        http_get=make_http_get(raise_exc=ConnectionRefusedError()),
        db_opener=plaintext_handle_opener(),
    )
    assert report["backend"]["supported"] is False
    assert report["cloud"]["active"] is False
    assert report["runtime_ok"] is False


def test_mlx_model_strings_under_litellm_not_ready(tmp_path):
    # mlx/* model strings with the default litellm backend are unrunnable (litellm
    # cannot serve them); preflight must not report READY.
    s = Settings(
        data_dir=tmp_path,
        embed_dim=768,
        llm_model="mlx/Qwen",
        embed_model="mlx/embed",
    )
    report = run_preflight(
        s, http_get=make_http_get(), db_opener=plaintext_handle_opener()
    )
    assert report["backend"]["supported"] is False
    assert report["runtime_ok"] is False


def test_litellm_backend_supported(tmp_path):
    s = Settings(data_dir=tmp_path, embed_dim=768)  # default litellm
    report = run_preflight(
        s, http_get=make_http_get(), db_opener=plaintext_handle_opener()
    )
    assert report["backend"]["supported"] is True
    assert report["runtime_ok"] is True


def test_cloud_only_route_reports_no_ollama_requirements(tmp_path):
    # A cloud-only route must not report default Ollama models as required/missing.
    s = Settings(
        data_dir=tmp_path,
        embed_dim=1536,
        llm_model="gpt-4o-mini",
        embed_model="text-embedding-3-small",
        allow_cloud=True,
    )
    report = run_preflight(
        s, http_get=make_http_get(), db_opener=plaintext_handle_opener()
    )
    assert report["cloud"]["uses_local_ollama"] is False
    assert report["ollama"]["required_models"] == []
    assert report["ollama"]["missing_models"] == []
    # Cloud-only route: the Ollama probe is skipped and reported not-applicable
    # (no misleading "down" for a service the route never uses).
    assert report["ollama"]["reachable"] == "n/a"
    assert report["ollama"]["auto_pull_allowed"] is False


def test_preflight_and_provider_agree_on_ollama_host(monkeypatch, tmp_path):
    # Regression: preflight host == runtime provider api_base host. Use a
    # loopback custom host so the default ollama/* route stays local (the host is
    # threaded the same way regardless of loopback/remote).
    monkeypatch.setenv("OPENBIRD_OLLAMA_HOST", "http://127.0.0.1:4242")
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    s = Settings(data_dir=tmp_path, embed_dim=768)
    report = run_preflight(
        s, http_get=make_http_get(), db_opener=plaintext_handle_opener()
    )
    preflight_host = report["ollama"]["host"]

    # Build the runtime provider and capture the api_base it would use.
    from openbird.llm.provider import LLMProvider

    def _fake_embedding(self, *, model, **kw):
        # Pop the texts out of **kw (rather than an `input=` param) so we don't
        # shadow the `input` builtin (Ruff A006) while keeping embedding_kwargs
        # to the remaining kwargs only (e.g. api_base) — matching the assertion.
        texts = kw.pop("input")
        self.embedding_kwargs = kw
        return {"data": [{"embedding": [0.0] * 768} for _ in texts]}

    fake = type(
        "F", (), {"embedding_kwargs": None, "embedding": _fake_embedding}
    )()
    monkeypatch.setitem(__import__("sys").modules, "litellm", fake)
    LLMProvider(s).embed(["x"])
    assert preflight_host == "http://127.0.0.1:4242"
    assert fake.embedding_kwargs["api_base"] == preflight_host


def test_preflight_preserves_ollama_model_tag(tmp_path):
    # A configured tag (llama3.2:3b) must be checked EXACTLY: a server that only
    # has llama3.2:latest must NOT satisfy it (preflight green != runtime works).
    s = Settings(data_dir=tmp_path, embed_dim=768, llm_model="ollama/llama3.2:3b")
    report = run_preflight(
        s,
        http_get=make_http_get(models=("llama3.2:latest", "nomic-embed-text:latest")),
        db_opener=plaintext_handle_opener(),
    )
    assert "llama3.2:3b" in report["ollama"]["required_models"]
    assert "llama3.2:3b" in report["ollama"]["missing_models"]
    assert report["runtime_ok"] is False


def test_preflight_tagged_model_present_is_ok(tmp_path):
    s = Settings(data_dir=tmp_path, embed_dim=768, llm_model="ollama/llama3.2:3b")
    report = run_preflight(
        s,
        http_get=make_http_get(models=("llama3.2:3b", "embeddinggemma:latest")),
        db_opener=plaintext_handle_opener(),
    )
    assert report["ollama"]["missing_models"] == []
    assert report["runtime_ok"] is True
