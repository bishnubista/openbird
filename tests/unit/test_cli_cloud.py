"""CLI cloud opt-in behavior: refusal, banner, and confirm paths."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from openbird import cli
from openbird.config import Settings, reset_settings_cache


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENBIRD_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.delenv("OPENBIRD_OLLAMA_HOST", raising=False)
    reset_settings_cache()
    yield
    reset_settings_cache()


def test_ingest_refuses_cloud_without_opt_in(monkeypatch, tmp_path):
    # ingest EMBEDS, so a cloud route with no opt-in must refuse non-interactively.
    f = tmp_path / "doc.txt"
    f.write_text("hello world")
    monkeypatch.setenv("OPENBIRD_LLM_MODEL", "gpt-4o-mini")
    reset_settings_cache()
    res = CliRunner().invoke(cli.app, ["ingest", str(f)])
    assert res.exit_code == 2
    assert "CLOUD MODEL CONFIGURED" in res.output
    assert "gpt-4o-mini" in res.output
    assert "OPENBIRD_ALLOW_CLOUD=1" in res.output


def test_data_purge_not_gated_by_cloud(monkeypatch):
    # Purge never embeds — it must NOT be blocked behind cloud opt-in (privacy
    # path). A cloud model configured without opt-in still allows deletion.
    monkeypatch.setenv("OPENBIRD_EMBED_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("OPENBIRD_EMBED_DIM", "1536")
    reset_settings_cache()
    res = CliRunner().invoke(cli.app, ["data", "purge", "--all", "--yes"])
    assert res.exit_code == 0, res.output
    assert "CLOUD MODEL CONFIGURED" not in res.output
    assert "Deleted" in res.output


def test_purge_works_after_embed_model_switch(monkeypatch, tmp_path):
    # Regression: a populated store + a switched embed model/dim must still purge
    # (the maintenance path must not hit the cohort-mismatch guard).
    from openbird.memory.store import MemoryStore

    class _FakeP:
        def __init__(self, dim=768, tag="a"):
            self.embed_dim = dim
            self.tag = tag

        def embed(self, texts):
            return [[0.1] * self.embed_dim for _ in texts]

        def cohort_key(self):
            return f"fake:{self.tag}:{self.embed_dim}:x"

    # Populate under cohort "a" / dim 768.
    s = Settings(data_dir=tmp_path, embed_dim=768)
    st = MemoryStore(settings=s, provider=_FakeP(768, "a"))
    st.add_observation("switch test", source="ingest", window="t")
    st.close()

    # Now the configured model is a different cohort/dim.
    monkeypatch.setenv("OPENBIRD_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OPENBIRD_EMBED_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("OPENBIRD_EMBED_DIM", "1536")
    reset_settings_cache()

    res = CliRunner().invoke(cli.app, ["data", "purge", "--all", "--yes"])
    assert res.exit_code == 0, res.output
    assert "Deleted 1" in res.output


def test_data_stats_not_gated_by_cloud(monkeypatch):
    # Stats only counts rows — also gate-free (and no banner).
    monkeypatch.setenv("OPENBIRD_EMBED_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("OPENBIRD_EMBED_DIM", "1536")
    reset_settings_cache()
    res = CliRunner().invoke(cli.app, ["data", "stats"])
    assert res.exit_code == 0, res.output
    assert "CLOUD MODEL CONFIGURED" not in res.output


def test_local_default_has_no_cloud_banner():
    res = CliRunner().invoke(cli.app, ["data", "stats"])
    assert res.exit_code == 0, res.output
    assert "CLOUD ACTIVE" not in res.output
    assert "CLOUD MODEL CONFIGURED" not in res.output


def test_capture_refuses_cloud_without_opt_in(monkeypatch):
    # Regression: capture sends screen text to the embed model, so it must go
    # through the cloud opt-in gate + banner like every other store command.
    monkeypatch.setenv("OPENBIRD_EMBED_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("OPENBIRD_EMBED_DIM", "1536")
    reset_settings_cache()
    res = CliRunner().invoke(cli.app, ["capture", "--helper", "echo noop", "--allow-unsigned"])
    assert res.exit_code == 2
    assert "CLOUD MODEL CONFIGURED" in res.output


def test_preflight_shows_cloud_blocked_row(monkeypatch):
    monkeypatch.setenv("OPENBIRD_LLM_MODEL", "claude-3-5-sonnet")
    reset_settings_cache()
    # Skip network probes; we only assert the cloud row renders.
    res = CliRunner().invoke(cli.app, ["preflight", "--no-ollama"])
    assert "cloud" in res.output
    assert "blocked" in res.output or "CLOUD" in res.output
