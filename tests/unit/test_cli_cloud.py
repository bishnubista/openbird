"""CLI cloud opt-in behavior [H3]: refusal, banner, and confirm paths."""

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


def test_data_stats_refuses_cloud_without_opt_in(monkeypatch):
    # A cloud-configured route with no opt-in must refuse non-interactively.
    monkeypatch.setenv("OPENBIRD_LLM_MODEL", "gpt-4o-mini")
    reset_settings_cache()
    res = CliRunner().invoke(cli.app, ["data", "stats"])
    assert res.exit_code == 2
    assert "CLOUD MODEL CONFIGURED" in res.output
    assert "gpt-4o-mini" in res.output
    assert "OPENBIRD_ALLOW_CLOUD=1" in res.output


def test_data_stats_proceeds_with_opt_in(monkeypatch):
    # With OPENBIRD_ALLOW_CLOUD=1 the factory builds the provider; the CLI prints
    # the CLOUD ACTIVE banner and the command runs.
    monkeypatch.setenv("OPENBIRD_EMBED_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("OPENBIRD_EMBED_DIM", "1536")
    monkeypatch.setenv("OPENBIRD_ALLOW_CLOUD", "1")
    reset_settings_cache()
    res = CliRunner().invoke(cli.app, ["data", "stats"])
    assert res.exit_code == 0, res.output
    assert "CLOUD ACTIVE" in res.output


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
    res = CliRunner().invoke(
        cli.app, ["capture", "--helper", "echo noop", "--allow-unsigned"]
    )
    assert res.exit_code == 2
    assert "CLOUD MODEL CONFIGURED" in res.output


def test_preflight_shows_cloud_blocked_row(monkeypatch):
    monkeypatch.setenv("OPENBIRD_LLM_MODEL", "claude-3-5-sonnet")
    reset_settings_cache()
    # Skip network probes; we only assert the cloud row renders.
    res = CliRunner().invoke(cli.app, ["preflight", "--no-ollama"])
    assert "cloud" in res.output
    assert "blocked" in res.output or "CLOUD" in res.output
