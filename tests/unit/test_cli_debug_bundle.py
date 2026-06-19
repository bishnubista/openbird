"""debug-bundle: emits diagnostics without leaking captured content or secrets."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from openbird import cli
from openbird.config import reset_settings_cache


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENBIRD_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.delenv("OPENBIRD_OLLAMA_HOST", raising=False)
    reset_settings_cache()
    yield
    reset_settings_cache()


def test_debug_bundle_json_has_expected_shape(monkeypatch):
    res = CliRunner().invoke(cli.app, ["debug-bundle", "--no-ollama", "--json"])
    assert res.exit_code == 0, res.output
    bundle = json.loads(res.stdout)
    for key in ("openbird_version", "python", "platform", "extras", "config", "preflight"):
        assert key in bundle
    # Home directory is redacted out of paths.
    assert "data_dir" in bundle["config"]
    assert "/Users/" not in json.dumps(bundle)


def test_debug_bundle_never_leaks_allow_block_patterns(monkeypatch):
    # The allow/block lists fingerprint which apps/sites the user runs, so a
    # shareable bundle must reduce them to counts only.
    monkeypatch.setenv("OPENBIRD_ALLOWLIST", "SecretBankApp,PrivateChatXYZ")
    reset_settings_cache()

    res = CliRunner().invoke(cli.app, ["debug-bundle", "--no-ollama", "--json"])
    assert res.exit_code == 0, res.output

    assert "SecretBankApp" not in res.stdout
    assert "PrivateChatXYZ" not in res.stdout

    bundle = json.loads(res.stdout)
    assert bundle["config"]["allowlist_entries"] == 2
    assert bundle["preflight"]["privacy"]["allowlist_entries"] == 2
    # The raw list keys must not survive into the redacted privacy block.
    assert "allowlist" not in bundle["preflight"]["privacy"]
