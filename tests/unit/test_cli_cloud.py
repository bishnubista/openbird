"""CLI cloud opt-in behavior: refusal, banner, and confirm paths."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from openbird import cli
from openbird.config import Settings, reset_settings_cache
from openbird.memory.store import MemoryStore
from tests.unit.conftest import FakeProvider


REASONING_LEDGER_FIELDS = {
    "created_at",
    "feature",
    "packet_route",
    "reasoning_route",
    "egress",
    "route_class",
    "provider_family",
    "model",
    "packet_hash",
    "packet_bytes",
    "selected_source_count",
    "citation_count",
    "excluded_observations",
    "excluded_by",
    "outcome",
    "error_kind",
    "deletion_caveat",
}


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
    from openbird.config import Settings
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


def test_data_reasoning_ledger_json_is_redacted_and_not_gated_by_cloud(
    monkeypatch, tmp_path
):
    settings = Settings(data_dir=tmp_path, embed_dim=64)
    store = MemoryStore(settings=settings, provider=FakeProvider(embed_dim=64))
    try:
        store.record_reasoning_send(
            feature="deep_brain.ask",
            packet_route="deep_brain.preview",
            reasoning_route="cloud_reasoning_active",
            egress="active_model_route",
            route_class="third-party-cloud",
            provider_family="openai",
            model="gpt-5.5",
            packet_hash="a" * 64,
            packet_bytes=321,
            selected_source_count=2,
            citation_count=1,
            excluded_observations=7,
            excluded_by={"app": 2, "source": 1, "observation_id": 4, "obs-secret-123": 9},
            outcome="error",
            error_kind="RuntimeError: secret provider text",
        )
    finally:
        store.close()

    monkeypatch.setenv("OPENBIRD_EMBED_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("OPENBIRD_EMBED_DIM", "1536")
    reset_settings_cache()

    try:
        res = CliRunner().invoke(cli.app, ["data", "reasoning-ledger", "--json"])

        assert res.exit_code == 0, res.output
        assert "CLOUD MODEL CONFIGURED" not in res.output
        payload = json.loads(res.stdout)
        assert set(payload) == {"rows"}
        assert len(payload["rows"]) == 1
        row = payload["rows"][0]
        assert set(row) == REASONING_LEDGER_FIELDS
        assert row["feature"] == "deep_brain.ask"
        assert row["packet_hash"] == "a" * 64
        assert row["error_kind"] == "error"
        assert row["excluded_observations"] == 7
        assert row["excluded_by"] == {"app": 2, "observation_id": 4, "source": 1}
        serialized = json.dumps(payload, sort_keys=True)
        assert "secret provider text" not in serialized
        assert "raw question" not in serialized
        assert "generated answer" not in serialized
        assert "packet json" not in serialized
        assert "obs-secret-123" not in serialized

        table = CliRunner().invoke(cli.app, ["data", "reasoning-ledger"])
        assert table.exit_code == 0, table.output
        assert "Reasoning Send Ledger" in table.output
        assert "No remote reasoning sends recorded" not in table.output
        assert "secret provider text" not in table.output
    finally:
        reset_settings_cache()


def test_data_reasoning_ledger_text_empty_and_limit_validation():
    empty = CliRunner().invoke(cli.app, ["data", "reasoning-ledger"])
    assert empty.exit_code == 0, empty.output
    assert "No remote reasoning sends recorded" in empty.output

    invalid = CliRunner().invoke(cli.app, ["data", "reasoning-ledger", "--limit", "0"])
    assert invalid.exit_code != 0


def test_data_vacuum_not_gated_by_cloud(monkeypatch):
    # Vacuum never embeds — it must stay available for local cleanup even when
    # the configured embed model is cloud-backed and not opted in.
    monkeypatch.setenv("OPENBIRD_EMBED_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("OPENBIRD_EMBED_DIM", "1536")
    reset_settings_cache()
    res = CliRunner().invoke(cli.app, ["data", "vacuum"])
    assert res.exit_code == 0, res.output
    assert "CLOUD MODEL CONFIGURED" not in res.output
    assert "Vacuumed" in res.output


def test_data_export_not_gated_by_cloud(monkeypatch, tmp_path):
    # Export is explicit local egress to a user-chosen file; it must warn, but it
    # must not construct a cloud provider or block cleanup/export workflows.
    monkeypatch.setenv("OPENBIRD_EMBED_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("OPENBIRD_EMBED_DIM", "1536")
    reset_settings_cache()
    out = tmp_path / "memory.jsonl"
    res = CliRunner().invoke(
        cli.app, ["data", "export", "--output", str(out), "--yes"]
    )
    assert res.exit_code == 0, res.output
    assert "CLOUD MODEL CONFIGURED" not in res.output
    assert "EXPORT WARNING" in res.output
    assert out.exists()


def test_vacuum_works_after_embed_model_switch(monkeypatch, tmp_path):
    # Regression: a populated store + a switched embed model/dim must still
    # vacuum. Maintenance must not hit the cohort-mismatch guard.
    from openbird.memory.store import MemoryStore

    class _FakeP:
        def __init__(self, dim=768, tag="a"):
            self.embed_dim = dim
            self.tag = tag

        def embed(self, texts):
            return [[0.1] * self.embed_dim for _ in texts]

        def cohort_key(self):
            return f"fake:{self.tag}:{self.embed_dim}:x"

    s = Settings(data_dir=tmp_path, embed_dim=768)
    st = MemoryStore(settings=s, provider=_FakeP(768, "a"))
    st.add_observation("switch test", source="ingest", window="t")
    st.close()

    monkeypatch.setenv("OPENBIRD_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OPENBIRD_EMBED_MODEL", "ollama/other-embed")
    monkeypatch.setenv("OPENBIRD_EMBED_DIM", "1536")
    reset_settings_cache()

    res = CliRunner().invoke(cli.app, ["data", "vacuum"])
    assert res.exit_code == 0, res.output
    assert "Embedding cohort mismatch" not in res.output
    assert "Vacuumed" in res.output


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
