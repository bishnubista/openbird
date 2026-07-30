"""Tests for `openbird reindex`: cohort rebuild + atomic rollback.

Exercises the real CLI command against an on-disk SQLite store populated through
MemoryStore, swapping in a provider with a DIFFERENT cohort/dimension to prove the
vectors and cohort are rebuilt — and that a mid-reindex failure rolls back so the
old, searchable state survives.
"""

from __future__ import annotations

import hashlib
import math

import pytest
from typer.testing import CliRunner

from openbird import cli
from openbird.config import Settings, reset_settings_cache
from openbird.exit_codes import EXIT_REINDEX_REQUIRED
from openbird.memory.store import EmbeddingCohortMismatch, MemoryStore


class _FakeProvider:
    """Deterministic embedder with a configurable cohort + dimension."""

    def __init__(self, embed_dim: int = 768, tag: str = "a") -> None:
        self.embed_dim = embed_dim
        self.tag = tag

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def _vec(self, text: str) -> list[float]:
        h = int(hashlib.sha256((self.tag + text).encode("utf-8")).hexdigest(), 16)
        state = h % (2**61 - 1) or 1
        vec: list[float] = []
        for _ in range(self.embed_dim):
            state = (state * 6364136223846793005 + 1442695040888963407) % (2**64)
            vec.append((state / 2**64) - 0.5)
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def cohort_key(self) -> str:
        return f"fake:{self.tag}:{self.embed_dim}:deadbeef"


class _FailingProvider(_FakeProvider):
    """Embeds the first batch, then raises — to test rollback mid-reindex."""

    def __init__(self, embed_dim: int = 768, tag: str = "b", fail_after: int = 1) -> None:
        super().__init__(embed_dim=embed_dim, tag=tag)
        self.fail_after = fail_after
        self.batches = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.batches += 1
        if self.batches > self.fail_after:
            raise ConnectionError("ollama down mid-reindex")
        return super().embed(texts)


@pytest.fixture
def env(tmp_path, monkeypatch):
    """On-disk store + settings pointed at a temp data dir."""
    monkeypatch.setenv("OPENBIRD_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.delenv("OPENBIRD_OLLAMA_HOST", raising=False)
    reset_settings_cache()
    yield tmp_path
    reset_settings_cache()


def _populate(settings: Settings, provider) -> int:
    store = MemoryStore(settings=settings, provider=provider)
    try:
        for i in range(5):
            store.add_observation(
                f"unique chunk number {i} about openbird local memory",
                source="ingest",
                window=f"f{i}",
            )
        return store.stats()["vectors"]
    finally:
        store.close()


def _counts(settings: Settings, provider):
    store = MemoryStore(settings=settings, provider=provider)
    try:
        return store.stats()
    finally:
        store.close()


def test_reindex_rebuilds_cohort_and_dimension(env, monkeypatch):
    settings = Settings(data_dir=env, embed_dim=768)
    n = _populate(settings, _FakeProvider(embed_dim=768, tag="old"))
    assert n == 5

    # Switch to a provider with a different dimension + cohort.
    new_provider = _FakeProvider(embed_dim=512, tag="new")
    monkeypatch.setattr(cli, "_provider", lambda: new_provider)
    monkeypatch.setattr(cli, "get_settings", lambda: Settings(data_dir=env, embed_dim=512))

    res = CliRunner().invoke(cli.app, ["reindex", "--yes"])
    assert res.exit_code == 0, res.output
    assert "Reindexed 5" in res.output

    # The store now opens cleanly under the new provider (cohort adopted), and the
    # vector table is rebuilt at the new dimension with all 5 vectors.
    stats = _counts(Settings(data_dir=env, embed_dim=512), _FakeProvider(512, "new"))
    assert stats["vectors"] == 5
    assert stats["embed_dim"] == 512
    assert stats["cohort_key"] == new_provider.cohort_key()


def test_reindex_noop_when_cohort_matches(env, monkeypatch):
    settings = Settings(data_dir=env, embed_dim=768)
    _populate(settings, _FakeProvider(embed_dim=768, tag="same"))

    same = _FakeProvider(embed_dim=768, tag="same")
    monkeypatch.setattr(cli, "_provider", lambda: same)
    monkeypatch.setattr(cli, "get_settings", lambda: Settings(data_dir=env, embed_dim=768))

    res = CliRunner().invoke(cli.app, ["reindex", "--yes"])
    assert res.exit_code == 0, res.output
    assert "nothing to do" in res.output


def test_reindex_rolls_back_on_embed_failure(env, monkeypatch):
    settings = Settings(data_dir=env, embed_dim=768)
    old = _FakeProvider(embed_dim=768, tag="old")
    _populate(settings, old)
    before = _counts(settings, _FakeProvider(768, "old"))

    # Failing provider: same dim so the store can still open afterwards to verify.
    failing = _FailingProvider(embed_dim=768, tag="old", fail_after=1)
    monkeypatch.setattr(cli, "_provider", lambda: failing)
    monkeypatch.setattr(cli, "get_settings", lambda: Settings(data_dir=env, embed_dim=768))

    # force, since cohort tag matches; batch_size=2 so the 2nd batch fails.
    res = CliRunner().invoke(cli.app, ["reindex", "--yes", "--force", "--batch-size", "2"])
    assert res.exit_code == 1
    assert "rolled back" in res.output

    # Old vectors + cohort survive intact (rollback worked).
    after = _counts(Settings(data_dir=env, embed_dim=768), _FakeProvider(768, "old"))
    assert after["vectors"] == before["vectors"] == 5
    assert after["cohort_key"] == before["cohort_key"]


def test_reindex_non_interactive_refuses_without_yes(env, monkeypatch):
    settings = Settings(data_dir=env, embed_dim=768)
    _populate(settings, _FakeProvider(embed_dim=768, tag="old"))

    new_provider = _FakeProvider(embed_dim=512, tag="new")
    monkeypatch.setattr(cli, "_provider", lambda: new_provider)
    monkeypatch.setattr(cli, "get_settings", lambda: Settings(data_dir=env, embed_dim=512))

    # CliRunner has no TTY; without --yes the command must refuse, not hang.
    res = CliRunner().invoke(cli.app, ["reindex"])
    assert res.exit_code == 1
    assert "Refusing" in res.output


def test_reindex_empty_db_succeeds(env, monkeypatch):
    # An empty store (no chunks) reindexes to zero vectors without error.
    new_provider = _FakeProvider(embed_dim=512, tag="new")
    monkeypatch.setattr(cli, "_provider", lambda: new_provider)
    monkeypatch.setattr(cli, "get_settings", lambda: Settings(data_dir=env, embed_dim=512))
    # Create the empty DB first under the old cohort.
    _counts(Settings(data_dir=env, embed_dim=768), _FakeProvider(768, "old"))

    res = CliRunner().invoke(cli.app, ["reindex", "--yes"])
    assert res.exit_code == 0, res.output
    assert "Reindexed 0" in res.output


def test_reindex_fresh_uninitialized_db_succeeds(env, monkeypatch):
    # Regression: reindex on a data dir whose DB was NEVER opened by MemoryStore
    # (no schema applied) must NOT crash with "no such table: embedding_meta".
    new_provider = _FakeProvider(embed_dim=768, tag="new")
    monkeypatch.setattr(cli, "_provider", lambda: new_provider)
    monkeypatch.setattr(cli, "get_settings", lambda: Settings(data_dir=env, embed_dim=768))

    res = CliRunner().invoke(cli.app, ["reindex", "--yes"])
    assert res.exit_code == 0, res.output
    assert "Reindexed 0" in res.output


# --------------------------------------------------------------------------- #
# Embedding-cohort-mismatch UX (PR2a safety net)                              #
# --------------------------------------------------------------------------- #


def test_populated_store_with_changed_cohort_raises_mismatch(env):
    # Opening a populated store under a DIFFERENT embedding cohort must raise the
    # dedicated EmbeddingCohortMismatch (a ValueError subclass, for back-compat),
    # carrying both cohort keys so callers can render a precise recovery hint.
    settings = Settings(data_dir=env, embed_dim=768)
    old = _FakeProvider(embed_dim=768, tag="old")
    _populate(settings, old)

    new = _FakeProvider(embed_dim=768, tag="new")
    with pytest.raises(EmbeddingCohortMismatch) as ei:
        MemoryStore(settings=settings, provider=new)
    assert isinstance(ei.value, ValueError)
    assert ei.value.stored == old.cohort_key()
    assert ei.value.current == new.cohort_key()


def test_empty_store_tolerates_cohort_change(env):
    # An empty store (no vectors) must NOT raise — _record_cohort adopts the new
    # cohort and rebuilds the vec table. Guards against a false mismatch on a fresh
    # or post-purge store.
    settings = Settings(data_dir=env, embed_dim=768)
    MemoryStore(settings=settings, provider=_FakeProvider(768, "old")).close()
    # Reopen empty under a different cohort: tolerated.
    store = MemoryStore(settings=settings, provider=_FakeProvider(768, "new"))
    try:
        assert store.stats()["cohort_key"] == _FakeProvider(768, "new").cohort_key()
    finally:
        store.close()


def test_chat_cli_renders_reindex_hint_on_cohort_mismatch(env, monkeypatch):
    # The EMBED CLI path (chat -> _store) must convert a cohort mismatch into a
    # friendly `openbird reindex` hint + non-zero exit, NOT a raw traceback.
    settings = Settings(data_dir=env, embed_dim=768)
    _populate(settings, _FakeProvider(embed_dim=768, tag="old"))

    # Make the command's provider report a different cohort than the stored one.
    monkeypatch.setattr(cli, "_provider", lambda: _FakeProvider(embed_dim=768, tag="new"))

    result = CliRunner().invoke(cli.app, ["chat", "what did I work on?"])
    assert result.exit_code == EXIT_REINDEX_REQUIRED
    assert "openbird reindex" in result.output
    assert "Embedding model changed" in result.output
    # The exception must be HANDLED, not surfaced as an unhandled traceback.
    assert not isinstance(result.exception, EmbeddingCohortMismatch)


def test_routine_run_cli_renders_reindex_hint_on_cohort_mismatch(env, monkeypatch):
    # A second EMBED consumer of the shared _store() seam (routine run) must get the
    # same friendly reindex hint + non-zero exit on a cohort mismatch, proving the
    # recovery path is uniform across commands (not just chat).
    settings = Settings(data_dir=env, embed_dim=768)
    _populate(settings, _FakeProvider(embed_dim=768, tag="old"))

    monkeypatch.setattr(cli, "_provider", lambda: _FakeProvider(embed_dim=768, tag="new"))

    result = CliRunner().invoke(cli.app, ["routine", "run", "yesterday"])
    assert result.exit_code == EXIT_REINDEX_REQUIRED
    assert "openbird reindex" in result.output
    assert "Embedding model changed" in result.output
    assert not isinstance(result.exception, EmbeddingCohortMismatch)


# --------------------------------------------------------------------------- #
# Summary index (Phase E1): cohort counts BOTH vec tables; reindex rebuilds both
# --------------------------------------------------------------------------- #


def _seed_summary_index(settings: Settings, provider) -> str:
    """Store + index one block summary (span-backed, no chunks touched)."""
    store = MemoryStore(settings=settings, provider=provider)
    try:
        span_id = store.open_span(
            epoch_id="e", start_ts=1000.0, end_ts=1900.0, bundle_id="b",
            detail_tier=1,
        )
        saved = store.save_block_summary(
            local_date="2026-06-29", block_key="k1", block_fingerprint="f1",
            start_ts=1000.0, end_ts=1900.0, dominant_bundle="b", level=None,
            summary_text="Worked on the openbird summary retrieval index.",
            model="m", extractor_version="block-summary-v1",
            observation_ids=[], span_ids=[span_id],
        )
        store.index_summary(
            summary_kind="block", summary_id=saved["id"], fingerprint="f1",
            text=saved["summary_text"],
        )
        return saved["id"]
    finally:
        store.close()


def test_cohort_mismatch_raises_when_only_vec_summaries_populated(env):
    # Adoption must count vec_summaries too: a store whose ONLY vectors are
    # summary-index embeddings is still a populated cohort.
    settings = Settings(data_dir=env, embed_dim=768)
    _seed_summary_index(settings, _FakeProvider(embed_dim=768, tag="old"))

    with pytest.raises(EmbeddingCohortMismatch):
        MemoryStore(settings=settings, provider=_FakeProvider(768, "new"))


def test_reindex_rebuilds_both_vec_tables(env, monkeypatch):
    settings = Settings(data_dir=env, embed_dim=768)
    old = _FakeProvider(embed_dim=768, tag="old")
    assert _populate(settings, old) == 5
    _seed_summary_index(settings, old)

    new_provider = _FakeProvider(embed_dim=512, tag="new")
    monkeypatch.setattr(cli, "_provider", lambda: new_provider)
    monkeypatch.setattr(cli, "get_settings", lambda: Settings(data_dir=env, embed_dim=512))

    res = CliRunner().invoke(cli.app, ["reindex", "--yes"])
    assert res.exit_code == 0, res.output
    assert "1 summary" in res.output

    store = MemoryStore(
        settings=Settings(data_dir=env, embed_dim=512),
        provider=_FakeProvider(512, "new"),
    )
    try:
        stats = store.stats()
        assert stats["vectors"] == 5
        assert stats["summary_index_entries"] == 1
        vec = store.conn.execute("SELECT COUNT(*) c FROM vec_summaries").fetchone()["c"]
        assert int(vec) == 1  # re-embedded into the rebuilt FLOAT[512] table
        assert stats["cohort_key"] == new_provider.cohort_key()
        # The re-embedded summary is retrievable under the new cohort.
        hits = store.search_summaries("summary retrieval index", k=3)
        assert hits and hits[0]["summary_kind"] == "block"
    finally:
        store.close()


def test_reindex_rollback_preserves_summary_vectors(env, monkeypatch):
    settings = Settings(data_dir=env, embed_dim=768)
    old = _FakeProvider(embed_dim=768, tag="old")
    _populate(settings, old)
    _seed_summary_index(settings, old)

    failing = _FailingProvider(embed_dim=768, tag="old", fail_after=1)
    monkeypatch.setattr(cli, "_provider", lambda: failing)
    monkeypatch.setattr(cli, "get_settings", lambda: Settings(data_dir=env, embed_dim=768))

    res = CliRunner().invoke(cli.app, ["reindex", "--yes", "--force", "--batch-size", "2"])
    assert res.exit_code == 1
    assert "rolled back" in res.output

    store = MemoryStore(settings=settings, provider=_FakeProvider(768, "old"))
    try:
        vec = store.conn.execute("SELECT COUNT(*) c FROM vec_summaries").fetchone()["c"]
        assert int(vec) == 1  # the old summary vector survived the rollback
    finally:
        store.close()
