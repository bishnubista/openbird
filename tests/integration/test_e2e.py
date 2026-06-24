"""End-to-end integration tests for the OpenBird core path.

Two tiers:

* **Fake-provider E2E (CI-able, no Ollama):** ingest sample text into a real
  :class:`MemoryStore`, then run the real :class:`RAG` pipeline with a *fake*
  :class:`LLMProvider` so retrieval, context assembly, citation validation, and
  the CLI's ``ingest``/``chat``/``data purge`` commands are exercised without any
  network or model. The fake completer cites a real in-context source id, so the
  test asserts a grounded, occurrence-level citation comes back.

* **Ollama-gated E2E (skipped if unreachable):** the same ingest→chat round-trip
  through the *real* LiteLLM-backed provider and Ollama. Skipped automatically
  when Ollama is not reachable or the required models are missing, so the suite
  stays green on CI.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shlex
import stat
import sys

import pytest

from openbird.chat.rag import RAG
from openbird.config import Settings
from openbird.memory.store import MemoryStore
from openbird.preflight import check_ollama


# --------------------------------------------------------------------------- #
# Fakes                                                                        #
# --------------------------------------------------------------------------- #


class FakeProvider:
    """Deterministic embed + a completer that cites a real in-context source.

    ``embed`` returns stable, L2-normalized ``embed_dim`` vectors derived from a
    hash of the text (identical text → identical vector). ``complete`` ignores
    the model and, when handed the RAG response schema, returns a JSON object
    that answers from — and cites — the FIRST ``[source_id: <id>]`` present in the
    assembled context. This proves the citation-validation path end-to-end
    without a real model.
    """

    def __init__(self, embed_dim: int = 768) -> None:
        self.embed_dim = embed_dim

    # -- embeddings -----------------------------------------------------------

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def _vec(self, text: str) -> list[float]:
        h = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)
        state = h % (2**61 - 1) or 1
        vec: list[float] = []
        for _ in range(self.embed_dim):
            state = (state * 6364136223846793005 + 1442695040888963407) % (2**64)
            vec.append((state / 2**64) - 0.5)
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    # -- completion -----------------------------------------------------------

    def complete(self, messages: list[dict], *, json_schema: dict | None = None):
        """Answer from + cite the first source id found in the user message."""
        user = next((m for m in messages if m.get("role") == "user"), None)
        content = user["content"] if user else ""
        match = re.search(r"\[source_id:\s*([^\]]+)\]", content)
        source_id = match.group(1).strip() if match else ""

        if json_schema is not None:
            return {
                "answer": "OpenBird keeps your data local and uses Ollama by default.",
                "citations": [source_id] if source_id else [],
            }
        return "OpenBird keeps your data local and uses Ollama by default."

    def cohort_key(self) -> str:
        return f"fake:fake-embed:{self.embed_dim}:deadbeef"


SAMPLE_TEXT = (
    "OpenBird is an open-source, local-first personal memory app for macOS. "
    "Your captured data stays on-device and the default LLM is Ollama, with "
    "cloud opt-in available through LiteLLM."
)


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(data_dir=tmp_path, embed_dim=768)


@pytest.fixture
def store(settings) -> MemoryStore:
    s = MemoryStore(db_path=str(settings.data_dir / "e2e.db"), settings=settings,
                    provider=FakeProvider(embed_dim=768))
    yield s
    s.close()


# --------------------------------------------------------------------------- #
# Fake-provider E2E                                                            #
# --------------------------------------------------------------------------- #


def test_ingest_then_chat_returns_cited_answer(store):
    """Ingest sample text, then RAG retrieves it and returns a valid citation."""
    obs = store.add_observation(
        SAMPLE_TEXT,
        app="Notes",
        window="OpenBird overview",
        source="ingest",
    )

    rag = RAG(store, store.provider)
    result = rag.answer("What is OpenBird and which LLM does it use?", k=5)

    assert result.answer  # non-empty grounded answer
    assert result.used_hits, "retrieval should surface the ingested chunk"
    assert result.citations, "answer should carry an occurrence-level citation"

    cite = result.citations[0]
    # The citation must resolve to the REAL observation we just stored.
    assert cite.observation_id == obs.id
    assert cite.app == "Notes"
    assert cite.window == "OpenBird overview"
    assert cite.snippet  # a non-empty snippet from the cited chunk


def test_chat_with_empty_memory_has_no_citations(store):
    """With nothing ingested, chat returns the no-memory answer and no citations."""
    rag = RAG(store, store.provider)
    result = rag.answer("anything at all?")
    assert result.citations == []
    assert result.used_hits == []


def test_cli_ingest_chat_purge_roundtrip(tmp_path, monkeypatch):
    """Drive the CLI end-to-end (ingest -> chat -> data purge) with a fake provider.

    Patches the CLI's provider/store factories to use the fake provider and the
    tmp data dir, then invokes the Typer commands through ``CliRunner`` so the
    real command wiring (argument parsing, store lifecycle, citation printing,
    cascade delete) is exercised without Ollama.
    """
    from typer.testing import CliRunner

    import openbird.cli as cli

    db_path = str(tmp_path / "cli.db")
    settings = Settings(data_dir=tmp_path, embed_dim=768)
    provider = FakeProvider(embed_dim=768)

    def fake_get_settings() -> Settings:
        return settings

    def fake_provider():
        return provider

    def fake_store(*, provider=None):
        return MemoryStore(db_path=db_path, settings=settings,
                           provider=provider or FakeProvider(embed_dim=768))

    monkeypatch.setattr(cli, "get_settings", fake_get_settings)
    monkeypatch.setattr(cli, "_provider", fake_provider)
    monkeypatch.setattr(cli, "_store", fake_store)
    # purge/stats use the maintenance store (no cloud gate); same fake DB here.
    monkeypatch.setattr(
        cli, "_store_maintenance",
        lambda: MemoryStore(db_path=db_path, settings=settings,
                            provider=FakeProvider(embed_dim=768)),
    )

    sample = tmp_path / "sample.txt"
    sample.write_text(SAMPLE_TEXT, encoding="utf-8")

    runner = CliRunner()

    # ingest
    res = runner.invoke(cli.app, ["ingest", str(sample)])
    assert res.exit_code == 0, res.output
    assert "Ingested 1" in res.output

    # chat -> grounded answer + a printed source line
    res = runner.invoke(cli.app, ["chat", "What is OpenBird?"])
    assert res.exit_code == 0, res.output
    assert "Sources" in res.output
    # A real citation entry must be emitted (numbered source line) AND it must
    # point at the ingested file's provenance — not just the "Sources" header.
    assert "[1]" in res.output, res.output
    assert "sample.txt" in res.output, res.output

    # data stats reflect the ingested observation
    res = runner.invoke(cli.app, ["data", "stats"])
    assert res.exit_code == 0, res.output
    stats = json.loads(res.output)
    assert stats["observations"] == 1

    export_path = tmp_path / "memory-export.jsonl"
    res = runner.invoke(cli.app, ["data", "export", "--output", str(export_path), "--yes"])
    assert res.exit_code == 0, res.output
    assert "EXPORT WARNING" in res.output
    exported = [json.loads(line) for line in export_path.read_text().splitlines()]
    assert len(exported) == 1
    assert "OpenBird is an open-source, local-first personal memory app" in exported[0]["text"]
    assert stat.S_IMODE(export_path.stat().st_mode) == 0o600

    export_path.chmod(0o644)
    res = runner.invoke(
        cli.app,
        ["data", "export", "--output", str(export_path), "--overwrite", "--yes"],
    )
    assert res.exit_code == 0, res.output
    assert stat.S_IMODE(export_path.stat().st_mode) == 0o600

    # data purge --all (cascade delete) with confirmation skipped
    res = runner.invoke(cli.app, ["data", "purge", "--all", "--yes"])
    assert res.exit_code == 0, res.output
    assert "Deleted 1" in res.output

    res = runner.invoke(cli.app, ["data", "stats"])
    stats = json.loads(res.output)
    assert stats["observations"] == 0


def test_cli_prune_and_vacuum_roundtrip(tmp_path, monkeypatch):
    """Drive `data prune --older-than` + `data vacuum` through the CLI (H10).

    Ingests two files with controlled timestamps via direct store writes, prunes
    the old one through the CLI, and vacuums — proving the retention + reclaim
    commands are wired and report sane output.
    """
    import time

    from typer.testing import CliRunner

    import openbird.cli as cli

    db_path = str(tmp_path / "cli_prune.db")
    settings = Settings(data_dir=tmp_path, embed_dim=768)

    def fake_store(*, provider=None):
        return MemoryStore(db_path=db_path, settings=settings,
                           provider=provider or FakeProvider(embed_dim=768))

    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(cli, "_store", fake_store)
    # prune + stats are maintenance ops that go through _store_maintenance (no
    # cloud gate); point it at the same test DB so the roundtrip is observable.
    monkeypatch.setattr(cli, "_store_maintenance", lambda: fake_store())

    now = time.time()
    seed = fake_store()
    seed.add_observation("ancient memory", source="t", ts=now - 100 * 86400)
    seed.add_observation("fresh memory", source="t", ts=now - 1 * 86400)
    seed.close()

    runner = CliRunner()

    # prune everything older than 30 days (the ancient one).
    res = runner.invoke(cli.app, ["data", "prune", "--older-than", "30d", "--yes"])
    assert res.exit_code == 0, res.output
    assert "Pruned 1" in res.output

    res = runner.invoke(cli.app, ["data", "stats"])
    stats = json.loads(res.output)
    assert stats["observations"] == 1

    # vacuum reclaims; emits JSON stats.
    res = runner.invoke(cli.app, ["data", "vacuum"])
    assert res.exit_code == 0, res.output
    assert "Vacuumed" in res.output
    payload = json.loads(res.output.split("\n", 1)[1])
    assert payload["bytes_after"] <= payload["bytes_before"]


def test_cli_capture_then_chat_roundtrip_with_fake_helper(tmp_path, monkeypatch):
    """Drive the product-shaped path: capture helper -> store -> chat -> purge.

    This is the closest CI-safe stand-in for the OpenBird workflow:
    a helper emits active-window text, the capture daemon applies allowlist and
    redaction policy, the real store indexes it, and the chat CLI retrieves it
    with a validated citation. The helper is a fake JSON-lines subprocess, so no
    macOS Accessibility permission, signed bundle, Ollama, or network is needed.
    """
    from typer.testing import CliRunner

    import openbird.capture.cli as capture_cli
    import openbird.cli as cli
    from openbird.memory import store as store_mod

    db_path = str(tmp_path / "capture-chat.db")
    settings = Settings(
        data_dir=tmp_path,
        embed_dim=768,
        allowlist=["com.apple.mail"],
    )
    provider = FakeProvider(embed_dim=768)

    def fake_get_settings() -> Settings:
        return settings

    def fake_provider():
        return provider

    def fake_store(*, provider=None, settings=settings):
        # Mirror the real _store signature: capture now passes its resolved settings
        # through _store(provider=..., settings=...). Default to the closure settings.
        return MemoryStore(
            db_path=db_path,
            settings=settings,
            provider=provider or FakeProvider(embed_dim=768),
        )

    monkeypatch.setattr(capture_cli, "get_settings", fake_get_settings)
    monkeypatch.setattr(
        store_mod,
        "MemoryStore",
        lambda *, settings, provider=provider: MemoryStore(
            db_path=db_path,
            settings=settings,
            provider=provider,
        ),
    )
    monkeypatch.setattr(cli, "get_settings", fake_get_settings)
    monkeypatch.setattr(cli, "_provider", fake_provider)
    monkeypatch.setattr(cli, "_store", fake_store)
    # purge/stats use the maintenance store (no cloud gate); use the fake here too
    # so its cohort matches the FakeProvider-built store.
    monkeypatch.setattr(
        cli, "_store_maintenance",
        lambda: MemoryStore(db_path=db_path, settings=settings, provider=provider),
    )

    events = [
        {
            "app": "com.apple.mail",
            "window": "OpenBird planning note",
            "url": "https://example.com/note?access_token=secret#fragment",
            "text": (
                "OpenBird planning note: the local-first assistant keeps "
                "screen text in local SQLite memory and uses Ollama by default."
            ),
            "ts": 1_700_000_000.0,
        },
        {
            "app": "com.unknown.private",
            "window": "Should not capture",
            "text": "This disallowed app text must never enter memory.",
            "ts": 1_700_000_001.0,
        },
    ]
    emitter = (
        "import json,sys\n"
        f"events={events!r}\n"
        "for e in events:\n"
        "    sys.stdout.write(json.dumps(e) + '\\n')\n"
        "sys.stdout.flush()\n"
    )
    helper = f"{sys.executable} -c {shlex.quote(emitter)}"

    runner = CliRunner()

    res = runner.invoke(
        cli.app,
        ["capture", "--helper", helper, "--allow-unsigned", "--max-events", "2"],
    )
    assert res.exit_code == 0, res.output
    assert "received=2" in res.output
    assert "ingested=1" in res.output
    assert "rejected=1" in res.output

    res = runner.invoke(cli.app, ["data", "stats"])
    assert res.exit_code == 0, res.output
    stats = json.loads(res.output)
    assert stats["observations"] == 1

    check_store = MemoryStore(db_path=db_path, settings=settings, provider=provider)
    try:
        observations = check_store.time_range(0, 2_000_000_000)
    finally:
        check_store.close()
    assert len(observations) == 1
    assert observations[0].url == "https://example.com/note"
    assert "access_token" not in (observations[0].url or "")
    assert "secret" not in (observations[0].url or "")

    res = runner.invoke(
        cli.app,
        [
            "chat",
            "Which product keeps screen text in local SQLite memory and uses Ollama?",
            "--no-semantic",
        ],
    )
    assert res.exit_code == 0, res.output
    assert "OpenBird keeps your data local and uses Ollama by default." in res.output
    assert "Sources" in res.output
    assert "[1]" in res.output
    assert "com.apple.mail / OpenBird planning note" in res.output
    assert "local SQLite memory" in res.output
    assert "disallowed app text" not in res.output

    res = runner.invoke(cli.app, ["data", "purge", "--all", "--yes"])
    assert res.exit_code == 0, res.output
    assert "Deleted 1" in res.output


def test_cli_routine_list_and_meeting_stub(monkeypatch):
    """`routine list` and the `meeting` stub run without any services."""
    from typer.testing import CliRunner

    import openbird.cli as cli

    runner = CliRunner()

    res = runner.invoke(cli.app, ["routine", "list"])
    assert res.exit_code == 0, res.output
    assert "daily-briefing" in res.output
    assert "weekly-summary" in res.output

    res = runner.invoke(cli.app, ["meeting"])
    assert res.exit_code == 0, res.output
    assert "meetings" in res.output.lower()


def test_cli_preflight_runs_without_ollama(monkeypatch, tmp_path):
    """`preflight --no-ollama` produces a report and a clean exit path."""
    from typer.testing import CliRunner

    import openbird.cli as cli

    settings = Settings(data_dir=tmp_path, embed_dim=768)
    monkeypatch.setattr(cli, "get_settings", lambda: settings)

    runner = CliRunner()
    res = runner.invoke(cli.app, ["preflight", "--no-ollama", "--json"])
    # Exit code is 1 when runtime-not-ready (ollama skipped) — that's expected;
    # what matters is it produced a parseable report and did not crash.
    report = json.loads(res.output)
    assert "sqlite" in report
    assert "encryption" in report
    assert report["ollama"]["reachable"] == "unknown"


# --------------------------------------------------------------------------- #
# Ollama-gated E2E                                                             #
# --------------------------------------------------------------------------- #


def _ollama_ready() -> bool:
    """True iff Ollama is reachable with the EXACT default-route models pulled.

    Resolves the RAM-tiered generation tag (e.g. ``qwen3:8b`` on a 32 GB host) plus
    the default embedder and checks those specific tags, rather than the family-level
    ``_REQUIRED_MODELS`` fallback — otherwise a host with only the other tier pulled
    would run the test and then fail when the runtime requests the exact tag.
    """
    try:
        from openbird.config import Settings, _default_llm_model, ollama_bare_model

        embed_default = Settings.__dataclass_fields__["embed_model"].default
        required = tuple(
            bare
            for bare in (
                ollama_bare_model(_default_llm_model()),
                ollama_bare_model(embed_default),
            )
            if bare
        )
        info = check_ollama(required_models=required, timeout=2.0)
    except ImportError:
        # Only a missing/renamed import means "can't probe -> treat as unavailable".
        # check_ollama never raises (it reports reachable=False), so any other
        # exception is a real regression and must surface, not silently skip.
        return False
    return bool(info.get("reachable")) and not info.get("missing_models")


@pytest.mark.skipif(not _ollama_ready(), reason="Ollama unreachable or models missing")
def test_ingest_then_chat_with_real_ollama(tmp_path):
    """Real round-trip through LiteLLM + Ollama: ingest -> embed -> chat -> cite.

    Skipped automatically unless Ollama is reachable and the required models are
    present, so this never breaks CI. When it runs it proves the embedding
    dimension guard, vector indexing, and grounded chat all work against the live
    local stack.
    """
    from openbird.llm.provider import LLMProvider

    settings = Settings(data_dir=tmp_path, embed_dim=768)
    provider = LLMProvider(settings)
    store = MemoryStore(db_path=str(tmp_path / "ollama.db"), settings=settings,
                        provider=provider)
    try:
        obs = store.add_observation(
            SAMPLE_TEXT, app="Notes", window="OpenBird overview", source="ingest"
        )

        hits = store.search("which LLM does OpenBird use by default?", k=5)
        assert hits, "real embeddings should retrieve the ingested chunk"

        rag = RAG(store, provider)
        result = rag.answer("Which LLM does OpenBird use by default?", k=5)
        assert result.answer
        # With short positional source labels the local model reliably grounds its
        # answer: assert a real, validated citation is produced (this is the
        # cited-answer behavior README promises), and every citation resolves to
        # the one real observation — no hallucinated ids survive.
        assert result.grounded, "expected a grounded, cited answer over one observation"
        assert result.citations
        for c in result.citations:
            assert c.observation_id == obs.id
    finally:
        store.close()
