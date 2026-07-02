"""Unit tests for the deterministic entity-ledger aggregation pass (Phase E2).

Real MemoryStore + deterministic fake embeddings (conftest) — no network. The
pass is STRUCTURALLY LLM-free (no provider argument), so every behavior here
is exact and repeatable: regex mining, item anchoring, cursors, rehydration,
resolution, dormancy.
"""

from __future__ import annotations

import datetime as _dt

import pytest

from openbird.config import Settings
from openbird.entities import run_entity_aggregation
from openbird.memory.store import MemoryStore, entity_id_for

from tests.unit.conftest import FakeProvider

# A fixed, realistic "now" (mid-2025) so local-date derivation stays sane.
NOW = 1_750_000_000.0
DAY = 86_400.0


@pytest.fixture
def store(tmp_path) -> MemoryStore:
    s = MemoryStore(
        db_path=":memory:",
        settings=Settings(data_dir=tmp_path, embed_dim=64),
        provider=FakeProvider(embed_dim=64),
    )
    yield s
    s.close()


def _settings(tmp_path, **kw) -> Settings:
    return Settings(data_dir=tmp_path, embed_dim=64, **kw)


def _run(store, tmp_path, *, now=NOW, **kw) -> dict:
    return run_entity_aggregation(store, now=now, settings=_settings(tmp_path, **kw))


def _evidence_details(store, entity_id, kind=None) -> set[tuple[str, str]]:
    rows = store.entity_evidence_for(entity_id)
    return {
        (r["kind"], r["detail"]) for r in rows if kind is None or r["kind"] == kind
    }


# -- no provider, ever -------------------------------------------------------------


def test_signature_is_structurally_llm_free():
    import inspect

    params = inspect.signature(run_entity_aggregation).parameters
    assert "provider" not in params
    assert list(params) == ["store", "now", "settings"]


# -- repo/domain derivation ----------------------------------------------------------


def test_repo_and_domain_entities_derived_deterministically(store, tmp_path):
    obs = store.add_observation(
        "Reviewing github.com/bbista/openbird changes",
        source="capture",
        ts=NOW - 3600.0,
        window="openbird — Pull requests",
        url="https://github.com/bbista/openbird",
    )
    counts = _run(store, tmp_path)
    assert counts["entities"] == 2  # the repo + the github.com domain

    repo = store.get_entity(entity_id_for("repo", "bbista/openbird"))
    assert repo is not None and repo["kind"] == "repo"
    assert repo["last_seen_source_kind"] == "observation"
    assert repo["last_seen_source_id"] == obs.id

    domain = store.get_entity(entity_id_for("domain", "github.com"))
    assert domain is not None and domain["kind"] == "domain"


def test_domain_entities_from_span_url_hosts(store, tmp_path):
    span_id = store.open_span(
        epoch_id="e", start_ts=NOW - 7200.0, end_ts=NOW - 3600.0,
        bundle_id="com.google.chrome", detail_tier=1, url_host="docs.rs",
    )
    _run(store, tmp_path)
    domain = store.get_entity(entity_id_for("domain", "docs.rs"))
    assert domain is not None
    assert domain["last_seen_source_kind"] == "span"
    assert domain["last_seen_source_id"] == span_id


def test_idempotent_rerun_row_counts_stable(store, tmp_path):
    store.add_observation(
        "Merged bbista/openbird pull request",
        source="capture",
        ts=NOW - 3600.0,
        window="Merge pull request #12 · bbista/openbird",
        url="https://github.com/bbista/openbird/pull/12",
    )
    first = _run(store, tmp_path)
    assert first["evidence"] == 1
    stats = store.stats()

    second = _run(store, tmp_path)  # overlap re-scan re-mines; dedup absorbs
    assert second["evidence"] == 0
    assert store.stats()["entities"] == stats["entities"]
    assert store.stats()["entity_evidence"] == stats["entity_evidence"]


# -- completion mining: GitHub ---------------------------------------------------------


def test_pr_merged_url_is_item_status_anywhere(store, tmp_path):
    store.add_observation(
        "This pull request was successfully merged by bbista",
        source="capture",
        ts=NOW - 3600.0,
        window="Add entity ledger by bbista · Pull Request #12",
        url="https://github.com/bbista/openbird/pull/12",
    )
    _run(store, tmp_path)
    repo_id = entity_id_for("repo", "bbista/openbird")
    assert ("pr_merged", "github:bbista/openbird#12") in _evidence_details(
        store, repo_id
    )


def test_mixed_list_mints_evidence_for_the_merged_item_only(store, tmp_path):
    filler = "review queue item " * 20  # >120 chars between the two items
    store.add_observation(
        f"github.com/bbista/openbird/pull/1 Open {filler} "
        "github.com/bbista/openbird/pull/2 Merged",
        source="capture",
        ts=NOW - 3600.0,
        window="Pull requests",
        url="https://github.com/bbista/openbird/pulls",
    )
    _run(store, tmp_path)
    repo_id = entity_id_for("repo", "bbista/openbird")
    merged = _evidence_details(store, repo_id, kind="pr_merged")
    assert merged == {("pr_merged", "github:bbista/openbird#2")}


def test_merged_prose_without_item_ref_does_not_fire(store, tmp_path):
    store.add_observation(
        "I merged the two branches locally and closed the tab afterwards.",
        source="capture",
        ts=NOW - 3600.0,
        window="Terminal",
    )
    counts = _run(store, tmp_path)
    assert counts["evidence"] == 0
    assert store.stats()["entity_evidence"] == 0


def test_issue_closed_via_url_item(store, tmp_path):
    store.add_observation(
        "bbista closed this issue yesterday",
        source="capture",
        ts=NOW - 3600.0,
        window="Capture bug · Issue #9 · bbista/openbird",
        url="https://github.com/bbista/openbird/issues/9",
    )
    _run(store, tmp_path)
    repo_id = entity_id_for("repo", "bbista/openbird")
    assert ("ticket_closed", "github:bbista/openbird#9") in _evidence_details(
        store, repo_id
    )


# -- completion mining: Linear/Jira -----------------------------------------------------


def test_jira_ticket_closed_requires_host_key_and_word(store, tmp_path):
    store.add_observation(
        "ENG-42 moved to Done",
        source="capture",
        ts=NOW - 3600.0,
        window="ENG-42 — Jira",
        url="https://acme.atlassian.net/browse/ENG-42",
    )
    _run(store, tmp_path)
    domain_id = entity_id_for("domain", "acme.atlassian.net")
    assert ("ticket_closed", "ENG-42") in _evidence_details(store, domain_id)


def test_linear_ticket_closed_on_linear_host(store, tmp_path):
    store.add_observation(
        "OPS-7 marked as resolved by the on-call",
        source="capture",
        ts=NOW - 3600.0,
        window="OPS-7 — Linear",
        url="https://linear.app/acme/issue/OPS-7",
    )
    _run(store, tmp_path)
    domain_id = entity_id_for("domain", "linear.app")
    assert ("ticket_closed", "OPS-7") in _evidence_details(store, domain_id)


def test_jira_words_off_host_do_not_fire(store, tmp_path):
    store.add_observation(
        "ENG-42 is finally Done according to the standup notes",
        source="capture",
        ts=NOW - 3600.0,
        window="standup notes",
        url="https://example.com/notes",
    )
    counts = _run(store, tmp_path)
    assert counts["evidence"] == 0
    assert store.stats()["entity_evidence"] == 0


# -- aliases -----------------------------------------------------------------------


def test_bare_repo_alias_added_while_unique_and_removed_on_collision(
    store, tmp_path
):
    store.add_observation(
        "work on github.com/bbista/openbird today",
        source="capture", ts=NOW - 7200.0,
    )
    _run(store, tmp_path)
    repo = store.get_entity(entity_id_for("repo", "bbista/openbird"))
    assert repo["aliases"] == ["openbird"]

    # A second repo with the SAME bare name collides: alias removed from both.
    store.add_observation(
        "reviewing the fork github.com/acme/openbird as well",
        source="capture", ts=NOW - 3600.0,
    )
    _run(store, tmp_path)
    assert store.get_entity(entity_id_for("repo", "bbista/openbird"))["aliases"] == []
    assert store.get_entity(entity_id_for("repo", "acme/openbird"))["aliases"] == []


# -- shipped_language over block summaries ------------------------------------------------


def _seed_summary(store, *, text, key="k1", fingerprint="f1", generated_at=None,
                  start_ts=NOW - 7200.0, end_ts=NOW - 3600.0):
    obs = store.add_observation(
        "grounding text for the block", source="capture", ts=start_ts + 10.0
    )
    span_id = store.open_span(
        epoch_id="e", start_ts=start_ts, end_ts=end_ts, bundle_id="b",
        detail_tier=1,
    )
    return store.save_block_summary(
        local_date=_dt.datetime.fromtimestamp(start_ts).strftime("%Y-%m-%d"),
        block_key=key,
        block_fingerprint=fingerprint,
        start_ts=start_ts,
        end_ts=end_ts,
        dominant_bundle="b",
        level=None,
        summary_text=text,
        model="m",
        extractor_version="block-summary-v1",
        observation_ids=[obs.id],
        span_ids=[span_id],
    )


def test_shipped_language_cites_summary_id(store, tmp_path):
    store.add_observation(
        "hacking on github.com/bbista/openbird", source="capture",
        ts=NOW - 7000.0,
    )
    summary = _seed_summary(
        store, text="Shipped the openbird entity ledger and tests."
    )
    _run(store, tmp_path)
    repo_id = entity_id_for("repo", "bbista/openbird")
    rows = [
        r for r in store.entity_evidence_for(repo_id)
        if r["kind"] == "shipped_language"
    ]
    assert rows and rows[0]["source_kind"] == "summary"
    assert rows[0]["source_id"] == summary["id"]
    assert rows[0]["detail"] == "shipped"


def test_summary_regeneration_is_remined_by_generation_cursor(store, tmp_path):
    """REGRESSION (binding): regenerating an old block AFTER the watermark
    advanced must recreate the shipped_language evidence — the summary cursor
    is generation-time, so the replacement row is always re-mined."""
    store.add_observation(
        "hacking on github.com/bbista/openbird", source="capture",
        ts=NOW - 7000.0,
    )
    _seed_summary(
        store, text="Shipped the openbird milestone.", generated_at=None
    )
    _run(store, tmp_path)  # watermark now past the summary's generated_at
    repo_id = entity_id_for("repo", "bbista/openbird")
    assert any(
        r["kind"] == "shipped_language" for r in store.entity_evidence_for(repo_id)
    )

    # Regenerate the SAME block (same key, drifted fingerprint): the old row's
    # trigger cascade deletes the old evidence…
    regenerated = _seed_summary(
        store, text="Shipped the openbird milestone, refined.",
        key="k1", fingerprint="f2",
    )
    assert not any(
        r["kind"] == "shipped_language" for r in store.entity_evidence_for(repo_id)
    )

    # …and the next run re-mines the replacement despite the old watermark.
    _run(store, tmp_path)
    rows = [
        r for r in store.entity_evidence_for(repo_id)
        if r["kind"] == "shipped_language"
    ]
    assert rows and rows[0]["source_id"] == regenerated["id"]


# -- open-loop promotion + resolution -----------------------------------------------------


def _seed_loop_day_memory(store, *, loops, local_date=None, source_ids=None):
    local_date = local_date or _dt.datetime.fromtimestamp(NOW).strftime("%Y-%m-%d")
    return store.save_day_memory(
        local_date=local_date,
        source_scope="capture",
        extractor_version="day-memory-v9",
        payload={"local_date": local_date, "open_loops": loops},
        source_ids=source_ids or [],
    )


def test_open_loop_promotion_rehydrates_and_cites_true_earliest_row(
    store, tmp_path
):
    early = store.add_observation(
        "waiting on review github.com/bbista/openbird/pull/7",
        source="capture", ts=NOW - 2 * DAY,
    )
    late = store.add_observation(
        "still open github.com/bbista/openbird/pull/7",
        source="capture", ts=NOW - DAY,
    )
    # Payload source_ids are SORTED (never chronological) — the pass must
    # rehydrate and pick the true earliest row, whatever the sort order says.
    _seed_loop_day_memory(
        store,
        loops=[{
            "kind": "github_pr",
            "title": "openbird PR 7",
            "cue": "bbista/openbird pull #7",
            "source_ids": sorted([early.id, late.id]),
            "source_count": 2,
        }],
        source_ids=[early.id, late.id],
    )
    counts = _run(store, tmp_path)
    assert counts["loops_promoted"] == 1
    repo_id = entity_id_for("repo", "bbista/openbird")
    loops = [
        r for r in store.entity_evidence_for(repo_id) if r["kind"] == "open_loop"
    ]
    assert len(loops) == 1
    assert loops[0]["source_id"] == early.id
    assert loops[0]["ts"] == early.ts
    assert loops[0]["detail"] == "github:bbista/openbird#7"


def test_open_loop_with_unstable_item_id_is_skipped(store, tmp_path):
    mixed = store.add_observation(
        "github.com/bbista/openbird/pull/7 and github.com/bbista/openbird/pull/8",
        source="capture", ts=NOW - DAY,
    )
    _seed_loop_day_memory(
        store,
        loops=[{
            "kind": "github_pr",
            "title": "ambiguous",
            "cue": "bbista/openbird pull #7",
            "source_ids": [mixed.id],
            "source_count": 1,
        }],
        source_ids=[mixed.id],
    )
    counts = _run(store, tmp_path)
    assert counts["loops_promoted"] == 0
    assert counts["loops_skipped"] == 1


def test_generic_cue_loops_stay_day_scoped(store, tmp_path):
    obs = store.add_observation(
        "TODO follow up on the design review", source="capture", ts=NOW - DAY
    )
    _seed_loop_day_memory(
        store,
        loops=[{
            "kind": "cue",
            "title": "TODO follow up on the design review",
            "cue": "todo",
            "source_ids": [obs.id],
            "source_count": 1,
        }],
        source_ids=[obs.id],
    )
    counts = _run(store, tmp_path)
    assert counts["loops_promoted"] == 0
    assert counts["loops_skipped"] == 0
    assert store.stats()["entity_evidence"] == 0


def test_resolution_requires_exact_detail_and_later_ts(store, tmp_path):
    opened = store.add_observation(
        "open github.com/bbista/openbird/pull/7", source="capture",
        ts=NOW - 2 * DAY,
    )
    _seed_loop_day_memory(
        store,
        loops=[{
            "kind": "github_pr",
            "title": "openbird PR 7",
            "cue": "bbista/openbird pull #7",
            "source_ids": [opened.id],
            "source_count": 1,
        }],
        source_ids=[opened.id],
    )
    # A LATER merge of the SAME item resolves it; a merge of a DIFFERENT item
    # (#8) must not.
    store.add_observation(
        "This pull request was merged", source="capture", ts=NOW - DAY,
        window="Pull Request #8 merged",
        url="https://github.com/bbista/openbird/pull/8",
    )
    counts = _run(store, tmp_path)
    assert counts["loops_promoted"] == 1
    assert counts["loops_resolved"] == 0

    merged = store.add_observation(
        "This pull request was merged", source="capture", ts=NOW - 0.5 * DAY,
        window="Pull Request #7 merged",
        url="https://github.com/bbista/openbird/pull/7",
    )
    counts = _run(store, tmp_path)
    assert counts["loops_resolved"] == 1
    repo_id = entity_id_for("repo", "bbista/openbird")
    resolved = [
        r for r in store.entity_evidence_for(repo_id)
        if r["kind"] == "open_loop_resolved"
    ]
    assert len(resolved) == 1
    assert resolved[0]["detail"] == "github:bbista/openbird#7"
    assert resolved[0]["source_id"] == merged.id  # cites the RESOLVING source
    assert resolved[0]["ts"] > NOW - 2 * DAY

    # Idempotent: a third run inserts nothing new.
    counts = _run(store, tmp_path)
    assert counts["loops_resolved"] == 0


# -- cursors / bounds -----------------------------------------------------------------


def test_first_run_bounded_by_lookback_days(store, tmp_path):
    store.add_observation(
        "ancient github.com/bbista/oldproject work", source="capture",
        ts=NOW - 20 * DAY,  # beyond the 14-day default lookback
    )
    _run(store, tmp_path)
    assert store.get_entity(entity_id_for("repo", "bbista/oldproject")) is None


def test_batch_limit_is_a_real_row_cap_and_cursor_resumes(store, tmp_path):
    ids = []
    for i, repo in enumerate(("bbista/a", "bbista/b", "bbista/c")):
        obs = store.add_observation(
            f"work on github.com/{repo}", source="capture",
            ts=NOW - 3600.0 + i,
        )
        ids.append(obs.id)

    _run(store, tmp_path, entity_evidence_batch_limit=2)
    assert store.get_entity(entity_id_for("repo", "bbista/a")) is not None
    assert store.get_entity(entity_id_for("repo", "bbista/b")) is not None
    assert store.get_entity(entity_id_for("repo", "bbista/c")) is None
    # The composite cursor advanced only through the last processed row.
    assert store.get_kv("entity_aggregation.obs_id") == ids[1]

    _run(store, tmp_path, entity_evidence_batch_limit=2)
    assert store.get_entity(entity_id_for("repo", "bbista/c")) is not None
    assert store.get_kv("entity_aggregation.obs_id") == ids[2]


def test_dormancy_at_aggregation_time(store, tmp_path):
    stale = store.upsert_entity("repo", "bbista/stale", seen_ts=NOW - 30 * DAY)
    done = store.upsert_entity("repo", "bbista/done", seen_ts=NOW - 30 * DAY)
    store.set_entity_status(done["id"], "user_marked_done")
    counts = _run(store, tmp_path)
    assert counts["dormant"] == 1
    assert store.get_entity(stale["id"])["status"] == "dormant"
    assert store.get_entity(done["id"])["status"] == "user_marked_done"


# -- CLI surfacing (Phase E2) -----------------------------------------------------


def _cli_store(tmp_path):
    s = MemoryStore(
        db_path=str(tmp_path / "cli-entities.db"),
        settings=Settings(data_dir=tmp_path, embed_dim=64),
        provider=FakeProvider(embed_dim=64),
    )
    obs = s.add_observation(
        "This pull request was merged", source="capture", ts=NOW - DAY,
        window="Merge PR #12 · bbista/openbird",
        url="https://github.com/bbista/openbird/pull/12",
    )
    entity = s.upsert_entity(
        "repo", "bbista/openbird", seen_ts=NOW - DAY,
        source_kind="observation", source_id=obs.id,
    )
    s.add_entity_evidence(
        entity["id"], ts=NOW - DAY, kind="pr_merged",
        source_kind="observation", source_id=obs.id,
        detail="github:bbista/openbird#12",
    )
    return s


def test_cli_entities_list_withholds_names_non_interactive(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from openbird import cli

    store = _cli_store(tmp_path)
    monkeypatch.setattr(cli, "_store_maintenance", lambda: store)
    res = CliRunner().invoke(cli.app, ["entities", "list"])
    assert res.exit_code == 0, res.output
    # Metadata yes; DERIVED SENSITIVE name no (CliRunner output is not a TTY).
    assert "kind=repo" in res.output
    assert "evidence=1" in res.output
    assert "bbista/openbird" not in res.output
    assert "withheld (non-interactive output)" in res.output


def test_cli_entities_list_json_non_interactive_omits_names(tmp_path, monkeypatch):
    import json as _json

    from typer.testing import CliRunner

    from openbird import cli

    store = _cli_store(tmp_path)
    monkeypatch.setattr(cli, "_store_maintenance", lambda: store)
    res = CliRunner().invoke(cli.app, ["entities", "list", "--json"])
    assert res.exit_code == 0, res.output
    assert "bbista/openbird" not in res.output
    payload = _json.loads(res.output[: res.output.rindex("}") + 1])
    assert payload["entities"][0]["kind"] == "repo"
    assert "name" not in payload["entities"][0]


def test_cli_entities_show_withholds_bodies_non_interactive(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from openbird import cli

    store = _cli_store(tmp_path)
    monkeypatch.setattr(cli, "_store_maintenance", lambda: store)
    res = CliRunner().invoke(cli.app, ["entities", "show", "openbird"])
    assert res.exit_code == 0, res.output
    assert "kind=repo" in res.output
    assert "github:bbista/openbird#12" not in res.output  # detail withheld
    assert "evidence rows withheld" in res.output


def test_cli_entities_show_unknown_name_exits_nonzero(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from openbird import cli

    store = _cli_store(tmp_path)
    monkeypatch.setattr(cli, "_store_maintenance", lambda: store)
    res = CliRunner().invoke(cli.app, ["entities", "show", "no-such-entity"])
    assert res.exit_code == 1


def test_cli_data_integrity_includes_entity_probe(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from openbird import cli
    from openbird.config import Settings as _Settings

    store = _cli_store(tmp_path)
    db_path = str(tmp_path / "cli-entities.db")
    store.close()
    settings = _Settings(data_dir=tmp_path, db_path=db_path, embed_dim=64)
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    res = CliRunner().invoke(cli.app, ["data", "integrity"])
    assert res.exit_code == 0, res.output
    assert "entity ledger: ok" in res.output


def test_cli_briefing_reports_dormant_unresolved_loop_count_only(
    tmp_path, monkeypatch
):
    """The default briefing gains ONE count-only review line — never names."""
    import time as _time

    from typer.testing import CliRunner

    from openbird import cli

    s = MemoryStore(
        db_path=str(tmp_path / "briefing-entities.db"),
        settings=Settings(data_dir=tmp_path, embed_dim=64),
        provider=FakeProvider(embed_dim=64),
    )
    now = _time.time()
    obs = s.add_observation(
        "waiting on github.com/bbista/secretname/pull/7",
        source="capture", ts=now - 600.0,
    )
    entity = s.upsert_entity(
        "repo", "bbista/secretname", seen_ts=now - 600.0,
        source_kind="observation", source_id=obs.id,
    )
    s.add_entity_evidence(
        entity["id"], ts=now - 600.0, kind="open_loop",
        source_kind="observation", source_id=obs.id,
        detail="github:bbista/secretname#7",
    )
    s.set_entity_status(entity["id"], "dormant")

    monkeypatch.setattr(cli, "_store_maintenance", lambda: s)
    res = CliRunner().invoke(cli.app, ["briefing", "--day", "0"])
    assert res.exit_code == 0, res.output
    assert "1 dormant project(s) have unresolved open loops" in res.output
    # The NEW review line itself is count-only — no entity name. (The day
    # prose above it may echo today's captured cue text; that is the existing
    # day-memory behavior over the seeded observation, not the ledger line.)
    joined = " ".join(res.output.split())
    line_start = joined.index("1 dormant project(s)")
    assert "secretname" not in joined[line_start:]
