"""Unit tests for RRF fusion and MMR dedup (pure functions, no DB)."""

from __future__ import annotations

from openbird.memory.search import mmr, rrf
from openbird.types import SearchHit


def test_rrf_rewards_agreement_across_rankings():
    vec = ["a", "b", "c", "d"]
    bm25 = ["b", "a", "e", "f"]
    fused = rrf([vec, bm25])
    ids = [i for i, _ in fused]
    # "a" and "b" appear high in both -> should top the fused list.
    assert set(ids[:2]) == {"a", "b"}
    # Scores are descending.
    scores = [s for _, s in fused]
    assert scores == sorted(scores, reverse=True)


def test_rrf_single_ranking_preserves_order():
    fused = rrf([["x", "y", "z"]])
    assert [i for i, _ in fused] == ["x", "y", "z"]


def test_rrf_empty():
    assert rrf([]) == []
    assert rrf([[]]) == []


def test_rrf_k_constant_changes_spread():
    ranking = [["a", "b"]]
    tight = dict(rrf([ranking[0]], k=1))
    loose = dict(rrf([ranking[0]], k=1000))
    # Larger k flattens the gap between rank 0 and rank 1.
    assert (tight["a"] - tight["b"]) > (loose["a"] - loose["b"])


def _hit(chunk_id: str, text: str, score: float) -> SearchHit:
    return SearchHit(chunk_id=chunk_id, content_hash=chunk_id, text=text, score=score)


def test_mmr_drops_near_duplicates():
    hits = [
        _hit("1", "the quarterly revenue report shows strong growth", 1.0),
        _hit("2", "the quarterly revenue report shows strong growth", 0.9),  # dup
        _hit("3", "an unrelated note about lunch plans tomorrow", 0.8),
    ]
    out = mmr(hits, k=10)
    texts = [h.text for h in out]
    # The duplicate is removed; the unrelated hit survives.
    assert texts.count("the quarterly revenue report shows strong growth") == 1
    assert "an unrelated note about lunch plans tomorrow" in texts


def test_mmr_respects_k():
    hits = [_hit(str(i), f"distinct topic number {i} alpha beta", 1.0 - i * 0.1) for i in range(5)]
    out = mmr(hits, k=3)
    assert len(out) == 3


def test_mmr_keeps_relevance_order_when_diverse():
    hits = [
        _hit("1", "cats are wonderful animals", 0.9),
        _hit("2", "rockets launch into deep space", 0.8),
        _hit("3", "baking sourdough bread at home", 0.7),
    ]
    out = mmr(hits, k=3)
    assert [h.chunk_id for h in out] == ["1", "2", "3"]


def test_mmr_empty():
    assert mmr([]) == []
