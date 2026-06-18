"""Rank fusion (RRF) and diversity dedup (MMR) for hybrid retrieval.

These are pure functions over rankings/hits so they can be unit-tested without a
database, Ollama, or embeddings.
"""

from __future__ import annotations

from collections.abc import Sequence

from openbird.types import SearchHit

RRF_K = 60  # standard reciprocal-rank-fusion damping constant


def rrf(
    rankings: Sequence[Sequence[str]],
    *,
    k: int = RRF_K,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion over several ranked id lists.

    Each input ranking is an ordered sequence of ids (best first). The fused
    score for an id is ``sum(1 / (k + rank))`` over the rankings it appears in
    (rank is 0-based). Returns ``(id, score)`` pairs sorted by descending score.

    Args:
        rankings: One ranked id list per retriever (e.g. vector, BM25).
        k: RRF damping constant; larger ``k`` flattens rank influence.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def _tokens(text: str) -> set[str]:
    """Lowercased word-token set used for cheap lexical similarity."""
    return {t for t in "".join(c if c.isalnum() else " " for c in text.lower()).split() if t}


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two token sets (0..1)."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def mmr(
    hits: Sequence[SearchHit],
    *,
    k: int | None = None,
    lambda_: float = 0.7,
    sim_threshold: float = 0.92,
) -> list[SearchHit]:
    """Greedy Maximal Marginal Relevance dedup over ranked hits.

    Selects hits that balance relevance (their existing ``score``) against
    novelty versus already-selected hits, using token-Jaccard similarity as a
    cheap stand-in for embedding similarity. Near-duplicate hits (similarity
    above ``sim_threshold``, e.g. repeated boilerplate or the same blob) are
    dropped outright.

    Args:
        hits: Candidate hits, ideally already fused/ranked.
        k: Max results to return; ``None`` returns all surviving hits.
        lambda_: Relevance vs. diversity trade-off (1.0 = pure relevance).
        sim_threshold: Similarity above which a candidate is treated as a dup.
    """
    if not hits:
        return []

    remaining = list(hits)
    token_cache: dict[int, set[str]] = {i: _tokens(h.text) for i, h in enumerate(hits)}
    index_of = {id(h): i for i, h in enumerate(hits)}

    selected: list[SearchHit] = []
    selected_tokens: list[set[str]] = []
    limit = k if k is not None else len(remaining)

    while remaining and len(selected) < limit:
        best_hit: SearchHit | None = None
        best_mmr = float("-inf")
        best_pos = -1
        is_dup = False

        for pos, cand in enumerate(remaining):
            cand_tokens = token_cache[index_of[id(cand)]]
            max_sim = max((_jaccard(cand_tokens, st) for st in selected_tokens), default=0.0)
            if max_sim >= sim_threshold:
                # Hard-drop near-duplicates.
                remaining.pop(pos)
                is_dup = True
                break
            mmr_score = lambda_ * cand.score - (1.0 - lambda_) * max_sim
            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_hit = cand
                best_pos = pos

        if is_dup:
            continue
        if best_hit is None:
            break

        selected.append(best_hit)
        selected_tokens.append(token_cache[index_of[id(best_hit)]])
        remaining.pop(best_pos)

    return selected


__all__ = ["rrf", "mmr", "RRF_K"]
