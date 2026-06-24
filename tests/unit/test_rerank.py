"""Tests for the optional cross-encoder reranker stage.

Covers the HTTP client's rigorous /v1/rerank alignment + hard failure handling,
and the store's _rerank stage: normalization, RRF fallback on every failure
(including a hanging server within the deadline), and the privacy cloud-route
classification of a remote rerank host.
"""

from __future__ import annotations

import math

import pytest

from openbird.config import Settings
from openbird.llm.provider import classify_models
from openbird.llm.rerank import (
    HTTPReranker,
    RerankError,
    build_reranker,
    rerank_is_remote,
)
from openbird.memory.store import MemoryStore
from openbird.types import SearchHit


@pytest.fixture(autouse=True)
def _hermetic_data_dir(tmp_path, monkeypatch):
    # Several helpers build Settings(embed_dim=768) without an explicit data_dir;
    # Settings.__post_init__ creates/chmods the data dir, so point it at a tmp dir
    # to keep the suite hermetic (never touch the developer's ~/.openbird).
    monkeypatch.setenv("OPENBIRD_DATA_DIR", str(tmp_path))


# --------------------------------------------------------------------------- #
# config + wiring                                                              #
# --------------------------------------------------------------------------- #


def test_reranker_disabled_by_default(tmp_path):
    s = Settings(data_dir=tmp_path, embed_dim=768)
    assert build_reranker(s) is None
    assert "rerank" not in classify_models(s)


def test_rerank_timeout_must_be_finite_positive(tmp_path):
    for bad in (0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            Settings(data_dir=tmp_path, embed_dim=768, rerank_timeout=bad)


def test_rerank_top_n_must_be_non_negative(tmp_path):
    # A negative top_n must be rejected, not silently treated as "rerank all".
    with pytest.raises(ValueError):
        Settings(data_dir=tmp_path, embed_dim=768, rerank_top_n=-1)
    # 0 (rerank all) and positive caps are valid.
    assert Settings(data_dir=tmp_path, embed_dim=768, rerank_top_n=0).rerank_top_n == 0
    assert Settings(data_dir=tmp_path, embed_dim=768, rerank_top_n=5).rerank_top_n == 5


def test_remote_rerank_host_is_cloud_classified(tmp_path):
    s = Settings(
        data_dir=tmp_path,
        embed_dim=768,
        rerank_model="bge-reranker-v2-m3",
        rerank_host="http://10.0.0.9:8080",
    )
    assert rerank_is_remote(s) is True
    assert classify_models(s)["rerank"] == "bge-reranker-v2-m3"


def test_loopback_rerank_host_is_local(tmp_path):
    s = Settings(data_dir=tmp_path, embed_dim=768, rerank_model="bge-reranker-v2-m3")
    assert rerank_is_remote(s) is False
    assert "rerank" not in classify_models(s)


# --------------------------------------------------------------------------- #
# HTTPReranker /v1/rerank alignment + failure handling                        #
# --------------------------------------------------------------------------- #


class _FakeResp:
    def __init__(self, status, body):
        self.status_code = status
        self._body = body

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


def _patch_post(monkeypatch, *, resp=None, exc=None):
    # The reranker issues its request inside `with httpx.Client(...) as c: c.post(...)`,
    # so patch the Client (not module-level httpx.post).
    import httpx

    class _FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, json=None):
            if exc is not None:
                raise exc
            return resp

    monkeypatch.setattr(httpx, "Client", _FakeClient)


def _reranker():
    return HTTPReranker(model="m", host="http://localhost:8080", timeout=5.0)


def test_align_by_index_not_array_position(monkeypatch):
    # Results come back SORTED by score; must map back to input order by `index`.
    body = {"results": [
        {"index": 2, "relevance_score": 0.9},
        {"index": 0, "relevance_score": 0.1},
        {"index": 1, "relevance_score": 0.5},
    ]}
    _patch_post(monkeypatch, resp=_FakeResp(200, body))
    scores = _reranker().rerank("q", ["a", "b", "c"])
    assert scores == [0.1, 0.5, 0.9]  # aligned to input order a,b,c


def test_partial_results_is_bad_response(monkeypatch):
    # A truncated response (fewer scores than docs) must NOT silently drop a doc.
    body = {"results": [{"index": 0, "relevance_score": 0.9}]}
    _patch_post(monkeypatch, resp=_FakeResp(200, body))
    with pytest.raises(RerankError) as ei:
        _reranker().rerank("q", ["a", "b", "c"])
    assert ei.value.reason == "bad_response"


@pytest.mark.parametrize("body", [
    {"results": [{"index": 0, "relevance_score": 0.5}, {"index": 0, "relevance_score": 0.6}]},  # dup index
    {"results": [{"index": 5, "relevance_score": 0.5}]},  # index out of range
    {"results": [{"index": 0, "relevance_score": float("nan")}]},  # non-finite
    {"results": "nope"},  # results not a list
    {"no_results": []},  # missing results key
])
def test_malformed_responses_are_bad_response(monkeypatch, body):
    # Two input docs for every case: any duplicate/out-of-range/non-finite/missing
    # mapping leaves a document unscored or invalid -> bad_response (no silent drop).
    _patch_post(monkeypatch, resp=_FakeResp(200, body))
    with pytest.raises(RerankError) as ei:
        _reranker().rerank("q", ["a", "b"])
    assert ei.value.reason == "bad_response"


def test_http_error_status_is_reasoned(monkeypatch):
    _patch_post(monkeypatch, resp=_FakeResp(503, {}))
    with pytest.raises(RerankError) as ei:
        _reranker().rerank("q", ["a", "b"])
    assert ei.value.reason == "http_503"


def test_timeout_maps_to_timeout_reason(monkeypatch):
    import httpx

    _patch_post(monkeypatch, exc=httpx.TimeoutException("slow"))
    with pytest.raises(RerankError) as ei:
        _reranker().rerank("q", ["a", "b"])
    assert ei.value.reason == "timeout"


def test_transport_error_maps_to_transport_reason(monkeypatch):
    import httpx

    _patch_post(monkeypatch, exc=httpx.ConnectError("refused"))
    with pytest.raises(RerankError) as ei:
        _reranker().rerank("q", ["a", "b"])
    assert ei.value.reason == "transport"


def test_real_http_timeout_is_wall_clock_bounded():
    # Prove the deadline is real wall-clock: point at a socket that ACCEPTS but
    # never responds; rerank must raise RerankError('timeout') within ~the deadline
    # (not hang). This is the guarantee that a wedged llama-server can't freeze
    # search — the store then falls back to RRF order.
    import socket
    import threading
    import time as _time

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    conns: list = []

    def _accept():
        try:
            c, _ = srv.accept()
            conns.append(c)  # accept the connection but never send a response
        except OSError:
            pass

    threading.Thread(target=_accept, daemon=True).start()
    rr = HTTPReranker(model="m", host=f"http://127.0.0.1:{port}", timeout=0.5)
    start = _time.monotonic()
    try:
        with pytest.raises(RerankError) as ei:
            rr.rerank("q", ["a", "b"])
        elapsed = _time.monotonic() - start
        assert ei.value.reason == "timeout"
        assert elapsed < 3.0  # bounded by the 0.5s deadline + margin, never hangs
    finally:
        for c in conns:
            c.close()
        srv.close()


def test_empty_documents_short_circuits(monkeypatch):
    # No HTTP call for an empty candidate set.
    called = {"n": 0}
    import httpx

    def _post(*a, **k):
        called["n"] += 1
        raise AssertionError("should not be called")

    monkeypatch.setattr(httpx, "post", _post)
    assert _reranker().rerank("q", []) == []
    assert called["n"] == 0


# --------------------------------------------------------------------------- #
# store._rerank: normalization, reordering, graceful fallback                 #
# --------------------------------------------------------------------------- #


def _hit(chunk_id, text, score):
    return SearchHit(chunk_id=chunk_id, content_hash="h" + chunk_id, text=text, score=score)


class _FakeReranker:
    def __init__(self, scores=None, exc=None):
        self._scores = scores
        self._exc = exc

    def rerank(self, query, documents):
        if self._exc is not None:
            raise self._exc
        return self._scores


def _store(reranker):
    # In-memory store; provider unused for these search-less unit calls.
    return MemoryStore(db_path=":memory:", settings=Settings(embed_dim=768),
                       provider=_NoProvider(), reranker=reranker)


class _NoProvider:
    embed_dim = 768

    def cohort_key(self):
        return "fake:fake:768:x"

    def embed(self, texts):
        return [[0.0] * 768 for _ in texts]


def test_rerank_reorders_and_normalizes():
    hits = [_hit("0", "a", 0.9), _hit("1", "b", 0.5), _hit("2", "c", 0.1)]
    # Reranker prefers c (idx2) > a (idx0) > b (idx1).
    st = _store(_FakeReranker(scores=[0.2, 0.0, 1.0]))
    try:
        out = st._rerank("q", hits)
    finally:
        st.close()
    assert [h.chunk_id for h in out] == ["2", "0", "1"]
    assert out[0].score == pytest.approx(1.0)  # normalized to [0,1]
    assert out[-1].score == pytest.approx(0.0)
    assert all(0.0 <= h.score <= 1.0 for h in out)


def test_rerank_all_equal_keeps_rrf_order():
    hits = [_hit("0", "a", 0.9), _hit("1", "b", 0.5)]
    st = _store(_FakeReranker(scores=[0.7, 0.7]))
    try:
        out = st._rerank("q", hits)
    finally:
        st.close()
    assert out == hits  # no signal -> unchanged RRF order


def test_rerank_error_falls_back_to_rrf_order():
    hits = [_hit("0", "a", 0.9), _hit("1", "b", 0.5)]
    st = _store(_FakeReranker(exc=RerankError("timeout")))
    try:
        out = st._rerank("q", hits)
    finally:
        st.close()
    assert out == hits


def test_rerank_hang_returns_within_deadline_via_fallback():
    # A reranker that never returns (raises after the bound) must not hang search:
    # the client's hard timeout surfaces as RerankError -> RRF fallback. We simulate
    # the post-deadline RerankError the HTTP client would raise.
    hits = [_hit("0", "a", 0.9), _hit("1", "b", 0.5)]
    st = _store(_FakeReranker(exc=RerankError("timeout")))
    try:
        out = st._rerank("q", hits)
    finally:
        st.close()
    assert [h.chunk_id for h in out] == ["0", "1"]


def test_rerank_unexpected_exception_falls_back():
    hits = [_hit("0", "a", 0.9), _hit("1", "b", 0.5)]
    st = _store(_FakeReranker(exc=RuntimeError("boom")))
    try:
        out = st._rerank("q", hits)  # must not propagate
    finally:
        st.close()
    assert out == hits


def test_rerank_wrong_length_falls_back():
    hits = [_hit("0", "a", 0.9), _hit("1", "b", 0.5)]
    st = _store(_FakeReranker(scores=[0.1]))  # length mismatch
    try:
        out = st._rerank("q", hits)
    finally:
        st.close()
    assert out == hits


def test_rerank_disabled_is_noop():
    hits = [_hit("0", "a", 0.9), _hit("1", "b", 0.5)]
    st = _store(None)
    try:
        out = st._rerank("q", hits)
    finally:
        st.close()
    assert out is hits  # exact passthrough


def test_store_refuses_remote_reranker_without_opt_in(tmp_path):
    # Store-direct path (provider injected, CLI/provider cloud gate skipped): a
    # remote rerank host without opt-in must fail closed, not silently send text.
    from openbird.llm.provider import CloudOptInRequired

    s = Settings(
        data_dir=tmp_path,
        embed_dim=768,
        rerank_model="bge-reranker-v2-m3",
        rerank_host="http://10.0.0.9:8080",
        allow_cloud=False,
    )
    with pytest.raises(CloudOptInRequired):
        MemoryStore(db_path=":memory:", settings=s, provider=_NoProvider())


def test_store_allows_remote_reranker_with_opt_in(tmp_path):
    s = Settings(
        data_dir=tmp_path,
        embed_dim=768,
        rerank_model="bge-reranker-v2-m3",
        rerank_host="http://10.0.0.9:8080",
        allow_cloud=True,
    )
    st = MemoryStore(db_path=":memory:", settings=s, provider=_NoProvider())
    try:
        assert st.reranker is not None
    finally:
        st.close()


def test_store_search_does_not_hang_on_wedged_reranker():
    # End-to-end: a real reranker pointed at a socket that never responds must not
    # freeze search — _rerank times out and falls back to RRF order within ~deadline.
    import socket
    import threading
    import time as _time

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    conns: list = []

    def _accept():
        try:
            c, _ = srv.accept()
            conns.append(c)
        except OSError:
            pass

    threading.Thread(target=_accept, daemon=True).start()

    rr = HTTPReranker(model="m", host=f"http://127.0.0.1:{port}", timeout=0.5)
    st = MemoryStore(db_path=":memory:", settings=Settings(embed_dim=768),
                     provider=_NoProvider(), reranker=rr)
    try:
        st.add_observation("alpha bravo charlie delta echo", source="ingest", window="w")
        st.add_observation("alpha bravo foxtrot golf hotel", source="ingest", window="w")
        start = _time.monotonic()
        hits = st.search("alpha bravo", k=5, semantic=False)  # BM25 only; reranker wedged
        elapsed = _time.monotonic() - start
        assert hits, "search must still return results (RRF fallback)"
        assert elapsed < 3.0  # bounded by the reranker deadline, never hangs
    finally:
        st.close()
        for c in conns:
            c.close()
        srv.close()


def test_rerank_log_is_content_free(caplog):
    hits = [_hit("0", "secret query text", 0.9), _hit("1", "more secret", 0.5)]
    st = _store(_FakeReranker(exc=RerankError("transport")))
    try:
        with caplog.at_level("INFO", logger="openbird.memory"):
            st._rerank("SENSITIVE QUERY", hits)
    finally:
        st.close()
    blob = " ".join(r.getMessage() for r in caplog.records)
    assert "transport" in blob
    assert "SENSITIVE QUERY" not in blob
    assert "secret" not in blob
