"""Optional cross-encoder reranker over a llama.cpp-compatible ``/v1/rerank`` endpoint.

OpenBird's retrieval fuses BM25 + vector with RRF, then dedups with MMR. A
cross-encoder reranker inserted between RRF and MMR re-scores each candidate
against the query and is the single biggest retrieval-accuracy lever — but
Ollama cannot serve rerankers (as of 2026), so this talks to a vendor-neutral
``/v1/rerank`` HTTP endpoint (e.g. ``llama-server --reranking`` with a
``bge-reranker-v2-m3`` GGUF).

Design invariants (privacy- and reliability-critical):

* **Default off.** No ``rerank_model`` configured -> :func:`build_reranker`
  returns ``None`` and search behavior is unchanged.
* **Never breaks search.** Every failure raises :class:`RerankError` carrying a
  structured, content-free ``reason`` code; the caller (the store) logs the
  reason and falls back to the RRF order. A wedged server costs at most the
  configured wall-clock ``timeout`` (single call, no retries).
* **Rigorous alignment.** ``/v1/rerank`` may return results SORTED and possibly
  truncated by ``top_n``; scores are mapped back to documents by the returned
  ``index`` (never array position). EVERY input document must receive a finite
  score, else it is a handled failure (the caller drops no candidate).
* **No content leaks.** Errors carry only the reason code / HTTP status — never
  the query, chunk text, or the server's (possibly document-echoing) body.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from openbird.config import Settings, is_loopback_host

# Default endpoint when a rerank model is configured without an explicit host —
# the conventional llama.cpp server port on this machine (loopback = on-device).
DEFAULT_RERANK_HOST = "http://localhost:8080"


def resolved_rerank_host(settings: Settings) -> str:
    """The rerank base URL: explicit ``rerank_host`` or the llama.cpp localhost default."""
    host = (settings.rerank_host or "").strip()
    return host or DEFAULT_RERANK_HOST


def rerank_is_remote(settings: Settings) -> bool:
    """True iff a configured reranker would send query+chunk text OFF this machine.

    Only meaningful when a reranker is configured (``rerank_model`` set). A
    loopback host is on-device; anything else is a remote route that must be
    cloud-gated like the llm/embed roles.
    """
    if not (settings.rerank_model or "").strip():
        return False
    return not is_loopback_host(resolved_rerank_host(settings))


class RerankError(RuntimeError):
    """A handled reranker failure — the caller falls back to the pre-rerank order.

    ``reason`` is a short structured code (``timeout|transport|http_<status>|
    bad_response``) safe to log; it never contains the query or document text.
    """

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


@dataclass
class HTTPReranker:
    """Cross-encoder reranker client over a llama.cpp-compatible ``/v1/rerank``.

    ``rerank(query, documents)`` returns a list of finite relevance scores aligned
    to the INPUT order of ``documents``. Any transport/parse/alignment problem
    raises :class:`RerankError` so the caller can fall back.
    """

    model: str
    host: str
    timeout: float
    top_n: int = 0

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        import threading

        import httpx

        payload: dict = {"model": self.model, "query": query, "documents": documents}
        # top_n only limits how many results the server returns; we still require a
        # score for every input document below, so a truncating server -> bad_response.
        if self.top_n and self.top_n > 0:
            payload["top_n"] = self.top_n
        url = self.host.rstrip("/") + "/v1/rerank"

        # TRUE total wall-clock deadline for the CALLER: httpx's scalar timeout is
        # per-phase (connect/read/write), so we run the single (no-retry) request on
        # a daemon thread and join for at most `self.timeout`. If it overruns we
        # abandon the worker and fall back — search never hangs.
        #
        # Python can't force-kill the worker, but it can't linger unbounded either:
        # the request runs inside an `httpx.Client` context manager, so when httpx's
        # read timeout fires (≤ self.timeout for an accept-but-no-response server) the
        # client is closed and the socket released. Only a malicious server trickling
        # a byte just under the read timeout could extend a worker — and that requires
        # a non-loopback rerank_host, which is cloud-gated (opt-in) anyway.
        box: dict = {}

        def _call() -> None:
            try:
                with httpx.Client(timeout=httpx.Timeout(self.timeout)) as client:
                    box["resp"] = client.post(url, json=payload)
            except BaseException as exc:  # noqa: BLE001 - carried to the main thread
                box["exc"] = exc

        worker = threading.Thread(target=_call, daemon=True)
        worker.start()
        worker.join(self.timeout)
        if worker.is_alive():
            raise RerankError("timeout")
        exc = box.get("exc")
        if exc is not None:
            if isinstance(exc, httpx.TimeoutException):
                raise RerankError("timeout") from exc
            if isinstance(exc, httpx.HTTPError):
                # Connection/DNS/protocol errors — class name only, never the body.
                raise RerankError("transport") from exc
            raise RerankError("transport") from exc
        resp = box["resp"]
        if resp.status_code != 200:
            raise RerankError(f"http_{resp.status_code}")
        try:
            body = resp.json()
        except ValueError as exc:
            raise RerankError("bad_response") from exc
        return self._align_scores(body, len(documents))

    @staticmethod
    def _align_scores(body: object, n_docs: int) -> list[float]:
        """Map ``{results:[{index, relevance_score}]}`` back to input order.

        Results may be sorted and/or truncated; we index by the returned ``index``.
        EVERY document must get exactly one finite score, else ``bad_response`` (so
        the caller drops no candidate).
        """
        if not isinstance(body, dict):
            raise RerankError("bad_response")
        results = body.get("results")
        if not isinstance(results, list):
            raise RerankError("bad_response")
        scores: list[float | None] = [None] * n_docs
        for item in results:
            if not isinstance(item, dict):
                raise RerankError("bad_response")
            idx = item.get("index")
            score = item.get("relevance_score", item.get("score"))
            if not isinstance(idx, int) or not (0 <= idx < n_docs):
                raise RerankError("bad_response")
            try:
                value = float(score)
            except (TypeError, ValueError) as exc:
                raise RerankError("bad_response") from exc
            if not math.isfinite(value):
                raise RerankError("bad_response")
            if scores[idx] is not None:
                raise RerankError("bad_response")  # duplicate index
            scores[idx] = value
        if any(s is None for s in scores):
            raise RerankError("bad_response")  # a document went unscored
        return [s for s in scores]  # type: ignore[misc]


def build_reranker(settings: Settings) -> HTTPReranker | None:
    """Construct the configured reranker, or ``None`` when reranking is disabled.

    Disabled (the default) whenever ``rerank_model`` is empty — search is then
    unchanged. The cloud opt-in for a remote ``rerank_host`` is enforced separately
    (see :func:`rerank_is_remote` and the provider route classification); this
    factory only wires the client.
    """
    model = (settings.rerank_model or "").strip()
    if not model:
        return None
    return HTTPReranker(
        model=model,
        host=resolved_rerank_host(settings),
        timeout=settings.rerank_timeout,
        top_n=settings.rerank_top_n,
    )


__all__ = [
    "HTTPReranker",
    "RerankError",
    "build_reranker",
    "rerank_is_remote",
    "resolved_rerank_host",
    "DEFAULT_RERANK_HOST",
]
