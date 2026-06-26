"""Offline answer-quality evals for the local RAG chat path.

This harness measures whether a given **local model** produces *reliable* answers
grounded in captured text, by running the EXACT production answer path
(:class:`openbird.chat.rag.RAG`) over synthetic fixtures and scoring the result on
deterministic, gold-bound axes.

Design (approved via Codex adversarial review — see
``docs/design/chat-model-evals.md``):

- **Deterministic core, model-bound runner.** Unlike the signal eval (which is
  model-free), answer quality is inherently model-bound: you cannot score "does the
  model answer reliably" without running the model. So the SCORERS here are pure and
  unit-tested with canned outputs (CI-safe), while the RUNNER that calls a model is
  opt-in / ``integration``-marked and auto-skips when Ollama is unavailable.
- **Lane A (this module): generation isolation.** A fake searcher returns pre-baked
  ``SearchHit``s so retrieval is fixed and the score reflects only the model's
  generation + citation behavior. (Lane B — a seeded ``MemoryStore`` end-to-end —
  is a planned follow-up that reuses these same scorers and fixtures.)
- **Gold binds facts to sources.** A model that states the right fact but cites the
  wrong/zero source is NOT reliable for a provenance product, so each gold fact
  carries ``required_source_ids`` (see :class:`GoldFact`).
- **Content-free.** Reports carry metrics / ids / reason codes only — never captured
  fixture text or model output bodies — on every exit path (privacy discipline).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from openbird.chat.rag import RAG, AnswerResult
from openbird.types import Observation, SearchHit

# Retrieval modes a fixture row can exercise. Mirrors the branches in
# ``RAG.answer``: plain semantic ``search``; a ``temporal-phrase`` query
# ("what did I do yesterday?") that routes through the time-range scan; and an
# ``explicit-window`` (start_ts, end_ts) hard scope.
VALID_MODES: frozenset[str] = frozenset({"search", "temporal-phrase", "explicit-window"})

# Fixed reference clock for ``temporal-phrase`` cases so "yesterday"/"today" resolve
# deterministically (RAG exposes an injectable ``_now``). 2,000,000,000 = 2033-05-18.
EVAL_NOW = 2_000_000_000.0
_DAY_S = 86_400.0


# --------------------------------------------------------------------------- #
# Fixture schema                                                              #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GoldSource:
    """One synthetic captured source the model is given as context (S1..Sn)."""

    sid: str
    text: str
    app: str | None = None
    window: str | None = None
    ts: float = 0.0


@dataclass(frozen=True)
class GoldFact:
    """A fact that must appear in the answer, bound to the source(s) that back it.

    ``any_of`` is satisfied when ANY listed phrase is present (after
    normalization). ``required_source_ids`` are the source ids that MUST be cited
    for the fact to count as grounded — stating the fact without the citation is a
    reliability failure, not a pass.
    """

    any_of: tuple[str, ...]
    required_source_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ChatEvalInput:
    """One labeled fixture row for the chat answer-quality eval."""

    id: str
    question: str
    mode: str
    sources: tuple[GoldSource, ...]
    facts: tuple[GoldFact, ...] = ()
    forbidden: tuple[str, ...] = ()
    expect_grounded: bool = True
    window: tuple[float, float] | None = None


# --------------------------------------------------------------------------- #
# Scoring (pure / deterministic — unit-tested with canned outputs)           #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CaseScore:
    """Per-case deterministic outcome. Content-free: ids and booleans only."""

    id: str
    mode: str
    json_valid: bool
    facts_ok: bool
    citations_precise: bool  # every cited id is an allowed source id
    citations_complete: bool  # all required source ids were cited
    forbidden_clean: bool  # no forbidden / distractor term leaked
    refusal_ok: bool  # grounding matched expect_grounded
    window_ok: bool  # every citation ts inside the requested window (or n/a)
    reason_codes: tuple[str, ...]  # which checks failed, as stable codes

    @property
    def passed(self) -> bool:
        return not self.reason_codes


_WS = re.compile(r"\s+")
_PCT = re.compile(r"\bpercent\b")


def normalize(text: str) -> str:
    """Lowercase, collapse whitespace, and fold ``percent`` -> ``%`` for matching.

    Keeps fact matching robust to trivial surface variation without an LLM judge.
    Date folding (ISO) can be added here as fixtures need it.
    """
    folded = _PCT.sub("%", text.lower())
    return _WS.sub(" ", folded).strip()


def score_case(
    case: ChatEvalInput,
    result: AnswerResult,
    *,
    json_valid: bool,
    claimed_ids: list[str] | None = None,
) -> CaseScore:
    """Score one model answer against the gold fixture. Pure; no I/O, no logging.

    ``json_valid`` is supplied by the runner (it knows whether the provider's
    structured-output parse succeeded), so this function stays model-agnostic and
    unit-testable with canned :class:`AnswerResult` + flag.

    ``claimed_ids`` are the RAW source ids the model claimed, captured BEFORE
    ``RAG._validate_citations`` drops hallucinated ones. Precision is scored from
    these (so a claim of ``["S1","FAKE9"]`` fails) — falling back to the validated
    citations only when raw claims are unavailable (e.g. canned unit tests).
    """
    allowed = {s.sid for s in case.sources}
    cited = {_sid_for(c, case) for c in result.citations}
    cited.discard(None)
    # Precision uses raw claims when available; else the (already-validated) cited set.
    precision_ids = set(claimed_ids) if claimed_ids is not None else cited
    answer_norm = normalize(result.answer)

    reasons: list[str] = []

    # -- refusal axis (worst failure first): grounding must match expectation.
    refusal_ok = result.grounded == case.expect_grounded
    if not refusal_ok:
        reasons.append("refusal" if case.expect_grounded else "hallucinated")

    # -- faithfulness, bound to citations.
    facts_ok = True
    cite_complete = True
    if case.expect_grounded:
        for fact in case.facts:
            if not any(normalize(p) in answer_norm for p in fact.any_of):
                facts_ok = False
            if not set(fact.required_source_ids).issubset(cited):
                cite_complete = False
        if not facts_ok:
            reasons.append("fact_missing")
        if not cite_complete:
            reasons.append("citation_incomplete")

    # -- citation precision: no hallucinated / out-of-context ids claimed (raw claims).
    cite_precise = precision_ids.issubset(allowed)
    if not cite_precise:
        reasons.append("citation_imprecise")

    # -- forbidden / distractor leakage (and injection sentinels).
    forbidden_clean = not any(normalize(t) in answer_norm for t in case.forbidden)
    if not forbidden_clean:
        reasons.append("forbidden_leak")

    # -- temporal window: every citation ts must fall inside the requested span.
    window_ok = True
    if case.window is not None:
        start, end = case.window
        window_ok = all(start <= c.ts <= end for c in result.citations)
        if not window_ok:
            reasons.append("window_violation")

    if not json_valid:
        reasons.append("json_invalid")

    return CaseScore(
        id=case.id,
        mode=case.mode,
        json_valid=json_valid,
        facts_ok=facts_ok,
        citations_precise=cite_precise,
        citations_complete=cite_complete,
        forbidden_clean=forbidden_clean,
        refusal_ok=refusal_ok,
        window_ok=window_ok,
        reason_codes=tuple(reasons),
    )


def _sid_for(citation, case: ChatEvalInput) -> str | None:
    """Resolve a validated Citation back to its fixture source id.

    Lane A assigns S1..Sn positionally to the fake hits in fixture order, and the
    RAG validator preserves that mapping via the observation id, so we recover the
    source id from the observation id we stamped (``eval:<sid>``).
    """
    obs_id = getattr(citation, "observation_id", "") or ""
    if obs_id.startswith("eval:"):
        # Return the sid unconditionally (even if not an allowed source) so the
        # precision check (cited ⊆ allowed) can flag a hallucinated id. In the real
        # RAG path hallucinated ids are dropped before becoming Citations, so
        # precision is always satisfied there; this only bites a fake provider.
        return obs_id.split(":", 1)[1]
    return None


# --------------------------------------------------------------------------- #
# Lane A runner: fixed retrieval, real RAG + real provider                    #
# --------------------------------------------------------------------------- #


class _FixedSearcher:
    """A ``_Searcher`` that returns pre-baked hits — retrieval is held constant.

    This isolates the variable under test (the model's generation + citation
    behavior) from retrieval quality. The hits carry real ``Observation``s whose
    ids encode the fixture source id (``eval:<sid>``) so scoring can map a
    validated citation back to its gold source.
    """

    def __init__(self, case: ChatEvalInput) -> None:
        self._hits = [
            SearchHit(
                chunk_id=f"eval:{s.sid}:chunk",
                content_hash=f"eval:{s.sid}",
                text=s.text,
                score=float(len(case.sources) - i),  # fixture order = rank order
                observation=Observation(
                    id=f"eval:{s.sid}",
                    content_hash=f"eval:{s.sid}",
                    ts=s.ts,
                    app=s.app,
                    window=s.window,
                    source="eval",
                    session_id=f"eval:{case.id}",
                ),
            )
            for i, s in enumerate(case.sources)
        ]

    def search(self, query: str, k: int = 10, *, semantic: bool = True) -> list[SearchHit]:
        return list(self._hits[:k])

    def time_range_text(self, start: float, end: float) -> list[tuple[Observation, str]]:
        """Back the temporal path: occurrences whose ts falls in the window."""
        return [
            (h.observation, h.text)
            for h in self._hits
            if h.observation is not None and start <= h.observation.ts <= end
        ]


@dataclass
class ChatEvalReport:
    """Aggregate, content-free chat-eval metrics across all cases for one model."""

    scores: tuple[CaseScore, ...]
    model: str

    @property
    def total(self) -> int:
        return len(self.scores)

    @property
    def passed(self) -> bool:
        return all(s.passed for s in self.scores)


def run_chat_eval(
    cases: Iterable[ChatEvalInput],
    *,
    provider,
    model: str,
    max_context: int = 6,
) -> ChatEvalReport:
    """Run Lane A over ``cases`` with a real ``provider`` and score each answer.

    ``provider`` is any object exposing ``.complete`` (the production
    :class:`LLMProvider`, or a fake for tests). The runner builds a fixed-retrieval
    :class:`RAG`, asks it each question, detects JSON validity from the provider
    response shape, and scores deterministically.
    """
    recorder = _JsonValidRecorder(provider)
    scores: list[CaseScore] = []
    for case in cases:
        # temporal-phrase exercises RAG's phrase router (not an explicit window), so
        # stamp the sources into "yesterday" relative to a FIXED clock and pin
        # RAG._now to match — making the phrase path deterministic and reproducible.
        if case.mode == "temporal-phrase":
            case = _stamp_yesterday(case)
        rag = RAG(_FixedSearcher(case), recorder, max_context=max_context)
        if case.mode == "temporal-phrase":
            rag._now = lambda: EVAL_NOW
        recorder.reset()
        json_valid = True
        try:
            if case.window is not None:
                result = rag.answer(case.question, window=case.window)
            else:
                result = rag.answer(case.question)
        except Exception:
            result = AnswerResult(answer="", citations=[], grounded=False)
            json_valid = False
        else:
            # The structured call succeeded only if the provider returned a parsed
            # dict. An empty string / best-effort str fallback (e.g. qwen3:4b's empty
            # json_object output) means JSON generation FAILED — do not let "didn't
            # throw" read as valid JSON.
            json_valid = recorder.last_json_valid
        scores.append(
            score_case(
                case, result, json_valid=json_valid, claimed_ids=recorder.last_claimed_ids
            )
        )
    return ChatEvalReport(scores=tuple(scores), model=model)


class _JsonValidRecorder:
    """Wrap a provider to record structured-output validity + the RAW claimed ids.

    ``LLMProvider.complete(json_schema=...)`` returns a parsed ``dict`` on success, but
    it ALSO returns a parsed dict on the best-effort fallback path AFTER schema
    validation fails — so "is a dict" is not enough. We treat output as json-valid only
    when it is a dict carrying the schema's required shape (a string ``answer`` and a
    list ``citations``); an empty string / prose fallback / shape-invalid dict is
    json-invalid. We also capture the raw ``citations`` the model claimed, BEFORE
    ``RAG._validate_citations`` drops hallucinated ids, so precision can be scored
    honestly. ``last_claimed_ids`` is ``None`` when no structured call was made.
    """

    def __init__(self, provider) -> None:
        self._provider = provider
        self.last_json_valid = True
        self.last_claimed_ids: list[str] | None = None

    def reset(self) -> None:
        self.last_json_valid = True
        self.last_claimed_ids = None

    def complete(self, messages: list[dict], *, json_schema: dict | None = None):
        raw = self._provider.complete(messages, json_schema=json_schema)
        if json_schema is not None:
            cites = raw.get("citations") if isinstance(raw, dict) else None
            valid_shape = (
                isinstance(raw, dict)
                and isinstance(raw.get("answer"), str)
                and isinstance(cites, list)
                # schema requires citations.items.type == "string" (rag.py:_RESPONSE_SCHEMA);
                # a list with non-string items (e.g. [1]) is schema-invalid.
                and all(isinstance(c, str) for c in cites)
            )
            self.last_json_valid = valid_shape
            if isinstance(raw, dict) and isinstance(raw.get("citations"), list):
                self.last_claimed_ids = [str(c) for c in raw["citations"]]
            else:
                self.last_claimed_ids = []
        return raw


def _stamp_yesterday(case: ChatEvalInput) -> ChatEvalInput:
    """Return a copy of ``case`` with every source ts placed inside yesterday.

    Spreads sources across yesterday's span so the temporal scan returns them in
    order, relative to the fixed :data:`EVAL_NOW` the runner pins onto RAG._now.
    Mirrors RAG._temporal_window's calendar-day computation (yesterday = the day
    before today's local midnight), so the stamped ts land inside the window RAG
    derives from the phrase — not merely "24h ago".
    """
    import dataclasses
    import datetime as _dt

    today_midnight = _dt.datetime.fromtimestamp(EVAL_NOW).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    yday_midnight = (today_midnight - _dt.timedelta(days=1)).timestamp()
    stamped = tuple(
        dataclasses.replace(s, ts=yday_midnight + (i + 1) * 3600.0)
        for i, s in enumerate(case.sources)
    )
    return dataclasses.replace(case, sources=stamped)


# --------------------------------------------------------------------------- #
# Fixture loading                                                             #
# --------------------------------------------------------------------------- #


def load_chat_eval_jsonl(path: str | Path) -> list[ChatEvalInput]:
    """Load and validate chat-eval fixtures from UTF-8 JSONL.

    Raises ``ValueError`` with a ``file:line`` prefix on any malformed row. Error
    messages reference field NAMES and line numbers only — never fixture text —
    so the content-free rule holds on the validation path too.
    """
    fixture = Path(path)
    cases: list[ChatEvalInput] = []
    with fixture.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{fixture}:{line_no}: invalid JSON") from exc
            cases.append(_case_from_obj(obj, fixture=fixture, line_no=line_no))
    if not cases:
        raise ValueError(f"{fixture}: no eval cases found")
    return cases


def _case_from_obj(obj: object, *, fixture: Path, line_no: int) -> ChatEvalInput:
    where = f"{fixture}:{line_no}"
    if not isinstance(obj, dict):
        raise ValueError(f"{where}: expected a JSON object")
    for name in ("id", "question", "mode", "sources"):
        if name not in obj:
            raise ValueError(f"{where}: missing required field: {name!r}")
    mode = obj["mode"]
    if mode not in VALID_MODES:
        raise ValueError(f"{where}: mode must be one of {sorted(VALID_MODES)}")

    raw_sources = obj["sources"]
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError(f"{where}: 'sources' must be a non-empty list")
    sources = tuple(_source_from_obj(s, where=where) for s in raw_sources)

    facts = tuple(_fact_from_obj(f, where=where) for f in obj.get("facts", []))
    forbidden = tuple(str(t) for t in obj.get("forbidden", []))
    expect_grounded = bool(obj.get("expect_grounded", True))
    window = _window_from_obj(obj.get("window"), where=where)

    return ChatEvalInput(
        id=str(obj["id"]),
        question=str(obj["question"]),
        mode=str(mode),
        sources=sources,
        facts=facts,
        forbidden=forbidden,
        expect_grounded=expect_grounded,
        window=window,
    )


def _num(value: object, *, field: str, where: str, cast=float):
    """Coerce ``value`` with ``cast``, raising a CONTENT-FREE error on failure.

    Names only the field + line — never echoes the bad value — so the CLI's
    ``Invalid ... fixture: {exc}`` surface cannot leak fixture content.
    """
    try:
        return cast(value)
    except (TypeError, ValueError):
        raise ValueError(f"{where}: {field!r} must be a {cast.__name__}") from None


def _source_from_obj(obj: object, *, where: str) -> GoldSource:
    if not isinstance(obj, dict) or "sid" not in obj or "text" not in obj:
        raise ValueError(f"{where}: each source needs 'sid' and 'text'")
    return GoldSource(
        sid=str(obj["sid"]),
        text=str(obj["text"]),
        app=str(obj["app"]) if obj.get("app") is not None else None,
        window=str(obj["window"]) if obj.get("window") is not None else None,
        ts=_num(obj.get("ts", 0.0), field="ts", where=where),
    )


def _fact_from_obj(obj: object, *, where: str) -> GoldFact:
    if not isinstance(obj, dict) or "any_of" not in obj:
        raise ValueError(f"{where}: each fact needs 'any_of'")
    any_of = obj["any_of"]
    if not isinstance(any_of, list) or not any_of:
        raise ValueError(f"{where}: 'any_of' must be a non-empty list")
    req = obj.get("required_source_ids", [])
    if not isinstance(req, list):
        raise ValueError(f"{where}: 'required_source_ids' must be a list")
    return GoldFact(
        any_of=tuple(str(p) for p in any_of),
        required_source_ids=tuple(str(s) for s in req),
    )


def _window_from_obj(obj: object, *, where: str) -> tuple[float, float] | None:
    if obj is None:
        return None
    if not isinstance(obj, list) or len(obj) != 2:
        raise ValueError(f"{where}: 'window' must be [start_ts, end_ts]")
    return _num(obj[0], field="window[0]", where=where), _num(obj[1], field="window[1]", where=where)


def chat_eval_report_payload(report: ChatEvalReport) -> dict[str, object]:
    """Return a JSON-serializable, content-free report.

    Carries per-case ids, mode, the boolean axis outcomes and reason codes, plus
    aggregate rates — never the question, source text, or model answer body.
    """
    n = report.total

    def rate(attr: str) -> float | None:
        if not n:
            return None
        return round(sum(1 for s in report.scores if getattr(s, attr)) / n, 3)

    return {
        "model": report.model,
        "passed": report.passed,
        "total_cases": n,
        "pass_rate": rate("passed"),
        "json_valid_rate": rate("json_valid"),
        "facts_ok_rate": rate("facts_ok"),
        "citations_precise_rate": rate("citations_precise"),
        "citations_complete_rate": rate("citations_complete"),
        "forbidden_clean_rate": rate("forbidden_clean"),
        "refusal_ok_rate": rate("refusal_ok"),
        "window_ok_rate": rate("window_ok"),
        "cases": [
            {
                "id": s.id,
                "mode": s.mode,
                "passed": s.passed,
                "reason_codes": list(s.reason_codes),
            }
            for s in report.scores
        ],
    }


__all__ = [
    "VALID_MODES",
    "GoldSource",
    "GoldFact",
    "ChatEvalInput",
    "CaseScore",
    "ChatEvalReport",
    "normalize",
    "score_case",
    "run_chat_eval",
    "load_chat_eval_jsonl",
    "chat_eval_report_payload",
]
