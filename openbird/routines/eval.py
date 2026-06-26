"""Offline prose-adherence + faithfulness eval for the daily briefing path.

Measures whether a local model honors the briefing's "clean prose" contract when run
through the REAL routine path (`build_routine_messages` + the actual `yesterday`
template prompt + `provider.complete` as PLAIN text — no json_object). Motivated by a
real qwen3:8b briefing that came out as structured markdown; see
``docs/design/briefing-prose-eval.md`` (Codex-approved).

Scoring is deterministic (no LLM judge), uses line-anchored regexes that do NOT
false-positive on ordinary prose (colons, em-dashes, inline numbers), and scores the
RAW model output — which on this tree is what the Today card renders (no Swift
normalizer wired into ``TodayView``). Reports are content-free (ids/booleans/counts).
"""

from __future__ import annotations

import hashlib
import json
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from openbird.prompts import render
from openbird.routines.templates import (
    _ROUTINE_PROMPT,
    _resolve_system_prompt,
    build_routine_messages,
    get_template,
    render_context_text,
)
from openbird.types import Observation

# Candidate prose persona (re-instates the contract dropped in HEAD 7a7fd4a, from
# commit 6b96c2d) — the "treatment" variant measured against the current "well-
# structured" persona baseline.
CANDIDATE_PROSE_PERSONA = (
    "Write the briefing as plain flowing prose grounded only in that data: a single "
    "short paragraph of a few sentences. Use **bold** sparingly for at most a couple "
    "of key terms. Do NOT use headings, bullet or numbered lists, horizontal rules, "
    "section labels, or any reasoning narration. If the window contains no activity, "
    "say so plainly in one sentence."
)

VALID_VARIANTS: frozenset[str] = frozenset({"current", "candidate"})

# -- line-anchored structure regexes (from Codex review; prose-safe) ---------- #
_HEADING_ATX = re.compile(r"(?m)^\s{0,3}#{1,6}\s+\S")
_HEADING_BOLD = re.compile(r"(?m)^\s{0,3}\*\*[^*\n]{1,80}:?\*\*\s*$")
_HEADING_COLON = re.compile(r"(?m)^\s{0,3}[A-Z][A-Za-z0-9 /&()'-]{2,60}:\s*$")
_NUMBERED_HEADING = re.compile(r"(?m)^\s{0,3}\d{1,2}[.)]\s+[A-Z][^.!?\n]{1,80}:?\s*$")
_BULLET = re.compile(r"(?m)^\s{0,3}(?:[-*+•])\s+\S")
_ORDERED = re.compile(r"(?m)^\s{0,3}\d{1,2}[.)]\s+\S")
_HRULE = re.compile(r"(?m)^\s{0,3}(?:-{3,}|\*{3,}|_{3,})\s*$")
_REASONING = [
    re.compile(r"(?is)<think>.*?</think>"),
    re.compile(r"(?im)^\s*(summary of observations|final answer)\s*:?\s*$"),
    re.compile(r"(?im)^\s*here(?:'s| is)\s+(?:a\s+)?(?:structured\s+)?breakdown\b"),
    re.compile(r"(?im)^\s*let(?:'s| us)\s+(?:see|think|analy[sz]e)\b"),
]
_HEADINGS = (_HEADING_ATX, _HEADING_BOLD, _HEADING_COLON, _NUMBERED_HEADING)
_LISTS = (_BULLET, _ORDERED)
_WORD = re.compile(r"\b[\w']+\b")
_SENT = re.compile(r"[.!?]+(?:\s|$)")
_MD_SYMBOL = re.compile(r"(?m)(^\s{0,3}#{1,6}\s|\*\*|^\s{0,3}[-*+•]\s|^\s{0,3}-{3,}\s*$)")


# --------------------------------------------------------------------------- #
# Fixture schema                                                              #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BriefObservation:
    """One synthetic captured row fed into the briefing context."""

    ts: float
    app: str
    text: str
    window: str | None = None


@dataclass(frozen=True)
class Fact:
    """A fact the briefing must convey; satisfied if ANY phrase appears (normalized)."""

    any_of: tuple[str, ...]


@dataclass(frozen=True)
class BriefingEvalInput:
    """One labeled briefing-eval fixture row."""

    id: str
    observations: tuple[BriefObservation, ...]
    required_facts: tuple[Fact, ...] = ()
    forbidden: tuple[str, ...] = ()
    conflicting_facts: tuple[str, ...] = ()
    max_words: int = 90
    max_sentences: int = 6
    max_paragraphs: int = 1


# --------------------------------------------------------------------------- #
# Scoring (pure / deterministic — unit-tested with canned outputs)           #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ProseScore:
    """Per-(case, repeat) outcome. Content-free."""

    id: str
    no_headings: bool
    no_lists: bool
    no_rules: bool
    no_reasoning: bool
    length_ok: bool
    facts_ok: bool
    forbidden_clean: bool
    md_symbol_count: int  # advisory: how much a normalizer would have to strip
    reason_codes: tuple[str, ...]

    @property
    def prose_clean(self) -> bool:
        return self.no_headings and self.no_lists and self.no_rules and self.no_reasoning and self.length_ok

    @property
    def grounded(self) -> bool:
        return self.facts_ok and self.forbidden_clean

    @property
    def passed(self) -> bool:
        return not self.reason_codes


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def score_briefing(text: str, case: BriefingEvalInput, *, errored: bool = False) -> ProseScore:
    """Score one raw briefing output against the prose + faithfulness contract. Pure.

    ``errored`` marks a provider failure (the runner caught an exception). An errored
    or empty/whitespace output is an UNCONDITIONAL failure — a model crash or blank
    briefing must never read as a clean pass, even for a fixture with no required
    facts. (Codex diff review: empty text was passing such fixtures.)
    """
    reasons: list[str] = []
    if errored:
        reasons.append("model_error")
    if not text.strip():
        reasons.append("empty_output")

    no_headings = not any(p.search(text) for p in _HEADINGS)
    no_lists = not any(p.search(text) for p in _LISTS)
    no_rules = not _HRULE.search(text)
    no_reasoning = not any(p.search(text) for p in _REASONING)
    if not no_headings:
        reasons.append("headings")
    if not no_lists:
        reasons.append("lists")
    if not no_rules:
        reasons.append("rules")
    if not no_reasoning:
        reasons.append("reasoning_narration")

    words = len(_WORD.findall(text))
    sentences = len(_SENT.findall(text)) or (1 if text.strip() else 0)
    paragraphs = len([p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()])
    length_ok = (
        words <= case.max_words
        and sentences <= case.max_sentences
        and paragraphs <= case.max_paragraphs
    )
    if not length_ok:
        reasons.append("too_long")

    norm = _norm(text)
    facts_ok = all(any(_norm(p) in norm for p in f.any_of) for f in case.required_facts)
    if not facts_ok:
        reasons.append("fact_missing")
    leaks = [t for t in (*case.forbidden, *case.conflicting_facts) if _norm(t) in norm]
    forbidden_clean = not leaks
    if not forbidden_clean:
        reasons.append("forbidden_leak")

    return ProseScore(
        id=case.id,
        no_headings=no_headings,
        no_lists=no_lists,
        no_rules=no_rules,
        no_reasoning=no_reasoning,
        length_ok=length_ok,
        facts_ok=facts_ok,
        forbidden_clean=forbidden_clean,
        md_symbol_count=len(_MD_SYMBOL.findall(text)),
        reason_codes=tuple(reasons),
    )


# --------------------------------------------------------------------------- #
# Prompt-variant message building (uses the REAL routine path)               #
# --------------------------------------------------------------------------- #


def _system_prompt_for(variant: str) -> str:
    if variant == "current":
        return _resolve_system_prompt()
    if variant == "candidate":
        return render(_ROUTINE_PROMPT, CANDIDATE_PROSE_PERSONA)
    raise ValueError(f"unknown variant {variant!r}; expected one of {sorted(VALID_VARIANTS)}")


def build_briefing_messages(case: BriefingEvalInput, *, variant: str) -> list[dict]:
    """Build the real routine messages for ``case`` under prompt ``variant``.

    Uses the actual ``yesterday`` template prompt (which itself contains the
    structure-inviting "grouped sensibly" phrase in BOTH variants) and the same
    context renderer the product uses, so only the system persona differs.
    """
    rows = [
        (
            Observation(
                id=f"brief:{case.id}:{i}",
                content_hash=f"brief:{case.id}:{i}",
                ts=o.ts,
                app=o.app,
                window=o.window,
                source="eval",
            ),
            o.text,
        )
        for i, o in enumerate(case.observations)
    ]
    context = render_context_text(rows)
    user_prompt = get_template("yesterday").prompt
    return build_routine_messages(_system_prompt_for(variant), user_prompt, 0.0, 86_400.0, context)


def prompt_hashes(variant: str) -> dict[str, str]:
    sysp = _system_prompt_for(variant)
    persona = CANDIDATE_PROSE_PERSONA if variant == "candidate" else "(active)"
    return {
        "system_prompt_hash": hashlib.sha256(sysp.encode()).hexdigest()[:16],
        "persona_hash": hashlib.sha256(persona.encode()).hexdigest()[:16],
    }


# --------------------------------------------------------------------------- #
# Runner (repeats → pass-probability) + report                               #
# --------------------------------------------------------------------------- #


@dataclass
class BriefingEvalReport:
    model: str
    variant: str
    repeats: int
    scores: list[ProseScore] = field(default_factory=list)  # len = cases * repeats

    @property
    def passed(self) -> bool:
        return all(s.passed for s in self.scores)


def run_briefing_eval(
    cases: Iterable[BriefingEvalInput],
    *,
    provider,
    model: str,
    variant: str = "current",
    repeats: int = 5,
) -> BriefingEvalReport:
    """Run the briefing eval R times per case under one prompt variant, scoring raw output."""
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    case_list = list(cases)
    scores: list[ProseScore] = []
    for _ in range(repeats):
        for case in case_list:
            messages = build_briefing_messages(case, variant=variant)
            errored = False
            try:
                raw = provider.complete(messages)
                text = raw if isinstance(raw, str) else str(raw)
            except Exception:
                text = ""
                errored = True
            scores.append(score_briefing(text, case, errored=errored))
    return BriefingEvalReport(model=model, variant=variant, repeats=repeats, scores=scores)


def briefing_eval_report_payload(report: BriefingEvalReport) -> dict[str, object]:
    """Content-free aggregate: per-case pass-probability + per-reason rates."""
    n = len(report.scores)

    def rate(pred) -> float | None:
        return round(statistics.mean([1.0 if pred(s) else 0.0 for s in report.scores]), 3) if n else None

    by_case: dict[str, list[ProseScore]] = {}
    for s in report.scores:
        by_case.setdefault(s.id, []).append(s)
    reason_counts: dict[str, int] = {}
    for s in report.scores:
        for r in s.reason_codes:
            reason_counts[r] = reason_counts.get(r, 0) + 1

    return {
        "model": report.model,
        "variant": report.variant,
        "repeats": report.repeats,
        **prompt_hashes(report.variant),
        "pass_rate": rate(lambda s: s.passed),
        "prose_clean_rate": rate(lambda s: s.prose_clean),
        "grounded_rate": rate(lambda s: s.grounded),
        "mean_md_symbols": round(statistics.mean([s.md_symbol_count for s in report.scores]), 2) if n else None,
        "reason_rates": {k: round(v / n, 3) for k, v in sorted(reason_counts.items())} if n else {},
        "cases": [
            {
                "id": cid,
                "pass_prob": round(statistics.mean([1.0 if s.passed else 0.0 for s in ss]), 3),
                "prose_clean_prob": round(statistics.mean([1.0 if s.prose_clean else 0.0 for s in ss]), 3),
                "flaky": 0 < sum(s.passed for s in ss) < len(ss),
            }
            for cid, ss in by_case.items()
        ],
    }


# --------------------------------------------------------------------------- #
# Fixture loading                                                            #
# --------------------------------------------------------------------------- #


def load_briefing_eval_jsonl(path: str | Path) -> list[BriefingEvalInput]:
    """Load + validate briefing fixtures from UTF-8 JSONL (content-free errors)."""
    fixture = Path(path)
    cases: list[BriefingEvalInput] = []
    with fixture.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{fixture}:{line_no}: invalid JSON") from exc
            cases.append(_case_from_obj(obj, fixture=fixture, line_no=line_no))
    if not cases:
        raise ValueError(f"{fixture}: no eval cases found")
    return cases


def _case_from_obj(obj: object, *, fixture: Path, line_no: int) -> BriefingEvalInput:
    where = f"{fixture}:{line_no}"
    if not isinstance(obj, dict):
        raise ValueError(f"{where}: expected a JSON object")
    for name in ("id", "observations"):
        if name not in obj:
            raise ValueError(f"{where}: missing required field: {name!r}")
    raw_obs = obj["observations"]
    if not isinstance(raw_obs, list) or not raw_obs:
        raise ValueError(f"{where}: 'observations' must be a non-empty list")
    obs = tuple(_obs_from(o, where=where) for o in raw_obs)
    facts = tuple(_fact_from(f, where=where) for f in obj.get("required_facts", []))
    return BriefingEvalInput(
        id=str(obj["id"]),
        observations=obs,
        required_facts=facts,
        forbidden=tuple(str(t) for t in obj.get("forbidden", [])),
        conflicting_facts=tuple(str(t) for t in obj.get("conflicting_facts", [])),
        max_words=_num(obj.get("max_words", 90), field="max_words", where=where, cast=int),
        max_sentences=_num(obj.get("max_sentences", 6), field="max_sentences", where=where, cast=int),
        max_paragraphs=_num(obj.get("max_paragraphs", 1), field="max_paragraphs", where=where, cast=int),
    )


def _num(value: object, *, field: str, where: str, cast=float):
    """Coerce ``value`` with ``cast``, raising a CONTENT-FREE error (field + line only)."""
    try:
        return cast(value)
    except (TypeError, ValueError):
        raise ValueError(f"{where}: {field!r} must be a {cast.__name__}") from None


def _obs_from(o: object, *, where: str) -> BriefObservation:
    if not isinstance(o, dict) or "ts" not in o or "app" not in o or "text" not in o:
        raise ValueError(f"{where}: each observation needs 'ts', 'app', 'text'")
    return BriefObservation(
        ts=_num(o["ts"], field="ts", where=where),
        app=str(o["app"]),
        text=str(o["text"]),
        window=str(o["window"]) if o.get("window") is not None else None,
    )


def _fact_from(f: object, *, where: str) -> Fact:
    if not isinstance(f, dict) or "any_of" not in f:
        raise ValueError(f"{where}: each required_fact needs 'any_of'")
    any_of = f["any_of"]
    if not isinstance(any_of, list) or not any_of:
        raise ValueError(f"{where}: 'any_of' must be a non-empty list")
    return Fact(any_of=tuple(str(p) for p in any_of))


__all__ = [
    "CANDIDATE_PROSE_PERSONA",
    "VALID_VARIANTS",
    "BriefObservation",
    "Fact",
    "BriefingEvalInput",
    "ProseScore",
    "BriefingEvalReport",
    "score_briefing",
    "build_briefing_messages",
    "prompt_hashes",
    "run_briefing_eval",
    "briefing_eval_report_payload",
    "load_briefing_eval_jsonl",
]
