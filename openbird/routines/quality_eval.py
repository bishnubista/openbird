"""On-demand quality eval for the briefing + ask surfaces, over the real store
with the configured (local) model.

This is a *manual pre-PR gate*, not a CI unit test: it makes live LLM calls, so
it is non-deterministic and slow. It runs each surface N times and checks the
acceptance gates that the deterministic unit tests cannot (faithful prose,
grounded answers, no self-capture leakage) — exercising the SAME code paths as
``openbird chat`` and ``openbird briefing``.

Gates (a surface passes when a MAJORITY of its runs pass):
  * Ask ("Summarize my day", "What should I follow up on?", "What did I work on
    yesterday?"): the answer is grounded with >=1 citation, and no citation points
    at OpenBird's own UI (self-capture). Additionally, the per-query grounded rate
    must clear :data:`GROUNDED_RATE_FLOOR` — a strict majority alone can hide a
    flaky grounding gate (e.g. "Summarize my day" grounded only 2/5 on a small
    model before the synthesis-persona fix), so the floor fails grounding flakiness
    before it degrades into silent answer-blanking.
  * Day-scoped Ask: the same gate over the explicit ``--day`` path used by the
    Today view's "Ask about this day" button. This catches regressions where the
    generic intent path grounds but the app/CLI hard-scope path blanks.
  * Briefing (--day 0 and --day 1): the prose has zero ungrounded ``#N`` refs
    (numbers it invented that are absent from the grounding context), and no
    grounding source is self-capture. Briefing runs do not report a grounded flag,
    so the grounded-rate floor is vacuous for them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from openbird.capture.redact import _is_self_capture
from openbird.chat.rag import RAG
from openbird.routines.templates import (
    count_ungrounded_refs,
    get_template,
    render_context_text,
    select_briefing_sources,
)

ASK_QUERIES: tuple[str, ...] = (
    "Summarize my day",
    "What did I work on yesterday?",
    "What should I follow up on?",
)
DAY_SCOPED_ASK_QUERIES: tuple[tuple[int, str], ...] = (
    (0, "Summarize my day"),
    (0, "What did I work on?"),
    (0, "What should I follow up on?"),
    # Non-synthesis by design: drives the explicit-window specific-question
    # branch instead of the broad temporal summary selector.
    (0, "What about OpenBird?"),
)
BRIEFING_DAYS: tuple[int, ...] = (0, 1)

# Minimum fraction of grounding-reporting runs that must be grounded for an ask
# surface to pass. Set above the bare majority (0.5) so a query that grounds only
# ~half the time — flaky on a small local model — fails the gate instead of
# sneaking through on a coin flip. Tunable; 0.8 = at most one slip in five runs.
GROUNDED_RATE_FLOOR: float = 0.8


@dataclass
class CheckResult:
    """One surface (an ask query or a briefing day) across its runs."""

    label: str
    runs: list[dict] = field(default_factory=list)

    @property
    def grounded_rate(self) -> float | None:
        """Fraction of grounding-reporting runs that grounded, or ``None``.

        Only ask runs carry a ``grounded`` flag; briefing runs omit it, so this is
        ``None`` for a briefing check (and the floor in :meth:`passed` is skipped).
        """
        grounded_runs = [r for r in self.runs if "grounded" in r]
        if not grounded_runs:
            return None
        return sum(1 for r in grounded_runs if r.get("grounded")) / len(grounded_runs)

    @property
    def passed(self) -> bool:
        if not self.runs:
            return False
        ok = sum(1 for r in self.runs if r.get("ok"))
        # Strict majority: a tie (e.g. 1/2) is NOT a pass — a quality gate should
        # not green on a coin flip.
        if ok * 2 <= len(self.runs):
            return False
        # Grounded-rate floor (ask surfaces only): the majority check above can
        # still green on a query that grounds just over half the time, which is the
        # exact flakiness the synthesis-persona fix targets. Runs that report a
        # ``grounded`` flag must additionally clear GROUNDED_RATE_FLOOR. Briefing
        # runs omit the flag, so ``grounded_rate`` is None and the floor is vacuous.
        rate = self.grounded_rate
        return rate is None or rate >= GROUNDED_RATE_FLOOR


@dataclass
class QualityReport:
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(c.passed for c in self.checks)


def _eval_ask(
    rag: RAG,
    query: str,
    runs: int,
    *,
    window: tuple[float, float] | None = None,
    label: str | None = None,
) -> CheckResult:
    res = CheckResult(label=label or f"ask:{query}")
    for _ in range(runs):
        ans = rag.answer(query, window=window)
        self_cap = any(_is_self_capture(c.app) for c in ans.citations)
        ok = ans.grounded and len(ans.citations) >= 1 and not self_cap
        # Metadata/counts ONLY — never the answer text or citation snippets: this
        # payload feeds `eval quality --json`, which is likely saved to a log, and
        # the answer can contain captured text / window titles / #numbers.
        res.runs.append(
            {
                "ok": ok,
                "grounded": ans.grounded,
                "citations": len(ans.citations),
                "self_capture": self_cap,
            }
        )
    return res


def _eval_briefing(store: Any, provider: Any, day_window, day: int, runs: int) -> CheckResult:
    res = CheckResult(label=f"briefing:day{day}")
    start, end = day_window(day)
    rows = store.time_range_text(start, end, source="capture")
    context = render_context_text(rows)
    sources, _total = select_briefing_sources(rows)
    self_cap = any(_is_self_capture(s.get("app")) for s in sources)
    template = get_template("yesterday")
    for _ in range(runs):
        text = template.run_rows(provider, start, end, rows)
        ungrounded = count_ungrounded_refs(text, context)
        ok = ungrounded == 0 and not self_cap
        # Metadata/counts ONLY (see _eval_ask) — the briefing prose is never put
        # into the report payload.
        res.runs.append(
            {
                "ok": ok,
                "ungrounded_refs": ungrounded,
                "self_capture_source": self_cap,
            }
        )
    return res


def run_quality_eval(store: Any, provider: Any, *, day_window, runs: int = 3) -> QualityReport:
    """Run the ask + briefing quality gates over ``store``/``provider``.

    ``day_window`` is a callable ``day_offset -> (start_ts, end_ts)`` (the CLI's
    ``_day_window``), injected so the eval shares the briefing command's exact
    bounds. Returns a :class:`QualityReport`.
    """
    report = QualityReport()
    rag = RAG(store, provider)
    for q in ASK_QUERIES:
        report.checks.append(_eval_ask(rag, q, runs))
    for day, q in DAY_SCOPED_ASK_QUERIES:
        report.checks.append(
            _eval_ask(
                rag,
                q,
                runs,
                window=day_window(day),
                label=f"ask:day{day}:{q}",
            )
        )
    for day in BRIEFING_DAYS:
        report.checks.append(_eval_briefing(store, provider, day_window, day, runs))
    return report


def quality_eval_payload(report: QualityReport) -> dict:
    """Machine-readable summary for ``--json``."""
    return {
        "passed": report.passed,
        "checks": [
            {
                "label": c.label,
                "passed": c.passed,
                "grounded_rate": c.grounded_rate,
                "runs": c.runs,
            }
            for c in report.checks
        ],
    }
