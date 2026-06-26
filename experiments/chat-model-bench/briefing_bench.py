"""Briefing prose-adherence bench: current vs candidate persona across models.

Imports the committed core (openbird.routines.eval). Quantifies the regression (current
"well-structured" persona) vs the candidate prose persona, per model, with repeats.
Run (background it): uv run python experiments/chat-model-bench/briefing_bench.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import dataclasses  # noqa: E402

from openbird.config import get_settings  # noqa: E402
from openbird.llm.provider import LLMProvider  # noqa: E402
from openbird.routines.eval import (  # noqa: E402
    briefing_eval_report_payload,
    load_briefing_eval_jsonl,
    run_briefing_eval,
)

FIXTURE = ROOT / "tests" / "fixtures" / "routines" / "briefing_prose.jsonl"
OUT = Path(__file__).resolve().parent / "results" / "briefing_bench.json"
MODELS = ["ollama/qwen3:8b", "ollama/granite3.3:8b", "ollama/llama3.2", "ollama/gemma3:4b"]
REPEATS = 5


def main() -> int:
    cases = load_briefing_eval_jsonl(FIXTURE)
    rows = []
    for model in MODELS:
        provider = LLMProvider(dataclasses.replace(get_settings(), llm_model=model))
        for variant in ("current", "candidate"):
            report = run_briefing_eval(
                cases, provider=provider, model=model, variant=variant, repeats=REPEATS
            )
            p = briefing_eval_report_payload(report)
            rows.append(p)
            print(
                f"{model:22} [{variant:9}] prose_clean={p['prose_clean_rate']} "
                f"grounded={p['grounded_rate']} pass={p['pass_rate']} "
                f"md_symbols={p['mean_md_symbols']} reasons={p['reason_rates']}"
            )
    OUT.write_text(json.dumps({"results": rows}, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT}")
    # Delta summary: candidate prose_clean minus current, per model.
    print("\n=== prose_clean delta (candidate - current) ===")
    by = {(r["model"], r["variant"]): r for r in rows}
    for model in MODELS:
        cur = by.get((model, "current"), {}).get("prose_clean_rate") or 0
        cand = by.get((model, "candidate"), {}).get("prose_clean_rate") or 0
        print(f"  {model:22} {cur:.3f} -> {cand:.3f}  (Δ {cand - cur:+.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
