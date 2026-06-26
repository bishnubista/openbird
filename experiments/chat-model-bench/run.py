"""Multi-model answer-quality benchmark for OpenBird's local RAG chat.

Thin driver: imports the COMMITTED, Codex-approved eval core
(``openbird.chat.eval``) and runs it against several candidate local models, then
prints a comparison + a recommendation. Keeping the N-model orchestration here (not
in the shipped CLI) follows the design doc's resolved decisions.

Run:  uv run python experiments/chat-model-bench/run.py --models ollama/qwen3:8b ollama/llama3.2
Records a run fingerprint (ollama version, model digest, litellm version, prompt/
schema hash, gen params) so a scorecard is comparable across time — Ollama tags are
mutable, so a bare model name is not a stable identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

# Make the repo importable when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from openbird.chat.eval import (  # noqa: E402
    chat_eval_report_payload,
    load_chat_eval_jsonl,
    run_chat_eval,
)

DEFAULT_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "tests" / "fixtures" / "chat" / "answer_quality.jsonl"
)
DEFAULT_MODELS = ["ollama/qwen3:8b", "ollama/llama3.2"]
# Rank weights: refusal correctness and citation-bound faithfulness matter most for
# a privacy/provenance product; raw json validity is table stakes.
WEIGHTS = {
    "refusal_ok_rate": 0.30,
    "citations_complete_rate": 0.25,
    "facts_ok_rate": 0.20,
    "forbidden_clean_rate": 0.15,
    "json_valid_rate": 0.10,
}


def _cmd(args: list[str]) -> str:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return ""


def _ollama_ids() -> dict[str, str]:
    """Map ``name[:tag] -> short id`` from ``ollama list`` (the stable digest)."""
    out: dict[str, str] = {}
    for line in _cmd(["ollama", "list"]).splitlines()[1:]:  # skip header
        parts = line.split()
        if len(parts) >= 2:
            out[parts[0]] = parts[1]
    return out


def fingerprint(models: list[str], fixture: Path) -> dict:
    try:
        from importlib.metadata import version
        litellm_v = version("litellm")
    except Exception:
        litellm_v = "unavailable"
    ids = _ollama_ids()
    digests = {}
    for m in models:
        bare = m.split("/", 1)[1] if "/" in m else m
        # Tag-aware: a bare name (no ':') resolves to the ':latest' row.
        key = bare if ":" in bare else f"{bare}:latest"
        digests[m] = ids.get(bare) or ids.get(key) or "n/a"
    prompt_hash = hashlib.sha256(fixture.read_bytes()).hexdigest()[:16]
    return {
        "ollama_version": _cmd(["ollama", "--version"]),
        "litellm_version": litellm_v,
        "model_digests": digests,
        "fixture_sha256_16": prompt_hash,
        "gen_params": "litellm/ollama defaults (provider-controlled)",
    }


def composite(payload: dict) -> float:
    total = 0.0
    for key, w in WEIGHTS.items():
        v = payload.get(key)
        total += w * (v if isinstance(v, (int, float)) else 0.0)
    return round(total, 3)


def build_provider(model: str):
    import dataclasses

    from openbird.config import get_settings
    from openbird.llm.provider import LLMProvider

    # Override only the model; keep the user's configured host/timeouts/etc.
    settings = dataclasses.replace(get_settings(), llm_model=model)
    return LLMProvider(settings)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    ap.add_argument("--out", type=Path, default=Path(__file__).with_name("results.json"))
    args = ap.parse_args(argv or sys.argv[1:])

    cases = load_chat_eval_jsonl(args.fixture)
    fp = fingerprint(args.models, args.fixture)
    rows = []
    for model in args.models:
        try:
            provider = build_provider(model)
            report = run_chat_eval(cases, provider=provider, model=model)
            payload = chat_eval_report_payload(report)
            payload["composite"] = composite(payload)
            rows.append(payload)
            print(f"[ok] {model}: composite={payload['composite']} pass_rate={payload['pass_rate']}")
        except Exception as exc:  # content-free: type only
            rows.append({"model": model, "error": type(exc).__name__})
            print(f"[err] {model}: {type(exc).__name__}")

    ranked = sorted(
        (r for r in rows if "composite" in r), key=lambda r: r["composite"], reverse=True
    )
    out = {"fingerprint": fp, "results": rows,
           "recommendation": ranked[0]["model"] if ranked else None}
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n=== Ranking (composite) ===")
    for r in ranked:
        print(f"  {r['composite']:>5}  {r['model']}")
    print(f"\nRecommendation: {out['recommendation']}")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
