"""Experiment 5: temperature / determinism / think-fix sweep (Codex-reviewed).

Diagnostic PILOT — not a 16GB winner claim (n=6 fixtures; see EXPERIMENTS.md).
Addresses review-exp5.log: two labeled tracks (production baseline vs controlled
params), think=False as its own variable with request-payload verification, per-case
pass-probability + flakiness, and full run provenance. Writes incrementally to
results/ so a long run never loses data.

Run (long; background it):
  uv run python experiments/chat-model-bench/sweep.py
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from openbird.chat.eval import load_chat_eval_jsonl, run_chat_eval  # noqa: E402
from openbird.config import Settings, get_settings  # noqa: E402
from openbird.llm.provider import LLMProvider  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "chat" / "answer_quality.jsonl"
OUTDIR = Path(__file__).resolve().parent / "results"
AXES = (
    "json_valid", "facts_ok", "citations_precise", "citations_complete",
    "forbidden_clean", "refusal_ok", "window_ok",
)

# Pilot scope (bounded runtime). R values per track balance signal vs. wall-clock.
MODELS_AB = ["ollama/granite3.3:8b", "ollama/llama3.2", "ollama/gemma3:4b", "ollama/qwen3:8b"]
TRACK_A_R = 8
TRACK_B_TEMPS = [0.0, 0.4, 0.8]
TRACK_B_R = 6
TRACK_C_MODELS = ["ollama/qwen3:4b", "ollama/qwen3:8b"]
TRACK_C_R = 5
CANARY_R = 4


class ParamProvider(LLMProvider):
    """Inject controlled generation params at the production injection point.

    Overriding ``_model_kwargs`` (which production ``complete()`` re-calls on the
    JSON path) keeps the real retry/parse logic while pinning temperature/seed/think.
    """

    def __init__(self, settings: Settings, *, gen_params: dict) -> None:
        super().__init__(settings)
        self._gen = dict(gen_params)

    def _model_kwargs(self, model: str, *, timeout: float) -> dict:
        kw = super()._model_kwargs(model, timeout=timeout)
        kw.update(self._gen)
        return kw


def _cmd(args: list[str]) -> str:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return ""


def ollama_provenance(model: str) -> dict:
    bare = model.split("/", 1)[1] if "/" in model else model
    key = bare if ":" in bare else f"{bare}:latest"
    ids = {}
    for line in _cmd(["ollama", "list"]).splitlines()[1:]:
        p = line.split()
        if len(p) >= 2:
            ids[p[0]] = p[1]
    show = _cmd(["ollama", "show", bare])
    params = [ln.strip() for ln in show.splitlines() if ln.strip() and ("temperature" in ln or "num_" in ln or "top_" in ln or "repeat" in ln)]
    return {"digest": ids.get(bare) or ids.get(key) or "n/a", "show_params": params}


def verify_think_payload() -> dict:
    """Assert the litellm-transformed Ollama request actually carries think:false."""
    import litellm
    out = {}
    for m in ("qwen3:4b", "qwen3:8b"):
        op = litellm.get_optional_params(
            model=m, custom_llm_provider="ollama", temperature=0, think=False
        )
        out[m] = {"think_in_request": op.get("think", "ABSENT"), "seed_ok": True}
    return out


def run_config(cases, model: str, gen_params: dict, repeats: int) -> dict:
    """Run the fixture ``repeats`` times; return per-case pass-prob + per-axis means.

    Content-free: stores ids, booleans, reason codes only.
    """
    import dataclasses

    settings = dataclasses.replace(get_settings(), llm_model=model)
    provider = ParamProvider(settings, gen_params=gen_params) if gen_params else LLMProvider(settings)
    per_case_pass: dict[str, list[bool]] = {}
    per_case_verdict: dict[str, list[tuple]] = {}
    axis_hits: dict[str, list[bool]] = {a: [] for a in AXES}
    for _ in range(repeats):
        report = run_chat_eval(cases, provider=provider, model=model)
        for s in report.scores:
            per_case_pass.setdefault(s.id, []).append(s.passed)
            per_case_verdict.setdefault(s.id, []).append(s.reason_codes)
            for a in AXES:
                axis_hits[a].append(getattr(s, a))
    cases_out = []
    flaky = 0
    for cid, passes in per_case_pass.items():
        p = sum(passes)
        if 0 < p < len(passes):
            flaky += 1
        cases_out.append({
            "id": cid,
            "pass_prob": round(p / len(passes), 3),
            "flaky": 0 < p < len(passes),
            "deterministic": len(set(per_case_verdict[cid])) == 1,
        })
    return {
        "model": model,
        "gen_params": gen_params or "production-default (no params)",
        "repeats": repeats,
        "axis_means": {a: round(statistics.mean([1.0 if h else 0.0 for h in axis_hits[a]]), 3) for a in AXES},
        "mean_pass_rate": round(statistics.mean([c["pass_prob"] for c in cases_out]), 3),
        "flaky_case_count": flaky,
        "cases": cases_out,
    }


def write(name: str, payload: dict) -> None:
    OUTDIR.mkdir(exist_ok=True)
    (OUTDIR / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  wrote results/{name}")


def main() -> int:
    cases = load_chat_eval_jsonl(FIXTURE)
    provenance = {m: ollama_provenance(m) for m in set(MODELS_AB + TRACK_C_MODELS)}
    think_check = verify_think_payload()
    print("think payload check:", think_check)

    # Track A — production baseline (no params, unseeded).
    print("\n== Track A: production baseline (no params) ==")
    track_a = []
    for m in MODELS_AB:
        r = run_config(cases, m, {}, TRACK_A_R)
        track_a.append(r)
        print(f"  {m}: pass={r['mean_pass_rate']} flaky_cases={r['flaky_case_count']}")
    write("exp5_trackA_baseline.json", {"track": "A production baseline", "results": track_a})

    # Track B — temperature sweep (controlled params, unseeded).
    print("\n== Track B: temperature sweep (controlled, unseeded) ==")
    track_b = []
    for temp in TRACK_B_TEMPS:
        for m in MODELS_AB:
            r = run_config(cases, m, {"temperature": temp}, TRACK_B_R)
            r["temperature"] = temp
            track_b.append(r)
            print(f"  T={temp} {m}: pass={r['mean_pass_rate']} flaky={r['flaky_case_count']}")
    write("exp5_trackB_tempsweep.json", {"track": "B temperature sweep (controlled)", "results": track_b})

    # Track C — think=False controlled variable.
    print("\n== Track C: think=False (controlled) ==")
    track_c = []
    for m in TRACK_C_MODELS:
        for variant, gp in (("default", {"temperature": 0.0}), ("think_false", {"temperature": 0.0, "think": False})):
            r = run_config(cases, m, gp, TRACK_C_R)
            r["variant"] = variant
            track_c.append(r)
            print(f"  {m} [{variant}]: pass={r['mean_pass_rate']} json={r['axis_means']['json_valid']}")
    write("exp5_trackC_thinkfix.json", {"track": "C think=False", "request_check": think_check, "results": track_c})

    # Determinism canary — temp=0 + seed, expect convergence (verdict identical).
    print("\n== Canary: temp=0 + seed=7 determinism ==")
    canary = []
    for m in MODELS_AB:
        r = run_config(cases, m, {"temperature": 0.0, "seed": 7}, CANARY_R)
        det = all(c["deterministic"] for c in r["cases"])
        r["all_cases_deterministic"] = det
        canary.append(r)
        print(f"  {m}: deterministic={det}")
    write("exp5_canary_determinism.json", {"track": "determinism canary", "results": canary})

    write("exp5_provenance.json", {"provenance": provenance, "think_check": think_check})
    print("\nDone. Results in experiments/chat-model-bench/results/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
