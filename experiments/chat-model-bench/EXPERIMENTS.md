# Chat-model answer-quality evals — lab notebook

Append-only record of every experiment and eval run for OpenBird's local RAG chat
model selection. Newest sections at the bottom. All fixtures are synthetic; reports
are content-free (ids/metrics only).

- Worktree / branch: `experiment/chat-model-evals` (off `origin/main` @ 37a845b)
- Eval core (Codex-approved design): `openbird/chat/eval.py` — see
  `docs/design/chat-model-evals.md`
- Fixtures: `tests/fixtures/chat/answer_quality.jsonl` (6 cases; modes: search,
  explicit-window, temporal-phrase)
- Driver: `experiments/chat-model-bench/run.py`
- Unit tests (deterministic, no model): `tests/unit/test_chat_eval.py` — 11 passing

## Scored axes (per case)
json_valid · facts_ok (bound to required_source_ids) · citations_precise ·
citations_complete · forbidden_clean (distractor/injection leak) · refusal_ok
(grounding matched expectation) · window_ok (citation ts inside requested span).
Composite weights: refusal 0.30, citations_complete 0.25, facts 0.20,
forbidden_clean 0.15, json_valid 0.10.

## Environment
- ollama 0.14.3 · litellm 1.88.1 · CPython 3.13.6 · darwin (Apple Silicon)
- Production RAG path has **no temperature/seed** → ollama defaults (temp ≈ 0.8).
  This is the source of run-to-run nondeterminism observed below.

---

## Run 1 — initial 2-model smoke (default sampling)
- Models: qwen3:8b, llama3.2 · fixture: 5 cases (pre temporal-phrase)
- Result: qwen3:8b composite **0.97** (pass 0.8); llama3.2 **0.94** (pass 0.8)
- qwen3:8b sole fail: forbidden_leak (decision-grounded). llama3.2 sole fail:
  hallucinated on unanswerable-refuse.
- Note: fingerprint was half-broken (litellm "unknown", digest "n/a").

## Run 2 — fingerprint fixed + temporal-phrase case added (default sampling)
- Models: qwen3:8b, llama3.2 · fixture: 6 cases
- Result: qwen3:8b **0.975** (5/6); llama3.2 **0.916** (4/6 — now also fails
  temporal? no: pass-rate denominator moved to /6).
- Fingerprint now resolves: litellm 1.88.1; digests qwen3:8b=500a1f067a9f,
  llama3.2=a80c4f17acd5.

## Run 3 — 3-model, qwen3:4b added (default sampling)  ← KEY FINDING
- Models: qwen3:8b, qwen3:4b, llama3.2 · 6 cases
- Result: qwen3:8b **0.975**, llama3.2 **0.95**, **qwen3:4b 0.375 (1/6)**.
- qwen3:4b failed 5/6 grounded cases identically (refusal + fact_missing +
  citation_incomplete) — i.e. produced NO valid citation.

### Diagnostic probes (qwen3:4b, deterministic)
- Plain completion (no schema): returns correct text + loose `citations: ["S1"]`. OK.
- `response_format=json_object` (the path RAG uses): returns **EMPTY STRING**. ← bug.
- System-prompt suffix `/no_think`: still empty. NOT a fix.
- Ollama option **`think=False`** + json_object: returns clean
  `{"answer": "...twelve percent", "citations": ["S1"]}`. ← **verified fix.**
- Conclusion: qwen3 thinking-mode × ollama json_object → empty content. Impacts the
  16GB-Mac default. Recorded in Claude memory (qwen3-json-empty-thinkfalse).

## Run 4 — 6-model bench (default sampling)
- Models: qwen3:8b, qwen3:4b, llama3.2, gemma3:4b, qwen2.5:7b-instruct, granite3.3:8b
- Ranking (composite): granite3.3:8b **0.975** · llama3.2 **0.95** · qwen3:8b **0.925**
  · gemma3:4b **0.925** · qwen2.5:7b-instruct **0.85** · qwen3:4b **0.375**.
- **Nondeterminism observed:** qwen3:8b's `unanswerable-refuse` flipped PASS→FAIL vs
  Run 3 (same model/prompt). Confirms default sampling (~temp 0.8) makes the refusal
  axis flaky. granite3.3:8b was the only model with refusal_ok=1.0 this run.
- Caveat: with n=6 cases and unseeded sampling, the 0.925–0.975 cluster is within
  noise; only qwen3:4b (deterministic empty-JSON) and "small models hallucinate on
  refusal" are robust conclusions.

---

## Experiment 5: temperature / determinism / think-fix (CODEX-REVIEWED → revised)
Goal: quantify the stochasticity seen in Runs 3–4 and test the qwen3 `think=False`
fix, as a **diagnostic pilot** — NOT a 16GB winner claim (n=6 is too small for that;
see "winner bar" below).

Codex review (review-exp5.log) verdict: **revise**. Key corrections incorporated:
1. `seed + R` measures plumbing, not flakiness → split into two tracks (below).
2. Subclassing `LLMProvider._model_kwargs` is a valid injection point (production
   `complete()` re-calls it on the JSON path), but controlled results MUST be labeled
   "controlled params," reported beside the no-param production baseline.
3. Don't assume temp=0 determinism — measure it (warm repeats; cold-reload noted).
4. `think=False` is its own controlled variable, not bundled. Verify the transformed
   request actually carries `think:false`.
5. Keep `json_valid` separate; record ollama params (num_predict/top_p/top_k/num_ctx/
   repeat_penalty), digest, keep_alive policy.

Design as run (pilot scope, bounded runtime):
- Injection: bench-local `ParamProvider(LLMProvider)` overriding `_model_kwargs`.
- **Track A — production baseline:** no injected params, unseeded, R=8 repeats/model.
  Honest proxy for today's app. Report per-case pass-probability + flakiness flag.
- **Track B — temperature sweep:** inject `temperature ∈ {0.0, 0.4, 0.8}`, unseeded,
  R=8. Sensitivity curve. (Labeled "controlled params.")
- **Track C — think=False (controlled):** {qwen3:4b, qwen3:8b} × {default, think=False}
  at temp=0, R=5; capture and assert the litellm-transformed request includes
  `think:false`.
- **Determinism canary:** temp=0 + seed=7, warm, R=5 — expect convergence (verifies
  seed plumbing only). Cold-reload determinism left as a noted follow-up.
- Models (A/B): granite3.3:8b, llama3.2, gemma3:4b, qwen3:8b. keep_alive warmed once.
- Output: `results/exp5_*.json` (per track) + summary table appended here.

**Winner bar (NOT met by this pilot, per Codex):** a defensible 16GB recommendation
needs n≥20 fixtures stratified (≥5 refusal, ≥5 injection/distractor), R≥10 for
finalists at both production-default and chosen controlled setting, with paired
per-fixture outcomes + bootstrap CIs; don't call a winner inside ~0.05 composite.
Tracked as Experiment 6 (dataset expansion).

### Experiment 5 — RESULTS (rolling; written per track to results/)
**Track A — production baseline (no params, unseeded, R=8):**
| model | mean_pass | refusal_ok | flaky cases | notes |
|---|---|---|---|---|
| granite3.3:8b | **0.896** | **1.0** | 1 | best; only decision-grounded flaky (0.375) |
| llama3.2 | 0.854 | 0.875 | 2 | unanswerable flaky **0.25**, window flaky |
| gemma3:4b | 0.667 | 0.833 | 0 | decision-grounded 0.0, unanswerable 0.0 (consistently) |
| qwen3:8b | 0.667 | 0.833 | 2 | decision-grounded 0.0, unanswerable flaky **0.125** |

Findings (now repeated-measures, not single-shot):
- **Refusal is the flaky axis**: unanswerable-refuse passes only 25% (llama3.2) / 12.5%
  (qwen3:8b) of repeats under production sampling → quantified reliability defect.
- **granite3.3:8b most reliable** at production default (refusal_ok=1.0 across 8 repeats).
- **FIXTURE BUG (decision-grounded):** `forbidden` includes "screenshots", but the gold
  decision is "store text… never raw screenshots" — a faithful answer must say
  "screenshots". So the case penalizes correctness. FIX in Experiment 6: forbidden must
  be the DISTRACTOR's unique content, never a token the correct answer needs.

**Track B — temperature sweep (controlled, unseeded, R=6):** pass rate is essentially
FLAT across T∈{0.0,0.4,0.8}: granite3.3:8b 0.833/0.861/0.833, llama3.2 0.833 flat,
gemma3:4b 0.667 flat, qwen3:8b 0.667 flat. → Temperature in [0,0.8] does NOT move the
aggregate; failures are capability/prompt-driven. The stochastic axis is *refusal*
(Track A), not temperature.

**Track C — think=False (R=5, temp=0):** request_check confirms `think:false` reaches the
Ollama request for both qwen tags.
| model | variant | pass | json_valid | facts_ok |
|---|---|---|---|---|
| qwen3:4b | default | 0.167 | 1.0* | 0.167 |
| qwen3:4b | think_false | 0.167 | 1.0* | **0.333** |
| qwen3:8b | default | 0.667 | 1.0 | 1.0 |
| qwen3:8b | think_false | **0.833** | 1.0 | 1.0 |
- **think=False helps the qwen family on the REAL RAG prompt**: qwen3:8b 0.667→0.833;
  qwen3:4b facts 0.167→0.333 (PARTIAL — the long RAG system prompt + json_object is harder
  than the isolated probe, so 4b is NOT fully rescued. More honest than the earlier
  "think=False fixes 4b" claim.)
- ***EVAL DEFECT FOUND (json_valid axis):** `run_chat_eval` marked json_valid=True whenever
  `complete()` did not raise — but qwen3:4b returns an EMPTY string (no exception), so the
  real empty-output failure was mis-attributed to fact_missing/refusal and json_valid read
  1.0. FIXED via a recording provider wrapper that flags empty/non-dict structured output
  as json_invalid (see openbird/chat/eval.py).

**Determinism canary (temp=0 + seed=7, R=4):** ALL four models deterministic=True (verdicts
byte-identical across repeats) → seed plumbing works; reproducible runs are achievable. We
deliberately use unseeded elsewhere to MEASURE stochasticity.

### Experiment 5 — conclusions
- **qwen3:4b unfit for the RAG path** even with think=False (partial recovery only).
- **granite3.3:8b** most reliable overall (Track A 0.896, refusal 1.0); **think=False is a
  beneficial default for the qwen family** on the JSON path.
- **temperature is not the lever** — refusal flakiness + model capability are.
- A defensible 16GB winner still needs Experiment 6 (n≥20 stratified, R≥10).

---

## Experiment 6 (added): briefing prose-adherence eval — Codex-approved, built
Motivated by a real qwen3:8b briefing rendering as structured markdown (user screenshot,
feat/clean-briefing-card). Root cause: the prose-prompt fix (6b96c2d) was DROPPED in HEAD
7a7fd4a (presentation-only); the active `yesterday` persona still says "well-structured"
and the user prompt says "grouped sensibly".

- Design: docs/design/briefing-prose-eval.md (Codex: revise×1 → approve; raw-output gate,
  line-anchored prose regexes, repeats/pass-prob, BOTH prompt variants current|candidate,
  structured required_facts not must_mention).
- Built: `openbird/routines/eval.py` + `tests/fixtures/routines/briefing_prose.jsonl` (3
  synthetic) + `tests/unit/test_briefing_eval.py` (13 tests). All 24 eval unit tests pass.
- Measures: no_headings/no_lists/no_rules/no_reasoning/length + grounded(required_facts/
  forbidden), per-reason rates, advisory md_symbol_count, per-variant prompt/persona hash.
- Live cross-model run: `experiments/chat-model-bench/briefing_bench.py` (4 models × 2
  variants × 3 cases × R=5). Results → results/briefing_bench.json.

> NOTE: these results were measured against the PRE-#137 routine prompt (the old
> "well-structured summary" persona). `origin/main` has since shipped a prose-tightened
> prompt (#137) + Swift renderer (#135), so the "current" baseline below is now
> historical; the eval framework now guards the shipped prose contract. Re-run against
> current main to get the post-fix baseline.

### Experiment 6 — briefing prose RESULTS (current vs candidate persona, pre-#137)
| model | prose_clean current → candidate | Δ | grounded (cand) | md_symbols current→cand |
|---|---|---|---|---|
| **qwen3:8b** | **0.0 → 1.0** | **+1.000** | 0.80 | 12.9 → 3.6 |
| llama3.2 | 0.333 → 0.867 | +0.534 | 0.53 | 2.7 → 0.7 |
| granite3.3:8b | 0.6 → 0.8 | +0.200 | 0.73 | 0.07 → 1.2 |
| gemma3:4b | 1.0 → 1.0 | +0.000 | 0.67 | 0 → 0 |

**Findings:**
- **Reproduces + explains the user screenshot:** qwen3:8b (the Today default) violates the
  prose contract 93% of the time under the CURRENT "well-structured" persona (headings 0.93,
  lists 0.93, too_long 0.93). The candidate prose persona (re-instating 6b96c2d's dropped
  contract) takes it to 100% prose-clean. → **Fix: re-add the prose persona.** The eval is
  its regression test.
- **Model-dependent:** gemma3:4b is immune (always prose); qwen3 is the worst offender and
  most prompt-sensitive; granite already fairly clean.
- **Prose↔coverage tradeoff:** the terse prose persona raises fact_missing slightly
  (llama3.2 candidate 0.47). granite balances prose (0.8) + grounding (0.73) best.
- Caveat: n=3 fixtures, R=5 — small, but the qwen3:8b 0.0→1.0 signal is far outside noise.
