# Design: Local-model answer-quality evals for OpenBird chat (RAG)

## Goal
Let a user (and maintainers) measure whether a given **local Ollama model** produces
**reliable answers grounded in captured text**, so model selection is evidence-based,
not vibes. Output: a per-model scorecard + a recommendation, runnable offline.

## Why this is different from the existing signal eval
`openbird/signals/eval.py` is **deterministic and model-free** (`provider=None`) → CI-safe.
Answer quality is **inherently model-bound** — you cannot score "does the local model
answer reliably" without running the model. So the architecture must split:

- **Deterministic core (CI-safe, unit-tested):** dataset loader, the scoring functions,
  and report aggregation. Tested by feeding *canned* model outputs — no Ollama in CI.
- **Model run (opt-in, `integration`-marked, auto-skips if Ollama down):** the loop that
  actually calls each candidate model through the **production** `RAG`/`build_rag_messages`
  path and scores the result.

## What "reliable answer" decomposes into (the scored axes)
Derived from what `chat/rag.py` already enforces:
1. **JSON validity** — `provider.complete(json_schema=_RESPONSE_SCHEMA)` must parse.
2. **Citation correctness** — cited ids ⊆ allowed ids AND include the gold id(s)
   (`_validate_citations` drops hallucinated; we measure both precision and recall of ids).
3. **Faithfulness / answer correctness** — gold "must-include" facts present; "must-not"
   absent. Deterministic, but **structured** (see fixture schema below), not loose substring.
4. **Correct refusal** — for `unanswerable` cases the answer must trip the grounding gate
   (no valid citation ⇒ `_UNGROUNDED_MESSAGE`) or explicitly decline. Hallucinating here
   is the worst failure for a privacy/trust product.
5. **Injection resistance** — reuse the existing `prompts/harness.py` sentinel approach
   per model (banned-term must not appear; must not cite the injected id).
6. **Latency / RSS** — reuse mlx-runtime's measurement approach (ns timings from Ollama
   `/api/chat`, RSS via ps), so model choice respects the RAM tiers.
7. **Temporal / window grounding** *(added — Codex review #1)* — `RAG.answer` has a
   distinct path (`rag.py:289` explicit window, `rag.py:359` `_answer_temporal`) that
   routes through `time_range_text` instead of semantic search. Fixtures must include
   `search`, `temporal-phrase` ("what did I do yesterday?"), and `explicit-window` modes,
   and score that **every cited occurrence's `ts` falls inside the requested window** plus
   the expected source count/order. Omitting this would leave the most failure-prone path
   (chronological recall) unmeasured.

## Scoring: deterministic first, model-judge advisory only (local-first constraint)
A cloud judge breaks the privacy model; a local judge is circular. So gold-based
deterministic checks are the **gate**; an optional local-model judge is **advisory** and
never changes pass/fail — mirroring how `prompts test` treats its model probe.

### Structured gold fixture schema (Codex review #3 — replaces loose substring)
Loose "must-include substring" is brittle and accidentally satisfiable. Fixtures bind
each fact to the source that must back it:
```jsonc
{
  "id": "revenue-q3",
  "mode": "search",                       // search | temporal-phrase | explicit-window
  "question": "how did revenue do?",
  "sources": [                            // becomes the seeded store / fake hits (S1..Sn)
    {"sid": "S1", "app": "Numbers", "window": "Q3 Report", "ts": 500.0,
     "text": "The quarterly revenue grew by twelve percent."},
    {"sid": "S2", "app": "Notes", "ts": 600.0, "text": "Lunch options: sushi, tacos."}  // distractor
  ],
  "facts": [                              // ALL must hold
    {"any_of": ["12%", "twelve percent"], "required_source_ids": ["S1"]}
  ],
  "forbidden": ["sushi", "tacos"],        // distractor leakage / hallucination
  "expect_grounded": true,                // false => refusal case (axis 4)
  "window": null                          // [start_ts, end_ts] for explicit-window mode
}
```
- `any_of` + date/number **normalization** (collapse `%`/`percent`, ISO dates) before match.
- `required_source_ids` ties faithfulness to citation correctness (a model that states the
  fact but cites the wrong/zero source FAILS — that is the real reliability bar).
- Include **distractor** and **conflicting-fact** rows so a model can't pass by parroting.

### Two execution lanes (Codex review #2 — fake-store alone hides interaction)
- **Lane A — generation isolation:** a fake `_Searcher` returns pre-baked `SearchHit`s, so
  retrieval is fixed and the score reflects *only* the model's generation/citation behavior.
- **Lane B — end-to-end:** a seeded `MemoryStore(db_path=":memory:", provider=fake_provider)`
  (fake embeddings, no Ollama for embeds) — exercises chunking, dedupe, context cap, ranking,
  and the temporal scan, catching retrieval×generation failures that decide real reliability.
  Mirrors the existing pattern at `tests/unit/test_chat.py:537`.
  Both lanes share the same fixtures and scorers; the report labels which lane produced each
  number so a regression is attributable to retrieval vs generation.

## House-style alignment (mirror signals eval)
- Module: `openbird/chat/eval.py` (next to `chat/rag.py`).
- Fixtures: `tests/fixtures/chat/*.jsonl` — synthetic captured-text Q&A, **no real data**.
- CLI: `openbird chat eval [--model ollama/...]... [--json]` (subcommand pattern from the
  signals branch).
- Unit tests: `tests/unit/test_chat_eval.py` — deterministic scorers with canned outputs.
- **Content-free reporting (hardened — Codex review)**: reports carry metrics/ids/
  reason-codes, never captured text or model output bodies. This rule covers **every** exit
  path, not just the happy path: validation errors, failure diagnostics, advisory local-
  judge notes, and the `--json` payload. Fixtures are synthetic so the dataset itself is
  safe to commit; a unit test asserts no fixture/answer text leaks into any report field
  (precedent: signal eval's content-free CLI tests).

## Run provenance / fingerprinting (Codex review #4 — scores drift without it)
Ollama tags are mutable, so a bare "qwen3:8b: 0.9" is meaningless next month. Every run
records and the report embeds: `ollama --version`, the exact model **digest** + params via
`ollama show` (not just the tag), LiteLLM version, the embed model, generation params
(temperature/seed/num_predict), and a **hash of the rendered system prompt + JSON schema**.
A recommendation is only comparable across runs with matching prompt/schema hashes. Reuse
the exact-tag availability guard from `tests/unit/test_chat.py:578` so a host missing a tag
skips cleanly instead of erroring.

## Key design decision: build the gold dataset on a FAKE in-memory store
The eval must exercise the real retrieval+citation path, but must not touch the user's
store. Plan: build a tiny in-memory `_Searcher` stub that returns pre-baked `SearchHit`s
with `Observation`s (source ids S1..Sn), so retrieval is fixed and the eval isolates the
**model's** behavior (the variable under test). This avoids conflating retrieval quality
with generation quality, and keeps the eval deterministic except for the model.

## Candidate models to evaluate (Apple Silicon, RAM-tiered)
Defaults today: `qwen3:4b` (16GB), `qwen3:8b` (24/32GB), embed `embeddinggemma`.
Proposed generation candidates (pull via ollama as needed):
- `qwen3:4b` and `qwen3:8b` (current defaults — the baseline to beat)
- `llama3.2:3b` (already installed; smallest/fastest floor)
- `qwen2.5:7b-instruct` (strong JSON/structured-output reputation)
- `granite3.3:8b` *or* `gemma3:4b` (one extra contender, decided after baseline)
Embedding models are evaluated separately (retrieval recall) and are out of scope for v1.

## Repo-inclusion recommendation
**Yes, commit it** — consistent with the signals-eval precedent and valuable for an OSS
project (contributors can prove a model regression, users can validate their hardware).
Guardrails: deterministic core in CI; model runs `integration`-marked + opt-in; synthetic
fixtures only; content-free output. Retire/keep `experiments/mlx-runtime` separately.

## Resolved decisions (post Codex round 1)
- **Module location:** `openbird/chat/eval.py` — mirrors `openbird/signals/eval.py`; a
  top-level `openbird/eval/` is premature with one feature.
- **Candidate-model loop:** single-model scoring lives in the committed core + `openbird
  chat eval --model X`. The *multi-model comparison + recommendation* lives in a thin
  `experiments/chat-model-bench/` driver that imports the committed core (keeps N-model
  orchestration out of the shipped CLI; retires the parallel mlx-runtime re-implementation).
- **Isolation:** both lanes (A fake-store + B seeded store), per review #2 — not either/or.
