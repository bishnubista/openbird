# Design: briefing prose-adherence eval (routine summary path)

> **Status update:** `origin/main` shipped a prose-tightened routine prompt in #137
> ("write the briefing as one short paragraph… Output only the paragraph") AND a Swift
> renderer in #135. So the prose contract this eval describes is now LIVE on main — the
> eval's role shifts from *proving a needed fix* to **guarding the shipped contract**
> against model/prompt regressions. The motivation below is the historical failure that
> prompted it (measured against the pre-#137 "well-structured" persona).

## Motivation (from a real failure)
On `feat/clean-briefing-card` (HEAD 7a7fd4a) the Today briefing rendered as a heavily
structured markdown breakdown (numbered sections, nested bullets, "Key Features:")
instead of the intended clean prose — produced by **qwen3:8b**. Root cause: the prose
*prompt* contract from 6b96c2d was dropped in the 7a7fd4a recommit (which is
"presentation only"), so the persona still says *"Produce a concise, well-structured
summary"* — which invites structure. The Swift `BriefingProse` normalizer strips literal
`###`/`**` symbols but not the structural shape.

This eval measures **how well a model honors the briefing prose contract**, so we can:
1. quantify the regression (current "well-structured" prompt → structure), and
2. provide the regression test when the prose contract is re-added, and
3. compare models on prose adherence (is qwen3 uniquely bad here?).

## What it scores (deterministic, no LLM judge)
Runs the REAL routine path — `build_routine_messages(persona, prompt, start, end,
context)` + `provider.complete(messages)` (plain text, NO json_object) — using the actual
**`yesterday` template prompt** (`get_template("yesterday")`, the one the Today card
uses; note its current prompt also says "grouped sensibly" — itself structure-inviting),
over synthetic observation logs, then scores the raw model output.

Checks use **line-anchored regexes** (Codex-supplied) so ordinary prose with colons,
em-dashes, or inline numbers ("3 PM", "1.5 hours", "2026 was busy") is NOT flagged:
```python
heading_atx        = r"(?m)^\s{0,3}#{1,6}\s+\S"
heading_bold_line  = r"(?m)^\s{0,3}\*\*[^*\n]{1,80}:?\*\*\s*$"
heading_colon_line = r"(?m)^\s{0,3}[A-Z][A-Za-z0-9 /&()'-]{2,60}:\s*$"
numbered_heading   = r"(?m)^\s{0,3}\d{1,2}[.)]\s+[A-Z][^.!?\n]{1,80}:?\s*$"
bullet_line        = r"(?m)^\s{0,3}(?:[-*+•])\s+\S"
ordered_item       = r"(?m)^\s{0,3}\d{1,2}[.)]\s+\S"
horizontal_rule    = r"(?m)^\s{0,3}(?:-{3,}|\*{3,}|_{3,})\s*$"
reasoning = [r"(?is)<think>.*?</think>",
             r"(?im)^\s*(summary of observations|final answer)\s*:?\s*$",
             r"(?im)^\s*here(?:'s| is)\s+(?:a\s+)?(?:structured\s+)?breakdown\b",
             r"(?im)^\s*let(?:'s| us)\s+(?:see|think|analy[sz]e)\b"]
```
- **no_headings** = none of heading_atx/heading_bold_line/heading_colon_line/numbered_heading.
- **no_lists** = none of bullet_line/ordered_item.
- **no_rules** = no horizontal_rule.
- **no_reasoning_narration** = none of the reasoning patterns.
- **length_bound** = fixture-configured `max_words`, `max_sentences`, `max_paragraphs`
  (default 1; blank-line-separated paragraphs count as structure). Words via token regex;
  sentences via terminal punctuation.
- **grounded (faithfulness)** = chat-eval-style **structured facts**, NOT a weak
  must_mention count: `required_facts: [{any_of:[...]}]` (all must hold), `forbidden`
  (distractor/injection absent), optional `conflicting_facts` and `coverage_min` for soft
  details. **No source-binding** — briefings are uncited, so we do NOT claim chat-eval's
  citation-bound faithfulness.

A case PASSES only if prose-clean AND grounded. Per-check reason codes.

## Scoping decision (Codex-confirmed): score RAW model output as the gate
Codex verified this matches product behavior on this tree: `briefing --json` emits raw
`text`, `TodayModel` assigns `briefing.text` directly, and `TodayView` renders
`Text(briefing)` with **no `BriefingProse` layer wired in** (TodayView.swift:185) — so raw
output is what the user sees. Therefore:
- **Gate = raw model output** (the prompt-contract measure).
- **Advisory `display_clean`** is added ONLY if it calls / golden-tests the REAL Swift
  `BriefingProse` normalizer — we do NOT port a "minimal normalizer" into Python (it would
  drift from the Swift impl and give false assurance).

## Repeats + both prompt variants (Codex-required)
- **Repeats:** briefing generation is one-shot daily, and the local model is stochastic,
  so a flaky prompt is a real reliability bug. Run R repeats (default 5; 10 for model
  comparison) and report `pass_count/repeat_count`, `prose_clean_rate`, `grounded_rate`,
  per-reason rates. Temperature pinned via the bench `ParamProvider` (production
  `complete()` exposes no temperature/seed).
- **Both variants in one run:** `prompt_variant = current | candidate | both`.
  - *current* = the active `yesterday` persona (baseline; the "well-structured" +
    "grouped sensibly" prompt) — should score LOW on prose, proving the regression.
  - *candidate* = a prose persona (re-instating 6b96c2d's contract: "plain flowing prose,
    a single short paragraph of a few sentences; no headings, lists, rules, or section
    labels"). Same fixtures, model digest, repeats.
  - Report `system_prompt_hash` + `persona_hash` per variant so the delta is attributable.

## House-style alignment
- Module: `openbird/routines/eval.py` (mirrors `signals/eval.py`, `chat/eval.py`).
- Fixtures: `tests/fixtures/routines/briefing_prose.jsonl` — synthetic observation logs
  + structured `required_facts` (each `{any_of:[...]}`) + `forbidden` +
  optional `conflicting_facts` / `coverage_min` + length thresholds
  (`max_words`/`max_sentences`/`max_paragraphs`). NO real captured data.
- CLI: `openbird eval briefing <fixture> [--model ...]` (same `eval` Typer group).
- Unit tests: `tests/unit/test_briefing_eval.py` — deterministic scorers, canned outputs.
- Content-free reports (ids/booleans/reason-codes/counts only) on every exit path.

## Execution
The bench driver (`experiments/chat-model-bench/`) gains a briefing mode that runs this
across the same model set + temperature tracks, so we get a prose-adherence scorecard
per model and can see whether temperature affects format adherence.

## Resolved (Codex review round 1 → all 4 medium findings incorporated)
1. Raw-output scoring = gate (confirmed it matches product; TodayView has no normalizer).
   display_clean advisory only via the real Swift normalizer.
2. Line-anchored regexes adopted verbatim (no prose false-positives).
3. Length bound is fixture-configured (max_words/max_sentences/max_paragraphs).
4. Run BOTH prompt variants (current baseline + candidate prose) with prompt/persona
   hashes; target the real `yesterday` template prompt ("grouped sensibly" included).
5. Replaced must_mention with structured `required_facts`/`forbidden`/`conflicting_facts`;
   no source-binding; local judge advisory-only if added later.
Plus: repeats/pass-probability (R≥5; 10 for comparison) instead of single-shot.
