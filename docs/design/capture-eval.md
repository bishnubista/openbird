# Deterministic capture evaluation

OpenBird uses a synthetic, local evaluation harness to measure capture extractors before they are allowed to change production capture. It is intentionally separate from the live encrypted store: the command reads only an explicit JSONL fixture, makes no model calls, and emits aggregate metrics without captured, reference, or forbidden text.

This is a source-level contract test, not a live end-to-end accessibility benchmark. The checked-in `current_helper` rows model known extractor shapes for generic AX, ChatGPT, Codex, and Chrome. A later controlled-app runner must validate those assumptions against signed builds and macOS TCC behavior.

## Run it

Baseline diagnostic:

```console
openbird eval capture tests/fixtures/capture/synthetic.jsonl --json
```

A fixture may also contain rows from a proposed extractor. Gate it against the baseline with:

```console
openbird eval capture results.jsonl \
  --baseline current_helper \
  --candidate proposed_extractor \
  --json
```

Invalid fixtures exit `2`. A baseline-only diagnostic exits `0` even when the current helper misses future promotion thresholds. A candidate exits `1` when a promotion gate fails.

## Fixture contract

Each JSONL row contains:

- `case_id`, `target`, `state`, and `strategy`
- `captured_text`, `reference_text`, and a list of `forbidden_text` sentinels
- `extraction_ms`
- optional paired `changed_at_ms` / `captured_at_ms`
- optional `repeat_group` and `budget_exceeded`

Candidate and baseline rows must have identical case sets and matching target, state, repeat group, reference, forbidden sentinels, and change-start time. Only observed output, capture completion time, extraction duration, and budget outcome may differ. This prevents a candidate from weakening the ground truth or silently dropping privacy and stability cases.

Fixture validation and reports never echo the three text fields. Checked-in fixtures must be artificial; never copy real captures into the repository.

## Metric definitions

Text is normalized with Unicode NFKC and `casefold`, then split with `\w+|[^\w\s]`. The latter preserves code punctuation as tokens. Extraction cases have non-empty reference tokens; their token-multiset precision, recall, and F1 are scored per case and macro-averaged. Empty-reference cases are excluded from those metrics and instead measure exclusion accuracy: only a token-empty capture passes.

Repeat stability is the minimum multiset-Jaccard score across all pairs in each repeat group. Change latency uses `captured_at_ms - changed_at_ms`. Extraction p95 is nearest-rank, which intentionally approaches the maximum for small corpora. An extraction is a budget breach when the fixture marks it or duration exceeds two seconds.

Candidate promotion requires:

- precision at least `0.75` and recall at least `0.90`;
- repeat similarity at least `0.95` when repeat groups exist;
- all measured changes captured within three seconds;
- p95 extraction at most one second and no two-second budget breach;
- exclusion accuracy `1.0` and zero forbidden-sentinel leaks;
- no per-target precision or recall regression greater than `0.05`; and
- at least `0.20` mean per-case F1 gain on targets whose baseline F1 is below `0.70`.

## Checked-in baseline calibration

The 26 synthetic `current_helper` cases establish this deterministic baseline:

| Metric | Baseline | Promotion bar | Gap / result |
|---|---:|---:|---|
| Precision | 0.582198 | >= 0.75 | -0.167802 |
| Recall | 0.664931 | >= 0.90 | -0.235069 |
| Mean per-case F1 | 0.613476 | diagnostic | — |
| Minimum repeat similarity | 1.0 | >= 0.95 | pass |
| Changes within 3 seconds | 0.75 (3/4) | 1.0 | one slow case |
| p95 extraction | 1150 ms | <= 1000 ms | +150 ms |
| Two-second budget breaches | 0 | 0 | pass |
| Exclusion accuracy | 0.5 (1/2) | 1.0 | one modeled form leak |
| Forbidden leak cases | 1 | 0 | fail |

Per-target extraction quality:

| Target | Precision | Recall | Mean per-case F1 |
|---|---:|---:|---:|
| ChatGPT | 0.111111 | 0.166667 | 0.133333 |
| Chrome | 0.486111 | 0.493056 | 0.479293 |
| Codex | 0.799560 | 1.000000 | 0.885024 |
| Generic AX | 0.913087 | 1.000000 | 0.946078 |

The result is deliberately diagnostic: it makes the expected weaknesses visible instead of grandfathering them. A future candidate must clear the absolute product bars and improve low-quality ChatGPT and Chrome results by at least 0.20 F1 without regressing stronger targets.
