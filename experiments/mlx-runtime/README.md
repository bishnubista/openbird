# MLX Runtime Experiment

Small isolated benchmark for comparing MLX/MLX-LM against OpenBird's current
local Ollama path. Production provider selection now has a reserved `mlx`
insertion point in `openbird/llm/provider.py`, but this experiment remains the
acceptance gate before any MLX runtime backend is wired in.

## Model

MLX backend:

- `mlx-community/Qwen3-4B-Instruct-2507-4bit`

This is a 4B-class MLX community instruction model. Hugging Face's model page
shows direct `mlx_lm` usage with `load()` and `generate()`.

Ollama backend:

- `llama3.2`

This matches OpenBird's current default `OPENBIRD_LLM_MODEL=ollama/llama3.2`
path, but the experiment calls Ollama's local HTTP API directly so production
provider behavior is not exercised by the benchmark.

## Run

Ollama-only smoke run:

```bash
uv run python experiments/mlx-runtime/run_experiment.py --backend ollama
```

MLX plus Ollama run in an ephemeral environment:

```bash
HF_HUB_DISABLE_XET=1 uv run --with mlx-lm python experiments/mlx-runtime/run_experiment.py --backend both --mlx-load-timeout 900
```

The first MLX run downloads the model from Hugging Face and records that setup
friction in the output. Generated artifacts are ignored by git:

- `experiments/mlx-runtime/results.json`
- `experiments/mlx-runtime/results.md`

## Workloads

The runner tests five local-first OpenBird-shaped workloads:

1. Grounded Q&A over supplied context
2. JSON response with citation IDs
3. Prompt-injection attempt inside retrieved context
4. Routine-style summary over fake observations
5. Short draft in the user's voice from examples

It records setup friction, cold load time, total latency, memory deltas, JSON
parse success, citation validity, and simple qualitative usefulness checks.
