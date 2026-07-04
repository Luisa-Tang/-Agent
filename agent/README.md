# Local Agent Framework

This directory implements a deterministic local Agent for the two circle-packing tasks.

Run a quick reproducibility pass:

```bash
python agent/run.py --task both --iterations 5 --fast
```

If the system `python` command is not available, use the Python environment that contains NumPy and SciPy, for example:

```bash
/opt/Anaconda3/bin/conda run -n sp python agent/run.py --task both --iterations 5 --fast
```

## Modules

- `run.py`: command-line entrypoint and orchestration loop.
- `evaluator_adapter.py`: writes candidates and invokes official evaluators through subprocesses.
- `candidate_generators.py`: deterministic candidate templates, SLSQP search, multi-start search, and perturb-and-repair.
- `geometry_utils.py`: validation, margins, fixed-center LP radius optimization, and safety repair.
- `archive.py`: candidate metadata, code snapshots, raw evaluator output, and best-candidate tracking.
- `log_utils.py`: JSONL and human-readable logging helpers.
- `report_data.py`: submission report generation.

The final exported `task_A/solution.py` and `task_B/solution.py` are standalone and do not require network access or an LLM API.

## Optional LLM Reflection

The Agent can connect to an OpenAI-compatible Chat Completions endpoint for strategy reflection. It only chooses among the built-in deterministic strategies; generated final solutions remain offline.

```bash
export DEEPSEEK_API_KEY="..."
python agent/run.py --task both --iterations 5 --fast --use-llm \
  --llm-base-url https://api.deepseek.com \
  --llm-model deepseek-v4-pro
```

If `DEEPSEEK_API_KEY`/`OPENAI_API_KEY` is absent or the endpoint is unavailable, the loop records the failure and falls back to the local deterministic policy.
