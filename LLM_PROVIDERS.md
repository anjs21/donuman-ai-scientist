# LLM backends for the campaign planner and critic

The LLM proposes the next campaign action (search, propose, analogs, refine,
repair, finish) and a separate critic reviews proposals. It does **not** choose
calculations (`warden decide` is a deterministic value-of-information policy),
score hypotheses, or verify quotes; those stay in code.

Backends: `ollama` (local, default) and `openai` (Chat Completions with
JSON-schema output; any OpenAI-compatible endpoint via `base_url`). Replies are
validated locally against the same schemas for both.

## Configure

The key is read only from an environment variable (default `OPENAI_API_KEY`);
never put it in a config file.

```bash
export OPENAI_API_KEY=sk-...        # current shell only
```

`llm` section forms:

```json
{"provider": "openai", "model": "gpt-4.1"}                       // both roles
{"planner": {"provider": "ollama", "model": "qwen2.5:7b"},
 "critic":  {"provider": "openai", "model": "gpt-4.1", "temperature": 0.0}}   // split roles
{"host": "http://localhost:11434", "model": "qwen2.5:7b"}         // legacy: Ollama for both
```

OpenAI options: `model`, `api_key_env`, `base_url`, `temperature` (`null` for
models that reject it), `max_completion_tokens`, `timeout_s`, `max_retries`
(429/5xx, honours `Retry-After`), `max_prompt_chars`. Keys shared at the top
level of a split config apply to both roles.

Ready-made: `config/v3-openai.json` (OpenAI for both roles) and
`config/v3-hybrid.json` (local planner, OpenAI critic). Set `model` to an
OpenAI model your account can use that supports Structured Outputs.

## Check, then run

The campaign's MACE screen needs a checkpoint at `models/mace.model`:

```bash
python -m warden fetch-model --list
python -m warden fetch-model --name matpes-r2scan-0 --accept-license --verify
```

```bash
env -u PYTHONPATH python -m pytest -q tests/test_llm.py          # mock server, no key needed
python -m warden llm-check --config config/v3-openai.json        # one tiny request per role; no project data
WARDEN_LIVE_LLM=1 env -u PYTHONPATH python -m pytest -q tests/test_llm.py -k live   # optional real call

python -m warden campaign --db runs/campaign_openai/campaign.sqlite \
    --manifest <manifest.json> --config config/v3-openai.json \
    --controls examples/controls-v3.json --evidence examples/evidence-literature.json
```

The campaign database records `llm_backends` (provider/model per role) at the
start and `llm_usage` (calls and tokens) at the end.

## Data leaving the machine

With `openai`, each planning step sends the campaign context: retrieved
literature text (truncated per item), candidate SMILES and hypotheses, computed
results and the agent ranking. Check this is acceptable before use.

## Comparing backends fairly

Use a new database per backend (the campaign refuses a changed config on an
existing one) and identical budgets, manifest, evidence and controls. Compare
from each `report.json`: valid proposals vs critic/validation rejections,
hypotheses `supported`/`refuted`, best dose-window margin from `decide`, and
calculations spent. One campaign per backend is an anecdote; repeat several
times before attributing a difference to the model.
