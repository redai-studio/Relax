# DeepEyes Agentic Example

DeepEyes integration for Relax agentic rollout.

## Entry points

- `run_deepeyes_agentic.sh`
- `run_deepeyes_agentic_pr.sh`
- `run_deepeyes_agentic_qwen35_9B_async.sh`

## Launch experiments

```bash
MODEL_DIR=/path/to/models \
DATA_DIR=/path/to/data \
SAVE_DIR=/path/to/save \
bash examples/deepeyes_agentic/run_deepeyes_agentic.sh
# bash examples/deepeyes_agentic/run_deepeyes_agentic_pr.sh
# bash examples/deepeyes_agentic/run_deepeyes_agentic_qwen35_9B_async.sh
```

Each training script enables agentic rollout and points Relax to the agent process:

```bash
--use-agentic-rollout
--agent-command "bash ${SCRIPT_DIR}/run_agent_app.sh"
--agent-cwd "${SCRIPT_DIR}"
--custom-rm-path examples.deepeyes_agentic.reward_deepeyes.reward_func
```

## Agent process

- `run_agent_app.sh` adapts Relax runtime variables to the agent CLI.
- `app/agent.py` runs one DeepEyes session.
- `app/env_deepeyes.py` implements image tools.
- `app/deepeyes_config.yaml` stores task settings.

Relax provides:

```bash
RELAX_INPUT_JSON
RELAX_OUTPUT_JSON
RELAX_BASE_URL
RELAX_SESSION_ID
```

`run_agent_app.sh` maps them to:

```bash
export OPENAI_BASE_URL="${RELAX_BASE_URL}"
export OPENAI_API_KEY="${RELAX_SESSION_ID}"

python -m app.agent \
    --input-json "${RELAX_INPUT_JSON}" \
    --output-json "${RELAX_OUTPUT_JSON}"
```

## Data flow

1. Relax writes session input JSON with `messages`.
2. The agent process calls the Relax OpenAI-compatible session API.
3. The agent process appends assistant messages and tool observation messages.
4. The agent process writes session output JSON.
5. Relax finalizes the session into training samples.
6. Relax calls `reward_deepeyes.reward_func` for reward scoring.
