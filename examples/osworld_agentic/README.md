# OSWorld GUI+MCP hybrid agentic RL

Relax agentic recipe for the two headline runs from
[hybrid-routing-agent](https://github.com/redai-infra/hybrid-routing-agent):

- `outcome_only_rl` - pure a1 outcome-only baseline (74 tasks).
- `context_rl` - same rollout with `skip_on_mcp_success` + 2-image history.

Model: `Qwen3-VL-8B-Thinking`. Env: OSWorld on **nex sandboxes** (no Docker, no
KVM, no qcow2). Eval: the 309-task OSWorld set.

## Layout

```
app/
  agent.py                     per-session driver subprocess
  env_osworld.py               async wrapper over DesktopEnv (provider_name=nex)
  hybrid_agent.py              RelaxHybridAgent (call_llm -> AgenticChatAPIService)
  osworld_config.yaml          context_policy=baseline            (outcome_only_rl)
  osworld_config_context.yaml  context_policy=skip_on_mcp_success (context_rl)
reward_osworld.py              outcome + length + cap + optional MCP bonus
run_agent_app.sh               agent-side entrypoint (invoked by Relax)
run_qwen3vl_8B_osworld_*.sh    training launchers for the two variants
scripts/prepare_data.py        evaluation_examples/*.json -> parquet
scripts/smoke_test.sh          single-session end-to-end, no Ray
env.sh.example                 template - copy to env.sh (git-ignored)
PITFALLS.md                    integration footguns; read before touching
VENDOR.md                      snapshot provenance
vendor/                        read-only upstream: agents/ + prompts/ (agent-side only)
```

`desktop_env` is **not** vendored. It is imported from the nex-enabled OSWorld
checkout at `$OSWORLD_REPO` (providers/nex, MCP-aware DesktopEnv, guest fixes),
so a single copy of the env serves both offline eval and RL rollouts.

## Prerequisites

- An OSWorld checkout with the nex provider - `$OSWORLD_REPO`.
- nex credentials: `NEX_API_KEY`, `NEX_WORKSPACE_ID`, `NEX_TEMPLATE`.
- `OSWORLD_CACHE_DIR` pointing at that checkout's `cache/`, so the evaluator
  reuses prefetched ground-truth files instead of reaching huggingface.
- An OpenAI-compatible endpoint (sglang / vLLM) for smoke tests. During training
  Relax supplies the per-session endpoint itself.

## Quickstart

```bash
# 0. One-time env setup
cp examples/osworld_agentic/env.sh.example examples/osworld_agentic/env.sh
$EDITOR examples/osworld_agentic/env.sh    # OSWORLD_REPO / MODEL_DIR / DATA_DIR / ...

# 1. Convert OSWorld task JSON -> parquet
python3 examples/osworld_agentic/scripts/prepare_data.py \
    --input-dir ${OSWORLD_REPO}/evaluation_examples \
    --output-dir ${DATA_DIR}/osworld/data

# 2. Sanity check - one session, one sandbox, no Ray
OSWORLD_TASK_JSON=${OSWORLD_REPO}/evaluation_examples/examples/<domain>/<id>.json \
OPENAI_BASE_URL=http://127.0.0.1:8000/v1 \
OPENAI_API_KEY=dummy OPENAI_MODEL=qwen3-vl \
bash examples/osworld_agentic/scripts/smoke_test.sh

# 3. Training
bash examples/osworld_agentic/run_qwen3vl_8B_osworld_outcome.sh
# or
bash examples/osworld_agentic/run_qwen3vl_8B_osworld_context.sh
```

## Two variants - what actually differs

Both launchers share the exact same GRPO / SGLang / Megatron / colocate config.
The only behavioural knob is which yaml the session driver loads, selected via
`OSWORLD_CONFIG_PATH`:

| File                              | context_policy        | Image history | Recipe            |
| --------------------------------- | --------------------- | ------------- | ----------------- |
| `app/osworld_config.yaml`         | `baseline`            | 2 images      | `outcome_only_rl` |
| `app/osworld_config_context.yaml` | `skip_on_mcp_success` | 2 images      | `context_rl`      |

`skip_on_mcp_success` drops the screenshot from turns immediately following a
successful MCP call - it frees up context tokens and biases the agent toward MCP
over pixel-diffing. It only has an effect when MCP is actually available.

## MCP is optional

The guest-side MCP server is injected by DesktopEnv only when `MCP_SRC_ROOT` is
mounted on the host. When it is not, `_get_obs()` returns `tool_list: []`, the
agent suppresses its local BM25 tool fallback and runs GUI-only, which is what
`outcome_only_rl` needs anyway. Set `OSWORLD_DISABLE_MCP=1` to skip the ~74 s
per-task readiness probe in that case.

## Reward

`reward_osworld.py` shapes the raw VM-evaluator outcome:

```
score = outcome                              # a1 from OSWorld's postconfig+metrics
      - 0.05 * turns                         # per-turn length penalty
      + 0.2 if turns <= max_turns/2          # cap bonus for terminating early
      + 0.05 if OSWORLD_MCP_BONUS=1 and at-least-one-MCP-succeeded
```

Total lives in roughly `[-1.1, 1.1]`. Set `OSWORLD_MCP_BONUS=1` to enable the
diagnostic MCP bonus (off by default - matches upstream `outcome_only_rl`).

## Troubleshooting

Read `PITFALLS.md` before changing anything. TL;DR:

1. Every session log lives at `${AGENT_DEBUG_LOG_DIR}/<session_id>.log`.
   That's where real tracebacks live; the Ray/Relax top-level stack is a red
   herring.
2. `reward=0` on every session + `agent_turns=1` on every session means the
   agent never entered its loop - usually the URL / `/v1` / `--agent-env` triad
   (PITFALLS 1-3).
3. GPU utilisation stuck on 2/8 cards -> the SGLang router policy isn't
   `consistent_hashing` (PITFALLS 4).
4. Sessions failing at step 0 -> nex sandbox quota or credentials; check
   `NEX_WORKSPACE_ID` / `NEX_API_KEY` and the per-session log.
