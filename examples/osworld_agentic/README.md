# OSWorld GUI+MCP hybrid agentic RL

Relax agentic recipe for the two headline runs from
[hybrid-routing-agent](https://github.com/redai-infra/hybrid-routing-agent):

- `outcome_only_rl` — pure ±1 outcome-only baseline (74 tasks).
- `context_rl` — same rollout with `skip_on_mcp_success` + 2-image history.

Model: `Qwen3-VL-8B-Thinking`. Env: OSWorld (Docker + QEMU/KVM, MCP-injected
Ubuntu guest). Eval: 309-task OSWorld-MCP set.

## Layout

```
app/
  agent.py                           per-session driver subprocess
  env_osworld.py                     async wrapper over vendor DesktopEnv
  hybrid_agent.py                    RelaxHybridAgent (call_llm → AgenticChatAPIService)
  osworld_config.yaml                context_policy=baseline (outcome_only_rl)
  osworld_config_context.yaml        context_policy=skip_on_mcp_success (context_rl)
reward_osworld.py                    outcome + length + cap + optional MCP bonus
run_agent_app.sh                     agent-side entrypoint (invoked by Relax)
run_qwen3vl_8B_osworld_outcome.sh    training launcher for outcome_only_rl
run_qwen3vl_8B_osworld_context.sh    training launcher for context_rl
scripts/
  prepare_data.py                    evaluation_examples/*.json -> parquet
  prepare_vm.sh                      docker pull + qcow2 stage
  smoke_test.sh                      single-session end-to-end without Ray
env.sh.example                       template — copy to env.sh (git-ignored)
PITFALLS.md                          integration footguns; read before touching
VENDOR.md                            snapshot provenance
vendor/                              read-only upstream (desktop_env + mm_agents + …)
```

## Prerequisites

- Host with `/dev/kvm` accessible (bare-metal or nested-virt enabled node).
- Docker with the KVM device passed through.
- Persistent disk for `${OSWORLD_CACHE_DIR}` (~30 GB free) — `/tmp` gets
  cleaned, see `PITFALLS.md §8`.
- Python deps from `vendor/desktop_env/` and `vendor/mm_agents/`
  (installed as part of the Relax env).

## Quickstart

```bash
# 0. One-time env setup
cp examples/osworld_agentic/env.sh.example examples/osworld_agentic/env.sh
$EDITOR examples/osworld_agentic/env.sh     # set MODEL_DIR / DATA_DIR / SAVE_DIR / ...

# 1. Stage the OSWorld VM assets (docker image + qcow2)
bash examples/osworld_agentic/scripts/prepare_vm.sh

# 2. Convert OSWorld task JSON → parquet
python3 examples/osworld_agentic/scripts/prepare_data.py \
    --input-dir /path/to/evaluation_examples \
    --output-dir ${DATA_DIR}/osworld/data \
    --train-manifest scripts/manifests/train_74.txt \
    --eval-manifest  scripts/manifests/eval_309.txt

# 3. Sanity check — one session, no Ray
OSWORLD_TASK_JSON=/path/to/evaluation_examples/some-task.json \
OPENAI_BASE_URL=http://your-openai-compatible-endpoint \
OPENAI_API_KEY=sk-... \
bash examples/osworld_agentic/scripts/smoke_test.sh

# 4. Training
bash examples/osworld_agentic/run_qwen3vl_8B_osworld_outcome.sh
# or
bash examples/osworld_agentic/run_qwen3vl_8B_osworld_context.sh
```

## Two variants — what actually differs

Both launchers share the exact same GRPO / SGLang / Megatron / colocate
config. The only behavioural knob is which yaml the session driver loads,
selected via `OSWORLD_CONFIG_PATH`:

| File                                  | context_policy         | Image history | Recipe            |
| ------------------------------------- | ---------------------- | ------------- | ----------------- |
| `app/osworld_config.yaml`             | `baseline`             | 2 images      | `outcome_only_rl` |
| `app/osworld_config_context.yaml`     | `skip_on_mcp_success`  | 2 images      | `context_rl`      |

`skip_on_mcp_success` drops the screenshot from turns immediately following
a successful MCP call — this frees up context tokens and biases the agent
toward MCP over pixel-diffing.

## Reward

`reward_osworld.py` shapes the raw VM-evaluator outcome:

```
score = outcome                       # ±1 from OSWorld's postconfig+metrics
      - 0.05 * turns                  # per-turn length penalty
      + 0.2 if turns <= max_turns/2   # cap bonus for terminating early
      + 0.05 if OSWORLD_MCP_BONUS=1 and at-least-one-MCP-succeeded
```

Total lives in roughly `[-1.1, 1.1]`. Set `OSWORLD_MCP_BONUS=1` to enable
the diagnostic MCP bonus (off by default — matches upstream `outcome_only_rl`).

## Troubleshooting

Read `PITFALLS.md` before changing anything. TL;DR:

1. Every session log lives at `${AGENT_DEBUG_LOG_DIR}/<session_id>.log`.
   That's where real tracebacks live; the Ray/Relax top-level stack is a
   red herring.
2. `reward=0` on every session + `agent_turns=1` on every session means the
   agent never entered its loop — usually the URL/`/v1`/`--agent-env` triad
   (PITFALLS.md §1–3).
3. GPU utilisation stuck on 2/8 cards → the SGLang router policy isn't
   `consistent_hashing` (PITFALLS.md §4).
