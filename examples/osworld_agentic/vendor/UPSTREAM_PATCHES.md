# OSWorld-main — divergence from upstream

This directory is a **vendored, modified fork** of the
[OSWorld](https://github.com/xlang-ai/OSWorld) benchmark. It is kept in-tree
(rather than as a submodule) because the agent / rollout / MCP-injection changes
are deep and co-evolve with the RL code in the parent repo. Upstream `LICENSE`
is preserved; see `NOTICE` at the repo root for attribution.

## What this project added / changed

**New agent + rollout entry points**
- `rl_rollout_local_qwen3vl.py` — unified rollout driver used for BOTH RL
  training rollouts and evaluation (`--rollout_type {rollout,evaluation}`).
  Manages the Docker+QEMU env fleet, vLLM lifecycle, hybrid GUI+MCP action
  parsing, and context-compression policies (`--context_policy`,
  `--max_image_history_length`).
- `start_8gpus_qwen3vl.sh` / `stop_8gpus.sh` — 8× vLLM launcher for Qwen3-VL-8B.
- `mm_agents/hybrid_agent_local.py` — the hybrid GUI+MCP agent (tool retrieval
  via `agents.tool_retriever.ToolRetriever`, BM25, top-18).
- `mm_agents/qwen3vl_agent_local.py`, `mm_agents/tool_doc_hints.py`.

**MCP injection into the VM**
- `desktop_env/desktop_env.py` — injects an MCP server + client into the guest
  VM at boot. Source dir is `MCP_SRC_ROOT` (env var; default `OSWorld-MCP`).
  MCP server listens on :9292 inside the VM; screenshots/pyautogui on :5000.

**Task splits** (`evaluation_examples/*.json`)
- Added: `test_all_no_internet.json` (309, the headline eval set),
  `train_v2_*.json`, `held_out_v2_*.json`, `ctx_degraded_*.json`, `smoke_test*`.
- See `../data/splits/` and the repo README for split semantics.

**Removed from this export** (products, not source)
- `cache/`, `logs/`, `results/`, `docker_vm_data/*.qcow2`, `.env`,
  `osworld.egg-info/`.

## Upstream

Regenerate a clean diff by cloning upstream OSWorld at the commit this fork was
based on and diffing against this directory. Stock OSWorld runners and agents for
*other* models (`run_multienv_*.py`, `run_*.py`, `sample_*`, `serve_*`, and the
non-qwen3vl `mm_agents/*`) have been **removed** — this fork keeps only the
qwen3vl rollout path (`rl_rollout_local_qwen3vl.py` + `lib_run_single.py` +
`mm_agents/{qwen3vl_agent_local,hybrid_agent_local,...}`) actually used here.
