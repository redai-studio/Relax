# Vendor snapshot — hybrid-routing-agent

> **本仓库不再 vendor `desktop_env`。** 它从 `$OSWORLD_REPO` 指向的 nex 版 OSWorld
> checkout 导入 (providers/nex + MCP 版 DesktopEnv + guest 修复), 这样评测和 RL
> 共用同一份环境代码。vendor/ 下只保留 `mm_agents` 与 `agents`。
> 同时删除的还有 `vendor/mcp_tools` (guest 侧 MCP 载荷, 由 OSWorld 侧 MCP_SRC_ROOT 注入)。

This directory vendors code from
[`redai-infra/hybrid-routing-agent`](https://github.com/redai-infra/hybrid-routing-agent)
so the OSWorld GUI+MCP recipe can be built on top of Relax's agentic stack
without git-submodule friction.

## Provenance

- **Upstream repo**: <https://github.com/redai-infra/hybrid-routing-agent>
- **Commit**: `a54d59c8049a0cd7192d95863155e133a1ba41ea`
- **Fetched**: 2026-08-18
- **License**: Apache-2.0 (see `vendor/hybrid-routing-agent-LICENSE`);
  the nested `OSWorld-main/` subtree it originally lived in carries its own
  Apache-2.0 (`vendor/OSWorld-LICENSE`).

## What is vendored

| Local path                   | Upstream path               | Purpose                                                                 |
| ---------------------------- | --------------------------- | ----------------------------------------------------------------------- |
| `vendor/desktop_env/`        | `OSWorld-main/desktop_env/` | OSWorld VM controller + MCP-injection layer                             |
| `vendor/mm_agents/`          | `OSWorld-main/mm_agents/`   | `hybrid_agent_local.py` (message array, tool_call parse, image history) |
| `vendor/mcp_tools/`          | `mcp_tools/`                | MCP server + curated tool set, injected into the guest VM               |
| `vendor/agents/`             | `agents/`                   | BM25 `ToolRetriever` (top-K MCP tool retrieval)                         |
| `tools/tools_registry.json`  | `tools/tools_registry.json` | Retriever corpus (tool name → doc string)                               |
| `vendor/UPSTREAM_PATCHES.md` | `OSWorld-main/PATCHES.md`   | Upstream's own diff record vs. `xlang-ai/OSWorld`                       |

## What is NOT vendored

- `evaluation_examples/` (309 task JSONs, ~10s of MB) — dataset, lives under
  `$DATA_DIR/osworld/evaluation_examples/`. See `scripts/prepare_data.py`.
- `rl_rollout_local_qwen3vl.py` / `osworld_rl.py` / `train/` / `reward/` —
  verl-side orchestrator + training loop. Superseded by Relax core.
- `configs/` — Hydra configs, superseded by `osworld_config.yaml` + CLI flags.
- Docker + qcow2 assets — fetched by `scripts/prepare_vm.sh` at deploy time.

## Local modifications

**None yet.** This is a P0 pristine snapshot.

Any future edit MUST be recorded here with:

1. Section header: `### <YYYY-MM-DD> — <short summary>`
2. Files touched (path relative to this directory).
3. Reason + upstream issue link if applicable.
4. Whether the change was upstreamed / is pending upstream.

The goal is to keep the diff auditable so future re-syncs from upstream can be
done mechanically (`diff -ru vendor/ /tmp/fresh-upstream/` should equal the
concatenation of the entries below).

## How to refresh

```bash
# 1. Save current patches.
cd examples/osworld_agentic
diff -ruN /tmp/hybrid-routing-<prev-sha>/OSWorld-main/desktop_env vendor/desktop_env > /tmp/desktop_env.patch
# ... same for mm_agents, mcp_tools, agents.

# 2. Re-fetch upstream (see scripts/prepare.sh, TBD).
# 3. Re-apply saved patches; resolve conflicts; append entries to this file.
```

## Modifications log

<!-- add entries here as edits accrue -->
