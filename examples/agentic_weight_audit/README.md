# Agentic weight-version audit

Track committed generation fragments through actual training consumption. This is a diagnostic side channel: sampling, rewards, loss masks and GRPO grouping retain their existing behavior. Implements [contributor task 15](https://github.com/redai-studio/community/blob/main/contributor-program/2026-cohort-2/official-task.md#task-15).

## Run the CPU example

With the repository's CPU dependencies installed, run from the repository root:

```bash
python -m examples.agentic_weight_audit.demo --output /tmp/agentic-weight-audit.json
```

This simulates successful publications using the real SessionForest and training-data converter; it does not run GPU inference or training. Four tokens come from version 10, a tool waits across publications 11 and 12, then six tokens come from version 12. Three observation tokens occupy continuation coordinates but have zero loss mask. Publication 13 occurs before consumption.

Expected response spans are `[0,4)` and `[7,13)`, generation span is 2, consumption ages are 3 and 1, post-generation wait is 1, and token-weighted consumption age is 1.8. Raw labels are mapped to successful-publication sequences, not subtracted directly.

## Inspect training

Enable the existing `--save-debug-train-data '/path/train-{rollout_id}-{rank}.pt'` option. Each reporting rank's sample records contain `weight_version_audit`. Aggregate trusted local dumps from TP0 / last PP stage / CP0 for every DP rank, without duplicate files:

```bash
python -m examples.agentic_weight_audit.demo \
  --train-dump /path/train-0-0.pt /path/train-1-0.pt \
  --output /tmp/agentic-weight-summary.json
```

The existing dump format uses pickle; only load trusted files. Aggregation uses token counts and weighted sums rather than averaging rank means. Custom-converter rows without provenance make the overall missing rate unavailable.

The GPU recipe is `scripts/training/text/run-agentic-weight-audit-smoke.sh`. Set `MODEL_PATH` to an existing Qwen2.5-7B-Instruct directory, `OUTPUT_DIR` to a fresh directory, and `RAY_ADDRESS` to an isolated Ray dashboard. Submit through the existing `scripts/entrypoint/ray-job.sh`. `SMOKE_MODE=single` is the default; `dp2` uses two colocated GPUs and `async2` uses one GPU per role. `PARTIAL_SMOKE=1` exercises a tool wait across two publications in colocate modes only.

## Semantics and limits

- Each committed attempt retains continuation-relative token coordinates and Session, turn, request and response identities. Only the selected lineage is exported; counts use the final loss mask.
- Versions are comparable only within one publisher epoch. Successful rollout publication hooks record a monotonic sequence before generation resumes. Missing labels, publication failures and incomparable epochs remain unknown.
- For source sequence `v`, generation-end snapshot `e` and consumption snapshot `c`: generation age is `e-v`, post-generation wait is `c-e`, and consumption age is `c-v`. Generation span is the sample's `max(v)-min(v)`.
- Generation end is the final attempt's permit-release control-plane boundary, not the physical completion instant of the final GPU token.
- Consumption is observed after training and before the next publication, using the existing serial Actor lifecycle. All ranks participate in the snapshot broadcast; only reporting ranks emit reports. Streaming mini-windows are included; legacy VPP replays and dummy batches are excluded.
- The scalar backend version relies on Relax's abort-before-update contract. A backend-reported mixed-version attempt, adapter-only LoRA and custom converters currently yield unknown numerical provenance.
- Full auditing is limited to Agentic rollout. GPU TP/PP/CP and multi-node combinations have not been validated by this example.

## 中文说明

本示例实现任务 15 的旁路审计：逐 attempt 保存成功提交的 token 来源，按最终 loss mask 计数，通过成功发布序号分别计算生成跨度、生成年龄、生成后等待和实际训练消费年龄。原始版本标签可能跳号，不能直接作为发布次数。

CPU 示例使用真实 SessionForest 和转换器模拟发布，不冒充 GPU 实训。预期区间为 `[0,4)`、`[7,13)`，消费年龄为 3、1，加权年龄为 1.8；工具 observation 不参与计数。

真实训练复用 `--save-debug-train-data`，仅汇总各 DP reporting rank 的可信 dump，不重复输入文件。上述命令和 GPU 配方支持单卡、双卡 DP2 及 1+1 fully-async；跨两次发布的工具等待使用 colocate 的 `PARTIAL_SMOKE=1`。每次使用新的输出目录。

未知版本、登记失败、跨 epoch、混合版本 attempt、adapter-only LoRA 和自定义转换器不会被伪装成零年龄。生成结束取控制面 permit release 边界。普通非 Agentic rollout 不在本次范围；GPU TP/PP/CP 与多节点组合仍未验证。
