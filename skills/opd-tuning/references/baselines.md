# OPD baselines — 已跑通的教师侧配置

**这是一份快照（2026-08 抄自 `examples/on_policy_distillation/**`），会随仓库漂移。**
用之前先跑一次下面的命令刷新，再和表格对照；对不上以命令输出为准。

```bash
# 刷新：列出所有 OPD 脚本实际用的 teacher flag 与 env
grep -rn -- "--teacher-sglang\|--teacher-num-gpus-per-engine\|--opd-teacher-routes\|--teacher-hf-checkpoint" \
  examples/on_policy_distillation/
grep -rn "SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK\|SGLANG_LOGITS_PROCESSER_CHUNK_SIZE\|SGLANG_VLM_CACHE_SIZE_MB\|RELAX_PROPAGATE_ENV_VARS" \
  examples/on_policy_distillation/
# 拓扑：resource / rollout / teacher 三者
grep -rn "RESOURCE_JSON=\|--resource\|ACTOR_GPUS=\|ROLLOUT_GPUS=\|TEACHER_GPUS=" \
  examples/on_policy_distillation/
```

表格是**合理区间锚点**，不是推荐值 —— 其中好几条明显没调过（见最后一节）。
`replicas` 是推导值不是脚本里写的：

```
replicas = teacher_gpus / num_teachers / --teacher-num-gpus-per-engine
```

（推导逻辑在 `relax/utils/opd/opd_utils.py` 的 `maybe_start_managed_opd_teacher`
与 `_start_managed_multi_teacher`。不写该 flag = 1 副本吃满 teacher 卡数。）

## 单教师（`--teacher-hf-checkpoint`）

| 脚本 | actor/rollout/teacher | teacher | TP | replicas | mem-frac | chunked-prefill | max-prefill | max-running |
|---|---|---|---|---|---|---|---|---|
| `math_opd/run-opd-qwen35-35B-A3B-8xgpu-colocate.sh` | 8 / 4 / 4 | 35B-A3B (RL ckpt) | 4 | 1 | 0.5 | (default) | (default) | (default) |
| `vision_opd/run-vision-opd-qwen3.5_9b-35ba3b-8xgpu-colocate.sh` | 8 / 4 / 4 | 35B-A3B | 4* | 1 | 0.8 | 4096 | (default) | 32 |
| `vision_opd/run-vision-opd-qwen3.5_9b-35ba3b-8xgpu-2teacher-colocate.sh` | 8 / 4 / 4 | 35B-A3B | 2 | 2 | 0.8 | 4096 | (default) | 32 |
| `vision_opd/run-opd-qwen3.5_35ba3b-122ba10b-128xgpu-colocate.sh` | 128 / 64 / 64 | 122B-A10B | 8 | 8 | 0.6 | 8192 | 16384 | 64 |

\* 该脚本没写 `--teacher-num-gpus-per-engine`，落到默认。

## Agentic OPD（都是 actor 8 / rollout 4 / teacher 4）

| 脚本 | student | TP | replicas | mem-frac | student engine |
|---|---|---|---|---|---|
| `agentic_opd/alfworld/run-alfworld-opd-qwen35-35B-A3B-8xgpu.sh` | 35B-A3B | 4 | 1 | 0.7 | 1×TP4 @0.7 |
| `agentic_opd/search_qa/run-search_qa-opd-qwen3-1.7B-8xgpu.sh` | 1.7B | 1 | 4 | 0.7 | 4×TP1 @0.7 |
| `agentic_opd/webshop/run-webshop-opd-qwen35-35B-A3B-8xgpu.sh` | 35B-A3B | 4 | 1 | **0.25** | 1×TP4 @0.7 |

三个都没动 chunk / prefill token 预算，都没开 logits chunking。

## 多教师 MOPD（`--opd-teacher-routes`）

| 脚本 | actor/rollout/teacher | 教师数 | TP | replicas/教师 | mem-frac | chunked-prefill | max-prefill | max-running |
|---|---|---|---|---|---|---|---|---|
| `mopd/run-mopd-qwen3-vl-2b-8xgpu-colocate.sh` | 8 / 4 / 4 | 2 (4B text + 4B VL) | 1 | 2 | 0.5 | 16384 | (default) | 256 |
| `mopd/run-mopd-qwen35-9b-8xgpu-colocate.sh` | 8 / 4 / 4 | 1 (9B VL) | 2 | 2 | **0.4** | **32768** | **32768** | 16 |
| `mopd/run-mopd-qwen35-35ba3b-16xgpu-colocate.sh` | 16 / 8 / 8 | 2 (27B text + 27B VL) | 2 | 2 | 0.5 | 16384 | (default) | 128 |

## 环境变量矩阵

| 脚本组 | logits chunk 开关 | chunk size | VLM cache |
|---|---|---|---|
| math_opd / agentic_opd ×3 | — | — | n/a（纯文本） |
| vision_opd 8xgpu ×2 | — | — | 显式设成 sglang 默认值（等于没设） |
| vision_opd 128xgpu | — | — | — |
| mopd ×3 | 1 | 8192 | 只有 9b 那个调大了 |

所有脚本都 `export RELAX_OPD_PREEXPANDED_PATCH=1`（多模态 OPD 必需）。
注意 **传播链路各脚本不一致**：直接 `python3 -m relax.entrypoints.train` 的脚本
只需 `RELAX_PROPAGATE_ENV_VARS`；走 `ray job submit` 的还要并进 runtime-env JSON
（见 `teacher-knobs.md` R-T09）。

## 学生侧对照（同一脚本内）

| 脚本 | rollout 引擎 | student mem-frac | student chunked-prefill | batch × n_samples |
|---|---|---|---|---|
| math_opd 8xgpu | 1×TP4 | 0.6 | (default) | 128 × 1 |
| vision_opd 8xgpu | 2×TP2 | 0.8 | (default) | 4 × 8 |
| vision_opd 128xgpu | 8×TP8 | 0.7 | (default) | 32 × 8 |
| mopd 2b 8xgpu | 4×TP1 | 0.8 | (default) | 32 × 4 |
| mopd 9b 8xgpu | 2×TP2 | 0.7 | 16384 | 32 × 8 |
| mopd 35ba3b 16xgpu | 2×TP4 | 0.6 | (default) | 16 × 8 |

teacher 与 student 的 `mem-fraction-static` **取值方向本应相反**（R-T01）：
mopd 组 teacher 0.4–0.5 < student 0.6–0.8 是对的；
vision_opd 组 teacher 0.8 == student 0.8 是**没调过**的痕迹，不要照抄。

## 快照时点上「明显没调过」的地方

这一节是**判断模式**，比上面的具体数值更耐用 —— 换成新脚本也照样适用：

- **只有 mopd 那三个开了 logits chunking**，其余脚本因此不敢抬 chunk 预算。
  这是最普遍的一处漏项（R-T03 → R-T04 的依赖被卡住）。
- **所有脚本都用旧的 `--teacher-sglang-disable-cuda-graph` 写法**（R-T05）。
  colocate 下影响为零，但写法过时且连 prefill 图一起关。
- **teacher mem-frac 在同模型同拓扑之间差 3 倍**（0.25 vs 0.7，两个 agentic 脚本）——
  这种不一致本身就是"至少有一个没调过"的信号。
- **`--image-max-token-num` 与 `--rollout-max-prompt-len` 不匹配**：
  前者大于后者时，吃满图像预算的样本必然超长被丢。
- **头部注释与实际默认值不符**（有脚本注释写的 GPU 划分早已过时）——
  读脚本时以变量默认值为准，不以注释为准。

## 追加新 baseline

按上表格式追加一行，并在环境变量矩阵补一行。
只有真跑通过的配置才进来，并注明 GPU 型号与显存。
