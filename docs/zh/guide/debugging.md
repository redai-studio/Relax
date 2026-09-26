# 调试指南

用于排查 Relax 训练中精度问题和隔离训练/推理组件的实用指南。

## 精度排查

### 1. 检查 Rollout 生成的 Response

在日志中搜索 `Finish rollout`，定位当前的 rollout ID 和生成的 response。首先判断 response 是否为正常的、连贯的回复。

- **如果第一步就出现了乱码/胡言乱语**，一般是由于 checkpoint 加载异常或模型转换异常。一个比较彻底的方法是在对应模型的 SGLang `load_weights` 实现中保存所有参数，查看与加载的 checkpoint 中是否一致。如果所有参数更新都正确还出现问题，有可能是 SGLang 里有一些特殊的 buffer 在 `release_memory_occupation` 阶段被释放了。如果是用 pretrain 模型进行的测试，可以换成同结构模型的 instruct 版本，查看这种乱码是不是 pretrain 模型特有的。

- **如果第一步回复正常，但后续 step 出现乱码**，说明训崩了。需要更细致地去检查每一步的 reward 计算是否符合预期、超参数配置等因素。

### 2. 检查 log_probs 和 ref_log_probs

查看第一步打印的 rollout stats，确认 `log_probs` 和 `ref_log_probs` 是否完全相等（即第一步 KL = 0），且值较小。

- **如果不是完全相等**，一般是 Transformer Engine 中的某些非确定性 kernel 导致的。例如，在某些版本的 TE 里，Megatron 需要 `--attention-backend flash` 来强制使用 Flash Attention，从而避免 Context Parallelism (CP) 下 fused attention 的数值不稳定。

- **如果数值较大**（例如 > 1），一般有两种可能：
  - 值非常大，应该是训练配置有问题。
  - 值只是比 SFT loss 的状态略大（例如 instruct 模型的 logprob 到了 0.8），有可能是数据不符合训练的 chat template，或者不符合冷启动的分布。

### 3. 检查第一步的 KL 和 grad_norm

在推一训一（`num_steps_per_rollout == 1`）的配置下，检查第一步的 KL 是否为 0，`grad_norm` 是否较小。

此阶段的问题基本上是 Megatron / Transformer Engine 相关的 bug。例如：

- MoE 模型需要开启 `--moe-permute-fusion`。

## SGLang 运行时崩溃

### 1. Stop string 在 agentic 高并发下触发 IMA

**症状**：agentic rollout 跑到中段（rollout 已经运行了几十秒到几分钟）时，SGLang scheduler 抛异常，紧接着 `SIGQUIT received. ... It usually means one child failed`，engine 被 kill，整个 rollout 停摆：

```
[TP0] Scheduler hit an exception: Traceback (most recent call last):
  ...
  File "sglang/srt/managers/scheduler_output_processor_mixin.py", line 371, in process_batch_result_decode
    next_token_ids = next_token_ids.tolist()
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
```

**根因**：OpenAI-compatible request 里的 `stop` 字符串由 SGLang detokenizer 子进程按 per-token 增量解码 + 子串匹配处理。agentic 场景下 100+ session 并发（每个 session 内又有多轮 chat completion），detokenizer 的单 token 工作量随 stop string 数量线性放大，容易 starve 掉它的 20s 心跳，GPU 侧 async kernel 就会以 IMA 的形式浮现。

**规避**：能用 `stop_token_ids` 就不要用 `stop`。`stop_token_ids` 走 GPU 侧 token id 匹配，完全绕过 detokenizer 子进程。

- 单 token 就能表示的终止符（如 `</tool_call>`、`<|im_end|>`）：查 tokenizer 拿到 id 后直接传 `stop_token_ids`。
- 多 token 序列（如 `</answer>` = `[510, 8944, 29]`）无法用 `stop_token_ids` 表达。这种情况优先用 chat-template 的 turn 终止符（`<|im_end|>`）作为停止条件，再在生成后 regex 提取答案，仍然可以避开 `stop`。

在 agent app 里通过 `extra_body` 传 `stop_token_ids`：

```python
extra_body = {"stop_token_ids": [248059, 248046]}  # </tool_call>, <|im_end|>
resp = await client.chat.completions.create(
    model=...,
    messages=messages,
    extra_body=extra_body,
    ...
)
```

具体示例可参考 `examples/deepeyes_v2_agentic/app/agent.py` 与 `examples/deepeyes_v2_agentic/app/deepeyes_v2_config.yaml` 的注释。

## 训练推理单独调试

Relax 支持将训练和推理组件分开独立运行，从而实现：

- 在调试推理部分时，只用少量卡就可以启动任务。
- 在调试训练部分时，可以保证模型输入固定，去除 rollout 的随机性。

### 可用的调试参数

以下 CLI 参数用于开启隔离调试：

| 参数 | 描述 |
|---|---|
| `--debug-rollout-only` | 仅初始化 SGLang（跳过 Megatron）。用于推理调试。 |
| `--debug-train-only` | 仅初始化 Megatron（跳过 SGLang）。用于训练调试。 |
| `--save-debug-rollout-data <path>` | 将 rollout 结果保存到指定路径，供后续回放使用。 |
| `--load-debug-rollout-data <path>` | 从指定路径加载 rollout 数据。自动设置 `--debug-train-only`。 |
| `--dump-details <dir>` | 保存训练的全部细节（自动开启 rollout 数据保存）。 |
| `--load-forge-rollout-data <path>` | 回放已保存的 rollout dump，**同时保持 SGLang 存活**（不会设置 `--debug-train-only`）。需配合 `--rollout-function-path relax.engine.rollout.forge_load.generate_rollout`。 |

### 工作流 1：仅调试推理

使用 `--debug-rollout-only` 跳过 Megatron 初始化。只会启动 SGLang 引擎，可以用更少的 GPU 来测试推理部分。

```bash
python3 relax/entrypoints/train.py \
    --debug-rollout-only \
    --rollout-num-gpus 8 \
    --rollout-num-gpus-per-engine 8 \
    # ... 其他 rollout 参数
```

可以配合 `--save-debug-rollout-data` 来保存 rollout 结果：

```bash
python3 relax/entrypoints/train.py \
    --debug-rollout-only \
    --save-debug-rollout-data /your/saved/debug/data_{rollout_id}.pt \
    # ... 其他 rollout 参数
```

### 工作流 2：仅调试训练

使用 `--load-debug-rollout-data` 加载预先保存的 rollout 数据，仅运行训练流程。这会自动设置 `debug_train_only=True`，因此不会初始化 SGLang。

```bash
python3 relax/entrypoints/train.py \
    --load-debug-rollout-data /your/saved/debug/data_{rollout_id}.pt \
    # ... 其他训练参数
```

这种方式特别适用于：

- 在不等待 rollout 的情况下调优并行配置（TP、PP、EP、CP）。
- 用确定性输入来复现和修复训练相关的问题。
- 快速迭代 loss 计算或优化器的修改。

### 工作流 3：全量细节转储

使用 `--dump-details` 保存训练的全部细节供事后分析。设置后会自动启用：

- `--save-debug-rollout-data`，保存路径为 `<dir>/rollout_data/{rollout_id}.pt`
- `--save-debug-train-data`，保存路径为 `<dir>/train_data/{rollout_id}_{rank}.pt`

```bash
python3 relax/entrypoints/train.py \
    --dump-details /path/to/dump/dir \
    # ... 其他参数
```

::: tip 提示
`--dump-details` 在收集 bug 报告数据时也非常有用 — 它会捕获复现问题所需的所有信息。
:::

### 工作流 4：保持 SGLang 存活回放 Rollout（显存分析）

`--load-debug-rollout-data`（工作流 2）会完全跳过 SGLang，因此只能孤立地测量训练循环。若要分析**真实的 colocate 显存占用** —— 训练峰值、权重同步、以及 offload/onload 切换 —— 就需要 SGLang、router 和权重更新全程存活。`forge_load` 用回放已保存的 rollout dump 来替代真实生成，从而保持整条流水线存活，只省掉（缓慢的）真实生成。

先抓一份 dump（例如通过工作流 1 或 `--dump-details`），再回放：

```bash
python3 relax/entrypoints/train.py \
    --rollout-function-path relax.engine.rollout.forge_load.generate_rollout \
    --load-forge-rollout-data /path/to/rollout_data/data_{rollout_id}.pt \
    # ... 保持与产生该 dump 的那次运行相同的 模型 / colocate / batch 参数
```

`--load-forge-rollout-data` 支持两种写法，沿用 `--save-debug-rollout-data` 的 `format(rollout_id=...)` 约定：

- **字面路径（literal）** —— `--load-forge-rollout-data /path/to/rollout_data/0.pt`
  不含 `{rollout_id}` 占位符：**每个** rollout step 都复用同一个文件。适合用一份 dump 反复回放 `--num-rollout > 1` 的显存测试。
- **模板路径（template）** —— `--load-forge-rollout-data /path/to/rollout_data/data_{rollout_id}.pt`
  含 `{rollout_id}`：每个 step 加载**各自**的文件（`data_0.pt`、`data_1.pt` ……）。若某个 step 的文件缺失，训练路径会回退到 `0` 那份 dump。

::: warning 注意
batch 相关参数（`--global-batch-size`、`--n-samples-per-prompt`、并行度）要与产生 dump 的那次运行保持一致 —— dump 必须恰好包含一个 step 的样本量。**不要**与 `--debug-rollout-only` 或 `--load-debug-rollout-data` 同时使用，二者都会破坏其目的（不训练，或跳过 SGLang）。`forge_load` 不会触发解码期显存，因此不能替代对生成阶段峰值的测量。
:::
