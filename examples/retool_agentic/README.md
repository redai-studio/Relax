# ReTool：Relax Agentic 训练示例

使用 Qwen2.5-7B 进行多轮 Python 工具调用与 GRPO 训练。模型生成 `<code> ... </code>` 后执行一次隔离的 Python 程序，将真实结果放入 `<interpreter>`，再继续推理。`relax/agentic` 管理每条轨迹的进程、上下文、生成 token/logprob 和训练导出；工具反馈只作为上下文，不参与 loss。

本目录包含 agent、数据准备、判分、词表 hook、sandbox、checkpoint 校验、训练脚本和测试。除 Relax 框架及其标准训练依赖外，不依赖其他 example，可单独加入兼容的 Relax 仓库。目录内不包含模型权重、训练数据或 sandbox 镜像。

## 环境与模型

需要已安装 Relax 及其 Megatron、SGLang 训练依赖的 Linux 单机 8 卡环境。模型目录应包含完整 HF 权重、config 和 `tokenizer.json`。本例针对 Qwen2.5-7B 的词表和模板；可以使用公开的 Qwen2.5-7B-Instruct，或使用该模型前序 SFT 导出的 HF checkpoint。本例提供 RL 训练，不包含 SFT 训练流程；更换初始权重会影响结果。

本地工具执行需要以下任一后端：

- Landlock：Linux x86_64、支持 Landlock 的内核、root 驱动、`libseccomp.so.2`，以及可用的 NumPy/SciPy。
- Apptainer：root 驱动、已有 `apptainer`，并允许建立 mount/PID/network namespace 和 cgroup。使用下文固定的 SandboxFusion 镜像。
- 外部服务：设置 `SANDBOX_FUSION_URL`，服务应实现相同的隔离和执行协议，并对运行 agent 的 worker 可达。

工具服务启动时检查运行条件与隔离，失败会阻止训练启动。外部服务由运行者负责部署和隔离。

## 准备数据

`MODEL_DIR` 指向具体 HF 模型目录，`DATA_DIR` 存放原始和转换后的数据，`SAVE_DIR` 是持久化 checkpoint 根目录。先设置三个目录：

```bash
export MODEL_DIR=/path/to/Qwen2.5-7B-Instruct
export DATA_DIR=/path/to/retool-data
export SAVE_DIR=/path/to/checkpoints
```

如果还没有初始模型和原始数据，可以下载以下固定版本：

```bash
hf download Qwen/Qwen2.5-7B-Instruct \
  --revision a09a35458c702b33eeacc393d103063234e8bc28 \
  --local-dir "$MODEL_DIR"
hf download BytedTsinghua-SIA/DAPO-Math-17k --repo-type dataset \
  --revision 65877096c24ffa7abc4e4fa5edb95cf3413a5674 \
  --local-dir "$DATA_DIR/dapo_raw"
```

将 `--input` 指向下载的完整训练 parquet：

```bash
python -m examples.retool_agentic.prepare_data \
  --input "$DATA_DIR/dapo_raw/data/dapo-math-17k.parquet" \
  --model "$MODEL_DIR" --out "$DATA_DIR/train.jsonl"
```

转换器验证固定源数据的重复记录，隔离同题答案冲突的 14 个 native ID，得到 17903 个问题；不截断、静默过滤超长 prompt，也不执行数据中的代码。输出旁的 manifest 记录转换统计和摘要。已有同协议的 `train.jsonl` 可以直接使用，无需重新转换；已有输出不会被覆盖。

## 启动

独占的单机训练容器中，从 Relax 仓库根目录运行：

```bash
bash examples/retool_agentic/scripts/run-qwen25-7B-grpo-8xgpu.sh
```

脚本自动 source `scripts/entrypoint/local.sh`，建立本机 Ray，再阻塞提交任务并跟随日志。该入口会清理本机旧进程，适用于新的独占任务环境。

已有 Ray 集群时使用框架入口，复用该集群：

```bash
RAY_ADDRESS=http://ray-head:8265 \
  bash scripts/entrypoint/ray-job.sh \
  examples/retool_agentic/scripts/run-qwen25-7B-grpo-8xgpu.sh
```

默认 ClearML project 为 `Relax/retool-agentic`，稳定实验名为 `qwen25-7b-retool-agentic-grpo-8xgpu`；分别可用 `PROJECT_NAME` 和 `EXP_NAME` 覆盖。ClearML 使用运行环境已有凭证，每次启动创建带时间戳的新 task。无需 ClearML 时在命令后追加 `--no-use-clearml`。显式设置非空 `RAY_NO_WAIT` 才非阻塞，此时脚本日志只包含提交输出，需用 `ray job logs` 查看训练。

### Sandbox

`train.py` 为整个 Ray Job 管理工具服务，训练退出后关闭服务。自动优先使用 Landlock；内核未实现 Landlock 时使用 Apptainer。首次使用 Apptainer 会将固定镜像缓存到 `$SAVE_DIR/.retool-sandbox/`，也可预先准备：

```bash
apptainer pull /path/to/sandboxfusion.sif \
  docker://vemlp-cn-beijing.cr.volces.com/preset-images/code-sandbox:server-20250609
export SANDBOX_IMAGE=/path/to/sandboxfusion.sif
bash examples/retool_agentic/scripts/run-qwen25-7B-grpo-8xgpu.sh
```

Apptainer 启动器只绑定可信 bootstrap 和 cgroup 控制目录，不挂载模型、数据、仓库或 GPU。模型代码以 uid/gid 65534 执行，每次调用是独立的无状态 Python 程序，限 30 秒、1 GiB；禁止网络、任意文件访问、子进程和线程。镜像内的 SciPy/HiGHS 限为单线程。

设置 `SANDBOX_FUSION_URL=http://sandbox-host:8080/run_code` 时复用外部服务，不启动或关闭它。需要手动运行本例的 Apptainer 服务时：

```bash
SANDBOX_IMAGE=/path/to/sandboxfusion.sif \
  bash examples/retool_agentic/scripts/run-sandbox-apptainer.sh
```

服务默认监听 `127.0.0.1:8080`，可用 `SANDBOX_HOST`、`SANDBOX_PORT` 覆盖。多节点或跨容器运行时，需要自行确保服务可达和访问范围。

## 保存与恢复

模型、optimizer、scheduler 和数据 RNG 保存在 `$SAVE_DIR/$EXP_NAME`，默认每 25 次更新保存，最多保留 8 份 native checkpoint。HF 导出位于同目录的 `hf_{rollout_id}`；rollout ID 从 0 开始，第一次保存为 `iter_0000024` / `hf_24`。

保持 `SAVE_DIR`、`EXP_NAME`、初始模型配置/tokenizer、数据内容、seed、batch 和每题轨迹数不变，再执行同一脚本，即从最新完整 checkpoint 恢复。迁移训练环境时携带完整实验目录，包括 tracker、native 模型/optimizer 状态和 `dataset/retool_*.json`。单独复制 `hf_*` 只能从权重开始新实验，不能恢复完整训练状态。改变 `SAVE_DIR` 或 `EXP_NAME` 可以新开实验。

Agentic 在某个 agent 进程失败时丢弃整组，并重新抽样补足 batch。抽样编号和训练轮次分别计数，checkpoint 同时记录 RNG 与下一抽样编号；权重版本校验使用 Session 的实际生成轮次。兼容补样前格式的同协议 RNG checkpoint。动态过滤、额外 oversampling、prelaunch、partial rollout、fully async 和 Eval 尚未接入本例，请勿开启这些选项。

## 默认配置与行为

| 配置               | 默认值                                                      |
| ------------------ | ----------------------------------------------------------- |
| 算法               | GRPO，200 rollout，每轮一次训练更新                         |
| 训练 batch         | 64 组 × 每组 16 条轨迹 = 1024                               |
| Resident 并发      | 16 组，最多 256 个 agent 进程                               |
| 推理               | 8 个 SGLang engine，每 engine 32 个请求，consistent hashing |
| Token 预算         | prompt 2048；回复与反馈合计 8192；context 10240             |
| Assistant 轮数上限 | 16，终止轮不执行多余代码                                    |
| Sampling           | temperature=1，top-p=1，top-k=-1                            |
| 训练               | Actor TP=2，lr=1e-6，clip=0.2，KL/entropy=0                 |
| 工具执行           | 32 路并发，每次 30 秒 / 1 GiB                               |

每次模型请求发送完整历史，使用固定模板和 `stream=false`。工具结果原样追加为非空的 `\n<interpreter>\n...\n</interpreter>\n`，继续同一 assistant token 流，不额外插入 ChatML 边界；无需 reasoning/tool-call parser。Session 形成线性上下文并导出一份训练样本。生成 token 的 mask 为 1，实际反馈 token 的 mask 为 0，logprob 来自原生推理后端。词表 hook 和 sampling bias 屏蔽 padded token。

奖励为答案判分的 ±1，减去每次工具失败的 0.1，无下限；保留 DAPO verifier 的 `Answer:` 优先、boxed 回退规则。这是 GRPO 配方，不是原论文 PPO 配方的等价复现。

代码错误、超时、非法调用和隔离程序的信号退出作为工具反馈；HTTP、服务或响应协议异常直接报错，不伪装成模型负奖励。每段新代码只执行一次，不做 HTTP 重试或历史重放。模型请求不设 wall timeout，以允许 Runtime 暂停；`--agent-timeout 1800` 限制 agent 的 active time，sandbox 请求另有 70 秒 wall timeout。

机器若出现 NVSwitch multicast 建链故障且 NCCL 返回 CUDA 999，可在重新启动前设置 `NCCL_NVLS_ENABLE=0`。脚本会将显式值传入 Ray worker；这属于通信故障绕过配置。

## 代码与测试

- `agent.py`、`protocol.py`：模型请求、工具循环、提示词与上下文模板。
- `data_source.py`、`prepare_data.py`：抽样、恢复和 DAPO 转换。
- `reward.py`、`math_dapo.py`、`vocab.py`：奖励、答案判分和词表训练 hook。
- `train.py`、`checkpoint.py`：工具服务生命周期与完整恢复检查。
- `sandbox*.py`、`landlock*.py`、`tool_service.py`：隔离执行后端及客户端。
- `scripts/`：训练与独立 sandbox 入口。

```bash
python -m pytest examples/retool_agentic/tests -q

# 使用实际模型的 tokenizer 检查模板和 token 对齐。
RETOOL_TEST_MODEL_DIR="$MODEL_DIR" \
  python -m pytest examples/retool_agentic/tests -q
```

测试通过真实 compiler、上下文匹配和 SessionForest 导出，与目录内冻结的原生参考循环比较 token、mask、logprob 和终止行为；同时覆盖子进程 HTTP、数据补样恢复、sandbox 生命周期和启动命令。CPU 测试用固定模型响应和执行结果替代 GPU 推理；真实吞吐和 GPU 兼容性需在目标训练环境验证。
