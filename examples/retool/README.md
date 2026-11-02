# ReTool：原生 SGLang + GRPO

本例使用 Relax 原生 Megatron、SGLang、checkpoint 和周期评测。模型在生成 `</code>` 后暂停，将新代码交给独立的 SandboxFusion 服务，把实际返回放入 `<interpreter>` 后继续生成。模型 token 参与 loss，反馈 token 仅作为上下文。

默认算法与 `liyuanfan/retool@691aa8062` 的作者数据主线对齐：Qwen2.5-7B-Instruct、作者 SFT 两轮/50 步、GRPO 200 步；每步 64 次有放回抽题，每次生成 16 条轨迹。温度/top-p 为 1，top-k 为 -1，clip 两侧均为 0.2，lr 为 1e-6，KL/entropy 为 0。答案判分保留该分支固定 DAPO verifier 的 `Answer:` 优先、boxed 回退规则。总奖励为答案 ±1 减去每次工具失败的 0.1，没有下限、调用奖励或成功奖励。这是该分支的 GRPO 变体；原论文使用 PPO 和纯答案奖励。

一次调用是无状态 Python 程序，限 30 秒/1 GiB；代码错误和超时作为反馈返回模型，非法调用也计失败。HTTP、服务故障等基础设施异常直接报错，不重试执行或转换成模型负奖励。执行服务必须独立部署并提供符合上述约束的隔离；这个 HTTP 客户端本身不提供代码隔离。

## Apptainer 沙箱与直接 RL 验证

可以直接用公开的 Qwen2.5-7B-Instruct 验证 RL 链路，无需先跑 SFT。在 rollout 主机上单独启动不占 GPU 的沙箱：

```bash
apptainer pull /path/to/sandboxfusion.sif \
  docker://vemlp-cn-beijing.cr.volces.com/preset-images/code-sandbox:server-20250609
export SANDBOX_IMAGE=/path/to/sandboxfusion.sif
bash examples/retool/scripts/run-sandbox-apptainer.sh
```

驱动需要 root 建立 namespace 和 cgroup；模型代码实际以 uid/gid 65534、no-new-privs 运行。适配固定镜像的 SandboxFusion lite runner，每次调用使用独立 overlay 和 PID/network namespace，限制 1 GiB/30 秒，禁止创建进程和线程。仅绑定可信启动文件及 cgroup 控制目录，不挂载宿主机 home、仓库、模型、数据或 GPU。启动器修复 Apptainer 根挂载与脚本目录的嵌套挂载问题，收紧镜像原有的 4 GiB 和桥接网络默认值；不能直接使用镜像默认的 isolation=none。

默认监听 `127.0.0.1:8080`，可用 `SANDBOX_HOST`、`SANDBOX_PORT` 覆盖。在另一个终端中，准备好下面的数据后提交：

```bash
export MODEL_DIR=/path/to/Qwen2.5-7B-Instruct
export DATA_DIR=/path/to/retool-data
export SAVE_DIR=/path/to/checkpoints
export SANDBOX_FUSION_URL=http://127.0.0.1:8080/run_code
export RAY_ADDRESS=http://ray-head:8265
bash scripts/entrypoint/ray-job.sh examples/retool/scripts/run-qwen25-7B-grpo-8xgpu.sh
```

直接 RL 使用相同正式参数：200 步、每步 64 次抽题、每题 16 条轨迹，response/context 为 8192/10240。更换为公开基座不代表已复现作者 SFT 后的训练效果。跨容器或节点时，URL 必须对实际 rollout worker 可达。

## 数据准备

在标准 Relax 训练环境中，从仓库根目录执行。`MODEL_DIR` 是初始模型的具体 HF 目录，`DATA_DIR` 放原始及转换后的数据，`SAVE_DIR` 放持久 checkpoint；各节点须能访问这些路径。

```bash
export MODEL_DIR=/path/to/Qwen2.5-7B-Instruct
export DATA_DIR=/path/to/retool-data
export SAVE_DIR=/path/to/checkpoints

hf download Qwen/Qwen2.5-7B-Instruct \
  --revision a09a35458c702b33eeacc393d103063234e8bc28 --local-dir "$MODEL_DIR"
hf download JoeYing/ReTool-SFT train_2000.parquet --repo-type dataset \
  --revision 74943ce52f389f16926702302fcff6255875cbb2 --local-dir "$DATA_DIR/author_raw"
hf download BytedTsinghua-SIA/DAPO-Math-17k --repo-type dataset \
  --revision 65877096c24ffa7abc4e4fa5edb95cf3413a5674 --local-dir "$DATA_DIR/dapo_raw"

python -m examples.retool.prepare_sft \
  --parquet "$DATA_DIR/author_raw/train_2000.parquet" --model "$MODEL_DIR" --out "$DATA_DIR/sft"
python -m examples.retool.prepare_data train \
  --input /path/to/downloaded-dapo-train.parquet --model "$MODEL_DIR" --out "$DATA_DIR/train.jsonl"
```

将最后一条命令的 input 指向下载的完整训练 parquet。转换验证 1791700 行/17917 个 native ID 的重复内容，隔离同题矛盾 gold 的 14 个 ID，得到 17903 题，保留问题和标签。原始 SFT 的工具反馈内嵌在 assistant 文本中，因此保留原文和 token 顺序，仅屏蔽反馈 span；不执行历史代码，不把它改写为其他 chat 协议。

## SFT 与 GRPO

向已运行的 Ray 集群提交，`RAY_ADDRESS` 使用 Ray Jobs 的 HTTP 地址。脚本不负责建立或清理共享集群。默认 8 卡、训练 TP2、SGLang 每引擎 1 卡；SFT batch80、16K；GRPO prompt2048、response8192（包含反馈）、context10240，最多 16 个 assistant turn。

```bash
export RAY_ADDRESS=http://ray-head:8265
bash scripts/entrypoint/ray-job.sh examples/retool/scripts/run-qwen25-7B-sft-8xgpu.sh

# SFT 完成后，从导出的同策略 HF 权重建立新的 GRPO 实验。
export MODEL_DIR="$SAVE_DIR/qwen25-7b-retool-sft-8xgpu/hf_49"
export SANDBOX_FUSION_URL=http://sandbox-host:8080/run_code
bash scripts/entrypoint/ray-job.sh examples/retool/scripts/run-qwen25-7B-grpo-8xgpu.sh
```

SandboxFusion URL 必须能从 rollout workers 访问。SGLang 使用内置禁词 processor，Megatron 用已有 logprob/train hooks 屏蔽相同词表尾部，保持该分支 token\<151665 的策略分布。两端采用不同计算内核，logprob 不保证逐位一致。

实验默认名由脚本给出，可用 `EXP_NAME`、`PROJECT_NAME` 覆盖。训练恢复使用稳定的 `$SAVE_DIR/$EXP_NAME`，保存 model/optimizer/scheduler/RNG 和数据游标；SFT 保留最近 2 个、GRPO 最近 8 个 checkpoint，每 25 步保存。重新运行同一脚本即恢复，总步数仍是原目标。不要给完整恢复追加 `--no-load-optim`、`--no-load-rng` 或 `--finetune`。启动前会拒绝缺少原生状态或数据游标的恢复目录；加载时再次核对采样 seed、分组和数据摘要。新实验须显式更换名称或保存根目录。

ClearML 和 metrics service 默认启用，使用现有凭证；W&B 未启用。Ray 提交默认等待结束，可显式设置 `RAY_NO_WAIT=1` 改为非阻塞提交。脚本末尾接受 CLI 覆盖，参数仍须满足模型架构、分组及资源约束。

## 周期评测

把原始 benchmark 记录转换为相同 prompt/label/metadata 格式：

```bash
python -m examples.retool.prepare_data eval --benchmark math500 \
  --input /path/to/math500.json --model "$MODEL_DIR" --out "$DATA_DIR/eval/math500.jsonl"
export EVAL_CONFIG="$PWD/examples/retool/eval.yaml"
bash scripts/entrypoint/ray-job.sh examples/retool/scripts/run-qwen25-7B-grpo-8xgpu.sh
```

转换器支持 JSON 数组或 JSONL；GSM8K 使用 question/answer，GSM-Hard 使用 input/target，其余使用 problem/answer。模板列出五套数据各开/关工具的十个面板，需先准备对应文件，或复制模板仅保留需要的数据集。评测 K1/greedy，使用相同 token 预算；`tool_enabled=false` 更换为无工具 prompt 且不构造执行客户端。指标读取独立 `acc`，不把工具惩罚当准确率。数据重合分析和污染控制由实验的数据选择负责。

开启后每 25 步用当前 rollout 引擎在线评测，不退出训练或重新恢复 Adam。原生 eval 的部分异常仅记录 warning；本例不提供“全部面板验收通过才准许续训”的额外调度器。未设置 EVAL_CONFIG 时不运行周期评测。

协议/奖励/采样恢复的 CPU 回归可运行：

```bash
python -m pytest tests/examples/test_retool.py -q
```
