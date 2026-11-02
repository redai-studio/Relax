# DeepCoder 0/1 baseline

Qwen3.5-4B 的代码生成 GRPO recipe。训练样本来自固定版本的 DeepCoder 训练集，SandboxFusion 只对最终 Python 代码判题，返回原始 0/1 reward。

## 准备

先按照 Relax 的安装说明准备训练环境、Qwen3.5-4B 权重和 Ray 集群。

判题服务采用 [SandboxFusion](https://github.com/bytedance/SandboxFusion/tree/469de0845cb7d665bc68044d541aa3aa28c5b30a)。在专用执行环境安装服务与 Python runtime，运行后向训练节点提供 HTTP 地址；该 recipe 调用服务，不负责创建服务的隔离边界。

```bash
git clone https://github.com/bytedance/SandboxFusion.git
cd SandboxFusion
git checkout 469de0845cb7d665bc68044d541aa3aa28c5b30a
conda create -n sandbox -y python=3.12
conda activate sandbox
uv pip install --python "$(which python)" . numpy
mkdir -p docs/build
(cd runtime/python && bash install-python-runtime.sh)
make run HOST=0.0.0.0 PORT=8080
```

从 Relax 根目录准备数据。脚本固定 [DeepCoder 数据版本](https://huggingface.co/datasets/agentica-org/DeepCoder-Preview-Dataset/tree/177913a7bd43791646ef6a43645caa3c871ab3db)，仅下载 PrimeIntellect、TACO 和 LCBv5 的训练分片，共 24,287 题；保留题目和 starter code，将测试转换为 SandboxFusion 的 provided-data 格式。

```bash
export DATA_DIR=/path/to/deepcoder-train
python examples/code_generate/prepare_data.py --output-dir "${DATA_DIR}"
```

## 启动

```bash
export MODEL_DIR=/path/to/models  # 包含 Qwen3.5-4B 子目录
export SAVE_DIR=/path/to/checkpoints
export SANDBOX_URL=http://sandbox-host:8080
RAY_NO_WAIT=1 bash scripts/entrypoint/ray-job.sh examples/code_generate/baseline.sh
```

| 配置            | 值                                                                   |
| --------------- | -------------------------------------------------------------------- |
| Actor           | 8 GPU；TP2 / CP1 / PP1 / DP4                                         |
| Rollout         | 同 8 GPU 分时复用；4 个 TP2 引擎                                     |
| Batch           | 32 题 × 8 个候选 = 256                                               |
| 输入 / 回复上限 | 2,048 / 16,384 tokens                                                |
| 采样            | thinking 开启，temperature 1，seed 42                                |
| 优化            | Adam，lr 1e-6，clip 0.2 / 0.28，KL 与 entropy 系数为 0               |
| 优势            | GRPO；组内减均值和 std 归一化开启；额外 advantage normalization 关闭 |
| 显存清理        | `--empty-unused-memory-level 1`                                      |

`app/reward.py` 从闭合的 `</think>` 后提取最后一个非空代码块；格式缺失记 0，不修改训练 tokens。调用 SandboxFusion `live_code_bench_v1` 的 `/submit`，Python 每次测试超时 6 秒，HTTP 超时 180 秒，最多并发 16 个判题请求。通过记 1，否则记 0；`SandboxError` 等基础设施故障报错，不当作解题失败。

## 已记录的训练曲线

训练曲线见 [PR #444 的 Screenshots / Logs](https://github.com/redai-studio/Relax/pull/444)。

图来自同一次历史 baseline 的指标导出，快照时间为 **2026-10-16 04:57:56 UTC**。导出的 238 个 reward、grad norm、entropy、logprob-difference 点覆盖 iteration 0–237；step time 有 237 个点，覆盖 0–236。横轴保留指标系统的 iteration，不额外推断为独立评测轮次。

浅线为未平滑、未降采样的原始数据；深线为包含当前点的至多 20 点移动平均。`rollout/raw_reward` 是当前训练采样的通过率，**不是独立测试集准确率**。前 20 点平均通过率 0.3064，最后 20 点为 0.6334；题目随训练变化，这两个窗口只描述该次运行。其他面板展示优化统计与 step 耗时，不据此声称算法对比收益。

上述历史记录使用 TP2 / CP2 / DP2、std 开启配置。当前 recipe 已将默认 CP 改为 1（DP4）；其余训练参数不变。本 recipe 是纯 0/1 baseline，没有质量 grader 或 group reward 重分配。

## CPU 回归

```bash
pytest -q tests/examples/test_code_generate.py
bash -n examples/code_generate/baseline.sh
```
