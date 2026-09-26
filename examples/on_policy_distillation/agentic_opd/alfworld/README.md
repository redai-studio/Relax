# Agentic OPD - ALFWorld

## 1. 环境

ALFWorld 依赖单独的 conda 环境（与训练环境解耦，agent 进程通过 `run_agent_app.sh`
里的 `conda activate` 使用它）：

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /root/miniconda3
source /root/miniconda3/etc/profile.d/conda.sh
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

```bash
conda create -n relax-opd-alfworld python==3.12
conda activate relax-opd-alfworld
pip3 install gymnasium==0.29.1
pip3 install stable-baselines3==2.6.0
pip3 install alfworld
pip3 install pyarrow pyyaml openai httpx   
```

## 2. 下载 & 准备数据

数据（ALFWorld 语料 + parquet）放**本地盘**，避免 JFS/网络盘的慢 IO。脚本默认：
`ALFWORLD_DATA=/root/alfworld`（语料）、`DATA_DIR=/root/alfworld-relax`（parquet）；

```bash
conda activate relax-opd-alfworld
export ALFWORLD_DATA=/root/alfworld     # 语料下载 & 读取路径（与训练脚本默认一致）
alfworld-download -f                    # 下到 $ALFWORLD_DATA
# --output-dir 与训练脚本 DATA_DIR 默认一致（/root/alfworld-relax）
cd /path/to/your/Relax
python examples/on_policy_distillation/agentic_opd/alfworld/prepare_data.py \
    --output-dir /root/alfworld-relax \
    --eval-dataset eval_out_of_distribution \
    --train-size 2048 \
    --eval-size -1
```

产出 `/root/alfworld-relax/{train,test}.parquet`。改路径时，训练用同名
`ALFWORLD_DATA` / `DATA_DIR` 覆盖即可保持三处（下载 / 准备 / 训练）一致。

## 3. grpo

```bash
# 模型放在 $EXP_DIR/Qwen3.5-35B-A3B/；数据 & 语料默认在本地 /root 下。
conda deactivate 
hf download Qwen/Qwen3.5-35B-A3B --local-dir ${EXP_DIR}/Qwen3.5-35B-A3B
EXP_DIR=/path/to/your/exp_dir bash examples/on_policy_distillation/agentic_opd/alfworld/run-alfworld-grpo-qwen35-35B-A3B-8xgpu.sh
```

## 4. opd

```bash
conda deactivate 
hf download Qwen/Qwen3.5-35B-A3B --local-dir ${EXP_DIR}/Qwen3.5-35B-A3B
EXP_DIR=/path/to/your/exp_dir bash examples/on_policy_distillation/agentic_opd/alfworld/run-alfworld-opd-qwen35-35B-A3B-8xgpu-sampled-kl.sh
```
