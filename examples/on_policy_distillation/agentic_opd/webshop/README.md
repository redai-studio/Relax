# WebShop 环境搭建（Relax Agentic OPD）

WebShop 装在独立 conda 环境（Python 3.10）。内网连不上 Google Drive / spaCy 下载源，
故用 **HuggingFace 数据集**替代 gdown、**直连 wheel** 装 spaCy 模型。
默认跑 **1000 商品子集**（对齐 SDAR，`DEFAULT_FILE_PATH=items_shuffle_1000.json`）。

```bash
# -1. conda（没装过才需要；同机只需一次）
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /root/miniconda3
source /root/miniconda3/etc/profile.d/conda.sh
# 新版 conda 首次用默认 channel 前要先接受 ToS，否则 conda create 会中止（同机只需一次）
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# 0. 源码 + 环境
export EXP_DIR=/root
git clone https://github.com/princeton-nlp/WebShop.git ${EXP_DIR}/WebShop
export WEBSHOP_HOME=${EXP_DIR}/WebShop
conda create -n relax-opd-webshop python==3.10 -y
conda activate relax-opd-webshop

# 1. 依赖（Python + faiss + JDK）
cd "$WEBSHOP_HOME"
conda install -y -c conda-forge openjdk=11 faiss-cpu mkl
grep -v '^gradio' requirements.txt > /tmp/webshop-req.txt
pip install -r /tmp/webshop-req.txt
pip install "spacy==3.7.2" "werkzeug==2.0.3"
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl
pip install "spacy==3.7.2" "pydantic==1.10.13"

# 2. 数据（HF 替代 gdown，1000 子集只需 3 个文件）
mkdir -p "$WEBSHOP_HOME/data"
# 若 huggingface.co 不通： export HF_ENDPOINT=https://hf-mirror.com
hf download YWZBrandon/webshop-data --repo-type dataset --local-dir "$WEBSHOP_HOME/data" \
  --include "items_shuffle_1000.json" "items_ins_v2_1000.json" "items_human_ins.json"

# 4. 建 Lucene 索引
cd "$WEBSHOP_HOME/search_engine"
mkdir -p resources resources_100 resources_1k resources_100k indexes
python convert_product_file_format.py
./run_indexing.sh
```

冒烟验证：

```bash
cd "$WEBSHOP_HOME"
python -c "
import gym
from web_agent_site.envs import WebAgentTextEnv
env = gym.make('WebAgentTextEnv-v0', observation_mode='text', num_products=None)
obs, info = env.reset(session=0)
print(obs[:500])
obs, reward, done, info = env.step('search[red shirt]')
print('reward', reward, 'done', done)
"
```

> 全量（可选）：额外 `--include "items_shuffle.json" "items_ins_v2.json"`，并把
> `web_agent_site/utils.py` 的 `DEFAULT_FILE_PATH`/`DEFAULT_ATTR_PATH` 指向全量文件后重建索引。

## 进训练（训练脚本管理 server）

架构：每次实验启动一个 cluster-shared `SimServer`（几 GB 商品库常驻内存），每个 rollout session 起一个
瘦客户端 agent 进程。训练脚本在提交 Ray Job 前启动 server，等待 `/health` 返回 ready，把集群可达的
`WEBSHOP_URL` 传给所有节点，并在训练命令退出时回收 server。`run_agent_app.sh` 始终作为瘦客户端运行。

先把 WebShop goal 下标枚举成 parquet（在上面这个 conda 环境里跑，需要 `$WEBSHOP_HOME`；
`prepare_data.py` 是 Relax 仓库里的模块，须先 `cd` 到 Relax 根目录再跑，否则找不到 `examples` 包）：

```bash
conda activate relax-opd-webshop
pip install fastparquet pyarrow
cd Relax
python examples/on_policy_distillation/agentic_opd/webshop/prepare_data.py \
  --output-dir /root/webshop-relax --webshop-home "$WEBSHOP_HOME"
# 已知 goal 总数可加 --num-goals N 跳过加载商品库（快）。
```

再跑训练。`WEBSHOP_HOME`、`WEBSHOP_PORT` 和 `WEBSHOP_CONDA_ENV` 用于训练脚本启动本机 server；
`WEBSHOP_CONDA_ENV` 指向 conda 环境目录。Agent 进程使用 Relax 主环境中的 `openai` 和 `httpx`，并通过
`WEBSHOP_URL` 访问 server；下面的 import 检查用于确认训练环境具备客户端依赖：

> 训练前先 `conda deactivate` 退出 `relax-opd-webshop`。训练脚本用的是驱动进程自身的 Python
> 环境（Relax 主环境），`relax-opd-webshop` 供 WebShop server 使用。两者依赖（如 pydantic 版本）
> 会冲突，留在 webshop 环境里直接跑训练脚本可能拿错解释器/依赖。

```bash
conda deactivate
cd Relax
python -c "import openai, httpx"

WEBSHOP_HOME="$WEBSHOP_HOME" WEBSHOP_CONDA_ENV=/root/miniconda3/envs/relax-opd-webshop EXP_DIR=/path/to/exp \
bash examples/on_policy_distillation/agentic_opd/webshop/run-webshop-grpo-qwen35-35B-A3B-8xgpu.sh
# OPSD（loss 模式自蒸馏）版：run-webshop-opsd-qwen3-1_7B-8xgpu.sh
```

- 端口：`WEBSHOP_PORT`（默认 36001）；交互轮数：`WEBSHOP_MAX_TURNS`（默认 15，对齐 SDAR）。
- server 侧 num_products / 语料路径 / step 并发上限在 [`app/config.yaml`](app/config.yaml) 调。
- server 日志写入 `log/<experiment>-webshop-server.log`。
- `WEBSHOP_HOST` 默认 `0.0.0.0`；`WEBSHOP_ADVERTISE_HOST` 默认取训练脚本所在 Ray 节点的 IP，也可以显式指定。
