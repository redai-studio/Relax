# Agentic OPD - Search-QA

## 1. 环境

检索服务和 agent 客户端共用一个独立 conda 环境（与训练环境解耦，agent 进程通过
`run_agent_app.sh` 里的 `conda activate` 使用它）。机器初始没有 conda，先装 miniconda：

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /root/miniconda3
source /root/miniconda3/etc/profile.d/conda.sh
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

再建 Search-QA 环境：

```bash
conda create -n relax-opd-search python==3.10 -y
conda activate relax-opd-search
conda install -y -c conda-forge faiss-gpu
pip install "torch==2.6.0+cu124" --extra-index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.version.cuda, torch.cuda.is_available())"
pip install transformers datasets uvicorn fastapi pydantic tqdm
pip install "openai>=1,<2" httpx requests pyarrow pandas huggingface_hub
```

> 驱动 CUDA 上限不是 12.4 时，把 `cu124` 换成对应档（如 12.1 用 `cu121`）。

## 2. 下载 index + 语料

wiki-18 的 E5 稠密向量索引较大，峰值磁盘约 130GB，放本地 NVMe。

```bash
conda activate relax-opd-search
export SEARCH_CORPUS=/root/search_qa_corpus
python examples/on_policy_distillation/agentic_opd/search_qa/retriever/searchr1_download.py \
  --save_path "$SEARCH_CORPUS"
cat "$SEARCH_CORPUS/part_aa" "$SEARCH_CORPUS/part_ab" > "$SEARCH_CORPUS/e5_Flat.index"
rm "$SEARCH_CORPUS/part_aa" "$SEARCH_CORPUS/part_ab"
gunzip "$SEARCH_CORPUS/wiki-18.jsonl.gz"
```

## 3. 起检索服务

检索服务是外部共享服务，Relax 只是它的 HTTP 客户端。**起一次、out of band**，别占训练的
8000 端口（Relax 的 agentic API 用 8000），示例用 8001：

```bash
conda activate relax-opd-search
python examples/on_policy_distillation/agentic_opd/search_qa/retriever/retrieval_server.py \
  --index_path "$SEARCH_CORPUS/e5_Flat.index" \
  --corpus_path "$SEARCH_CORPUS/wiki-18.jsonl" \
  --port 8001 \
  --topk 3 --retriever_name e5 --retriever_model intfloat/e5-base-v2 --faiss_gpu \
  > /root/retrieval_server.log 2>&1 &

# 冷启动要几分钟加载 index + 语料 + e5 模型；就绪判据不是看日志，而是这条 curl
# 能返回 JSON（返回空/连接拒绝 = 还在加载，等一会儿再试）：
curl -s --noproxy '*' http://127.0.0.1:8001/retrieve -H 'Content-Type: application/json' \
  -d '{"queries":["who wrote hamlet"],"topk":3,"return_scores":true}' | head -c 500
```

> 机器设了 `http_proxy`/`https_proxy` 时，把检索服务地址加进 `no_proxy`，否则 agent→检索请求会被代理拦：
> `export no_proxy="127.0.0.1,localhost,<检索服务IP>,${no_proxy}"`

> 检索服务和训练**共用同一 8 卡**时，`--faiss_gpu` 会和 SGLang/teacher 抢显存。要么去掉
> `--faiss_gpu` 走 CPU，要么用 `retriever/start_after_relax_model_load.sh`——等 Relax 模型
> 加载完显存峰值过去再拉起检索。

## 4. 准备数据

从 HuggingFace `PeterJinGo/nq_hotpotqa_train`（NQ + HotpotQA）抽 `(question, golden_answers)`
写 parquet。`prepare_data.py` 是 Relax 仓库里的模块，须先 `cd` 到 Relax 根目录再跑：

```bash
conda activate relax-opd-search
cd /path/to/your/Relax
python examples/on_policy_distillation/agentic_opd/search_qa/prepare_data.py \
  --output-dir /root/search_qa-relax-full
```

会额外按原始 data_source 切出 NQ/TriviaQA/PopQA/HotpotQA/2Wiki/MuSiQue/Bamboogle 七份
`test_*.parquet`，训练脚本逐数据集报 EM。eval 默认只取 512 条（`--eval-size`，round-robin
让七个源都有代表，一次 eval 几分钟；传 `--eval-size 0` 才用全量测试集——共 6 万+ 条、单次
eval 要几小时）。冒烟测试还可 `--train-size 2048` 限制训练集。

## 5. grpo

训练用 Relax 主环境，先 `conda deactivate` 退出 `relax-opd-search`（它只给节点上的
`run_agent_app.sh` 子进程用）。`SEARCH_RETRIEVAL_URL` 经 `--agent-env` 广播到各节点；
服务在别的节点就填那台的 IP。

```bash
conda deactivate
cd /path/to/your/Relax
SEARCH_RETRIEVAL_URL="http://127.0.0.1:8001/retrieve" \
MODEL_DIR=/path/to/base_models DATA_DIR=/root/search_qa-relax-full EXP_DIR=/path/to/exp \
bash examples/on_policy_distillation/agentic_opd/search_qa/run-search_qa-grpo-qwen3-1.7B-8xgpu.sh
```

> student base 权重放在 `${MODEL_DIR}/Qwen3-1.7B/`。

## 6. opd

On-Policy Distillation：student = Qwen3-1.7B，teacher = 一个**训练好的** 1.7B search
checkpoint（HF 格式，如第 5 步 grpo 产出的转换结果）；从 1.7B base 蒸自己没意义，`TEACHER_MODEL_PATH`
必须指向更强的 checkpoint。

```bash
conda deactivate
cd /path/to/your/Relax
SEARCH_RETRIEVAL_URL="http://127.0.0.1:8001/retrieve" \
MODEL_DIR=/path/to/base_models DATA_DIR=/root/search_qa-relax-full EXP_DIR=/path/to/exp \
TEACHER_MODEL_PATH=/path/to/search-grpo-teacher-hf \
bash examples/on_policy_distillation/agentic_opd/search_qa/run-search_qa-opd-qwen3-1.7B-8xgpu.sh
```
