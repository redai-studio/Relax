# 模型评测：V\*Bench本地裁剪工具

[汇总](README.md) · [公开来源与协议差异](benchmark-references.md)

对每个模型分别评测基座与SFT最终权重。模型自主调用本地`crop`，CPU返回真实裁剪图，模型继续观察和作答；只使用V\*Bench，不需要搜索、图片托管、OCR或模型裁判服务。训练轨迹中的验证loss/PPL另行记录。

## 固定协议

- 官方`craigwu/vstar_bench`共191题，revision为`d9ae62c903da0c98336e85c5ee89cd863b04b4da`，题目及对应图片约281 MB。
- [LMMs-Lab 的 Parquet 版](https://huggingface.co/datasets/lmms-lab-encoder/vstar-bench/tree/b44023b4dca749ed8a76b85eb576627d05a1c174) 的 191 行记录与上述版本逐行核对一致，包括题号、顺序、题文、选项、类别及答案。两份图片未逐张核验。下方命令读取官方 JSONL 和对应原图。
- 使用`test_questions.jsonl`自带的选项及标签。参考标签只参与评分，不发送给模型，也不读取原始标注中的目标bbox作为工具参数。
- 温度0、输入输出合计32K、最多10轮，每张显示图片最多1,024视觉token。两组保留完整历史，共用提示、工具与图像处理预算。
- 工具为`crop(image, x, y, width, height)`；坐标采用当前显示图的整数像素，映射到该图对应的原始分辨率区域裁剪，再缩放显示。新图引用依次为`img_2`、`img_3`等。
- 最终答案要求`<response>A</response>`。v2评分读取`</think>`之后的唯一`<response>`内容或最终输出，提取末行明确选项；支持`(A) white`、`A. white`、`Answer: (A)`、Markdown粗体和`\boxed{A}`。仅列多个选项、末行有多个不同候选、未闭合或多个`<response>`均拒绝；不从分析段猜字母，不按解释改写模型所选字母，不调用裁判。每题生成一条轨迹，不以重试挑选更好答案。
- 超限、无有效选项、工具格式错误及轮数耗尽均计入191题分母。服务、文件或token计数异常必须排查，不作为正式模型成绩。另报与实际SFT train去重后的分数。

## 准备数据与配置

沿用当前环境中的Pillow、requests、Transformers、OpenAI SDK和httpx。下载器只取191题实际引用的图片，校验固定revision中的对象及本地清单，完整缓存可直接复用：

```bash
python3 examples/opensearch_vl/tools/vstar_benchmark.py \
  --data "${DATA_DIR}/benchmark/vstar" --prepare-only
cp examples/opensearch_vl/configs/benchmark.env.example /path/to/benchmark.env
# 填入本次实际路径和服务配置后加载：
source /path/to/benchmark.env
```

`BENCHMARK_OVERLAP_INDEX`指向从实际训练切分生成的问题/原图像素hash索引，生成方式见[数据准备](data-preparation.md#评测去重索引)。不得用全量原始数据或其他实验的训练切分代替。

`AGENT_PROVENANCE_FILE`为仓库外的模型来源记录。基座`kind=base`，SFT为`kind=trained-final`；记录完整训练分支/commit、实际训练量、`served_model_name`和可信文件hash。对照可信清单检查配置、权重索引及所有分片后，才可标记`hashes_verified=true`。服务名须与真实服务相同，不以目录存在代替校验。

评测中期模型时，向评测器传入 `--label relax --checkpoint-step N`，来源文件须记录 `kind=trained-checkpoint` 和整数 `completed_updates=N`。将结果写入与 `base/vstar` 同级的 `checkpoint-stepN/vstar`，再向比较器传入 `--checkpoint-step N` 即可复用同一份 Base 评分。默认比较器继续读取 `relax/vstar` 并核实 `kind=trained-final`。如另传 `--training-overlap-index`，评测器还会依据该权重训练步数的样本索引保存独立去重审计。三组用于配对评分的 `--overlap-index` 必须相同。

快速对照可给评测器和比较器传入同一份 `--case-manifest`。清单指定数据版本、官方题目文件 SHA-256 和互不重复的原始题号。评测记录清单 SHA-256，比较器逐条核验两组实际作答题号。复用已在加载的服务时，也可在该次输出目录的父目录预置 `vstar-run-plan.json`，用 `case_manifest` 指定清单的绝对路径。子集分数须标明分母，正式验收保留全量 191 题。

## 部署模型服务

设置本组模型的显式 HF 目录，基座与 SFT 分别启动。下面的命令用于 Qwen：Qwen3-VL-8B-Instruct 使用原生模板，Qwen3.5/3.6 保留完整 thinking 历史。

```bash
export HF_CHECKPOINT=/path/to/base-or-final-hf
export BENCHMARK_CHAT_TEMPLATE="${SAVE_DIR}/${EXP_NAME}/benchmark-chat-template.jinja"
PYTHONPATH=examples/opensearch_vl/tools:. python3 - <<'PYTEMPLATE'
import os
from pathlib import Path
from transformers import AutoProcessor
from benchmark import preserve_history_template
processor = AutoProcessor.from_pretrained(os.environ["HF_CHECKPOINT"], local_files_only=True)
path = Path(os.environ["BENCHMARK_CHAT_TEMPLATE"])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(preserve_history_template(processor))
PYTEMPLATE
python3 examples/opensearch_vl/tools/serve_sglang.py \
  --model-path "${HF_CHECKPOINT}" --served-model-name benchmark-model \
  --host 127.0.0.1 --port 30000 --tp-size 1 --context-length 32768 \
  --chat-template "${BENCHMARK_CHAT_TEMPLATE}"
```

DeepSeek-V4.1-Flash-1017 使用 BF16 基座或从训练 DCP 导出的独立 HF 目录。评测客户端读取目录内的 `encoding/encoding.py`，向 SGLang 的 `/generate` 提交原生 token ID 和每一轮图片。四层模型在单机 8×H800 上使用以下服务配置；共享 Engram 表约需要 95 GiB 主机内存。

```bash
export HF_CHECKPOINT=/path/to/deepseek-1017-4layer-base-or-trained-hf
SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1 SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=shared \
SGLANG_DSV4_FP4_EXPERTS=0 SGLANG_OPT_FP8_WO_A_GEMM=0 \
SGLANG_OPT_DEEPGEMM_HC_PRENORM=0 SGLANG_OPT_FUSE_WQA_WKV=0 \
python -m sglang.launch_server \
  --model-path "$HF_CHECKPOINT" --served-model-name benchmark-model \
  --tp-size 8 --ep-size 8 --dtype bfloat16 --load-format auto \
  --attention-backend dsv4 --context-length 32768 --image-processor-backend pil \
  --mem-fraction-static 0.60 --max-running-requests 2 \
  --disable-cuda-graph --disable-radix-cache --trust-remote-code \
  --host 127.0.0.1 --port 30000
```

按实际空卡和显存选择 TP/DP。评测请求已有模型服务，保留原始 `<tool_call>` 文本。客户端核对包含全部历史图片的实际输入 token 数，并在 32K 预算内保留完整历史。

## 运行与比较

```bash
export AGENT_BASE_URL=http://127.0.0.1:30000/v1
export AGENT_MODEL=benchmark-model
export AGENT_PROVENANCE_FILE=/path/to/verified-base-provenance.json
export BENCHMARK_OVERLAP_INDEX=/path/to/train-overlap-index.json
export BENCHMARK_DATA_ROOT="${DATA_DIR}/benchmark"
export BENCHMARK_WORKERS=16
SCRIPT=examples/opensearch_vl/scripts/run-qwen3-vl-8B-8xgpu.sh
bash "$SCRIPT" benchmark vstar base
```

换成SFT最终权重，更新来源记录和共用模板，重启对应服务，再执行：

```bash
export AGENT_PROVENANCE_FILE=/path/to/verified-sft-provenance.json
bash "$SCRIPT" benchmark vstar relax
bash "$SCRIPT" compare
```

其他模型换用各自脚本。先用`BENCHMARK_MAX_CASES=2`检查真实裁剪、返回图片与后续推理；这是冒烟测试，不填完整分数。正式运行取消限题设置。工具可用不意味着每题一定调用，不能强制每题调用后将结果当作自主选择工具的协议。

## 结果与记录

- 逐题：`${SAVE_DIR}/${EXP_NAME}/benchmark/<base|relax>/vstar/trajectories.jsonl`，包含原始输出、真实调用参数、执行状态、最终选项与用量。
- 实际显示图：同目录`cases/<id>/`；汇总为`summary.json`，记录题数、分数、失败原因、工具调用率与成功数、去重分数。
- 来源：同目录`protocol.json`、`model-provenance.json`和`evaluator.json`，记录数据/提示/模板/源码hash、模型来源及评测commit。
- 对照：`${SAVE_DIR}/${EXP_NAME}/benchmark-comparison.json`。比较器只接受同协议的完整191题；基础设施错误和限题测试不能混入。

JSON、逐题输出、日志和权重全部留在仓库外。每次评测在`results/<model>.md`前方插入一段，完成后填入实际分数并更新总表。修改评分提取时保留原始答案，不重新生成后挑选成绩。

若评测源码以归档形式交付、没有 `.git`，运行前设置 `RELAX_GIT_COMMIT` 为该源码的完整 commit，作为 `evaluator.json` 的版本来源；未设置时从当前 Git checkout 读取。

评测与评分均使用仓库中的[vstar_benchmark.py](tools/vstar_benchmark.py)。结果文档只记录一个评测代码commit，以该版本的脚本和本页协议复现。若修正答案提取，对所有已完成的基座与SFT统一重评分，原始答案及逐题差异保留在仓库外。
