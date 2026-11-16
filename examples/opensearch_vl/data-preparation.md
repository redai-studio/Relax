# 数据准备

使用 `OpenSearch-VL/Search-VL-SFT-36K` 的固定 revision `2c1c460af4fa15bd63210cbf426a96664b959944`。七域为 `fvqa`、`livevqa`、`palace`、`webqa`、`wiki_art`、`wiki_en`、`wiki_zh`。原始数据约 11.38 GB，不含模型。

## 输入与处理

首次准备可在下面的命令中加 `--download`，脚本会从上述固定版本下载七域 JSON 和图片 ZIP，并写入 `source-revision.json`；不需要手动生成该文件。已有原始数据时直接使用下面的处理命令。

`SOURCE_DIR` 中应有 `source-revision.json` 和七个域目录，每个域包含原始 JSON、`images.zip`。若原始 tar 带 `SHA256SUMS`，先在源目录执行 `sha256sum --check SHA256SUMS`。

```bash
python3 examples/opensearch_vl/tools/prepare_data.py \
  --source "${SOURCE_DIR}" \
  --output "${DATA_DIR}/opensearch_vl/32k" \
  --workers 16
```

输出目录必须是新的；脚本拒绝覆盖已准备的数据。处理不需要模型，不运行 tokenizer，也不按 32K 长度过滤。`32k` 是配方目录名；实际长度策略在训练阶段通过 `--sft-oversize-strategy skip` 执行。

保留 role、工具调用、最终答案格式、图片引用及 ChatML 注入标记检查。按问题和原图像素关联组切分，关联轨迹不会分到不同集合。默认保留全部合格训练轨迹，固定验证/测试各 32 条；此数据版本得到 34,607 / 32 / 32 条。

图片由每个 worker 直接读取 ZIP，避免将数万张原图先解到共享盘；也支持仅有已解压图片的输入。RGB 图按原始像素 hash 去重，输出为 32 像素网格、最多 262,144 像素的 PNG。原始大小和 RGB 像素 hash 写入每行 metadata，供训练去重与评测重叠检查使用。

输出包括 `train.jsonl`、`val.jsonl`、`test.jsonl`、`images/`、`rejected.json` 和 `manifest.json`。manifest 记录源 revision、样本 ID、计数、worker 数和处理时间；图片路径指向本次输出目录。移动数据时应同时处理这些路径，不能只复制 JSONL。

## 训练前审计

```bash
python3 examples/opensearch_vl/tools/audit_data.py \
  --model "${HF_CHECKPOINT}" \
  --data-dir "${DATA_DIR}/opensearch_vl/32k" \
  --max-samples-per-split 8 \
  --output "${SAVE_DIR}/data-audit.json"
```

审计使用真实 SFT Processor 检查图片 token 展开、assistant loss mask 和 thinking 监督。训练入口会执行该检查；JSON 可解析或图片文件存在不能替代它。

## 评测去重索引

从实际训练切分生成索引，不使用全部原始数据或验证/测试切分代替：

```bash
PYTHONPATH=examples/opensearch_vl/tools:. python3 - "${DATA_DIR}/opensearch_vl/32k/train.jsonl" /path/to/train-overlap-index.json <<'PY'
import json
import sys
from pathlib import Path
from benchmark import question_hash
questions, images = set(), set()
for line in Path(sys.argv[1]).open():
    row = json.loads(line)
    questions.add(question_hash(row["messages"][1]["content"]))
    images.update(row["metadata"]["image_hashes"])
Path(sys.argv[2]).write_text(json.dumps({"questions": sorted(questions), "images": sorted(images)}))
PY
```

该操作只读取训练记录，不重复处理图片。评测问题或任一原图 hash 命中时，记录为与 SFT train 重叠；完整分数和去重分数都保留。
