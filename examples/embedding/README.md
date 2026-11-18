# Embedding SFT with Qwen3.5-4B

This example trains a text-pair encoder on two data-parallel GPUs using
`--loss-type sft --task-type embedding`. It uses the standard Qwen3.5-4B
Hugging Face checkpoint and pools the last real token from each sequence.

Start a Ray cluster with two available GPUs and set `RAY_ADDRESS` to its Jobs
HTTP address. Set `MODEL_DIR` to the parent directory of `Qwen3.5-4B`, `DATA_DIR`
to the input directory, and `SAVE_DIR` to a persistent checkpoint root.

Prepare eight sentence pairs from the `pair` training split of
[AllNLI](https://huggingface.co/datasets/sentence-transformers/all-nli).
Run the download with Hugging Face Datasets installed and make `DATA_DIR`
accessible to the Ray workers.

```bash
mkdir -p "$DATA_DIR"
python3 - "$DATA_DIR/pairs.jsonl" <<'PY'
import json
import sys
from pathlib import Path

from datasets import load_dataset

pairs = load_dataset(
    "sentence-transformers/all-nli",
    "pair",
    revision="d482672c8e74ce18da116f430137434ba2e52fab",
    split="train",
    streaming=True,
).take(8)
lines = [json.dumps({"query": row["anchor"], "positive": row["positive"]}) for row in pairs]
Path(sys.argv[1]).write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

bash examples/embedding/scripts/run-qwen35-4B-2xgpu.sh
```

The default run makes two optimizer updates with four pairs per update. Use a
whole number of batches in the JSONL file. Set `NUM_UPDATES` to change the
training length. Use the same `SAVE_DIR` and `EXP_NAME` to resume from a saved
native checkpoint. Set a new `EXP_NAME` for a fresh run.

The framework keeps each pair together through data-parallel assignment and
uses the positives from the full data-parallel batch as InfoNCE candidates.
Set `--max-tokens-per-gpu` to fit the whole local pair batch in one microbatch.
The supported setup is Megatron-Bridge with TP=PP=CP=1.

Custom datasets can supply `input_ids` and an optional zero-based
`embedding_position` per branch. A custom loss can be passed through
`--custom-loss-function-path`. It receives the local query and positive
tensors along with the keyword arguments `temperature` and `process_group`,
and returns a scalar mean loss.
