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
and returns a scalar mean loss. The built-in InfoNCE needs at least two pairs
in the global batch. Custom losses can use a single pair.

If every rank evaluates the global loss with only its own features attached to
autograd, set `requires_dp_gradient_sum = True` on the loss callable. Relax then
sums the rank-local gradient contributions through DDP. Callbacks that return
a rank-local mean keep the default gradient averaging.

## AllNLI training comparison

Relax and the [Sentence Transformers MNRL baseline](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss)
each ran 80 updates on two H800 GPUs from the same Qwen3.5-4B base. Both used
2,560 AllNLI training pairs, a global batch of 32 pairs, BF16 base weights,
LoRA rank 8 and alpha 16, a constant learning rate of `1e-4`, last-token
pooling, and one-way InfoNCE with temperature `0.05` and cross-rank negatives.
The baseline used Sentence Transformers 5.3.0 with Transformers and PEFT.
Each framework used its own adapter initialization and optimizer implementation.

Both runs used the AllNLI revision pinned above. The training subset takes 2,560
pairs after streaming shuffle with seed 42 and a 10,000-row buffer, skipping
pairs with a repeated sentence. Evaluation uses the first 1,024 records after
shuffling the `triplet` dev split with seed 43 and the same buffer size. The
same evaluator measures how often the positive has higher cosine similarity
to the anchor than the negative.

| Framework             | Step 0 | Step 20 | Step 40 | Step 60 | Step 80 |
| --------------------- | -----: | ------: | ------: | ------: | ------: |
| Relax                 | 65.43% |  81.15% |  84.18% |  86.04% |  87.30% |
| Sentence Transformers | 65.43% |  80.57% |  84.47% |  85.55% |  87.11% |
