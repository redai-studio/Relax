# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Measure the irreducible train/rollout mismatch floor of an MXFP4 release.

Same SGLang engine version, same token ids, two checkpoints:

- A) the native MXFP4 release (rollout-side weights in RL)
- B) its BF16 dequantized copy (train-side weight equivalent)

Per-token logprob differences between A and B isolate the pure quantization
effect from Megatron-vs-SGLang numerics and from the online push path. Compare
against the in-training ``train/mismatch_k3_kl`` / ``train_rollout_*`` metrics:
the in-training mismatch is healthy iff it is at or below this floor.

Usage (on a GPU node with the Relax training image):

    python3 examples/models/kimi-k3/tools/probe_mxfp4_mismatch_floor.py \
        --mxfp4-dir /path/to/Kimi-K3-MXFP4 --bf16-dir /path/to/Kimi-K3-BF16 \
        --data /path/to/dapo-math-17k.jsonl --n-prompts 8

Usage notes (see docs/en/guide/kimi-k3.md):

- this sglang fork takes ``return_logprob`` as a top-level ``Engine.generate``
  kwarg, not inside ``sampling_params``;
- the K3 chat template requires segmented content (``[{"type": "text", ...}]``);
- BF16 copies ship remote code — pass ``trust_remote_code=True``.
"""

import argparse
import json
import math
import os
import sys


os.environ.setdefault("SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK", "false")


def load_input_ids(tokenizer, data_path: str, n_prompts: int) -> list[list[int]]:
    rows = []
    with open(data_path) as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) >= n_prompts:
                break
    ids = []
    for row in rows:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": row["prompt"]}]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        ids.append(tokenizer(text, add_special_tokens=False)["input_ids"])
    return ids


def prompt_logprobs(model_path: str, all_ids: list[list[int]]):
    import sglang as sgl

    engine = sgl.Engine(
        model_path=model_path,
        tp_size=1,
        mem_fraction_static=0.6,
        random_seed=0,
        log_level="warning",
        trust_remote_code=True,
    )
    try:
        outs = engine.generate(
            input_ids=all_ids,
            sampling_params={"max_new_tokens": 1, "temperature": 0.0},
            return_logprob=True,
            logprob_start_len=0,
            top_logprobs_num=0,
        )
    finally:
        engine.shutdown()
    # input_token_logprobs: per-sample list of (logprob, token_id, text|None); pos0 is None
    return [
        [entry[0] if entry is not None else None for entry in out["meta_info"]["input_token_logprobs"]] for out in outs
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mxfp4-dir", required=True, help="native MXFP4 release dir")
    parser.add_argument("--bf16-dir", required=True, help="BF16 dequantized copy dir")
    parser.add_argument("--data", required=True, help="prompt jsonl with a 'prompt' key")
    parser.add_argument("--n-prompts", type=int, default=8)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.mxfp4_dir, trust_remote_code=True)
    all_ids = load_input_ids(tokenizer, args.data, args.n_prompts)
    print(f"probe set: {len(all_ids)} prompts, {sum(len(x) for x in all_ids)} tokens", flush=True)

    lp_bf16 = prompt_logprobs(args.bf16_dir, all_ids)
    print("bf16 engine done", flush=True)
    lp_mxfp4 = prompt_logprobs(args.mxfp4_dir, all_ids)
    print("mxfp4 engine done", flush=True)

    diffs = []
    prob_diffs = []
    for ids, lb, lm in zip(all_ids, lp_bf16, lp_mxfp4):
        assert len(ids) == len(lb) == len(lm), (len(ids), len(lb), len(lm))
        for i in range(1, len(ids)):
            if lb[i] is None or lm[i] is None:
                continue
            d = lm[i] - lb[i]
            diffs.append(d)
            prob_diffs.append(abs(math.exp(lm[i]) - math.exp(lb[i])))

    n = len(diffs)
    mean_abs = sum(abs(d) for d in diffs) / n
    rms = math.sqrt(sum(d * d for d in diffs) / n)
    k3 = sum(math.exp(d) - d - 1 for d in diffs) / n
    k1 = sum(diffs) / n
    prob_abs = sum(prob_diffs) / n
    mean_lp_b = sum(x for lb in lp_bf16 for x in lb if x is not None) / n
    mean_lp_m = sum(x for lm in lp_mxfp4 for x in lm if x is not None) / n
    print(f"n={n}")
    print(f"mean_lp bf16={mean_lp_b:.4f} mxfp4={mean_lp_m:.4f}")
    print(f"mean|dlogprob|={mean_abs:.6f}  rms(dlogprob)={rms:.6f}")
    print(f"k3_kl={k3:.6f}  signed_mean(mismatch_kl)={-k1:.6f}  prob_abs_diff={prob_abs:.6e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
