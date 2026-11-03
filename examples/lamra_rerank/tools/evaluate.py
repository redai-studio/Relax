#!/usr/bin/env python3
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Score fixed candidate pairs with the first generated-token Yes/No
probability."""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from examples.lamra_rerank.data import load_rank_image  # noqa: E402
from examples.lamra_rerank.tools.prepare_data import sha256  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    import torch
    import transformers
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    torch.manual_seed(42)
    processor = AutoProcessor.from_pretrained(args.model, use_fast=False)
    processor.tokenizer.padding_side = "left"
    yes_ids = processor.tokenizer("Yes", add_special_tokens=False)["input_ids"]
    no_ids = processor.tokenizer("No", add_special_tokens=False)["input_ids"]
    if len(yes_ids) != 1 or len(no_ids) != 1:
        raise ValueError("The reference scoring protocol requires single-token Yes and No")
    model, info = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, output_loading_info=True
    )
    if any(info.get(key) for key in ["missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"]):
        raise ValueError(f"Incomplete rank checkpoint load: {info}")
    model = model.to(local_rank).eval()
    args.output.mkdir(parents=True, exist_ok=True)
    receipt = args.output / f"loading_rank{rank}.json"
    if receipt.exists():
        raise FileExistsError(receipt)
    receipt.write_text(
        json.dumps(
            {
                "rank": rank,
                "pid": os.getpid(),
                "model": str(args.model.resolve()),
                "loader": "Qwen2VLForConditionalGeneration",
                "loading_info": {
                    key: sorted(value) if isinstance(value, set) else value for key, value in info.items()
                },
                "dtype": "torch.bfloat16",
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "yes_token_id": yes_ids[0],
                "no_token_id": no_ids[0],
                "score": "softmax(first_generated_logits[Yes, No].float())[Yes]",
                "batch_size": args.batch_size,
            },
            indent=2,
        )
    )
    pairs = []
    with args.pairs.open() as stream:
        for index, line in enumerate(stream):
            if index % world == rank:
                pairs.append(json.loads(line))
    score_chunks = []
    started = time.time()
    with ThreadPoolExecutor(max_workers=8) as pool, torch.inference_mode():
        for offset in range(0, len(pairs), args.batch_size):
            rows = pairs[offset : offset + args.batch_size]
            messages = [row["messages"] for row in rows]
            texts = [
                processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
                for message in messages
            ]
            images = list(
                pool.map(
                    load_rank_image,
                    [
                        part["image"]
                        for message in messages
                        for part in message[0]["content"]
                        if part["type"] == "image"
                    ],
                )
            )
            inputs = processor(text=texts, images=images or None, padding=True, return_tensors="pt")
            inputs = inputs.to(local_rank)
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
            # Only the first generated score is used by the reference. Keeping
            # generate applies the same generation-config logit processors.
            generated = model.generate(
                **inputs, max_new_tokens=1, do_sample=False, output_scores=True, return_dict_in_generate=True
            )
            logits = generated.scores[0]
            score_chunks.append(torch.softmax(logits[:, [yes_ids[0], no_ids[0]]].float(), dim=-1)[:, 0])
            if offset == 0 or (offset // args.batch_size + 1) % 16 == 0:
                progress = {
                    "at": time.time(),
                    "rank": rank,
                    "pid": os.getpid(),
                    "completed_pairs": offset + len(rows),
                    "total_pairs": len(pairs),
                    "runtime_s": time.time() - started,
                }
                target = args.output / f"progress_rank{rank}.json"
                temporary = target.with_suffix(".tmp")
                temporary.write_text(json.dumps(progress))
                temporary.replace(target)
    # One output-boundary transfer, outside the GPU scoring loop.
    values = torch.cat(score_chunks).cpu().tolist() if score_chunks else []
    target = args.output / f"scores_rank{rank}.jsonl"
    with target.open("x") as stream:
        for row, value in zip(pairs, values, strict=True):
            stream.write(
                json.dumps({"index": row["index"], "qid": row["qid"], "did": row["did"], "p_yes": value}) + "\n"
            )
    (args.output / f"completion_rank{rank}.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "rank": rank,
                "world_size": world,
                "pairs": len(values),
                "runtime_s": time.time() - started,
                "scores_sha256": sha256(target),
                "pairs_sha256": sha256(args.pairs),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
