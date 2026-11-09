# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Audit prepared trajectories through the real CPU SFT input pipeline, without
length filtering."""

import argparse
import random
import re
from pathlib import Path

from common import digest, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples-per-split", type=int, default=0, help="0 audits every row (default)")
    args = parser.parse_args()
    if args.max_samples_per_split < 0:
        parser.error("--max-samples-per-split must be nonnegative")
    from relax.engine.sft.dataset.streaming import SFTStreamingDataset
    from relax.utils.data.processing_utils import load_processor
    from relax.utils.data.processor_pool import ProcessorPool
    from relax.utils.multimodal.config import MultimodalConfig

    processor = load_processor(str(args.model), local_files_only=True)
    tokenizer = processor.tokenizer
    reports = []
    pool = ProcessorPool(
        str(args.model),
        pool_size=1,
        multimodal_config=MultimodalConfig(image_min_token_num=1, image_max_token_num=256),
    )
    try:
        for split in ("train", "val", "test"):
            dataset = SFTStreamingDataset(
                str(args.data_dir / f"{split}.jsonl"),
                tokenizer=tokenizer,
                processor_pool=pool,
                prompt_key="messages",
                tool_key="tools",
                multimodal_keys={"image": "images"},
                prefetch_max_cached=0,
                pad_token_ids=[tokenizer.convert_tokens_to_ids("<|image_pad|>")],
                apply_chat_template_kwargs={"preserve_thinking": True},
            )
            indices = list(range(len(dataset)))
            if args.max_samples_per_split and len(indices) > args.max_samples_per_split:
                indices = sorted(random.Random(42).sample(indices, args.max_samples_per_split))
            for index in indices:
                row = dataset.reader[index]
                processed = dataset.get_batch_by_indices([index])[0]
                ids = processed.tokens.tolist()
                learned = processed.tokens[processed.loss_mask.bool()].tolist()
                if not learned:
                    raise ValueError(f"No supervision in {row['id']}")
                decoded = tokenizer.decode(learned, skip_special_tokens=False)
                for message in row["messages"]:
                    if message["role"] == "assistant":
                        for thinking in re.findall(r"<think>(.*?)</think>", message["content"], re.DOTALL):
                            if thinking.strip() and thinking.strip() not in decoded:
                                raise ValueError(f"Reasoning supervision missing in {row['id']}")
                reports.append(
                    {
                        "id": row["id"],
                        "split": split,
                        "tokens": len(ids),
                        "learned_tokens": len(learned),
                        "input_ids_sha256": digest(ids),
                        "supervised_ids_sha256": digest(learned),
                    }
                )
    finally:
        pool.shutdown()
    write_json(
        args.output,
        {
            "rows": reports,
            "truncated": 0,
            "note": "Actual SFT tokens and expanded loss masks; no sequence-length filtering",
        },
    )


if __name__ == "__main__":
    main()
