# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU-only Relax preflight: fail instead of truncating selected 32K trajectories."""

import argparse
import re
from pathlib import Path

from common import chat_messages, digest, read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from PIL import Image
    from transformers import AutoProcessor

    from relax.engine.sft.dataset.chat_template import render_to_text, render_with_loss_mask
    from relax.engine.sft.dataset.sample import CanonicalMessage, CanonicalSample

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    processor.image_processor.min_pixels = 1024
    processor.image_processor.max_pixels = 262144
    tokenizer = processor.tokenizer
    reports = []
    for split in ("train", "val", "test"):
        for row in read_jsonl(args.data_dir / f"{split}.jsonl"):
            sample = CanonicalSample(
                messages=[CanonicalMessage(**m, learn=m["role"] == "assistant")
                          for m in chat_messages(row["messages"])],
                tools=row["tools"], images=row["images"],
                metadata={"source_dataset": "opensearch", "row_index": row["id"]},
            )
            short_ids, mask = render_with_loss_mask(
                sample, tokenizer=tokenizer, apply_chat_template_kwargs={"preserve_thinking": True},
            )
            learned = short_ids[mask.bool()].tolist()
            text = render_to_text(sample, tokenizer=tokenizer, apply_chat_template_kwargs={"preserve_thinking": True})
            images = [Image.open(path).convert("RGB") for path in row["images"]]
            inputs = processor(text=[text], images=images, return_tensors=None)
            ids = inputs["input_ids"][0]
            if len(ids) > 32768:
                raise ValueError(f"{row['id']} has {len(ids)} tokens; refuse truncation")
            decoded = tokenizer.decode(learned, skip_special_tokens=False)
            for message in row["messages"]:
                if message["role"] == "assistant":
                    for thinking in re.findall(r"<think>(.*?)</think>", message["content"], re.DOTALL):
                        if thinking.strip() and thinking.strip() not in decoded:
                            raise ValueError(f"Reasoning supervision missing in {row['id']}")
            reports.append({"id": row["id"], "split": split, "tokens": len(ids), "learned_tokens": len(learned),
                            "input_ids_sha256": digest(ids), "supervised_ids_sha256": digest(learned)})
    write_json(args.output, {"rows": reports, "truncated": 0,
                             "note": "Loss-mask hashes cover unexpanded text tokens; images are conditioning context"})


if __name__ == "__main__":
    main()
