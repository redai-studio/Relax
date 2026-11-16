#!/usr/bin/env python3
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Prepare the LamRA <emb> checkpoint and NLI pair JSONL for Ret PT."""

import argparse
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory


def write_pairs(source: Path, output: Path) -> None:
    with source.open(newline="", encoding="utf-8") as stream:
        rows = csv.DictReader(stream)
        if not {"sent0", "sent1"}.issubset(rows.fieldnames or ()):
            raise ValueError("NLI CSV requires sent0 and sent1 columns")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as destination:
            count = 0
            for row in rows:
                count += 1
                messages = []
                for column in ("sent0", "sent1"):
                    text = row[column] or "nan"
                    messages.extend(
                        [
                            {"role": "user", "content": f"{text}\nSummarize above sentence in one word: "},
                            {"role": "assistant", "content": "<emb>."},
                        ]
                    )
                destination.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
            if not count:
                raise ValueError("NLI CSV has no pairs")


def prepare_model(source: Path, output: Path) -> None:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(source, use_fast=False)
    model, loading = AutoModelForImageTextToText.from_pretrained(source, dtype="auto", output_loading_info=True)
    if model.config.model_type not in {"qwen2_vl", "qwen3_5"}:
        raise ValueError("Ret PT requires a Qwen2-VL or Qwen3.5 base")
    if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"Incomplete base checkpoint load: {loading}")
    tokenizer = processor.tokenizer
    existing = "<emb>" in tokenizer.get_vocab()
    if not existing:
        tokenizer.add_tokens(["<emb>"])
    ids = tokenizer.encode("<emb>", add_special_tokens=False)
    embedding_rows = model.get_input_embeddings().weight.shape[0]
    if len(ids) != 1 or (existing and ids[0] >= embedding_rows):
        raise ValueError("Existing <emb> must have a matching vocabulary row")
    if ids[0] >= embedding_rows:
        model.resize_token_embeddings(len(tokenizer))
    model.save_pretrained(output)
    processor.save_pretrained(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    args = parser.parse_args()
    for output in (args.pairs, args.output_model):
        if output.exists():
            raise FileExistsError(output)
    args.pairs.parent.mkdir(parents=True, exist_ok=True)
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    with (
        TemporaryDirectory(dir=args.pairs.parent, prefix=".nli-") as pairs_workspace,
        TemporaryDirectory(dir=args.output_model.parent, prefix=".ret-model-") as model_workspace,
    ):
        pairs = Path(pairs_workspace) / "pairs.jsonl"
        checkpoint = Path(model_workspace) / "model"
        write_pairs(args.input, pairs)
        prepare_model(args.base, checkpoint)
        checkpoint.rename(args.output_model)
        pairs.rename(args.pairs)


if __name__ == "__main__":
    main()
