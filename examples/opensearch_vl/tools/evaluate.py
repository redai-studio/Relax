# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Common HF evaluation of base, reference and Relax exports; no tools execute."""

import argparse
import json
import math
from pathlib import Path

from common import (
    canonical_sample,
    digest,
    extract_answer,
    extract_calls,
    normalized_text,
    read_jsonl,
    write_json,
)


def score_completion(reference: str, completion: str, kind: str) -> dict:
    if kind == "call":
        gold = extract_calls(reference)
        try:
            predicted = extract_calls(completion)
        except (ValueError, TypeError):
            predicted = []
        return {
            "call_parse_ok": bool(predicted),
            "tool_names_match": bool(predicted) and [c["name"] for c in predicted] == [c["name"] for c in gold],
            "call_exact_match": bool(predicted) and predicted == gold,
        }
    gold, predicted = extract_answer(reference), extract_answer(completion)
    return {"answer_format_ok": predicted is not None,
            "answer_exact_match": predicted is not None and normalized_text(predicted) == normalized_text(gold or "")}


def summarize(records: list[dict]) -> dict:
    result = {"expected_cases": len(records), "failures": sum("error" in r for r in records)}
    for kind in ("call", "answer"):
        subset = [r for r in records if r["kind"] == kind]
        result[f"{kind}_cases"] = len(subset)
        keys = ("call_parse_ok", "tool_names_match", "call_exact_match") if kind == "call" else (
            "answer_format_ok", "answer_exact_match")
        for key in keys:
            result[key] = sum(bool(r.get(key, False)) for r in subset) / len(subset) if subset else None
    count = sum(r.get("target_tokens", 0) for r in records)
    total = sum(r.get("nll_sum", 0.0) for r in records)
    result.update({"scored_tokens": count, "nll": total / count if count else None,
                   "ppl": math.exp(min(total / count, 700)) if count else None})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="One explicit HF checkpoint, never a parent directory")
    parser.add_argument("--processor", type=Path, required=True, help="Same original Qwen3.5-4B processor for all runs")
    parser.add_argument("--data", type=Path, required=True, help="Prepared test_steps.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["nll", "generate", "both"], default="both")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory for each checkpoint")
    import torch
    import torch.nn.functional as functional
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from relax.engine.sft.dataset.chat_template import render_to_text

    if not torch.cuda.is_available():
        raise RuntimeError("Run this evaluator on a CUDA worker; never execute model inference on the local Mac")
    torch.manual_seed(42)
    processor = AutoProcessor.from_pretrained(args.processor, local_files_only=True)
    processor.image_processor.min_pixels = 1024
    processor.image_processor.max_pixels = 262144
    model, loading = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", local_files_only=True,
        output_loading_info=True,
    )
    if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"Incomplete HF checkpoint: {loading}")
    model = model.to("cuda").eval()
    cases = read_jsonl(args.data)
    records = []
    args.output.mkdir(parents=True)
    with (args.output / "predictions.jsonl").open("w") as sink:
        for case in cases:
            record = {"id": case["id"], "kind": case["kind"], "metadata": case["metadata"]}
            try:
                row = {**case, "messages": case["messages"] + [{"role": "assistant", "content": case["reference"]}]}
                text = render_to_text(
                    canonical_sample(row), tokenizer=processor.tokenizer,
                    apply_chat_template_kwargs={"preserve_thinking": True},
                )
                marker = "<|im_start|>assistant\n"
                boundary = text.rfind(marker)
                if boundary < 0:
                    raise ValueError("Expected Qwen ChatML assistant boundary")
                prompt = text[:boundary + len(marker)]
                images = [Image.open(path).convert("RGB") for path in case["images"]]
                kwargs = {"images": images} if images else {}
                full = processor(text=[text], return_tensors="pt", **kwargs).to("cuda")
                prefix = processor(text=[prompt], return_tensors="pt", **kwargs).to("cuda")
                start, end = prefix["input_ids"].shape[1], full["input_ids"].shape[1]
                if not torch.equal(full["input_ids"][:, :start], prefix["input_ids"]):
                    raise ValueError("Prefix tokenization changed at the assistant boundary")
                if end > 32768 or end <= start:
                    raise ValueError(f"Invalid full sequence length {end}; no truncation is permitted")
                record.update({"prompt_tokens": start, "full_tokens": end,
                               "input_sha256": digest(full["input_ids"][0].tolist())})
                with torch.inference_mode():
                    if args.mode in {"nll", "both"}:
                        # Only materialize logits needed to score this assistant decision.
                        output = model(**full, logits_to_keep=end - start + 1, use_cache=False)
                        logits = output.logits[0, :-1]
                        labels = full["input_ids"][0, start:]
                        total = 0.0
                        for offset in range(0, len(labels), 128):
                            total += functional.cross_entropy(
                                logits[offset:offset + 128].float(), labels[offset:offset + 128], reduction="sum",
                            ).item()
                        record.update({"nll_sum": total, "target_tokens": len(labels)})
                        del output, logits
                    if args.mode in {"generate", "both"}:
                        budget = min(args.max_new_tokens, 32768 - start)
                        if budget <= 0:
                            raise ValueError("No generation budget remains at 32K")
                        tokens = model.generate(**prefix, do_sample=False, max_new_tokens=budget, use_cache=True)
                        generated = tokens[0, start:]
                        completion = processor.tokenizer.decode(generated, skip_special_tokens=False)
                        record.update({"completion": completion, "reference": case["reference"],
                                       "generated_tokens": len(generated), "budget_exhausted": len(generated) >= budget})
                        record.update(score_completion(case["reference"], completion, case["kind"]))
                del full, prefix
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
            records.append(record)
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            sink.flush()
    protocol = {"data_sha256": digest(cases), "mode": args.mode,
                "processor": str(args.processor.resolve()), "max_length": 32768,
                "max_new_tokens": args.max_new_tokens, "seed": 42, "renderer": "relax_qwen_preserve_thinking_v1"}
    write_json(args.output / "summary.json", {
        "model": str(args.model.resolve()), "protocol": protocol, "metrics": summarize(records),
        "length_buckets": {name: summarize([r for r in records if low <= r.get("prompt_tokens", -1) < high])
                           for name, low, high in [("under16k", 0, 16384), ("16k-24k", 16384, 24576),
                                                   ("24k-32k", 24576, 32769)]},
        "versions": {"torch": torch.__version__, "transformers": __import__("transformers").__version__},
    })
    if any("error" in r for r in records):
        raise SystemExit("Evaluation has failed cases; inspect predictions.jsonl. Do not report this run as passing.")


if __name__ == "__main__":
    main()
