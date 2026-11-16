#!/usr/bin/env python3
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Mine Rank training candidates or evaluate a merged LamRA Ret checkpoint."""

import argparse
import json
import random
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torch.nn.functional as F


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from examples.lamra_rerank.data import format_text, load_rank_image, make_entity  # noqa: E402
from examples.lamra_rerank.tools.prepare_data import sha256  # noqa: E402
from examples.seq_contrast_sft.mbeir import retrieval_messages  # noqa: E402
from examples.seq_contrast_sft.tools.prepare_mbeir import load_instructions  # noqa: E402


def read_rows(path: Path, id_key: str) -> dict:
    rows = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            identifier = row[id_key]
            if identifier in rows:
                raise ValueError(f"Duplicate {id_key} {identifier!r} in {path}")
            rows[identifier] = row
    if not rows:
        raise ValueError(f"Empty input: {path}")
    return rows


def read_qrels(path: Path) -> dict[str, set[str]]:
    qrels = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            qid, _, did, relevance = line.split()
            if int(relevance) > 0:
                qrels.setdefault(qid, set()).add(did)
    return qrels


@torch.inference_mode()
def encode(model, processor, entities: list[dict], batch_size: int, device: torch.device) -> torch.Tensor:
    marker = processor.tokenizer.encode("<emb>", add_special_tokens=False)
    if len(marker) != 1 or "<emb>" not in processor.tokenizer.get_vocab():
        raise ValueError("Ret checkpoint must contain a single <emb> token")
    if marker[0] >= model.get_input_embeddings().weight.shape[0]:
        raise ValueError("<emb> is outside the serialized model vocabulary")
    vectors = None
    for start in range(0, len(entities), batch_size):
        messages = [retrieval_messages(entity) for entity in entities[start : start + batch_size]]
        texts = [processor.apply_chat_template(row, tokenize=False, add_generation_prompt=False) for row in messages]
        images = [
            load_rank_image(part["image"]) for row in messages for part in row[0]["content"] if part["type"] == "image"
        ]
        inputs = processor(text=texts, images=images or None, padding=True, return_tensors="pt").to(device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
        mask = inputs["input_ids"].eq(marker[0])
        positions = mask.long().argmax(dim=-1)
        if not ((mask.sum(dim=-1) == 1) & (positions > 0)).all():
            raise ValueError("Each Ret example requires one <emb> with a preceding token")
        hidden = model.model(**inputs, use_cache=False).last_hidden_state
        selected = hidden[torch.arange(len(positions), device=device), positions - 1].float()
        if not selected.isfinite().all():
            raise ValueError("Non-finite Ret embedding")
        features = F.normalize(selected, dim=-1).cpu()
        if vectors is None:
            vectors = torch.empty((len(entities), features.shape[-1]), dtype=torch.float32)
        vectors[start : start + len(features)] = features
    return vectors


def search(
    queries: torch.Tensor,
    candidates: torch.Tensor,
    top_k: int,
    device: torch.device,
    *,
    query_batch: int = 256,
    candidate_batch: int = 8192,
) -> tuple[list[list[int]], list[list[float]]]:
    """Exact cosine search after normalization with a bounded score matrix."""
    names, scores = [], []
    for query_chunk in queries.split(query_batch):
        left = query_chunk.to(device)
        best_scores = left.new_empty((len(left), 0))
        best_ids = torch.empty((len(left), 0), dtype=torch.long, device=device)
        for start in range(0, len(candidates), candidate_batch):
            right = candidates[start : start + candidate_batch].to(device)
            block_scores, block_ids = (left @ right.T).topk(min(top_k, len(right)), dim=1)
            values = torch.cat((best_scores, block_scores), dim=1)
            indices = torch.cat((best_ids, block_ids + start), dim=1)
            best_scores, offsets = values.topk(min(top_k, values.shape[1]), dim=1)
            best_ids = indices.gather(1, offsets)
        names.extend(best_ids.tolist())
        scores.extend(best_scores.tolist())
    return names, scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "queries", "candidates", "instructions", "image-root", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, help="Evaluation qrels. Defaults to the query positives for mining.")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8, help="Embedding batch size")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    queries = read_rows(args.queries, "qid")
    candidates = read_rows(args.candidates, "did")
    qrels = read_qrels(args.qrels) if args.qrels else {qid: set(row["pos_cand_list"]) for qid, row in queries.items()}
    if set(queries) != set(qrels) or any(not values for values in qrels.values()):
        raise ValueError("Every query requires its complete positive labels")
    if args.batch_size <= 0 or not 1 <= args.top_k <= len(candidates):
        raise ValueError("Batch size must be positive and top-k must fit the candidate pool")

    instructions = load_instructions(args.instructions)
    rng = random.Random(42)
    query_entities = []
    for qid, row in queries.items():
        positive = candidates[row["pos_cand_list"][0]]
        key = (qid.split(":")[0], row["query_modality"], positive["modality"])
        text = format_text(f"{rng.choice(instructions[key])} {row.get('query_txt') or ''}")
        query_entities.append(make_entity(text, row.get("query_img_path"), args.image_root))
    candidate_entities = [
        make_entity(format_text(row.get("txt")), row.get("img_path"), args.image_root) for row in candidates.values()
    ]

    from transformers import AutoModelForImageTextToText, AutoProcessor

    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model, loading = AutoModelForImageTextToText.from_pretrained(args.model, dtype=dtype, output_loading_info=True)
    if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"Incomplete Ret checkpoint load: {loading}")
    model = model.to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model, use_fast=False)
    processor.tokenizer.padding_side = "left"
    left = encode(model, processor, query_entities, args.batch_size, device)
    right = encode(model, processor, candidate_entities, args.batch_size, device)
    del model
    indices, scores = search(left, right, args.top_k, device)
    candidate_ids = list(candidates)
    names = [[candidate_ids[index] for index in row] for row in indices]
    rankings = {
        qid: {"gt_docs": sorted(qrels[qid]), "top100_docs": row[:100]} for qid, row in zip(queries, names, strict=True)
    }
    recalls = {
        f"recall@{k}": 100
        * sum(bool(qrels[qid].intersection(row[:k])) for qid, row in zip(queries, names, strict=True))
        / len(queries)
        for k in (1, 5, 10, 20, 50, 100)
        if k <= args.top_k
    }
    model_files = sorted(args.model.glob("*.safetensors"))
    if not model_files:
        model_files = sorted(args.model.glob("pytorch_model*.bin"))
    if not model_files:
        raise ValueError("Ret checkpoint has no local weight files")
    identity = {"model": str(args.model.resolve()), "sha256": {path.name: sha256(path) for path in model_files}}
    outputs = {
        "query_names.json": list(queries),
        "candidate_names.json": names,
        "scores.json": scores,
        "rankings.json": rankings,
        "model_identity.json": identity,
        "metrics.json": {"queries": len(queries), "candidates": len(candidates), "top_k": args.top_k, **recalls},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent, prefix=".retrieve-") as temporary:
        staged = Path(temporary) / "results"
        staged.mkdir()
        for name, value in outputs.items():
            with (staged / name).open("w", encoding="utf-8") as stream:
                json.dump(value, stream, ensure_ascii=False)
                stream.write("\n")
        staged.rename(args.output)


if __name__ == "__main__":
    main()
