#!/usr/bin/env python3
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Prepare a prefix-truncated K3.1 checkpoint or OpenR1-MM SFT parquet."""

import argparse
import json
import re
import shutil
import struct
from pathlib import Path

from relax.utils.logging_utils import get_logger


LAYER = re.compile(r"^language_model\.model\.layers\.(\d+)\.")
logger = get_logger(__name__)


def prepare_model(source: Path, output: Path, num_layers: int) -> None:
    config = json.loads((source / "config.json").read_text())
    text = config["text_config"]
    if config["model_type"] != "kimi_k31" or not 1 < num_layers < text["num_hidden_layers"]:
        raise ValueError("Expected K3.1 checkpoint and a strictly reduced layer count greater than one")
    text["num_hidden_layers"] = num_layers
    linear = text["linear_attn_config"]
    for field in ("kda_layers", "full_attn_layers"):
        linear[field] = [n for n in linear[field] if n <= num_layers]
    dsa = text["dsa_config"]
    dsa["dsa_layers"] = [n for n in dsa["dsa_layers"] if n <= num_layers]
    if not linear["kda_layers"] or not dsa["dsa_layers"]:
        raise ValueError("Keep at least one KDA layer and one DSA layer")
    if set(dsa["dsa_layers"]) != set(range(1, num_layers + 1)) - set(linear["kda_layers"]):
        raise ValueError("DSA and KDA layout must partition all layers")
    source_index = json.loads((source / "model.safetensors.index.json").read_text())
    weight_map = {
        name: shard
        for name, shard in source_index["weight_map"].items()
        if not (match := LAYER.match(name)) or int(match[1]) < num_layers
    }
    headers = {}
    for shard in sorted(set(weight_map.values())):
        path = source / shard
        with path.open("rb") as stream:
            header_size = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(header_size))
        tensor_entries = {name: entry for name, entry in header.items() if name != "__metadata__"}
        expected_size = 8 + header_size + max(entry["data_offsets"][1] for entry in tensor_entries.values())
        if path.stat().st_size != expected_size:
            raise ValueError(f"Shard is incomplete: {path}")
        headers[shard] = tensor_entries
    total_size = 0
    for name, shard in weight_map.items():
        start, end = headers[shard][name]["data_offsets"]
        total_size += end - start
    # Validate everything before creating a new output. Never mutate source files.
    output.mkdir(parents=True, exist_ok=False)
    for path in source.iterdir():
        if path.is_file() and path.suffix in (".py", ".model", ".txt", ".json"):
            if path.name not in ("config.json", "model.safetensors.index.json"):
                shutil.copy2(path, output / path.name)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    metadata = dict(source_index.get("metadata", {}))
    metadata["total_size"] = total_size
    (output / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": metadata, "weight_map": weight_map}, indent=2) + "\n"
    )
    for shard in sorted(headers):
        (output / shard).symlink_to((source / shard).resolve())
    (output / "reduction_manifest.json").write_text(
        json.dumps(
            {
                "source": str(source.resolve()),
                "kept_layers_zero_based": list(range(num_layers)),
                "num_tensors": len(weight_map),
                "tensor_bytes": total_size,
                "shards": sorted(headers),
                "dsa_layers_one_based": dsa["dsa_layers"],
            },
            indent=2,
        )
        + "\n"
    )
    logger.info("Prepared %s tensors across %s shards in %s", len(weight_map), len(headers), output)


def prepare_data(source: Path, output: Path, max_rows: int) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if output.exists():
        raise FileExistsError(output)
    table = pq.read_table(source, columns=["prompt", "image", "label"])
    if max_rows:
        table = table.slice(0, max_rows)
    prompts = []
    for index, row in enumerate(table.select(["prompt", "label"]).to_pylist()):
        if not isinstance(row["prompt"], list) or not isinstance(row["label"], str) or not row["label"]:
            raise ValueError(f"Invalid OpenR1 prompt/label at row {index}")
        if any(message["role"] == "assistant" for message in row["prompt"]):
            raise ValueError(f"Row {index} already contains an assistant; avoid duplicating supervision")
        prompts.append(row["prompt"] + [{"role": "assistant", "content": row["label"]}])
    result = pa.table({"prompt": pa.array(prompts, type=table.schema.field("prompt").type), "image": table["image"]})
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(result, output)
    logger.info("Prepared %s SFT rows in %s", len(result), output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    model = subparsers.add_parser("model")
    model.add_argument("source", type=Path)
    model.add_argument("output", type=Path)
    model.add_argument("--num-layers", type=int, default=4)
    data = subparsers.add_parser("data")
    data.add_argument("source", type=Path)
    data.add_argument("output", type=Path)
    data.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()
    if args.command == "model":
        prepare_model(args.source, args.output, args.num_layers)
    else:
        prepare_data(args.source, args.output, args.max_rows)


if __name__ == "__main__":
    main()
