# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from Miles (Apache-2.0):
# https://github.com/radixark/miles/blob/6f52e0b3021484e8d6ad0d43c1074100adc8e4b9/tools/convert_deepseek_v4_1_to_bf16.py

"""Prepare text-only V4.1 BF16 HF weights, preserving frozen Engram bytes.

Process one tensor at a time; raw tensors (including the large Engram tables)
are streamed without loading them into Torch. No Miles or GPU dependency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import struct
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO

from relax.utils.logging_utils import get_logger


if TYPE_CHECKING:
    import torch

logger = get_logger(__name__)
COPY_BYTES = 64 * 1024**2
RAW_SUFFIXES = (".engram.embed.weight", ".engram.embed.scale")
DROP_PREFIXES = ("mtp.", "vision.", "aligner.", "image_")
DTYPE_BYTES = {"BF16": 2, "F32": 4, "F8_E4M3": 1, "F8_E8M0": 1, "I8": 1, "U8": 1}


def read_header(path: Path) -> tuple[dict[str, Any], int]:
    with path.open("rb") as source:
        prefix = source.read(8)
        if len(prefix) != 8:
            raise ValueError(f"Invalid safetensors header: {path}")
        size = struct.unpack("<Q", prefix)[0]
        if size > min(path.stat().st_size - 8, 100 * 1024**2):
            raise ValueError(f"Invalid safetensors header size: {path}")
        header = json.loads(source.read(size))
    header.pop("__metadata__", None)
    return header, size + 8


def convert_config(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("model_type") != "deepseek_v41" or not config.get("text_config"):
        raise ValueError("Expected an original V4.1 composite checkpoint with text_config.")
    quant = config.get("quantization_config", {})
    if quant.get("scale_fmt") != "ue8m0" or quant.get("expert_dtype") != "fp4":
        raise ValueError("Expected FP8 dense + FP4 experts with UE8M0 scales.")
    if quant.get("weight_block_size") != [32, 32]:
        raise ValueError("Only the V4.1 32x32 FP8 block format is supported.")
    text = {k: v for k, v in config["text_config"].items() if k != "model_type" and not k.startswith("dspark_")}
    text.update({k: v for k, v in config.items() if k not in ("text_config", "vision_config", "quantization_config")})
    text.pop("quantization_config", None)
    text["num_nextn_predict_layers"] = 0
    return text


def kept(name: str) -> bool:
    return not name.startswith(DROP_PREFIXES) and not name.endswith(".ffn.gate.bias_vl")


def converted_entry(name: str, entry: dict[str, Any]) -> dict[str, Any]:
    dtype, shape = entry["dtype"], list(entry["shape"])
    if name.endswith(RAW_SUFFIXES):
        allowed = ("F8_E4M3", "U8") if name.endswith(".weight") else ("F8_E8M0", "U8")
        if dtype not in allowed:
            raise ValueError(f"Unexpected frozen Engram dtype: {name}: {dtype}")
    elif dtype in ("F8_E4M3", "I8"):
        if not name.endswith(".weight") or len(shape) != 2:
            raise ValueError(f"Unsupported quantized tensor: {name}: {entry}")
        if dtype == "I8":
            if ".ffn.experts." not in name:
                raise ValueError(f"Packed FP4 is only supported for routed experts: {name}")
            shape[1] *= 2
        dtype = "BF16"
    elif dtype not in ("BF16", "F32"):
        raise ValueError(f"Unexpected dtype: {name}: {dtype}")
    return {"dtype": dtype, "shape": shape}


def plan_conversion(src: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Inspect all headers, scales and indexed tensors before writing anything."""
    config = convert_config(json.loads((src / "config.json").read_text()))
    weight_map = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    shards, output, all_entries = {}, {}, {}
    for filename in sorted(set(weight_map.values())):
        if Path(filename).name != filename:
            raise ValueError(f"Expected a shard filename, got {filename!r}")
        header, start = read_header(src / filename)
        file_size = (src / filename).stat().st_size
        shards[filename] = (header, start)
        for name, entry in header.items():
            if name in all_entries or weight_map.get(name) != filename:
                raise ValueError(f"Duplicate or incorrectly indexed source tensor: {name}")
            begin, end = entry["data_offsets"]
            size = math.prod(entry["shape"]) * DTYPE_BYTES.get(entry["dtype"], 0)
            if begin < 0 or end - begin != size or start + end > file_size:
                raise ValueError(f"Invalid tensor offsets: {name}")
            all_entries[name] = entry
    if all_entries.keys() != weight_map.keys():
        raise ValueError("Source index and shard tensors differ.")
    for name, entry in all_entries.items():
        if not kept(name):
            continue
        if name.endswith(".scale") and not name.endswith(RAW_SUFFIXES):
            weight = all_entries.get(name.removesuffix("scale") + "weight", {})
            if weight.get("dtype") not in ("F8_E4M3", "I8"):
                raise ValueError(f"Orphan scale: {name}")
            continue
        converted = converted_entry(name, entry)
        if entry["dtype"] in ("F8_E4M3", "I8") and not name.endswith(RAW_SUFFIXES):
            scale = all_entries.get(name.removesuffix("weight") + "scale", {})
            rows, cols = entry["shape"]
            expected = [math.ceil(rows / 32), math.ceil(cols / 32)]
            if entry["dtype"] == "I8":
                expected = [rows, cols * 2 // 32]
                if cols * 2 % 32:
                    raise ValueError(f"Invalid FP4 block width: {name}")
            if scale.get("dtype") not in ("F8_E8M0", "U8") or scale.get("shape") != expected:
                raise ValueError(f"Missing or incompatible scale: {name}: {scale}")
        output.setdefault(weight_map[name], {})[name] = converted
    names = {name for entries in output.values() for name in entries}
    required = {"embed.weight", "head.weight", "norm.weight"}
    for layer in range(config["num_hidden_layers"]):
        required.update(
            f"layers.{layer}.ffn.experts.{e}.w{i}.weight" for e in range(config["n_routed_experts"]) for i in (1, 2, 3)
        )
        required.update(f"layers.{layer}.attn.{proj}.weight" for proj in ("wq_a", "wkv", "wo_a", "wo_b"))
        required.add(f"layers.{layer}.ffn.gate.weight")
    for layer in config["engram_layer_ids"]:
        required.update(f"layers.{layer}.engram.{suffix}" for suffix in ("embed.weight", "embed.scale", "wkv.weight"))
    if missing := required - names:
        raise ValueError(f"Source is missing required text weights: {sorted(missing)[:5]}")
    return config, shards, output


def read_tensor(path: Path, entry: dict[str, Any], start: int, device: str) -> torch.Tensor:
    import torch

    begin, end = entry["data_offsets"]
    with path.open("rb") as source:
        source.seek(start + begin)
        data = bytearray(source.read(end - begin))
    if len(data) != end - begin:
        raise ValueError(f"Truncated tensor in {path}")
    dtype = torch.float8_e4m3fn if entry["dtype"] == "F8_E4M3" else torch.uint8
    return torch.frombuffer(data, dtype=dtype).reshape(entry["shape"]).to(device)


def dequantize(weight: torch.Tensor, scale: torch.Tensor, *, fp4: bool) -> torch.Tensor:
    import torch

    # E8M0 byte 255 encodes NaN; refuse corrupt weights instead of writing infinities.
    if (scale == 255).any():
        raise ValueError("Encountered a NaN E8M0 scale.")
    scales = torch.exp2(scale.float() - 127.0)
    if fp4:
        values = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
            device=weight.device,
            dtype=torch.float32,
        )
        unpacked = torch.stack((values[(weight & 15).long()], values[(weight >> 4).long()]), dim=-1)
        return (unpacked.reshape(weight.shape[0], -1) * scales.repeat_interleave(32, dim=1)).to(torch.bfloat16)
    rows, cols = weight.shape
    scales = scales.repeat_interleave(32, dim=0)[:rows].repeat_interleave(32, dim=1)[:, :cols]
    return (weight.float() * scales).to(torch.bfloat16)


def copy_bytes(source: Path, offset: int, size: int, target: BinaryIO) -> None:
    with source.open("rb") as stream:
        stream.seek(offset)
        while size:
            chunk = stream.read(min(size, COPY_BYTES))
            if not chunk:
                raise ValueError(f"Truncated tensor in {source}")
            target.write(chunk)
            size -= len(chunk)


def output_header(entries: dict[str, Any]) -> tuple[dict[str, Any], int]:
    header, offset = {}, 0
    for name, entry in entries.items():
        size = math.prod(entry["shape"]) * DTYPE_BYTES[entry["dtype"]]
        header[name] = {**entry, "data_offsets": [offset, offset + size]}
        offset += size
    return header, offset


def write_json(path: Path, data: dict[str, Any]) -> None:
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(data, indent=2) + "\n")
    pending.replace(path)


def convert_shard(
    src: Path,
    dst: Path,
    filename: str,
    shards: dict[str, Any],
    entries: dict[str, Any],
    source_map: dict[str, str],
    device: str,
) -> None:
    import torch

    header, _ = output_header(entries)
    blob = json.dumps({"__metadata__": {"format": "pt"}, **header}, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    pending = dst / (filename + ".tmp")
    source_header, start = shards[filename]
    with pending.open("wb") as target:
        target.write(struct.pack("<Q", len(blob)))
        target.write(blob)
        for name in entries:
            entry = source_header[name]
            if entry["dtype"] not in ("F8_E4M3", "I8") or name.endswith(RAW_SUFFIXES):
                begin, end = entry["data_offsets"]
                copy_bytes(src / filename, start + begin, end - begin, target)
                continue
            scale_name = name.removesuffix("weight") + "scale"
            scale_file = source_map[scale_name]
            scale_header, scale_start = shards[scale_file]
            weight = read_tensor(src / filename, entry, start, device)
            scale = read_tensor(src / scale_file, scale_header[scale_name], scale_start, device)
            tensor = dequantize(weight, scale, fp4=entry["dtype"] == "I8").cpu().contiguous()
            target.write(memoryview(tensor.view(torch.uint8).numpy()).cast("B"))
            del weight, scale, tensor
    pending.replace(dst / filename)


def convert(src: Path, dst: Path, *, device: str = "cpu", resume: bool = False, check_only: bool = False) -> None:
    src, dst = src.resolve(), dst.resolve()
    if src == dst or src in dst.parents or dst in src.parents:
        raise ValueError("Source and destination must be separate directories, not nested.")
    config, shards, output = plan_conversion(src)
    total_size = sum(output_header(entries)[1] for entries in output.values())
    logger.info(
        "Planned %d tensors in %d shards, %.2f GB", sum(map(len, output.values())), len(output), total_size / 1e9
    )
    if check_only:
        return
    source_map = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    identity = {"source": str(src), "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for filename in ("config.json", "model.safetensors.index.json"):
        identity[filename + "_sha256"] = hashlib.sha256((src / filename).read_bytes()).hexdigest()
    marker = dst / "conversion.json"
    if dst.exists() and any(dst.iterdir()):
        if not resume or not marker.is_file() or json.loads(marker.read_text()).get("source_identity") != identity:
            raise ValueError("Destination is nonempty; --resume requires this converter's matching conversion.json.")
    dst.mkdir(parents=True, exist_ok=True)
    record = {"source_identity": identity, "complete": False}
    write_json(marker, record)
    for filename, entries in output.items():
        path = dst / filename
        if path.exists():
            actual, start = read_header(path)
            expected, size = output_header(entries)
            if actual != expected or path.stat().st_size != start + size:
                raise ValueError(f"Existing shard is incomplete or incompatible: {path}")
            logger.info("Resume: keeping %s", filename)
            continue
        logger.info("Converting %s", filename)
        convert_shard(src, dst, filename, shards, entries, source_map, device)
    if {path.name for path in dst.glob("*.safetensors")} != output.keys():
        raise ValueError("Unexpected output shards; use a clean destination.")
    for pattern in ("tokenizer*", "chat_template*", "generation_config.json", "special_tokens_map.json", "LICENSE*"):
        for path in src.glob(pattern):
            if path.is_file():
                shutil.copy2(path, dst / path.name)
    write_json(dst / "config.json", config)
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": {name: filename for filename, entries in output.items() for name in entries},
    }
    write_json(dst / "model.safetensors.index.json", index)
    record.update(complete=True, tensors=len(index["weight_map"]), total_size=total_size)
    write_json(marker, record)
    logger.info("Complete: %s", dst)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--resume", action="store_true", help="Resume this tool's output with matching source metadata."
    )
    parser.add_argument(
        "--check-only", action="store_true", help="Check source headers and estimate output bytes; no writes."
    )
    args = parser.parse_args()
    convert(args.model_dir, args.save_dir, device=args.device, resume=args.resume, check_only=args.check_only)


if __name__ == "__main__":
    main()
