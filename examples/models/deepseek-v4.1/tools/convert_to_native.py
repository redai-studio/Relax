# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Restore a trained V4.1 BF16 export to its original 1017 mixed-precision HF layout.

Run on the shared Ray cluster with this tools directory, Relax and Megatron Bridge
on workers' PYTHONPATH. Each worker reads and writes one original model shard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections import Counter
from contextlib import ExitStack
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from convert_to_bf16 import dequantize, read_header

from relax.utils.logging_utils import get_logger


if TYPE_CHECKING:
    import torch

logger = get_logger(__name__)

INDEX = "model.safetensors.index.json"
CHUNK = 16 * 1024 * 1024


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def is_frozen(key: str) -> bool:
    return key.startswith(("vision.", "aligner.", "image_")) or key.endswith(
        (".ffn.gate.bias_vl", ".engram.embed.weight", ".engram.embed.scale")
    )


def sampled(key: str) -> bool:
    return (
        re.fullmatch(
            r"layers\.(?:0|20|39)\.(?:ffn\.experts\.0\.w[123]\.weight|attn\.wq_a\.weight|attn_norm\.weight|hc_attn_fn)",
            key,
        )
        is not None
    )


class Source:
    def __init__(self, root: Path, stack: ExitStack) -> None:
        self.root = root
        self.entries = json.loads((root / INDEX).read_bytes())["weight_map"]
        self.stack = stack
        self.shard = lru_cache(maxsize=None)(self._open)

    def _open(self, name: str) -> tuple[dict[str, Any], int, int]:
        header, offset = read_header(self.root / name)
        stream = self.stack.enter_context((self.root / name).open("rb", buffering=0))
        return header, offset, stream.fileno()

    def view(self, key: str) -> tuple[dict[str, Any], int, int, int]:
        header, offset, fd = self.shard(self.entries[key])
        item = header[key]
        beg, end = item["data_offsets"]
        return item, fd, offset + beg, end - beg


def read_exact(fd: int, start: int, size: int) -> bytes:
    result = os.pread(fd, size, start)
    if len(result) != size:
        raise OSError(f"short read at {start}, expected {size}, got {len(result)}")
    return result


def tensor(source: Source, name: str, device: str) -> torch.Tensor:
    import torch

    item, fd, pos, size = source.view(name)
    dtype = {
        "BF16": torch.bfloat16,
        "F32": torch.float32,
        "F8_E4M3": torch.float8_e4m3fn,
        "F8_E8M0": torch.uint8,
        "I8": torch.uint8,
    }[item["dtype"]]
    buf = bytearray(read_exact(fd, pos, size))
    return torch.frombuffer(buf, dtype=dtype).view(item["shape"]).to(device)


def quantized(
    native: Source, trained: Source, key: str, device: str
) -> tuple[torch.Tensor, torch.Tensor, bool, dict[str, Any] | None]:
    import torch
    import torch.nn.functional as F
    from megatron.bridge.models.conversion.quantization_utils import (
        quantize_fp8_e4m3fn_like_scale,
        quantize_mxfp4_e2m1_like_scale,
    )

    scale_key = key.removesuffix("weight") + "scale"
    fp4 = native.view(key)[0]["dtype"] == "I8"
    base = tensor(native, key, device)
    old_scale = tensor(native, scale_key, device)
    w = tensor(trained, key, device)
    if w.dtype != torch.bfloat16 or (w.shape[0], w.shape[1] // (2 if fp4 else 1)) != base.shape:
        raise ValueError(f"Unexpected BF16 input geometry for {key}")
    original = dequantize(base, old_scale, fp4=fp4)
    same = w == original
    rows, cols = w.shape
    if fp4:
        same_blocks = same.view(rows, -1, 32).all(-1)
    else:
        padded = F.pad(same, (0, -cols % 32, 0, -rows % 32), value=True)
        same_blocks = padded.view(old_scale.shape[0], 32, old_scale.shape[1], 32).all(dim=(1, 3))
    if same_blocks.all():
        packed, scale = base, old_scale
        unchanged = True
    else:
        if not torch.isfinite(w).all():
            raise ValueError(f"Nonfinite trained weight {key}")
        if fp4:
            packed, scale = quantize_mxfp4_e2m1_like_scale(w, old_scale, block_size=32, name=key)
            mask = same_blocks.repeat_interleave(16, dim=1)
        else:
            packed, scale = quantize_fp8_e4m3fn_like_scale(
                w, old_scale.view(torch.float8_e8m0fnu), block_size=32, name=key
            )
            mask = same_blocks.repeat_interleave(32, dim=0)[:rows].repeat_interleave(32, dim=1)[:, :cols]
        packed = torch.where(mask, base.view(torch.uint8), packed.view(torch.uint8)).view(base.dtype)
        scale = torch.where(same_blocks, old_scale, scale.view(torch.uint8))
        if (scale == 255).any() or (not fp4 and (packed.view(torch.uint8).bitwise_and(127) == 127).any()):
            raise ValueError(f"Nonfinite encoded data {key}")
        unchanged = False
    if packed.shape != base.shape or scale.shape != old_scale.shape:
        raise ValueError(f"Native geometry mismatch {key}")
    metrics = None
    if sampled(key):
        actual = dequantize(packed, scale, fp4=fp4).float()
        err = actual - w.float()
        denom = w.float().norm().item()
        metrics = dict(
            dtype="MXFP4" if fp4 else "FP8",
            elements=w.numel(),
            max_abs=err.abs().max().item(),
            rmse=err.norm().item() / math.sqrt(w.numel()),
            relative_l2=err.norm().item() / denom if denom else 0.0,
            changed_code_bytes=(packed.view(torch.uint8) != base.view(torch.uint8)).sum().item(),
            total_code_bytes=base.numel(),
            changed_scale_bytes=(scale != old_scale).sum().item(),
            total_scale_bytes=scale.numel(),
            changed_trained_elements=(~same).sum().item(),
            identical_base_blocks=same_blocks.sum().item(),
            blocks=same_blocks.numel(),
        )
    return packed, scale, unchanged, metrics


def shard_task(native_root: str, trained_root: str, staging: str, filename: str, device: str) -> dict[str, Any]:
    import torch

    torch.set_num_threads(1)
    staging = Path(staging)
    with ExitStack() as stack:
        native = Source(Path(native_root), stack)
        trained = Source(Path(trained_root), stack)
        header, start, in_fd = native.shard(filename)
        counts, metrics = Counter(), {}
        out_final = staging / filename
        out_tmp = staging / (filename + ".partial")
        with out_tmp.open("wb+") as output:
            output.write(read_exact(in_fd, 0, start))
            output.truncate(os.fstat(in_fd).st_size)
            done = set()
            for key, item in sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0]):
                if key in done:
                    continue
                scale = key.endswith(".scale")
                if scale and not key.startswith("mtp.") and not is_frozen(key):
                    key = key.removesuffix("scale") + "weight"
                    if key not in header:
                        raise ValueError(f"Scale partner missing in {filename}: {key}")
                    item = header[key]
                if key in done:
                    continue
                _, nfd, nstart, nsize = native.view(key)
                if key in trained.entries and not is_frozen(key) and item["dtype"] in {"I8", "F8_E4M3"}:
                    skey = key.removesuffix("weight") + "scale"
                    if skey not in header:
                        raise ValueError(f"Missing in-shard scale for {key}")
                    q, s, unchanged, stats = quantized(native, trained, key, device)
                    for name, data in ((key, q), (skey, s)):
                        raw = memoryview(data.contiguous().view(torch.uint8).cpu().numpy()).cast("B")
                        _, _, where, expected = native.view(name)
                        if len(raw) != expected:
                            raise ValueError(f"Wrong encoded size for {name}")
                        output.seek(where)
                        output.write(raw)
                    done.update((key, skey))
                    counts["mxfp4" if item["dtype"] == "I8" else "fp8"] += 1
                    counts["quantized_tensors_exactly_unchanged_from_base"] += int(unchanged)
                    if stats is not None:
                        metrics[key] = stats
                    continue
                copy_native = key.startswith("mtp.")
                if copy_native:
                    if key in trained.entries:
                        raise ValueError(f"Unexpected trained MTP weight {key}")
                    counts["original_mtp"] += 1
                    fd, pos = nfd, nstart
                else:
                    train_item, fd, pos, size = trained.view(key)
                    if (
                        train_item["dtype"] == "F32"
                        and item["dtype"] == "BF16"
                        and train_item["shape"] == item["shape"]
                        and key.endswith(
                            (
                                ".engram.k_weight",
                                ".engram.q_weight",
                                ".attn.compressor.norm.weight",
                                ".attn.indexer.k_norm.weight",
                            )
                        )
                    ):
                        w = tensor(trained, key, device)
                        cast = w.bfloat16()
                        if not torch.isfinite(cast).all():
                            raise ValueError(f"Nonfinite native BF16 data for {key}")
                        err = (cast.float() - w).norm().item()
                        metrics[key] = dict(
                            dtype="BF16",
                            source_dtype="F32",
                            elements=w.numel(),
                            relative_l2=err / max(w.norm().item(), 1e-30),
                            rmse=err / math.sqrt(w.numel()),
                            max_abs=(cast.float() - w).abs().max().item(),
                        )
                        output.seek(nstart)
                        output.write(memoryview(cast.contiguous().view(torch.uint8).cpu().numpy()).cast("B"))
                        counts["trained_fp32_to_native_bf16"] += 1
                        done.add(key)
                        continue
                    if (item["dtype"], item["shape"], nsize) != (train_item["dtype"], train_item["shape"], size):
                        raise ValueError(f"Raw tensor schema mismatch {key}")
                    counts["frozen_verified" if is_frozen(key) else "trained_raw"] += 1
                output.seek(nstart)
                changed = 0
                for offset in range(0, nsize, CHUNK):
                    size = min(nsize - offset, CHUNK)
                    buf = read_exact(fd, pos + offset, size)
                    if is_frozen(key) and buf != read_exact(nfd, nstart + offset, size):
                        raise ValueError(f"Frozen asset differs from native 1017: {key}")
                    if sampled(key):
                        import numpy as np

                        values = np.frombuffer(buf, dtype="V2" if item["dtype"] == "BF16" else "V4")
                        other = np.frombuffer(read_exact(nfd, nstart + offset, size), dtype=values.dtype)
                        changed += int((values != other).sum())
                    output.write(buf)
                if sampled(key):
                    metrics[key] = dict(
                        dtype=item["dtype"],
                        elements=math.prod(item["shape"]),
                        changed_trained_elements=changed,
                        max_abs=0.0,
                        rmse=0.0,
                        relative_l2=0.0,
                    )
                done.add(key)
            if done != header.keys():
                raise ValueError(f"Incomplete native shard {filename}")
        size = out_tmp.stat().st_size
        if size != os.fstat(in_fd).st_size:
            raise ValueError(f"Wrong file size {filename}")
        digest = file_digest(out_tmp)
        os.replace(out_tmp, out_final)
        return dict(file=filename, bytes=size, sha256=digest, counts=dict(counts), tensors=len(done), samples=metrics)


def main() -> None:
    import ray

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("native", "trained", "output", "source-provenance"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    native, trained, output = (getattr(args, key).resolve() for key in ("native", "trained", "output"))
    if output.exists():
        raise FileExistsError(output)
    provenance = json.loads(args.source_provenance.read_bytes())
    if Path(provenance["model_path"]).resolve() != trained or not provenance["hashes_verified"]:
        raise ValueError("Input BF16 archive does not match the supplied verified provenance")
    if file_digest(trained / INDEX) != provenance["files_sha256"][INDEX]:
        raise ValueError("BF16 index differs from source provenance")
    base_map = json.loads((native / INDEX).read_bytes())["weight_map"]
    trained_map = json.loads((trained / INDEX).read_bytes())["weight_map"]
    if trained_map.keys() - base_map.keys():
        raise ValueError("BF16 export contains weight names outside original 1017 schema")
    if any(not (name.startswith("mtp.") or name.endswith(".scale")) for name in base_map.keys() - trained_map.keys()):
        raise ValueError("BF16 export missing trained parameters")
    config = json.loads((native / "config.json").read_bytes())
    if config.get("model_type") != "deepseek_v41" or config["quantization_config"]["weight_block_size"] != [32, 32]:
        raise ValueError("Expected 1017 native FP8 32x32 and MXFP4 checkpoint")
    stage = Path(tempfile.mkdtemp(prefix="." + output.name + "-", dir=output.parent))
    logger.info("Native conversion staging %s", stage)
    for entry in native.iterdir():
        if entry.name.startswith(".") or entry.name.endswith(".safetensors"):
            continue
        if entry.is_dir():
            shutil.copytree(entry, stage / entry.name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            shutil.copyfile(entry, stage / entry.name)
    ray.init(address="auto")
    task = ray.remote(num_gpus=1, num_cpus=2, max_retries=0)(shard_task)
    files = sorted(set(base_map.values()), key=lambda s: -(native / s).stat().st_size)
    jobs = [task.remote(str(native), str(trained), str(stage), name, "cuda") for name in files]
    results = []
    try:
        while jobs:
            ready, jobs = ray.wait(jobs, num_returns=1)
            row = ray.get(ready[0])
            results.append(row)
            logger.info("%s/%s complete %s %s bytes", len(results), len(files), row["file"], row["bytes"])
    finally:
        for job in jobs:
            ray.cancel(job, force=True)
        ray.shutdown()
    results.sort(key=lambda row: row["file"])
    counts = Counter()
    for row in results:
        counts.update(row["counts"])
    if sum(r["tensors"] for r in results) != len(base_map):
        raise ValueError("Output shard tensor count differs from the native index")
    report = dict(
        status="complete",
        format="native-mixed-hf",
        model_path=str(output),
        native_path=str(native),
        trained_path=str(trained),
        completed_updates=provenance["completed_updates"],
        source_revision=provenance.get("source_revision"),
        source_provenance=str(args.source_provenance.resolve()),
        source_provenance_sha256=file_digest(args.source_provenance),
        source_bf16_weights_sha256=provenance["verification"]["weights_sha256"],
        native_config_sha256=file_digest(native / "config.json"),
        config_sha256=file_digest(stage / "config.json"),
        native_index_sha256=file_digest(native / INDEX),
        index_sha256=file_digest(stage / INDEX),
        algorithm="Megatron Bridge E2M1 nearest/amax-6 one E8M0 per 32, E4M3FN nearest/amax-448 one E8M0 per 32x32; unchanged blocks reuse original bytes; untrained MTP restored from native base",
        original_tensor_count=len(base_map),
        counts=dict(counts),
        weight_shards=results,
    )
    (stage / "conversion-result.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    stage.chmod(0o755)
    if output.exists():
        raise FileExistsError(output)
    stage.rename(output)
    logger.info("Published %s", output)


if __name__ == "__main__":
    main()
