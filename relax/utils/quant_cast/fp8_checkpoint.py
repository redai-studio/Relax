# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Shared checkpoint streaming and indexing helpers for HF/FP8 export."""

import json
import os
import shutil
import tempfile
import threading
from collections.abc import Iterable
from pathlib import Path

import safetensors.torch
import torch

from relax.utils.logging_utils import get_logger
from relax.utils.quant_cast.fp8 import quantize_hf_tensor, validate_fp8_options


logger = get_logger(__name__)


class ConversionResult:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.weight_map: dict[str, str] = {}
        self.total_size: int = 0
        self.modules_to_not_convert: list[str] = []

    def add_result(
        self,
        filename: str,
        q_weights: dict[str, torch.Tensor],
        module_names: list[str],
    ) -> None:
        with self.lock:
            duplicate_keys = self.weight_map.keys() & q_weights.keys()
            if duplicate_keys:
                raise ValueError(f"Duplicate checkpoint tensors across shards: {sorted(duplicate_keys)}")
            for k, v in q_weights.items():
                self.weight_map[k] = filename
                self.total_size += v.numel() * v.element_size()
            self.modules_to_not_convert.extend(module_names)

    def add_ignored_modules(self, module_names: list[str]) -> None:
        with self.lock:
            self.modules_to_not_convert.extend(module_names)


def write_safetensors_index(output_path: str | Path, result: ConversionResult) -> None:
    index_dict = {
        "metadata": {"total_size": result.total_size},
        "weight_map": result.weight_map,
    }
    with open(Path(output_path) / "model.safetensors.index.json", "w") as f:
        json.dump(index_dict, f, indent=2)


class StreamingHFWriter:
    """Write Bridge tensors in arrival order with a bounded shard buffer.

    The buffer owns compact CPU tensors so slices cannot retain a full PLE
    table or an autograd graph. A single tensor can exceed the shard target.
    """

    _temporary_prefix = "hf"

    def __init__(
        self,
        key_to_filename_map: dict[str, str],
        max_shard_size: int = 4 * 1024**3,
    ) -> None:
        if not key_to_filename_map:
            raise ValueError("Streaming conversion requires a safetensors source checkpoint")
        if max_shard_size <= 0:
            raise ValueError("max_shard_size must be positive")
        self.expected_names = set(key_to_filename_map)
        self.max_shard_size = max_shard_size
        self.result = ConversionResult()

    def _convert_tensor(self, name: str, tensor: torch.Tensor) -> tuple[dict[str, torch.Tensor], list[str]]:
        return {name: tensor}, []

    @torch.no_grad()
    def save_generator(
        self,
        generator: Iterable[tuple[str, torch.Tensor]],
        output_path: str | Path,
        strict: bool = True,
        distributed_save: bool = False,
        save_every_n_ranks: int = 1,
        ignored_source_key_prefixes: Iterable[str] | None = None,
        ignored_source_key_suffixes: Iterable[str] | None = None,
    ) -> None:
        del save_every_n_ranks
        if distributed_save:
            raise ValueError("Streaming conversion does not support Bridge distributed_save")

        ignored_prefixes = tuple(ignored_source_key_prefixes or ())
        ignored_suffixes = tuple(ignored_source_key_suffixes or ())
        expected_names = {
            name
            for name in self.expected_names
            if not name.startswith(ignored_prefixes) and not name.endswith(ignored_suffixes)
        }

        is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        rank = torch.distributed.get_rank() if is_distributed else 0
        if rank != 0:
            for _ in generator:
                pass
            return

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        temporary_dir = Path(tempfile.mkdtemp(prefix=f".{self._temporary_prefix}-export-", dir=output_path))
        cleanup_temporary_dir = True

        buffered_tensors: dict[str, torch.Tensor] = {}
        buffered_storages: set[int] = set()
        buffered_size = 0
        temporary_shards: list[tuple[str, list[str]]] = []
        yielded_names: set[str] = set()
        output_names: set[str] = set()

        def flush() -> None:
            nonlocal buffered_size, buffered_tensors
            if not buffered_tensors:
                return
            filename = f".{self._temporary_prefix}-shard-{len(temporary_shards) + 1:05d}.tmp"
            safetensors.torch.save_file(buffered_tensors, temporary_dir / filename, metadata={"format": "pt"})
            shard_keys = list(buffered_tensors)
            self.result.add_result(filename, buffered_tensors, [])
            temporary_shards.append((filename, shard_keys))
            buffered_tensors = {}
            buffered_storages.clear()
            buffered_size = 0

        try:
            for name, tensor in generator:
                if name in yielded_names:
                    raise ValueError(f"Bridge yielded duplicate tensor: {name}")
                yielded_names.add(name)

                if name not in expected_names:
                    if strict:
                        raise KeyError(f"Bridge tensor '{name}' is missing from the source safetensors index")
                    logger.warning(f"Skipping Bridge tensor missing from the source safetensors index: {name}")
                    del tensor
                    continue

                converted, ignored_modules = self._convert_tensor(name, tensor)
                duplicate_outputs = output_names & converted.keys()
                if duplicate_outputs:
                    raise ValueError(f"Duplicate output tensors: {sorted(duplicate_outputs)}")
                self.result.add_ignored_modules(ignored_modules)
                converted_size = sum(tensor.numel() * tensor.element_size() for tensor in converted.values())
                if buffered_tensors and buffered_size + converted_size > self.max_shard_size:
                    flush()
                # Detach even inputs already on CPU. Clone slices/noncontiguous
                # tensors so buffered bytes reflect the storage retained here.
                for output_name, weight in converted.items():
                    compact = weight.detach().to(device="cpu").contiguous()
                    storage = compact.untyped_storage()
                    if (
                        storage.nbytes() != compact.numel() * compact.element_size()
                        or storage.data_ptr() in buffered_storages
                    ):
                        compact = compact.clone()
                    buffered_storages.add(compact.untyped_storage().data_ptr())
                    buffered_tensors[output_name] = compact
                    del weight, compact, storage
                buffered_size += converted_size
                output_names.update(converted)
                if buffered_size >= self.max_shard_size:
                    flush()
                del tensor, converted

            missing_names = expected_names - yielded_names
            if missing_names:
                logger.warning(f"Bridge did not yield {len(missing_names)} tensors from the source safetensors index")
                if strict:
                    raise KeyError(f"Bridge did not yield {len(missing_names)} tensors required by the source index")

            flush()

            if not self.result.weight_map:
                raise ValueError("Bridge did not produce any checkpoint tensors")

            shard_count = len(temporary_shards)
            staged_shards: list[tuple[Path, Path]] = []
            for shard_idx, (temporary_filename, shard_keys) in enumerate(temporary_shards, start=1):
                if shard_count == 1:
                    final_filename = "model.safetensors"
                else:
                    final_filename = f"model-{shard_idx:05d}-of-{shard_count:05d}.safetensors"
                staged_shards.append((temporary_dir / temporary_filename, output_path / final_filename))
                for key in shard_keys:
                    self.result.weight_map[key] = final_filename
            write_safetensors_index(temporary_dir, self.result)

            existing_files = list(output_path.glob("model*.safetensors"))
            index_path = output_path / "model.safetensors.index.json"
            if index_path.exists():
                existing_files.append(index_path)

            backup_dir = temporary_dir / "previous-checkpoint"
            backup_dir.mkdir()
            backup_pairs = [(backup_dir / path.name, path) for path in existing_files]
            install_pairs = [*staged_shards, (temporary_dir / index_path.name, index_path)]
            try:
                for backup_path, existing_path in backup_pairs:
                    os.replace(existing_path, backup_path)
                for staged_path, final_path in install_pairs:
                    os.replace(staged_path, final_path)
            except BaseException:
                try:
                    for staged_path, installed_path in reversed(install_pairs):
                        if not staged_path.exists():
                            installed_path.unlink(missing_ok=True)
                    for backup_path, original_path in reversed(backup_pairs):
                        if backup_path.exists():
                            os.replace(backup_path, original_path)
                except BaseException:
                    cleanup_temporary_dir = False
                    raise
                raise
        finally:
            if cleanup_temporary_dir:
                shutil.rmtree(temporary_dir, ignore_errors=True)


class StreamingFP8Writer(StreamingHFWriter):
    """Quantize Bridge HF tensors before buffering them for output."""

    _temporary_prefix = "fp8"

    def __init__(
        self,
        key_to_filename_map: dict[str, str],
        strategy: str,
        block_size: list[int] | None,
        device: str | torch.device,
        max_shard_size: int = 4 * 1024**3,
    ) -> None:
        super().__init__(key_to_filename_map, max_shard_size)
        validate_fp8_options(strategy, block_size)
        self.strategy = strategy
        self.block_size = block_size
        self.device = device

    def _convert_tensor(self, name: str, tensor: torch.Tensor) -> tuple[dict[str, torch.Tensor], list[str]]:
        return quantize_hf_tensor(name, tensor, self.strategy, self.block_size, self.device)
