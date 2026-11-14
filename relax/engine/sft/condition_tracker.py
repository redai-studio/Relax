# Copyright (c) 2026 Relax Authors. All Rights Reserved.
from __future__ import annotations

import hashlib
import json
import math
import os
import resource
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from relax.engine.sft.image import image_tier_budget


HOST_STAGES = (
    "ref_image_preprocess_ms",
    "target_image_preprocess_ms",
)
MICRO_GPU_STAGES = (
    "ref_vision_encode_ms",
    "prefix_encode_ms",
    "gen_vision_encode_ms",
    "dit_forward_ms",
    "backward_ms",
)
STEP_GPU_STAGES = ("optimizer_ms",)
ALL_STAGE_NAMES = (
    "data_wait_ms",
    "ref_image_preprocess_ms",
    "ref_vision_encode_ms",
    "prefix_encode_ms",
    "target_image_preprocess_ms",
    "gen_vision_encode_ms",
    "dit_forward_ms",
    "backward_ms",
    "optimizer_ms",
    "train_ms",
)
TRANSFER_KINDS = (
    "ref_pixel_values",
    "ref_grid_hw",
    "prefix_input_ids",
    "target_input_latents",
)
ENCODE_KINDS = (
    "ref_vision_encode",
    "prefix_encode",
    "gen_vision_encode",
)
FEATURE_BYTE_KEYS = (
    "ref_vision_embeds_bytes",
    "past_key_values_bytes",
    "indexes_image_bytes",
)


def _sha256_key(payload: Any) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _read_cpu_memory_bytes() -> tuple[int, int]:
    statm_fields = Path("/proc/self/statm").read_text(encoding="utf-8").split()
    rss_pages = int(statm_fields[1])
    page_size = int(os.sysconf("SC_PAGE_SIZE"))
    current_rss_bytes = rss_pages * page_size
    peak_rss_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    return current_rss_bytes, peak_rss_bytes


def compute_past_key_values_bytes(past_key_values: Any) -> int:
    if past_key_values is None:
        return 0
    if isinstance(past_key_values, torch.Tensor):
        return int(past_key_values.numel() * past_key_values.element_size())
    if hasattr(past_key_values, "layers"):
        total = 0
        for layer in past_key_values.layers:
            keys = getattr(layer, "keys", None)
            values = getattr(layer, "values", None)
            if keys is not None:
                total += int(keys.numel() * keys.element_size())
            if values is not None:
                total += int(values.numel() * values.element_size())
        return total
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        return compute_past_key_values_bytes(past_key_values.key_cache) + compute_past_key_values_bytes(
            past_key_values.value_cache
        )
    if isinstance(past_key_values, (list, tuple)):
        return sum(compute_past_key_values_bytes(item) for item in past_key_values)
    if isinstance(past_key_values, dict):
        return sum(compute_past_key_values_bytes(value) for value in past_key_values.values())
    raise TypeError(f"Unsupported past_key_values type: {type(past_key_values)!r}")


def _summarize_timing_series(
    values: list[float],
    train_mean_ms: float,
    local_repeated_values: list[float] | None = None,
    global_repeated_values: list[float] | None = None,
) -> dict[str, Any]:
    if not values:
        mean_ms = 0.0
        p50_ms = 0.0
        p95_ms = 0.0
    else:
        arr = np.asarray(values, dtype=np.float64)
        mean_ms = float(np.mean(arr))
        p50_ms = float(np.percentile(arr, 50))
        p95_ms = float(np.percentile(arr, 95))
    share_of_step = float(mean_ms / train_mean_ms) if train_mean_ms > 0.0 else 0.0
    result: dict[str, Any] = {
        "mean_ms": mean_ms,
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
        "share_of_step": share_of_step,
        "share_pct": share_of_step * 100.0,
    }
    if local_repeated_values is not None:
        local_rep_mean = float(np.mean(local_repeated_values)) if local_repeated_values else 0.0
        result["repeated_work_ms"] = local_rep_mean
        result["local_repeated_work_ms"] = local_rep_mean
    if global_repeated_values is not None:
        result["global_repeated_work_ms"] = float(np.mean(global_repeated_values)) if global_repeated_values else 0.0
    return result


def _summarize_feature_bytes(values: list[int]) -> dict[str, Any]:
    if not values:
        mean_bytes = 0.0
        max_bytes = 0
    else:
        mean_bytes = float(sum(values) / len(values))
        max_bytes = int(max(values))
    return {
        "count": len(values),
        "mean_bytes": mean_bytes,
        "max_bytes": max_bytes,
        "mean_kib": mean_bytes / 1024.0,
        "max_kib": max_bytes / 1024.0,
        "mean_mib": mean_bytes / (1024.0 * 1024.0),
        "max_mib": max_bytes / (1024.0 * 1024.0),
    }


class ConditionTracker:
    def __init__(
        self,
        rank: int,
        world_size: int,
        micro_batches: int,
        dataset_size: int,
        output_dir: Path,
        warmup_steps: int = 5,
        seed: int | None = None,
    ) -> None:
        if rank < 0 or world_size <= 0 or micro_batches <= 0 or dataset_size <= 0 or warmup_steps < 0:
            raise ValueError("Invalid ConditionTracker configuration")
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.micro_batches = int(micro_batches)
        self.global_batch_size = self.micro_batches * self.world_size
        self.dataset_size = int(dataset_size)
        self.steps_per_epoch = math.ceil(self.dataset_size / self.global_batch_size)
        self.output_dir = Path(output_dir)
        self.warmup_steps = int(warmup_steps)
        self.seed = int(seed) if seed is not None else None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.steps_jsonl_path = self.output_dir / f"rank_{self.rank}_steps.jsonl"
        self.summary_json_path = self.output_dir / f"rank_{self.rank}_summary.json"
        self.steps_jsonl_path.write_text("", encoding="utf-8")

        # 预分配 CUDA Event 事件池，每个 (micro, stage) 预分配两对事件以支持正负双分支调用
        self._cuda_available = torch.cuda.is_available()
        self._use_cuda_events = False
        self._event_pool: dict[tuple[int, str], list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {}
        self._step_event_pool: dict[str, tuple[torch.cuda.Event, torch.cuda.Event]] = {}
        if self._cuda_available:
            for micro in range(self.micro_batches):
                for stage in MICRO_GPU_STAGES:
                    self._event_pool[(micro, stage)] = [
                        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)),
                        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)),
                    ]
            for stage in STEP_GPU_STAGES:
                self._step_event_pool[stage] = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )

        self.max_allocated: int = 0
        self.max_reserved: int = 0
        self.total_samples: int = 0
        self.padding_samples: int = 0
        self.repeated_samples: int = 0
        self.seen_sample_ids: set[int] = set()
        self._current_epoch: int = -1
        self._epoch_seen_sample_ids: set[int] = set()
        self._step_padding_flags: list[bool] | None = None

        self.seen_encode_keys: dict[str, set[str]] = {k: set() for k in ENCODE_KINDS}
        self.encode_stats: dict[str, dict[str, Any]] = {
            "ref_vision_encode": {
                "eligible": None,
                "calls": 0,
                "encoded_images": 0,
                "items": 0,
                "repeated_calls": 0,
                "cache_lookups": 0,
                "cache_hits": 0,
                "cache_misses": 0,
                "cache_hit_rate": 0.0,
            },
            "prefix_encode": {
                "eligible": None,
                "calls": 0,
                "items": 0,
                "repeated_calls": 0,
                "cache_lookups": 0,
                "cache_hits": 0,
                "cache_misses": 0,
                "cache_hit_rate": 0.0,
            },
            "gen_vision_encode": {
                "eligible": False,
                "calls": 0,
                "items": 0,
                "repeated_calls": 0,
                "cache_lookups": 0,
                "cache_hits": 0,
                "cache_misses": 0,
                "cache_hit_rate": 0.0,
            },
        }

        self.seen_transfer_keys: dict[str, set[str]] = {k: set() for k in TRANSFER_KINDS}
        self.transfer_stats: dict[str, dict[str, int]] = {
            k: {
                "total_transfer_calls": 0,
                "total_transfer_bytes": 0,
                "repeated_transfer_calls": 0,
                "repeated_transfer_bytes": 0,
            }
            for k in TRANSFER_KINDS
        }
        self.feature_byte_records: dict[str, list[int]] = {k: [] for k in FEATURE_BYTE_KEYS}
        self.step_records: list[dict[str, Any]] = []

        self.current_step: int = 0
        self.current_device: torch.device = torch.device("cpu")
        self.current_micro: int = 0
        self.current_sample_id: int = 0
        self.current_training_mode: str = "lora"
        self.current_height: int = 0
        self.current_width: int = 0
        self.current_budget: int | None = None
        self.current_is_positive: bool = True
        self.current_ref_image_keys: list[str] = []
        self.current_ref_vision_key: str = ""
        self.current_positive_prefix_kv_key: str = ""
        self.current_prefix_kv_key: str = ""
        self.current_target_image_key: str = ""
        self.current_eligible: bool = True

        self._recorded_micro_event_counts: dict[tuple[int, str], int] = {}
        self._recorded_step_events: set[str] = set()
        self._current_micro_records: dict[int, dict[str, Any]] = {}
        self._current_step_timings: dict[str, float] = {stage: 0.0 for stage in STEP_GPU_STAGES}
        self._current_step_peak_allocated: int | None = None
        self._current_step_peak_reserved: int | None = None

    def _compute_step_padding_flags(self, config: dict[str, Any]) -> list[bool]:
        epoch, offset = divmod(self.current_step, self.steps_per_epoch)
        start = offset * self.global_batch_size
        if start + self.global_batch_size <= self.dataset_size:
            return [False] * self.micro_batches
        if not config.get("dp_tier_micro_balance", False):
            return [
                (start + micro * self.world_size + self.rank) >= self.dataset_size
                for micro in range(self.micro_batches)
            ]
        sampler_seed = self.seed if self.seed is not None else int(config.get("seed") or 42)
        generator = torch.Generator().manual_seed(sampler_seed + epoch)
        order = torch.randperm(self.dataset_size, generator=generator).tolist()
        padded_size = self.steps_per_epoch * self.global_batch_size
        padded_order = (order * math.ceil(padded_size / self.dataset_size))[:padded_size]
        tagged_batch = [
            (padded_order[pos], pos >= self.dataset_size) for pos in range(start, start + self.global_batch_size)
        ]
        ordered = sorted(tagged_batch, key=lambda item: image_tier_budget(config, item[0]) or 0, reverse=True)
        if config.get("dp_tier_rank_rotation", False):
            rot = self.current_step % self.world_size
            if rot:
                ordered = [
                    item
                    for blk in range(0, len(ordered), self.world_size)
                    for item in (
                        ordered[blk + self.world_size - rot : blk + self.world_size]
                        + ordered[blk : blk + self.world_size - rot]
                    )
                ]
        rank_slice = ordered[self.rank :: self.world_size]
        return [bool(is_pad) for _, is_pad in rank_slice]

    def begin_step(self, step: int, device: torch.device) -> None:
        self.current_step = int(step)
        self.current_device = device
        self._use_cuda_events = self._cuda_available and device.type == "cuda"
        epoch = self.current_step // self.steps_per_epoch
        if epoch != self._current_epoch:
            self._current_epoch = epoch
            self._epoch_seen_sample_ids.clear()
        if self._use_cuda_events:
            self.max_allocated = max(self.max_allocated, int(torch.cuda.max_memory_allocated(device)))
            self.max_reserved = max(self.max_reserved, int(torch.cuda.max_memory_reserved(device)))
            torch.cuda.reset_peak_memory_stats(device)
        self.current_micro = 0
        self._step_padding_flags = None
        self._recorded_micro_event_counts.clear()
        self._recorded_step_events.clear()
        self._current_micro_records.clear()
        self._current_step_timings = {stage: 0.0 for stage in STEP_GPU_STAGES}
        self._current_step_peak_allocated = None
        self._current_step_peak_reserved = None

    def begin_micro(
        self,
        micro: int,
        sample_id: int,
        row: dict[str, Any],
        config: dict[str, Any],
        training_mode: str,
        height: int,
        width: int,
    ) -> None:
        self.current_micro = int(micro)
        self.current_sample_id = int(sample_id)
        self.current_training_mode = str(training_mode)
        self.current_height = int(height)
        self.current_width = int(width)
        self.current_is_positive = True
        max_pixels = int(config["max_pixels"])
        budget = image_tier_budget(config, self.current_sample_id)
        self.current_budget = budget

        edit_paths = [str(path) for path in row["edit_image"]]
        num_ref_images = len(edit_paths)
        ref_image_keys = [_sha256_key([path, max_pixels, budget, num_ref_images]) for path in edit_paths]
        ref_vision_key = _sha256_key(ref_image_keys)
        think_mode = bool(config.get("think_mode", False))
        prefix_kv_key = _sha256_key(
            [
                str(row["prompt"]),
                True,
                ref_vision_key,
                self.current_height,
                self.current_width,
                think_mode,
                self.current_training_mode,
            ]
        )
        target_image_key = _sha256_key([str(row["image"]), max_pixels, budget])
        eligible = self.current_training_mode == "lora"

        self.current_ref_image_keys = ref_image_keys
        self.current_ref_vision_key = ref_vision_key
        self.current_positive_prefix_kv_key = prefix_kv_key
        self.current_prefix_kv_key = prefix_kv_key
        self.current_target_image_key = target_image_key
        self.current_eligible = eligible

        self.total_samples += 1
        if self._step_padding_flags is None:
            self._step_padding_flags = self._compute_step_padding_flags(config)
        if self.current_micro < len(self._step_padding_flags):
            is_padding = self._step_padding_flags[self.current_micro]
        else:
            offset = self.current_step % self.steps_per_epoch
            epoch_pos = offset * self.global_batch_size + self.current_micro * self.world_size + self.rank
            is_padding = epoch_pos >= self.dataset_size
        self._epoch_seen_sample_ids.add(self.current_sample_id)
        if is_padding:
            self.padding_samples += 1

        is_repeated_sample = self.current_sample_id in self.seen_sample_ids
        if is_repeated_sample:
            self.repeated_samples += 1
        else:
            self.seen_sample_ids.add(self.current_sample_id)

        self._current_micro_records[self.current_micro] = {
            "micro": self.current_micro,
            "sample_id": self.current_sample_id,
            "training_mode": self.current_training_mode,
            "eligible": eligible,
            "height": self.current_height,
            "width": self.current_width,
            "fixed_pixels_budget": budget,
            "is_padding": is_padding,
            "is_repeated_sample": is_repeated_sample,
            "ref_image_keys": ref_image_keys,
            "ref_vision_key": ref_vision_key,
            "prefix_kv_key": prefix_kv_key,
            "target_image_key": target_image_key,
            "ref_image_repeated": False,
            "ref_vision_repeated": False,
            "prefix_kv_repeated": False,
            "target_image_repeated": False,
            "transfers": {
                k: {"calls": 0, "bytes": 0, "repeated_calls": 0, "repeated_bytes": 0} for k in TRANSFER_KINDS
            },
            "encodes": {},
            "feature_bytes": {},
            "timings_ms": {stage: 0.0 for stage in (*HOST_STAGES, *MICRO_GPU_STAGES)},
        }

    def set_prefix_context(self, prompt: str, is_positive: bool, think_mode: bool) -> None:
        self.current_is_positive = bool(is_positive)
        key = _sha256_key(
            [
                str(prompt),
                self.current_is_positive,
                self.current_ref_vision_key,
                self.current_height,
                self.current_width,
                bool(think_mode),
                self.current_training_mode,
            ]
        )
        self.current_prefix_kv_key = key
        if self.current_is_positive:
            self.current_positive_prefix_kv_key = key
            micro_rec = self._current_micro_records.get(self.current_micro)
            if micro_rec is not None:
                micro_rec["prefix_kv_key"] = key

    @staticmethod
    def compute_kv_bytes(past_key_values: Any) -> int:
        return compute_past_key_values_bytes(past_key_values)

    @contextmanager
    def host_stage(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            micro_rec = self._current_micro_records.get(self.current_micro)
            if micro_rec is not None:
                micro_rec["timings_ms"][name] = micro_rec["timings_ms"].get(name, 0.0) + elapsed_ms

    @contextmanager
    def gpu_stage(self, name: str) -> Iterator[None]:
        if self._use_cuda_events:
            if name in self._step_event_pool:
                start_ev, end_ev = self._step_event_pool[name]
                start_ev.record()
                try:
                    yield
                finally:
                    end_ev.record()
                    self._recorded_step_events.add(name)
            else:
                key = (self.current_micro, name)
                call_idx = self._recorded_micro_event_counts.get(key, 0)
                pool_list = self._event_pool.setdefault(key, [])
                while len(pool_list) <= call_idx:
                    pool_list.append(
                        (
                            torch.cuda.Event(enable_timing=True),
                            torch.cuda.Event(enable_timing=True),
                        )
                    )
                start_ev, end_ev = pool_list[call_idx]
                start_ev.record()
                try:
                    yield
                finally:
                    end_ev.record()
                    self._recorded_micro_event_counts[key] = call_idx + 1
        else:
            t0 = time.perf_counter()
            try:
                yield
            finally:
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                if name in STEP_GPU_STAGES:
                    self._current_step_timings[name] = self._current_step_timings.get(name, 0.0) + elapsed_ms
                else:
                    micro_rec = self._current_micro_records.get(self.current_micro)
                    if micro_rec is not None:
                        micro_rec["timings_ms"][name] = micro_rec["timings_ms"].get(name, 0.0) + elapsed_ms

    def record_transfer(self, kind: str, tensor: torch.Tensor, image_index: int = 0, key: str | None = None) -> None:
        if kind not in self.transfer_stats:
            raise ValueError(f"Unknown transfer kind: {kind!r}")
        if key is None:
            if kind in ("ref_pixel_values", "ref_grid_hw"):
                key = (
                    self.current_ref_image_keys[image_index]
                    if image_index < len(self.current_ref_image_keys)
                    else f"{self.current_ref_vision_key}:{image_index}"
                )
            elif kind == "prefix_input_ids":
                key = self.current_prefix_kv_key
            elif kind == "target_input_latents":
                key = self.current_target_image_key

        nbytes = int(tensor.numel() * tensor.element_size())
        is_repeat = bool(key) and (key in self.seen_transfer_keys[kind])
        if key:
            self.seen_transfer_keys[kind].add(key)

        stats = self.transfer_stats[kind]
        stats["total_transfer_calls"] += 1
        stats["total_transfer_bytes"] += nbytes
        if is_repeat:
            stats["repeated_transfer_calls"] += 1
            stats["repeated_transfer_bytes"] += nbytes

        micro_rec = self._current_micro_records.get(self.current_micro)
        if micro_rec is not None:
            m_tr = micro_rec["transfers"][kind]
            m_tr["calls"] += 1
            m_tr["bytes"] += nbytes
            if is_repeat:
                m_tr["repeated_calls"] += 1
                m_tr["repeated_bytes"] += nbytes
                if kind == "ref_pixel_values":
                    micro_rec["ref_image_repeated"] = True
                elif kind == "target_input_latents":
                    micro_rec["target_image_repeated"] = True

    def record_encode(
        self,
        kind: str,
        num_items: int,
        output_bytes: int = 0,
        key: str | None = None,
        eligible: bool | None = None,
        indexes_image_bytes: int = 0,
    ) -> None:
        if kind not in self.encode_stats:
            raise ValueError(f"Unknown encode kind: {kind!r}")
        if kind == "gen_vision_encode":
            resolved_eligible = False
            resolved_key = None
            is_repeat = False
        else:
            resolved_eligible = self.current_eligible if eligible is None else bool(eligible)
            if key is not None:
                resolved_key = key
            elif kind == "ref_vision_encode":
                resolved_key = self.current_ref_vision_key
            else:
                resolved_key = self.current_prefix_kv_key
            is_repeat = bool(resolved_key) and (resolved_key in self.seen_encode_keys[kind])
            if resolved_key:
                self.seen_encode_keys[kind].add(resolved_key)

        stats = self.encode_stats[kind]
        stats["eligible"] = resolved_eligible
        stats["calls"] += 1
        stats["items"] += int(num_items)
        if kind == "ref_vision_encode":
            stats["encoded_images"] += int(num_items)
        if is_repeat:
            stats["repeated_calls"] += 1

        out_bytes = int(output_bytes)
        idx_bytes = int(indexes_image_bytes)
        if kind == "ref_vision_encode" and out_bytes > 0:
            self.feature_byte_records["ref_vision_embeds_bytes"].append(out_bytes)
        elif kind == "prefix_encode":
            if out_bytes > 0:
                self.feature_byte_records["past_key_values_bytes"].append(out_bytes)
            if idx_bytes > 0:
                self.feature_byte_records["indexes_image_bytes"].append(idx_bytes)

        micro_rec = self._current_micro_records.get(self.current_micro)
        if micro_rec is not None:
            enc_entry = micro_rec["encodes"].get(kind)
            if enc_entry is None:
                enc_entry = {
                    "calls": 0,
                    "items": 0,
                    "repeated_calls": 0,
                    "repeated": False,
                    "eligible": resolved_eligible,
                    "key": resolved_key,
                    "keys": [],
                    "output_bytes": 0,
                }
                micro_rec["encodes"][kind] = enc_entry
            first_call_in_micro = enc_entry["calls"] == 0
            enc_entry["calls"] += 1
            enc_entry["items"] += int(num_items)
            if is_repeat:
                enc_entry["repeated_calls"] += 1
            if first_call_in_micro or self.current_is_positive:
                enc_entry["repeated"] = is_repeat if first_call_in_micro else (enc_entry["repeated"] or is_repeat)
                enc_entry["key"] = resolved_key
            if resolved_key is not None:
                enc_entry["keys"].append(resolved_key)
            enc_entry["output_bytes"] += out_bytes

            if kind == "ref_vision_encode":
                if first_call_in_micro:
                    micro_rec["ref_vision_repeated"] = is_repeat
                if out_bytes > 0 and (first_call_in_micro or self.current_is_positive):
                    micro_rec["feature_bytes"]["ref_vision_embeds_bytes"] = out_bytes
            elif kind == "prefix_encode":
                if self.current_is_positive or first_call_in_micro:
                    micro_rec["prefix_kv_repeated"] = is_repeat
                if out_bytes > 0 and (self.current_is_positive or first_call_in_micro):
                    micro_rec["feature_bytes"]["past_key_values_bytes"] = out_bytes
                if idx_bytes > 0 and (self.current_is_positive or first_call_in_micro):
                    micro_rec["feature_bytes"]["indexes_image_bytes"] = idx_bytes

    def consume_step_memory(self, device: torch.device) -> tuple[int, int]:
        if self._cuda_available and device.type == "cuda":
            step_alloc = int(torch.cuda.max_memory_allocated(device))
            step_reserved = int(torch.cuda.max_memory_reserved(device))
        else:
            step_alloc = 0
            step_reserved = 0
        self._current_step_peak_allocated = step_alloc
        self._current_step_peak_reserved = step_reserved
        self.max_allocated = max(self.max_allocated, step_alloc)
        self.max_reserved = max(self.max_reserved, step_reserved)
        return self.max_allocated, self.max_reserved

    def end_step(self, step: int, data_wait_seconds: float, train_seconds: float) -> None:
        if int(step) != self.current_step:
            raise ValueError(f"Expected step {self.current_step}, received {step}")
        if self._current_step_peak_allocated is None or self._current_step_peak_reserved is None:
            self.consume_step_memory(self.current_device)

        if self._use_cuda_events:
            for (micro, stage_name), call_count in sorted(self._recorded_micro_event_counts.items()):
                pool_list = self._event_pool[(micro, stage_name)]
                total_elapsed_ms = 0.0
                for call_idx in range(call_count):
                    start_ev, end_ev = pool_list[call_idx]
                    if not end_ev.query():
                        end_ev.synchronize()
                    total_elapsed_ms += float(start_ev.elapsed_time(end_ev))
                if micro in self._current_micro_records:
                    self._current_micro_records[micro]["timings_ms"][stage_name] = total_elapsed_ms
            for stage_name in sorted(self._recorded_step_events):
                start_ev, end_ev = self._step_event_pool[stage_name]
                if not end_ev.query():
                    end_ev.synchronize()
                self._current_step_timings[stage_name] = float(start_ev.elapsed_time(end_ev))

        micro_list = [self._current_micro_records[m] for m in sorted(self._current_micro_records)]
        stage_timings_ms: dict[str, float] = {
            "data_wait_ms": float(data_wait_seconds) * 1000.0,
        }
        for stage in (*HOST_STAGES, *MICRO_GPU_STAGES):
            stage_timings_ms[stage] = float(sum(m["timings_ms"].get(stage, 0.0) for m in micro_list))
        for stage in STEP_GPU_STAGES:
            stage_timings_ms[stage] = float(self._current_step_timings.get(stage, 0.0))
        stage_timings_ms["train_ms"] = float(train_seconds) * 1000.0

        ref_pre_rep_ms = float(
            sum(m["timings_ms"].get("ref_image_preprocess_ms", 0.0) for m in micro_list if m["ref_image_repeated"])
        )
        ref_vis_rep_ms = 0.0
        prefix_rep_ms = 0.0
        for m in micro_list:
            vis_enc = m["encodes"].get("ref_vision_encode")
            if vis_enc and vis_enc["calls"] > 0:
                ref_vis_rep_ms += float(m["timings_ms"].get("ref_vision_encode_ms", 0.0)) * (
                    vis_enc["repeated_calls"] / vis_enc["calls"]
                )
            elif m["ref_vision_repeated"]:
                ref_vis_rep_ms += float(m["timings_ms"].get("ref_vision_encode_ms", 0.0))

            pref_enc = m["encodes"].get("prefix_encode")
            if pref_enc and pref_enc["calls"] > 0:
                prefix_rep_ms += float(m["timings_ms"].get("prefix_encode_ms", 0.0)) * (
                    pref_enc["repeated_calls"] / pref_enc["calls"]
                )
            elif m["prefix_kv_repeated"]:
                prefix_rep_ms += float(m["timings_ms"].get("prefix_encode_ms", 0.0))
        target_pre_rep_ms = float(
            sum(
                m["timings_ms"].get("target_image_preprocess_ms", 0.0)
                for m in micro_list
                if m["target_image_repeated"]
            )
        )
        repeated_stage_timings_ms = {
            "ref_image_preprocess_ms": ref_pre_rep_ms,
            "ref_vision_encode_ms": ref_vis_rep_ms,
            "prefix_encode_ms": prefix_rep_ms,
            "target_image_preprocess_ms": target_pre_rep_ms,
            "condition_repeated_ms": ref_pre_rep_ms + ref_vis_rep_ms + prefix_rep_ms,
            "train_ms": ref_pre_rep_ms + ref_vis_rep_ms + prefix_rep_ms + target_pre_rep_ms,
        }

        step_transfers = {
            k: {
                "calls": sum(m["transfers"][k]["calls"] for m in micro_list),
                "bytes": sum(m["transfers"][k]["bytes"] for m in micro_list),
                "repeated_calls": sum(m["transfers"][k]["repeated_calls"] for m in micro_list),
                "repeated_bytes": sum(m["transfers"][k]["repeated_bytes"] for m in micro_list),
            }
            for k in TRANSFER_KINDS
        }
        cpu_rss_bytes, cpu_peak_rss_bytes = _read_cpu_memory_bytes()
        step_record = {
            "rank": self.rank,
            "world_size": self.world_size,
            "step": self.current_step,
            "epoch": self.current_step // self.steps_per_epoch,
            "in_perf_window": self.current_step >= self.warmup_steps,
            "sample_ids": [m["sample_id"] for m in micro_list],
            "data_wait_seconds": float(data_wait_seconds),
            "train_seconds": float(train_seconds),
            "stage_timings_ms": stage_timings_ms,
            "repeated_stage_timings_ms": repeated_stage_timings_ms,
            "transfers": step_transfers,
            "memory": {
                "step_peak_allocated_bytes": int(self._current_step_peak_allocated or 0),
                "step_peak_reserved_bytes": int(self._current_step_peak_reserved or 0),
                "lifetime_peak_allocated_bytes": int(self.max_allocated),
                "lifetime_peak_reserved_bytes": int(self.max_reserved),
                "cpu_rss_bytes": int(cpu_rss_bytes),
                "cpu_peak_rss_bytes": int(cpu_peak_rss_bytes),
            },
            "micros": micro_list,
        }
        self.step_records.append(step_record)
        with self.steps_jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(step_record, ensure_ascii=False) + "\n")

    def build_summary(self) -> dict[str, Any]:
        perf_records = [r for r in self.step_records if r["step"] >= self.warmup_steps]
        window_records = perf_records if perf_records else self.step_records

        encoding_summary: dict[str, Any] = {}
        for kind in ENCODE_KINDS:
            st = dict(self.encode_stats[kind])
            calls = int(st["calls"])
            repeated = int(st["repeated_calls"])
            st["local_repeat_rate"] = float(repeated / calls) if calls > 0 else 0.0
            encoding_summary[kind] = st

        transfer_summary: dict[str, Any] = {}
        tot_calls = 0
        tot_bytes = 0
        tot_rep_calls = 0
        tot_rep_bytes = 0
        for kind in TRANSFER_KINDS:
            st = self.transfer_stats[kind]
            c = int(st["total_transfer_calls"])
            b = int(st["total_transfer_bytes"])
            rc = int(st["repeated_transfer_calls"])
            rb = int(st["repeated_transfer_bytes"])
            tot_calls += c
            tot_bytes += b
            tot_rep_calls += rc
            tot_rep_bytes += rb
            transfer_summary[kind] = {
                "total_transfer_calls": c,
                "total_transfer_bytes": b,
                "total_transfer_mib": b / (1024.0 * 1024.0),
                "repeated_transfer_calls": rc,
                "repeated_transfer_bytes": rb,
                "repeated_transfer_mib": rb / (1024.0 * 1024.0),
                "repeat_byte_rate": float(rb / b) if b > 0 else 0.0,
            }
        transfer_summary["total"] = {
            "total_transfer_calls": tot_calls,
            "total_transfer_bytes": tot_bytes,
            "total_transfer_mib": tot_bytes / (1024.0 * 1024.0),
            "repeated_transfer_calls": tot_rep_calls,
            "repeated_transfer_bytes": tot_rep_bytes,
            "repeated_transfer_mib": tot_rep_bytes / (1024.0 * 1024.0),
            "repeat_byte_rate": float(tot_rep_bytes / tot_bytes) if tot_bytes > 0 else 0.0,
        }

        train_values = [float(r["stage_timings_ms"]["train_ms"]) for r in window_records]
        train_mean_ms = float(np.mean(train_values)) if train_values else 0.0
        repeatable_stages = {
            "ref_image_preprocess_ms",
            "ref_vision_encode_ms",
            "prefix_encode_ms",
            "target_image_preprocess_ms",
            "train_ms",
        }
        stage_summary: dict[str, Any] = {}
        for stage in ALL_STAGE_NAMES:
            vals = [float(r["stage_timings_ms"].get(stage, 0.0)) for r in window_records]
            rep_vals = (
                [float(r["repeated_stage_timings_ms"].get(stage, 0.0)) for r in window_records]
                if stage in repeatable_stages
                else None
            )
            stage_summary[stage] = _summarize_timing_series(vals, train_mean_ms, local_repeated_values=rep_vals)

        feature_summary = {key: _summarize_feature_bytes(self.feature_byte_records[key]) for key in FEATURE_BYTE_KEYS}

        step_alloc_vals = [int(r["memory"]["step_peak_allocated_bytes"]) for r in window_records]
        step_res_vals = [int(r["memory"]["step_peak_reserved_bytes"]) for r in window_records]
        cpu_rss_bytes, cpu_peak_rss_bytes = _read_cpu_memory_bytes()
        if self.step_records:
            cpu_peak_rss_bytes = max(
                cpu_peak_rss_bytes,
                max(int(r["memory"]["cpu_peak_rss_bytes"]) for r in self.step_records),
            )

        step_alloc_mean = float(np.mean(step_alloc_vals)) if step_alloc_vals else 0.0
        step_alloc_max = int(max(step_alloc_vals)) if step_alloc_vals else 0
        step_res_mean = float(np.mean(step_res_vals)) if step_res_vals else 0.0
        step_res_max = int(max(step_res_vals)) if step_res_vals else 0
        gib = float(1024**3)

        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "micro_batches": self.micro_batches,
            "global_batch_size": self.global_batch_size,
            "dataset_size": self.dataset_size,
            "warmup_steps": self.warmup_steps,
            "total_steps": len(self.step_records),
            "perf_window_steps": len(perf_records),
            "sample_stats": {
                "total_samples": self.total_samples,
                "unique_rows": len(self.seen_sample_ids),
                "padding_samples": self.padding_samples,
                "repeated_samples": self.repeated_samples,
                "local_sample_repeat_rate": (
                    float(self.repeated_samples / self.total_samples) if self.total_samples > 0 else 0.0
                ),
            },
            "encoding_stats": encoding_summary,
            "transfer_stats": transfer_summary,
            "stage_timings_ms": stage_summary,
            "feature_tensor_bytes": feature_summary,
            "memory_stats": {
                "step_peak_allocated_mean_bytes": step_alloc_mean,
                "step_peak_allocated_max_bytes": step_alloc_max,
                "step_peak_allocated_mean_gib": step_alloc_mean / gib,
                "step_peak_allocated_max_gib": step_alloc_max / gib,
                "step_peak_reserved_mean_bytes": step_res_mean,
                "step_peak_reserved_max_bytes": step_res_max,
                "step_peak_reserved_mean_gib": step_res_mean / gib,
                "step_peak_reserved_max_gib": step_res_max / gib,
                "lifetime_peak_allocated_bytes": int(self.max_allocated),
                "lifetime_peak_allocated_gib": int(self.max_allocated) / gib,
                "lifetime_peak_reserved_bytes": int(self.max_reserved),
                "lifetime_peak_reserved_gib": int(self.max_reserved) / gib,
                "cpu_rss_bytes": int(cpu_rss_bytes),
                "cpu_peak_rss_bytes": int(cpu_peak_rss_bytes),
                "cpu_peak_rss_gib": int(cpu_peak_rss_bytes) / gib,
            },
        }

    def save_summary(self) -> dict[str, Any]:
        summary = self.build_summary()
        self.summary_json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return summary


def summarize_baseline_dir(output_dir: Path) -> dict[str, Any]:
    out_path = Path(output_dir)
    step_files = sorted(out_path.glob("rank_*_steps.jsonl"))
    if not step_files:
        raise FileNotFoundError(f"No rank_*_steps.jsonl files found under {out_path}")

    rank_step_records: dict[int, list[dict[str, Any]]] = {}
    for file_path in step_files:
        records = [json.loads(line) for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if records:
            rank = int(records[0]["rank"])
            rank_step_records[rank] = records

    if not rank_step_records:
        raise ValueError(f"Step log files under {out_path} are empty")

    rank_summaries: dict[int, dict[str, Any]] = {}
    for file_path in sorted(out_path.glob("rank_*_summary.json")):
        data = json.loads(file_path.read_text(encoding="utf-8"))
        rank_summaries[int(data["rank"])] = data

    ranks = sorted(rank_step_records)
    first_rank_steps = rank_step_records[ranks[0]]
    world_size = int(first_rank_steps[0]["world_size"])
    warmup_steps = int(
        rank_summaries[ranks[0]]["warmup_steps"]
        if ranks[0] in rank_summaries
        else next((r["step"] for r in first_rank_steps if r.get("in_perf_window")), 5)
    )
    dataset_size = int(rank_summaries[ranks[0]]["dataset_size"]) if ranks[0] in rank_summaries else 0
    micro_batches = len(first_rank_steps[0]["micros"])
    global_batch_size = micro_batches * world_size
    steps_per_epoch = math.ceil(dataset_size / global_batch_size) if dataset_size > 0 else 1

    # 按 (step, micro, rank) 顺序遍历全量 micro 记录，统计本地与跨 rank 重复
    all_micro_events: list[tuple[int, int, int, dict[str, Any]]] = []
    for rank in ranks:
        for step_rec in rank_step_records[rank]:
            step_idx = int(step_rec["step"])
            for micro_rec in step_rec["micros"]:
                all_micro_events.append((step_idx, int(micro_rec["micro"]), rank, micro_rec))
    all_micro_events.sort(key=lambda item: (item[0], item[1], item[2]))

    total_samples = len(all_micro_events)
    global_seen_samples: set[int] = set()
    rank_seen_samples: dict[int, set[int]] = {r: set() for r in ranks}
    epoch_seen_samples: dict[int, set[int]] = {}
    local_repeated_samples = 0
    global_repeated_samples = 0
    padding_samples = 0

    global_seen_encode_keys: dict[str, set[str]] = {k: set() for k in ENCODE_KINDS}
    encode_agg: dict[str, dict[str, Any]] = {
        kind: {
            "eligible": False if kind == "gen_vision_encode" else None,
            "calls": 0,
            "items": 0,
            "local_repeated_calls": 0,
            "global_repeated_calls": 0,
            "cache_lookups": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_hit_rate": 0.0,
        }
        for kind in ENCODE_KINDS
    }
    encode_agg["ref_vision_encode"]["encoded_images"] = 0

    global_seen_transfer_keys: dict[str, set[str]] = {k: set() for k in TRANSFER_KINDS}
    transfer_agg: dict[str, dict[str, int]] = {
        k: {
            "total_transfer_calls": 0,
            "total_transfer_bytes": 0,
            "local_repeated_transfer_calls": 0,
            "local_repeated_transfer_bytes": 0,
            "global_repeated_transfer_calls": 0,
            "global_repeated_transfer_bytes": 0,
        }
        for k in TRANSFER_KINDS
    }

    feature_bytes_all: dict[str, list[int]] = {k: [] for k in FEATURE_BYTE_KEYS}

    first_seen_prefix_local: dict[tuple[int, str], dict[str, Any]] = {}
    first_seen_prefix_global: dict[str, dict[str, Any]] = {}
    local_prefix_repeat_count = 0
    global_prefix_repeat_count = 0
    local_repeat_evidence: list[dict[str, Any]] = []
    global_repeat_evidence: list[dict[str, Any]] = []

    # 记录每个 (rank, step) 的全局重复工作耗时
    global_repeated_step_timings: dict[tuple[int, int], dict[str, float]] = {}

    for step_idx, micro_idx, rank, micro_rec in all_micro_events:
        sample_id = int(micro_rec["sample_id"])
        epoch = step_idx // steps_per_epoch if steps_per_epoch > 0 else 0
        epoch_set = epoch_seen_samples.setdefault(epoch, set())
        is_pad = bool(micro_rec["is_padding"]) if "is_padding" in micro_rec else (sample_id in epoch_set)
        if is_pad:
            padding_samples += 1
        epoch_set.add(sample_id)

        if sample_id in rank_seen_samples[rank]:
            local_repeated_samples += 1
        else:
            rank_seen_samples[rank].add(sample_id)

        if sample_id in global_seen_samples:
            global_repeated_samples += 1
        else:
            global_seen_samples.add(sample_id)

        for fkey in FEATURE_BYTE_KEYS:
            if fkey in micro_rec.get("feature_bytes", {}):
                feature_bytes_all[fkey].append(int(micro_rec["feature_bytes"][fkey]))

        step_glob_rep = global_repeated_step_timings.setdefault(
            (rank, step_idx),
            {
                "ref_image_preprocess_ms": 0.0,
                "ref_vision_encode_ms": 0.0,
                "prefix_encode_ms": 0.0,
                "target_image_preprocess_ms": 0.0,
                "train_ms": 0.0,
            },
        )

        # 转移字节聚合
        ref_image_keys = micro_rec.get("ref_image_keys", [])
        ref_img_glob_rep = any(k in global_seen_transfer_keys["ref_pixel_values"] for k in ref_image_keys)
        for k in ref_image_keys:
            global_seen_transfer_keys["ref_pixel_values"].add(k)
            global_seen_transfer_keys["ref_grid_hw"].add(k)

        prefix_key = micro_rec.get("prefix_kv_key", "")
        prefix_tr_glob_rep = bool(prefix_key) and (prefix_key in global_seen_transfer_keys["prefix_input_ids"])
        if prefix_key:
            global_seen_transfer_keys["prefix_input_ids"].add(prefix_key)

        target_key = micro_rec.get("target_image_key", "")
        target_tr_glob_rep = bool(target_key) and (target_key in global_seen_transfer_keys["target_input_latents"])
        if target_key:
            global_seen_transfer_keys["target_input_latents"].add(target_key)

        glob_rep_by_kind = {
            "ref_pixel_values": ref_img_glob_rep,
            "ref_grid_hw": ref_img_glob_rep,
            "prefix_input_ids": prefix_tr_glob_rep,
            "target_input_latents": target_tr_glob_rep,
        }
        for tkind in TRANSFER_KINDS:
            tr_info = micro_rec["transfers"][tkind]
            c = int(tr_info["calls"])
            b = int(tr_info["bytes"])
            rc = int(tr_info["repeated_calls"])
            rb = int(tr_info["repeated_bytes"])
            agg = transfer_agg[tkind]
            agg["total_transfer_calls"] += c
            agg["total_transfer_bytes"] += b
            agg["local_repeated_transfer_calls"] += rc
            agg["local_repeated_transfer_bytes"] += rb
            if glob_rep_by_kind[tkind]:
                agg["global_repeated_transfer_calls"] += c
                agg["global_repeated_transfer_bytes"] += b

        if ref_img_glob_rep:
            t_ms = float(micro_rec["timings_ms"].get("ref_image_preprocess_ms", 0.0))
            step_glob_rep["ref_image_preprocess_ms"] += t_ms
            step_glob_rep["train_ms"] += t_ms
        if target_tr_glob_rep:
            t_ms = float(micro_rec["timings_ms"].get("target_image_preprocess_ms", 0.0))
            step_glob_rep["target_image_preprocess_ms"] += t_ms
            step_glob_rep["train_ms"] += t_ms

        # 编码调用聚合与证据收集
        for ekind in ENCODE_KINDS:
            enc_info = micro_rec.get("encodes", {}).get(ekind)
            if enc_info is None:
                continue
            agg = encode_agg[ekind]
            agg["eligible"] = bool(enc_info["eligible"])
            calls = int(enc_info["calls"])
            items = int(enc_info["items"])
            agg["calls"] += calls
            agg["items"] += items
            if ekind == "ref_vision_encode":
                agg["encoded_images"] += items
            local_rep_calls = int(
                enc_info["repeated_calls"]
                if "repeated_calls" in enc_info
                else (calls if enc_info.get("repeated", False) else 0)
            )
            agg["local_repeated_calls"] += local_rep_calls

            if ekind in ("ref_vision_encode", "prefix_encode"):
                ekeys = enc_info.get("keys")
                if not ekeys:
                    ekeys = [enc_info["key"]] if enc_info.get("key") else []
                glob_rep_in_micro = 0
                for ekey in ekeys:
                    if ekey in global_seen_encode_keys[ekind]:
                        glob_rep_in_micro += 1
                    global_seen_encode_keys[ekind].add(ekey)
                if glob_rep_in_micro > 0:
                    agg["global_repeated_calls"] += glob_rep_in_micro
                    stage_key = "ref_vision_encode_ms" if ekind == "ref_vision_encode" else "prefix_encode_ms"
                    t_ms = float(micro_rec["timings_ms"].get(stage_key, 0.0)) * (
                        glob_rep_in_micro / max(len(ekeys), 1)
                    )
                    step_glob_rep[stage_key] += t_ms
                    step_glob_rep["train_ms"] += t_ms

        if prefix_key:
            occurrence_meta = {
                "rank": rank,
                "step": step_idx,
                "micro": micro_idx,
                "sample_id": sample_id,
                "is_padding": bool(micro_rec.get("is_padding", False)),
                "prefix_kv_key": prefix_key,
                "ref_vision_key": micro_rec.get("ref_vision_key", ""),
                "ref_h2d_bytes": int(
                    micro_rec["transfers"]["ref_pixel_values"]["bytes"]
                    + micro_rec["transfers"]["ref_grid_hw"]["bytes"]
                    + micro_rec["transfers"]["prefix_input_ids"]["bytes"]
                ),
                "ref_h2d_mib": float(
                    (
                        micro_rec["transfers"]["ref_pixel_values"]["bytes"]
                        + micro_rec["transfers"]["ref_grid_hw"]["bytes"]
                        + micro_rec["transfers"]["prefix_input_ids"]["bytes"]
                    )
                    / (1024.0 * 1024.0)
                ),
                "ref_image_preprocess_ms": float(micro_rec["timings_ms"].get("ref_image_preprocess_ms", 0.0)),
                "ref_vision_encode_ms": float(micro_rec["timings_ms"].get("ref_vision_encode_ms", 0.0)),
                "prefix_encode_ms": float(micro_rec["timings_ms"].get("prefix_encode_ms", 0.0)),
            }
            local_pair = (rank, prefix_key)
            if local_pair in first_seen_prefix_local:
                local_prefix_repeat_count += 1
                if len(local_repeat_evidence) < 16:
                    local_repeat_evidence.append(
                        {
                            "first": first_seen_prefix_local[local_pair],
                            "repeat": occurrence_meta,
                        }
                    )
            else:
                first_seen_prefix_local[local_pair] = occurrence_meta

            if prefix_key in first_seen_prefix_global:
                global_prefix_repeat_count += 1
                if len(global_repeat_evidence) < 16:
                    global_repeat_evidence.append(
                        {
                            "first": first_seen_prefix_global[prefix_key],
                            "repeat": occurrence_meta,
                        }
                    )
            else:
                first_seen_prefix_global[prefix_key] = occurrence_meta

    for ekind, agg in encode_agg.items():
        calls = int(agg["calls"])
        agg["repeated_calls"] = int(agg["local_repeated_calls"])
        agg["local_repeat_rate"] = float(agg["local_repeated_calls"] / calls) if calls > 0 else 0.0
        agg["global_repeat_rate"] = float(agg["global_repeated_calls"] / calls) if calls > 0 else 0.0

    transfer_summary: dict[str, Any] = {}
    tot_tr = {
        "total_transfer_calls": 0,
        "total_transfer_bytes": 0,
        "local_repeated_transfer_calls": 0,
        "local_repeated_transfer_bytes": 0,
        "global_repeated_transfer_calls": 0,
        "global_repeated_transfer_bytes": 0,
    }
    for tkind in TRANSFER_KINDS:
        agg = transfer_agg[tkind]
        for k in tot_tr:
            tot_tr[k] += int(agg[k])
        b = int(agg["total_transfer_bytes"])
        lb = int(agg["local_repeated_transfer_bytes"])
        gb = int(agg["global_repeated_transfer_bytes"])
        transfer_summary[tkind] = {
            **agg,
            "repeated_transfer_calls": int(agg["local_repeated_transfer_calls"]),
            "repeated_transfer_bytes": lb,
            "total_transfer_mib": b / (1024.0 * 1024.0),
            "repeated_transfer_mib": lb / (1024.0 * 1024.0),
            "local_repeated_transfer_mib": lb / (1024.0 * 1024.0),
            "global_repeated_transfer_mib": gb / (1024.0 * 1024.0),
            "repeat_byte_rate": float(lb / b) if b > 0 else 0.0,
            "local_repeat_byte_rate": float(lb / b) if b > 0 else 0.0,
            "global_repeat_byte_rate": float(gb / b) if b > 0 else 0.0,
        }
    tb = int(tot_tr["total_transfer_bytes"])
    tlb = int(tot_tr["local_repeated_transfer_bytes"])
    tgb = int(tot_tr["global_repeated_transfer_bytes"])
    transfer_summary["total"] = {
        **tot_tr,
        "repeated_transfer_calls": int(tot_tr["local_repeated_transfer_calls"]),
        "repeated_transfer_bytes": tlb,
        "total_transfer_mib": tb / (1024.0 * 1024.0),
        "repeated_transfer_mib": tlb / (1024.0 * 1024.0),
        "local_repeated_transfer_mib": tlb / (1024.0 * 1024.0),
        "global_repeated_transfer_mib": tgb / (1024.0 * 1024.0),
        "repeat_byte_rate": float(tlb / tb) if tb > 0 else 0.0,
        "local_repeat_byte_rate": float(tlb / tb) if tb > 0 else 0.0,
        "global_repeat_byte_rate": float(tgb / tb) if tb > 0 else 0.0,
    }

    # 聚合性能窗口阶段耗时：按 step 取跨 rank 均值序列后计算 Mean / P50 / P95
    all_steps_by_idx: dict[int, list[dict[str, Any]]] = {}
    for rank in ranks:
        for rec in rank_step_records[rank]:
            all_steps_by_idx.setdefault(int(rec["step"]), []).append(rec)
    sorted_step_indices = sorted(all_steps_by_idx)
    perf_step_indices = [s for s in sorted_step_indices if s >= warmup_steps]
    window_step_indices = perf_step_indices if perf_step_indices else sorted_step_indices

    step_mean_train_ms = [
        float(np.mean([r["stage_timings_ms"]["train_ms"] for r in all_steps_by_idx[s]])) for s in window_step_indices
    ]
    overall_train_mean_ms = float(np.mean(step_mean_train_ms)) if step_mean_train_ms else 0.0

    repeatable_stages = {
        "ref_image_preprocess_ms",
        "ref_vision_encode_ms",
        "prefix_encode_ms",
        "target_image_preprocess_ms",
        "train_ms",
    }
    stage_summary: dict[str, Any] = {}
    for stage in ALL_STAGE_NAMES:
        step_means = [
            float(np.mean([r["stage_timings_ms"].get(stage, 0.0) for r in all_steps_by_idx[s]]))
            for s in window_step_indices
        ]
        local_rep_means = (
            [
                float(np.mean([r["repeated_stage_timings_ms"].get(stage, 0.0) for r in all_steps_by_idx[s]]))
                for s in window_step_indices
            ]
            if stage in repeatable_stages
            else None
        )
        global_rep_means = (
            [
                float(
                    np.mean(
                        [
                            global_repeated_step_timings.get((int(r["rank"]), s), {}).get(stage, 0.0)
                            for r in all_steps_by_idx[s]
                        ]
                    )
                )
                for s in window_step_indices
            ]
            if stage in repeatable_stages
            else None
        )
        stage_summary[stage] = _summarize_timing_series(
            step_means,
            overall_train_mean_ms,
            local_repeated_values=local_rep_means,
            global_repeated_values=global_rep_means,
        )

    feature_summary = {key: _summarize_feature_bytes(feature_bytes_all[key]) for key in FEATURE_BYTE_KEYS}

    window_rank_records = [r for s in window_step_indices for r in all_steps_by_idx[s]]
    all_rank_records = [r for s in sorted_step_indices for r in all_steps_by_idx[s]]
    step_alloc_vals = [int(r["memory"]["step_peak_allocated_bytes"]) for r in window_rank_records]
    step_res_vals = [int(r["memory"]["step_peak_reserved_bytes"]) for r in window_rank_records]
    lifetime_alloc_max = max(
        (int(r["memory"]["lifetime_peak_allocated_bytes"]) for r in all_rank_records),
        default=0,
    )
    lifetime_res_max = max(
        (int(r["memory"]["lifetime_peak_reserved_bytes"]) for r in all_rank_records),
        default=0,
    )
    cpu_peak_rss_max = max(
        (int(r["memory"]["cpu_peak_rss_bytes"]) for r in all_rank_records),
        default=0,
    )
    gib = float(1024**3)
    step_alloc_mean = float(np.mean(step_alloc_vals)) if step_alloc_vals else 0.0
    step_alloc_max = int(max(step_alloc_vals)) if step_alloc_vals else 0
    step_res_mean = float(np.mean(step_res_vals)) if step_res_vals else 0.0
    step_res_max = int(max(step_res_vals)) if step_res_vals else 0

    summary = {
        "world_size": world_size,
        "ranks": ranks,
        "micro_batches": micro_batches,
        "global_batch_size": global_batch_size,
        "dataset_size": dataset_size,
        "warmup_steps": warmup_steps,
        "total_steps": len(sorted_step_indices),
        "perf_window_steps": len(perf_step_indices),
        "sample_stats": {
            "total_samples": total_samples,
            "local_unique_rows_sum": sum(len(s) for s in rank_seen_samples.values()),
            "global_unique_rows": len(global_seen_samples),
            "padding_samples": padding_samples,
            "local_repeated_samples": local_repeated_samples,
            "global_repeated_samples": global_repeated_samples,
            "local_sample_repeat_rate": (float(local_repeated_samples / total_samples) if total_samples > 0 else 0.0),
            "global_sample_repeat_rate": (
                float(global_repeated_samples / total_samples) if total_samples > 0 else 0.0
            ),
        },
        "encoding_stats": encode_agg,
        "transfer_stats": transfer_summary,
        "stage_timings_ms": stage_summary,
        "feature_tensor_bytes": feature_summary,
        "memory_stats": {
            "step_peak_allocated_mean_bytes": step_alloc_mean,
            "step_peak_allocated_max_bytes": step_alloc_max,
            "step_peak_allocated_mean_gib": step_alloc_mean / gib,
            "step_peak_allocated_max_gib": step_alloc_max / gib,
            "step_peak_reserved_mean_bytes": step_res_mean,
            "step_peak_reserved_max_bytes": step_res_max,
            "step_peak_reserved_mean_gib": step_res_mean / gib,
            "step_peak_reserved_max_gib": step_res_max / gib,
            "lifetime_peak_allocated_bytes": lifetime_alloc_max,
            "lifetime_peak_allocated_gib": lifetime_alloc_max / gib,
            "lifetime_peak_reserved_bytes": lifetime_res_max,
            "lifetime_peak_reserved_gib": lifetime_res_max / gib,
            "cpu_peak_rss_bytes": cpu_peak_rss_max,
            "cpu_peak_rss_gib": cpu_peak_rss_max / gib,
        },
        "repeat_evidence": {
            "local_repeat_count": local_prefix_repeat_count,
            "global_repeat_count": global_prefix_repeat_count,
            "local_examples": local_repeat_evidence,
            "global_examples": global_repeat_evidence,
        },
    }
    baseline_summary_path = out_path / "baseline_summary.json"
    baseline_summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python3 -m relax.engine.sft.condition_tracker <output_dir>")
    result = summarize_baseline_dir(Path(sys.argv[1]))
    print(json.dumps(result, indent=2, ensure_ascii=False))
