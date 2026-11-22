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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

import numpy as np
import torch

from relax.engine.sft.image import ImageBatchSampler, assign_image_batch_positions, image_tier_budget
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


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


@dataclass(frozen=True)
class TransferSpec:
    """Declaration of one host-to-device condition transfer tracked per
    micro."""

    kind: str
    key_field: str
    per_image: bool = False
    repeat_flag: str | None = None


@dataclass(frozen=True)
class EncodeSpec:
    """Declaration of one encoding stage and its emitted feature byte keys."""

    kind: str
    key_field: str | None
    stage: str
    eligible: bool | None = None
    feature_bytes: tuple[str, ...] = ()
    items_field: str | None = None
    repeat_flag: str | None = None
    global_repeat: bool = True


@dataclass(frozen=True)
class EvidenceSpec:
    """Configuration for collecting repeated condition examples in
    summaries."""

    key_field: str
    extra_key_fields: tuple[str, ...]
    h2d_name: str
    h2d_transfers: tuple[str, ...]
    stages: tuple[str, ...]


@dataclass(frozen=True)
class TrackerSchema:
    """Model-specific declaration driving ConditionTracker recording and
    summaries."""

    name: str
    stage_order: tuple[str, ...]
    host_stages: tuple[str, ...]
    micro_gpu_stages: tuple[str, ...]
    key_fields: tuple[str, ...]
    micro_keys: Callable[..., dict[str, Any]]
    transfers: tuple[TransferSpec, ...]
    encodes: tuple[EncodeSpec, ...]
    host_repeat_stages: tuple[tuple[str, str], ...]
    condition_stages: tuple[str, ...]
    evidence: EvidenceSpec
    prefix_context: Callable[..., None] | None = None

    @property
    def feature_byte_keys(self) -> tuple[str, ...]:
        """Return all declared output feature byte metric keys."""
        return tuple(key for spec in self.encodes for key in spec.feature_bytes)

    @property
    def repeatable_stages(self) -> tuple[str, ...]:
        """Return ordered stage names whose work can repeat across epochs."""
        stages = {stage for stage, _ in self.host_repeat_stages}
        stages.update(spec.stage for spec in self.encodes if spec.global_repeat)
        return tuple(stage for stage in self.stage_order if stage in stages or stage == "train_ms")


def _sensenova_micro_keys(
    row: dict[str, Any], config: dict[str, Any], budget: int | None, height: int, width: int, training_mode: str
) -> dict[str, Any]:
    # SenseNova 在线基线按文件路径与分辨率配置构建 H2D 键，仅识别同路径重复；实际缓存键按像素内容哈希。
    edit_paths = [str(path) for path in row["edit_image"]]
    max_pixels = int(config["max_pixels"])
    ref_image_keys = [_sha256_key([path, max_pixels, budget, len(edit_paths)]) for path in edit_paths]
    ref_vision_key = _sha256_key(ref_image_keys)
    return {
        "ref_image_keys": ref_image_keys,
        "ref_vision_key": ref_vision_key,
        "prefix_kv_key": _sha256_key(
            [
                str(row["prompt"]),
                True,
                ref_vision_key,
                height,
                width,
                bool(config.get("think_mode", False)),
                training_mode,
            ]
        ),
        "target_image_key": _sha256_key([str(row["image"]), max_pixels, budget]),
        "eligible": training_mode == "lora",
    }


def _sensenova_prefix_context(tracker: Any, prompt: str, is_positive: bool, think_mode: bool) -> None:
    tracker.current_is_positive = bool(is_positive)
    key = _sha256_key(
        [
            str(prompt),
            tracker.current_is_positive,
            tracker.current_keys["ref_vision_key"],
            tracker.current_height,
            tracker.current_width,
            bool(think_mode),
            tracker.current_training_mode,
        ]
    )
    tracker.current_keys["prefix_kv_key"] = key
    if tracker.current_is_positive:
        micro_rec = tracker._current_micro_records.get(tracker.current_micro)
        if micro_rec is not None:
            micro_rec["prefix_kv_key"] = key


SENSENOVA_TRACKER_SCHEMA = TrackerSchema(
    name="relax.engine.sft.condition_tracker.SENSENOVA_TRACKER_SCHEMA",
    stage_order=ALL_STAGE_NAMES,
    host_stages=HOST_STAGES,
    micro_gpu_stages=MICRO_GPU_STAGES,
    key_fields=("ref_image_keys", "ref_vision_key", "prefix_kv_key", "target_image_key"),
    micro_keys=_sensenova_micro_keys,
    transfers=(
        TransferSpec("ref_pixel_values", "ref_image_keys", True, "ref_image_repeated"),
        TransferSpec("ref_grid_hw", "ref_image_keys", True),
        TransferSpec("prefix_input_ids", "prefix_kv_key"),
        TransferSpec("target_input_latents", "target_image_key", repeat_flag="target_image_repeated"),
    ),
    encodes=(
        EncodeSpec(
            "ref_vision_encode",
            "ref_vision_key",
            "ref_vision_encode_ms",
            feature_bytes=("ref_vision_embeds_bytes",),
            items_field="encoded_images",
            repeat_flag="ref_vision_repeated",
        ),
        EncodeSpec(
            "prefix_encode",
            "prefix_kv_key",
            "prefix_encode_ms",
            feature_bytes=("past_key_values_bytes", "indexes_image_bytes"),
            repeat_flag="prefix_kv_repeated",
        ),
        EncodeSpec("gen_vision_encode", None, "gen_vision_encode_ms", eligible=False, global_repeat=False),
    ),
    host_repeat_stages=(
        ("ref_image_preprocess_ms", "ref_pixel_values"),
        ("target_image_preprocess_ms", "target_input_latents"),
    ),
    condition_stages=("ref_image_preprocess_ms", "ref_vision_encode_ms", "prefix_encode_ms"),
    evidence=EvidenceSpec(
        "prefix_kv_key",
        ("ref_vision_key",),
        "ref_h2d",
        ("ref_pixel_values", "ref_grid_hw", "prefix_input_ids"),
        ("ref_image_preprocess_ms", "ref_vision_encode_ms", "prefix_encode_ms"),
    ),
    prefix_context=_sensenova_prefix_context,
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
    """Return the tensor bytes held by a K/V cache or a nested container."""
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
    """Record the online condition encoding baseline of one rank.

    The tracker detects repeated samples, encoder calls and host-to-device
    transfers by content keys, times host and GPU stages, and records step and
    lifetime memory peaks. ``end_step`` appends one JSON line per step to
    ``rank_<rank>_steps.jsonl`` under ``output_dir``, and ``save_summary``
    writes ``rank_<rank>_summary.json``. Steps before ``warmup_steps`` are
    excluded from the performance window.
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        micro_batches: int,
        dataset_size: int,
        output_dir: Path,
        warmup_steps: int = 5,
        seed: int | None = None,
        schema: TrackerSchema = SENSENOVA_TRACKER_SCHEMA,
        run_id: str | None = None,
    ) -> None:
        if rank < 0 or world_size <= 0 or micro_batches <= 0 or dataset_size <= 0 or warmup_steps < 0:
            raise ValueError("Invalid ConditionTracker configuration")
        self.rank = int(rank)
        self.schema = schema
        self.run_id = str(run_id) if run_id is not None else None
        self.io_failed = False
        self.io_error: str | None = None
        self.transfer_specs = {spec.kind: spec for spec in schema.transfers}
        self.encode_specs = {spec.kind: spec for spec in schema.encodes}
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

        # Preallocate two CUDA event pairs per (micro, stage) for the positive and negative branches
        self._cuda_available = torch.cuda.is_available()
        self._use_cuda_events = False
        self._event_pool: dict[tuple[int, str], list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {}
        self._step_event_pool: dict[str, tuple[torch.cuda.Event, torch.cuda.Event]] = {}
        if self._cuda_available:
            for micro in range(self.micro_batches):
                for stage in schema.micro_gpu_stages:
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

        self.seen_encode_keys: dict[str, set[str]] = {k: set() for k in self.encode_specs}
        self.encode_stats: dict[str, dict[str, Any]] = {
            spec.kind: {
                "eligible": spec.eligible,
                "calls": 0,
                **({spec.items_field: 0} if spec.items_field else {}),
                "items": 0,
                "repeated_calls": 0,
            }
            for spec in schema.encodes
        }

        self.seen_transfer_keys: dict[str, set[str]] = {k: set() for k in self.transfer_specs}
        self.transfer_stats: dict[str, dict[str, int]] = {
            k: {
                "total_transfer_calls": 0,
                "total_transfer_bytes": 0,
                "repeated_transfer_calls": 0,
                "repeated_transfer_bytes": 0,
            }
            for k in self.transfer_specs
        }
        self.feature_byte_records: dict[str, list[int]] = {k: [] for k in schema.feature_byte_keys}
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
        self.current_keys: dict[str, Any] = {}
        self.current_eligible: bool = True

        self._recorded_micro_event_counts: dict[tuple[int, str], int] = {}
        self._recorded_step_events: set[str] = set()
        self._current_micro_records: dict[int, dict[str, Any]] = {}
        self._current_step_timings: dict[str, float] = {stage: 0.0 for stage in STEP_GPU_STAGES}
        self._current_step_peak_allocated: int | None = None
        self._current_step_peak_reserved: int | None = None

    def _fallback_step_padding_flags(self, config: dict[str, Any]) -> list[bool]:
        sampler_seed = self.seed if self.seed is not None else int(config.get("seed") or 42)
        sampler = ImageBatchSampler(self.dataset_size, self.global_batch_size, sampler_seed)
        tagged = sampler.batch_with_padding(self.current_step)
        ids = [sample_id for sample_id, _ in tagged]
        positions = assign_image_batch_positions(ids, config, step=self.current_step, world=self.world_size)
        return [bool(tagged[pos][1]) for pos in positions[self.rank :: self.world_size]]

    def begin_step(self, step: int, device: torch.device) -> None:
        """Start ``step`` on ``device`` and reset the per-step records.

        On CUDA devices the previous peaks are folded into the lifetime peaks
        before the peak memory statistics are reset.
        """
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
        *,
        is_padding: bool | None = None,
    ) -> None:
        """Start a micro-batch and derive the content keys of its sample.

        ``row`` is the dataset row of ``sample_id``. Its reference images,
        prompt and target image, with ``config``, ``height`` and ``width``,
        form the keys that detect repeated work. Only LoRA training is eligible
        for reuse.
        """
        self.current_micro = int(micro)
        self.current_sample_id = int(sample_id)
        self.current_training_mode = str(training_mode)
        self.current_height = int(height)
        self.current_width = int(width)
        self.current_is_positive = True
        budget = image_tier_budget(config, self.current_sample_id)
        self.current_budget = budget

        keys = self.schema.micro_keys(row, config, budget, height, width, training_mode)
        eligible = keys["eligible"]
        self.current_keys = {key: keys[key] for key in self.schema.key_fields}
        self.current_eligible = eligible

        self.total_samples += 1
        if is_padding is None:
            if self._step_padding_flags is None:
                self._step_padding_flags = self._fallback_step_padding_flags(config)
            if self.current_micro < len(self._step_padding_flags):
                resolved_padding = self._step_padding_flags[self.current_micro]
            else:
                offset = self.current_step % self.steps_per_epoch
                epoch_pos = offset * self.global_batch_size + self.current_micro * self.world_size + self.rank
                resolved_padding = epoch_pos >= self.dataset_size
        else:
            resolved_padding = bool(is_padding)
        self._epoch_seen_sample_ids.add(self.current_sample_id)
        if resolved_padding:
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
            "is_padding": resolved_padding,
            "is_repeated_sample": is_repeated_sample,
            **self.current_keys,
            **{spec.repeat_flag: False for spec in (*self.schema.transfers, *self.schema.encodes) if spec.repeat_flag},
            "transfers": {
                k: {"calls": 0, "bytes": 0, "repeated_calls": 0, "repeated_bytes": 0, "events": []}
                for k in self.transfer_specs
            },
            "encodes": {},
            "feature_bytes": {},
            "timings_ms": {stage: 0.0 for stage in (*self.schema.host_stages, *self.schema.micro_gpu_stages)},
        }

    def set_prefix_context(self, prompt: str, is_positive: bool, think_mode: bool) -> None:
        """Set the prefix key of the next positive or negative encode."""
        if self.schema.prefix_context is None:
            raise ValueError(f"Tracker schema {self.schema.name} has no prefix context")
        self.schema.prefix_context(self, prompt, is_positive, think_mode)

    def add_host_time(self, stage: str, ms: float) -> None:
        """Add host time measured during sample preparation."""
        if stage not in self.schema.host_stages or not math.isfinite(ms) or ms < 0:
            raise ValueError(f"Invalid host timing: {stage}={ms}")
        micro_rec = self._current_micro_records.get(self.current_micro)
        if micro_rec is not None:
            micro_rec["timings_ms"][stage] += ms

    @staticmethod
    def compute_kv_bytes(past_key_values: Any) -> int:
        """Return the tensor bytes held by ``past_key_values``."""
        return compute_past_key_values_bytes(past_key_values)

    @contextmanager
    def host_stage(self, name: str) -> Iterator[None]:
        """Add the wall-clock time of a host stage to the micro-batch."""
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
        """Time a GPU stage of the current micro-batch or step.

        On CUDA devices the stage is timed with CUDA events that ``end_step``
        resolves; otherwise wall-clock time is used.
        """
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

    def record_transfer(
        self, kind: str, tensor: torch.Tensor | tuple[torch.Tensor, ...], image_index: int = 0, key: str | None = None
    ) -> None:
        """Record a host-to-device transfer of ``tensor``.

        A transfer repeats when ``key`` was seen before. ``key`` defaults to
        the content key of the current reference image, prefix or target image
        for ``kind``.
        """
        if kind not in self.transfer_stats:
            raise ValueError(f"Unknown transfer kind: {kind!r}")
        spec = self.transfer_specs[kind]
        if key is None:
            key = self.current_keys[spec.key_field]
            if spec.per_image:
                key = key[image_index]

        nbytes = compute_past_key_values_bytes(tensor)
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
            m_tr["events"].append({"key": key, "bytes": nbytes})
            if is_repeat:
                m_tr["repeated_calls"] += 1
                m_tr["repeated_bytes"] += nbytes
                if spec.repeat_flag:
                    micro_rec[spec.repeat_flag] = True

    def record_encode(
        self,
        kind: str,
        num_items: int,
        feature_bytes: Mapping[str, int] | None = None,
        key: str | None = None,
        eligible: bool | None = None,
    ) -> None:
        """Record one encoder call of ``kind`` over ``num_items`` inputs.

        A call repeats when ``key`` was seen before; ``key`` defaults to the
        schema's current key. ``feature_bytes`` records declared output sizes.
        """
        if kind not in self.encode_stats:
            raise ValueError(f"Unknown encode kind: {kind!r}")
        spec = self.encode_specs[kind]
        feature_bytes = {} if feature_bytes is None else feature_bytes
        if set(feature_bytes) - set(spec.feature_bytes):
            raise ValueError(f"Unknown feature byte fields for {kind}: {sorted(feature_bytes)}")
        if any(type(value) is not int or value < 0 for value in feature_bytes.values()):
            raise ValueError("Feature byte counts must be nonnegative integers")
        resolved_eligible = self.current_eligible if eligible is None else bool(eligible)
        if spec.eligible is not None:
            resolved_eligible = spec.eligible
        resolved_key = key if key is not None else self.current_keys.get(spec.key_field)
        if resolved_key is not None and not isinstance(resolved_key, str):
            raise ValueError(f"Encoding {kind} requires one explicit content key per call")
        is_repeat = bool(resolved_key) and (resolved_key in self.seen_encode_keys[kind])
        if resolved_key:
            self.seen_encode_keys[kind].add(resolved_key)

        stats = self.encode_stats[kind]
        stats["eligible"] = resolved_eligible
        stats["calls"] += 1
        stats["items"] += int(num_items)
        if spec.items_field:
            stats[spec.items_field] += int(num_items)
        if is_repeat:
            stats["repeated_calls"] += 1

        micro_rec = self._current_micro_records.get(self.current_micro)
        for field, nbytes in feature_bytes.items():
            if nbytes > 0:
                if (
                    isinstance(self.current_keys.get(spec.key_field), (list, tuple))
                    and micro_rec is not None
                    and field in micro_rec["feature_bytes"]
                ):
                    self.feature_byte_records[field][-1] += nbytes
                else:
                    self.feature_byte_records[field].append(nbytes)

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
            enc_entry["output_bytes"] += sum(feature_bytes.values())
            if first_call_in_micro or self.current_is_positive:
                if spec.repeat_flag:
                    micro_rec[spec.repeat_flag] = enc_entry["repeated"]
                for field, nbytes in feature_bytes.items():
                    if nbytes > 0:
                        if isinstance(self.current_keys.get(spec.key_field), (list, tuple)):
                            micro_rec["feature_bytes"][field] = micro_rec["feature_bytes"].get(field, 0) + nbytes
                        else:
                            micro_rec["feature_bytes"][field] = nbytes

    def consume_step_memory(self, device: torch.device) -> tuple[int, int]:
        """Record the step memory peaks of ``device``.

        Returns the lifetime ``(max_allocated, max_reserved)`` peaks in bytes.
        """
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
        """Finish ``step`` and append its record to the steps file.

        Waits for the CUDA events of the step. ``data_wait_seconds`` and
        ``train_seconds`` are measured by the caller.
        """
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
        for stage in (*self.schema.host_stages, *self.schema.micro_gpu_stages):
            stage_timings_ms[stage] = float(sum(m["timings_ms"].get(stage, 0.0) for m in micro_list))
        for stage in STEP_GPU_STAGES:
            stage_timings_ms[stage] = float(self._current_step_timings.get(stage, 0.0))
        stage_timings_ms["train_ms"] = float(train_seconds) * 1000.0

        repeated_stage_timings_ms = dict.fromkeys(self.schema.repeatable_stages, 0.0)
        for stage, kind in self.schema.host_repeat_stages:
            repeated_stage_timings_ms[stage] = float(
                sum(m["timings_ms"].get(stage, 0.0) for m in micro_list if m["transfers"][kind]["repeated_calls"])
            )
        for spec in self.schema.encodes:
            if not spec.global_repeat:
                continue
            for m in micro_list:
                enc = m["encodes"].get(spec.kind)
                if enc and enc["calls"]:
                    repeated_stage_timings_ms[spec.stage] += (
                        float(m["timings_ms"].get(spec.stage, 0.0)) * enc["repeated_calls"] / enc["calls"]
                    )
        repeated_stage_timings_ms["condition_repeated_ms"] = sum(
            repeated_stage_timings_ms.get(stage, 0.0) for stage in self.schema.condition_stages
        )
        repeated_stage_timings_ms["train_ms"] = sum(
            repeated_stage_timings_ms[stage] for stage in self.schema.repeatable_stages if stage != "train_ms"
        )

        step_transfers = {
            k: {
                "calls": sum(m["transfers"][k]["calls"] for m in micro_list),
                "bytes": sum(m["transfers"][k]["bytes"] for m in micro_list),
                "repeated_calls": sum(m["transfers"][k]["repeated_calls"] for m in micro_list),
                "repeated_bytes": sum(m["transfers"][k]["repeated_bytes"] for m in micro_list),
            }
            for k in self.transfer_specs
        }
        cpu_rss_bytes, cpu_peak_rss_bytes = _read_cpu_memory_bytes()
        step_record = {
            "tracker_schema": self.schema.name,
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
        if self.run_id is not None:
            step_record["run_id"] = self.run_id
        if hasattr(self, "condition_cache_stats"):
            step_record["condition_cache"] = self.condition_cache_stats
        self.step_records.append(step_record)
        if not self.io_failed:
            try:
                with self.steps_jsonl_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(step_record, ensure_ascii=False) + "\n")
            except OSError as exc:
                self.io_failed = True
                self.io_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "ConditionTracker rank=%s step=%s log write failed; stopping disk logging: %s",
                    self.rank,
                    self.current_step,
                    exc,
                )

    def build_summary(self) -> dict[str, Any]:
        """Aggregate the recorded steps of this rank into a summary."""
        perf_records = [r for r in self.step_records if r["step"] >= self.warmup_steps]
        window_records = perf_records if perf_records else self.step_records

        encoding_summary: dict[str, Any] = {}
        for kind in self.encode_specs:
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
        for kind in self.transfer_specs:
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
        repeatable_stages = self.schema.repeatable_stages
        stage_summary: dict[str, Any] = {}
        for stage in self.schema.stage_order:
            vals = [float(r["stage_timings_ms"].get(stage, 0.0)) for r in window_records]
            rep_vals = (
                [float(r["repeated_stage_timings_ms"].get(stage, 0.0)) for r in window_records]
                if stage in repeatable_stages
                else None
            )
            stage_summary[stage] = _summarize_timing_series(vals, train_mean_ms, local_repeated_values=rep_vals)

        feature_summary = {
            key: _summarize_feature_bytes(self.feature_byte_records[key]) for key in self.schema.feature_byte_keys
        }

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

        summary = {
            "tracker_schema": self.schema.name,
            "rank": self.rank,
            "world_size": self.world_size,
            "complete": not self.io_failed,
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
        if self.run_id is not None:
            summary["run_id"] = self.run_id
        if self.io_error is not None:
            summary["io_error"] = self.io_error
        return summary

    def save_summary(self) -> dict[str, Any]:
        """Write and return the summary of this rank."""
        summary = self.build_summary()
        if hasattr(self, "condition_cache_stats"):
            summary["condition_cache"] = self.condition_cache_stats
        if self.io_failed:
            return summary
        try:
            self.summary_json_path.write_text(
                json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
        except OSError as exc:
            self.io_failed = True
            self.io_error = f"{type(exc).__name__}: {exc}"
            summary["complete"] = False
            summary["io_error"] = self.io_error
            logger.warning("ConditionTracker rank=%s summary write failed: %s", self.rank, exc)
        return summary


def summarize_baseline_dir(output_dir: Path, *, allow_incomplete: bool = False) -> dict[str, Any]:
    """Combine the rank files under ``output_dir`` into a run summary.

    Reads every ``rank_*_steps.jsonl`` and ``rank_*_summary.json``, validates
    rank coverage, run consistency and step alignment, measures local and
    cross-rank repeats, and writes ``baseline_summary.json``.
    """
    out_path = Path(output_dir)
    step_files = sorted(out_path.glob("rank_*_steps.jsonl"))
    if not step_files:
        raise FileNotFoundError(f"No rank_*_steps.jsonl files found under {out_path}")

    rank_step_records: dict[int, list[dict[str, Any]]] = {}
    for file_path in step_files:
        records = [json.loads(line) for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not records:
            continue
        rank = int(records[0]["rank"])
        if file_path.name != f"rank_{rank}_steps.jsonl":
            raise ValueError(f"Step log filename {file_path.name} does not match recorded rank {rank}")
        if any(int(rec["rank"]) != rank for rec in records):
            raise ValueError(f"Step log {file_path.name} mixes records from multiple ranks")
        if rank in rank_step_records:
            raise ValueError(f"Duplicate baseline step log for rank {rank}")
        rank_step_records[rank] = records

    if not rank_step_records:
        raise ValueError(f"Step log files under {out_path} are empty")

    rank_summaries: dict[int, dict[str, Any]] = {}
    for file_path in sorted(out_path.glob("rank_*_summary.json")):
        data = json.loads(file_path.read_text(encoding="utf-8"))
        rank = int(data["rank"])
        if file_path.name != f"rank_{rank}_summary.json":
            raise ValueError(f"Summary filename {file_path.name} does not match recorded rank {rank}")
        rank_summaries[rank] = data

    ranks = sorted(rank_step_records)
    first_rank_steps = rank_step_records[ranks[0]]
    schema_paths = {
        record.get("tracker_schema", SENSENOVA_TRACKER_SCHEMA.name)
        for records in rank_step_records.values()
        for record in records
    }
    schema_paths.update(data.get("tracker_schema", SENSENOVA_TRACKER_SCHEMA.name) for data in rank_summaries.values())
    if len(schema_paths) != 1:
        raise ValueError("Baseline files use different tracker schemas")
    schema_path = schema_paths.pop()
    if schema_path == SENSENOVA_TRACKER_SCHEMA.name:
        schema = SENSENOVA_TRACKER_SCHEMA
    else:
        from relax.utils.misc import load_function

        schema = load_function(schema_path)
    if not isinstance(schema, TrackerSchema):
        raise TypeError(f"Invalid tracker schema: {schema_path}")
    transfer_specs = {spec.kind: spec for spec in schema.transfers}
    encode_specs = {spec.kind: spec for spec in schema.encodes}

    world_sizes = {int(record["world_size"]) for records in rank_step_records.values() for record in records} | {
        int(data["world_size"]) for data in rank_summaries.values()
    }
    if len(world_sizes) != 1:
        raise ValueError(f"Baseline files disagree on world_size: {sorted(world_sizes)}")
    world_size = world_sizes.pop()
    if world_size <= 0:
        raise ValueError(f"Invalid world_size in baseline logs: {world_size}")

    all_ranks = set(ranks) | set(rank_summaries)
    unexpected_ranks = sorted(r for r in all_ranks if r < 0 or r >= world_size)
    if unexpected_ranks:
        raise ValueError(
            f"Baseline files contain stale or out-of-range ranks {unexpected_ranks} for world_size={world_size}"
        )

    run_ids = {record.get("run_id") for records in rank_step_records.values() for record in records} | {
        data.get("run_id") for data in rank_summaries.values()
    }
    if len(run_ids) > 1:
        raise ValueError(f"Baseline files mix records from different runs: {sorted(str(v) for v in run_ids)}")
    run_id = next(iter(run_ids))

    missing_ranks = [r for r in range(world_size) if r not in rank_step_records]
    missing_summaries = [r for r in range(world_size) if r not in rank_summaries]
    steps_by_rank: dict[int, list[int]] = {}
    for rank in ranks:
        rank_steps = [int(rec["step"]) for rec in rank_step_records[rank]]
        if len(rank_steps) != len(set(rank_steps)):
            raise ValueError(f"Rank {rank} step log contains duplicate step indices")
        steps_by_rank[rank] = rank_steps
    expected_steps = steps_by_rank[ranks[0]]
    mismatched_steps = [rank for rank in ranks[1:] if steps_by_rank[rank] != expected_steps]
    summary_step_mismatches = [
        rank
        for rank in ranks
        if rank in rank_summaries and int(rank_summaries[rank].get("total_steps", -1)) != len(steps_by_rank[rank])
    ]
    incomplete_summaries = sorted(rank for rank, data in rank_summaries.items() if data.get("complete") is False)
    is_complete = (
        not missing_ranks
        and not missing_summaries
        and not mismatched_steps
        and not summary_step_mismatches
        and not incomplete_summaries
    )
    if not is_complete and not allow_incomplete:
        problems = []
        if missing_ranks:
            problems.append(f"missing ranks {missing_ranks} for world_size={world_size}")
        if missing_summaries:
            problems.append(f"missing rank summaries {missing_summaries} for world_size={world_size}")
        if mismatched_steps:
            problems.append(f"step mismatch across ranks { {r: steps_by_rank[r] for r in ranks} }")
        if summary_step_mismatches:
            problems.append(f"summary total_steps mismatch for ranks {summary_step_mismatches}")
        if incomplete_summaries:
            problems.append(f"incomplete rank summaries {incomplete_summaries}")
        raise ValueError("Incomplete baseline records: " + "; ".join(problems))

    warmup_steps = int(
        rank_summaries[ranks[0]]["warmup_steps"]
        if ranks[0] in rank_summaries
        else next((r["step"] for r in first_rank_steps if r.get("in_perf_window")), 5)
    )
    dataset_size = int(rank_summaries[ranks[0]]["dataset_size"]) if ranks[0] in rank_summaries else 0
    micro_batches = len(first_rank_steps[0]["micros"])
    global_batch_size = micro_batches * world_size
    steps_per_epoch = math.ceil(dataset_size / global_batch_size) if dataset_size > 0 else 1

    # Walk all micro records in (step, micro, rank) order to count local and cross-rank repeats
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

    global_seen_encode_keys: dict[str, set[str]] = {k: set() for k in encode_specs}
    encode_agg: dict[str, dict[str, Any]] = {
        spec.kind: {
            "eligible": spec.eligible,
            "calls": 0,
            "items": 0,
            "local_repeated_calls": 0,
            "global_repeated_calls": 0,
        }
        for spec in schema.encodes
    }
    for spec in schema.encodes:
        if spec.items_field:
            encode_agg[spec.kind][spec.items_field] = 0

    global_seen_transfer_keys: dict[str, set[str]] = {k: set() for k in transfer_specs}
    transfer_agg: dict[str, dict[str, int]] = {
        k: {
            "total_transfer_calls": 0,
            "total_transfer_bytes": 0,
            "local_repeated_transfer_calls": 0,
            "local_repeated_transfer_bytes": 0,
            "global_repeated_transfer_calls": 0,
            "global_repeated_transfer_bytes": 0,
        }
        for k in transfer_specs
    }

    feature_bytes_all: dict[str, list[int]] = {k: [] for k in schema.feature_byte_keys}

    first_seen_prefix_local: dict[tuple[int, str], dict[str, Any]] = {}
    first_seen_prefix_global: dict[str, dict[str, Any]] = {}
    local_prefix_repeat_count = 0
    global_prefix_repeat_count = 0
    local_repeat_evidence: list[dict[str, Any]] = []
    global_repeat_evidence: list[dict[str, Any]] = []

    # Time spent on globally repeated work per (rank, step)
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

        for fkey in schema.feature_byte_keys:
            if fkey in micro_rec.get("feature_bytes", {}):
                feature_bytes_all[fkey].append(int(micro_rec["feature_bytes"][fkey]))

        step_glob_rep = global_repeated_step_timings.setdefault(
            (rank, step_idx),
            dict.fromkeys(schema.repeatable_stages, 0.0),
        )

        glob_rep_by_kind = {}
        for tkind, spec in transfer_specs.items():
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
            seen = global_seen_transfer_keys[tkind]
            if "events" in tr_info:
                repeated = False
                for event in tr_info["events"]:
                    key = event["key"]
                    if key and key in seen:
                        repeated = True
                        agg["global_repeated_transfer_calls"] += 1
                        agg["global_repeated_transfer_bytes"] += int(event["bytes"])
                    if key:
                        seen.add(key)
            else:
                keys = micro_rec.get(spec.key_field, []) if spec.per_image else [micro_rec.get(spec.key_field, "")]
                repeated = any(key and key in seen for key in keys)
                seen.update(key for key in keys if key)
                if repeated:
                    agg["global_repeated_transfer_calls"] += c
                    agg["global_repeated_transfer_bytes"] += b
            glob_rep_by_kind[tkind] = repeated

        for stage, kind in schema.host_repeat_stages:
            if glob_rep_by_kind[kind]:
                t_ms = float(micro_rec["timings_ms"].get(stage, 0.0))
                step_glob_rep[stage] += t_ms
                step_glob_rep["train_ms"] += t_ms

        # Aggregate encoder calls and collect repeat examples
        for ekind, spec in encode_specs.items():
            enc_info = micro_rec.get("encodes", {}).get(ekind)
            if enc_info is None:
                continue
            agg = encode_agg[ekind]
            agg["eligible"] = bool(enc_info["eligible"])
            calls = int(enc_info["calls"])
            items = int(enc_info["items"])
            agg["calls"] += calls
            agg["items"] += items
            if spec.items_field:
                agg[spec.items_field] += items
            local_rep_calls = int(
                enc_info["repeated_calls"]
                if "repeated_calls" in enc_info
                else (calls if enc_info.get("repeated", False) else 0)
            )
            agg["local_repeated_calls"] += local_rep_calls

            if spec.global_repeat:
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
                    stage_key = spec.stage
                    t_ms = float(micro_rec["timings_ms"].get(stage_key, 0.0)) * (
                        glob_rep_in_micro / max(len(ekeys), 1)
                    )
                    step_glob_rep[stage_key] += t_ms
                    step_glob_rep["train_ms"] += t_ms

        evidence = schema.evidence
        prefix_key = micro_rec.get(evidence.key_field, "")
        if prefix_key:
            h2d_bytes = sum(micro_rec["transfers"][kind]["bytes"] for kind in evidence.h2d_transfers)
            occurrence_meta = {
                "rank": rank,
                "step": step_idx,
                "micro": micro_idx,
                "sample_id": sample_id,
                "is_padding": bool(micro_rec.get("is_padding", False)),
                evidence.key_field: prefix_key,
                **{key: micro_rec.get(key, "") for key in evidence.extra_key_fields},
                f"{evidence.h2d_name}_bytes": int(h2d_bytes),
                f"{evidence.h2d_name}_mib": float(h2d_bytes / (1024.0 * 1024.0)),
                **{stage: float(micro_rec["timings_ms"].get(stage, 0.0)) for stage in evidence.stages},
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
    for tkind in transfer_specs:
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

    # Stage timings in the performance window: average each step across ranks, then compute mean, P50 and P95
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

    repeatable_stages = schema.repeatable_stages
    stage_summary: dict[str, Any] = {}
    for stage in schema.stage_order:
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

    feature_summary = {key: _summarize_feature_bytes(feature_bytes_all[key]) for key in schema.feature_byte_keys}

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
        "complete": is_complete,
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
    cache_stats = {
        str(rank): data["condition_cache"] for rank, data in rank_summaries.items() if "condition_cache" in data
    }
    if run_id is not None:
        summary["run_id"] = run_id
    if not is_complete:
        summary["missing_ranks"] = missing_ranks
        summary["missing_summaries"] = missing_summaries
        summary["recorded_steps_by_rank"] = {str(rank): steps_by_rank[rank] for rank in ranks}
        if summary_step_mismatches:
            summary["summary_step_mismatches"] = summary_step_mismatches
    if "tracker_schema" in first_rank_steps[0]:
        summary["tracker_schema"] = schema.name
    if cache_stats:
        summary["condition_cache"] = cache_stats
        total_requests = sum(int(item.get("requests", 0)) for item in cache_stats.values())
        total_lookups = sum(int(item.get("lookups", 0)) for item in cache_stats.values())
        total_local_hits = sum(int(item.get("local_hits", 0)) for item in cache_stats.values())
        total_effective_reuses = sum(int(item.get("effective_reuses", 0)) for item in cache_stats.values())
        total_forced_online = sum(int(item.get("group_forced_online", 0)) for item in cache_stats.values())
        summary["condition_cache_summary"] = {
            "ranks_covered": sorted(int(rank) for rank in cache_stats),
            "requests": total_requests,
            "lookups": total_lookups,
            "local_hits": total_local_hits,
            "effective_reuses": total_effective_reuses,
            "group_forced_online": total_forced_online,
            "local_hit_rate": float(total_local_hits / total_lookups) if total_lookups > 0 else 0.0,
            "effective_reuse_rate": float(total_effective_reuses / total_requests) if total_requests > 0 else 0.0,
        }
    baseline_summary_path = out_path / "baseline_summary.json"
    baseline_summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    from relax.engine.sft.condition_tracker import summarize_baseline_dir as summarize_directory

    args_cli = sys.argv[1:]
    allow_incomplete = False
    if "--allow-incomplete" in args_cli:
        allow_incomplete = True
        args_cli = [arg for arg in args_cli if arg != "--allow-incomplete"]
    if len(args_cli) != 1:
        raise SystemExit("Usage: python3 -m relax.engine.sft.condition_tracker <output_dir> [--allow-incomplete]")
    result = summarize_directory(Path(args_cli[0]), allow_incomplete=allow_incomplete)
    print(json.dumps(result, indent=2, ensure_ascii=False))
