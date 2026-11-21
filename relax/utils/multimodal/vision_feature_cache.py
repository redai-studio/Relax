# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Opt-in reuse of frozen Qwen3.5 ViT outputs, before its trainable merger.

Image identities are computed on CPU at the Sample boundary. The cache stores
complete pre-merger features on the executing rank's device, under a byte
budget. Original CP distribution/gather, merger TP collectives and LM SP remain
outside the cached decoder. No cache is shared across model instances or DP
replicas.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import math
from collections import Counter, OrderedDict
from typing import Any

import torch
import torch.distributed as dist

from relax.utils.env import Envs
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)
VISION_CACHE_INFO = "_relax_vision_cache"


def attach_vision_cache_info(session_id: str | None, inputs: dict | None) -> dict | None:
    """Attach CPU-only identity metadata without mutating the source sample."""
    if inputs is None or not session_id or "pixel_values" not in inputs or "image_grid_thw" not in inputs:
        return inputs
    if inputs.get("pixel_values_videos") is not None:
        return inputs  # Mixed image/video order requires a separate adapter.
    pixels, grid = inputs["pixel_values"], torch.as_tensor(inputs["image_grid_thw"])
    if not isinstance(pixels, torch.Tensor) or pixels.device.type != "cpu" or grid.device.type != "cpu":
        raise ValueError("Vision cache identities require CPU processor tensors before training transfer")
    if pixels.requires_grad:
        raise ValueError("Vision cache cannot detach trainable image inputs")
    shapes = [tuple(map(int, row)) for row in grid.tolist()]
    if any(t != 1 for t, _, _ in shapes):
        return inputs  # Temporal inputs retain the normal model path.
    if sum(math.prod(shape) for shape in shapes) != len(pixels):
        raise ValueError("Vision cache image geometry does not match processor pixel rows")
    infos, offset = [], 0
    for shape in shapes:
        end = offset + math.prod(shape)
        content = pixels[offset:end].detach().contiguous().view(torch.uint8).numpy().tobytes()
        digest = hashlib.sha256(str((pixels.dtype, shape)).encode() + content).hexdigest()
        infos.append({"key": (session_id, digest), "grid": shape})
        offset = end
    return {**inputs, VISION_CACHE_INFO: infos}


class FeatureLRU:
    """Bound device storage; entries own detached tensors, never
    graphs/views."""

    def __init__(self, max_bytes: int) -> None:
        if max_bytes <= 0:
            raise ValueError("RELAX_TRAIN_VISION_CACHE_MAX_BYTES must be positive")
        self.max_bytes = max_bytes
        self.entries: OrderedDict[Any, torch.Tensor] = OrderedDict()
        self.nbytes = 0
        self.hits = self.misses = 0
        self.evictions = self.oversized_skips = self.peak_bytes = 0

    def get(self, key: Any) -> torch.Tensor | None:
        value = self.entries.get(key)
        if value is not None:
            self.entries.move_to_end(key)
        return value

    def put(self, key: Any, value: torch.Tensor) -> None:
        size = value.numel() * value.element_size()
        if size > self.max_bytes:
            self.oversized_skips += 1
            return
        previous = self.entries.pop(key, None)
        if previous is not None:
            self.nbytes -= previous.numel() * previous.element_size()
        while self.entries and self.nbytes + size > self.max_bytes:
            _, old = self.entries.popitem(last=False)
            self.nbytes -= old.numel() * old.element_size()
            self.evictions += 1
        self.entries[key] = value.detach().clone()
        self.nbytes += size
        self.peak_bytes = max(self.peak_bytes, self.nbytes)

    def clear(self) -> None:
        self.entries.clear()
        self.nbytes = 0


class _Qwen35DecoderCache:
    """Adapter for the actual Qwen35VLModelProvider's pre-merger interface."""

    def __init__(self, model: torch.nn.Module, max_bytes: int) -> None:
        self.model = model
        self.vision = model.vision_model
        self.decoder = self.vision.decoder
        self.cache = FeatureLRU(max_bytes)
        self.stats: Counter[str] = Counter()
        self.images: list[dict] | None = None
        self.signature: Any = None
        self.original = self.decoder.forward
        self.forward_signature = inspect.signature(self.original)
        self.validate()
        self.control_group = self.decoder.tp_group
        if self.control_group is not None and dist.get_world_size(self.control_group) > 1:
            if dist.get_backend(self.control_group) != "gloo":
                # Only CPU hit/identity decisions: avoid NCCL object collectives'
                # device synchronization and extra GPU buffers. TP members alone
                # create this disjoint group; it survives train/rollout offload.
                self.control_group = dist.new_group(
                    ranks=dist.get_process_group_ranks(self.control_group),
                    backend="gloo",
                    use_local_synchronization=True,
                )

    def frozen_modules(self) -> tuple[torch.nn.Module, ...]:
        rotary = getattr(self.vision, "rotary_pos_emb", None)
        return (self.vision.patch_embed, self.vision.pos_embed, self.decoder) + (
            (rotary,) if rotary is not None else ()
        )

    def validate(self) -> None:
        if getattr(self.model, "use_dist_train", False):
            raise ValueError("Vision cache does not map disaggregated encoder/decoder inputs")
        if getattr(self.decoder, "deepstack_visual_indexes", ()):
            raise ValueError(
                "Vision cache adapter needs an empty deepstack; its intermediate mergers may be trainable"
            )
        for module in self.frozen_modules():
            if any(p.requires_grad for p in module.parameters()):
                raise ValueError("Vision cache requires frozen patch/position embeddings and ViT decoder")
            for child in module.modules():
                if isinstance(child, torch.nn.modules.dropout._DropoutNd) and child.p:
                    raise ValueError("Vision cache requires deterministic frozen computation (dropout must be zero)")
        config = self.decoder.config
        if any(getattr(config, name, 0) for name in ("hidden_dropout", "attention_dropout", "fp8", "fp4")):
            raise ValueError("Vision cache requires zero dropout and a stateless non-FP8/non-FP4 vision encoder")
        if getattr(config, "cuda_graph_impl", "none") not in (None, "none"):
            raise ValueError("Vision decoder CUDA graph capture is not adapted for cache-dependent miss batches")
        if getattr(config, "sequence_parallel", False):
            raise ValueError("This adapter expects complete pre-merger visual sequences, not vision-SP shards")
        if not {"hidden_states", "rotary_pos_emb", "packed_seq_params"} <= set(self.forward_signature.parameters):
            raise ValueError("Unsupported Qwen3.5 visual decoder signature")

    def local_images(self, infos: list[dict] | None, kwargs: dict) -> list[dict] | None:
        if not infos or kwargs.get("pixel_values_videos") is not None:
            return None
        cp = self.model.pg_collection.cp  # Read the live group, including dynamic CP.
        if cp.size() == 1 or not getattr(self.model.config, "vision_dp_when_cp", False):
            return infos
        counts = kwargs.get("cp_img_num")
        if counts is None:
            # Same contiguous, image-count partition as Bridge qwen3vl_cp_split.
            counts = [len(infos) // cp.size() + (rank < len(infos) % cp.size()) for rank in range(cp.size())]
        if any(kwargs.get("images_padded") or ()) or sum(counts) != len(infos):
            raise ValueError("Vision cache cannot map custom CP image padding to source image identities")
        start = sum(counts[: cp.rank()])
        return infos[start : start + counts[cp.rank()]]

    def _weight_signature(self, hidden: torch.Tensor) -> tuple:
        tensors = [p for module in self.frozen_modules() for p in (*module.parameters(), *module.buffers())]
        return (
            hidden.dtype,
            hidden.device,
            torch.is_autocast_enabled(hidden.device.type),
            torch.get_autocast_dtype(hidden.device.type),
            tuple((id(p), p._version, p.dtype, p.device) for p in tensors),
        )

    def report_and_clear(self, reason: str) -> None:
        """Report one interval per visual rank without tensor
        synchronization."""
        fields = (
            "calls",
            "cache_calls",
            "image_refs",
            "batch_dedup_images",
            "encoded_images",
            "patch_refs",
            "cache_hit_patches",
            "batch_dedup_patches",
            "encoded_patches",
            "bypass_calls",
            "bypass_patches",
            "tp_forced_misses",
        )
        metrics = {name: self.stats[name] for name in fields}
        metrics.update(
            hits=self.cache.hits,
            misses=self.cache.misses,
            evictions=self.cache.evictions,
            oversized_skips=self.cache.oversized_skips,
            entries=len(self.cache.entries),
            bytes=self.cache.nbytes,
            peak_bytes=self.cache.peak_bytes,
            budget_bytes=self.cache.max_bytes,
            image_hit_ratio=self.cache.hits / metrics["image_refs"] if metrics["image_refs"] else 0.0,
            image_skip_ratio=(self.cache.hits + metrics["batch_dedup_images"]) / metrics["image_refs"]
            if metrics["image_refs"]
            else 0.0,
            patch_skip_ratio=1 - metrics["encoded_patches"] / metrics["patch_refs"] if metrics["patch_refs"] else 0.0,
        )
        ranks = {
            name: getattr(self.model.pg_collection, name).rank()
            if getattr(self.model.pg_collection, name, None) is not None
            else None
            for name in ("tp", "pp", "cp", "dp")
        }
        logger.info(
            "Training vision cache: hits=%d misses=%d bytes=%d; clearing reason=%s ranks=%s stats=%s",
            self.cache.hits,
            self.cache.misses,
            self.cache.nbytes,
            reason,
            ranks,
            json.dumps(metrics, sort_keys=True),
        )
        self.cache.clear()
        self.cache.hits = self.cache.misses = 0
        self.cache.evictions = self.cache.oversized_skips = self.cache.peak_bytes = 0
        self.stats.clear()

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        bound = self.forward_signature.bind(*args, **kwargs)
        hidden = bound.arguments["hidden_states"]
        signature = self._weight_signature(hidden)
        if signature != self.signature:
            if self.signature is not None:
                self.report_and_clear("encoder_state_changed")
            else:
                self.cache.clear()
            self.signature = signature
        reason = None
        try:
            self.validate()  # Detect a later unfreeze or change in execution mode.
            if hidden.requires_grad:
                raise ValueError("Vision cache cannot detach trainable encoder inputs")
            for name in (
                "attention_mask",
                "context",
                "context_mask",
                "attention_bias",
                "rotary_pos_cos",
                "rotary_pos_sin",
                "inference_context",
                "inference_params",
                "sequence_len_offset",
            ):
                if bound.arguments.get(name) is not None:
                    raise ValueError(f"Vision cache does not support visual decoder argument {name}")
            if self.images and sum(math.prod(x["grid"]) for x in self.images) != hidden.shape[0]:
                raise ValueError("Vision cache metadata does not match the actual local visual sequence")
        except ValueError as error:
            reason = str(error)
        keys = [tuple(x["key"]) for x in self.images] if self.images else None
        features = [self.cache.get(key) for key in keys] if keys else []
        state = (keys, [x is not None for x in features], reason)
        group = self.control_group  # No world/DP/PP/CP cache consensus.
        states = [state]
        if group is not None and dist.get_world_size(group) > 1:
            states = [None] * dist.get_world_size(group)
            # Exchange only CPU identity/control metadata. No feature or pixel tensor sync.
            dist.all_gather_object(states, state, group=group)
        errors = [s[2] for s in states if s[2]]
        if errors:
            raise ValueError(errors[0])
        self.stats["calls"] += 1
        self.stats["patch_refs"] += hidden.shape[0]
        if any(s[0] is None for s in states):
            self.stats["bypass_calls"] += 1
            self.stats["bypass_patches"] += hidden.shape[0]
            self.stats["encoded_patches"] += hidden.shape[0]
            return self.original(*args, **kwargs)
        if any(s[0] != keys for s in states):
            raise ValueError("Vision cache image order differs across collaborating TP ranks")
        hits = [all(s[1][i] for s in states) for i in range(len(keys))]
        self.cache.hits += sum(hits)
        self.cache.misses += len(hits) - sum(hits)
        self.stats["cache_calls"] += 1
        self.stats["image_refs"] += len(keys)
        self.stats["tp_forced_misses"] += sum(feature is not None and not hit for feature, hit in zip(features, hits))
        offsets = [0]
        for image in self.images:
            offsets.append(offsets[-1] + math.prod(image["grid"]))
        missing = list(dict.fromkeys(key for key, hit in zip(keys, hits) if not hit))
        self.stats["cache_hit_patches"] += sum(offsets[i + 1] - offsets[i] for i, hit in enumerate(hits) if hit)
        if missing:
            indices = [keys.index(key) for key in missing]
            encoded_patches = sum(offsets[i + 1] - offsets[i] for i in indices)
            selection = lambda tensor: torch.cat([tensor[offsets[i] : offsets[i + 1]] for i in indices], dim=0)
            call = dict(bound.arguments)
            call["hidden_states"] = selection(hidden)
            call["rotary_pos_emb"] = selection(call["rotary_pos_emb"])
            grid = torch.tensor([self.images[i]["grid"] for i in indices], device=hidden.device)
            call["packed_seq_params"] = self.vision.build_packed_seq_params(grid)
            with torch.no_grad():
                output, deepstack = self.original(**call)
            if deepstack or output.shape != call["hidden_states"].shape:
                raise ValueError("Visual decoder must return complete pre-merger features and no deepstack")
            self.stats["encoded_images"] += len(missing)
            self.stats["encoded_patches"] += encoded_patches
            self.stats["batch_dedup_images"] += len(hits) - sum(hits) - len(missing)
            self.stats["batch_dedup_patches"] += (
                sum(offsets[i + 1] - offsets[i] for i, hit in enumerate(hits) if not hit) - encoded_patches
            )
            split = torch.split(output.detach(), [offsets[i + 1] - offsets[i] for i in indices])
            computed = dict(zip(missing, split, strict=True))
            for key, feature in computed.items():
                self.cache.put(key, feature)
            features = [features[i] if hit else computed[key] for i, (key, hit) in enumerate(zip(keys, hits))]
        return torch.cat(features, dim=0), []


def install_train_vision_cache(model: torch.nn.Module) -> None:
    """Install only the verified model interface; leave all parallel config
    intact."""
    if not Envs.RELAX_TRAIN_VISION_CACHE:
        return
    from megatron.bridge.models.qwen_vl.qwen35_vl_provider import Qwen35VLModelProvider

    if not isinstance(getattr(model, "config", None), Qwen35VLModelProvider):
        raise ValueError("Training vision cache currently has an adapter only for the Qwen3.5 dense visual interface")
    if getattr(model, "_relax_train_vision_cache_installed", False):
        return
    adapter = (
        _Qwen35DecoderCache(model, Envs.RELAX_TRAIN_VISION_CACHE_MAX_BYTES) if model.vision_model is not None else None
    )
    original = model.forward

    @functools.wraps(original)
    def forward(*args: Any, **kwargs: Any) -> Any:
        infos = kwargs.pop(VISION_CACHE_INFO, None)
        if adapter is None:
            return original(*args, **kwargs)
        previous = adapter.images
        adapter.images = adapter.local_images(infos, kwargs)
        try:
            return original(*args, **kwargs)
        finally:
            adapter.images = previous

    # All PP/VPP stages strip private metadata before Bridge forwards LM kwargs.
    model.forward = forward
    model._relax_train_vision_cache_installed = True
    if adapter is None:
        return
    adapter.decoder.forward = adapter.forward
    model._relax_train_vision_cache = adapter
    model.register_load_state_dict_pre_hook(lambda *unused: adapter.report_and_clear("load_state_dict"))
    groups = model.pg_collection
    parallel = {
        name: getattr(groups, name).size() if getattr(groups, name, None) is not None else None
        for name in ("tp", "pp", "cp", "dp")
    }
    logger.info(
        "Training vision cache: frozen Qwen3.5 decoder, complete local features, budget=%d bytes, "
        "parallel=%s LM_SP=%s vision_SP=%s",
        adapter.cache.max_bytes,
        parallel,
        model.config.sequence_parallel,
        adapter.decoder.config.sequence_parallel,
    )


def clear_train_vision_cache(models: Any, *, reason: str = "explicit") -> None:
    """Invalidate before weight swaps/offload, including .data-based
    restores."""
    if not Envs.RELAX_TRAIN_VISION_CACHE:
        return
    for model in models if isinstance(models, (list, tuple)) else [models]:
        for module in model.modules():
            adapter = getattr(module, "_relax_train_vision_cache", None)
            if adapter is not None:
                adapter.report_and_clear(reason)
