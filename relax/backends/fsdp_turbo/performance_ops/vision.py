# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Frame-local vision execution and optional redistribution over an existing
FSDP group."""

import inspect
from contextvars import ContextVar
from dataclasses import dataclass
from types import MethodType

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs
from transformers.vision_utils import get_vision_interpolation_indices_and_weights, get_vision_position_ids


_VISION_CALL = ContextVar("relax_vision_call", default=None)


class _Exchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, send_sizes, recv_sizes, group):
        ctx.send_sizes, ctx.recv_sizes, ctx.group = send_sizes, recv_sizes, group
        output = value.new_empty((sum(recv_sizes), *value.shape[1:]))
        dist.all_to_all_single(output, value.contiguous(), recv_sizes, send_sizes, group=group)
        return output

    @staticmethod
    def backward(ctx, grad):
        output = grad.new_empty((sum(ctx.send_sizes), *grad.shape[1:]))
        dist.all_to_all_single(output, grad.contiguous(), ctx.send_sizes, ctx.recv_sizes, group=ctx.group)
        return output, None, None, None


@dataclass
class VisionTaskDistributor:
    group: dist.ProcessGroup
    cp_group: dist.ProcessGroup

    def encode(self, visual, pixels, grids, device, dtype):
        rank, world = dist.get_rank(self.group), dist.get_world_size(self.group)
        ranks = dist.get_process_group_ranks(self.group)
        owners = tuple(ranks.index(r) for r in dist.get_process_group_ranks(self.cp_group))
        metadata = (
            owners,
            grids,
            None if pixels is None else pixels.shape[-1],
            0 if pixels is None else pixels.shape[0],
            pixels is not None and pixels.requires_grad,
        )
        gathered = [None] * world
        dist.all_gather_object(gathered, metadata, group=self.group)
        merge = visual.config.spatial_merge_size
        tasks = []
        for source, (destinations, source_grids, _, count, _) in enumerate(gathered):
            if any(t <= 0 or h <= 0 or w <= 0 or h % merge or w % merge for t, h, w in source_grids):
                raise ValueError("Vision frames must have positive, merge-aligned dimensions")
            if count != sum(t * h * w for t, h, w in source_grids):
                raise ValueError("Vision grid sizes must match the input patch count")
            if source != min(destinations):
                if source_grids != gathered[min(destinations)][1]:
                    raise ValueError("CP replicas must have identical visual frame geometry")
                continue
            offset = 0
            for t, h, w in source_grids:
                for _ in range(t):
                    tasks.append(dict(source=source, owners=destinations, offset=offset, h=h, w=w, patch_count=h * w))
                    offset += h * w
        if not tasks:
            return None, torch.zeros((), device=device, dtype=dtype)
        width = next(m[2] for m in gathered if m[2] is not None)
        if any(m[2] is not None and m[2] != width for m in gathered):
            raise ValueError("Visual patch widths must match across the FSDP group")
        merge, dim, mlp = visual.config.spatial_merge_size, visual.config.hidden_size, visual.config.intermediate_size
        loads = [0] * world
        costs = [t["patch_count"] * (4 * dim * dim + 2 * dim * mlp) + 2 * t["patch_count"] ** 2 * dim for t in tasks]
        for i in sorted(range(len(tasks)), key=lambda i: (-costs[i], i)):
            target = min(range(world), key=lambda j: (loads[j], j))
            tasks[i]["executor"] = target
            loads[target] += costs[i]
        routes = [[[] for _ in range(world)] for _ in range(world)]
        for task in tasks:
            routes[task["source"]][task["executor"]].append(task)
        patch_send_sizes = [1 + sum(t["patch_count"] for t in routes[rank][j]) for j in range(world)]
        patch_recv_sizes = [1 + sum(t["patch_count"] for t in routes[j][rank]) for j in range(world)]
        track_pixels = torch.is_grad_enabled() and any(m[4] for m in gathered)
        if track_pixels and any(len(m[0]) > 1 for m in gathered):
            raise ValueError("Pixel-input gradients require CP=1; trainable vision weights remain supported with CP")
        zero = torch.zeros((1, width), device=device, dtype=dtype, requires_grad=track_pixels)
        outbound = [
            torch.cat(
                [zero] + [pixels[t["offset"] : t["offset"] + t["patch_count"]].to(dtype) for t in routes[rank][j]],
                dim=0,
            )
            for j in range(world)
        ]
        received = _Exchange.apply(torch.cat(outbound), patch_send_sizes, patch_recv_sizes, self.group)
        local_tasks, local_pixels = [], []
        cursor = 0
        for source, size in enumerate(patch_recv_sizes):
            pos = cursor + 1
            for task in routes[source][rank]:
                local_tasks.append(task)
                local_pixels.append(received[pos : pos + task["patch_count"]])
                pos += task["patch_count"]
            cursor += size
        packed_rows = [[1, t["h"], t["w"]] for t in local_tasks] or [[1, merge, merge]]
        packed = torch.cat(local_pixels) if local_pixels else zero.expand(merge * merge, -1).contiguous()
        packed = packed + received.reshape(-1)[:1].sum() * 0
        grid = torch.tensor(packed_rows, device=device, dtype=torch.long)
        encoded = visual(packed, grid_thw=grid, _relax_vision_grid=packed_rows, return_dict=True).pooler_output
        merged_token_counts = [t["patch_count"] // (merge * merge) for t in local_tasks]
        chunks = list(encoded.split(merged_token_counts)) if local_tasks else []
        sentinel = encoded[:1] * 0
        blocks = [
            torch.cat([sentinel] + [x for t, x in zip(local_tasks, chunks) if j in t["owners"]]) for j in range(world)
        ]
        token_send_sizes = [x.shape[0] for x in blocks]
        incoming = [[t for t in tasks if t["executor"] == j and rank in t["owners"]] for j in range(world)]
        token_recv_sizes = [1 + sum(t["patch_count"] // (merge * merge) for t in row) for row in incoming]
        returned = _Exchange.apply(torch.cat(blocks), token_send_sizes, token_recv_sizes, self.group)
        restored = {}
        cursor = 0
        for size, row in zip(token_recv_sizes, incoming):
            pos = cursor + 1
            for task in row:
                merged_token_count = task["patch_count"] // (merge * merge)
                restored[task["offset"]] = returned[pos : pos + merged_token_count]
                pos += merged_token_count
            cursor += size
        features = torch.cat([restored[k] for k in sorted(restored)]) if restored else None
        return features, returned.reshape(-1)[:1].sum() * 0


def _patch_embed(self, hidden_states, frame_lengths):
    outputs = []
    for frame in hidden_states.split(frame_lengths):
        for chunk in frame.split(self._relax_frame_chunk_size):
            valid = chunk.shape[0]
            chunk = F.pad(chunk, (0, 0, 0, self._relax_frame_chunk_size - valid))
            outputs.append(self._relax_patch_forward(chunk)[:valid])
    return torch.cat(outputs)


def _vision_block(self, hidden_states, cu_seqlens, position_embeddings=None, **kwargs):
    lengths = kwargs.pop("_relax_frame_lengths")
    outputs = []
    start = 0
    for length in lengths:
        local_cu = torch.tensor([0, length], dtype=torch.int32, device="cpu")
        positions = tuple(x[start : start + length] for x in position_embeddings)
        outputs.append(
            self._relax_block_forward(
                hidden_states[start : start + length], local_cu, positions, **{**kwargs, "max_seqlen": length}
            )
        )
        start += length
    return torch.cat(outputs)


@merge_with_config_defaults
@capture_outputs
def _vision_forward(self, hidden_states, grid_thw, **kwargs):
    rows = kwargs.pop("_relax_vision_grid")
    lengths = [h * w for t, h, w in rows for _ in range(t)]
    if not lengths or sum(lengths) != hidden_states.shape[0]:
        raise ValueError("Vision frame geometry must span all patch tokens")
    indices, weights = get_vision_interpolation_indices_and_weights(
        grid_thw,
        num_grid_per_side=self.num_grid_per_side,
        mode=self.interpolation_mode,
        align_corners=self.interpolation_align_corners,
        spatial_merge_size=self.spatial_merge_size,
        kwargs=kwargs,
    )
    position_ids = get_vision_position_ids(grid_thw, self.spatial_merge_size, kwargs=kwargs)
    hidden_states = self.patch_embed(hidden_states, frame_lengths=lengths)
    positions = (self.pos_embed(indices) * weights[:, :, None]).sum(1)
    hidden_states = hidden_states + positions.to(hidden_states.dtype)
    rotary = self.rotary_pos_emb(hidden_states, position_ids)
    for block in self.blocks:
        hidden_states = block(
            hidden_states, cu_seqlens=None, position_embeddings=rotary, _relax_frame_lengths=lengths, **kwargs
        )
    merged = self.merger(hidden_states)
    return BaseModelOutputWithPooling(last_hidden_state=hidden_states, pooler_output=merged)


def _features(self, pixels, grid, modality, **kwargs):
    state = _VISION_CALL.get()
    if state is None:
        raise ValueError("Frame vision adapters require Relax's CPU frame metadata")
    rows = state["grids"].get(modality, [])
    if state["balanced"]:
        merged_token_counts = [t * h * w // self.visual.spatial_merge_size**2 for t, h, w in rows]
        return BaseModelOutputWithPooling(pooler_output=state[modality].split(merged_token_counts))
    return self._relax_image_features(pixels, grid, _relax_vision_grid=rows, **kwargs)


def _image_features(self, pixel_values, image_grid_thw=None, **kwargs):
    return _features(self, pixel_values, image_grid_thw, "image", **kwargs)


def _video_features(self, pixel_values_videos, video_grid_thw=None, **kwargs):
    return _features(self, pixel_values_videos, video_grid_thw, "video", **kwargs)


def _embedding(self, *args, **kwargs):
    output = self._relax_embedding_forward(*args, **kwargs)
    state = _VISION_CALL.get()
    if state is not None and state["balanced"]:
        output = output + state["dependency"].to(output)
    return output


def _model_forward(self, *args, **kwargs):
    grids = kwargs.pop("_relax_vision_grids", {})
    bound = self._relax_forward_signature.bind(*args, **kwargs)
    values = bound.arguments
    state = {"grids": grids, "balanced": self._relax_vision_balance}
    if state["balanced"]:
        if values.get("past_key_values") is not None:
            raise ValueError("Vision balancing supports uncached SFT forwards")
        inputs = values.get("inputs_embeds")
        ids = values.get("input_ids")
        if (ids is None) ^ (inputs is not None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        device = inputs.device if inputs is not None else ids.device
        img, dep_img = self._relax_vision_distributor.encode(
            self.visual,
            values.get("pixel_values"),
            grids.get("image", []),
            device,
            self.visual.dtype,
        )
        vid, dep_vid = self._relax_vision_distributor.encode(
            self.visual,
            values.get("pixel_values_videos"),
            grids.get("video", []),
            device,
            self.visual.dtype,
        )
        state.update(image=img, video=vid, dependency=dep_img + dep_vid)
        if inputs is not None:
            values["inputs_embeds"] = inputs + state["dependency"].to(inputs)
    token = _VISION_CALL.set(state)
    try:
        return self._relax_model_forward(*bound.args, **bound.kwargs)
    finally:
        _VISION_CALL.reset(token)


def install_vision_adapters(model, chunk_size, balance):
    backbone = model.model
    visual = backbone.visual
    visual.forward = MethodType(_vision_forward, visual)
    visual.patch_embed._relax_patch_forward = visual.patch_embed.forward
    visual.patch_embed._relax_frame_chunk_size = chunk_size
    visual.patch_embed.forward = MethodType(_patch_embed, visual.patch_embed)
    for block in visual.blocks:
        block._relax_block_forward = block.forward
        block.forward = MethodType(_vision_block, block)
    backbone._relax_image_features = backbone.get_image_features
    backbone.get_image_features = MethodType(_image_features, backbone)
    backbone.get_video_features = MethodType(_video_features, backbone)
    backbone._relax_model_forward = backbone.forward
    backbone._relax_forward_signature = inspect.signature(backbone.forward)
    backbone._relax_vision_balance = balance
    backbone.forward = MethodType(_model_forward, backbone)
    embedding = backbone.get_input_embeddings()
    embedding._relax_embedding_forward = embedding.forward
    embedding.forward = MethodType(_embedding, embedding)
    model._relax_vision_metadata = True
    backbone._relax_vision_metadata = True


def initialize_vision_distribution(turbo):
    group = turbo.parallel_state.get_fsdp_group()
    cp_group = turbo.parallel_state.get_cp_group()
    if not set(dist.get_process_group_ranks(cp_group)).issubset(dist.get_process_group_ranks(group)):
        raise ValueError("Vision scheduling must contain complete CP groups")
    turbo.model.model._relax_vision_distributor = VisionTaskDistributor(group, cp_group)


def validate_vision_plan(model, config) -> None:
    """Reject rank-dependent child sharding and getter-only replay before
    wrapping."""
    from fsdp_turbo.utils.str_match import module_name_match

    visual = model.model.visual
    if visual.config._attn_implementation != "sdpa_vision":
        raise ValueError("Frame vision execution requires sdpa_vision with CPU frame boundaries")
    repeated = set(visual.patch_embed.modules()) - {visual.patch_embed}
    for block in visual.blocks:
        repeated.update(set(block.modules()) - {block})
    plan = config.distributed.fsdp_plan
    names = list(model.named_modules())
    for name, module in names:
        if module not in repeated or any(module_name_match(p, name) for p in plan.ignored_modules):
            continue
        if any(module_name_match(p, name) for p in plan.apply_modules):
            raise ValueError(f"Frame vision cannot independently shard repeated submodule {name!r}")
    if config.memory.recompute:
        plan = config.memory.recompute_plan
        entries = getattr(plan, "apply_modules", plan) or []
        for entry in entries:
            if any(module_name_match(entry, name) for name, _ in names):
                continue
            owner, _, method = entry.rpartition(".")
            if method in ("get_image_features", "get_video_features") and any(
                module is model.model and module_name_match(owner, name) for name, module in names
            ):
                raise ValueError("Frame vision requires model/block recompute, not getter-only method recompute")
