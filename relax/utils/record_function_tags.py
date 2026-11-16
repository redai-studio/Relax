# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Opt-in module ranges, with graph-preserving checkpoint backward ranges.

Backward checkpoint ranges include activation recomputation. Inner backward
ranges are autograd Node execution fragments, without added graph/view nodes.
"""

from __future__ import annotations

import inspect
import json
import os
import threading
from collections import deque
from contextlib import ExitStack
from contextvars import ContextVar
from functools import wraps
from typing import Any

import torch


_phase: ContextVar[str] = ContextVar("rcd_phase", default="fwd")
_checkpoint_layers: ContextVar[list[str] | None] = ContextVar("rcd_checkpoint_layers", default=None)
_layer_depth: ContextVar[int] = ContextVar("rcd_layer_depth", default=0)
_checkpoint_owner: ContextVar[str | None] = ContextVar("rcd_checkpoint_owner", default=None)
_node_backward_enabled = False
_grouped_checkpoint_layers = False
_NODE_METADATA_KEY = "relax_rcd_node_ranges_v1"
_profile_step = -1
_microbatch: ContextVar[int | None] = ContextVar("rcd_microbatch", default=None)
_moe_owner: ContextVar[str] = ContextVar("rcd_moe_owner", default="unknown")
_mok_schedule_records: list[dict[str, Any]] = []


def _wrap_mok_diagnostics() -> None:
    """Tag optional MoK calls and retain only tiny schedule counters for
    export."""
    from mok import functional

    for method, label in (("build_schedule", "Schedule"), ("forward", "Forward"), ("backward", "Backward")):
        original = getattr(functional, method)
        if getattr(original, "_relax_rcd_installed", False):
            continue

        def make_wrapper(original: Any, label: str) -> Any:
            @wraps(original)
            def tagged(*args: Any, **kwargs: Any) -> Any:
                with torch.profiler.record_function(f"[RCD]MoK{label}|{_moe_owner.get()}"):
                    result = original(*args, **kwargs)
                if label == "Schedule" and torch.autograd._profiler_enabled():
                    workspace = args[0] if args else kwargs["workspace"]
                    config = args[1] if len(args) > 1 else kwargs["config"]
                    # No device reads or copies in the measured path. Schedule
                    # owns these small tensors; keep references until export.
                    _mok_schedule_records.append(
                        {
                            "step": _profile_step,
                            "microbatch": _microbatch.get(),
                            "phase": _phase.get(),
                            "owner": _moe_owner.get(),
                            "local_input_tokens": workspace.num_local_tokens,
                            "ep_size": workspace.ep_size,
                            "topk": workspace.topk,
                            "macrobatch_size": config.macrobatch_size,
                            "minibatch_size": config.minibatch_size,
                            "fwd_num_comm_sms": config.fwd_num_comm_sms,
                            "bwd_num_comm_sms": config.bwd_num_comm_sms,
                            "num_tokens_tensor": result.num_tokens,
                            "tokens_per_expert_tensor": result.tokens_per_expert,
                        }
                    )
                return result

            tagged._relax_rcd_installed = True
            return tagged

        setattr(functional, method, make_wrapper(original, label))


def export_profile_diagnostics(trace_dir: str, worker_name: str) -> None:
    """Read schedule counters after capture, outside the measured training
    path."""
    from pathlib import Path

    if not _mok_schedule_records:
        return
    records = list(_mok_schedule_records)
    # The profiler has stopped. Batched device reads here cannot add CPU/GPU
    # synchronizations to the captured forward/backward hot path.
    totals = torch.cat([row["num_tokens_tensor"].reshape(-1) for row in records]).cpu().tolist()
    lengths = [row["tokens_per_expert_tensor"].numel() for row in records]
    experts = torch.cat([row["tokens_per_expert_tensor"].reshape(-1) for row in records]).cpu().tolist()
    exported = []
    offset = 0
    for row, total, length in zip(records, totals, lengths):
        values = {key: value for key, value in row.items() if not key.endswith("_tensor")}
        values.update(
            scheduled_routed_rows=total,
            tokens_per_expert=experts[offset : offset + length],
            macrobatches=(total + row["macrobatch_size"] - 1) // row["macrobatch_size"],
        )
        exported.append(values)
        offset += length
    output = Path(trace_dir) / f"{worker_name}.mok-schedules.json"
    output.write_text(json.dumps({"read_after_capture": True, "records": exported}))
    _mok_schedule_records.clear()


def _tensor_nodes(value: Any) -> set[Any]:
    if isinstance(value, torch.Tensor):
        return {value.grad_fn} if value.grad_fn is not None else set()
    if isinstance(value, dict):
        value = value.values()
    elif not isinstance(value, (tuple, list)):
        return set()
    nodes: set[Any] = set()
    for item in value:
        nodes.update(_tensor_nodes(item))
    return nodes


def _tag_backward_nodes(boundaries: set[Any], outputs: Any, label: str) -> None:
    # Only this invocation's new graph. Inputs bound traversal, parameter leaves
    # are excluded; no global graph cache, no captured Tensor/Node in hook closures.
    pending = list(_tensor_nodes(outputs))
    visited: set[Any] = set()
    while pending:
        node = pending.pop()
        if node in visited or node in boundaries:
            continue
        visited.add(node)
        name = node.name()
        if name.endswith("AccumulateGrad"):
            continue
        # CheckpointFunction has its own exact outer range; do not duplicate it
        # or walk into graph belonging to its inputs from an outer MTP module.
        if name.endswith("CheckpointFunctionBackward"):
            continue
        pending.extend(parent for parent, _ in node.next_functions if parent is not None)
        metadata = node.metadata
        state = metadata.get(_NODE_METADATA_KEY)
        if state is None:
            state = {"labels": [], "active": {}}
            metadata[_NODE_METADATA_KEY] = state

            def pre_hook(_grad_outputs: Any, state: dict = state) -> None:
                stack = ExitStack()
                # Children register first; enclosing parents append on return.
                # Reverse registration order reconstructs outer-to-inner scopes.
                for name in reversed(state["labels"]):
                    if name.startswith("[RCD]MoKbwd|"):
                        token = _moe_owner.set(name.split("|", 1)[1])
                        stack.callback(_moe_owner.reset, token)
                    stack.enter_context(torch.profiler.record_function(name))
                state["active"].setdefault(threading.get_ident(), []).append(stack)

            def post_hook(_grad_inputs: Any, _grad_outputs: Any, state: dict = state) -> None:
                thread = threading.get_ident()
                stacks = state["active"][thread]
                stack = stacks.pop()
                if not stacks:
                    del state["active"][thread]
                stack.close()

            node.register_prehook(pre_hook)
            node.register_hook(post_hook)
        if label not in state["labels"]:
            state["labels"].append(label)


def _wrap_checkpoint_owner(module: Any, suffix: str) -> None:
    original = module._checkpointed_forward

    @wraps(original)
    def tagged(*args: Any, **kwargs: Any) -> Any:
        token = _checkpoint_owner.set(f"[RCD]MTPbwd|{suffix}")
        try:
            return original(*args, **kwargs)
        finally:
            _checkpoint_owner.reset(token)

    module._checkpointed_forward = tagged


def _wrap_forward(
    module: Any,
    kind: str,
    suffix: str,
    *,
    layer: bool = False,
    method: str = "forward",
    cached_outputs: tuple[str, ...] = (),
) -> None:
    original = getattr(module, method)

    @wraps(original)
    def tagged(*args: Any, **kwargs: Any) -> Any:
        phase = _phase.get()
        labels = _checkpoint_layers.get()
        if layer and labels is not None and _layer_depth.get() == 0 and phase == "fwd":
            labels.append(f"[RCD]{kind}bwd|{suffix}")
        token = _layer_depth.set(_layer_depth.get() + 1) if layer else None
        node_scope = not layer or _grouped_checkpoint_layers or (_checkpoint_owner.get() or "").startswith("[RCD]MTP")
        record_nodes = _node_backward_enabled and torch.is_grad_enabled() and node_scope
        # Snapshot before forward: an in-place operation can replace input.grad_fn.
        cached_inputs = tuple(getattr(module, name, None) for name in cached_outputs)
        boundaries = _tensor_nodes((args, kwargs, cached_inputs)) if record_nodes else set()
        moe_token = _moe_owner.set(suffix) if kind == "MoK" else None
        try:
            with torch.profiler.record_function(f"[RCD]{kind}{phase}|{suffix}"):
                result = original(*args, **kwargs)
            # Ordinary layer backward is already the exact checkpoint scope.
            # Legacy MTP owns that scope instead, so its inner QSA/GDN gets node
            # fragments to avoid mislabelling projection/norm as pure layer bwd.
            if record_nodes:
                cached_results = tuple(getattr(module, name, None) for name in cached_outputs)
                _tag_backward_nodes(boundaries, (result, cached_results), f"[RCD]{kind}bwd|{suffix}")
            return result
        finally:
            if moe_token is not None:
                _moe_owner.reset(moe_token)
            if token is not None:
                _layer_depth.reset(token)

    setattr(module, method, tagged)


def _wrap_checkpoint(checkpoint_cls: Any) -> None:
    if getattr(checkpoint_cls, "_relax_rcd_installed", False):
        return
    original_forward = checkpoint_cls.forward
    original_backward = checkpoint_cls.backward

    @wraps(original_forward)
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        labels: list[str] = []
        token = _checkpoint_layers.set(labels)
        depth_token = _layer_depth.set(0)
        try:
            result = original_forward(ctx, *args, **kwargs)
            # A multi-layer checkpoint is one group, not multiple full layer times.
            owner = _checkpoint_owner.get()
            if owner is not None:
                ranges = (owner,)
            elif len(labels) > 1 or (_grouped_checkpoint_layers and labels):
                ranges = ("[RCD]CheckpointGroupbwd|" + "+".join(label.removeprefix("[RCD]") for label in labels),)
            else:
                ranges = tuple(labels)
            ctx._relax_rcd_labels = ranges
            ctx._relax_rcd_owner = owner
            return result
        finally:
            _layer_depth.reset(depth_token)
            _checkpoint_layers.reset(token)

    @wraps(original_backward)
    def backward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        phase_token = _phase.set("recomp")
        checkpoint_token = _checkpoint_layers.set(None)
        owner_token = _checkpoint_owner.set(getattr(ctx, "_relax_rcd_owner", None))
        try:
            with ExitStack() as stack:
                for label in getattr(ctx, "_relax_rcd_labels", ()):
                    stack.enter_context(torch.profiler.record_function(label))
                return original_backward(ctx, *args, **kwargs)
        finally:
            _checkpoint_owner.reset(owner_token)
            _checkpoint_layers.reset(checkpoint_token)
            _phase.reset(phase_token)

    checkpoint_cls.forward = staticmethod(forward)
    checkpoint_cls.backward = staticmethod(backward)
    checkpoint_cls._relax_rcd_installed = True


def _attention_kind(module: Any) -> str | None:
    name = type(module).__name__
    if name == "QwenSparseSelfAttention":
        return "QSA"
    if "GatedDeltaNet" in name or name == "GDN2":
        return "GDN"
    return None


def _wrap_pipeline_steps(schedules: Any, pp_rank: int) -> None:
    """Nest layers inside real non-interleaved pipeline microbatch calls."""
    if getattr(schedules, "_relax_rcd_installed", False):
        return
    original_forward, original_backward = schedules.forward_step, schedules.backward_step
    signature = inspect.signature(original_forward)
    pending: deque[int] = deque()

    @wraps(original_forward)
    def forward(*args: Any, **kwargs: Any) -> Any:
        global _profile_step
        bound = signature.bind(*args, **kwargs)
        mb = bound.arguments.get("current_microbatch")
        if mb is None:
            raise RuntimeError("RCD pipeline tags require current_microbatch")
        if mb == 0:
            pending.clear()  # Also handles a preceding forward-only evaluation.
            _profile_step += 1
        token = _microbatch.set(mb)
        try:
            with torch.profiler.record_function(f"[RCD]mb{mb}fwd|PP{pp_rank}"):
                result = original_forward(*args, **kwargs)
        finally:
            _microbatch.reset(token)
        pending.append(mb)
        return result

    @wraps(original_backward)
    def backward(*args: Any, **kwargs: Any) -> Any:
        if not pending:
            raise RuntimeError("RCD pipeline backward has no matching forward microbatch")
        mb = pending.popleft()
        token = _microbatch.set(mb)
        try:
            with torch.profiler.record_function(f"[RCD]mb{mb}bwd|PP{pp_rank}"):
                return original_backward(*args, **kwargs)
        finally:
            _microbatch.reset(token)

    schedules.forward_step, schedules.backward_step = forward, backward
    schedules._relax_rcd_installed = True


def _instrument_chunks(chunks: Any, rank_suffix: str) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for chunk_index, chunk in enumerate(chunks):
        modules = list(chunk.named_modules())
        layers: list[tuple[str, Any, str]] = []
        for path, module in modules:
            attention = getattr(module, "self_attention", getattr(module, "mixer", None))
            kind = _attention_kind(attention)
            if kind is not None:
                layers.append((path, module, kind))

        for path, module in modules:
            kind = None
            is_layer = False
            owner = next(
                ((p, m, k) for p, m, k in reversed(layers) if not p or path == p or path.startswith(p + ".")),
                None,
            )
            leaf = path.rsplit(".", 1)[-1]
            cls = type(module).__name__
            if owner is not None and owner[1] is module:
                kind, is_layer = owner[2], True
            elif _attention_kind(module) is not None:
                kind = "Attn"
            elif cls == "MoELayer":
                kind = "MoE"
            elif cls == "MoKMegakernel":
                kind = "MoK"
            elif cls == "SharedExpertMLP":
                kind = "SharedExpert"
            elif leaf in ("self_attention_hyper_connection", "mlp_hyper_connection"):
                kind = "ResidualPre"
            elif cls == "QSAIndexer":
                kind = "Indexer"
            elif leaf == "vision_model":
                kind = "Vision"
            elif leaf == "merger" or (
                ".deepstack_merger_list." in path and path.rsplit(".deepstack_merger_list.", 1)[-1].isdigit()
            ):
                kind = "VHead"
            elif leaf == "embedding" or cls == "PerLayerEmbedding":
                kind = "Embed"
            elif "LayerNorm" in cls and "Linear" in cls:
                kind = "NormLinear"
            elif "norm" in leaf.lower() and ("Norm" in cls or "norm" in cls):
                kind = "Norm"
            elif leaf == "mtp" or cls == "MultiTokenPredictionLayer":
                kind = "MTP"
            elif leaf == "output_layer":
                kind = "LMHead"
            if kind is None or (id(module), "forward") in seen:
                continue
            suffix = rank_suffix
            if owner is not None:
                suffix = f"L{getattr(owner[1], 'layer_number', '?')}"
            if ".mtp." in f".{path}.":
                mtp_path = path.split("mtp.", 1)[-1]
                parts = mtp_path.split(".")
                index = next((part for part in parts if part.isdigit()), None)
                suffix += "|mtp" + (index if index is not None else "")
            if kind == "Norm":
                suffix += f"|{leaf}"
            if kind == "ResidualPre":
                suffix += "|attn" if leaf.startswith("self_attention") else "|mlp"
            if "LayerNorm" in cls and "Linear" in cls:
                # This is a fused norm+linear module, never claim an isolated norm.
                kind = "NormLinear"
            _wrap_forward(module, kind, suffix, layer=is_layer)
            seen.add((id(module), "forward"))
            manifest.append(
                {
                    "chunk": chunk_index,
                    "path": path,
                    "class": cls,
                    "kind": kind,
                    "suffix": suffix,
                    "phases": ["fwd", "recomp", "bwd"],
                    "backward_scope": "checkpoint (MTP inner: node fragments)"
                    if is_layer and not _grouped_checkpoint_layers
                    else "node fragments",
                }
            )
            if cls == "SharedExpertMLP":
                # Overlap bypasses forward() and stages tensors in module
                # caches. Tag each real method and only its newly created graph.
                for method in (
                    "pre_forward_comm",
                    "linear_fc1_forward_and_act",
                    "linear_fc2_forward",
                    "post_forward_comm",
                    "get_output",
                ):
                    _wrap_forward(
                        module,
                        "SharedExpert",
                        suffix + "|" + method,
                        method=method,
                        cached_outputs=(
                            "cached_fc1_input",
                            "cached_fc2_input",
                            "cached_fc2_output",
                            "cached_output",
                            "gate_score",
                        ),
                    )
                    manifest.append(
                        {
                            "chunk": chunk_index,
                            "path": path + "." + method,
                            "class": cls,
                            "kind": "SharedExpert",
                            "suffix": suffix + "|" + method,
                            "phases": ["fwd", "recomp", "bwd"],
                            "backward_scope": "node fragments",
                        }
                    )
            if is_layer:
                for method, part in (("_apply_self_attn_bda_step", "attn"), ("_apply_mlp_bda_step", "mlp")):
                    if hasattr(module, method):
                        _wrap_forward(module, "Residual", suffix + "|" + part, method=method)
                        manifest.append(
                            {
                                "chunk": chunk_index,
                                "path": path + "." + method,
                                "class": cls,
                                "kind": "Residual",
                                "suffix": suffix + "|" + part,
                                "phases": ["fwd", "recomp", "bwd"],
                                "backward_scope": "node fragments",
                            }
                        )
            if cls == "MultiTokenPredictionLayer" and hasattr(module, "_checkpointed_forward"):
                _wrap_checkpoint_owner(module, suffix)
                _wrap_forward(module, "MTP", suffix, method="_proj_and_transformer_layer")
                manifest.append(
                    {
                        "chunk": chunk_index,
                        "path": path + "._proj_and_transformer_layer",
                        "class": cls,
                        "kind": "MTP",
                        "suffix": suffix,
                        "phases": ["fwd", "recomp", "bwd"],
                        "backward_scope": "legacy MTP checkpoint and node fragments",
                    }
                )
            if kind == "Indexer" and hasattr(module, "_select_blocks"):
                _wrap_forward(module, "TopK", suffix, method="_select_blocks")
                manifest.append(
                    {
                        "chunk": chunk_index,
                        "path": path + "._select_blocks",
                        "class": cls,
                        "kind": "TopK",
                        "suffix": suffix,
                        "phases": ["fwd", "recomp", "bwd"],
                        "backward_scope": "node fragments (none when frozen)",
                    }
                )
    return manifest


def install_record_function_tags(args: Any, model_chunks: Any) -> list[dict[str, Any]]:
    """Install only on enabled, selected profiler ranks after model setup."""
    if os.environ.get("RELAX_QWEN38_RECORD_FUNCTION_TAGS", "0") != "1" or not args.use_pytorch_profiler:
        return []
    from relax.utils.logging_utils import get_logger
    from relax.utils.profile_utils import _should_profile_train_rank

    if not _should_profile_train_rank(args):
        return []
    if (
        getattr(args, "recompute_granularity", None) != "full"
        or not isinstance(getattr(args, "recompute_num_layers", None), int)
        or getattr(args, "recompute_num_layers", 0) < 1
    ):
        raise RuntimeError("RCD profiling requires full recomputation with a positive recompute_num_layers")
    from megatron.core import parallel_state
    from megatron.core.pipeline_parallel import schedules
    from megatron.core.tensor_parallel import random as tp_random

    if any(getattr(chunk, "_relax_rcd_installed", False) for chunk in model_chunks):
        raise RuntimeError("RCD instrumentation was already installed on model chunks")
    if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
        raise RuntimeError("RCD microbatch tags currently require a non-interleaved pipeline")
    suffix = (
        f"pp{parallel_state.get_pipeline_model_parallel_rank()}"
        f"tp{parallel_state.get_tensor_model_parallel_rank()}"
        f"cp{parallel_state.get_context_parallel_rank()}"
    )
    global _node_backward_enabled, _grouped_checkpoint_layers
    _node_backward_enabled = True
    _grouped_checkpoint_layers = args.recompute_num_layers > 1
    manifest = _instrument_chunks(model_chunks, suffix)
    if any(item["kind"] == "MoK" for item in manifest):
        _wrap_mok_diagnostics()
    if not any(item["kind"] in ("QSA", "GDN") for item in manifest):
        raise RuntimeError("RCD found no QSA/GDN transformer layers on selected rank")
    _wrap_checkpoint(tp_random.CheckpointFunction)
    _wrap_pipeline_steps(schedules, parallel_state.get_pipeline_model_parallel_rank())
    for chunk in model_chunks:
        chunk._relax_rcd_installed = True
    get_logger(__name__).info("RCD tag manifest: %s", json.dumps(manifest, sort_keys=True))
    get_logger(__name__).info(
        "RCD single-layer bwd spans its checkpoint; multi-layer checkpoints use CheckpointGroupbwd. "
        "Grouped layer bwd ranges are node fragments; legacy MTP checkpoint is MTPbwd. "
        "Inner bwd ranges are autograd Node execution fragments, not contiguous module totals. "
        "Frozen components have no synthetic bwd. NormLinear marks fused norm+linear."
    )
    return manifest
