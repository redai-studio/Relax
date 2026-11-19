# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Opt-in decision representations for the inspected SGLang source snapshot."""

from __future__ import annotations

import functools
import hashlib
import importlib.metadata
import importlib.util
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    import torch


# The installed distribution reports 0.5.9 but carries local changes. Pin the
# actual code contract, not just the package version. Unreviewed builds fail closed.
_SOURCE_SHA256 = {
    "layers/logits_processor.py": "e5631a0e51e40a0665cbe93b673f301792b37b5dbf2bf28221cef8f9c484f5cd",
    "models/qwen3_vl.py": "8e899df54859680b545bd54a45d72e22ead4821eed6e4f3e62c491b5c5dddc45",
    "managers/scheduler_output_processor_mixin.py": "d536f7855f89b8c363b481c492612a33119a886d14b50b74ddf253dce77601f3",
    "managers/scheduler_update_weights_mixin.py": "d96dd1cbdc4109671ecff0962cd17a0fa7bb97f6da3f48bd77a8f03851e28e0b",
}
SEMANTICS = "mean_final_norm_pre_lm_head_predicting_generated_tokens_v1"


class DecisionPool:
    def __init__(self, vector: torch.Tensor, version: str) -> None:
        import torch

        self.total = torch.zeros_like(vector, dtype=torch.float32)
        self.count = torch.zeros((), device=vector.device, dtype=torch.float32)
        self.version = version

    def add(self, vector: torch.Tensor, *, include: bool, version: str) -> None:
        if version != self.version:
            raise RuntimeError("Weights changed within a hidden-state request")
        if include:
            self.total.add_(vector.float())
            self.count.add_(1)

    def finish(self) -> dict[str, Any]:
        import torch

        # A single transfer at completion; no per-token D2H or synchronization.
        packed = torch.cat((self.total / self.count.clamp_min(1), self.count.reshape(1))).cpu().tolist()
        return {
            "vector": packed[:-1],
            "token_count": int(packed[-1]),
            "hidden_dim": len(packed) - 1,
            "layer": "final_norm",
            "semantics": SEMANTICS,
            "policy_version": self.version,
        }


def validate_server(server_args: Any) -> None:
    if importlib.metadata.version("sglang") != "0.5.9":
        raise RuntimeError("Decision hidden collection supports inspected SGLang 0.5.9 only")
    root = Path(importlib.util.find_spec("sglang").origin).parent / "srt"
    for name, expected in _SOURCE_SHA256.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Unreviewed SGLang hidden collection interface: {name}")
    required = {"disable_cuda_graph": True, "disable_overlap_schedule": True, "pp_size": 1, "dp_size": 1}
    for name, value in required.items():
        if getattr(server_args, name, None) != value:
            raise RuntimeError(f"Decision hidden collection requires {name}={value}")
    for name in ("speculative_algorithm", "enable_lora", "enable_dp_attention", "enable_torch_compile"):
        if getattr(server_args, name, None):
            raise RuntimeError(f"Decision hidden collection does not support {name}")
    if getattr(server_args, "disaggregation_mode", "null") != "null":
        raise RuntimeError("Decision hidden collection does not support disaggregated inference")


def apply_patch() -> None:
    """Install only inside opted-in scheduler subprocesses, before model
    loading."""
    import torch
    from sglang.srt.layers.logits_processor import LogitsProcessor
    from sglang.srt.managers.scheduler import Scheduler

    if getattr(Scheduler, "_relax_decision_patch", False):
        return
    original_init = Scheduler.__init__

    @functools.wraps(original_init)
    def initialize(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        validate_server(self.server_args)
        if self.model_config.hf_config.architectures != ["Qwen3VLForConditionalGeneration"]:
            raise RuntimeError("Decision hidden collection supports Qwen3VLForConditionalGeneration only")
        self._relax_weight_instance = uuid.uuid4().hex
        self._relax_weight_revision = 0
        self._relax_checkpoint = str(self.server_args.model_path)

    Scheduler.__init__ = initialize
    original_prune = LogitsProcessor._get_hidden_states_to_store
    original_forward = LogitsProcessor.forward

    @functools.wraps(original_prune)
    def capture(self: Any, *args: Any, **kwargs: Any) -> Any:
        # Inspected snapshot signature: pruned_states is post final-norm, and
        # sample_indices selects only rows whose logits predict new tokens.
        states, indices = args[3], args[6]
        self._relax_decision_rows = (states[indices] if indices is not None else states).detach().clone()
        return original_prune(self, *args, **kwargs)

    @functools.wraps(original_forward)
    def forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            output = original_forward(self, *args, **kwargs)
            output._relax_decision_rows = self._relax_decision_rows
            return output
        finally:
            self.__dict__.pop("_relax_decision_rows", None)

    LogitsProcessor._get_hidden_states_to_store = capture
    LogitsProcessor.forward = forward
    original_collect = Scheduler.maybe_collect_customized_info

    @functools.wraps(original_collect)
    def collect(self: Any, i: int, req: Any, output: Any) -> None:
        original_collect(self, i, req, output)
        # Other TP ranks neither accumulate nor export. Qwen3-VL's final
        # residual hidden dimension is replicated, not vocabulary-sharded.
        if self.tp_rank != 0 or not req.rid.startswith("req_"):
            return
        if req.stream or req.return_hidden_states or req.grammar is not None:
            raise RuntimeError(
                "Decision hidden collection requires nonstreaming requests without grammar or return_hidden_states"
            )
        vector = output._relax_decision_rows[i]
        if vector.ndim != 1 or vector.numel() != self.model_config.hf_text_config.hidden_size:
            raise RuntimeError("Incomplete decision hidden vector")
        version = f"{self._relax_weight_instance}:{self._relax_weight_revision}"
        pool = getattr(req, "_relax_decision_pool", None)
        if pool is None:
            pool = req._relax_decision_pool = DecisionPool(vector, version)
        # Keep reasoning/action delimiters. Exclude only termination IDs,
        # not all tokenizer special tokens (e.g. think/tool tags).
        terminal_ids = set(req.eos_token_ids or ()) | set(req.sampling_params.stop_token_ids or ())
        terminal_ids.update(getattr(req.tokenizer, "additional_stop_token_ids", None) or ())
        eos = getattr(req.tokenizer, "eos_token_id", None)
        if eos is not None:
            terminal_ids.add(eos)
        pool.add(vector, include=req.output_ids[-1] not in terminal_ids, version=version)
        if req.finished():
            finish(self, req)

    def finish(self: Any, req: Any) -> None:
        if getattr(req, "_relax_decision_saved", False):
            return
        pool = getattr(req, "_relax_decision_pool", None)
        if pool is None:
            # Aborted before its first sampled token: an empty representation.
            pool = DecisionPool(
                torch.zeros(self.model_config.hf_text_config.hidden_size, device=self.device),
                f"{self._relax_weight_instance}:{self._relax_weight_revision}",
            )
        record = pool.finish()
        record["checkpoint"] = self._relax_checkpoint
        record["backend_request_id"] = req.rid
        # HTTP transport retries can execute the same rid again. Distinguish
        # physical generations even when the first response is lost.
        record["request_id"] = f"{req.rid}:{uuid.uuid4().hex}"
        context = (req.sampling_params.custom_params or {}).get("relax_decision_context")
        if context is None:
            raise RuntimeError("Decision hidden collection requires request task/session/turn metadata")
        from relax.agentic.decision_hidden import write_record

        write_record(
            "requests",
            {
                **context,
                **record,
                "finish_reason": req.finished_reason.to_json() if req.finished_reason is not None else None,
            },
        )
        req.customized_info = req.customized_info or {}
        req.customized_info["relax_decision_hidden"] = [None] * req.send_token_offset + [record]
        req.__dict__.pop("_relax_decision_pool", None)
        req._relax_decision_saved = True

    Scheduler.maybe_collect_customized_info = collect

    original_stream = Scheduler.stream_output_generation

    @functools.wraps(original_stream)
    def stream(self: Any, reqs: Any, *args: Any, **kwargs: Any) -> Any:
        if self.tp_rank == 0:
            for req in reqs:
                if req.rid.startswith("req_") and req.finished():
                    finish(self, req)
        return original_stream(self, reqs, *args, **kwargs)

    Scheduler.stream_output_generation = stream

    def wrap_update(method: Any) -> Any:
        @functools.wraps(method)
        def update(self: Any, request: Any) -> Any:
            result = method(self, request)
            if not result.success:
                raise RuntimeError("Weight update failed while decision collection is enabled")
            # Revision is tied to actual successful weight mutations, including
            # chunked transfers; never inferred from a training global_step.
            self._relax_weight_revision += 1
            self._relax_checkpoint = str(getattr(request, "model_path", None) or self._relax_checkpoint)
            return result

        return update

    for name in (
        "update_weights_from_disk",
        "update_weights_from_tensor",
        "update_weights_from_distributed",
        "update_weights_from_ipc",
        "post_process_weights",
    ):
        setattr(Scheduler, name, wrap_update(getattr(Scheduler, name)))
    Scheduler._relax_decision_patch = True
