# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Keep the final native two-layer decoder graph on PP1, retaining uniform2
elsewhere."""

from functools import wraps

import torch


def choose(native, minimum_layers=16):
    stats = {"retained": 0, "ordinary": 0}

    @wraps(native)
    def checkpoint(function, distribute, *inputs):
        if (
            getattr(function, "__module__", "").endswith(".modelling_qwen3_vl.transformer_block")
            and function.__name__ == "custom_forward"
            and function.__closure__
        ):
            closure = dict(zip(function.__code__.co_freevars, (cell.cell_contents for cell in function.__closure__)))
            from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.transformer_block import Qwen3VLTransformerBlock

            stack = closure.get("self")
            if isinstance(stack, Qwen3VLTransformerBlock) and stack.num_layers_per_pipeline_rank >= minimum_layers:
                end = closure.get("end")
                start = closure.get("start")
                if end == stack.num_layers_per_pipeline_rank and start == end - 2:
                    if distribute or not torch.is_grad_enabled() or not inputs[0].requires_grad:
                        raise RuntimeError("The final decoder group requires its ordinary native autograd graph")
                    stats["retained"] += 1
                    return function(*inputs)
                stats["ordinary"] += 1
        return native(function, distribute, *inputs)

    return checkpoint, stats


def install(args):
    from megatron.core import parallel_state, tensor_parallel

    if (
        args.tensor_model_parallel_size,
        args.pipeline_model_parallel_size,
        args.context_parallel_size,
        args.recompute_method,
        args.recompute_num_layers,
        args.recompute_granularity,
    ) != (2, 2, 1, "uniform", 2, "full"):
        raise ValueError("Retained group validated for TP2/PP2/CP1 uniform2")
    if parallel_state.get_pipeline_model_parallel_rank() == 1:
        tensor_parallel.checkpoint, stats = choose(tensor_parallel.checkpoint)
        return stats
