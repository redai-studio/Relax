# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""LoRA on Qwen3.8's replicated HC and PLE projections."""

from argparse import Namespace
from collections import Counter
from collections.abc import Sequence
from types import MethodType

import torch
import torch.distributed as dist

from relax.utils.env import Envs


HC_TARGETS = ("input_mix_weight_down", "input_mix_weight_up", "block_inject_weight")
PLE_TARGETS = ("key_proj", "value_proj")
_PLE_LORA_RNG_NAME = "relax-ple-lora"


def _ple_lora_forward(self: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
    from megatron.core.transformer.per_layer_embedding import PerLayerEmbedding

    if not self.training:
        return PerLayerEmbedding.forward(self, hidden_states)
    if self.config.sequence_parallel and self.tp_group.size() > 1 and self._cp_packed_seq_params is None:
        raise ValueError("PLE LoRA dropout with CP>1 and TP sequence parallelism requires packed THD inputs")
    tracker = get_cuda_rng_tracker()
    if _PLE_LORA_RNG_NAME not in tracker.get_states():
        raise RuntimeError(
            "Checkpoint lacks the PLE LoRA dropout RNG stream; resume a checkpoint saved with this support, "
            "or use --no-load-rng/--finetune for an intentional warm start"
        )
    # CP reconstructs the same logical tokens on every replica. The surrounding
    # SP RNG stream differs across TP ranks and cannot drive this region.
    with tracker.fork(_PLE_LORA_RNG_NAME):
        return PerLayerEmbedding.forward(self, hidden_states)


def qwen38_lora_targets(model: torch.nn.Module, targets: list[str], args: Namespace) -> list[str]:
    """Resolve special leaf names without changing the canonical export
    targets."""
    special = (*HC_TARGETS, *PLE_TARGETS)
    requested = set(targets).intersection(special)
    if getattr(getattr(model, "config", None), "mhc_variant", None) != "gated_residual" or (
        not requested and all(target.isidentifier() for target in targets)
    ):
        return targets

    from megatron.bridge.peft.utils import wildcard_match
    from megatron.core.transformer.hyper_connection import GatedResidualHyperConnection
    from megatron.core.transformer.per_layer_embedding import PerLayerEmbedding

    if requested and getattr(args, "lora_scope", "all") == "vision":
        raise ValueError("Qwen3.8 HC/PLE LoRA targets require --lora-scope language or all")
    if requested and getattr(args, "lora_adapter_mode", False):
        raise ValueError("Qwen3.8 HC/PLE LoRA requires merge mode for rollout; dynamic adapters are not supported")
    if requested.intersection(PLE_TARGETS) and Envs.RELAX_QWEN38_PLE_STREAMED:
        raise ValueError("PLE projection LoRA requires RELAX_QWEN38_PLE_STREAMED=0")

    resolved = []
    invalid_patterns = set()
    for prefix, owner in model.named_modules():
        if isinstance(owner, GatedResidualHyperConnection):
            leaves = HC_TARGETS
        elif isinstance(owner, PerLayerEmbedding):
            leaves = PLE_TARGETS
        else:
            continue
        for leaf in leaves:
            if getattr(owner, leaf, None) is None:
                continue
            name = f"{prefix}.{leaf}" if prefix else leaf
            if leaf in requested:
                resolved.append(name)
            invalid_patterns.update(
                target for target in targets if target not in special and wildcard_match(target, name)
            )

    # Keep expectations before freeze/scope filtering so a requested adapter cannot
    # silently disappear. A PP chunk without these owners is valid.
    model._relax_qwen38_lora_targets = tuple(resolved)
    # Full Megatron paths are not portable HF target names. Report these after
    # every PP stage is built, including stages on which the pattern matches none.
    model._relax_qwen38_lora_errors = tuple(
        f"{pattern}: use canonical HC/PLE target names instead of full paths or wildcards"
        for pattern in sorted(invalid_patterns)
    )
    remaining = [target for target in targets if target not in requested]
    return remaining + resolved or ["__relax_lora_no_match__"]


@torch.no_grad()
def validate_qwen38_lora(models: Sequence[torch.nn.Module], args: Namespace) -> None:
    """Check after freezing/DDP, then synchronize fresh adapters before
    restore.

    Called once after all VP chunks are built. World validation must not run in
    the provider, whose invocation count can differ between pipeline stages.
    """
    if not any(hasattr(model, "_relax_qwen38_lora_targets") for model in models):
        return

    from megatron.bridge.peft.lora_layers import LinearAdapter
    from megatron.core.transformer.per_layer_embedding import PerLayerEmbedding

    from relax.utils.megatron_peft_utils import is_lora_adapter_param

    counts: Counter[str] = Counter()
    errors: list[str] = []
    adapters: list[torch.nn.Parameter] = []
    ple_owners: list[torch.nn.Module] = []
    for model in models:
        errors.extend(getattr(model, "_relax_qwen38_lora_errors", ()))
        modules = dict(model.named_modules())
        targets = getattr(model, "_relax_qwen38_lora_targets", ())
        for owner_name in dict.fromkeys(name.rpartition(".")[0] for name in targets):
            if isinstance(modules[owner_name], PerLayerEmbedding):
                ple_owners.append(modules[owner_name])
            for name, parameter in modules[owner_name].named_parameters():
                if parameter.requires_grad and not is_lora_adapter_param(f".{name}"):
                    errors.append(f"{owner_name}.{name}: HC/PLE base parameters must stay frozen")
        for name in targets:
            counts[name.rsplit(".", 1)[-1]] += 1
            wrapper = modules[name]
            base, adapter = getattr(wrapper, "to_wrap", None), getattr(wrapper, "adapter", None)
            if not isinstance(base, torch.nn.Linear) or type(adapter) is not LinearAdapter:
                errors.append(f"{name}: expected a plain LinearAdapter")
                continue
            if adapter.linear_in.weight.shape != (args.lora_rank, base.in_features) or (
                adapter.linear_out.weight.shape != (base.out_features, args.lora_rank)
            ):
                errors.append(f"{name}: incorrect adapter shape")
            for parameter in adapter.parameters():
                if not parameter.requires_grad:
                    errors.append(f"{name}: requested adapter was frozen; check freeze/only-train patterns")
                if getattr(base.weight, "sequence_parallel", False) and (
                    not getattr(parameter, "sequence_parallel", False)
                    or getattr(parameter, "average_gradients_across_tp_domain", False)
                    or getattr(parameter, "tensor_model_parallel", False)
                ):
                    errors.append(f"{name}: adapter must inherit sequence-parallel TP SUM")
                adapters.append(parameter)

    gathered = [(counts, errors)]
    if dist.is_initialized():
        from relax.utils.distributed_utils import get_gloo_group

        group = get_gloo_group()
        gathered = [None] * dist.get_world_size(group=group)
        dist.all_gather_object(gathered, (counts, errors), group=group)
    totals: Counter[str] = Counter()
    failures = []
    for local_counts, local_errors in gathered:
        totals.update(local_counts)
        failures.extend(local_errors)
    for target in set(args.lora_target_modules).intersection((*HC_TARGETS, *PLE_TARGETS)):
        if not totals[target]:
            failures.append(f"{target}: no Qwen3.8 HC/PLE projection matched on any pipeline stage")
    if failures:
        raise ValueError("Invalid Qwen3.8 HC/PLE LoRA: " + "; ".join(dict.fromkeys(failures)))

    # get_model has already moved parameters to their training device. Its
    # optional DDP initialization broadcast covers DP/CP, not TP copies of
    # plain LinearAdapter. Never repeat this after loading a checkpoint.
    if dist.is_initialized():
        from megatron.core import parallel_state

        if args.lora_dropout > 0 and totals:
            if parallel_state.get_data_parallel_world_size() > 1 and not getattr(
                args, "data_parallel_random_init", False
            ):
                raise ValueError(
                    "HC/PLE LoRA dropout with DP>1 requires --data-parallel-random-init so checkpoints "
                    "preserve each data-parallel rank's RNG state for variable-length batches"
                )
            if ple_owners and parallel_state.get_context_parallel_world_size() > 1:
                from megatron.core.tensor_parallel.random import get_cuda_rng_tracker

                tracker = get_cuda_rng_tracker()
                if _PLE_LORA_RNG_NAME not in tracker.get_states():
                    # The default seed includes PP and optional DP, but not TP/CP.
                    # Register before activation checkpointing and RNG restore;
                    # never reset an existing stream when another VP chunk is built.
                    tracker.add(_PLE_LORA_RNG_NAME, torch.cuda.initial_seed() + 0x504C45)
                for owner in ple_owners:
                    owner.forward = MethodType(_ple_lora_forward, owner)
        group = parallel_state.get_tensor_model_parallel_group()
        if dist.get_world_size(group=group) > 1:
            source = dist.get_global_rank(group, 0)
            for parameter in dict.fromkeys(adapters):
                dist.broadcast(parameter.data, src=source, group=group)
