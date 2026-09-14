# Copyright (c) 2026 Relax Authors. All Rights Reserved.

# Adapt from https://github.com/NVIDIA/Megatron-LM/blob/b1efb3c7126ef7615e8c333432d76e08038e17ff/pretrain_gpt.py
import argparse
import inspect
import json
import logging
import os
import pickle
import re
from collections.abc import Sequence
from contextlib import nullcontext
from typing import Any, Literal

import torch
import torch.distributed as dist
from megatron.core import mpu
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.transformer.spec_utils import import_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.arguments import core_transformer_config_from_args

from relax.utils.device import is_npu_available, make_current_torch_device
from relax.utils.logging_utils import get_logger
from relax.utils.megatron_peft_utils import (
    _lora_module_key,
    build_lora_peft,
    count_adapter_parameters,
    install_gdn_gate_mask_hooks,
    is_lora_adapter_mode,
    is_lora_adapter_param,
    is_lora_enabled,
    is_lora_merge_mode,
    scope_target_modules_to_region,
    summarize_lora_modules,
)
from relax.utils.misc import load_function
from relax.utils.training.ppo_utils import (
    ensure_sequence_classification_head_trainable,
    install_critic_value_head_in_provider,
    install_sequence_classification_head_in_provider,
)

from .conditional_branch_sync import install_conditional_branch_sync


logger = get_logger(__name__)


def configure_mtp_detach_paths(args: argparse.Namespace, model: torch.nn.Module) -> None:
    """Propagate MTP detach-path settings to every Megatron model config."""
    detach_paths = frozenset(getattr(args, "mtp_detach_paths", ("embedding", "backbone", "lm-head")))
    seen_configs: set[int] = set()
    for module in model.modules():
        config = getattr(module, "config", None)
        if config is None or id(config) in seen_configs:
            continue
        setattr(config, "mtp_detach_embedding", "embedding" in detach_paths)
        setattr(config, "mtp_detach_backbone", "backbone" in detach_paths)
        setattr(config, "mtp_detach_lm_head", "lm-head" in detach_paths)
        seen_configs.add(id(config))


def _make_json_safe(value: Any, seen: set[int] | None = None) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    if seen is None:
        seen = set()

    if isinstance(value, dict):
        return {str(k): _make_json_safe(v, seen) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_make_json_safe(v, seen) for v in value]

    obj_id = id(value)
    if obj_id in seen:
        return str(value)

    if hasattr(value, "__dict__"):
        seen.add(obj_id)
        try:
            return {str(k): _make_json_safe(v, seen) for k, v in vars(value).items()}
        finally:
            seen.remove(obj_id)

    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _dump_provider_config(provider: Any, save_path: str) -> None:
    os.makedirs(save_path, exist_ok=True)

    pkl_path = os.path.join(save_path, "transformer_config.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(provider, f)
    logger.info(f"Provider config saved to {pkl_path}")

    json_path = os.path.join(save_path, "transformer_config.json")
    with open(json_path, "w") as f:
        json.dump(_make_json_safe(provider), f, indent=2, ensure_ascii=False)
    logger.info(f"Provider config saved to {json_path}")


# CP-PROBE: one-shot forward-pre-hook on the first attention module to verify that
# context parallelism actually splits the sequence dimension at the attention input.
# Compare seq_len across CP=1 vs CP=2 runs — it must halve.  Remove after verifying.
_CP_PROBE_INSTALLED = False


def _maybe_mark_unsplit_forward(args: argparse.Namespace, model: torch.nn.Module) -> None:
    """Mark `args.uses_unsplit_forward` when the bridge produces a model whose
    forward expects UNSPLIT input + global cu_seqlens + attention_mask and does
    CP+SP splitting internally (Qwen3VLModel family — used for Qwen3-VL and
    text-only Qwen3.5 / Qwen3.6 sharing the same architecture).

    Read by data.py / loss.py to build unsplit tokens + tp*cp*2-aligned
    cu_seqlens instead of the pre-split + cp-multiplied form, and by model.py
    to route those through the forward.
    """
    try:
        from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model import Qwen3VLModel
    except ImportError:
        return
    if isinstance(model, Qwen3VLModel):
        args.uses_unsplit_forward = True


def _install_cp_probe(model: torch.nn.Module) -> None:
    global _CP_PROBE_INSTALLED
    if _CP_PROBE_INSTALLED:
        return

    from megatron.core import mpu

    cp_size = mpu.get_context_parallel_world_size()
    cp_rank = mpu.get_context_parallel_rank()
    tp_rank = mpu.get_tensor_model_parallel_rank()
    state = {"n": 0}

    target_classes = (
        "DotProductAttention",
        "TEDotProductAttention",
        "FusedAttention",
        "FlashAttention",
    )

    def hook(module, args, kwargs):
        if state["n"] >= 2 or tp_rank != 0:
            return
        shapes: dict[str, object] = {}
        for name in (
            "query",
            "key",
            "value",
            "q",
            "k",
            "v",
            "hidden_states",
            "query_layer",
            "key_layer",
            "value_layer",
        ):
            t = kwargs.get(name)
            if torch.is_tensor(t):
                shapes[name] = tuple(t.shape)
        for i, t in enumerate(args):
            if torch.is_tensor(t):
                shapes[f"arg{i}"] = tuple(t.shape)
        for name in ("cu_seqlens_q", "cu_seqlens_kv"):
            t = kwargs.get(name)
            if torch.is_tensor(t):
                shapes[name] = t.tolist()  # one-shot sync, OK for probe
        logger.debug(f"[CP-PROBE] cp_rank={cp_rank}/{cp_size} module={type(module).__name__} shapes={shapes}")
        state["n"] += 1

    skip_prefixes = ("vision_model", "visual", "vit", "image_encoder", "projector", "audio")

    def is_llm_backbone(n: str) -> bool:
        return not any(p in n for p in skip_prefixes)

    matches = [(n, m) for n, m in model.named_modules() if type(m).__name__ in target_classes]
    llm_matches = [(n, m) for n, m in matches if is_llm_backbone(n)]
    chosen = llm_matches or matches  # fallback to vision if no LLM backbone in this stage

    if chosen:
        name, m = chosen[0]
        m.register_forward_pre_hook(hook, with_kwargs=True)
        logger.debug(
            f"[CP-PROBE] hook installed on '{name}' ({type(m).__name__}) "
            f"cp_size={cp_size} cp_rank={cp_rank} "
            f"(total_attn_modules={len(matches)}, llm_backbone={len(llm_matches)})"
        )
        _CP_PROBE_INSTALLED = True
        return

    candidates = [(n, type(m).__name__) for n, m in model.named_modules() if "attention" in n.lower()][:8]
    logger.warning(
        f"[CP-PROBE] no attention module matched on this stage (cp_rank={cp_rank}); "
        f"attention-like candidates: {candidates}"
    )


def get_model_provider_func(
    args: argparse.Namespace,
    role: Literal["actor", "critic"] = "actor",
):
    # Support custom model provider path (similar to --custom-rm-path for reward models)
    if getattr(args, "custom_model_provider_path", None):

        def wrapped_model_provider(
            pre_process: bool = True, post_process: bool = True, vp_stage: int | None = None
        ) -> GPTModel:
            custom_model_provider = load_function(args.custom_model_provider_path)
            # Check if the custom provider supports vp_stage parameter
            has_vp_stage = "vp_stage" in inspect.signature(custom_model_provider).parameters
            if has_vp_stage:
                model = custom_model_provider(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
            else:
                model = custom_model_provider(pre_process=pre_process, post_process=post_process)
            configure_mtp_detach_paths(args, model)
            # Apply critic output layer if needed
            install_critic_value_head_in_provider(model, role, post_process)
            install_sequence_classification_head_in_provider(model, args, role, post_process)
            _maybe_mark_unsplit_forward(args, model)
            install_conditional_branch_sync(args, model)
            _install_cp_probe(model)
            return model

        if is_lora_enabled(args) and getattr(args, "task_type", "causal_lm") == "seq_cls":
            wrapped_model_provider = wrap_model_provider_with_lora(wrapped_model_provider, args)
        return wrapped_model_provider

    if args.megatron_to_hf_mode == "bridge":
        from megatron.bridge import AutoBridge

        bridge = AutoBridge.from_hf_pretrained(args.hf_checkpoint, trust_remote_code=True)
        provider = bridge.to_megatron_provider(load_weights=False)
        # Override provider attributes with matching args values
        bridge_keys = [
            "attention_backend",
            "tensor_model_parallel_size",
            "sequence_parallel",
            "pipeline_model_parallel_size",
            "virtual_pipeline_model_parallel_size",
            "context_parallel_size",
            "linear_cp_mode",
            "expert_model_parallel_size",
            "expert_tensor_parallel_size",
            "variable_seq_lengths",
            "dsa_indexer_loss_coeff",
            "dsa_indexer_use_sparse_loss",
            "attention_softmax_in_fp32",
            "masked_softmax_fusion",
            "bias_dropout_fusion",
            "apply_rope_fusion",
            "recompute_granularity",
            "recompute_method",
            "recompute_num_layers",
            "distribute_saved_activations",
            "moe_router_load_balancing_type",
            "moe_router_dtype",
            "moe_aux_loss_coeff",
            "moe_token_dispatcher_type",
            "moe_shared_expert_overlap",
            "moe_enable_deepep",
            "moe_flex_dispatcher_backend",
            "use_audio_in_video",
            "freeze_language_model",
            "freeze_vision_model",
            "freeze_vision_projection",
            "freeze_audio_model",
            "freeze_audio_projection",
            # https://github.com/redai-studio/Megatron-Bridge/commit/960bb5f18800d3e1fb9815e95daa185ab06c09ea
            "vision_dp_when_tp",
            "vision_dp_when_cp",
            "calculate_per_token_loss",
            "cross_entropy_loss_fusion",
            "cross_entropy_fusion_impl",
            "mtp_num_layers",
            "mtp_loss_scaling_factor",
            # "position_embedding_type", # Use default values of megatron-bridge, no need to pass
            # Allow CLI to override layer count / MoE frequency for layer-reduced training
            "num_layers",
            "moe_layer_freq",
            # Kimi K2 / MLA / MoE override surface — required because published K2 configs
            # declare DeepseekV3ForCausalLM and route through DeepSeekV3Bridge, which has
            # different defaults than what slime's K2 launch scripts assume.
            "q_lora_rank",
            "kv_lora_rank",
            "qk_head_dim",
            "qk_pos_emb_head_dim",
            "v_head_dim",
            "rotary_scaling_factor",
            "rotary_base",
            "moe_router_pre_softmax",
            "moe_router_enable_expert_bias",
            "moe_router_bias_update_rate",
            "moe_permute_fusion",
            "moe_grouped_gemm",
            "moe_shared_expert_intermediate_size",
            "moe_router_topk",
            "moe_router_num_groups",
            "moe_router_group_topk",
            "moe_router_topk_scaling_factor",
            "moe_router_score_function",
            "moe_ffn_hidden_size",
            # "position_embedding_type", # Use default values of megatron-bridge, no need to pass
            # Dynamic CP related args
            "dynamic_context_parallel",
        ]

        args_dict = vars(args)
        for attr in vars(provider):
            if attr in args_dict and attr in bridge_keys:
                old_val = getattr(provider, attr)
                new_val = args_dict[attr]
                if getattr(args, "dynamic_context_parallel", False):
                    if attr == "dynamic_context_parallel":
                        new_val = False
                if old_val != new_val:
                    logger.info(f"Override provider.{attr}: {old_val!r} -> {new_val!r}")
                setattr(provider, attr, new_val)

        # Handle name-mismatched attributes that require explicit mapping
        if getattr(args, "decoder_first_pipeline_num_layers", None) is not None:
            provider.num_layers_in_first_pipeline_stage = args.decoder_first_pipeline_num_layers
        if getattr(args, "decoder_last_pipeline_num_layers", None) is not None:
            provider.num_layers_in_last_pipeline_stage = args.decoder_last_pipeline_num_layers
        if hasattr(args, "gradient_accumulation_fusion"):
            provider.gradient_accumulation_fusion = args.gradient_accumulation_fusion
        if is_npu_available:
            for key, value in vars(args).items():
                if not hasattr(provider, key):
                    setattr(provider, key, value)

        if args.fp16:
            provider.fp16 = True
            provider.bf16 = False
            provider.params_dtype = torch.float16
        elif args.bf16:
            provider.fp16 = False
            provider.bf16 = True
            provider.params_dtype = torch.bfloat16

        provider.finalize()

        # Pickle provider for offline inspection / reproducibility (only on rank 0)
        if not dist.is_initialized() or dist.get_rank() == 0:
            save_path = getattr(args, "save", None) or "/tmp/relax"
            _dump_provider_config(provider, save_path)

        original_provide = provider.provide

        def provide_with_cp_probe(*p_args, **p_kwargs):
            model = original_provide(*p_args, **p_kwargs)
            configure_mtp_detach_paths(args, model)
            post_process = p_kwargs.get("post_process", p_args[1] if len(p_args) > 1 else True)
            install_critic_value_head_in_provider(model, role, post_process, stash_lm_head=True)
            install_sequence_classification_head_in_provider(
                model,
                args,
                role,
                post_process,
                stash_lm_head=True,
            )
            _maybe_mark_unsplit_forward(args, model)
            install_conditional_branch_sync(args, model)
            _install_cp_probe(model)
            return model

        if is_lora_enabled(args):
            provide_with_cp_probe = wrap_model_provider_with_lora(provide_with_cp_probe, args)

        return provide_with_cp_probe

    def model_provider(pre_process: bool = True, post_process: bool = True, vp_stage: int | None = None) -> GPTModel:
        """Builds the model.

        If you set the use_legacy_models to True, it will return the legacy GPT model and if not the mcore GPT model.

        Args:
            pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
            post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


        Returns:
            Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
        """
        use_te = args.transformer_impl == "transformer_engine"

        # Experimental loading arguments from yaml
        config: TransformerConfig = core_transformer_config_from_args(args)

        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
            # Allow the spec to be a function so that user can use customized Megatron easier.
            if callable(transformer_layer_spec):
                transformer_layer_spec = transformer_layer_spec(args, config, vp_stage)
        else:
            if args.num_experts:
                # Define the decoder block spec
                kwargs = {
                    "use_transformer_engine": use_te,
                }
                if vp_stage is not None:
                    kwargs["vp_stage"] = vp_stage
                transformer_layer_spec = get_gpt_decoder_block_spec(config, **kwargs)
            else:
                # Define the decoder layer spec
                if use_te:
                    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                        num_experts=args.num_experts,
                        moe_grouped_gemm=args.moe_grouped_gemm,
                        qk_layernorm=args.qk_layernorm,
                        multi_latent_attention=args.multi_latent_attention,
                        moe_use_legacy_grouped_gemm=args.moe_use_legacy_grouped_gemm,
                    )
                else:
                    transformer_layer_spec = get_gpt_layer_local_spec(
                        num_experts=args.num_experts,
                        moe_grouped_gemm=args.moe_grouped_gemm,
                        qk_layernorm=args.qk_layernorm,
                        multi_latent_attention=args.multi_latent_attention,
                        moe_use_legacy_grouped_gemm=args.moe_use_legacy_grouped_gemm,
                    )

        build_model_context = nullcontext
        build_model_context_args = {}
        if args.fp8_param_gather:
            try:
                from transformer_engine.pytorch import fp8_model_init

                build_model_context = fp8_model_init
                build_model_context_args["enabled"] = True

                # Check if fp8_model_init supports preserve_high_precision_init_val
                if "preserve_high_precision_init_val" in inspect.signature(fp8_model_init).parameters:
                    build_model_context_args["preserve_high_precision_init_val"] = True
            except Exception as e:
                raise RuntimeError(
                    "--fp8-param-gather requires `fp8_model_init` from TransformerEngine, but not found."
                ) from e

        kwargs = {
            "config": config,
            "transformer_layer_spec": transformer_layer_spec,
            "vocab_size": args.padded_vocab_size,
            "max_sequence_length": args.max_position_embeddings,
            "pre_process": pre_process,
            "post_process": post_process,
            "fp16_lm_cross_entropy": args.fp16_lm_cross_entropy,
            "parallel_output": True,
            "share_embeddings_and_output_weights": not args.untie_embeddings_and_output_weights,
            "position_embedding_type": args.position_embedding_type,
            "rotary_percent": args.rotary_percent,
            "rotary_base": args.rotary_base,
            "rope_scaling": args.use_rope_scaling,
        }

        if vp_stage is not None:
            kwargs["vp_stage"] = vp_stage

        if args.mtp_num_layers:
            from megatron.core.models.gpt.gpt_layer_specs import get_gpt_mtp_block_spec

            mtp_kwargs = {
                "use_transformer_engine": use_te,
            }
            if vp_stage is not None:
                mtp_kwargs["vp_stage"] = vp_stage

            mtp_block_spec = get_gpt_mtp_block_spec(config, transformer_layer_spec, **mtp_kwargs)
            kwargs["mtp_block_spec"] = mtp_block_spec

        with build_model_context(**build_model_context_args):
            model = GPTModel(**kwargs)

        configure_mtp_detach_paths(args, model)
        install_critic_value_head_in_provider(model, role, post_process)
        install_sequence_classification_head_in_provider(model, args, role, post_process)

        _maybe_mark_unsplit_forward(args, model)
        install_conditional_branch_sync(args, model)
        _install_cp_probe(model)
        return model

    if is_lora_enabled(args):
        model_provider = wrap_model_provider_with_lora(model_provider, args)

    return model_provider


def _global_vision_lora_count(local_count: int, args) -> int:
    """Max vision-region LoRA module count across all ranks (adapter mode
    only).

    The vision tower lives on a single PP stage, so only some ranks can observe it.
    A rank-local check would raise on those ranks while the rest marched on into the
    next collective and hung, so the count is all-reduced first and every rank then
    reaches the same verdict.

    Gated on adapter mode, which is uniform across ranks — no rank can skip the
    collective while another enters it. Returns ``local_count`` unchanged when
    ``torch.distributed`` is not initialized (single-process tests).
    """
    if not is_lora_adapter_mode(args):
        return 0
    if not dist.is_initialized():
        return local_count
    group = dist.group.WORLD
    # NCCL/HCCL cannot reduce a CPU tensor; gloo cannot reduce an accelerator one.
    on_cpu = dist.get_backend(group) == "gloo"
    count = torch.tensor([local_count], dtype=torch.int32, device="cpu" if on_cpu else make_current_torch_device())
    dist.all_reduce(count, op=dist.ReduceOp.MAX, group=group)
    return int(count.item())


def wrap_model_provider_with_lora(original_provider, args):
    """Wrap model provider to add LoRA support.

    Only wraps if args.lora_rank > 0. Uses Megatron-Bridge's PEFT integration
    to freeze base model and enable only LoRA parameters.
    """
    if not is_lora_enabled(args):
        return original_provider

    def wrapped_provider(pre_process=True, post_process=True, vp_stage=None, **kwargs):
        sig = inspect.signature(original_provider)
        accepts_vp_stage = "vp_stage" in sig.parameters or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        if accepts_vp_stage:
            model = original_provider(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
        else:
            model = original_provider(pre_process=pre_process, post_process=post_process)

        try:
            peft = build_lora_peft(args)
            # Restrict adapter injection to the requested model region (e.g. language-only
            # for VL models). Overriding target_modules post-construction is supported:
            # PEFT.__call__ rebuilds its matcher state from the current target_modules.
            scope = getattr(args, "lora_scope", "all")
            if scope != "all":
                peft.target_modules = scope_target_modules_to_region(model, list(args.lora_target_modules), scope)
            model = peft(model, training=True)
            ensure_sequence_classification_head_trainable(model, args, "actor", post_process)
            gdn_gate_masked = install_gdn_gate_mask_hooks(model) if is_lora_adapter_mode(args) else 0
            adapter_names = [n for n, _ in model.named_parameters() if is_lora_adapter_param(n)]
            by_role = summarize_lora_modules(adapter_names)
            vision_wrapped = _global_vision_lora_count(by_role.get("vision", 0), args)
            if dist.is_initialized() and dist.get_rank() == 0:
                adapter_params, total_params, percentage = count_adapter_parameters(model)
                mode = "adapter" if is_lora_adapter_mode(args) else ("merge" if is_lora_merge_mode(args) else "plain")
                logger.info(
                    "LoRA enabled: mode=%s scope=%s rank=%d alpha=%d dropout=%s targets=%s | "
                    "wrapped modules by role (rank0-local): %s | adapter_params=%s (%.2f%% of %s total)",
                    mode,
                    scope,
                    args.lora_rank,
                    args.lora_alpha,
                    args.lora_dropout,
                    list(args.lora_target_modules),
                    by_role,
                    f"{adapter_params:,}",
                    percentage,
                    f"{total_params:,}",
                )
                if gdn_gate_masked:
                    logger.info(
                        "LoRA: pinned the GDN in_proj adapter's b/a (gate) rows to zero on %d "
                        "module(s); SGLang hosts the delta on the fused in_proj_qkvz slices only.",
                        gdn_gate_masked,
                    )
                if logger.isEnabledFor(logging.DEBUG):
                    for module_name in sorted({_lora_module_key(n) for n in adapter_names}):
                        logger.debug("LoRA wrapped module: %s", module_name)
            if vision_wrapped:
                raise ValueError(
                    f"LoRA adapter mode wrapped {vision_wrapped} vision-region module(s) (scope={scope}). "
                    "SGLang hosts LoRA on language-model layers only, so this adapter would be trained "
                    "but silently dropped before rollout, breaking the on-policy assumption. Pass "
                    "--lora-scope language, or use --lora-merge-mode (which folds the vision adapter "
                    "into the synced base weights)."
                )
        except RuntimeError:
            # build_lora_peft already raises a clear upgrade-hint message when the
            # Megatron-Bridge image lacks PEFT support; don't shadow it.
            raise

        return model

    return wrapped_provider


def wrap_model_provider_with_freeze(original_provider, args):
    def wrapped_provider(pre_process=True, post_process=True, vp_stage=None, **kwargs):
        if vp_stage is None and mpu.get_virtual_pipeline_model_parallel_world_size() is not None:
            vp_stage = mpu.get_virtual_pipeline_model_parallel_rank()

        sig = inspect.signature(original_provider)
        accepts_vp_stage = "vp_stage" in sig.parameters or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        if accepts_vp_stage:
            model = original_provider(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
        else:
            model = original_provider(pre_process=pre_process, post_process=post_process)

        freeze_model_params(model, args)
        ensure_sequence_classification_head_trainable(model, args, "actor", post_process)

        return model

    return wrapped_provider


def freeze_model_params(model: GPTModel, args: argparse.Namespace):
    if args.only_train_params_name_list:
        for name, param in model.named_parameters():
            param.requires_grad = False
            for pattern in args.only_train_params_name_list:
                if re.search(pattern, name):
                    param.requires_grad = True
                    break

    if args.freeze_params_name_list:
        for name, param in model.named_parameters():
            for pattern in args.freeze_params_name_list:
                if re.search(pattern, name):
                    param.requires_grad = False
                    break


def validate_mtp_only_trainable_params(args: argparse.Namespace, model: Sequence[torch.nn.Module]) -> None:
    """Fail fast when MTP-only mode exposes any non-MTP trainable parameter."""
    if not getattr(args, "mtp_only_training", False):
        return

    trainable = [
        (name, param) for model_chunk in model for name, param in model_chunk.named_parameters() if param.requires_grad
    ]
    unexpected = [name for name, _ in trainable if re.search(r"(^|\.)mtp(\.|$)", name) is None]
    local_param_count = len(trainable)
    global_param_count = local_param_count
    global_unexpected_count = len(unexpected)
    if dist.is_available() and dist.is_initialized():
        first_param = next(param for model_chunk in model for param in model_chunk.parameters())
        counts = torch.tensor(
            [local_param_count, len(unexpected)],
            dtype=torch.long,
            device=first_param.device,
        )
        dist.all_reduce(counts, group=dist.group.WORLD)
        global_param_count, global_unexpected_count = (int(value) for value in counts.tolist())

    if global_unexpected_count:
        details = ", ".join(unexpected[:10]) if unexpected else "reported by another distributed rank"
        raise RuntimeError(f"--mtp-only-training left non-MTP parameters trainable: {details}")

    if global_param_count == 0:
        raise RuntimeError("--mtp-only-training found no trainable MTP parameters in the distributed model.")

    local_numel = sum(param.numel() for _, param in trainable)
    logger.info(
        "MTP-only trainable parameters on this rank: tensors=%d, elements=%d; distributed tensors=%d",
        local_param_count,
        local_numel,
        global_param_count,
    )
