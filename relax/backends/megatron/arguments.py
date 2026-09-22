# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import ast
import math
from typing import Optional

from megatron.core.optimizer import OptimizerConfig
from megatron.training.arguments import parse_args as _megatron_parse_args
from megatron.training.arguments import validate_args as _megatron_validate_args


try:
    from megatron.training.tokenizer.tokenizer import _vocab_size_with_padding as vocab_size_with_padding
except ModuleNotFoundError:
    from megatron.core.tokenizers.utils.build_tokenizer import vocab_size_with_padding

from transformers import AutoConfig

from relax.utils import device as device_utils
from relax.utils.logging_utils import get_logger
from relax.utils.model_source import ModelSource


__all__ = ["validate_args", "megatron_parse_args", "set_default_megatron_args"]

logger = get_logger(__name__)

_FP16_OPTIMIZER_FALLBACKS = {
    "initial_loss_scale": 32768.0,
    "min_loss_scale": 1.0,
    "use_precision_aware_optimizer": True,
    "store_param_remainders": False,
}
_DYNAMIC_LOSS_SCALE_FIELDS = ("initial_loss_scale", "min_loss_scale")


def _optimizer_option(name: str) -> str:
    return f"--{name.replace('_', '-')}"


def _format_optimizer_fallback(name: str, value: object) -> str:
    option = _optimizer_option(name)
    if isinstance(value, bool):
        return option if value else f"--no-{option[2:]}"
    return f"{option} {value!r}"


def _validate_fp16_optimizer_args(args) -> None:
    if getattr(args, "loss_scale", None) is None:
        for name in _DYNAMIC_LOSS_SCALE_FIELDS:
            value = getattr(args, name)
            option = _optimizer_option(name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{option} must be a finite number greater than 0, got {value!r}.")

        if args.min_loss_scale > args.initial_loss_scale:
            raise ValueError(
                "--min-loss-scale must be less than or equal to --initial-loss-scale, "
                f"got {args.min_loss_scale!r} > {args.initial_loss_scale!r}."
            )

    for name, fallback in _FP16_OPTIMIZER_FALLBACKS.items():
        if not isinstance(fallback, bool):
            continue
        value = getattr(args, name)
        if not isinstance(value, bool):
            option = _optimizer_option(name)
            raise ValueError(f"{option} must resolve to a boolean, got {value!r}.")


def _resolve_optimizer_precision_args(args):
    """Resolve precision optimizer arguments before Megatron validates them."""
    native_defaults = None if args.fp16 else OptimizerConfig()
    missing = []
    for name, fp16_fallback in _FP16_OPTIMIZER_FALLBACKS.items():
        if args.fp16 and getattr(args, "loss_scale", None) is not None and name in _DYNAMIC_LOSS_SCALE_FIELDS:
            continue
        if getattr(args, name, None) is None:
            fallback = fp16_fallback if args.fp16 else getattr(native_defaults, name)
            setattr(args, name, fallback)
            if args.fp16:
                missing.append(name)

    if not args.fp16:
        return args

    _validate_fp16_optimizer_args(args)
    if missing:
        fallback_text = ", ".join(_format_optimizer_fallback(name, getattr(args, name)) for name in missing)
        logger.warning(
            "FP16 optimizer options were not explicitly configured; using Relax compatibility fallbacks: %s. "
            "Pass the active FP16 optimizer options explicitly to silence this warning.",
            fallback_text,
        )
    return args


def _validate_dynamic_context_parallel(args):
    import inspect

    from megatron.core import mpu

    assert "dynamic_context_parallel" in inspect.signature(mpu.initialize_model_parallel).parameters, (
        "dynamic_context_parallel is not supported in your megatron version, "
        "please update your megatron version to the corresponding version"
    )
    assert args.use_dynamic_batch_size, "--dynamic-context-parallel requires --use-dynamic-batch-size."
    assert args.rollout_max_context_len, (
        "--dynamic-context-parallel requires --rollout-max-context-len. (context = prompt + response)"
    )
    assert args.max_tokens_per_gpu, "--dynamic-context-parallel requires --max-tokens-per-gpu."

    cp_size = (args.rollout_max_context_len + args.max_tokens_per_gpu - 1) // args.max_tokens_per_gpu
    args.context_parallel_size = 1 << (cp_size - 1).bit_length()
    total_model_size = args.tensor_model_parallel_size * args.pipeline_model_parallel_size * args.context_parallel_size
    assert args.world_size % total_model_size == 0, (
        f"world_size ({args.world_size}) must be a multiple of tp*pp*cp ({total_model_size}), "
        f"dynamic_context_parallel make max context_parallel_size={args.context_parallel_size} from "
        f"(args.rollout_max_context_len + args.max_tokens_per_gpu - 1) // args.max_tokens_per_gpu."
    )
    args.data_parallel_size = args.world_size // total_model_size
    logger.info(f"dynamic_context_parallel init context_parallel_size = {args.context_parallel_size}")
    args.max_seqlen_per_dp_cp_rank = args.max_tokens_per_gpu


def validate_args(args):
    """Run megatron's own validate_args plus slime-specific megatron
    validations."""

    if getattr(args, "dynamic_context_parallel", False):
        _validate_dynamic_context_parallel(args)

    if not device_utils.is_available():
        from unittest.mock import patch

        class _DeviceProperty:
            major = 9
            minor = 0

        # Megatron internally calls torch.cuda.get_device_properties / get_device_capability.
        # When no real device is available, device_utils.get_device_name() returns "cpu",
        # so we must patch torch.cuda specifically — that's what Megatron actually invokes.
        with (
            patch("torch.cuda.get_device_properties", return_value=_DeviceProperty()),
            patch("torch.cuda.get_device_capability", return_value=(9, 0)),
        ):
            _megatron_validate_args(args)
    else:
        _megatron_validate_args(args)

    # always use varlen
    args.variable_seq_lengths = True
    if getattr(args, "moe_token_dispatcher_type", None) == "allgather":
        logger.info(
            "--moe-token-dispatcher-type allgather does not support variable sequence length, "
            "please use alltoall dispatcher instead."
        )
        args.moe_token_dispatcher_type = "alltoall"

    if args.pipeline_model_parallel_size == 1:
        assert args.decoder_first_pipeline_num_layers is None and args.decoder_last_pipeline_num_layers is None, (
            "decoder_first_pipeline_num_layers and decoder_last_pipeline_num_layers should be None when "
            "pipeline_model_parallel_size is 1."
        )

    # Megatron-Bridge requires --calculate-per-token-loss when context parallelism is enabled.
    # See https://github.com/NVIDIA-NeMo/Megatron-Bridge
    if args.context_parallel_size > 1 or getattr(args, "dynamic_context_parallel", False):
        assert args.calculate_per_token_loss, (
            "--calculate-per-token-loss must be set when context_parallel_size > 1 or dynamic_context_parallel is enabled (required by Megatron-Bridge)."
        )
    return args


def _has_dense_moe_layers(args):
    moe_layer_freq = getattr(args, "moe_layer_freq", None)
    if moe_layer_freq is None:
        return True

    if isinstance(moe_layer_freq, str):
        try:
            moe_layer_freq = ast.literal_eval(moe_layer_freq)
        except (SyntaxError, ValueError):
            return "0" in moe_layer_freq

    try:
        return any(int(layer_freq) == 0 for layer_freq in moe_layer_freq)
    except TypeError:
        return int(moe_layer_freq) == 0


def _is_moe_config(hf_config):
    return any(
        hasattr(hf_config, attr)
        for attr in (
            "moe_intermediate_size",
            "num_experts",
            "n_routed_experts",
            "num_local_experts",
        )
    )


def _hf_validate_args(args, hf_config):
    def equal(x, y):
        return x == y

    errors = []

    # Multimodal models (Qwen3-VL, Qwen3.5, Qwen3-Omni, etc.) use multi-axis RoPE whose
    # rotary_pos_emb is a Python list of tensors, not a single Tensor. Megatron's fused
    # RoPE kernel cannot handle this and produces numerically different results from the
    # unfused HF/SGLang implementation, causing training-inference log-prob mismatch.
    is_multimodal = hasattr(hf_config, "text_config") or hasattr(hf_config, "thinker_config")
    if is_multimodal and getattr(args, "apply_rope_fusion", False):
        errors.append(
            "Multimodal models use multi-axis RoPE (list of tensors) which is incompatible "
            "with fused RoPE kernels — this causes training-inference log-prob mismatch. "
            "Add --no-rope-fusion to the launch script."
        )

    # omni models have different config structure
    if hasattr(hf_config, "thinker_config"):
        hf_config = hf_config.thinker_config

    # multimodal models have different config structure
    if hasattr(hf_config, "text_config"):
        hf_config = hf_config.text_config

    validate_dense_ffn = not _is_moe_config(hf_config) or _has_dense_moe_layers(args)

    for hf_config_name, megatron_config_name, compare_fn in (
        [
            ("hidden_size", "hidden_size", equal),
            ("num_attention_heads", "num_attention_heads", equal),
            ("num_hidden_layers", "num_layers", equal),
            ("intermediate_size", "ffn_hidden_size", equal),
            ("moe_intermediate_size", "moe_ffn_hidden_size", equal),
            ("shared_expert_intermediate_size", "moe_shared_expert_intermediate_size", equal),
            ("tie_word_embeddings", "untie_embeddings_and_output_weights", lambda x, y: not x == y),
            ("rope_theta", "rotary_base", equal),
        ]
        + [("rms_norm_eps", "norm_epsilon", equal)]
        if hasattr(args, "norm_epsilon")
        else [("rms_norm_eps", "layernorm_epsilon", equal)]
    ):
        if hf_config_name == "intermediate_size" and not validate_dense_ffn:
            continue

        if hasattr(hf_config, hf_config_name):
            if not compare_fn(getattr(hf_config, hf_config_name), getattr(args, megatron_config_name)):
                errors.append(
                    f"{hf_config_name} in hf config {getattr(hf_config, hf_config_name)} is not equal to "
                    f"{megatron_config_name} {getattr(args, megatron_config_name)}, please check the config."
                )

    if len(errors) > 0:
        raise AssertionError("hf_validate_args failed: " + "; ".join(errors))


def _set_default_megatron_args(args):
    # always use zero optimizer
    args.use_distributed_optimizer = True
    # TODO: maybe change this after megatron has good fp8 support
    args.bf16 = not args.fp16
    _resolve_optimizer_precision_args(args)
    # placeholders
    if args.seq_length is None:
        args.seq_length = 4096
    args.max_position_embeddings = args.seq_length
    # TODO: revisit this when megatron(dev) have solved the optimizer-cpu-offload ckpt saving bug
    args.dist_ckpt_save_pre_mcore_014 = True
    # compatible for megatron
    if hasattr(args, "rope_type") and args.rope_type is None:
        args.rope_type = "yarn" if args.multi_latent_attention else "rope"

    if args.vocab_size and not args.padded_vocab_size:
        args.padded_vocab_size = vocab_size_with_padding(args.vocab_size, args)

    if not args.tokenizer_model and not args.tokenizer_type:
        logger.info("--tokenizer-model not set, use --hf-checkpoint as tokenizer model.")
        args.tokenizer_model = args.hf_checkpoint
        args.tokenizer_type = "HuggingFaceTokenizer"
    return args


# Public alias for external tools (e.g. convert_hf_to_torch_dist.py)
set_default_megatron_args = _set_default_megatron_args


def _derive_cluster_args_from_resource(args):
    """When ``--resource`` is provided, derive the legacy per-role GPU args
    (``actor_num_gpus_per_node``, ``actor_num_nodes``, ``rollout_num_gpus``,
    ``critic_num_*``, ``genrm_num_gpus``) so that users only need to specify
    ``--resource`` without duplicating resource information in separate flags.

    The function is intentionally conservative: it only overwrites a legacy arg
    when the user did **not** explicitly set it (i.e. still at its default).
    """
    if args.resource is None:
        return

    num_gpus_per_node = getattr(args, "num_gpus_per_node", 8)

    # --- actor ---
    if "actor" in args.resource:
        _, actor_total_gpus = args.resource["actor"]
        # Only override when the user relied on the defaults (1 node × 8 gpus)
        derived_gpus_per_node = min(num_gpus_per_node, actor_total_gpus)
        derived_num_nodes = max(1, actor_total_gpus // derived_gpus_per_node)
        if args.actor_num_gpus_per_node == 8 and args.actor_num_nodes == 1:
            # User did not explicitly set these; derive from --resource
            args.actor_num_gpus_per_node = derived_gpus_per_node
            args.actor_num_nodes = derived_num_nodes
            logger.info(
                f"Derived actor_num_gpus_per_node={args.actor_num_gpus_per_node}, "
                f"actor_num_nodes={args.actor_num_nodes} from --resource"
            )

    # --- rollout ---
    if "rollout" in args.resource:
        _, rollout_total_gpus = args.resource["rollout"]
        if args.rollout_num_gpus is None:
            args.rollout_num_gpus = rollout_total_gpus
            logger.info(f"Derived rollout_num_gpus={args.rollout_num_gpus} from --resource")

    # --- critic ---
    if "critic" in args.resource:
        _, critic_total_gpus = args.resource["critic"]
        derived_gpus_per_node = min(num_gpus_per_node, critic_total_gpus)
        derived_num_nodes = max(1, critic_total_gpus // derived_gpus_per_node) if derived_gpus_per_node > 0 else 0
        if args.critic_num_gpus_per_node is None and args.critic_num_nodes is None:
            # User did not explicitly set these; derive from --resource
            args.critic_num_gpus_per_node = derived_gpus_per_node
            args.critic_num_nodes = derived_num_nodes
            logger.info(
                f"Derived critic_num_gpus_per_node={args.critic_num_gpus_per_node}, "
                f"critic_num_nodes={args.critic_num_nodes} from --resource"
            )

    # --- genrm ---
    if "genrm" in args.resource:
        _, genrm_total_gpus = args.resource["genrm"]
        if getattr(args, "genrm_num_gpus", 1) == 1 and genrm_total_gpus != 1:
            args.genrm_num_gpus = genrm_total_gpus
            logger.info(f"Derived genrm_num_gpus={args.genrm_num_gpus} from --resource")


def megatron_parse_args(
    extra_args_provider,
    skip_hf_validate: bool = False,
    model_source: Optional[ModelSource] = None,
):
    """Parse megatron args, validate HF config, and set defaults."""
    args = _megatron_parse_args(extra_args_provider=extra_args_provider, ignore_unknown_args=True)

    if model_source is not None:
        # Keep the CLI path so node-local materialization can remap other
        # arguments that explicitly referenced the same model.
        args._model_source_original_hf_checkpoint = args.hf_checkpoint
        args.hf_checkpoint = model_source.uri

    if args.hf_checkpoint and not skip_hf_validate:
        hf_config = AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
        _hf_validate_args(args, hf_config)

    # Derive legacy cluster args from --resource when available, so users
    # don't have to specify both --resource and --actor-num-nodes / etc.
    _derive_cluster_args_from_resource(args)

    args.rank = 0
    if args.critic_train_only:
        args.world_size = args.critic_num_nodes * args.critic_num_gpus_per_node
    else:
        args.world_size = args.actor_num_nodes * args.actor_num_gpus_per_node
    args = _set_default_megatron_args(args)
    return args
