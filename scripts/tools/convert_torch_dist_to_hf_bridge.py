# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import argparse
import os
import sys
from pathlib import Path

import torch


_RELAX_ROOT = str(Path(__file__).resolve().parents[2])
if _RELAX_ROOT in sys.path:
    sys.path.remove(_RELAX_ROOT)
sys.path.insert(0, _RELAX_ROOT)

_pythonpath_entries = [
    entry for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep) if entry and entry != _RELAX_ROOT
]
os.environ["PYTHONPATH"] = os.pathsep.join([_RELAX_ROOT, *_pythonpath_entries])


_model_load_save_module = None
_original_load_model_config = None
_original_save_file = None
_provider_override = {}
_lora_checkpoint_spec = None


def _resolve_checkpoint_dir(input_dir):
    checkpoint_dir = Path(input_dir)
    latest_file = checkpoint_dir / "latest_checkpointed_iteration.txt"
    if latest_file.is_file():
        tag = latest_file.read_text().strip()
        iteration_dir = checkpoint_dir / f"iter_{int(tag):07d}"
        if iteration_dir.is_dir():
            checkpoint_dir = iteration_dir
    return checkpoint_dir


def _read_checkpoint_metadata(input_dir):
    from torch.distributed.checkpoint import FileSystemReader

    return FileSystemReader(_resolve_checkpoint_dir(input_dir)).read_metadata()


def _build_lora_checkpoint_spec(metadata):
    """Describe the LoRA structure stored in a torch-dist checkpoint.

    Full module paths are recovered from the checkpoint instead of recomputing
    scope/freeze rules from launcher-only environment variables. This makes the
    converter recreate exactly the adapters that were actually saved, including
    merger-only vision LoRA and per-expert grouped MoE adapters.
    """
    adapter_shapes = {}
    for key, value in metadata.state_dict_metadata.items():
        if ".adapter." not in key or not key.endswith(".weight"):
            continue
        size = getattr(value, "size", None)
        if size is not None:
            adapter_shapes[key] = tuple(size)

    if not adapter_shapes:
        return None

    suffixes = (".adapter.linear_in.weight", ".adapter.linear_out.weight")
    pairs = {}
    unrecognized = []
    for key in adapter_shapes:
        for side, suffix in zip(("in", "out"), suffixes):
            if key.endswith(suffix):
                pairs.setdefault(key[: -len(suffix)], {})[side] = key
                break
        else:
            unrecognized.append(key)

    if unrecognized:
        raise ValueError(
            f"Checkpoint contains {len(unrecognized)} unsupported adapter weight key(s), "
            f"e.g. {sorted(unrecognized)[:3]}"
        )
    incomplete = sorted(prefix for prefix, pair in pairs.items() if set(pair) != {"in", "out"})
    if incomplete:
        raise ValueError(f"Checkpoint contains {len(incomplete)} incomplete LoRA adapter pair(s): {incomplete[:3]}")

    ranks = set()
    for pair in pairs.values():
        in_shape = adapter_shapes[pair["in"]]
        out_shape = adapter_shapes[pair["out"]]
        if len(in_shape) not in (2, 3) or len(out_shape) != len(in_shape):
            raise ValueError(f"Unsupported LoRA adapter shapes: linear_in={in_shape}, linear_out={out_shape}")
        ranks.add(in_shape[-2])
        if out_shape[-1] != in_shape[-2]:
            raise ValueError(f"LoRA rank mismatch: linear_in={in_shape}, linear_out={out_shape}")

    if len(ranks) != 1:
        raise ValueError(
            f"Checkpoint contains multiple LoRA ranks {sorted(ranks)}; automatic reconstruction is unsafe"
        )

    model_targets = {_checkpoint_adapter_prefix_to_model_target(prefix) for prefix in pairs}
    if len(model_targets) != len(pairs):
        raise ValueError("Multiple checkpoint LoRA paths collapse to the same model module path")

    return {
        "rank": ranks.pop(),
        "target_modules": sorted(model_targets),
        "adapter_keys": set(adapter_shapes),
        "share_expert_adapters": not any(len(shape) == 3 for shape in adapter_shapes.values()),
    }


def _checkpoint_adapter_prefix_to_model_target(prefix):
    target = prefix.replace(".mlp.experts.experts.", ".mlp.experts.")
    vision_layers_prefix = "vision_model.decoder.layers."
    if target.startswith(vision_layers_prefix):
        suffix = target[len(vision_layers_prefix) :]
        first_component = suffix.split(".", 1)[0]
        if not first_component.isdigit():
            # Qwen3-VL's vision sharded_state_dict collapses the physical layer
            # index into the ShardedTensor offsets. Recreate every concrete
            # vision layer with a wildcard; adapter coverage below still checks
            # the persistent DCP keys exactly before any weights are loaded.
            target = f"{vision_layers_prefix}*.{suffix}"
    return target


def _assert_adapter_coverage(expected, actual):
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise ValueError(
            "LoRA checkpoint/model structure mismatch before torch-dist load: "
            f"missing={len(missing)} {missing[:3]}, unexpected={len(unexpected)} {unexpected[:3]}"
        )


def _adapter_checkpoint_keys_from_model(model_chunks):
    actual = set()
    for chunk in model_chunks:
        for dict_key, sharded_value in chunk.sharded_state_dict().items():
            checkpoint_key = getattr(sharded_value, "key", dict_key)
            if ".adapter." in checkpoint_key and checkpoint_key.endswith(".weight"):
                actual.add(checkpoint_key)
    return actual


def _register_lora_hook(provider, checkpoint_args, spec):
    if checkpoint_args is None:
        raise ValueError(
            "LoRA checkpoint detected, but its training args are unavailable; refusing a base-only export"
        )

    rank = getattr(checkpoint_args, "lora_rank", None)
    alpha = getattr(checkpoint_args, "lora_alpha", None)
    if rank != spec["rank"]:
        raise ValueError(f"LoRA rank mismatch: checkpoint args={rank}, adapter tensors={spec['rank']}")
    if alpha is None:
        raise ValueError("LoRA checkpoint args do not contain lora_alpha; automatic merge is unsafe")

    peft_config = {
        "rank": rank,
        "alpha": alpha,
        "dropout": getattr(checkpoint_args, "lora_dropout", 0.0),
        "bias": "none",
        "target_modules": spec["target_modules"],
        "share_expert_adapters": spec["share_expert_adapters"],
        "normalize_moe_lora": getattr(checkpoint_args, "normalize_moe_lora", False),
    }

    def apply_checkpoint_lora(model_chunks):
        try:
            from megatron.bridge.peft.utils import create_peft
        except ImportError:
            # Older Bridge revisions expose LoRA directly but not the config
            # factory. Keep this fallback strict: unsupported constructor fields
            # (notably per-expert adapters) must raise instead of being dropped.
            from megatron.bridge.peft.lora import LoRA

            peft = LoRA(
                target_modules=peft_config["target_modules"],
                dim=peft_config["rank"],
                alpha=peft_config["alpha"],
                dropout=peft_config["dropout"],
                share_expert_adapters=peft_config["share_expert_adapters"],
                normalize_moe_lora=peft_config["normalize_moe_lora"],
            )
        else:
            peft = create_peft(peft_config)
        model_chunks = peft(model_chunks, training=False)
        actual = _adapter_checkpoint_keys_from_model(model_chunks)
        _assert_adapter_coverage(spec["adapter_keys"], actual)
        print(
            f"[convert] Reconstructed {len(actual) // 2} LoRA module(s): rank={rank}, alpha={alpha}, "
            f"share_expert_adapters={spec['share_expert_adapters']}"
        )
        return model_chunks

    provider.register_pre_wrap_hook(apply_checkpoint_lora)


# Some megatron_to_hf mappings in Megatron Bridge yield non-contiguous tensors (e.g. after
# transpose/narrow/chunk without a trailing .contiguous()). The shared-tensor dedup check in
# safetensors.save_file runs `tensor.view(-1)[-1]`, which raises
# "view size is not compatible with input tensor's size and stride" on such tensors.
# Force .contiguous() at the save boundary as a safety net.


def _save_file_ensure_contiguous(tensors, filename, metadata=None):
    fixed = {}
    for k, v in tensors.items():
        if hasattr(v, "is_contiguous") and not v.is_contiguous():
            print(
                f"[convert] forcing .contiguous() on non-contig tensor: {k} shape={tuple(v.shape)} stride={v.stride()}"
            )
            v = v.contiguous()
        fixed[k] = v
    return _original_save_file(fixed, filename, metadata=metadata)


# Here we need to patch Megatron Bridge's `load_model_config`, since the checkpoint is saved
# by Megatron and lack of provider information.
def _patched_load_model_config(checkpoint_path):
    provider = _provider_override.get("provider")
    if provider is not None:
        # The HF Bridge provider replaces the MLM TransformerConfig. Load only the
        # checkpoint Namespace here: converting old Relax args into the current
        # TransformerConfig can fail on newly-added fields even though that config
        # would immediately be discarded below.
        from megatron.bridge.training.mlm_compat.arguments import _load_args_from_checkpoint

        mlm_args = _load_args_from_checkpoint(checkpoint_path)
        checkpoint_moe_layer_freq = getattr(mlm_args, "moe_layer_freq", None)
        if isinstance(checkpoint_moe_layer_freq, list):
            provider.moe_layer_freq = checkpoint_moe_layer_freq
            print(
                "[convert] Preserving checkpoint moe_layer_freq list "
                f"({len(checkpoint_moe_layer_freq)} layers) for sharded key compatibility"
            )
        if _lora_checkpoint_spec is not None and not getattr(provider, "_relax_lora_export_hook", False):
            _register_lora_hook(provider, mlm_args, _lora_checkpoint_spec)
            provider._relax_lora_export_hook = True
        print(f"[convert] Overriding MLM TransformerConfig with Bridge provider: {type(provider).__name__}")
        return provider, mlm_args
    return _original_load_model_config(checkpoint_path)


def _initialize_bridge_patches():
    global _model_load_save_module, _original_load_model_config, _original_save_file

    import megatron.bridge.training.model_load_save as model_load_save_module
    import safetensors.torch as safetensors_torch

    _model_load_save_module = model_load_save_module
    _original_load_model_config = model_load_save_module.load_model_config
    _original_save_file = safetensors_torch.save_file
    model_load_save_module.load_model_config = _patched_load_model_config
    safetensors_torch.save_file = _save_file_ensure_contiguous


def _checkpoint_has_mtp(input_dir):
    """Return True if the torch-dist checkpoint actually stores MTP weights.

    `input_dir` may be a Megatron checkpoint root (containing
    `latest_checkpointed_iteration.txt` and `iter_*` subdirs) or a single
    checkpoint directory. Detection reads the torch DCP `.metadata` and looks
    for any `mtp` key.
    """
    metadata = _read_checkpoint_metadata(input_dir)
    return any("mtp" in k.lower() for k in metadata.state_dict_metadata)


def _export_checkpoint(bridge, input_dir, output_dir, strict):
    """Load torch-dist weights and explicitly merge any reconstructed LoRA on
    HF export."""
    from megatron.bridge.training.model_load_save import temporary_distributed_context

    with temporary_distributed_context(backend="gloo"):
        megatron_model = bridge.load_megatron_model(input_dir, wrap_with_ddp=False)
        bridge.save_hf_pretrained(
            megatron_model,
            output_dir,
            strict=strict,
            merge_adapter_weights=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert torch distributed checkpoint to HuggingFace format using Megatron Bridge"
    )
    parser.add_argument(
        "--input-dir", type=str, required=True, help="Path to the torch distributed checkpoint directory"
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Path to save the HuggingFace checkpoint")
    parser.add_argument(
        "--origin-hf-dir",
        type=str,
        required=True,
        help="Path to the original HuggingFace model directory (for config)",
    )
    parser.add_argument(
        "-f", "--force", action="store_true", help="Force overwrite the output directory if it exists."
    )
    parser.add_argument(
        "--fp8",
        action="store_true",
        help="Quantize each exported HF tensor to FP8 before it is buffered for safetensors output.",
    )
    parser.add_argument(
        "--fp8-strategy",
        choices=["block", "channel", "tensor"],
        default="block",
        help="FP8 quantization strategy (default: block).",
    )
    parser.add_argument(
        "--fp8-block-size",
        type=int,
        nargs=2,
        default=None,
        metavar=("ROWS", "COLS"),
        help="Block shape for block FP8 (default: 128 128).",
    )
    parser.add_argument(
        "--fp8-device",
        type=str,
        default="cuda",
        help="Device used for one-tensor-at-a-time FP8 quantization (default: cuda).",
    )
    parser.add_argument(
        "--fp8-max-shard-size-mb",
        type=int,
        default=4096,
        help="Target FP8 safetensors shard size in MiB; one converted tensor group may exceed it (default: 4096).",
    )
    args = parser.parse_args()

    _initialize_bridge_patches()
    from megatron.bridge import AutoBridge

    checkpoint_metadata = _read_checkpoint_metadata(args.input_dir)
    _lora_checkpoint_spec = _build_lora_checkpoint_spec(checkpoint_metadata)
    if _lora_checkpoint_spec is None:
        print("[convert] No LoRA adapter weights detected; exporting the full checkpoint directly")
    else:
        print(
            f"[convert] Detected {len(_lora_checkpoint_spec['adapter_keys']) // 2} LoRA module(s); "
            "they will be reconstructed and merged into the exported HF weights"
        )

    if args.fp8:
        try:
            output_is_origin = os.path.samefile(args.output_dir, args.origin_hf_dir)
        except FileNotFoundError:
            output_is_origin = os.path.realpath(args.output_dir) == os.path.realpath(args.origin_hf_dir)
        if output_is_origin:
            raise ValueError("--output-dir must differ from --origin-hf-dir for online FP8 conversion")
    if os.path.exists(args.output_dir) and not args.force:
        raise ValueError(f"Output directory {args.output_dir} already exists. Use --force to overwrite it.")

    fp8_block_size = args.fp8_block_size
    if args.fp8 and args.fp8_strategy == "block" and fp8_block_size is None:
        fp8_block_size = [128, 128]
    if args.fp8:
        from relax.utils.quant_cast.fp8 import build_quantization_config, validate_fp8_options
        from relax.utils.quant_cast.fp8_checkpoint import StreamingFP8Writer

        validate_fp8_options(args.fp8_strategy, fp8_block_size)
        if args.fp8_max_shard_size_mb <= 0:
            raise ValueError("--fp8-max-shard-size-mb must be positive")
        fp8_device = torch.device(args.fp8_device)
        if fp8_device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("--fp8-device points to CUDA, but torch.cuda.is_available() is false")

    print(f"Loading config from {args.origin_hf_dir}")
    bridge = AutoBridge.from_hf_pretrained(args.origin_hf_dir, trust_remote_code=True)

    # Use Bridge's provider so the correct model class is created (e.g., Qwen3VLModel
    # instead of GPTModel). This is needed because MLM checkpoints lack run_config.yaml.
    provider = bridge.to_megatron_provider(load_weights=False)

    # Some HF configurations enable MTP layers, but RL-trained Megatron checkpoints lack MTP weights, which causes a loading error.
    allow_missing_mtp_keys = False
    if getattr(provider, "mtp_num_layers", 0) and not _checkpoint_has_mtp(args.input_dir):
        print(f"[convert] Checkpoint has no MTP weights; disabling MTP (was mtp_num_layers={provider.mtp_num_layers})")
        provider.mtp_num_layers = 0
        allow_missing_mtp_keys = True

    _provider_override["provider"] = provider
    print(f"[convert] Using Bridge provider: {type(provider).__name__}")

    fp8_writer = None
    source = None
    original_save_generator = None
    if args.fp8:
        state = getattr(bridge.hf_pretrained, "state", None)
        source = getattr(state, "source", None)
        if source is None or not hasattr(source, "key_to_filename_map"):
            raise ValueError("Online FP8 conversion requires --origin-hf-dir to contain safetensors weights")
        fp8_writer = StreamingFP8Writer(
            source.key_to_filename_map,
            args.fp8_strategy,
            fp8_block_size,
            args.fp8_device,
            args.fp8_max_shard_size_mb * 1024**2,
        )
        original_save_generator = source.save_generator
        source.save_generator = fp8_writer.save_generator
        print(
            f"[convert] Enabled streaming FP8: strategy={args.fp8_strategy}, "
            f"block_size={fp8_block_size}, device={args.fp8_device}, "
            f"max_shard_size_mb={args.fp8_max_shard_size_mb}"
        )

    print(f"Exporting checkpoint from {args.input_dir} to {args.output_dir}")
    try:
        _export_checkpoint(
            bridge,
            args.input_dir,
            args.output_dir,
            strict=not allow_missing_mtp_keys,
        )
    finally:
        if source is not None and original_save_generator is not None:
            source.save_generator = original_save_generator

    # Work around a Megatron-Bridge bug: when MTP is explicitly absent and strict=False,
    # export_ckpt writes each incomplete shard without the missing MTP tensors, yet still
    # lists those keys in model.safetensors.index.json, producing "ghost" entries that
    # point at shards which do not contain them. Reconcile the index against the shards
    # and supplement missing MTP weights from the reference model so the export stays
    # loadable/deployable (e.g. for EAGLE speculative decoding). The FP8 path uses a
    # separate streaming writer with its own index and is left untouched here.
    if not args.fp8:
        from relax.utils.hf_export import reconcile_hf_export_index

        reconcile_summary = reconcile_hf_export_index(
            args.output_dir,
            reference_hf_dir=args.origin_hf_dir,
            supplement_mtp=allow_missing_mtp_keys,
        )
        if reconcile_summary["ghosts"]:
            print(
                f"[convert] reconciled index: {len(reconcile_summary['ghosts'])} ghost key(s), "
                f"{len(reconcile_summary['supplemented'])} MTP supplemented, "
                f"{len(reconcile_summary['dropped'])} dropped"
            )

    # Make the output dir consumable by older transformers 4.x releases.
    import json
    import shutil

    cfg_path = os.path.join(args.output_dir, "config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        rope_params = cfg.get("rope_parameters")
        if isinstance(rope_params, dict):
            cfg.setdefault("rope_theta", rope_params.get("rope_theta"))
            cfg.setdefault("rope_scaling", None)
        if "dtype" in cfg and "torch_dtype" not in cfg:
            cfg["torch_dtype"] = cfg["dtype"]
        cfg["transformers_version"] = "4.51.0"
        if fp8_writer is not None:
            cfg["quantization_config"] = build_quantization_config(
                args.fp8_strategy,
                fp8_block_size,
                fp8_writer.result.modules_to_not_convert,
            )
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        print(f"[convert] post-processed {cfg_path} for transformers 4.x compatibility")

    for fname in ("tokenizer_config.json", "vocab.json", "merges.txt"):
        src = os.path.join(args.origin_hf_dir, fname)
        dst = os.path.join(args.output_dir, fname)
        if os.path.isfile(src):
            shutil.copyfile(src, dst)
            print(f"[convert] copied {fname} from origin (tokenizer-compatibility)")

    if fp8_writer is not None:
        print(
            f"[convert] wrote {len(fp8_writer.result.weight_map)} FP8 checkpoint tensors "
            f"({fp8_writer.result.total_size} bytes)"
        )

    print("Done!")
