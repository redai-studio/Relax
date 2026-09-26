# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Export a Relax sequence-classification checkpoint for HF serving.

The training model replaces Megatron's vocabulary output layer with a
replicated ``hidden_size -> num_labels`` head.  Megatron-Bridge still names
that tensor ``lm_head.weight`` during HF export when the original model has an
untied output layer.  Tied-embedding models make Bridge omit ``output_layer``
entirely, so this tool captures the trained task head from the reconstructed
model before Bridge export.  Bridge still merges LoRA and reshards the base
weights; the captured task head is then written as HF ``score.weight``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

import safetensors.torch as safetensors_torch
import torch
from safetensors import safe_open


RELAX_ROOT = str(Path(__file__).resolve().parents[3])
if RELAX_ROOT in sys.path:
    sys.path.remove(RELAX_ROOT)
sys.path.insert(0, RELAX_ROOT)

pythonpath_entries = [entry for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep) if entry != RELAX_ROOT]
os.environ["PYTHONPATH"] = os.pathsep.join([RELAX_ROOT, *pythonpath_entries])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Megatron checkpoint root or iter_XXXXXXX dir")
    parser.add_argument("--origin-hf-dir", type=Path, required=True, help="Original Qwen3.5 Hugging Face directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="SGLang-loadable output directory")
    parser.add_argument(
        "--label-names",
        nargs="+",
        default=None,
        help="Optional labels in class-id order. Defaults to LABEL_0 ... LABEL_N.",
    )
    parser.add_argument("--force", action="store_true", help="Replace an existing output directory")
    parser.add_argument("--no-progress", action="store_true", help="Disable Megatron-Bridge progress output")
    return parser.parse_args()


def resolve_iteration_dir(input_dir: Path) -> Path:
    input_dir = input_dir.resolve()
    if input_dir.name.startswith("iter_"):
        if not input_dir.is_dir():
            raise FileNotFoundError(f"Checkpoint iteration directory does not exist: {input_dir}")
        return input_dir

    latest_file = input_dir / "latest_checkpointed_iteration.txt"
    if not latest_file.is_file():
        raise FileNotFoundError(
            f"Expected {input_dir} to be an iter_XXXXXXX directory or contain latest_checkpointed_iteration.txt"
        )
    latest = latest_file.read_text(encoding="utf-8").strip()
    if not latest.isdigit():
        raise ValueError(f"Unsupported checkpoint iteration marker {latest!r} in {latest_file}")
    iteration_dir = input_dir / f"iter_{int(latest):07d}"
    if not iteration_dir.is_dir():
        raise FileNotFoundError(f"Latest checkpoint directory does not exist: {iteration_dir}")
    return iteration_dir


def load_checkpoint_args(iteration_dir: Path) -> argparse.Namespace:
    from megatron.bridge.training.mlm_compat.arguments import _load_args_from_checkpoint

    checkpoint_args = _load_args_from_checkpoint(str(iteration_dir))
    if getattr(checkpoint_args, "task_type", None) != "seq_cls":
        raise ValueError(
            f"Checkpoint {iteration_dir} is not a sequence-classification checkpoint: "
            f"task_type={getattr(checkpoint_args, 'task_type', None)!r}"
        )
    num_labels = getattr(checkpoint_args, "num_labels", None)
    if not isinstance(num_labels, int) or num_labels < 2:
        raise ValueError(f"Checkpoint has invalid num_labels={num_labels!r}")
    problem_type = getattr(checkpoint_args, "problem_type", None)
    if problem_type not in ("single_label_classification", "multi_label_classification"):
        raise ValueError(f"Checkpoint has invalid problem_type={problem_type!r}")
    return checkpoint_args


def checkpoint_has_mtp(iteration_dir: Path) -> bool:
    from torch.distributed.checkpoint import FileSystemReader

    metadata = FileSystemReader(str(iteration_dir)).read_metadata()
    return any("mtp" in key.lower() for key in metadata.state_dict_metadata)


def _install_classification_and_lora(model: list[torch.nn.Module], checkpoint_args: argparse.Namespace):
    from relax.utils.megatron_peft_utils import build_lora_peft, is_lora_enabled
    from relax.utils.training.ppo_utils import (
        ensure_sequence_classification_head_trainable,
        install_sequence_classification_head_in_provider,
    )

    peft = build_lora_peft(checkpoint_args) if is_lora_enabled(checkpoint_args) else None
    wrapped_models = []
    for model_chunk in model:
        install_sequence_classification_head_in_provider(
            model_chunk,
            checkpoint_args,
            role="actor",
            post_process=True,
        )
        if peft is not None:
            model_chunk = peft(model_chunk, training=True)
            ensure_sequence_classification_head_trainable(
                model_chunk,
                checkpoint_args,
                role="actor",
                post_process=True,
            )
        wrapped_models.append(model_chunk)
    return wrapped_models


@contextmanager
def _bridge_export_overrides(provider):
    import megatron.bridge.training.model_load_save as model_load_save

    original_load_model_config = model_load_save.load_model_config
    original_save_file = safetensors_torch.save_file

    def patched_load_model_config(checkpoint_path):
        _model_config, mlm_args = original_load_model_config(checkpoint_path)
        return provider, mlm_args

    def save_file_contiguous(tensors, filename, metadata=None):
        contiguous = {
            name: tensor.contiguous() if hasattr(tensor, "is_contiguous") and not tensor.is_contiguous() else tensor
            for name, tensor in tensors.items()
        }
        return original_save_file(contiguous, filename, metadata=metadata)

    model_load_save.load_model_config = patched_load_model_config
    safetensors_torch.save_file = save_file_contiguous
    try:
        yield
    finally:
        model_load_save.load_model_config = original_load_model_config
        safetensors_torch.save_file = original_save_file


def export_with_bridge(
    *,
    iteration_dir: Path,
    origin_hf_dir: Path,
    output_dir: Path,
    checkpoint_args: argparse.Namespace,
    show_progress: bool,
) -> torch.Tensor:
    from megatron.bridge import AutoBridge
    from megatron.bridge.models.conversion.param_mapping import AutoMapping
    from megatron.bridge.training.model_load_save import temporary_distributed_context

    bridge = AutoBridge.from_hf_pretrained(str(origin_hf_dir), trust_remote_code=True)
    provider = bridge.to_megatron_provider(load_weights=False)

    if getattr(provider, "mtp_num_layers", 0) and not checkpoint_has_mtp(iteration_dir):
        print(f"[seq-cls-export] disabling MTP from origin config (mtp_num_layers={provider.mtp_num_layers})")
        provider.mtp_num_layers = 0

    provider.register_pre_wrap_hook(lambda model: _install_classification_and_lora(model, checkpoint_args))
    AutoMapping.register_module_type("LinearForLastLayer", "replicated")

    with _bridge_export_overrides(provider), temporary_distributed_context(backend="gloo"):
        megatron_model = bridge.load_megatron_model(str(iteration_dir), wrap_with_ddp=False)
        task_head = _task_head_from_model(megatron_model, checkpoint_args.num_labels)
        bridge.save_hf_pretrained(
            megatron_model,
            str(output_dir),
            show_progress=show_progress,
            strict=False,
            source_path=str(origin_hf_dir),
        )
    return task_head


def _task_head_from_model(model: list[torch.nn.Module], num_labels: int) -> torch.Tensor:
    from relax.utils.training.ppo_utils import LinearForLastLayer, _find_output_layer_owner

    task_heads = []
    for model_chunk in model:
        owner = _find_output_layer_owner(model_chunk)
        if owner is None:
            continue
        head = owner.output_layer
        if not isinstance(head, LinearForLastLayer):
            raise TypeError(f"Expected LinearForLastLayer task head, got {type(head).__name__}")
        task_heads.append(head)

    if len(task_heads) != 1:
        raise ValueError(f"Expected exactly one sequence-classification task head, found {len(task_heads)}")
    weight = task_heads[0].weight
    if weight.ndim != 2 or weight.shape[0] != num_labels:
        raise ValueError(
            f"Sequence-classification task head has shape {tuple(weight.shape)}, expected [{num_labels}, hidden_size]"
        )
    return weight.detach().to(device="cpu").contiguous().clone()


def reconcile_export_index(output_dir: Path) -> dict[str, list[str]]:
    """Drop Bridge ghost entries for MTP tensors absent from classification
    checkpoints."""
    from relax.utils.hf_export import reconcile_hf_export_index

    summary = reconcile_hf_export_index(
        str(output_dir),
        reference_hf_dir=None,
        supplement_mtp=False,
    )
    if summary["ghosts"]:
        print(
            f"[seq-cls-export] reconciled {len(summary['ghosts'])} ghost index entries; "
            f"dropped {len(summary['dropped'])} absent MTP tensors"
        )
    return summary


def _load_index(output_dir: Path) -> tuple[Path | None, dict[str, str]]:
    index_paths = sorted(output_dir.glob("*.safetensors.index.json"))
    if len(index_paths) > 1:
        raise ValueError(f"Found multiple safetensors index files in {output_dir}: {index_paths}")
    if index_paths:
        index_path = index_paths[0]
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError(f"Invalid weight_map in {index_path}")
        return index_path, weight_map

    safetensor_paths = sorted(output_dir.glob("*.safetensors"))
    if len(safetensor_paths) != 1:
        raise ValueError(
            f"Expected one unindexed safetensors file or one index in {output_dir}, found {safetensor_paths}"
        )
    with safe_open(safetensor_paths[0], framework="pt", device="cpu") as handle:
        return None, {name: safetensor_paths[0].name for name in handle.keys()}


def _task_head_key(weight_map: dict[str, str]) -> str | None:
    exact = [key for key in weight_map if key == "lm_head.weight"]
    if exact:
        return exact[0]
    suffix = [key for key in weight_map if key.endswith(".lm_head.weight")]
    if len(suffix) > 1:
        raise ValueError(f"Expected exactly one exported lm_head.weight, found {suffix}")
    return suffix[0] if suffix else None


def rewrite_task_head(
    output_dir: Path,
    num_labels: int,
    task_head: torch.Tensor | None = None,
) -> tuple[int, int]:
    index_path, weight_map = _load_index(output_dir)
    source_key = _task_head_key(weight_map)
    if source_key is None and task_head is None:
        raise ValueError("Export has no lm_head.weight and no captured task head was provided")
    if "score.weight" in weight_map:
        raise ValueError("Export already contains score.weight; refusing an ambiguous rewrite")

    score = task_head
    source_size = 0
    shard_path = None
    metadata = None
    tensors = None
    if source_key is not None:
        shard_path = output_dir / weight_map[source_key]
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
        tensors = safetensors_torch.load_file(str(shard_path), device="cpu")
        if source_key not in tensors:
            raise KeyError(f"Index maps {source_key!r} to {shard_path}, but the tensor is absent")
        exported_head = tensors.pop(source_key)
        source_size = exported_head.numel() * exported_head.element_size()
        if score is None:
            score = exported_head

    assert score is not None
    if score.ndim != 2 or score.shape[0] != num_labels:
        raise ValueError(
            f"Exported classification head has shape {tuple(score.shape)}, expected [{num_labels}, hidden_size]"
        )

    if shard_path is None and index_path is not None:
        shard_path = output_dir / "sequence-classification-head.safetensors"
        if shard_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing task-head shard: {shard_path}")
        safetensors_torch.save_file({"score.weight": score}, str(shard_path), metadata={"format": "pt"})
    else:
        if shard_path is None:
            shard_path = output_dir / next(iter(weight_map.values()))
            with safe_open(shard_path, framework="pt", device="cpu") as handle:
                metadata = handle.metadata()
            tensors = safetensors_torch.load_file(str(shard_path), device="cpu")
        assert tensors is not None
        tensors["score.weight"] = score
        safetensors_torch.save_file(
            {name: tensor.contiguous() for name, tensor in tensors.items()},
            str(shard_path),
            metadata=metadata or {"format": "pt"},
        )

    if index_path is not None:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        if source_key is not None:
            payload["weight_map"].pop(source_key)
        payload["weight_map"]["score.weight"] = shard_path.name
        total_size = payload.get("metadata", {}).get("total_size")
        if isinstance(total_size, int):
            score_size = score.numel() * score.element_size()
            payload["metadata"]["total_size"] = total_size - source_size + score_size
        index_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return tuple(score.shape)


def _is_moe_config(config: dict[str, Any]) -> bool:
    text_config = config.get("text_config") if isinstance(config.get("text_config"), dict) else config
    values: Iterable[Any] = (
        config.get("model_type"),
        text_config.get("model_type"),
        *(config.get("architectures") or []),
    )
    return any("moe" in str(value).lower() for value in values)


def _classification_architecture(config: dict[str, Any]) -> str:
    model_type = str(config.get("model_type", "")).lower()
    architectures = [str(value).lower() for value in config.get("architectures") or []]
    if model_type == "qwen3_vl" or any("qwen3vl" in value for value in architectures):
        return "Qwen3VLForSequenceClassification"
    if model_type == "qwen2_5_vl" or any("qwen2_5_vl" in value for value in architectures):
        return "Qwen2_5_VLForSequenceClassification"
    if model_type == "qwen3" or any(value.startswith("qwen3for") for value in architectures):
        return "Qwen3ForSequenceClassification"
    return "Qwen3_5MoeForSequenceClassification" if _is_moe_config(config) else "Qwen3_5ForSequenceClassification"


def update_classification_config(
    output_dir: Path,
    *,
    num_labels: int,
    problem_type: str,
    label_names: list[str] | None,
) -> str:
    config_path = output_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Bridge export did not create {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))

    if label_names is None:
        label_names = [f"LABEL_{index}" for index in range(num_labels)]
    if len(label_names) != num_labels:
        raise ValueError(f"--label-names has {len(label_names)} entries, expected {num_labels}")
    if len(set(label_names)) != len(label_names):
        raise ValueError("--label-names entries must be unique")

    id2label = {str(index): label for index, label in enumerate(label_names)}
    label2id = {label: index for index, label in enumerate(label_names)}
    architecture = _classification_architecture(config)
    config.update(
        {
            "architectures": [architecture],
            "num_labels": num_labels,
            "problem_type": problem_type,
            "id2label": id2label,
            "label2id": label2id,
            "pooling_type": "LAST",
            "normalize": False,
        }
    )
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        text_config.update(
            {
                "num_labels": num_labels,
                "problem_type": problem_type,
                "id2label": id2label,
                "label2id": label2id,
            }
        )
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return architecture


def validate_export(output_dir: Path, *, num_labels: int) -> None:
    _index_path, weight_map = _load_index(output_dir)
    physical_keys: set[str] = set()
    for shard_name in set(weight_map.values()):
        with safe_open(output_dir / shard_name, framework="pt", device="cpu") as handle:
            physical_keys.update(handle.keys())
    ghost_keys = sorted(set(weight_map) - physical_keys)
    if ghost_keys:
        raise ValueError(f"Export index references tensors absent from its shards: {ghost_keys[:20]}")
    if "score.weight" not in weight_map:
        raise ValueError("Export is missing score.weight")
    forbidden = [
        key
        for key in weight_map
        if key.endswith("lm_head.weight")
        or ".lora_A." in key
        or ".lora_B." in key
        or ".adapter." in key
        or ".to_wrap." in key
    ]
    if forbidden:
        raise ValueError(f"Export contains unmerged or generative-head tensors: {forbidden[:20]}")

    score_path = output_dir / weight_map["score.weight"]
    with safe_open(score_path, framework="pt", device="cpu") as handle:
        score_shape = tuple(handle.get_slice("score.weight").get_shape())
    if len(score_shape) != 2 or score_shape[0] != num_labels:
        raise ValueError(f"score.weight has invalid shape {score_shape}")


def prepare_output_dir(output_dir: Path, force: bool) -> None:
    if output_dir.exists():
        if not force:
            raise FileExistsError(f"Output directory already exists: {output_dir}; pass --force to replace it")
        shutil.rmtree(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)


def validate_path_separation(output_dir: Path, iteration_dir: Path, origin_hf_dir: Path) -> None:
    output_dir = output_dir.resolve()
    protected_roots = {Path(output_dir.anchor), Path.home().resolve(), Path(RELAX_ROOT).resolve()}
    if output_dir in protected_roots:
        raise ValueError(f"Refusing to use protected directory as --output-dir: {output_dir}")

    for source_name, source_dir in (("checkpoint", iteration_dir.resolve()), ("origin HF", origin_hf_dir.resolve())):
        try:
            output_dir.relative_to(source_dir)
            overlaps = True
        except ValueError:
            try:
                source_dir.relative_to(output_dir)
                overlaps = True
            except ValueError:
                overlaps = False
        if overlaps:
            raise ValueError(f"--output-dir {output_dir} must not overlap the {source_name} directory {source_dir}")


def main() -> None:
    args = parse_args()
    iteration_dir = resolve_iteration_dir(args.input_dir)
    checkpoint_args = load_checkpoint_args(iteration_dir)
    origin_hf_dir = args.origin_hf_dir.resolve()
    output_dir = args.output_dir.resolve()
    validate_path_separation(output_dir, iteration_dir, origin_hf_dir)
    prepare_output_dir(output_dir, args.force)

    print(f"[seq-cls-export] checkpoint: {iteration_dir}")
    print(f"[seq-cls-export] origin HF:  {origin_hf_dir}")
    print(f"[seq-cls-export] output:     {output_dir}")
    try:
        task_head = export_with_bridge(
            iteration_dir=iteration_dir,
            origin_hf_dir=origin_hf_dir,
            output_dir=output_dir,
            checkpoint_args=checkpoint_args,
            show_progress=not args.no_progress,
        )
        reconcile_export_index(output_dir)
        head_shape = rewrite_task_head(output_dir, checkpoint_args.num_labels, task_head=task_head)
        architecture = update_classification_config(
            output_dir,
            num_labels=checkpoint_args.num_labels,
            problem_type=checkpoint_args.problem_type,
            label_names=args.label_names,
        )
        validate_export(output_dir, num_labels=checkpoint_args.num_labels)
    except Exception:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        raise

    print(
        f"[seq-cls-export] done: architecture={architecture}, problem_type={checkpoint_args.problem_type}, "
        f"score.weight={head_shape}"
    )


if __name__ == "__main__":
    main()
