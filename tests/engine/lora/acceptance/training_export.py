# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Train one real LoRA step, save through Relax DCP, export and seal TRAINED.

This subprocess needs one visible CUDA device and a local dense HF model. It
exercises the existing FSDP checkpoint/export functions with a plain module,
which those functions support. It does not claim distributed FSDP, Megatron,
Actor scheduling, or a complete RL training job was exercised.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from relax.engine.lora.artifact import PROVENANCE, ModelContract, canonical, provenance
from relax.engine.lora.snapshot import fingerprint_model, snapshot_adapter


def export_training_step(model_path: Path, store: Path, output: Path) -> dict:
    import torch
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from relax.backends.fsdp.checkpoint import export_peft_adapter, save_checkpoint
    from relax.backends.fsdp.lora import (
        inject_lora_adapter,
        lora_metadata_dict,
        named_lora_params,
        peft_adapter_state_dict,
    )

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("training export requires exactly one visible CUDA GPU; set CUDA_VISIBLE_DEVICES")
    model_path, store, output = model_path.resolve(), store.resolve(), output.resolve()
    config = json.loads((model_path / "config.json").read_text())
    contract = ModelContract(fingerprint_model(model_path), config, 8, ("q_proj", "v_proj"))
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(1707)
    torch.cuda.manual_seed_all(1707)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        local_files_only=True,
        trust_remote_code=False,
    ).to("cuda:0")
    injected = inject_lora_adapter(
        model, rank=8, alpha=16, target_modules=contract.target_modules, task_type="CAUSAL_LM"
    )
    adapter_params = named_lora_params(model)
    trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if not adapter_params or trainable_names != {name for name, _ in adapter_params}:
        raise AssertionError("only the injected LoRA parameters may be trainable")
    initial = {name: parameter.detach().cpu().clone() for name, parameter in adapter_params}
    optimizer = torch.optim.AdamW([parameter for _, parameter in adapter_params], lr=1e-3)
    text = "A versioned adapter keeps an agent session on the same policy throughout its conversation."
    batch = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to("cuda:0")
    if batch["input_ids"].shape[1] < 2:
        raise AssertionError("training example needs at least two tokens for a causal loss")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = model(**batch, labels=batch["input_ids"], use_cache=False).loss
    if not torch.isfinite(loss).item():
        raise AssertionError("non-finite training loss")
    loss.backward()
    optimizer.step()
    updated = {name: parameter.detach().cpu().clone() for name, parameter in adapter_params}
    deltas = {name: float((value.float() - initial[name].float()).abs().max()) for name, value in updated.items()}
    if not all(torch.isfinite(value).all().item() for value in updated.values()) or not any(deltas.values()):
        raise AssertionError("optimizer step did not produce finite, nonzero adapter updates")

    checkpoint_root = output / "checkpoints"
    metadata = lora_metadata_dict(
        rank=8, alpha=16, dropout=0.0, target_modules=contract.target_modules, task_type="CAUSAL_LM"
    )
    checkpoint = save_checkpoint(
        str(checkpoint_root),
        "publication_acceptance",
        1,
        model,
        optimizer,
        trainer_state={"optimizer_step": 1, "seed": 1707, "base_model_digest": contract.base_digest},
        weight_sync_manifest={},
        adapter_contract={"save_mode": "adapter", "lora": metadata},
        adapter_only=True,
    )
    exported = Path(export_peft_adapter(str(checkpoint_root), "publication_acceptance", 1, str(output / "adapter")))
    # Compare the DCP -> PEFT round-trip to actual post-step tensors, not a mock writer.
    expected = peft_adapter_state_dict(list(updated.items()))
    actual = load_file(str(exported / "adapter_model.safetensors"))
    if actual.keys() != expected.keys() or any(not torch.equal(actual[key], value) for key, value in expected.items()):
        raise AssertionError("exported PEFT weights differ from the optimizer-step weights")
    producer = "relax.backends.fsdp.checkpoint.export_peft_adapter"
    (exported / PROVENANCE).write_bytes(canonical(provenance(contract, exported, producer=producer, export_step=1)))
    snapshot = snapshot_adapter(exported, store, version_id="TRAINED", base_model_digest=contract.base_digest)
    contract.validate(snapshot.path)
    evidence = {
        "state": "SEALED",
        "producer": producer,
        "exported_step": 1,
        "export_step": 1,
        "scope": "single-process real HF optimizer step and existing Relax DCP save/export; no distributed trainer",
        "checkpoint": checkpoint,
        "save_function": "relax.backends.fsdp.checkpoint.save_checkpoint",
        "model_path": str(model_path),
        "base_model_digest": contract.base_digest,
        "loss": float(loss.detach().cpu()),
        "training_tokens": batch["input_ids"].shape[1],
        "injected_layers": injected,
        "changed_tensors": sum(value > 0 for value in deltas.values()),
        "max_adapter_update": max(deltas.values()),
        "exported_tensor_count": len(actual),
        "round_trip_exact": True,
        "snapshot": {"version_id": "TRAINED", "digest": snapshot.digest, "path": str(snapshot.path)},
    }
    (output / "export.json").write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New output directory; never overwrites evidence")
    args = parser.parse_args()
    export_training_step(args.model, args.store, args.output)


if __name__ == "__main__":
    main()
