# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Generate deterministic test adapters; these are not training-export
evidence."""

import argparse
import json
from pathlib import Path

from relax.engine.lora.artifact import PROVENANCE, ModelContract, canonical, provenance
from relax.engine.lora.snapshot import fingerprint_model


def make_fixtures(model_path: Path, output: Path, rank: int = 8) -> None:
    import torch

    from relax.utils.megatron_peft_utils import write_hf_peft_adapter

    config = json.loads((model_path / "config.json").read_text())
    if config.get("model_type") not in {"qwen2", "qwen3", "llama"}:
        raise ValueError("fixture generator supports dense Qwen2/Qwen3/Llama text models")
    if config.get("num_experts") or config.get("num_local_experts"):
        raise ValueError("use a dense model for the fixture experiment")
    hidden = config["hidden_size"]
    head_dim = config.get("head_dim", hidden // config["num_attention_heads"])
    widths = {
        "q_proj": config["num_attention_heads"] * head_dim,
        "v_proj": config.get("num_key_value_heads", config["num_attention_heads"]) * head_dim,
    }
    generator = torch.Generator().manual_seed(1707)
    weights = {}
    for layer in range(config["num_hidden_layers"]):
        for module, width in widths.items():
            prefix = f"base_model.model.model.layers.{layer}.self_attn.{module}"
            weights[f"{prefix}.lora_A.weight"] = torch.randn(rank, hidden, generator=generator) * 0.1
            weights[f"{prefix}.lora_B.weight"] = torch.randn(width, rank, generator=generator) * 0.1
    contract = ModelContract(fingerprint_model(model_path), config, rank, tuple(widths))
    for name, sign in (("A", 1), ("B", -1)):
        variant = {key: value * sign if ".lora_B." in key else value for key, value in weights.items()}
        write_hf_peft_adapter(
            variant,
            output / name,
            lora_rank=rank,
            lora_alpha=rank * 2,
            target_modules=list(widths),
            lora_dropout=0.0,
        )
        (output / name / PROVENANCE).write_bytes(
            canonical(
                provenance(
                    contract,
                    output / name,
                    producer="relax.fixture",
                    export_step=0,
                )
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=8)
    args = parser.parse_args()
    if args.rank <= 0:
        parser.error("rank must be positive")
    make_fixtures(args.model, args.output, args.rank)


if __name__ == "__main__":
    main()
