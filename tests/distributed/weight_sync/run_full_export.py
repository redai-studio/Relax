# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""FULL export from real Megatron-Core training, with an independent HF
oracle."""

import argparse
import copy
import json
import os
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace


def run(root: Path, bridge: bool = False) -> None:
    import torch
    import torch.distributed as dist
    from megatron.core import parallel_state as mpu
    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_local_spec,
        get_gpt_layer_with_transformer_engine_spec,
    )
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.enums import AttnBackend
    from megatron.core.transformer.transformer_config import TransformerConfig
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from relax.backends.megatron.weight_update.export_full_checkpoint import export_full_checkpoint
    from relax.distributed.weight_sync.checkpoint import write_checkpoint
    from relax.distributed.weight_sync.store import VersionStore
    from relax.utils.distributed_utils import init_gloo_group

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=90))
    group = init_gloo_group(distributed_timeout_minutes=2)
    mpu.initialize_model_parallel(tensor_model_parallel_size=world)
    model_parallel_cuda_manual_seed(0)
    config = TransformerConfig(
        num_layers=2,
        hidden_size=128,
        num_attention_heads=2,
        num_query_groups=2,
        kv_channels=64,
        ffn_hidden_size=256,
        tensor_model_parallel_size=world,
        params_dtype=torch.bfloat16,
        bf16=True,
        normalization="RMSNorm",
        qk_layernorm=True,
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
        add_bias_linear=False,
        add_qkv_bias=False,
        attention_dropout=0.0,
        attention_backend=AttnBackend.unfused,
        hidden_dropout=0.0,
        persist_layer_norm=False,
        gradient_accumulation_fusion=False,
        bias_activation_fusion=False,
        bias_dropout_fusion=False,
        masked_softmax_fusion=False,
        apply_rope_fusion=False,
        use_cpu_initialization=False,
    )
    spec = (
        get_gpt_layer_with_transformer_engine_spec(qk_layernorm=True)
        if bridge
        else get_gpt_layer_local_spec(qk_layernorm=True, normalization="RMSNorm")
    )
    model = (
        GPTModel(
            config=config,
            transformer_layer_spec=spec,
            vocab_size=1024,
            max_sequence_length=64,
            position_embedding_type="rope",
            rotary_base=1000000,
            parallel_output=True,
            share_embeddings_and_output_weights=False,
        )
        .cuda()
        .to(torch.bfloat16)
    )

    class Wrapped(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.config = module.config

    wrapped = [Wrapped(model)]
    args = SimpleNamespace(
        num_experts=None,
        mtp_num_layers=None,
        update_weight_buffer_size=1024**2,
        hidden_size=128,
        num_attention_heads=2,
        num_query_groups=2,
        kv_channels=64,
        vocab_size=1024,
        q_lora_rank=None,
        num_layers=2,
        transformer_impl="transformer_engine" if bridge else "local",
        model_name="qwen3",
        megatron_to_hf_mode="raw",
    )
    hf_config = Qwen3Config(
        vocab_size=1024,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=64,
        tie_word_embeddings=False,
        rms_norm_eps=config.layernorm_epsilon,
        rope_parameters={"rope_type": "default", "rope_theta": 1000000.0},
    )
    hf_config.architectures = ["Qwen3ForCausalLM"]
    root.mkdir(parents=True, exist_ok=True)
    store = VersionStore(root / "store", disk_budget=512 * 1024**2, max_weight_bytes=32 * 1024**2)
    if bridge:
        args.megatron_to_hf_mode = "bridge"
        args.hf_checkpoint = str(root / "hf-source")
        if rank == 0:
            Qwen3ForCausalLM(copy.deepcopy(hf_config)).to(torch.bfloat16).save_pretrained(args.hf_checkpoint)
        dist.barrier(group=group)
    actor = SimpleNamespace(args=args, model=[model] if bridge else wrapped, hf_config=hf_config)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], device="cuda")
    positions = torch.arange(8, device="cuda").unsqueeze(0)
    attention = torch.triu(torch.ones(1, 1, 8, 8, device="cuda", dtype=torch.bool), diagonal=1)
    reports = []
    try:
        for step in range(6):
            loss_value = None
            if step:
                optimizer.zero_grad(set_to_none=True)
                loss = model(tokens, positions, attention, labels=tokens).mean()
                loss.backward()
                # This small driver uses raw modules rather than Megatron DDP.
                # Match finalize_model_grads for norms shared across TP heads.
                for name, param in model.named_parameters():
                    if param.grad is not None and ("q_layernorm" in name or "k_layernorm" in name):
                        dist.all_reduce(param.grad, group=mpu.get_tensor_model_parallel_group())
                optimizer.step()
                loss_value = float(loss.detach())
            receipt = export_full_checkpoint(
                actor,
                group,
                store_path=root / "store",
                stream="megatron",
                version=step,
                scratch=root / f"scratch-{rank}",
                disk_budget=512 * 1024**2,
                max_weight_bytes=32 * 1024**2,
                update_id=f"step-{step}",
                metrics_dir=root / "metrics",
            )
            with torch.no_grad():
                reference = model(tokens, positions, attention, runtime_gather_output=True).float().cpu()
            error = None
            if rank == 0:
                try:
                    manifest = store.manifest("megatron", step)
                    assert manifest["kind"] == "full" and manifest["base_version"] is None
                    assert receipt["root"] == manifest["root"]
                    checkpoint = write_checkpoint(store, "megatron", step, root / f"checkpoint-{step}")
                    hf_model, loading = Qwen3ForCausalLM.from_pretrained(
                        checkpoint,
                        output_loading_info=True,
                        dtype=torch.float32,
                    )
                    assert not loading.get("missing_keys") and not loading.get("unexpected_keys"), loading
                    with torch.no_grad():
                        logits = hf_model(tokens.cpu()).logits.float()
                    torch.testing.assert_close(logits, reference, atol=0.02, rtol=0.03)
                    reports.append(
                        {
                            "receipt": receipt,
                            "loss": loss_value,
                            "hf_forward_max_abs_error": float((logits - reference).abs().max()),
                            "logical_tensors": len(manifest["tensors"]),
                        }
                    )
                    (root / "progress.json").write_text(json.dumps(reports, indent=2))
                    del hf_model
                except Exception as caught:
                    error = f"{type(caught).__name__}: {caught}"
            errors = [error]
            dist.broadcast_object_list(errors, src=0, group=group)
            if errors[0]:
                raise AssertionError(errors[0])
        if rank == 0:
            assert len({report["receipt"]["root"] for report in reports}) == 6
            (root / "result.json").write_text(
                json.dumps(
                    {
                        "passed": True,
                        "tp": world,
                        "bridge": bridge,
                        "optimizer_updates": 5,
                        "steps": reports,
                    },
                    indent=2,
                )
            )
    finally:
        mpu.destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bridge", action="store_true")
    options = parser.parse_args()
    run(options.root, options.bridge)
