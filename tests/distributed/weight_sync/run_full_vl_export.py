# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Exercise native Megatron Qwen3-VL export with real image inputs and SGD."""

import argparse
import json
import os
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace


def run(root: Path) -> None:
    import torch
    import torch.distributed as dist
    from megatron.bridge import AutoBridge
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.rope import get_rope_index
    from megatron.core import parallel_state as mpu
    from megatron.core.transformer.enums import AttnBackend
    from transformers import Qwen3VLForConditionalGeneration
    from transformers.models.qwen3_vl.configuration_qwen3_vl import (
        Qwen3VLConfig,
        Qwen3VLTextConfig,
        Qwen3VLVisionConfig,
    )

    from relax.backends.megatron.weight_update.export_full_checkpoint import export_full_checkpoint
    from relax.distributed.weight_sync.checkpoint import write_checkpoint
    from relax.distributed.weight_sync.store import VersionStore
    from relax.utils.distributed_utils import init_gloo_group

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=180))
    group = init_gloo_group(distributed_timeout_minutes=3)
    rank, world = dist.get_rank(group), dist.get_world_size(group)
    if world not in (1, 2):
        raise ValueError("The tiny vision model supports TP=1 or TP=2")
    text_config = Qwen3VLTextConfig(
        vocab_size=151936,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=256,
        rms_norm_eps=1e-6,
        attention_bias=False,
        attention_dropout=0.0,
        use_cache=False,
        bos_token_id=151643,
        eos_token_id=151645,
        rope_parameters={"rope_type": "default", "rope_theta": 5000000.0, "mrope_section": [12, 10, 10]},
        dtype=torch.bfloat16,
    )
    vision_config = Qwen3VLVisionConfig(
        depth=2,
        hidden_size=128,
        intermediate_size=256,
        num_heads=2,
        hidden_act="gelu_pytorch_tanh",
        in_channels=3,
        patch_size=16,
        temporal_patch_size=2,
        spatial_merge_size=2,
        out_hidden_size=128,
        num_position_embeddings=64,
        deepstack_visual_indexes=[0, 1],
        dtype=torch.bfloat16,
    )
    hf_config = Qwen3VLConfig(
        text_config=text_config,
        vision_config=vision_config,
        image_token_id=151655,
        video_token_id=151656,
        vision_start_token_id=151652,
        vision_end_token_id=151653,
        tie_word_embeddings=False,
        architectures=["Qwen3VLForConditionalGeneration"],
        dtype=torch.bfloat16,
    )
    config_dir = root / "hf-source"
    if rank == 0:
        hf_config.save_pretrained(config_dir)
    dist.barrier(group=group)
    # Vision position embeddings are initially constructed on CPU. Seed it
    # before the provider initializes independent model-parallel CUDA RNGs.
    torch.manual_seed(0)
    provider = AutoBridge.from_hf_config(hf_config).to_megatron_provider(load_weights=False)
    provider.apply_overrides_and_finalize(
        dtype=torch.bfloat16,
        overrides={
            "tensor_model_parallel_size": world,
            "pipeline_model_parallel_size": 1,
            "virtual_pipeline_model_parallel_size": None,
            "context_parallel_size": 1,
            "expert_model_parallel_size": 1,
            "expert_tensor_parallel_size": 1,
            "sequence_parallel": False,
            "attention_backend": AttnBackend.unfused,
            "seq_length": 256,
            "language_max_sequence_length": 256,
            "should_pad_vocab": False,
            "share_embeddings_and_output_weights": False,
            "use_hf_vision_model": False,
            "gradient_accumulation_fusion": False,
            "cross_entropy_loss_fusion": False,
            "bias_activation_fusion": False,
            "bias_dropout_fusion": False,
            "masked_softmax_fusion": False,
            "apply_rope_fusion": False,
            "persist_layer_norm": False,
            "tp_comm_overlap": False,
            "use_cpu_initialization": False,
        },
    )
    provider.initialize_model_parallel(seed=0)
    models = provider.provide_distributed_model(wrap_with_ddp=False, use_cpu_initialization=False)
    assert len(models) == 1
    model = models[0]
    assert type(model.module.vision_model).__name__ == "Qwen3VLVisionModel"
    args = SimpleNamespace(
        num_experts=None,
        mtp_num_layers=None,
        update_weight_buffer_size=1024**2,
        hidden_size=128,
        num_attention_heads=2,
        num_query_groups=2,
        kv_channels=64,
        vocab_size=151936,
        q_lora_rank=None,
        num_layers=2,
        transformer_impl="transformer_engine",
        model_name="qwen3vl",
        megatron_to_hf_mode="raw",
        hf_checkpoint=str(config_dir),
    )
    actor = SimpleNamespace(args=args, model=models, hf_config=hf_config)
    tokens = torch.tensor([[1, 151652, 151655, 151655, 151655, 151655, 151653, 2]], device="cuda")
    grid = torch.tensor([[1, 4, 4]], device="cuda")
    pixels = torch.linspace(-1, 1, 16 * 1536, device="cuda").reshape(16, -1).to(torch.bfloat16)
    keep = torch.ones_like(tokens)
    positions, _ = get_rope_index(
        spatial_merge_size=2,
        image_token_id=151655,
        video_token_id=151656,
        vision_start_token_id=151652,
        input_ids=tokens,
        image_grid_thw=grid,
        attention_mask=keep,
    )
    assert positions.shape == (3, 1, 8)
    assert not torch.equal(positions[1], positions[2])
    native_batch = {
        "input_ids": tokens,
        "position_ids": positions,
        "attention_mask": torch.ones(1, 1, 8, 8, dtype=torch.bool, device="cuda").triu(diagonal=1),
        "pixel_values": pixels,
        "image_grid_thw": grid,
        "mm_token_type_ids": (tokens == 151655).long(),
    }
    hf_batch = {**native_batch, "attention_mask": keep, "use_cache": False}
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005)
    store = VersionStore(root / "store", disk_budget=4 * 1024**3, max_weight_bytes=256 * 1024**2)
    reports = []
    try:
        for step in range(6):
            loss_value = None
            vision_gradient_l1 = None
            if step:
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss = model(**native_batch, labels=tokens).mean()
                loss.backward()
                vision_gradients = [
                    param.grad.detach().float().abs().sum()
                    for param in model.module.vision_model.parameters()
                    if param.grad is not None
                ]
                assert vision_gradients, "The native vision tower did not receive gradients"
                vision_gradient = torch.stack(vision_gradients).sum()
                assert torch.isfinite(vision_gradient) and vision_gradient > 0
                vision_gradient_l1 = float(vision_gradient)
                for name, param in model.named_parameters():
                    if param.requires_grad and ("q_layernorm" in name or "k_layernorm" in name):
                        assert param.grad is not None, name
                        dist.all_reduce(param.grad, group=mpu.get_tensor_model_parallel_group())
                optimizer.step()
                loss_value = float(loss.detach())
            model.eval()
            receipt = export_full_checkpoint(
                actor,
                group,
                store_path=root / "store",
                stream="megatron-vl",
                version=step,
                scratch=root / f"scratch-{rank}",
                disk_budget=4 * 1024**3,
                max_weight_bytes=256 * 1024**2,
                update_id=f"vl-step-{step}",
                metrics_dir=root / "metrics",
            )
            with torch.no_grad():
                reference = model(**native_batch, runtime_gather_output=True).float().cpu()
                zero_pixels = {**native_batch, "pixel_values": torch.zeros_like(pixels)}
                no_image = model(**zero_pixels, runtime_gather_output=True).float().cpu()
            error = None
            if rank == 0:
                try:
                    image_effect = float((reference - no_image).abs().max())
                    assert image_effect > 1e-4, "The image input did not affect source logits"
                    manifest = store.manifest("megatron-vl", step)
                    assert manifest["kind"] == "full" and manifest["base_version"] is None
                    assert receipt["root"] == manifest["root"]
                    checkpoint = write_checkpoint(store, "megatron-vl", step, root / f"checkpoint-{step}")
                    hf_model, loading = Qwen3VLForConditionalGeneration.from_pretrained(
                        checkpoint,
                        dtype=torch.bfloat16,
                        attn_implementation="eager",
                        output_loading_info=True,
                    )
                    assert not loading.get("missing_keys") and not loading.get("unexpected_keys"), loading
                    hf_model = hf_model.cuda().eval()
                    with torch.no_grad():
                        logits = hf_model(**hf_batch).logits.float().cpu()
                    torch.testing.assert_close(logits, reference, atol=0.02, rtol=0.03)
                    reports.append(
                        {
                            "receipt": receipt,
                            "loss": loss_value,
                            "vision_gradient_l1": vision_gradient_l1,
                            "hf_forward_max_abs_error": float((logits - reference).abs().max()),
                            "image_input_max_abs_effect": image_effect,
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
                        "model_type": "qwen3_vl",
                        "source_vision_model": type(model.module.vision_model).__name__,
                        "initialized_from_hf_weights": False,
                        "optimizer_updates": 5,
                        "image_grid_thw": [[1, 4, 4]],
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
    run(parser.parse_args().root)
