# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Compare full checkpoint replay with and without GLM Flash KPool selection
reuse."""

from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank: int, world_size: int, rendezvous: str) -> None:
    from megatron.bridge.models.glm5next.glm53flash_bridge import GLM53FlashModelProvider, glm5next_hybrid_stack_spec
    from megatron.bridge.peft.lora import LoRA
    from megatron.core import parallel_state as ps
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.enums import AttnBackend
    from megatron.core.transformer.experimental_attention_variant import dsa, dsa_layout

    from relax.models.glm53_flash import configure_dsa_selection_reuse

    tp_size = min(2, world_size)
    cp_size = world_size // tp_size
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=rendezvous, rank=rank, world_size=world_size, timeout=timedelta(seconds=180)
    )
    try:
        ps.initialize_model_parallel(tensor_model_parallel_size=tp_size, context_parallel_size=cp_size)
        model_parallel_cuda_manual_seed(817)
        groups = ProcessGroupCollection.use_mpu_process_groups()
        generator = torch.Generator(device="cuda").manual_seed(177)
        hidden = torch.randn(1024, 1, 256, device="cuda", dtype=torch.bfloat16, generator=generator)
        inputs = []
        for split in (320, 576):
            cu = torch.tensor([0, split, 1024], device="cuda", dtype=torch.int32)
            packed = PackedSeqParams(
                qkv_format="thd",
                cu_seqlens_q=cu,
                cu_seqlens_kv=cu,
                cu_seqlens_q_padded=cu,
                cu_seqlens_kv_padded=cu,
                max_seqlen_q=max(split, 1024 - split),
                max_seqlen_kv=max(split, 1024 - split),
            )
            rows = dsa_layout.build_packed_allgather_cp_local_positions(
                cu, cp_size, dist.get_rank(groups.cp), cu.device, output_size=1024 // cp_size
            )
            local = hidden[rows].chunk(tp_size)[dist.get_rank(groups.tp)].contiguous()
            inputs.append((local, packed))

        for uniform in (1, 2):
            provider = GLM53FlashModelProvider(
                num_layers=2,
                hybrid_layer_pattern="D-",
                hybrid_stack_spec=glm5next_hybrid_stack_spec,
                hidden_size=256,
                ffn_hidden_size=512,
                num_attention_heads=32,
                vocab_size=256,
                make_vocab_size_divisible_by=64,
                add_bias_linear=False,
                gated_linear_unit=True,
                normalization="RMSNorm",
                position_embedding_type="none",
                qk_layernorm=True,
                q_lora_rank=128,
                kv_lora_rank=512,
                kv_channels=128,
                qk_head_dim=128,
                qk_pos_emb_head_dim=0,
                v_head_dim=128,
                apply_rope_fusion=False,
                multi_latent_attention=True,
                experimental_attention_variant="dsa",
                dsa_kernel_backend="tilelang",
                dsa_indexer_head_dim=128,
                dsa_indexer_n_heads=32,
                dsa_indexer_topk=128,
                dsa_indexer_kpool=4,
                dsa_indexer_loss_coeff=0.0,
                dsa_indexer_qk_proj_use_quantization=False,
                dsa_indexer_weights_proj_use_quantization=False,
                dsa_indexer_weights_proj_output_dtype="fp32",
                dsa_indexer_kpool_use_quantization=True,
                dsa_indexer_kpool_always_select_tail=True,
                enable_mhc_connections=True,
                mhc_num_residual_streams=4,
                mhc_learned_output_contract=False,
                mhc_norm_eps_inside_sqrt=True,
                mhc_keep_mappings_in_fp32=True,
                attention_backend=AttnBackend.flash,
                attention_dropout=0.0,
                hidden_dropout=0.0,
                tensor_model_parallel_size=tp_size,
                context_parallel_size=cp_size,
                cp_comm_type="allgather",
                linear_cp_mode="headwise",
                linear_num_key_heads=32,
                linear_num_value_heads=32,
                linear_key_head_dim=128,
                linear_value_head_dim=128,
                linear_conv_kernel_dim=4,
                sequence_parallel=True,
                bf16=True,
                params_dtype=torch.bfloat16,
                recompute_granularity="full",
                recompute_method="uniform",
                recompute_num_layers=uniform,
                gradient_accumulation_fusion=False,
            )
            provider.finalize()
            provider._pg_collection = groups
            language_model = provider.provide_language_model().cuda().train()
            peft = LoRA(target_modules=["linear_q_up_proj", "linear_proj"], dim=4, alpha=8, dropout=0.0)
            language_model = peft(language_model, training=True)

            selections: list[torch.Tensor] = []
            selection_calls = 0
            native_attention, native_selection = dsa._run_sparse_attention, dsa.fused_qk_topk_kpool

            def capture_attention(**kwargs):
                selections.append(kwargs["topk_indices"].detach().clone())
                return native_attention(**kwargs)

            def count_selection(*args, **kwargs):
                nonlocal selection_calls
                selection_calls += 1
                return native_selection(*args, **kwargs)

            def forward() -> list[torch.Tensor]:
                return [
                    language_model.decoder(local.detach().clone().requires_grad_(True), None, packed_seq_params=packed)
                    .float()
                    .square()
                    .mean()
                    for local, packed in inputs
                ]

            def backward(losses: list[torch.Tensor]) -> dict[str, torch.Tensor]:
                language_model.zero_grad(set_to_none=True)
                for loss in reversed(losses):
                    loss.backward()
                return {
                    name: param.grad.detach().clone()
                    for name, param in language_model.named_parameters()
                    if param.requires_grad and param.grad is not None
                }

            with (
                patch.object(dsa, "_run_sparse_attention", capture_attention),
                patch.object(dsa, "fused_qk_topk_kpool", count_selection),
            ):
                baseline_losses = forward()
                baseline_gradients = backward(baseline_losses)
                assert selection_calls == 4
                baseline_ids = selections.copy()

                configure_dsa_selection_reuse(SimpleNamespace(language_model=language_model))
                selections.clear()
                losses = forward()
                assert selection_calls == 6
                reused_gradients = backward(losses)
                assert selection_calls == 6, "Checkpoint replay must not recompute KPool selection"

            # Each run records two original forwards and then replays them in
            # reverse order. The indexer is the first layer so both runs score
            # identical inputs even if later fused layers round differently.
            assert len(baseline_ids) == len(selections) == 4
            for actual, expected in zip(selections, baseline_ids, strict=True):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(torch.stack(losses), torch.stack(baseline_losses), rtol=0, atol=2e-4)
            assert reused_gradients.keys() == baseline_gradients.keys()
            assert any(torch.count_nonzero(grad) for grad in baseline_gradients.values())
            for name, grad in reused_gradients.items():
                torch.testing.assert_close(grad, baseline_gradients[name], msg=name)
    finally:
        ps.destroy_model_parallel()
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="requires at least four CUDA GPUs for TP2/CP2")
def test_glm53_kpool_selection_reuse(tmp_path):
    world_size = 8 if torch.cuda.device_count() >= 8 else 4
    mp.spawn(
        _worker,
        args=(world_size, (tmp_path / "glm53-kpool").as_uri()),
        nprocs=world_size,
        join=True,
    )
