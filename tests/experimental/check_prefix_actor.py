# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Reduced actual Qwen3.5 VLM/TE smoke test of actor_forward dispatch."""

import copy
import json
import os
import sys
import tempfile
import time
from pathlib import Path


sys.path.insert(0, os.environ["MEGATRON_ROOT"])
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from megatron.bridge.models.qwen_vl.qwen35_vl_provider import Qwen35VLModelProvider
from megatron.core import parallel_state as mpu
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed, vocab_parallel_cross_entropy
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig

from relax.utils.training.prefix_runtime import PrefixRuntime, actor_forward


world = int(os.environ.get("WORLD_SIZE", "1"))
rank = int(os.environ.get("RANK", "0"))
tp_size = int(os.environ.get("PREFIX_TP_SIZE", str(world)))
torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
rendezvous = tempfile.TemporaryDirectory()
torch.distributed.init_process_group(
    "nccl", init_method="env://" if world > 1 else "file://" + rendezvous.name + "/init", rank=rank, world_size=world
)
mpu.initialize_model_parallel(tensor_model_parallel_size=tp_size)
model_parallel_cuda_manual_seed(123)
vision = Qwen3_5VisionConfig(
    depth=1, hidden_size=288, intermediate_size=576, num_heads=4, out_hidden_size=128, deepstack_visual_indexes=[]
)
cfg = Qwen35VLModelProvider(
    num_layers=4,
    hidden_size=128,
    ffn_hidden_size=256,
    num_attention_heads=4,
    num_query_groups=4,
    kv_channels=32,
    linear_key_head_dim=32,
    linear_value_head_dim=32,
    linear_num_key_heads=4,
    linear_num_value_heads=4,
    vocab_size=512,
    seq_length=512,
    language_max_sequence_length=512,
    mrope_section=[2, 1, 1],
    vision_config=vision,
    params_dtype=torch.bfloat16,
    bf16=True,
    hidden_dropout=0.0,
    attention_dropout=0.0,
    image_token_id=500,
    video_token_id=501,
    vision_start_token_id=502,
    vision_end_token_id=503,
    bos_token_id=504,
    eos_token_id=505,
    attention_softmax_in_fp32=True,
    freeze_vision_model=True,
)
full_config_path = os.environ.get("PREFIX_FULL_CONFIG")
if full_config_path:
    source = json.loads(Path(full_config_path).read_text())
    text_config = source["text_config"]
    fields = {
        "num_layers": "num_hidden_layers",
        "hidden_size": "hidden_size",
        "ffn_hidden_size": "intermediate_size",
        "num_attention_heads": "num_attention_heads",
        "num_query_groups": "num_key_value_heads",
        "kv_channels": "head_dim",
        "vocab_size": "vocab_size",
        "layernorm_epsilon": "rms_norm_eps",
        "linear_conv_kernel_dim": "linear_conv_kernel_dim",
        "linear_key_head_dim": "linear_key_head_dim",
        "linear_value_head_dim": "linear_value_head_dim",
        "linear_num_key_heads": "linear_num_key_heads",
        "linear_num_value_heads": "linear_num_value_heads",
    }
    for target, origin in fields.items():
        setattr(cfg, target, text_config[origin])
    cfg.vision_config = Qwen3_5VisionConfig(**source["vision_config"])
    for name in ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id"):
        setattr(cfg, name, source[name])
    cfg.rotary_base = text_config["rope_parameters"]["rope_theta"]
    cfg.rotary_percent = text_config["rope_parameters"]["partial_rotary_factor"]
    cfg.mrope_section = text_config["rope_parameters"]["mrope_section"]
    cfg.linear_attention_freq = text_config["full_attention_interval"]


def snapshot(tensor):
    value = tensor.detach()
    return value.cpu().clone() if full_config_path else value.clone()


cfg.activation_func = torch.nn.functional.silu
fusion = os.environ.get("PREFIX_TEST_FUSION", "0") == "1"
cfg.gradient_accumulation_fusion = fusion
cfg.tensor_model_parallel_size = tp_size
cfg.sequence_parallel = tp_size > 1
cfg.recompute_granularity = "full"
cfg.recompute_method = "uniform"
cfg.recompute_num_layers = 1
cfg.finalize()
cfg._pg_collection = ProcessGroupCollection.use_mpu_process_groups()
model = cfg.provide(pre_process=True, post_process=True).cuda().to(torch.bfloat16).train()
row_length = 256
host_ids = [i % 400 for i in range(2 * row_length)]
with_images = os.environ.get("PREFIX_TEST_IMAGES", "0") == "1"
if with_images:
    for offset in (160, 416):
        host_ids[offset : offset + 18] = (
            [cfg.vision_start_token_id] + [cfg.image_token_id] * 16 + [cfg.vision_end_token_id]
        )
ids = torch.tensor(host_ids, device="cuda").reshape(1, -1)
cu = torch.tensor([0, row_length, 2 * row_length], device="cuda", dtype=torch.int32)
packed = PackedSeqParams(
    qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=row_length, max_seqlen_kv=row_length
)
packed.cu_seqlens_q_cpu = [0, row_length, 2 * row_length]
batch = {
    "packed_seq_params": packed,
    "prefix_inputs": [(10, tuple(host_ids[:row_length]), 32), (11, tuple(host_ids[row_length:]), 32)],
}
kwargs = dict(input_ids=ids, position_ids=None, attention_mask=None, labels=None, packed_seq_params=packed)
if with_images:
    from relax.utils.multimodal.vision_feature_cache import attach_vision_cache_info, install_train_vision_cache

    os.environ["RELAX_TRAIN_VISION_CACHE"] = "1"
    install_train_vision_cache(model)
    pixel_inputs = attach_vision_cache_info(
        "validation-session",
        {
            "pixel_values": torch.randn(
                128, 3 * 2 * 16 * 16, dtype=torch.bfloat16, generator=torch.Generator().manual_seed(71)
            ),
            "image_grid_thw": torch.tensor([[1, 8, 8], [1, 8, 8]]),
        },
    )
    kwargs.update(
        {key: value.cuda() if isinstance(value, torch.Tensor) else value for key, value in pixel_inputs.items()}
    )
if fusion and os.environ.get("PREFIX_BENCHMARK", "0") == "1":
    raise ValueError("Fusion mode is a correctness probe; benchmark currently uses ordinary AdamW")
if fusion:
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig

    model = DistributedDataParallel(
        cfg,
        DistributedDataParallelConfig(grad_reduce_in_fp32=True, overlap_grad_reduce=False),
        model,
        disable_bucketing=True,
        pg_collection=cfg._pg_collection,
    )
zero_prefix = os.environ.get("PREFIX_TEST_ZERO_PREFIX", "0") == "1"
limits = (0, 0, 0) if zero_prefix else (64, 128, 128)
initial = {k: snapshot(v) if isinstance(v, torch.Tensor) else copy.deepcopy(v) for k, v in model.state_dict().items()}
if os.environ.get("PREFIX_BENCHMARK_ONLY", "0") != "1":
    results = {}
    for mode in ("baseline", "detached_prefix", "detached_prefix_reuse"):
        model.load_state_dict(initial)
        for module in model.modules():
            if hasattr(module, "is_first_microbatch"):
                module.is_first_microbatch = True
        model.zero_grad(set_to_none=True)
        if fusion:
            model.zero_grad_buffer()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        runtime = None if mode == "baseline" else PrefixRuntime(mode, 2**26)
        outputs = []
        for limit in limits:
            batch["prefix_inputs"] = [
                (10, tuple(host_ids[:row_length]), limit),
                (11, tuple(host_ids[row_length:]), limit),
            ]
            output = actor_forward(model, kwargs, batch, runtime)
            selected = torch.tensor(
                list(range(limit, row_length)) + list(range(row_length + limit, 2 * row_length)), device="cuda"
            )
            chosen = output[:, selected]
            outputs.append(chosen.detach().clone())
            logprobs = -vocab_parallel_cross_entropy(output.float(), ids.roll(-1, dims=1))[:, selected]
            outputs.append(logprobs.detach().clone())
            (-logprobs.mean()).backward()
        if fusion:
            model.finish_grad_sync()
            for parameter in model.parameters():
                if parameter.requires_grad:
                    parameter.grad = parameter.main_grad.to(parameter.dtype)
        grads = {n: snapshot(p.grad) for n, p in model.named_parameters() if p.grad is not None}
        optimizer.step()
        weights = {n: snapshot(p) for n, p in model.named_parameters()}
        results[mode] = (outputs, grads, weights)
        if runtime is not None:
            if mode == "detached_prefix_reuse" and not zero_prefix:
                assert runtime.cache.hits >= 2 and runtime.cache.extensions >= 1
            runtime.close()
    for mode in ("detached_prefix", "detached_prefix_reuse"):
        for a, b in zip(results["baseline"][0], results[mode][0]):
            torch.testing.assert_close(a, b, atol=0.01, rtol=0.02)
    for slot in (1, 2):
        for name, a in results["detached_prefix"][slot].items():
            torch.testing.assert_close(a, results["detached_prefix_reuse"][slot][name], atol=0.001, rtol=0.02)
    if zero_prefix:
        for name, gradient in results["baseline"][1].items():
            torch.testing.assert_close(gradient, results["detached_prefix"][1][name], atol=0.001, rtol=0.02)
    print(
        json.dumps(
            {
                "scope": "full configured Qwen3.5, random weights, synthetic packed input"
                if full_config_path
                else "reduced Qwen3.5, synthetic packed input",
                "rank": rank,
                "fusion_ddp": fusion,
                "zero_prefix_gradient_check": zero_prefix,
                "images_and_existing_vision_cache": with_images,
                "tp": tp_size,
                "dp": world // tp_size,
                "sp": tp_size > 1,
                "recompute": "full",
                "three_mode_logits_and_logprobs": "passed",
                "detached_vs_reuse_gradients_and_adamw": "passed",
                "prefix_extension_and_hit": "not exercised (zero prefix)" if zero_prefix else "passed",
            }
        ),
        flush=True,
    )
if os.environ.get("PREFIX_BENCHMARK", "0") == "1":
    for mode in ("baseline", "detached_prefix", "detached_prefix_reuse"):
        model.load_state_dict(initial)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        for repeat in range(4):
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            runtime = None if mode == "baseline" else PrefixRuntime(mode, 2**26)
            forward_seconds = backward_seconds = 0.0
            for limit in limits:
                batch["prefix_inputs"] = [
                    (10, tuple(host_ids[:row_length]), limit),
                    (11, tuple(host_ids[row_length:]), limit),
                ]
                tick = time.perf_counter()
                output = actor_forward(model, kwargs, batch, runtime)
                selected = torch.tensor(
                    list(range(limit, row_length)) + list(range(row_length + limit, 2 * row_length)), device="cuda"
                )
                loss = vocab_parallel_cross_entropy(output.float(), ids.roll(-1, dims=1))[:, selected].mean()
                torch.cuda.synchronize()
                forward_seconds += time.perf_counter() - tick
                tick = time.perf_counter()
                loss.backward()
                torch.cuda.synchronize()
                backward_seconds += time.perf_counter() - tick
            stats = (
                {}
                if runtime is None
                else {
                    "prefix_tokens": runtime.prefix_tokens,
                    "suffix_tokens": runtime.suffix_tokens,
                    "computed_prefix_tokens": runtime.cache.computed_tokens,
                    "hits": runtime.cache.hits,
                    "builds": runtime.cache.builds,
                    "extensions": runtime.cache.extensions,
                    "cache_bytes": runtime.cache.bytes,
                    "windows": runtime.windows,
                }
            )
            if runtime is not None:
                runtime.close()
            optimizer.step()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            if repeat:
                print(
                    json.dumps(
                        {
                            "benchmark": "full_config_synthetic_random" if full_config_path else "reduced_synthetic",
                            "mode": mode,
                            "rank": rank,
                            "repeat": repeat,
                            "forward_s": forward_seconds,
                            "backward_s": backward_seconds,
                            "update_s": elapsed,
                            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                            **stats,
                        }
                    ),
                    flush=True,
                )
torch.distributed.destroy_process_group()
