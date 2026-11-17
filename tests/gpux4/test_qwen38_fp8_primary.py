# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Four-rank DP4/EP2 regression for mixed MXFP8/BF16 storage and native resume.

Operator tests cannot exercise dense/expert optimizer siblings or sharded
master weights. This small model uses real FP8 safetensors, Qwen Bridge import,
four GPUs and a bounded spawn timeout. Only the model's mapping registry is
adapted to the small fixture. A separate TP2 phase checks real NCCL scatter.
"""

import json
import time
from argparse import Namespace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist


def _write_hf_fixture(root: Path) -> None:
    from safetensors.torch import save_file
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

    folder = root / "hf"
    folder.mkdir()
    config = Qwen4ExpTextConfig(
        architectures=["Qwen4ExpForCausalLM"],
        hidden_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=128,
        dtype="bfloat16",
        quantization_config={"quant_method": "fp8", "weight_block_size": [128, 128]},
    )
    config.save_pretrained(folder)
    generator = torch.Generator().manual_seed(731)
    state, reference = {}, {}

    def quantized(name, shape):
        payload = torch.randn(shape, generator=generator).to(torch.float8_e4m3fn)
        scales = torch.linspace(0.017, 0.023, steps=(shape[0] // 128) * (shape[1] // 128))
        scales = scales.reshape(shape[0] // 128, shape[1] // 128).bfloat16()
        state[name], state[name + "_scale_inv"] = payload, scales
        return payload.float() * scales.float().repeat_interleave(128, 0).repeat_interleave(128, 1)

    for expert in range(4):
        prefix = f"model.layers.0.mlp.experts.{expert}"
        gate = quantized(prefix + ".gate_proj.weight", (128, 256))
        up = quantized(prefix + ".up_proj.weight", (128, 256))
        reference[f"expert{expert}"] = torch.cat((gate, up))
    prefix = "model.layers.0.mlp.shared_expert"
    reference["mlp.shared_experts.linear_fc1.weight"] = torch.cat(
        [quantized(prefix + f".{projection}.weight", (128, 256)) for projection in ("gate_proj", "up_proj")]
    )
    reference["mtp.weight"] = quantized("mtp.layers.0.mlp.experts.0.down_proj.weight", (256, 256))
    vision = torch.randn(256, 256, generator=generator).bfloat16() * 0.02
    state["visual.weight"] = vision
    reference["vision_model.weight"] = vision.float()
    prefix = "model.layers.1.ple.ple_embedding.ngram_embedding"
    ple = torch.randn(32, 256, generator=generator).to(torch.float8_e4m3fn)
    scale = torch.tensor([0.019775390625], dtype=torch.bfloat16)
    for index, shard in enumerate(torch.split(ple, [12, 12, 8])):
        state[prefix + f".shard_{index}.weight"] = shard.clone()
    state[prefix + ".weight_scale"] = scale
    reference["ple.weight"] = ple.float() * scale.float()
    assert any(torch.any(value != value.bfloat16().float()) for value in reference.values())
    save_file(state, folder / "model.safetensors")
    (folder / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "model.safetensors" for name in state}})
    )
    save_file(reference, root / "reference.safetensors")


def _check_tp_import(root: Path) -> None:
    from megatron.bridge.models.conversion.model_bridge import WeightConversionTask
    from megatron.bridge.models.conversion.param_mapping import (
        ColumnParallelMapping,
        GatedMLPMapping,
        RowParallelMapping,
    )
    from megatron.bridge.models.qwen.qwen4_exp_bridge import Qwen4ExpBridge
    from megatron.core import parallel_state
    from safetensors.torch import load_file

    source = load_file(root / "hf/model.safetensors")
    expected = load_file(root / "reference.safetensors")["mtp.weight"]
    name = "mtp.layers.0.mlp.experts.0.down_proj.weight"
    bridge = Qwen4ExpBridge()
    bridge.hf_pretrained = SimpleNamespace(
        config=SimpleNamespace(quantization_config={"weight_block_size": [128, 128]})
    )
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    for kind in (ColumnParallelMapping, RowParallelMapping, GatedMLPMapping):
        if kind is GatedMLPMapping:
            mapping = kind("weight", gate=name, up=name)
            half = expected.chunk(2, dim=0)[tp_rank]
            reference = torch.cat((half, half))
        else:
            mapping = kind("weight", name)
            reference = expected.chunk(2, dim=0 if kind is ColumnParallelMapping else 1)[tp_rank]
        module = torch.nn.Module()
        module.weight = torch.nn.Parameter(torch.zeros(reference.shape, dtype=torch.bfloat16, device="cuda"))
        task = WeightConversionTask("weight", "weight", mapping, megatron_module=module, param_weight=module.weight)
        loaded = bridge.maybe_modify_loaded_hf_weight(mapping.hf_param, source)
        converted = bridge._convert_loaded_hf_weight(task, loaded)
        assert converted.dtype == torch.float32 and converted.is_cuda
        torch.testing.assert_close(converted.cpu(), reference, rtol=0, atol=0)


def _worker(rank: int, root: str) -> None:
    from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
    from megatron.bridge.models.conversion.param_mapping import GatedMLPMapping, ReplicatedMapping, RowParallelMapping
    from megatron.bridge.models.qwen.qwen4_exp_bridge import PLENGramEmbeddingMapping, Qwen4ExpBridge
    from megatron.core import dist_checkpointing, parallel_state
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.extensions.transformer_engine import TEColumnParallelGroupedLinear, TEColumnParallelLinear
    from megatron.core.fp8_utils import get_fp8_context
    from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
    from megatron.core.tensor_parallel.layers import set_defaults_if_not_set_tensor_model_parallel_attributes
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.module import MegatronModule
    from megatron.core.transformer.transformer_config import TransformerConfig
    from safetensors.torch import load_file
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor

    from relax.backends.megatron import checkpoint as checkpoint_module
    from relax.backends.megatron import model as training
    from relax.models.qwen3_8.precision import configure_fp8

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=(Path(root) / "rendezvous").as_uri(),
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=180),
    )
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=2)
    _check_tp_import(Path(root))
    parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(expert_model_parallel_size=2)
    model_parallel_cuda_manual_seed(1234)
    config = TransformerConfig(
        num_layers=1,
        hidden_size=256,
        num_attention_heads=4,
        bf16=True,
        params_dtype=torch.bfloat16,
        fp8="e4m3",
        fp8_recipe="mxfp8",
        num_moe_experts=4,
        expert_model_parallel_size=2,
        expert_tensor_parallel_size=1,
        gradient_accumulation_fusion=False,
        add_bias_linear=False,
    )
    args = Namespace(
        bf16=True,
        fp16=False,
        fp8="e4m3",
        fp8_recipe="mxfp8",
        fp8_param_gather=True,
        use_distributed_optimizer=True,
        reuse_grad_buf_for_mxfp8_param_ag=True,
        overlap_param_gather=False,
        empty_unused_memory_level=0,
        custom_megatron_before_train_step_hook_path=None,
        ci_test=False,
        seq_length=64,
        micro_batch_size=1,
        decoder_seq_length=None,
        loss_type="sft",
        megatron_to_hf_mode="bridge",
        hf_checkpoint=str(Path(root) / "hf"),
        ref_load=None,
        load_main_params_from_ckpt=False,
    )

    class TinyModel(MegatronModule):
        def __init__(self):
            super().__init__(config)
            self.share_embeddings_and_output_weights = False
            self.mlp = MegatronModule(config)
            self.mlp.experts = MegatronModule(config)
            self.mlp.experts.linear_fc1 = TEColumnParallelGroupedLinear(
                2,
                256,
                256,
                config=config,
                init_method=lambda w: torch.nn.init.normal_(w, std=0.02),
                bias=False,
                skip_bias_add=False,
                is_expert=True,
            )
            self.mlp.shared_experts = MegatronModule(config)
            self.mlp.shared_experts.linear_fc1 = TEColumnParallelLinear(
                256,
                256,
                config=config,
                init_method=lambda w: torch.nn.init.normal_(w, std=0.02),
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
                tp_group=parallel_state.get_tensor_model_parallel_group(),
            )
            self.vision_model = torch.nn.Linear(256, 256, bias=False, device="cuda", dtype=torch.bfloat16)
            self.ple = torch.nn.Embedding(32, 256, device="cuda", dtype=torch.bfloat16)
            self.mtp = torch.nn.Linear(256, 256, bias=False, device="cuda", dtype=torch.bfloat16)

        def forward(self, x):
            routed = self.mlp.experts.linear_fc1(x, [32, 32])[0]
            shared = self.mlp.shared_experts.linear_fc1(x)[0]
            positions = torch.arange(x.shape[0], device=x.device) % 32
            return routed + shared + self.vision_model(x) + self.mtp(x) + self.ple(positions)

    reference = load_file(Path(root) / "reference.safetensors")
    original = {name: weight for name, weight in reference.items() if not name.startswith("expert")}
    expert_start = parallel_state.get_expert_model_parallel_rank() * 2
    original.update({f"mlp.experts.linear_fc1.weight{i}": reference[f"expert{expert_start + i}"] for i in range(2)})

    def build_stack(*_):
        model = TinyModel()
        configure_fp8(model, args)
        for parameter in model.parameters():
            set_defaults_if_not_set_tensor_model_parallel_attributes(parameter)
        ddp = DistributedDataParallel(
            config,
            DistributedDataParallelConfig(
                use_distributed_optimizer=True,
                grad_reduce_in_fp32=True,
                overlap_grad_reduce=True,
                overlap_param_gather=False,
                bucket_size=262144,
                fp8_param_gather=True,
                reuse_grad_buf_for_mxfp8_param_ag=True,
            ),
            model,
        )
        optimizer = get_megatron_optimizer(
            OptimizerConfig(
                optimizer="adam",
                lr=0.01,
                min_lr=0.001,
                weight_decay=0.01,
                adam_beta1=0.9,
                adam_beta2=0.98,
                clip_grad=1.0,
                bf16=True,
                params_dtype=torch.bfloat16,
                use_distributed_optimizer=True,
                use_precision_aware_optimizer=True,
                store_param_remainders=False,
                main_params_dtype=torch.float32,
                main_grads_dtype=torch.float32,
                exp_avg_dtype=torch.float32,
                exp_avg_sq_dtype=torch.float32,
                fp8_recipe="mxfp8",
                reuse_grad_buf_for_mxfp8_param_ag=True,
                overlap_param_gather=False,
            ),
            [ddp],
        )
        assert len(optimizer.chained_optimizers) == 2
        return [ddp], optimizer, SimpleNamespace(step=lambda **_: None)

    def tiny_mapping_registry(bridge):
        return MegatronMappingRegistry(
            GatedMLPMapping(
                "mlp.experts.linear_fc1.weight*",
                gate="model.layers.0.mlp.experts.*.gate_proj.weight",
                up="model.layers.0.mlp.experts.*.up_proj.weight",
            ),
            GatedMLPMapping(
                "mlp.shared_experts.linear_fc1.weight",
                gate="model.layers.0.mlp.shared_expert.gate_proj.weight",
                up="model.layers.0.mlp.shared_expert.up_proj.weight",
            ),
            RowParallelMapping("mtp.weight", "mtp.layers.0.mlp.experts.0.down_proj.weight"),
            ReplicatedMapping("vision_model.weight", "visual.weight"),
            PLENGramEmbeddingMapping(
                "ple.weight", "model.layers.1.ple.ple_embedding.ngram_embedding.shard_{}.weight", num_shards=3
            ),
        )

    def load_hf(models, optimizer, *_args, **_kwargs):
        return checkpoint_module._load_checkpoint_hf(models, optimizer, args, None)

    def master_snapshot(models, optimizer, *, check_initial=False):
        names = {p: name for name, p in models[0].module.named_parameters()}
        result = {}
        for child in optimizer.chained_optimizers:
            for group, masters in zip(child.model_float16_groups, child.shard_fp32_from_float16_groups, strict=True):
                for parameter, master in zip(group, masters, strict=True):
                    assert master.dtype == torch.float32
                    name = names[parameter]
                    result[name] = master.detach().cpu().clone()
                    if check_initial:
                        span = child._get_model_param_range_map(parameter)["param"]
                        expected = original[name].flatten()[span.start : span.end].float()
                        torch.testing.assert_close(result[name], expected, rtol=0, atol=0)
        return result

    def snapshot(models, optimizer):
        result = {"master." + key: value for key, value in master_snapshot(models, optimizer).items()}
        for name, parameter in models[0].module.named_parameters():
            if isinstance(parameter, MXFP8Tensor):
                for suffix in ("_rowwise_data", "_columnwise_data", "_rowwise_scale_inv", "_columnwise_scale_inv"):
                    result[name + suffix] = getattr(parameter, suffix).detach().cpu().clone()
            else:
                result[name] = parameter.detach().cpu().clone()
        for index, child in enumerate(optimizer.chained_optimizers):
            for pindex, state in enumerate(child.optimizer.state.values()):
                for key in ("exp_avg", "exp_avg_sq"):
                    result[f"adam.{index}.{pindex}.{key}"] = state[key].detach().cpu().clone()
        return result

    metadata = {
        "dp_cp_group": parallel_state.get_data_parallel_group(with_context_parallel=True),
        "distrib_optim_sharding_type": "fully_reshardable",
    }

    def state(models, optimizer, loading=False):
        result = {"model": models[0].module.sharded_state_dict(metadata=metadata)}
        result["optimizer"] = optimizer.sharded_state_dict(result, is_loading=loading, metadata=metadata)
        return result

    context = {}

    def forward_backward(**_):
        models = context["models"]
        with get_fp8_context(config):
            output = models[0](context["input"])
        loss = (output.float() - context["target"]).square().mean()
        context["loss"] = loss.detach().clone()
        loss.backward()
        models[0].finish_grad_sync()
        return []

    def train(models, optimizer, scheduler, step):
        context["models"] = models
        training.train_one_step(args, 0, step, [], models, optimizer, scheduler, 1, 4)
        return context["loss"]

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Qwen4ExpBridge, "mapping_registry", tiny_mapping_registry)
        patch.setattr(training, "setup_model_and_optimizer", build_stack)
        patch.setattr(training, "load_checkpoint", load_hf)
        patch.setattr(training, "get_args", lambda: args)
        patch.setattr(training, "get_forward_backward_func", lambda: forward_backward)
        patch.setattr(training.mpu, "is_pipeline_last_stage", lambda **_: False)
        patch.setattr(training, "_is_global_zero_token_step", lambda _: False)
        models, optimizer, scheduler, _ = training.initialize_model_and_optimizer(args)
        master_snapshot(models, optimizer, check_initial=True)
        assert all(
            isinstance(p, MXFP8Tensor) == ("experts." in name) for name, p in models[0].module.named_parameters()
        )
        context.update(
            input=torch.randn(64, 256, device="cuda", dtype=torch.bfloat16),
            target=torch.randn(64, 256, device="cuda"),
        )
        initial = master_snapshot(models, optimizer)
        train(models, optimizer, scheduler, 0)
        trained = master_snapshot(models, optimizer)
        for name in initial:
            assert not torch.equal(initial[name], trained[name]), (rank, name)
        checkpoint = Path(root) / "checkpoint"
        checkpoint.mkdir(exist_ok=True)
        dist_checkpointing.save(state(models, optimizer), str(checkpoint))
        saved = snapshot(models, optimizer)
        expected_loss = train(models, optimizer, scheduler, 1)
        expected = snapshot(models, optimizer)

        def load_native(models, optimizer, *_args, **_kwargs):
            loaded = dist_checkpointing.load(state(models, optimizer, loading=True), str(checkpoint))
            models[0].module.load_state_dict(loaded["model"])
            optimizer.load_state_dict(loaded["optimizer"])
            return 1, 0

        patch.setattr(training, "load_checkpoint", load_native)
        models, optimizer, scheduler, iteration = training.initialize_model_and_optimizer(args)
        assert iteration == 1
        restored = snapshot(models, optimizer)
        for name in saved:
            torch.testing.assert_close(
                restored[name], saved[name], rtol=0, atol=0, msg=f"restored rank={rank}: {name}"
            )
        actual_loss = train(models, optimizer, scheduler, 1)
        torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
        actual = snapshot(models, optimizer)
        assert actual.keys() == expected.keys()
        for name in actual:
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0, msg=name)
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="Requires four MXFP8-capable GPUs")
def test_qwen38_fp8_primary_master_updates_and_native_resume(tmp_path):
    import torch.multiprocessing as mp
    import transformer_engine.pytorch as te

    if not te.is_mxfp8_available():
        pytest.skip("Requires MXFP8-capable Blackwell GPUs")

    _write_hf_fixture(tmp_path)
    context = mp.spawn(_worker, args=(str(tmp_path),), nprocs=4, join=False)
    deadline = time.monotonic() + 240
    try:
        while not context.join(timeout=5):
            if time.monotonic() > deadline:
                raise TimeoutError("Qwen3.8 FP8 distributed regression exceeded 240 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=10)
