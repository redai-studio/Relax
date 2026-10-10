# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Tiny installed DDP/DistributedOptimizer/DCP resume probe on one selected
GPU."""

from __future__ import annotations

import argparse
import copy
import dataclasses
import datetime
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compute", choices=("bf16", "fp8", "stock_fp8"), default="bf16")
    parser.add_argument("--seed", type=int, default=731)
    parser.add_argument("--load-checkpoint", type=Path, help="Compatible step5 fixture generated with the same seed")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)

    import torch
    import torch.distributed as dist
    import transformer_engine.pytorch as te
    from megatron.core import dist_checkpointing, parallel_state
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.extensions.transformer_engine import TEGroupedLinear
    from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
    from megatron.core.transformer.module import MegatronModule
    from megatron.core.transformer.transformer_config import TransformerConfig
    from transformer_engine.common.recipe import Float8BlockScaling, Format

    from relax.backends.megatron.compat import preserve_hdo_dp_reshardable_steps_on_load
    from relax.models.deepseek_v4.master_weights import (
        _master,
        install_master_weight_hooks,
        materialize,
    )
    from relax.models.deepseek_v4.qat import install_mxfp4_qat
    from relax.models.deepseek_v4.quantization import mxfp4_qdq

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    torch.cuda.set_device(0)
    install_master_weight_hooks()
    config = TransformerConfig(
        num_layers=1,
        hidden_size=256,
        num_attention_heads=4,
        bf16=True,
        params_dtype=torch.bfloat16,
        num_moe_experts=2,
        fp8="e4m3",
        fp8_recipe="blockwise",
        gradient_accumulation_fusion=True,
        calculate_per_token_loss=True,
    )
    recipe = Float8BlockScaling(fp8_format=Format.E4M3)
    metadata = {"distrib_optim_sharding_type": "dp_reshardable"}

    class Tiny(MegatronModule):
        def __init__(self, compute: str) -> None:
            super().__init__(config)
            self.experts = torch.nn.Module()
            self.experts.linear_fc1 = TEGroupedLinear(
                2,
                256,
                256,
                parallel_mode="column",
                config=config,
                init_method=torch.nn.init.normal_,
                bias=False,
                skip_bias_add=False,
                is_expert=True,
            )
            with torch.no_grad():
                for p in self.parameters():
                    p.normal_(0, 0.025)
            self.splits = [128, 128]
            original_workspace = self.experts.linear_fc1.get_weight_workspace.__func__
            install_mxfp4_qat(self, compute=compute)
            if compute == "stock_fp8":
                assert self.experts.linear_fc1.get_weight_workspace.__func__ is original_workspace
                assert self.experts.linear_fc1.te_quant_params is None

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            result, _ = self.experts.linear_fc1(value, self.splits)
            return result

    def create(compute: str = args.compute) -> tuple[Any, Any]:
        model = Tiny(compute)
        ddp = DistributedDataParallel(
            config,
            DistributedDataParallelConfig(
                grad_reduce_in_fp32=True,
                use_distributed_optimizer=True,
                overlap_grad_reduce=False,
                overlap_param_gather=False,
            ),
            model,
        )
        optimizer = get_megatron_optimizer(
            OptimizerConfig(
                optimizer="adam",
                lr=0.001,
                bf16=True,
                use_distributed_optimizer=True,
                optimizer_cpu_offload=True,
                optimizer_offload_fraction=1.0,
                overlap_cpu_optimizer_d2h_h2d=True,
                use_precision_aware_optimizer=True,
                fp8_recipe="blockwise",
                clip_grad=1.0,
            ),
            [ddp],
        )
        optimizer.reload_model_params()
        return ddp, optimizer

    def sharded(ddp: Any, optimizer: Any, *, loading: bool = False) -> dict[str, Any]:
        model_state = ddp.module.sharded_state_dict(metadata=metadata)
        return {
            "model": model_state,
            "optimizer": optimizer.sharded_state_dict(model_state, is_loading=loading, metadata=metadata),
        }

    def update(ddp: Any, optimizer: Any, number: int) -> None:
        compute = ddp.module.experts.linear_fc1._relax_mxfp4_qat.compute
        generations = [
            (wrapper, wrapper._relax_dsv4_fp4_generation)
            for wrapper in getattr(optimizer, "chained_optimizers", [optimizer])
            if getattr(wrapper, "_relax_dsv4_fp4_bindings", ())
        ]
        assert generations, "Optimizer did not bind expert masters"
        optimizer.zero_grad()
        ddp.zero_grad_buffer()
        reference = te.GroupedLinear(
            2, 256, 256, bias=False, params_dtype=torch.bfloat16, fuse_wgrad_accumulation=True, device="cuda"
        )
        with torch.no_grad():
            for name, param in ddp.module.experts.linear_fc1.named_parameters():
                getattr(reference, name).copy_(param)
                getattr(reference, name).main_grad = torch.zeros_like(param, dtype=torch.float32)
        reference_recipe = copy.deepcopy(recipe)
        if compute == "fp8":
            reference_recipe.fp8_quant_fwd_weight = dataclasses.replace(
                recipe.fp8_quant_fwd_weight, power_2_scale=True
            )
        for microbatch in range(2):
            splits = [0, 256] if microbatch else [128, 128]
            ddp.module.splits = splits
            inputs = torch.randn(
                (256, 256),
                dtype=torch.bfloat16,
                device="cuda",
                generator=torch.Generator(device="cuda").manual_seed(321 + number * 2 + microbatch),
                requires_grad=True,
            )
            reference_input = inputs.detach().clone().requires_grad_(True)
            with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
                result = te.distributed.checkpoint(ddp, inputs, use_reentrant=True) if microbatch else ddp(inputs)
                result.float().square().mean().backward()
                from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

                assert FP8GlobalStateManager.is_fp8_enabled(), "Expert override leaked outside its module"
                dense = te.Linear(256, 256, bias=False, params_dtype=torch.bfloat16, device="cuda")
                dense(inputs.detach())
                assert dense.fp8, "Non-expert projection did not retain FP8"
            with te.fp8_autocast(enabled=compute != "bf16", fp8_recipe=reference_recipe):
                expected = reference(reference_input, splits, is_first_microbatch=None)
                expected.float().square().mean().backward()
            assert torch.equal(result, expected), (
                "DeepSeek-V4 forward differs from independent effective-weight reference"
            )
            assert torch.equal(inputs.grad, reference_input.grad), (
                "DeepSeek-V4 dgrad differs from independent reference"
            )
            expert = ddp.module.experts.linear_fc1
            assert expert.fp8 == (compute != "bf16"), "Expert compute mode was not applied"
            if compute == "bf16":
                assert expert._relax_mxfp4_qat.workspace_calls == 0 and not expert._fp8_workspaces
            assert torch.isfinite(inputs.grad).all() and inputs.grad.abs().max() > 0
        for name, param in ddp.module.experts.linear_fc1.named_parameters():
            # Both references accumulate wgrad in FP32, avoiding a BF16 cast
            # of each microbatch's gradient before summation.
            assert torch.equal(param.main_grad, getattr(reference, name).main_grad), "Fused wgrad differs"
        assert all(wrapper._relax_dsv4_fp4_generation == before for wrapper, before in generations), (
            "Microbatches/recomputation unexpectedly repeated master QDQ"
        )
        ddp.finish_grad_sync()
        success, _, _ = optimizer.step()
        if not success:
            raise AssertionError("Tiny optimizer step was skipped")
        assert all(wrapper._relax_dsv4_fp4_generation == before + 1 for wrapper, before in generations), (
            "Successful optimizer update must materialize Q4 exactly once"
        )
        for wrapper in getattr(optimizer, "chained_optimizers", [optimizer]):
            for param, module in getattr(wrapper, "_relax_dsv4_fp4_bindings", ()):
                expected = mxfp4_qdq(_master(wrapper, param).to(param.device)).to(param.dtype)
                assert torch.equal(param, expected), "Post-step carrier differs from Q4(master)"
                assert not module._fp8_workspaces, "Stale FP8 workspace survived optimizer step"

    def capture(ddp: Any, optimizer: Any) -> dict[str, Any]:
        values = {f"model/{name}": param.detach().cpu().clone() for name, param in ddp.module.named_parameters()}
        for wrapper in getattr(optimizer, "chained_optimizers", [optimizer]):
            hdo = wrapper.optimizer
            for name, param in ddp.module.named_parameters():
                if param not in wrapper.model_param_group_index_map:
                    continue
                group, offset = wrapper.model_param_group_index_map[param]
                shard = hdo.param_groups[group]["params"][offset]
                for key, value in hdo.state[shard].items():
                    if isinstance(value, torch.Tensor):
                        values[f"state/{name}/{key}"] = value.detach().cpu().clone()
                values[f"working_param/{name}"] = hdo.param_to_inner_param[shard].detach().cpu().clone()
        return values

    def compare(actual: dict[str, Any], expected: dict[str, Any]) -> int:
        assert actual.keys() == expected.keys(), "Optimizer state keys changed after resume"
        for key in actual:
            assert torch.equal(actual[key], expected[key]), f"Checkpoint state differs: {key}"
        return len(actual)

    def restore(path: Path, compute: str = args.compute) -> tuple[Any, Any]:
        ddp, optimizer = create(compute)
        with preserve_hdo_dp_reshardable_steps_on_load():
            state = dist_checkpointing.load(sharded(ddp, optimizer, loading=True), str(path))
            ddp.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
        return ddp, optimizer

    report = {
        "scope": "2x256x256 experts, TE fused main_grad + DO/HDO + DCP; empty expert and recomputation",
        "expert_compute": args.compute,
        "seed": args.seed,
    }
    try:
        dist.init_process_group(
            "nccl",
            init_method=(output / "rendezvous").as_uri(),
            rank=0,
            world_size=1,
            timeout=datetime.timedelta(seconds=30),
        )
        parallel_state.initialize_model_parallel()
        ddp, optimizer = create()
        # Deliberate boundary case: FP32 master differs from BF16 before Q4.
        for wrapper in getattr(optimizer, "chained_optimizers", [optimizer]):
            for param, _ in getattr(wrapper, "_relax_dsv4_fp4_bindings", ()):
                master = _master(wrapper, param)
                master.zero_()
                master[:, 0::32] = 6.0
                master[:, 1::32] = 0.7501
                latent = master.clone()
                materialize(wrapper)
                assert torch.equal(master, latent), "QAT modified latent master"
                assert (param[:, 1::32] == 1).all()
                assert (mxfp4_qdq(master.to(torch.bfloat16))[:, 1::32] == 0.5).all()
                # Return to a realistic weight scale before numerical optimizer tests.
                master.normal_(0, 0.025)
            materialize(wrapper)
        for number in range(3):
            update(ddp, optimizer, number)
        first_path = output / "step3"
        first_path.mkdir()
        dist_checkpointing.save(sharded(ddp, optimizer), str(first_path))
        expected_loaded = capture(ddp, optimizer)
        for number in (3, 4):
            update(ddp, optimizer, number)
        expected_final = capture(ddp, optimizer)
        if args.load_checkpoint is not None:
            update(ddp, optimizer, 5)
            expected_external_next_step = capture(ddp, optimizer)
        ddp, optimizer = restore(first_path)
        report["loaded_comparison"] = compare(capture(ddp, optimizer), expected_loaded)
        for number in (3, 4):
            update(ddp, optimizer, number)
        last_path = output / "step5"
        last_path.mkdir()
        dist_checkpointing.save(sharded(ddp, optimizer), str(last_path))
        report["uninterrupted_comparison"] = compare(capture(ddp, optimizer), expected_final)
        alternate = "fp8" if args.compute == "bf16" else "bf16"
        ddp, optimizer = restore(last_path, alternate)
        report["cross_precision_loaded"] = compare(capture(ddp, optimizer), expected_final)
        update(ddp, optimizer, 5)
        report["cross_precision_step"] = {"from": args.compute, "to": alternate, "status": "PASS"}
        if args.load_checkpoint is not None:
            ddp, optimizer = restore(args.load_checkpoint.resolve())
            report["external_checkpoint"] = str(args.load_checkpoint.resolve())
            report["external_loaded_comparison"] = compare(capture(ddp, optimizer), expected_final)
            update(ddp, optimizer, 5)
            report["external_next_step_comparison"] = compare(capture(ddp, optimizer), expected_external_next_step)
        report["status"] = "PASS"
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        if parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
