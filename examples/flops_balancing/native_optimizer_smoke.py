# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""One-GPU smoke test of real Megatron optimizer capture (no model kernels).

Use the pinned Megatron source on PYTHONPATH. This is not Qwen3.5 training,
multi-rank reduction, or TP/PP/CP/EP validation.
"""

import argparse
import json
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist

from relax.utils.training.optimizer_step_capture import OptimizerStepCapture


def main() -> None:
    from megatron.core import parallel_state
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.optimizer import OptimizerConfig
    from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import HybridDeviceOptimizer
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
    from megatron.core.optimizer.optimizer import ChainedOptimizer, Float16OptimizerWithFloat16Params, FP32Optimizer
    from megatron.core.transformer.transformer_config import TransformerConfig

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("A visible CUDA GPU is required")
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    with tempfile.TemporaryDirectory() as temporary:
        dist.init_process_group("nccl", init_method=f"file://{temporary}/init", rank=0, world_size=1)
        parallel_state.initialize_model_parallel()
        try:
            for mode in ("fp32", "bf16_master", "chained", "cpu_offload"):
                config = OptimizerConfig(
                    lr=0.01, bf16=mode == "bf16_master", clip_grad=1.0, optimizer_cpu_offload=mode == "cpu_offload"
                )
                leaves = []
                for _ in range(2 if mode == "chained" else 1):
                    parameter = torch.nn.Parameter(
                        torch.tensor(
                            [1.0, 2.0], device="cuda", dtype=torch.bfloat16 if mode == "bf16_master" else torch.float32
                        )
                    )
                    base = (
                        HybridDeviceOptimizer(
                            [parameter],
                            offload_fraction=1.0,
                            cpu_optimizer_cls=torch.optim.AdamW,
                            gpu_optimizer_cls=torch.optim.AdamW,
                            overlap_cpu_optimizer_d2h_h2d=False,
                            param_update_in_fp32=True,
                            lr=0.01,
                        )
                        if mode == "cpu_offload"
                        else torch.optim.AdamW([parameter], lr=0.01)
                    )
                    leaf = (
                        Float16OptimizerWithFloat16Params(base, config, None, lambda _: None)
                        if mode == "bf16_master"
                        else FP32Optimizer(base, config, lambda _: None)
                    )
                    parameter.main_grad = torch.tensor([3.0, 4.0], device="cuda")
                    leaves.append(leaf)
                optimizer = ChainedOptimizer(leaves) if mode == "chained" else leaves[0]
                capture = OptimizerStepCapture(
                    optimizer, args.output / mode, group=dist.group.WORLD, device=torch.device("cuda")
                )
                with capture:
                    successful, _, _ = optimizer.step()
                capture.finish({"diagnostic_objective": 1.0}, bool(successful))
                total = 0
                for entry in capture.manifest["parameters"]:
                    payload = torch.load(args.output / mode / entry["file"], weights_only=True)
                    torch.testing.assert_close(payload["gradient"], torch.tensor([3.0, 4.0]), atol=0, rtol=0)
                    if payload["initial"].dtype != torch.float32 or not bool(payload["update"].abs().max() > 0):
                        raise AssertionError("Expected nonzero actual FP32 updates")
                    total += payload["update"].numel()
                results.append(dict(mode=mode, parameters=len(capture.parameters), elements=total, passed=True))
            # Exercise the actual target BF16 -> reduced decoupled gradient ->
            # FP32 CPU master path. DP=1 checks API/capture, not multi-rank math.
            config = OptimizerConfig(
                lr=0.01,
                bf16=True,
                use_distributed_optimizer=True,
                use_precision_aware_optimizer=True,
                optimizer_cpu_offload=True,
            )
            model_config = TransformerConfig(
                num_layers=1, hidden_size=8, num_attention_heads=1, params_dtype=torch.bfloat16, bf16=True
            )
            ddp_config = DistributedDataParallelConfig(grad_reduce_in_fp32=True, use_distributed_optimizer=True)
            module = torch.nn.Linear(8, 8, bias=False, device="cuda", dtype=torch.bfloat16)
            model = DistributedDataParallel(model_config, ddp_config, module, disable_bucketing=True)
            base = HybridDeviceOptimizer(
                list(model.parameters()),
                offload_fraction=1.0,
                param_update_in_fp32=True,
                cpu_optimizer_cls=torch.optim.AdamW,
                gpu_optimizer_cls=torch.optim.AdamW,
                overlap_cpu_optimizer_d2h_h2d=False,
                lr=0.01,
            )
            optimizer = DistributedOptimizer(
                base,
                config,
                None,
                None,
                [model],
                {0: model.buffers},
                parallel_state.get_data_parallel_group(),
                parallel_state.get_data_parallel_group_gloo(),
                0,
                0,
            )
            model.zero_grad_buffer()
            model(torch.ones(2, 8, device="cuda", dtype=torch.bfloat16)).float().sum().backward()
            model.finish_grad_sync()
            capture = OptimizerStepCapture(
                optimizer,
                args.output / "distributed_bf16_offload",
                group=dist.group.WORLD,
                device=torch.device("cuda"),
            )
            with capture:
                successful, _, _ = optimizer.step()
            capture.finish({"diagnostic_objective": 1.0}, bool(successful))
            elements = 0
            for entry in capture.manifest["parameters"]:
                payload = torch.load(capture.directory / entry["file"], weights_only=True)
                torch.testing.assert_close(
                    payload["gradient"], torch.full_like(payload["gradient"], 2.0), atol=0, rtol=0
                )
                assert payload["initial"].dtype == torch.float32 and bool(payload["update"].abs().max() > 0)
                elements += payload["update"].numel()
            assert elements == 64
            results.append(dict(mode="distributed_bf16_offload", elements=elements, dp_size=1, passed=True))
        finally:
            parallel_state.destroy_model_parallel()
            dist.destroy_process_group()
    report = dict(scope=__doc__, cases=results, passed=all(row["passed"] for row in results))
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
