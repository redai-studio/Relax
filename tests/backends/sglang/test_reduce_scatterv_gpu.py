# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise variable-size DP MoE reductions with real NCCL and CUDA graphs."""

from inspect import signature
from pathlib import Path
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")


def _check_reduce_scatterv(rank: int, init_method: str) -> None:
    from sglang.srt.distributed.parallel_state import (
        cleanup_dist_env_and_memory,
        get_tp_group,
        graph_capture,
        init_distributed_environment,
        initialize_model_parallel,
        set_custom_all_reduce,
    )
    from sglang.srt.environ import envs
    from sglang.srt.layers import dp_attention
    from sglang.srt.layers.moe.utils import should_use_dp_reduce_scatterv
    from sglang.srt.runtime_context import publish
    from sglang.srt.server_args import ServerArgs

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    args = ServerArgs(
        model_path="dummy",
        device="cuda",
        tp_size=4,
        dp_size=4,
        ep_size=4,
        enable_dp_attention=True,
        moe_a2a_backend="none",
        disable_custom_all_reduce=True,
    )
    if "ranks" in signature(publish).parameters:
        from sglang.srt.runtime_context import SpawnRanks

        publish(args, role="test", ranks=SpawnRanks(world_rank=rank, gpu_id=rank))
    else:
        publish(args, role="test")
    set_custom_all_reduce(False)
    init_distributed_environment(
        world_size=4, rank=rank, local_rank=rank, distributed_init_method=init_method, backend="nccl", timeout=120
    )
    try:
        if "tensor_model_parallel_size" in signature(initialize_model_parallel).parameters:
            initialize_model_parallel(
                tensor_model_parallel_size=4, expert_model_parallel_size=4, attention_data_parallel_size=4
            )
        else:
            # SGLang 0.5.21 reads the widths from the published runtime context.
            initialize_model_parallel()
        group = get_tp_group()
        model_config = SimpleNamespace(hf_config=SimpleNamespace(), hidden_size=4, dtype=torch.bfloat16)
        if "model_config" in signature(dp_attention.initialize_dp_attention).parameters:
            dp_attention.initialize_dp_attention(args, model_config)
        else:
            dp_attention.initialize_dp_attention(args)
            dp_attention.init_dp_gathered_buffer(model_config)
        assert should_use_dp_reduce_scatterv()
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            for sizes in ([0, 1, 3, 4], [1, 3, 0, 4], [0, 0, 0, 0], None):
                counts = sizes if sizes is not None else [2] * 4
                count, offset = counts[rank], sum(counts[:rank])
                rows = sum(counts)
                positions = 2 * torch.arange(rows, device=device).reshape(-1, 1)
                positions = positions + torch.arange(4, device=device) / 4
                for output_mode in ("provided", "allocated", "alias"):
                    value = torch.empty((rows, 4), device=device, dtype=dtype)
                    output = None
                    if output_mode == "provided":
                        output = torch.empty((count, 4), device=device, dtype=dtype)
                    elif output_mode == "alias":
                        output = value.narrow(0, offset, count)

                    def reset(phase: int) -> None:
                        if rank == 1:
                            value.copy_(positions + 1 + phase)
                        else:
                            value.fill_((256, 0, -256, 0.5)[rank])

                    def check(result: torch.Tensor, phase: int) -> None:
                        assert result.dtype == dtype and result.shape == (count, 4)
                        if output is not None:
                            assert result is output
                        expected = (positions + 1.5 + phase).to(dtype).narrow(0, offset, count)
                        torch.testing.assert_close(result, expected, rtol=0, atol=0)

                    reset(0)
                    check(group.reduce_scatterv(value, output=output, sizes=sizes), 0)
                    with graph_capture() as capture:
                        for _ in range(2):
                            reset(0)
                            result = group.reduce_scatterv(value, output=output, sizes=sizes)
                        torch.cuda.synchronize()
                        check(result, 0)
                        reset(0)
                        torch.cuda.synchronize()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, stream=capture.stream):
                            result = group.reduce_scatterv(value, output=output, sizes=sizes)
                    for phase in range(3):
                        reset(phase)
                        graph.replay()
                        torch.cuda.synchronize()
                        check(result, phase)
                    del graph, result

        if not hasattr(group, "_deterministic_reduce_scatterv"):
            return  # 0.5.17 is covered by the regular NCCL and graph matrix above.

        # The deterministic dispatch must use the same FP32 accumulation as the
        # regular PyNCCL path, including when replayed from a CUDA graph.
        envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.set(True)
        sizes = [0, 1, 3, 4]
        value = torch.full((8, 4), (256, 1, -256, 0.5)[rank], device=device, dtype=torch.bfloat16)
        original = value.clone()
        output = torch.empty((sizes[rank], 4), device=device, dtype=torch.bfloat16)
        with graph_capture() as capture:
            group.reduce_scatterv(value, output=output, sizes=sizes)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=capture.stream):
                result = group.reduce_scatterv(value, output=output, sizes=sizes)
        graph.replay()
        torch.cuda.synchronize()
        assert result is output
        torch.testing.assert_close(result, torch.full_like(output, 1.5), rtol=0, atol=0)
        torch.testing.assert_close(value, original, rtol=0, atol=0)
    finally:
        cleanup_dist_env_and_memory()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 4,
    reason="The real NCCL reduce_scatterv and CUDA graph test requires four CUDA GPUs",
)
def test_reduce_scatterv_nccl_and_cuda_graph(tmp_path: Path) -> None:
    torch.multiprocessing.spawn(
        _check_reduce_scatterv, args=(f"file://{tmp_path / 'distributed-init'}",), nprocs=4, join=True
    )
