# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise variable-size DP MoE reductions with real NCCL and CUDA graphs."""

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
    publish(args, role="test")
    set_custom_all_reduce(False)
    init_distributed_environment(
        world_size=4, rank=rank, local_rank=rank, distributed_init_method=init_method, backend="nccl", timeout=120
    )
    try:
        initialize_model_parallel(
            tensor_model_parallel_size=4, expert_model_parallel_size=4, attention_data_parallel_size=4
        )
        group = get_tp_group()
        dp_attention.initialize_dp_attention(
            args, SimpleNamespace(hf_config=SimpleNamespace(), hidden_size=4, dtype=torch.bfloat16)
        )
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
