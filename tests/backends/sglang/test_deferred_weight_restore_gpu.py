# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""GPU evidence for the same-policy, same-round deferred scoring restore.

Two engine loads total: the required backup case and a negative control. No
training/Ray cluster is needed. CI supplies Qwen3-4B through MODEL_DIR; a local
checkpoint can be selected with RELAX_GPU_TEST_MODEL.
"""

import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


class _RetainedUnbackedPages(AssertionError):
    """The CUDA allocator reused contents, making the control inconclusive."""


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a CUDA GPU and SGLang GPU runtime")
@pytest.mark.parametrize(
    "backup",
    [
        pytest.param(True, id="restore-current-policy"),
        pytest.param(
            False,
            id="negative-control",
            marks=pytest.mark.xfail(
                raises=_RetainedUnbackedPages,
                strict=False,
                reason="CUDA VMM makes no content guarantee without backup; physical pages may retain their contents",
            ),
        ),
    ],
)
def test_deferred_policy_logprobs_survive_release_resume(backup: bool) -> None:
    model = Path(
        os.environ.get("RELAX_GPU_TEST_MODEL", str(Path(os.environ.get("MODEL_DIR", "/models")) / "Qwen3-4B"))
    )
    if not model.is_dir():
        pytest.skip(f"GPU checkpoint missing: {model}; set RELAX_GPU_TEST_MODEL")
    try:
        from sglang.srt.entrypoints.engine import Engine
        from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

        TorchMemorySaverAdapter.create(enable=True)
    except (ImportError, OSError) as exc:
        pytest.skip(f"SGLang GPU runtime unavailable: {exc}")

    from relax.backends.sglang.sglang_engine import _compute_server_args
    from relax.engine.inference.deferred import needs_rollout_cpu_backup

    args = SimpleNamespace(
        hf_checkpoint=str(model),
        seed=42,
        offload_rollout=True,
        rollout_num_gpus_per_engine=1,
        num_gpus_per_node=4,
        sglang_pp_size=1,
        sglang_dp_size=1,
        sglang_ep_size=1,
        use_rollout_routing_replay=False,
        fp16=False,
        colocate=True,
        hybrid=False,
        rollout_num_gpus=4,
        resource={"actor": [1, 4], "rollout": [1, 4], "teacher": [1, 4]},
        use_opd=True,
        opd_type="sglang",
        teacher_hf_checkpoint=str(model),
        opd_teacher_defer=True,
    )
    assert needs_rollout_cpu_backup(args)
    server_args, _ = _compute_server_args(
        args,
        0,
        None,
        None,
        "127.0.0.1",
        0,
        base_gpu_id=0,
        sglang_overrides={
            "mem_fraction_static": 0.25,
            "max_total_tokens": 512,
            "context_length": 512,
            "disable_cuda_graph": True,
            "disable_radix_cache": True,
            "attention_backend": "triton",
            "dtype": "bfloat16",
            "log_level": "warning",
        },
    )
    assert server_args["tp_size"] == 1
    assert server_args["enable_memory_saver"] is True
    assert server_args["enable_weights_cpu_backup"] is True
    # The negative control deliberately bypasses Relax's safety check, to test
    # the old unsafe engine configuration rather than weaken production code.
    server_args["enable_weights_cpu_backup"] = backup

    engine = Engine(**server_args)
    try:

        def logprobs() -> list[tuple[float | None, int, str | None]]:
            output = engine.generate(
                prompt="The capital of France is Paris. Two plus two equals four.",
                sampling_params={"temperature": 0, "max_new_tokens": 0},
                return_logprob=True,
                logprob_start_len=0,
            )
            return output["meta_info"]["input_token_logprobs"]

        before = logprobs()
        scored = [item[0] for item in before if item[0] is not None]
        assert len(scored) > 5 and all(math.isfinite(value) for value in scored)
        engine.release_memory_occupation()
        engine.resume_memory_occupation()
        after = logprobs()
        assert [item[1] for item in before] == [item[1] for item in after]
        # TP=1, identical prefill shapes, no radix-cache hits or sampling:
        # backup copies bytes, so 1e-6 absolute tolerance only allows FP32
        # log-softmax rounding, far below typical BF16 logprob differences.
        equal = all(
            (left[0] is None and right[0] is None)
            or (
                left[0] is not None
                and right[0] is not None
                and math.isclose(left[0], right[0], rel_tol=0, abs_tol=1e-6)
            )
            for left, right in zip(before, after, strict=True)
        )
        if backup:
            assert equal, "CPU backup did not restore the current policy logprobs"
        else:
            # cuMemCreate without backup makes no content guarantee: a driver
            # may return the same physical pages. Only that inconclusive
            # outcome is xfailed; engine/API failures still fail the test.
            if equal:
                raise _RetainedUnbackedPages("No-backup CUDA VMM allocation happened to retain its contents")
            assert not equal
    finally:
        engine.shutdown()
