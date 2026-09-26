# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

from relax.engine.inference import config_adapters


def test_teacher_env_matches_rollout_genrm_stability_envs(monkeypatch):
    # RELAX_OPD_PREEXPANDED_PATCH is passed through from the driver env (default
    # "0"); set it so the test verifies the pass-through, not the default value.
    monkeypatch.setenv("RELAX_OPD_PREEXPANDED_PATCH", "1")
    args = SimpleNamespace(fp16=True)

    env = config_adapters._build_teacher_engine_env(args)

    assert env["RELAX_OPD_PREEXPANDED_PATCH"] == "1"
    assert env["SGLANG_JIT_DEEPGEMM_PRECOMPILE"] == "false"
    assert env["SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK"] == "true"
    assert env["SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK"] == "true"
    assert env["SGLANG_MEMORY_SAVER_CUDA_GRAPH"] == "true"
    assert env["SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT"] == "true"
    assert env["SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION"] == "false"
    assert env["SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE"] == "false"
    assert env["SGLANG_MAMBA_CONV_DTYPE"] == "float16"
