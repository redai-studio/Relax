# Copyright (c) 2026 Relax Authors. All Rights Reserved.


from types import SimpleNamespace

import pytest

from relax.inference.engine_spec import (
    STATIC_UPDATE_ENDPOINTS,
    InferenceEngineSpec,
    build_static_engine_config,
    drain_static_engine,
    validate_static_server_args,
)


def _policy_args() -> SimpleNamespace:
    return SimpleNamespace(
        hf_checkpoint="policy-checkpoint",
        teacher_hf_checkpoint="teacher-checkpoint",
        genrm_model_path="judge-checkpoint",
        genrm_engine_config={"mem_fraction_static": 0.45},
        sglang_model_path="policy-checkpoint",
        sglang_tokenizer_path="policy-tokenizer",
        sglang_served_model_name="policy-name",
        sglang_load_format="dummy",
        sglang_quantization="fp8",
        sglang_enable_lora=True,
        sglang_lora_paths=["policy-adapter"],
        sglang_speculative_algorithm="EAGLE",
        sglang_speculative_draft_model_path="policy-draft",
        sglang_pp_size=2,
        sglang_dp_size=2,
        sglang_ep_size=2,
        sglang_moe_dense_tp_size=1,
        sglang_chunked_prefill_size=1024,
        model_source=object(),
        rollout_external=True,
        lora_rank=8,
        lora_adapter_mode=True,
        enable_mtp_training=True,
        use_rollout_routing_replay=True,
        optimize_routing_replay=True,
    )


@pytest.mark.parametrize("role,checkpoint", [("teacher", "teacher-checkpoint"), ("genrm", "judge-checkpoint")])
def test_static_engine_config_isolates_policy_identity_and_dynamic_features(role, checkpoint):
    policy = _policy_args()
    before = vars(policy).copy()

    isolated, overrides = build_static_engine_config(policy, role)

    assert vars(policy) == before
    assert isolated is not policy
    assert isolated.hf_checkpoint == overrides["model_path"] == checkpoint
    assert isolated.model_source is None
    assert isolated.rollout_external is False
    assert isolated.sglang_load_format == "auto"
    assert isolated.sglang_enable_lora is False
    assert isolated.sglang_enable_weights_cpu_backup is True
    assert isolated.lora_rank == 0
    assert isolated.enable_mtp_training is False
    assert isolated.use_rollout_routing_replay is False
    assert not hasattr(isolated, "sglang_tokenizer_path")
    assert not hasattr(isolated, "sglang_served_model_name")
    assert not hasattr(isolated, "sglang_speculative_algorithm")
    assert not hasattr(isolated, "sglang_lora_paths")
    assert not hasattr(isolated, "sglang_quantization")
    assert isolated.sglang_chunked_prefill_size == 1024


def test_teacher_role_overrides_preserve_its_parallelism_and_quantization():
    role_overrides = {
        "model_path": "teacher-v2",
        "tokenizer_path": "teacher-tokenizer",
        "served_model_name": "teacher-name",
        "quantization": "awq",
        "speculative_algorithm": "EAGLE",
        "pp_size": 1,
        "dp_size": 4,
        "enable_memory_saver": True,
    }

    isolated, overrides = build_static_engine_config(_policy_args(), "teacher", role_overrides)

    for key, value in role_overrides.items():
        assert overrides[key] == value
        assert getattr(isolated, f"sglang_{key}") == value
    assert isolated.sglang_ep_size == 1
    assert isolated.sglang_moe_dense_tp_size is None
    assert "enable_weights_cpu_backup" not in role_overrides


@pytest.mark.parametrize("role", ["teacher", "genrm"])
@pytest.mark.parametrize(
    "invalid",
    [
        {"load_format": "dummy"},
        {"enable_weights_cpu_backup": False},
        {"enable_lora": True},
        {"lora_paths": ["adapter"]},
    ],
)
def test_static_engine_rejects_unrecoverable_or_dynamic_role_overrides(role, invalid):
    args = _policy_args()
    if role == "genrm":
        args.genrm_engine_config = invalid
    with pytest.raises(ValueError):
        build_static_engine_config(args, role, invalid if role == "teacher" else None)


@pytest.mark.parametrize("role", ["teacher", "genrm"])
@pytest.mark.parametrize("operation", sorted(STATIC_UPDATE_ENDPOINTS))
def test_static_engine_spec_rejects_all_dynamic_weight_operations(role, operation):
    with pytest.raises(RuntimeError, match="STATIC"):
        InferenceEngineSpec(role).require_operation(operation)


@pytest.mark.parametrize(
    "operation", ["generate", "release_memory_occupation", "resume_memory_occupation", "abort_request"]
)
def test_static_engine_spec_allows_inference_and_memory_lifecycle(operation):
    InferenceEngineSpec("teacher").require_operation(operation)


def test_static_server_validation_rejects_backend_without_backup_support():
    spec = InferenceEngineSpec("teacher")
    with pytest.raises(ValueError, match="CPU weight backup"):
        validate_static_server_args(spec, {"load_format": "auto"})
    validate_static_server_args(spec, {"load_format": "auto", "enable_weights_cpu_backup": True})


def _drain_scenario(*, failure=None, strict=True, flush_results=(False, True)):
    calls, now = [], [0.0]
    results = iter(flush_results)

    def step(name, timeout):
        assert 0 < timeout <= 3
        calls.append(name)
        if name == failure:
            raise RuntimeError(f"{name} failed")
        return next(results, False) if name == "flush" else {"ok": True}

    def sleep(duration):
        now[0] += duration

    def execute():
        return drain_static_engine(
            pause=lambda timeout: step("pause", timeout),
            abort=lambda timeout: step("abort", timeout),
            flush=lambda timeout: step("flush", timeout),
            release=lambda timeout: step("release", timeout),
            strict=strict,
            timeout=3,
            release_timeout=3,
            clock=lambda: now[0],
            sleep=sleep,
        )

    return execute, calls, now


def test_managed_static_drain_requires_pause_and_idle_ack_before_release():
    execute, calls, _now = _drain_scenario()
    assert execute() == {"ok": True}
    assert calls == ["pause", "abort", "flush", "abort", "flush", "release"]


@pytest.mark.parametrize("failure", ["pause", "abort", "flush"])
def test_managed_static_drain_failure_never_releases_memory(failure):
    execute, calls, _now = _drain_scenario(failure=failure)
    with pytest.raises(RuntimeError, match=failure):
        execute()
    assert "release" not in calls


def test_managed_static_drain_deadline_does_not_release_busy_engine():
    execute, calls, now = _drain_scenario(flush_results=(False,))
    with pytest.raises(TimeoutError):
        execute()
    assert now[0] == 3
    assert "release" not in calls


def test_legacy_genrm_drain_explicitly_retains_best_effort_pause():
    execute, calls, _now = _drain_scenario(failure="pause", strict=False, flush_results=(True,))
    execute()
    assert calls == ["pause", "abort", "flush", "release"]


def test_actual_sglang_static_engine_guards_and_partial_resume(monkeypatch):
    pytest.importorskip("sglang")

    from relax.backends.sglang.sglang_engine import SGLangEngine

    engine = SGLangEngine.__new__(SGLangEngine)
    engine.engine_spec = InferenceEngineSpec("teacher")
    engine.node_rank = 0
    engine.server_host = "teacher.example"
    engine.server_port = 15000
    engine._resident_tags = set()
    engine._admission_paused = True
    calls = []

    def post(url, **kwargs):
        calls.append(url.rsplit("/", 1)[-1])
        assert kwargs.get("timeout") is not None
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {})

    monkeypatch.setattr("relax.backends.sglang.sglang_engine.requests.post", post)
    with pytest.raises(RuntimeError, match="STATIC"):
        engine.update_weights_from_tensor([])
    with pytest.raises(RuntimeError, match="STATIC"):
        engine.register_dcs()
    engine.resume_memory_occupation(tags=["weights"])
    assert calls == ["resume_memory_occupation"]
    engine.resume_memory_occupation(tags=["kv_cache", "cuda_graph"])
    engine.continue_generation()
    assert calls == ["resume_memory_occupation", "resume_memory_occupation", "continue_generation"]


def test_actual_sglang_launch_keeps_follower_process_and_rolls_back_failed_head(monkeypatch):
    pytest.importorskip("sglang")

    from relax.backends.sglang import sglang_engine as module

    alive = [True]
    process = SimpleNamespace(pid=1234, start=lambda: None, is_alive=lambda: alive[0], join=lambda timeout: None)
    killed = []
    monkeypatch.setattr(module.multiprocessing, "set_start_method", lambda *args, **kwargs: None)
    monkeypatch.setattr(module.multiprocessing, "Process", lambda **kwargs: process)

    def kill(pid):
        killed.append(pid)
        alive[0] = False

    monkeypatch.setattr(module, "kill_process_tree", kill)

    def fail_health(**kwargs):
        raise TimeoutError("backend startup timed out")

    monkeypatch.setattr(module, "_wait_server_healthy", fail_health)
    assert module.launch_server_process(SimpleNamespace(node_rank=1)) is process
    assert killed == []
    with pytest.raises(TimeoutError):
        module.launch_server_process(SimpleNamespace(node_rank=0, url=lambda: "http://engine.example", api_key=None))
    assert killed == [process.pid]


def test_actual_sglang_preinit_shutdown_acknowledges_empty_ownership():
    pytest.importorskip("sglang")

    from relax.backends.sglang.sglang_engine import SGLangEngine

    engine = SGLangEngine.__new__(SGLangEngine)
    engine.args = SimpleNamespace(rollout_external=False)
    engine.shutdown()


def test_actual_sglang_shutdown_does_not_acknowledge_live_process(monkeypatch):
    pytest.importorskip("sglang")

    from relax.backends.sglang import sglang_engine as module

    engine = module.SGLangEngine.__new__(module.SGLangEngine)
    engine.args = SimpleNamespace(rollout_external=False)
    alive = [True]
    process = SimpleNamespace(pid=1234, join=lambda timeout: None, is_alive=lambda: alive[0])
    engine.process = process
    monkeypatch.setattr(module, "kill_process_tree", lambda pid: None)

    with pytest.raises(RuntimeError, match="remains alive"):
        engine.shutdown()
    assert engine.process is process
    alive[0] = False
    engine.shutdown()
    assert engine.process is None
