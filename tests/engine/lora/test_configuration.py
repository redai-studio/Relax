# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import argparse
import json
import sys
from types import ModuleType, SimpleNamespace

import httpx
import pytest

from relax.distributed.ray.rollout_validation import validate_server_group_gpu_indices
from relax.engine.lora.cli import run_command, service_url
from relax.engine.lora.publication import PublicationConfig
from tests.utils.test_arguments_opd_teacher_colocate import arguments_module as arguments_module_fixture


base_arguments_module = arguments_module_fixture


@pytest.fixture
def arguments_module(monkeypatch, request):
    # OPD is unrelated to this parser branch and imports PyTorch eagerly.
    # Keep the actual Agentic argument registration/validation under test.
    opd = ModuleType("relax.utils.opd.opd_utils")
    for name in (
        "add_opd_arguments",
        "is_managed_opd_teacher_enabled",
        "teacher_sglang_parse_args",
        "validate_managed_opd_teacher_colocate_args",
        "validate_opd_args",
    ):
        setattr(opd, name, lambda value, **kwargs: value)
    monkeypatch.setitem(sys.modules, opd.__name__, opd)
    return request.getfixturevalue("base_arguments_module")


def parse(module, tmp_path, extra=()):
    module.RouterArgs = SimpleNamespace(add_cli_args=lambda parser, **kwargs: parser)
    parser = argparse.ArgumentParser()
    module.get_slime_extra_args_provider()(parser)
    return parser.parse_args(
        [
            "--use-agentic-rollout",
            "--agent-command",
            "python agent.py",
            "--agent-cwd",
            str(tmp_path),
            *extra,
        ]
    )


def config_path(tmp_path):
    path = tmp_path / "publication.yaml"
    path.write_text("artifact_store: " + str(tmp_path / "store") + "\ncapacity: 2\n")
    return str(path)


def test_publication_argument_uses_existing_agentic_parser(arguments_module, tmp_path):
    args = parse(arguments_module, tmp_path, ["--lora-publication-config", config_path(tmp_path)])
    arguments_module._validate_agentic_rollout_args(args)
    assert args.lora_publication_config.endswith("publication.yaml")
    assert args.rollout_function_path == "relax.agentic.rollout.generate_rollout"


def test_publication_argument_accepts_native_session_lifecycle(arguments_module, tmp_path):
    args = parse(
        arguments_module, tmp_path, ["--lora-publication-config", config_path(tmp_path), "--agentic-session-lifecycle"]
    )
    args.sglang_enable_session_radix_cache = True
    args.sglang_radix_eviction_policy = "priority"
    arguments_module._validate_agentic_rollout_args(args)


def test_publication_argument_rejects_program_admission(arguments_module, tmp_path):
    args = parse(
        arguments_module, tmp_path, ["--lora-publication-config", config_path(tmp_path), "--agentic-program-admission"]
    )
    with pytest.raises(ValueError, match="publication"):
        arguments_module._validate_agentic_rollout_args(args)


@pytest.mark.parametrize(
    "field,value",
    [
        ("capacity", "true"),
        ("prepare_timeout_seconds", ".nan"),
        ("cleanup_timeout_seconds", ".inf"),
        ("auto_publish", "1"),
    ],
)
def test_publication_profile_rejects_invalid_limits(tmp_path, field, value):
    path = tmp_path / "bad.yaml"
    path.write_text(f"artifact_store: /tmp/store\n{field}: {value}\n")
    with pytest.raises(ValueError):
        PublicationConfig.read(str(path))


def test_publication_bootstrap_rejects_path_escape(tmp_path):
    from relax.engine.lora.publication import PublicationConfig

    path = tmp_path / "publication.yaml"
    path.write_text("artifact_store: /store\nbootstrap_version_id: ../mutable\n")
    with pytest.raises(ValueError, match="path-safe"):
        PublicationConfig.read(str(path))


@pytest.mark.parametrize("flag", ["true_on_policy_mode", "mask_offpolicy_in_partial_rollout"])
def test_publication_rejects_step_based_on_policy_shortcuts(flag, tmp_path):
    from types import SimpleNamespace

    args = SimpleNamespace(use_agentic_rollout=True, lora_adapter_mode=True, fully_async=True)
    setattr(args, flag, True)
    with pytest.raises(ValueError, match="on-policy"):
        PublicationConfig(str(tmp_path)).validate_args(args)


# These checks are independent; representative rows cover every branch/value
# without a 108-case Cartesian product that only repeats argument validation.
@pytest.mark.parametrize(
    "mode,engines,train_offload,rollout_offload,tensor_parallel",
    [
        ("sync", 2, False, False, 1),
        ("sync", 3, True, False, 2),
        ("sync", 8, False, True, 4),
        ("hybrid", 2, False, True, 2),
        ("hybrid", 3, True, True, 4),
        ("hybrid", 8, False, False, 1),
        ("fully_async", 2, True, True, 4),
        ("fully_async", 3, False, False, 1),
        ("fully_async", 8, True, False, 2),
    ],
)
def test_publication_accepts_resident_training_modes_and_multiple_engines(
    mode, engines, train_offload, tensor_parallel, rollout_offload
):
    args = SimpleNamespace(
        use_agentic_rollout=True,
        lora_adapter_mode=True,
        fully_async=mode != "sync",
        hybrid=mode == "hybrid",
        colocate=mode == "hybrid",
        offload_train=train_offload,
        offload_rollout=rollout_offload,
        rollout_num_gpus=engines * tensor_parallel,
        rollout_num_gpus_per_engine=tensor_parallel,
        train_backend="megatron",
    )
    PublicationConfig("/store").validate_args(args)


@pytest.mark.parametrize("fraction", [None, 0.5, 0.8, float("nan"), True])
def test_shared_gpu_requires_an_explicit_bounded_memory_budget(fraction):
    args = SimpleNamespace(
        use_agentic_rollout=True,
        lora_adapter_mode=True,
        rollout_num_gpus=1,
        rollout_num_gpus_per_engine=1,
        sglang_mem_fraction_static=fraction,
    )
    with pytest.raises(ValueError, match="per-engine"):
        PublicationConfig("/store", engines_per_gpu=2).validate_args(args)


def test_single_gpu_can_host_two_publication_targets():
    args = SimpleNamespace(
        use_agentic_rollout=True,
        lora_adapter_mode=True,
        rollout_num_gpus=1,
        rollout_num_gpus_per_engine=1,
        sglang_mem_fraction_static=0.35,
    )
    PublicationConfig("/store", engines_per_gpu=2).validate_args(args)


def test_colocated_publication_requires_explicit_memory_handoffs():
    args = SimpleNamespace(
        use_agentic_rollout=True,
        lora_adapter_mode=True,
        colocate=True,
        hybrid=False,
        fully_async=False,
        offload_train=True,
        offload_rollout=True,
        rollout_num_gpus=2,
        rollout_num_gpus_per_engine=1,
    )
    PublicationConfig("/store").validate_args(args)
    args.offload_train = False
    with pytest.raises(ValueError, match="both train and rollout offload"):
        PublicationConfig("/store").validate_args(args)


def test_publish_client_uses_existing_service_and_preserves_intent_on_replay():
    requests = []

    def handle(request):
        requests.append((request.url.path, json.loads(request.content)))
        return httpx.Response(202, json={"state": "PREPARING", "operation_id": "op"})

    args = SimpleNamespace(
        command="publish",
        rollout_url="http://serve/rollout/",
        version_id="B",
        request_id="intent",
        retry_of="failed",
        expected_default_epoch=1,
        wait_seconds=0,
    )
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        assert run_command(client, args)["operation_id"] == "op"
        assert run_command(client, args)["operation_id"] == "op"
    assert requests[0] == requests[1]
    assert requests[0][0] == "/rollout/lora/publications"
    assert requests[0][1] == {
        "version_id": "B",
        "request_id": "intent",
        "retry_of": "failed",
        "expected_default_epoch": 1,
    }


def test_service_role_is_appended_only_once():
    assert service_url("http://serve/", "actor") == "http://serve/actor"
    assert service_url("http://serve/actor/", "actor") == "http://serve/actor"


def test_export_has_stable_id_and_does_not_call_collective_in_client():
    calls = []

    def handle(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json={"state": "WAITING_BOUNDARY", "request_id": "export-1"})

    args = SimpleNamespace(
        command="export", actor_url="http://serve", version_id="B", request_id="export-1", publish=True, wait_seconds=0
    )
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        assert run_command(client, args)["state"] == "WAITING_BOUNDARY"
    assert calls == [{"version_id": "B", "request_id": "export-1", "publish": True}]


@pytest.mark.parametrize("sharing,engines,gpus", [(1, 2, 2), (2, 2, 1), (2, 4, 2), (3, 5, 2)])
def test_shared_engine_layout_counts_physical_gpu_slots(sharing, engines, gpus):
    args = dict(
        worker_type="regular",
        gpu_offset=0,
        num_gpus_per_engine=1,
        num_gpu_per_engine=1,
        num_engines=engines,
        num_available_gpus=gpus,
        rollout_num_gpus=gpus,
        rollout_num_gpus_per_engine=1,
        engines_per_gpu=sharing,
    )
    validate_server_group_gpu_indices(**args)
    args["num_available_gpus"] -= 1
    with pytest.raises(ValueError, match="GPU placement"):
        validate_server_group_gpu_indices(**args)
