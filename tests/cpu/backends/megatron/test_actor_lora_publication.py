# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import torch

from tests.cpu.engine.lora_versions.test_training import ThreadRanks, assert_rank_errors
from tests.cpu.utils.test_rollout_quantization import (
    compressed_checkpoint as compressed_checkpoint,
)
from tests.cpu.utils.test_rollout_quantization import (
    quantization_args as quantization_args,
)


pytest.importorskip("megatron.core")
from relax.backends.megatron import actor as actor_module
from relax.distributed.checkpoint_service.client.engine import CheckpointEngineClient


@pytest.mark.parametrize(
    "overrides", [None, {"quantization_config": {"quant_method": "fp8"}}, {"quantization_config": None}]
)
def test_checkpoint_sender_uses_effective_rollout_quantization(monkeypatch, tmp_path, overrides):
    actor = object.__new__(actor_module.MegatronTrainRayActor)
    checkpoint_quantization = {"quant_method": "fp8", "weight_block_size": [128, 128]}
    actor.hf_config = SimpleNamespace(quantization_config=checkpoint_quantization)
    (tmp_path / "config.json").write_text(json.dumps({"quantization_config": checkpoint_quantization}))
    actor.args = SimpleNamespace(
        sglang_json_model_override_args=None if overrides is None else json.dumps(overrides),
        hf_checkpoint=str(tmp_path),
        coordinator_url="http://coordinator.test",
        model_name="test-model",
        checkpoint_engine_backend="test",
    )
    actor.model = []
    actor.lock = object()
    for name in ("tensor_model", "pipeline_model", "expert_model", "context"):
        monkeypatch.setattr(actor_module.mpu, f"get_{name}_parallel_world_size", lambda: 1)
    monkeypatch.setattr(actor_module.mpu, "get_data_parallel_world_size", lambda **kwargs: 1)
    monkeypatch.setattr(actor_module.mpu, "get_data_parallel_rank", lambda **kwargs: 1)
    monkeypatch.setattr(actor_module.mpu, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(actor_module.mpu, "get_pipeline_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(actor_module.dist, "get_rank", lambda: 1)
    monkeypatch.setattr(actor_module, "run", asyncio.run)
    client = AsyncMock(return_value=object())
    monkeypatch.setattr(actor_module, "create_client", client)

    actual = actor._create_checkpoint_engine_client("actor")

    expected = checkpoint_quantization if overrides is None else overrides["quantization_config"]
    assert actual is client.return_value
    assert client.call_args.kwargs["quantization_config"] == expected


@pytest.mark.parametrize("failure", ["none", "acquire", "publish", "release"])
def test_managed_weight_sync_releases_topology_lease_on_all_exit_paths(monkeypatch, failure):
    ranks = ThreadRanks(monkeypatch)
    monkeypatch.setattr(actor_module, "get_gloo_group", lambda: ranks)
    monkeypatch.setattr(actor_module, "print_memory", lambda *args, **kwargs: None)
    monkeypatch.setattr(actor_module, "run", asyncio.run)
    monkeypatch.setattr(actor_module.ray, "get", lambda value, **kwargs: value)
    manager = SimpleNamespace(set_weight_updating=SimpleNamespace(remote=Mock(return_value=True)))
    lock = SimpleNamespace(
        acquire=SimpleNamespace(remote=Mock(return_value=True)),
        release=SimpleNamespace(remote=Mock()),
    )
    if failure == "acquire":
        lock.acquire.remote.side_effect = RuntimeError("acquire failed")
    elif failure == "release":
        lock.release.remote.side_effect = RuntimeError("release failed")
    actors = []
    for _rank in range(2):
        actor = object.__new__(actor_module.MegatronTrainRayActor)
        actor.args = SimpleNamespace(rollout_http_timeout=10)
        actor._managed_lora_publication = True
        actor.rollout_manager = manager
        actor._weight_sync_lock = lock
        actor.checkpoint_engine_client = SimpleNamespace(
            update_weights_for_rollout=AsyncMock(return_value=[]),
            init_process_groups_for_actor_fwd_ref=AsyncMock(),
        )
        if failure == "publish":
            actor.checkpoint_engine_client.update_weights_for_rollout.side_effect = RuntimeError("publish failed")
        actors.append(actor)
    results = ranks.run(
        lambda rank: actor_module.MegatronTrainRayActor.update_weights_fully_async.__wrapped__(
            actors[rank], 1, rollout_only=True
        )
    )
    if failure == "none":
        assert results == [None, None]
    else:
        assert_rank_errors(results, f"{failure} failed")
    assert [call.args for call in manager.set_weight_updating.remote.call_args_list] == [(True,), (False,)]
    assert lock.release.remote.call_count == (0 if failure == "acquire" else 1)


@pytest.mark.parametrize("actor_fwd_absent", [False, True])
def test_managed_health_check_never_enters_legacy_rollout_pause(monkeypatch, actor_fwd_absent):
    ranks = ThreadRanks(monkeypatch)
    monkeypatch.setattr(actor_module, "get_gloo_group", lambda: ranks)

    def reduce(tensor, *, op, group):
        values = ranks.exchange("reduce", tensor.tolist())
        tensor.copy_(torch.tensor(values).max(dim=0).values)

    monkeypatch.setattr(actor_module.dist, "all_reduce", reduce)
    get_url = Mock(return_value="http://actor-fwd")
    request = Mock(return_value=Mock())
    monkeypatch.setattr(actor_module, "get_serve_url", get_url)
    monkeypatch.setattr(actor_module.requests, "get", request)
    actors = []
    for _rank in range(2):
        actor = object.__new__(actor_module.MegatronTrainRayActor)
        actor._managed_lora_publication = True
        actor.args = SimpleNamespace(true_on_policy_mode=actor_fwd_absent, hybrid=False, rollout_http_timeout=10)
        actors.append(actor)
    assert (
        ranks.run(lambda rank: actor_module.MegatronTrainRayActor._check_services_health.__wrapped__(actors[rank]))
        == [(actor_fwd_absent, False)] * 2
    )
    if actor_fwd_absent:
        get_url.assert_not_called()
        request.assert_not_called()
    else:
        get_url.assert_called_once_with("actor_fwd")
        request.assert_called_once_with("http://actor-fwd/get_step", timeout=10)


@pytest.mark.parametrize("failure", ["lookup", "group_init"])
def test_single_rank_actor_fwd_initialization_failure_exits_all_trainers(monkeypatch, failure):
    ranks = ThreadRanks(monkeypatch)
    monkeypatch.setattr(actor_module, "get_gloo_group", lambda: ranks)
    monkeypatch.setattr(actor_module, "print_memory", lambda *args, **kwargs: None)
    monkeypatch.setattr(actor_module, "run", asyncio.run)
    monkeypatch.setattr(actor_module.ray, "get", lambda value, **kwargs: value)
    manager = SimpleNamespace(set_weight_updating=SimpleNamespace(remote=Mock(return_value=True)))
    lock = SimpleNamespace(
        acquire=SimpleNamespace(remote=Mock(return_value=True)), release=SimpleNamespace(remote=Mock())
    )
    actors = []
    for rank in range(2):
        client = object.__new__(CheckpointEngineClient)
        client.args = SimpleNamespace(ref_update_interval=None)
        client.role_info = SimpleNamespace(role_name="actor", rank=rank)
        client.coordinator_url = "http://coordinator"
        client._backend = SimpleNamespace(
            managed_lora_publication_enabled=True, init_process_groups_for_actor_fwd_ref=Mock()
        )
        client._http_client = SimpleNamespace(get=AsyncMock(return_value=Mock()))
        client.update_weights_for_rollout = AsyncMock(return_value=[])
        if rank == 1:
            if failure == "lookup":
                client._http_client.get.side_effect = RuntimeError("one rank failed")
            else:
                client._backend.init_process_groups_for_actor_fwd_ref.side_effect = RuntimeError("one rank failed")
        actor = object.__new__(actor_module.MegatronTrainRayActor)
        actor._managed_lora_publication = True
        actor.rollout_manager, actor._weight_sync_lock = manager, lock
        actor.checkpoint_engine_client = client
        actors.append(actor)
    assert_rank_errors(
        ranks.run(
            lambda rank: actor_module.MegatronTrainRayActor.update_weights_fully_async.__wrapped__(actors[rank], 1)
        ),
        "one rank failed",
    )
    assert all(actor.checkpoint_engine_client.update_weights_for_rollout.call_count == 0 for actor in actors)
    lock.release.remote.assert_called_once()
    assert [call.args for call in manager.set_weight_updating.remote.call_args_list] == [(True,), (False,)]


@pytest.mark.parametrize("managed", [False, True])
def test_manager_assignment_configures_publication_on_every_training_rank(monkeypatch, managed):
    monkeypatch.setattr(actor_module.ray, "get", lambda value, **kwargs: value)
    config = object() if managed else None
    lock = object()
    manager = SimpleNamespace(
        set_train_parallel_config=SimpleNamespace(remote=Mock()),
        get_weight_sync_lock=SimpleNamespace(remote=Mock(return_value=lock)),
        get_lora_training_config=SimpleNamespace(remote=Mock(return_value=config)),
    )
    for rank in (0, 1):
        actor = object.__new__(actor_module.MegatronTrainRayActor)
        actor.args = SimpleNamespace(
            rank=rank,
            debug_rollout_only=False,
            fully_async=True,
            hybrid=False,
            colocate=False,
            offload_rollout=False,
            lora_adapter_mode=True,
        )
        actor.train_parallel_config = {}
        backend = SimpleNamespace(configure_lora_publication=Mock())
        actor.checkpoint_engine_client = SimpleNamespace(backend=backend)
        actor.set_rollout_manager(manager)
        assert actor._managed_lora_publication is managed
        assert actor._weight_sync_lock is lock
        if managed:
            backend.configure_lora_publication.assert_called_once_with(config, manager)
        else:
            backend.configure_lora_publication.assert_not_called()
    manager.set_train_parallel_config.remote.assert_called_once()


@pytest.fixture
def resolve_rollout_quantization_config():
    def resolve(args, hf_config):
        actor = object.__new__(actor_module.MegatronTrainRayActor)
        actor.args = args
        actor.hf_config = hf_config
        return actor._resolve_push_quantization_config()

    return resolve


def _write_groups(args, tmp_path, models):
    pytest.importorskip("relax.distributed.ray.rollout")
    path = tmp_path / "rollout.yaml"
    path.write_text(json.dumps({"sglang": models}))
    args.sglang_config = str(path)
    args.rollout_num_gpus = sum(group["num_gpus"] for model in models for group in model["engine_groups"])


@pytest.mark.parametrize("method_source", ["metadata", "global", "group"])
def test_receiver_compression_metadata_drives_packing(
    resolve_rollout_quantization_config, quantization_args, compressed_checkpoint, tmp_path, method_source
):
    path, config = compressed_checkpoint
    quantization_args.sglang_hf_checkpoint = str(path)
    if method_source != "metadata":
        config = {key: value for key, value in config.items() if key != "quant_method"}
        (path / "config.json").write_text(json.dumps({"compression_config": config}))
        quantization_args.sglang_quantization = "compressed-tensors"
    if method_source == "group":
        quantization_args.sglang_quantization = "fp8"
        _write_groups(
            quantization_args,
            tmp_path,
            [
                {
                    "name": "policy",
                    "engine_groups": [
                        {"worker_type": "regular", "num_gpus": 2, "overrides": {"quantization": "compressed-tensors"}}
                    ],
                }
            ],
        )
    actual = resolve_rollout_quantization_config(
        quantization_args, SimpleNamespace(quantization_config={"quant_method": "fp8"})
    )
    assert actual["quant_method"] == "compressed-tensors"
    assert actual["format"] == "pack-quantized"
    assert "re:.*vision_tower.*" in actual["ignore"]
    assert config["ignore"] == []
    # Execute the real sender dispatch on a plain namespace excluded by the
    # checkpoint scan; this CPU case does not invoke the CUDA packing kernel.
    import torch

    from relax.backends.megatron.weight_conversion.processors import quantize_params

    named = [("vision_tower.proj.weight", torch.ones(1, 1))]
    assert quantize_params(quantization_args, "unused", named, actual) == named


def test_rollout_quantization_uses_receiver_checkpoint(
    resolve_rollout_quantization_config, quantization_args, tmp_path
):
    pytest.importorskip("transformers")
    receiver = tmp_path / "receiver"
    receiver.mkdir()
    expected = {"quant_method": "other"}
    (receiver / "config.json").write_text(json.dumps({"text_config": {"quantization_config": expected}}))
    quantization_args.sglang_hf_checkpoint = str(receiver)
    assert (
        resolve_rollout_quantization_config(
            quantization_args, SimpleNamespace(quantization_config={"quant_method": "fp8"})
        )
        == expected
    )


@pytest.mark.parametrize("order", ["single", "empty-first", "plain-first"])
def test_rollout_empty_quantization_preserves_bf16_dispatch(
    resolve_rollout_quantization_config, quantization_args, tmp_path, order
):
    from pathlib import Path

    import torch

    from relax.backends.megatron.weight_conversion.processors import quantize_params

    (Path(quantization_args.hf_checkpoint) / "config.json").write_text(json.dumps({"quantization_config": {}}))
    quantization_args.sglang_quantization = "compressed-tensors"
    groups = [{"worker_type": "regular", "num_gpus": 1}]
    if order != "single":
        groups.append(
            {
                "worker_type": "regular",
                "num_gpus": 1,
                "overrides": {"quantization": None, "json_model_override_args": '{"quantization_config": null}'},
            }
        )
    if order == "plain-first":
        groups.reverse()
    _write_groups(quantization_args, tmp_path, [{"name": "policy", "engine_groups": groups}])
    actual = resolve_rollout_quantization_config(quantization_args, SimpleNamespace())
    named = [("model.weight", torch.ones(1, 1))]
    assert quantize_params(quantization_args, "unused", named, actual) is named
    assert actual is None


@pytest.mark.parametrize(
    "override,expected",
    [
        ("{}", {"quant_method": "fp8"}),
        ('{"quantization_config": null}', None),
    ],
)
def test_rollout_quantization_group_json_replaces_global(
    resolve_rollout_quantization_config, quantization_args, tmp_path, override, expected
):
    args = quantization_args
    args.sglang_json_model_override_args = '{"quantization_config": {"quant_method": "other"}}'
    _write_groups(
        args,
        tmp_path,
        [
            {
                "name": "policy",
                "engine_groups": [
                    {"worker_type": "regular", "num_gpus": 2, "overrides": {"json_model_override_args": override}},
                ],
            }
        ],
    )
    assert resolve_rollout_quantization_config(args, SimpleNamespace()) == expected


@pytest.mark.parametrize("fully_async,hybrid", [(False, False), (True, False), (False, True)])
def test_rollout_quantization_checks_publication_recipients(
    resolve_rollout_quantization_config, quantization_args, tmp_path, fully_async, hybrid
):
    args = quantization_args
    args.fully_async, args.hybrid = fully_async, hybrid
    _write_groups(
        args,
        tmp_path,
        [
            {
                "name": "policy",
                "engine_groups": [
                    {"worker_type": "regular", "num_gpus": 1},
                    {
                        "worker_type": "placeholder",
                        "num_gpus": 1,
                        "overrides": {"json_model_override_args": '{"quantization_config": null}'},
                    },
                ],
            },
            {
                "name": "second",
                "engine_groups": [
                    {
                        "worker_type": "regular",
                        "num_gpus": 1,
                        "overrides": {"json_model_override_args": '{"quantization_config": null}'},
                    },
                ],
            },
        ],
    )
    if fully_async or hybrid:
        with pytest.raises(ValueError, match="policy/0:regular and second/0:regular"):
            resolve_rollout_quantization_config(args, SimpleNamespace())
    else:
        assert resolve_rollout_quantization_config(args, SimpleNamespace()) == {"quant_method": "fp8"}


@pytest.mark.parametrize("worker_types", [("regular", "regular"), ("prefill", "decode")])
def test_rollout_quantization_rejects_mixed_group_formats(
    resolve_rollout_quantization_config, quantization_args, tmp_path, worker_types
):
    args = quantization_args
    _write_groups(
        args,
        tmp_path,
        [
            {
                "name": "policy",
                "engine_groups": [
                    {"worker_type": worker_types[0], "num_gpus": 1},
                    {
                        "worker_type": worker_types[1],
                        "num_gpus": 1,
                        "overrides": {"json_model_override_args": '{"quantization_config": null}'},
                    },
                ],
            }
        ],
    )
    with pytest.raises(ValueError, match="different quantization configs"):
        resolve_rollout_quantization_config(args, SimpleNamespace())


def test_sender_still_rejects_unsupported_serialized_layout(resolve_rollout_quantization_config, quantization_args):
    from pathlib import Path

    import torch

    from relax.backends.megatron.weight_conversion.processors import quantize_params

    unsupported = {"quant_method": "modelopt", "quant_algo": "NVFP4"}
    (Path(quantization_args.hf_checkpoint) / "config.json").write_text(
        json.dumps({"quantization_config": unsupported})
    )
    quantization_args.sglang_quantization = "modelopt_fp4"
    actual = resolve_rollout_quantization_config(quantization_args, SimpleNamespace())
    with pytest.raises(ValueError, match="Unsupported quantization method"):
        quantize_params(quantization_args, "unused", [("model.weight", torch.ones(1, 1))], actual)
