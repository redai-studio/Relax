# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
from types import SimpleNamespace

import pytest
import torch

from relax.backends.megatron.weight_update.shard_p2p_source import plan_expert_views
from relax.backends.sglang.shard_p2p_weight_loader import apply_command
from relax.distributed.checkpoint_service.backends.shard_p2p.plan import fingerprint


@pytest.fixture
def transfer(monkeypatch):
    from relax.distributed.checkpoint_service.backends.shard_p2p import ipc

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(ipc, "import_tensor", lambda payload, device: payload)
    geometry = dict(num_experts=8, hidden_size=4, intermediate_size=4, tp_size=2, num_layers=1, source_world_size=8)
    prefix = "model.layers.0.mlp.experts."
    parameters = {
        prefix + "w13_weight": torch.empty(8, 4, 4, dtype=torch.bfloat16),
        prefix + "w2_weight": torch.empty(8, 4, 2, dtype=torch.bfloat16),
    }
    writes, bindings = [], []
    for rank in range(8):
        flat = torch.empty(48, dtype=torch.bfloat16)
        for projection in range(3):
            flat[projection * 16 : (projection + 1) * 16].fill_(rank * 3 + projection)
        canonical = {
            prefix + f"{rank}.gate_proj.weight": flat[:16].view(4, 4),
            prefix + f"{rank}.up_proj.weight": flat[16:32].view(4, 4),
            prefix + f"{rank}.down_proj.weight": flat[32:].view(4, 4),
        }
        sources, plans = plan_expert_views(
            canonical, num_experts=8, hidden_size=4, intermediate_size=4, tp_size=2, source_rank=rank
        )
        writes.extend(plans[0])
        bindings.extend(dict(source_rank=rank, source=key, payload=value) for key, value in sources.items())
        # Independent native-reference layout, not produced by the planner.
        parameters[prefix + "w13_weight"][rank, :2].fill_(rank * 3)
        parameters[prefix + "w13_weight"][rank, 2:].fill_(rank * 3 + 1)
        parameters[prefix + "w2_weight"][rank].fill_(rank * 3 + 2)
    model = SimpleNamespace(named_parameters=lambda: parameters.items())
    expected = {name: value.clone() for name, value in parameters.items()}
    seed = dict(
        op="seed",
        version=7,
        session="session",
        engine=0,
        tp_rank=0,
        nonce="seed",
        geometry=geometry,
        writes=writes,
        plan_hash=fingerprint(dict(geometry=geometry, writes=writes)),
        audit_native=True,
    )
    update = dict(
        op="apply",
        version=7,
        session="session",
        engine=0,
        tp_rank=0,
        nonce="update7",
        plan_hash=seed["plan_hash"],
        bindings=bindings,
    )
    return model, parameters, expected, seed, update


def _endpoint(rank, *, remote=False):
    return dict(
        boot_id="00000000-0000-0000-0000-00000000000" + str(int(remote)),
        hostname="same-container-name",
        pid=100 + rank,
        gpu_uuid=f"GPU-{rank}",
        device_index=0,
        generation=f"worker-{rank}",
    )


@pytest.fixture
def delivery(transfer, monkeypatch):
    from relax.backends.sglang import shard_p2p_weight_loader as loader

    model, parameters, expected, seed, update = transfer
    identity = _endpoint(20)
    monkeypatch.setattr(loader, "worker_identity", lambda generation, device: identity.copy())
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 1)
    monkeypatch.setattr(
        torch.distributed, "all_gather_object", lambda output, value, group: output.__setitem__(0, value)
    )
    probe = dict(
        seed,
        op="prepare",
        sources=[dict(rank=rank, endpoint=_endpoint(rank)) for rank in range(8)],
    )
    args = SimpleNamespace(
        tp_size=2,
        pp_size=1,
        dp_size=1,
        ep_size=1,
        quantization=None,
        enable_dp_attention=False,
        enable_lora=False,
        speculative_algorithm=None,
    )

    def control(command):
        loader._control_tp(model, [(json.dumps(command), torch.empty(0))], args, 0, "cpu-control")

    return probe, control


def test_physical_expert_prepare_seed_and_warm_audit(transfer, delivery):
    model, parameters, expected, seed, update = transfer
    probe, control = delivery
    control(probe)
    assert not hasattr(model, "_relax_expert_ipc")
    assert all(torch.equal(parameters[name], value) for name, value in expected.items())
    feedback = json.loads(json.dumps(model._relax_expert_delivery))
    reply = json.loads(model._relax_weight_sync_reply)["shard_p2p_receivers"][0]
    assert reply["identity"] == feedback["identity"] and reply["plan_hash"] == feedback["plan_hash"]
    binding = {key: value for key, value in feedback.items() if key != "identity"}
    seed["delivery"] = binding
    control(seed)
    assert not hasattr(model, "_relax_bind_delivery")
    assert all(torch.isnan(value).all() for value in parameters.values())
    update["delivery_hash"] = binding["plan_hash"]
    apply_command(model, update, 0)
    assert all(torch.equal(parameters[name], value) for name, value in expected.items())
    legacy = {key: value for key, value in seed.items() if key != "delivery"}
    legacy.update(version=8, previous_version=7, nonce="legacy-reseed")
    state = model._relax_expert_ipc
    with pytest.raises(ValueError, match="cannot be omitted"):
        apply_command(model, legacy, 0)
    assert model._relax_expert_ipc is state and not state["failed"]
    assert all(torch.equal(parameters[name], value) for name, value in expected.items())
    # The warm identity is private and remains bound to this incarnation.
    seed["delivery"]["delivery"]["receiver"]["pid"] = 999
    assert model._relax_expert_ipc["delivery"]["receiver"]["pid"] == 120
    update.update(version=8, nonce="next-version", delivery_hash="another-worker")
    with pytest.raises(RuntimeError, match="misrouted"):
        apply_command(model, update, 0)
    assert all(torch.equal(parameters[name], value) for name, value in expected.items())


@pytest.mark.parametrize("fault", ["remote", "missing_owner", "duplicate_owner", "logical_hash"])
def test_bind_delivery_failure_does_not_poison_native_seed(transfer, delivery, fault):
    model, parameters, expected, seed, _ = transfer
    probe, control = delivery
    if fault == "remote":
        probe["sources"][0]["endpoint"] = _endpoint(0, remote=True)
    elif fault == "missing_owner":
        probe["sources"].pop()
    elif fault == "duplicate_owner":
        probe["sources"].append(probe["sources"][0])
    else:
        probe["plan_hash"] = "wrong-plan"
    with pytest.raises(RuntimeError, match="receiver failed"):
        control(probe)
    assert not hasattr(model, "_relax_weight_sync_reply")
    assert not hasattr(model, "_relax_expert_ipc")
    assert all(torch.equal(parameters[name], value) for name, value in expected.items())
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("field", ["nonce", "engine", "receiver", "routes", "missing"])
def test_expert_seed_rejects_unprepared_receiver_binding(transfer, delivery, field):
    model, parameters, expected, seed, _ = transfer
    probe, control = delivery
    control(probe)
    feedback = json.loads(json.dumps(model._relax_expert_delivery))
    seed["delivery"] = {key: value for key, value in feedback.items() if key != "identity"}
    if field in ("nonce", "engine"):
        seed[field] = "different-seed" if field == "nonce" else 1
    elif field == "receiver":
        seed["delivery"]["delivery"]["receiver"]["pid"] += 1
    elif field == "missing":
        seed.pop("delivery")
    else:
        seed["delivery"]["delivery"]["routes"]["cuda_ipc"].pop()
    with pytest.raises(RuntimeError, match="not prepared|Missing or misrouted"):
        control(seed)
    assert not hasattr(model, "_relax_expert_ipc")
    assert all(torch.equal(parameters[name], value) for name, value in expected.items())


@pytest.mark.parametrize("invalid", ["tp_rank", "session", "engine", "version", "plan_hash", "nonce"])
def test_stale_or_misrouted_update_rejected_before_writes(transfer, invalid):
    model, parameters, _, seed, update = transfer
    apply_command(model, seed, 0)
    if invalid in ("tp_rank", "engine", "version"):
        update[invalid] += 1
    else:
        update[invalid] = "seed" if invalid == "nonce" else "wrong"
    with pytest.raises((ValueError, RuntimeError)):
        apply_command(model, update, 0)
    assert all(torch.isnan(parameter).all() for parameter in parameters.values())


@pytest.mark.parametrize("invalid", ["duplicate_binding", "audit_mismatch"])
def test_failed_transfer_cannot_publish_or_accept_later_updates(transfer, invalid):
    model, _, _, seed, update = transfer
    apply_command(model, seed, 0)
    if invalid == "duplicate_binding":
        update["bindings"].append(update["bindings"][0])
    else:
        update["bindings"][-1]["payload"].add_(1)
    with pytest.raises((ValueError, RuntimeError)):
        apply_command(model, update, 0)
    assert model._relax_expert_ipc["failed"]
    assert model._relax_expert_ipc["version"] == 6
    update.update(version=8, nonce="update8")
    with pytest.raises(RuntimeError, match="failed"):
        apply_command(model, update, 0)


@pytest.mark.parametrize("new_receiver", [False, True])
@pytest.mark.parametrize("mode", ["expert", "ordinary", "mixed"])
def test_completed_session_can_reseed_after_full_native_recovery(transfer, new_receiver, mode):
    model, parameters, expected, seed, update = transfer
    if mode != "expert":
        if mode == "ordinary":
            parameters.clear()
            expected.clear()
            seed["writes"].clear()
            update["bindings"].clear()
        value = torch.arange(12, dtype=torch.bfloat16).view(3, 4)
        parameters["projection"], expected["projection"] = value.clone(), value.clone()
        seed["geometry"].update(expert_enabled=mode == "mixed", ordinary_targets={"projection": [3, 4]})
        seed["writes"].append(
            dict(
                id="ordinary",
                target="tp-0",
                source_rank=0,
                source="ordinary",
                param="projection",
                src_offset=0,
                dst_offset=0,
                src_stride=24,
                dst_stride=24,
                rows=1,
                row_bytes=24,
            )
        )
        update["bindings"].append(dict(source_rank=0, source="ordinary", payload=value.view(-1)))
        seed["plan_hash"] = update["plan_hash"] = fingerprint(dict(geometry=seed["geometry"], writes=seed["writes"]))
    apply_command(model, seed, 0)
    apply_command(model, update, 0)
    if new_receiver:
        parameters = {name: value.clone() for name, value in parameters.items()}
        model = SimpleNamespace(
            named_parameters=lambda: parameters.items(),
            **({"get_submodule": model.get_submodule} if hasattr(model, "get_submodule") else {}),
        )
    # Model the completed full native update before direct transfer resumes.
    for name, parameter in parameters.items():
        parameter.copy_(expected[name] + 1)
    for binding in update["bindings"]:
        binding["payload"] = binding["payload"].clone() + 1
    seed.update(previous_version=7, version=8, nonce="reseed8")
    apply_command(model, seed, 0)
    assert all(torch.isnan(parameter).all() for parameter in parameters.values())
    update.update(version=8, nonce="recovery8")
    apply_command(model, update, 0)
    for name, parameter in parameters.items():
        torch.testing.assert_close(parameter, expected[name] + 1, rtol=0, atol=0)
    assert model._relax_expert_ipc["version"] == 8
    assert "reference" not in model._relax_expert_ipc


def _tp_control_error_worker(rank, rendezvous, output):
    from datetime import timedelta
    from pathlib import Path
    from unittest.mock import patch

    import torch.distributed as dist

    from relax.backends.sglang.shard_p2p_weight_loader import _control_tp
    from relax.distributed.checkpoint_service.backends.shard_p2p.lifecycle import NativeWeightLease

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=20))
    args = SimpleNamespace(
        tp_size=2,
        pp_size=1,
        dp_size=1,
        ep_size=1,
        quantization=None,
        enable_dp_attention=False,
        enable_lora=False,
        speculative_algorithm=None,
    )
    try:
        for fault in ("decode", "reply", "mode", "copy", "header"):
            model = torch.nn.Linear(2, 2, bias=False)
            original = model.weight.detach().clone()
            model._relax_expert_ipc = dict(geometry=dict(tp_size=2), tp={}, version=6, failed=False)
            model._relax_native_weight_lease = NativeWeightLease()
            command = dict(op="apply", session="s", engine=0, tp_rank=rank, version=7, nonce="n")
            if fault in ("decode", "reply"):
                command.update(op="prepare", geometry=dict(tp_size=2))
            elif fault == "mode":
                command.update(op="seed", geometry=dict(tp_size=2), delivery={}, batched=True, cache_plan=rank == 0)
            elif fault == "header" and rank == 1:
                command["version"] += 1
            payload = "{" if fault == "decode" and rank == 1 else json.dumps(command)
            receipt = (
                {}
                if fault == "reply" and rank == 1
                else dict(identity={}, delivery={"receiver": {}}, logical_plan_hash="l", plan_hash="p")
            )
            applied = []

            def apply(*args):
                applied.append(True)
                if rank == 1:
                    raise RuntimeError("copy or fence failed")

            with (
                patch("relax.backends.sglang.shard_p2p_weight_loader._prepare_delivery", return_value=receipt),
                patch("relax.backends.sglang.shard_p2p_weight_loader._accept_delivery"),
                patch("relax.backends.sglang.shard_p2p_weight_loader.apply_command", side_effect=apply),
            ):
                with pytest.raises(RuntimeError, match="receiver failed"):
                    _control_tp(model, [(payload, torch.empty(0))], args, rank, dist.group.WORLD)
            assert model._relax_expert_ipc["failed"] and model._relax_native_weight_lease.failed
            assert model._relax_expert_ipc["version"] == 6 and torch.equal(model.weight, original)
            if fault != "copy":
                assert not applied
            marker = torch.tensor(1)
            dist.all_reduce(marker, group=dist.group.WORLD)
            assert marker.item() == 2 and not torch.cuda.is_initialized()
        Path(output, f"{rank}.json").write_text("passed")
    finally:
        dist.destroy_process_group()


def test_tp_control_malformed_peer_fails_without_stranding_other_ranks(tmp_path):
    import torch.multiprocessing as mp

    mp.spawn(_tp_control_error_worker, args=((tmp_path / "rdzv").as_uri(), str(tmp_path)), nprocs=2, join=True)
    assert all((tmp_path / f"{rank}.json").read_text() == "passed" for rank in range(2))


@pytest.mark.parametrize("fail", [False, True])
def test_default_sync_writes_uninitialized_targets_without_native_audit(transfer, monkeypatch, fail):
    from relax.backends.sglang import shard_p2p_weight_loader as loader

    model, parameters, expected, seed, update = transfer
    seed.pop("audit_native")
    monkeypatch.setattr(loader, "_digests", lambda _: pytest.fail("Disabled audit must not hash weights"))
    for parameter in parameters.values():
        parameter.fill_(float("nan"))
    apply_command(model, seed, 0)
    assert model._relax_expert_ipc["seed_pending"]
    assert model._relax_expert_ipc["reference"] is None
    with pytest.raises(RuntimeError, match="already bound"):
        apply_command(model, seed, 0)
    if fail:

        def fail_fence():
            raise RuntimeError("injected fence failure")

        monkeypatch.setattr(torch.cuda, "synchronize", fail_fence)
        with pytest.raises(RuntimeError, match="injected fence failure"):
            apply_command(model, update, 0)
        assert model._relax_expert_ipc["seed_pending"] and model._relax_expert_ipc["failed"]
        assert model._relax_expert_ipc["version"] == 6
        with pytest.raises(RuntimeError, match="already bound"):
            apply_command(model, seed, 0)
        return
    apply_command(model, update, 0)
    assert not model._relax_expert_ipc["seed_pending"]
    assert model._relax_expert_ipc["version"] == update["version"]
    assert all(torch.equal(parameters[name], value) for name, value in expected.items())


def test_compiled_target_alias_failure_locks_session_before_import(transfer, monkeypatch):
    from unittest.mock import Mock

    from relax.distributed.checkpoint_service.backends.shard_p2p import ipc

    model, parameters, _, seed, update = transfer
    seed.update(batched=True, cache_plan=True)
    apply_command(model, seed, 0)
    prefix = "model.layers.0.mlp.experts."
    parameters[prefix + "w2_weight"] = parameters[prefix + "w13_weight"].as_strided(
        (8, 4, 2), (8, 2, 1), storage_offset=1
    )
    importer = Mock(side_effect=AssertionError("must reject before import"))
    monkeypatch.setattr(ipc, "import_tensor", importer)
    with pytest.raises(ValueError, match="physically overlap"):
        apply_command(model, update, 0)
    assert model._relax_expert_ipc["failed"] and model._relax_expert_ipc["version"] == 6
    assert model._relax_expert_ipc["compiled_plan"].failed
    with pytest.raises(RuntimeError, match="failed"):
        apply_command(model, update, 0)
    importer.assert_not_called()


@pytest.mark.parametrize("fault", [None, "stale", "hash", "missing", "duplicate", "failure"])
def test_prepare_rpc_response_binds_actual_receiver_without_files(transfer, delivery, monkeypatch, fault):
    from relax.distributed.checkpoint_service.backends.shard_p2p.plan import TransferPlan
    from relax.distributed.checkpoint_service.backends.shard_p2p.synchronizer import ShardP2PSynchronizer

    model, parameters, expected, seed, _ = transfer
    probe, control = delivery
    helper = object.__new__(ShardP2PSynchronizer)
    helper.rank, helper.geometry = 0, seed["geometry"]
    helper._phase = lambda operation: operation()
    helper.group, helper._actor_endpoints = "actor-group", []
    helper._plan_hashes = [seed["plan_hash"]]
    helper.validate_devices = lambda actors, receivers: None
    monkeypatch.setattr(torch.distributed, "broadcast_object_list", lambda values, **kwargs: None)
    endpoints = [_endpoint(rank) for rank in range(8)]
    builder = TransferPlan("tp-0", 8, tuple(seed["writes"])).bind_sources(dict(enumerate(endpoints)))

    def respond(commands):
        control(commands[0][0])
        message = json.loads(model._relax_weight_sync_reply)
        replies = message["shard_p2p_receivers"]
        if fault == "stale":
            replies[0]["identity"]["nonce"] = "previous-session"
        elif fault == "hash":
            replies[0]["plan_hash"] = "wrong"
        elif fault == "missing":
            replies.clear()
        elif fault == "duplicate":
            replies.append(replies[0])
        return [dict(success=fault != "failure", message=json.dumps(message))]

    helper._fanout = respond
    commands = [[seed.copy()]]
    if fault:
        with pytest.raises(ValueError):
            helper._negotiate_delivery(commands, endpoints, [builder])
    else:
        helper._negotiate_delivery(commands, endpoints, [builder])
        assert commands[0][0]["delivery"] == {k: v for k, v in model._relax_expert_delivery.items() if k != "identity"}
        assert helper._receiver_plan_hashes == (commands[0][0]["delivery"]["plan_hash"],)
    assert all(torch.equal(parameters[name], value) for name, value in expected.items())
