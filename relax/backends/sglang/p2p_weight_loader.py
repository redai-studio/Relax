# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Remote expert shard receiver through SGLang's native custom loader API."""

import base64
import os
import socket
import time
import uuid
from pathlib import Path

import torch

from relax.distributed.checkpoint_service.backends.shard_p2p.remote.common import (
    atomic_json,
    create_agent,
    digest,
    layout_of,
)


# Only fields needed to reconstruct this supported model's native weight layout.
_REPLICA_FIELDS = (
    "model_path",
    "trust_remote_code",
    "revision",
    "dtype",
    "tp_size",
    "pp_size",
    "dp_size",
    "ep_size",
    "quantization",
    "enable_dp_attention",
    "enable_dp_lm_head",
    "json_model_override_args",
    "model_impl",
    "model_config_parser",
    "disable_shared_experts_fusion",
    "moe_runner_backend",
    "moe_a2a_backend",
)


def control(model: torch.nn.Module, weights: list[tuple[str, torch.Tensor]]) -> None:
    """Apply the remote shard protocol through the actual native worker
    guard."""
    from relax.distributed.checkpoint_service.backends.shard_p2p.remote.common import REMOTE_SHARD_LOADER

    _run_control(model, weights, REMOTE_SHARD_LOADER)


def _run_control(model: torch.nn.Module, weights: list, loader: str) -> None:
    from sglang.srt.runtime_context import get_parallel, get_server_args

    from .p2p_weight_lease import run_control

    def apply(command):
        return _apply_control(model, command, get_server_args(), get_parallel())

    run_control(model, weights, loader, apply)


def _apply_control(model: torch.nn.Module, command: dict, args, parallel):
    if command["op"] == "capability":
        from relax.distributed.checkpoint_service.backends.shard_p2p.remote.common import NATIVE_LEASE_PROTOCOL

        from .weight_sync_guard import native_weight_lease

        lease = native_weight_lease(model)
        manager = model._relax_native_weight_guard.manager
        guarded = manager.update_weights_from_tensor
        sessions = tuple(
            callable(getattr(manager, name, None)) for name in ("begin_weight_update", "end_weight_update")
        )
        if (
            lease.failed
            or lease.active is not None
            or lease.pending
            or not getattr(guarded, "_relax_remote_weight_guard", False)
            or sessions[0] != sessions[1]
        ):
            raise NotImplementedError("Actual scheduler does not support the remote allocation protocol")
        directory = Path(command["directory"])
        if not directory.is_dir() or directory.stat().st_mode & 0o077:
            raise ValueError("Capability challenge requires a private metadata directory")
        from .shard_p2p_reload import receiver_capability

        capability = receiver_capability(model, args, parallel.tp_rank)
        capability["checks"]["no_draft_worker"] = manager.draft_worker is None
        atomic_json(
            directory / f"{parallel.tp_rank}.json",
            dict(
                engine=command["engine"],
                challenge=command["challenge"],
                tp_rank=parallel.tp_rank,
                tp_size=args.tp_size,
                pid=os.getpid(),
                hostname=socket.gethostname(),
                supported=all(capability["checks"].values()),
                capability=capability,
                weight_update_session=sessions[0],
                native_weight_lease_protocol=NATIVE_LEASE_PROTOCOL,
            ),
        )
        return
    if any(field in command for field in ("ipc", "ipc_generations", "ipc_plan_hashes")):
        raise ValueError("Retired IPC weight-transfer protocol")
    if any(
        field in command
        for field in (
            "tp_plan_envelopes",
            "tp_source_receipts",
            "vocab_plan_envelopes",
            "vocab_source_receipts",
        )
    ):
        raise RuntimeError("Unsupported TP or vocabulary protocol payload")
    scoped_fields = {
        "collective_groups": ("prepare",),
        "collective_peers": ("expert_allgather",),
        "collective_bootstraps": ("expert_allgather",),
        "fallback_receipt_keys": ("prepare",),
        "fallback_receipts": ("commit",),
        "plan_envelopes": ("plan",),
        "source_receipts": ("commit", "expert_allgather"),
    }
    if any(field in command and command.get("op") not in operations for field, operations in scoped_fields.items()):
        raise RuntimeError("Unexpected P2P protocol metadata for this operation")
    from relax.utils.env import Envs

    if (
        type(args.tp_size) is not int
        or args.tp_size <= 0
        or args.pp_size != 1
        or args.dp_size != 1
        or args.ep_size != 1
        or args.quantization is not None
        or args.enable_dp_attention
        or getattr(args, "enable_memory_saver", None) is not False
        or args.enable_lora
        or args.lora_paths
        or args.speculative_algorithm is not None
    ):
        raise NotImplementedError(
            "P2P receiver requires unquantized PP1/DP1/EP1, without LoRA, speculation or DP Attention; TP other than 2 requires expert shard transfer"
        )
    profile, mark = {}, time.monotonic()

    def checkpoint(name: str) -> None:
        nonlocal mark
        now = time.monotonic()
        profile[name] = now - mark
        mark = now

    directory = Path(command["directory"])
    if not directory.is_dir() or directory.stat().st_mode & 0o077:
        raise ValueError("P2P manifest directory must exist with mode 0700")
    key = f"{command['engine']}-{parallel.tp_rank}"
    if command["op"] == "prepare":
        from relax.distributed.checkpoint_service.backends.shard_p2p.remote.common import NATIVE_LEASE_PROTOCOL

        from .p2p_weight_lease import P2PWeightLease

        manager = model._relax_native_weight_guard.manager
        native_session = tuple(
            callable(getattr(manager, name, None)) for name in ("begin_weight_update", "end_weight_update")
        )
        if (
            native_session[0] != native_session[1]
            or command.get("native_weight_lease_protocol") != NATIVE_LEASE_PROTOCOL
            or type(command.get("weight_update_session")) is not bool
            or command["weight_update_session"] != native_session[0]
        ):
            raise NotImplementedError("P2P weight-update capability differs from the actual scheduler")
        state = getattr(model, "_relax_p2p_state", None)
        if state is not None and (state["receipt_tp_size"] != args.tp_size or state["key"] != key):
            raise RuntimeError("Receiver TP identity changed; recreate the P2P session")
        from sglang.srt.layers.layernorm import GemmaRMSNorm

        from relax.distributed.checkpoint_service.backends.shard_p2p.remote.fallback_receipts import owner_keys
        from relax.distributed.checkpoint_service.backends.shard_p2p.remote.layout import (
            BLOCK_CAPABILITY,
            validate_native_model,
        )

        fallback_keys = owner_keys(command.get("fallback_receipt_keys"), command["engine"], args.tp_size)
        fallback_buffers = sorted(
            name + ".gemma_weight" for name, module in model.named_modules() if isinstance(module, GemmaRMSNorm)
        )
        if state is not None and (
            state.get("fallback_receipt_keys") != fallback_keys
            or state.get("fallback_loader_buffers") != fallback_buffers
        ):
            raise RuntimeError("Native fallback schema changed; recreate the remote shard session")
        collective_allgather = bool(command.get("collective_groups"))
        if not collective_allgather and "collective_groups" in command:
            raise RuntimeError("Unexpected collective receiver groups")
        if state is not None and state.get("collective_allgather", False) != collective_allgather:
            raise RuntimeError("Collective mode changed; recreate the remote shard session")
        if state is not None and state.get("collective_failed"):
            raise RuntimeError("A failed collective session cannot be reused")
        native_experts = validate_native_model(model, args, parallel.tp_rank)
        if state is not None and state.get("block_native_layouts") != native_experts:
            raise RuntimeError("Native expert schema changed; recreate the remote shard session")
        compact_manifest = state is not None and not command["audit"]
        if state is not None and state["active"]:
            raise RuntimeError("A P2P update is already active")
        if state is not None and (
            state.get("completed_version") != state["version"]
            or command["version"] != state["version"] + 1
            or Path(state["directory"]).parent != directory.parent
        ):
            raise RuntimeError("P2P prepare requires the preceding completed allocation version")
        params = dict(model.named_parameters())
        layouts = {name: layout_of(param) for name, param in params.items()}
        generation = state["generation"] if state is not None else str(uuid.uuid4())
        native_reload = model._relax_native_weight_guard.native_reload
        if native_reload is None or native_reload.model is not model:
            raise RuntimeError("Native receiver has no current reload capability")
        reload_epoch = native_reload.capture_epoch()
        checkpoint("qualification_layout_s")
        transaction = P2PWeightLease(model, command, generation, params, layouts)
        checkpoint("native_lease_s")
        if state is None:
            from relax.distributed.checkpoint_service.backends.shard_p2p.remote.transport import RegisteredMemory

            agent = create_agent()
            checkpoint("nixl_agent_s")
            registration = RegisteredMemory(agent, params)
            registration.register(batched=True)
            checkpoint("nixl_registration_s")
            state = dict(
                agent=agent,
                registrations=registration.registrations,
                registration=registration,
                native_reload=native_reload,
                params=params,
                layouts=layouts,
                active=False,
                generation=generation,
                block_native_layouts=native_experts,
                fallback_receipt_keys=fallback_keys,
                fallback_loader_buffers=fallback_buffers,
                collective_allgather=collective_allgather,
                receipt_tp_size=args.tp_size,
                retain_source_receipts=bool(Envs.RELAX_P2P_AUDIT_DIR),
            )
            model._relax_p2p_state = state
        elif layouts != state["layouts"]:
            raise RuntimeError("Rollout parameter storage changed; recreate the P2P session")
        else:
            state["registration"].validate(params, layouts=layouts)
        state.update(
            active=True,
            version=command["version"],
            directory=str(directory),
            key=key,
            audit=command["audit"],
            direct_params=set(),
            plan=None,
            receipt_delivery="file",
            plan_delivery="file",
            received_plan_envelope=None,
            collective_completion=None,
            reload_buffer_epoch=reload_epoch,
            native_weight_lease_protocol=NATIVE_LEASE_PROTOCOL,
            weight_update_session=command["weight_update_session"],
        )
        transaction.bind(state)
        state.setdefault("initial_version", command["version"])
        checkpoint("registration_validation_s")
        state["reference"] = {name: digest(param) for name, param in params.items()} if command["audit"] else {}
        state["reference_buffers"] = (
            {name: digest(buf) for name, buf in model.named_buffers() if name.endswith(".gemma_weight")}
            if command["audit"]
            else {}
        )
        checkpoint("reference_audit_s")
        if command["audit"]:
            # The initial native broadcast already matches the reference. Poison
            # it so an omitted first write cannot pass the same-snapshot audit.
            with torch.no_grad():
                for param in params.values():
                    param.fill_(float("nan"))
            torch.cuda.synchronize()
        checkpoint("poison_fence_s")
        manifest = dict(
            generation=state["generation"],
            version=state["version"],
            hostname=socket.gethostname(),
            pid=os.getpid(),
            tp_rank=parallel.tp_rank,
            gpu_id=torch.cuda.current_device(),
        )
        if compact_manifest:
            # The actual storage was checked above. Publish only the dynamic
            # receipt instead of rebuilding the same 664-parameter JSON graph.
            manifest["cached_layout"] = True
        else:
            manifest.update(
                native_weight_lease_protocol=NATIVE_LEASE_PROTOCOL,
                weight_update_session=command["weight_update_session"],
                agent_metadata=base64.b64encode(state["agent"].get_agent_metadata()).decode(),
                params=layouts,
                server_fields={name: getattr(args, name) for name in _REPLICA_FIELDS},
            )
            from relax.distributed.checkpoint_service.backends.shard_p2p.remote.fallback_receipts import (
                RPC_FALLBACK_DELIVERY,
            )
            from relax.distributed.checkpoint_service.backends.shard_p2p.remote.receiver import RPC_PLAN_DELIVERY

            manifest.update(
                block_moe_layout=BLOCK_CAPABILITY,
                rpc_plan_delivery=RPC_PLAN_DELIVERY,
                fallback_receipt_delivery=RPC_FALLBACK_DELIVERY,
                fallback_receipt_owner=fallback_keys.index(key),
                fallback_loader_buffers=fallback_buffers,
            )
        state.update({name: manifest[name] for name in ("hostname", "pid", "tp_rank", "gpu_id")})
        if collective_allgather:
            from relax.distributed.checkpoint_service.backends.shard_p2p.remote.expert_collective import configure

            bootstrap = configure(state, command.get("collective_groups"), key, parallel.tp_rank)
            if bootstrap is not None and not compact_manifest:
                manifest["collective_bootstrap"] = bootstrap
        checkpoint("manifest_collective_setup_s")
        if command["audit"]:
            manifest["cold_profile"] = profile
        atomic_json(directory / f"{key}.manifest.json", manifest)
        return
    if command["op"] not in ("plan", "commit", "expert_allgather"):
        raise ValueError(f"Unknown P2P operation {command['op']}")
    state = getattr(model, "_relax_p2p_state", None)
    if state is None or not state["active"] or state["version"] != command["version"]:
        raise RuntimeError("No matching active P2P session")
    if state["directory"] != str(directory) or state["key"] != key:
        raise RuntimeError("P2P session identity changed")
    if "fallback_receipts" in command and command["op"] != "commit":
        raise RuntimeError("Fallback completions are valid only for commit")
    if command["op"] == "expert_allgather":
        from relax.distributed.checkpoint_service.backends.shard_p2p.remote.expert_collective import execute

        if not state.get("collective_allgather") or state["plan"] is None:
            raise RuntimeError("Unexpected replica AllGather operation")
        execute(state, command, directory, key)
        return
    if command["op"] == "plan":
        from relax.distributed.checkpoint_service.backends.shard_p2p.remote.plan import (
            ReceiptValidator,
            fingerprint,
            validate_coverage,
        )

        if state["plan"] is not None:
            raise RuntimeError("Unexpected or duplicate remote shard plan")
        from relax.distributed.checkpoint_service.backends.shard_p2p.remote.receiver import (
            RPC_PLAN_DELIVERY,
            read_plan_envelope,
        )

        envelope = read_plan_envelope(state, command, directory, key)
        if envelope["generation"] != state["generation"] or envelope["version"] != state["version"]:
            raise RuntimeError("Stale remote shard plan")
        if "cached_plan_hash" in envelope:
            if envelope["cached_plan_hash"] != state.get("validated_plan_hash") or "validated_plan" not in state:
                raise RuntimeError("Unknown cached remote shard plan")
            plan = state["validated_plan"]
        else:
            plan = envelope["plan"]
        from relax.distributed.checkpoint_service.backends.shard_p2p.remote.plan import (
            COLLECTIVE_DELIVERY,
            RPC_RECEIPT_DELIVERY,
        )

        receipt_delivery = plan.get("receipt_delivery")
        if receipt_delivery != RPC_RECEIPT_DELIVERY:
            raise RuntimeError("remote shard receipt delivery differs from the receiver session")
        collective = plan.get("delivery") == COLLECTIVE_DELIVERY
        if collective != state.get("collective_allgather", False):
            raise RuntimeError("remote shard collective delivery differs from the receiver session")
        if plan.get("delivery") not in (None, COLLECTIVE_DELIVERY) or plan.get("relay_consumers"):
            raise RuntimeError("Unsupported expert delivery protocol")
        if any(write["target"] != key for write in plan["writes"]):
            raise RuntimeError("remote shard plan targets another rank")
        if plan.get("source_catalog") is None:
            raise RuntimeError("Expert shard transfer requires an original expert source catalog")
        plan_hash = envelope.get("cached_plan_hash") or fingerprint(plan)
        if state.get("validated_plan_hash") != plan_hash:
            if collective:
                from relax.distributed.checkpoint_service.backends.shard_p2p.remote.plan import (
                    validate_collective_plan,
                )

                direct = validate_collective_plan(plan, state["layouts"], key, parallel.tp_rank)
                if direct != set(state["block_native_layouts"]):
                    raise RuntimeError("Collective plan must cover the complete guarded native expert set")
            else:
                direct = validate_coverage(
                    plan["writes"], state["layouts"], source_world_size=plan.get("source_world_size", 4)
                )
                from relax.distributed.checkpoint_service.backends.shard_p2p.remote.plan import validate_expert_catalog

                validate_expert_catalog(plan, key, state["layouts"], required=True)
            if not direct:
                raise RuntimeError("Empty remote shard coverage")
            state["validated_plan_hash"] = plan_hash
            state["validated_direct_params"] = direct
            state["validated_plan"] = plan
            state["receipt_validator"] = ReceiptValidator(plan)
        state["direct_params"] = state["validated_direct_params"]
        state["plan"] = plan
        if collective and state.get("collective_cached_plan_hash") != plan_hash:
            from relax.distributed.checkpoint_service.backends.shard_p2p.remote.expert_collective import cache_plan

            cache_plan(state)
            state["collective_cached_plan_hash"] = plan_hash
        if state["version"] != state["initial_version"]:
            state["plan_delivery"] = RPC_PLAN_DELIVERY
            state["received_plan_envelope"] = envelope
        return
    from relax.distributed.checkpoint_service.backends.shard_p2p.remote.fallback_receipts import (
        RPC_FALLBACK_DELIVERY,
        read_receipt,
    )

    receipt = read_receipt(state, command, directory, key)
    if receipt["generation"] != state["generation"] or receipt["version"] != state["version"]:
        raise RuntimeError("Stale P2P completion receipt")
    params = state["params"]
    if state["plan"] is None:
        raise RuntimeError("remote shard commit requires a validated plan")
    from relax.distributed.checkpoint_service.backends.shard_p2p.remote.plan import RPC_RECEIPT_DELIVERY
    from relax.distributed.checkpoint_service.backends.shard_p2p.remote.rpc_receipts import read_commit_receipts

    receipts = read_commit_receipts(state, command, directory, key)
    if state.get("collective_allgather"):
        from relax.distributed.checkpoint_service.backends.shard_p2p.remote.expert_collective import verify_commit

        verify_commit(state, receipts)
    if state["version"] != state["initial_version"]:
        state["receipt_delivery"] = RPC_RECEIPT_DELIVERY
        if state["retain_source_receipts"]:
            # Preserve what this receiver actually got over RPC, not a
            # completion reconstructed from its static routing plan.
            atomic_json(directory / f"{key}.rpc-receipts.json", command["source_receipts"][key])
            for rank, source_receipt in receipts.items():
                atomic_json(directory / f"{key}.source-{rank}.done.json", source_receipt)
    expected_fallback = set(params) - state["direct_params"]
    if set(receipt["params"]) != expected_fallback or len(receipt["params"]) != len(expected_fallback):
        raise RuntimeError("P2P completion does not cover every target parameter exactly once")
    if {name: layout_of(param) for name, param in params.items()} != state["layouts"]:
        raise RuntimeError("Parameter layout changed during P2P write")
    checkpoint("receipts_layout_s")
    torch.cuda.synchronize()
    checkpoint("transfer_fence_s")
    if state["audit"]:
        for name, param in params.items():
            actual = digest(param)
            if actual != state["reference"][name] or (
                name in expected_fallback and actual != receipt["digests"][name]
            ):
                raise RuntimeError(f"P2P/reference byte mismatch: {name}")
    checkpoint("parameter_audit_s")
    refreshed = state["native_reload"].refresh(set(receipt["loader_buffers"]), epoch=state["reload_buffer_epoch"])
    checkpoint("derived_postload_s")
    torch.cuda.synchronize()
    checkpoint("derived_fence_s")
    if state["audit"]:
        for name, buffer in model.named_buffers():
            if name in state["reference_buffers"] and digest(buffer) != state["reference_buffers"][name]:
                raise RuntimeError(f"P2P/reference derived buffer mismatch: {name}")
    checkpoint("derived_audit_s")
    return dict(
        cold_profile=profile if state["audit"] else None,
        version=state["version"],
        params=len(params),
        nbytes=sum(p.numel() * p.element_size() for p in params.values()),
        audit=state["audit"],
        derived_buffers=len(refreshed),
        success=True,
        direct_params=len(state["direct_params"]),
        direct_nbytes=sum(state["layouts"][name]["nbytes"] for name in state["direct_params"]),
        source_receipt_delivery=state["receipt_delivery"],
        source_receipts=(command["source_receipts"][key] if state["receipt_delivery"] != "file" else None),
        plan_delivery=state["plan_delivery"],
        received_plan_envelope=state["received_plan_envelope"],
        **({"collective_completion": state["collective_completion"]} if state.get("collective_allgather") else {}),
        fallback_receipt_delivery=("file" if state["version"] == state["initial_version"] else RPC_FALLBACK_DELIVERY),
        fallback_receipt=(None if state["version"] == state["initial_version"] else command["fallback_receipts"][key]),
    )
