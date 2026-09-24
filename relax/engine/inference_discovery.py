# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Read manager-owned, cached metadata without calling inference actors."""

from typing import Any

from relax.engine.inference import SnapshotVersion, actor_identity, base_url, model_state


def rollout_snapshot(manager: Any) -> dict[str, Any]:
    models: dict[str, Any] = {}
    identities = []
    state = getattr(manager, "_inference_state", "starting")
    if any(
        getattr(manager, flag, False)
        for flag in ("_training_weight_updating", "_scale_out_weight_updating", "_inference_weight_updating")
    ):
        state = "onloading"
    for name, server in tuple(manager.servers.items()):
        replicas, diagnostics = [], []
        ready_worker_types: set[str] = set()
        backend_model = None
        groups = tuple(server.engine_groups)
        pd = any(group.worker_type in ("prefill", "decode") for group in groups)
        for index, group in enumerate(groups):
            slots = tuple(group.all_engines)
            urls = dict(getattr(group, "inference_urls", {}))
            blocked = set(getattr(group, "inference_blocked", ()))
            group_state = state
            lifecycle = getattr(group.lifecycle_status, "value", group.lifecycle_status)
            if lifecycle != "ACTIVE":
                group_state = "draining"
            overrides = group.sglang_overrides
            backend_model = (
                backend_model
                or overrides.get("served_model_name")
                or getattr(manager.args, "sglang_served_model_name", None)
                or overrides.get("model_path")
            )
            workers = []
            for slot, engine in enumerate(slots):
                rank = group.rank_offset + slot
                identity = actor_identity(engine) if engine is not None else None
                identities.append((name, rank, identity))
                url = urls.get(rank) or getattr(manager, "_inference_external_urls", {}).get(identity)
                head = slot % group.nodes_per_engine == 0
                live = engine is not None and slot not in blocked
                worker = {"rank": rank, "status": "active" if live else "dead"}
                if live:
                    worker.update(getattr(group, "inference_metadata", {}).get(identity, {}))
                if head and url:
                    worker["url"] = url
                workers.append(worker)
                if not head or group.worker_type == "placeholder":
                    continue
                end = slot + group.nodes_per_engine
                complete = end <= len(slots) and all(
                    slots[i] is not None and i not in blocked for i in range(slot, end)
                )
                replica_state = group_state if live and complete else "dead"
                if replica_state == "ready" and identity in getattr(manager, "_inference_pending_weights", ()):
                    replica_state = "onloading"
                if replica_state == "ready":
                    ready_worker_types.add(group.worker_type)
                replicas.append(
                    {
                        "engine_id": f"{name}/{rank}",
                        "base_url": url,
                        "state": replica_state,
                        "direct_eligible": replica_state == "ready" and bool(url) and not pd,
                    }
                )
            diagnostics.append(
                {
                    "group_index": index,
                    "worker_type": group.worker_type,
                    "num_gpus_per_engine": group.num_gpus_per_engine,
                    "num_new_engines": group.num_new_engines,
                    "engines": workers,
                }
            )
        router = base_url(server.router_ip, server.router_port) if server.router_ip and server.router_port else None
        effective_state = model_state(replicas, state)
        # A shared router can still select a recovered head while its weights
        # are stale, so block this whole model until its transfer is confirmed.
        if effective_state == "ready":
            transitional = next(
                (engine["state"] for engine in replicas if engine["state"] in ("starting", "onloading", "draining")),
                None,
            )
            if transitional is not None:
                effective_state = transitional
        if effective_state == "ready" and pd and not {"prefill", "decode"}.issubset(ready_worker_types):
            effective_state = "unavailable"
        models[name] = {
            "state": effective_state,
            "router_url": router,
            "router_required": True,
            "backend_model": backend_model or getattr(manager.args, "hf_checkpoint", name),
            "engines": replicas,
            "engine_groups": diagnostics,
            "router_ip": server.router_ip,
            "router_port": server.router_port,
            "total_engines": sum(len(group["engines"]) for group in diagnostics),
        }
    return manager._inference_version.publish(models, identities)


def static_snapshot(manager: Any) -> dict[str, Any]:
    state = manager._inference_state
    name = getattr(manager.args, "_inference_model_name", "default")
    engines = []
    slots = tuple(manager.all_engines)
    identities = [(rank, actor_identity(engine) if engine is not None else None) for rank, engine in enumerate(slots)]
    addresses = dict(manager._engine_addr_and_ports)
    for rank in range(0, len(slots), manager.nodes_per_engine):
        address = addresses.get(rank)
        end = rank + manager.nodes_per_engine
        live = end <= len(slots) and all(
            slots[i] is not None and i not in manager._inference_blocked for i in range(rank, end)
        )
        replica_state = state if live else "dead"
        if replica_state == "ready" and rank in manager._inference_unhealthy:
            replica_state = "unavailable"
        url = base_url(address["host"], address["port"]) if address else None
        engines.append(
            {
                "engine_id": f"default/{rank}",
                "base_url": url,
                "state": replica_state,
                "direct_eligible": replica_state == "ready" and bool(url),
            }
        )
    teacher = hasattr(manager, "_overrides")
    overrides = manager._overrides if teacher else (getattr(manager.args, "genrm_engine_config", None) or {})
    engine_args = manager._teacher_args if teacher else manager.args
    backend_model = (
        overrides.get("served_model_name")
        or getattr(engine_args, "sglang_served_model_name", None)
        or overrides.get("model_path")
        or getattr(engine_args, "genrm_model_path", None)
        or getattr(engine_args, "hf_checkpoint", None)
    )
    return manager._inference_version.publish(
        {
            name: {
                "state": model_state(engines, state),
                "router_url": None,
                "backend_model": backend_model,
                "engines": engines,
            }
        },
        identities,
    )


def initialize_discovery(manager: Any) -> None:
    manager._inference_version = SnapshotVersion()
    manager._inference_state = "starting"
    manager._inference_blocked = set()
    manager._inference_unhealthy = set()
    manager._inference_weight_updating = False
    manager._inference_sync_serial = 0
    manager._inference_pending_weights = set()


def weight_update_notification(manager: Any, completed: dict[str, Any] | None = None) -> dict[str, Any]:
    """Keep failed transfers closed; confirm only the engines actually
    updated."""
    if completed is None:
        manager._inference_weight_updating = True
        manager._inference_sync_serial += 1
        urls = {}
        for server in manager.servers.values():
            for group in server.engine_groups:
                for slot in range(0, len(group.all_engines), group.nodes_per_engine):
                    engine = group.all_engines[slot]
                    if engine is not None:
                        identity = actor_identity(engine)
                        urls[identity] = group.inference_urls.get(group.rank_offset + slot) or getattr(
                            manager, "_inference_external_urls", {}
                        ).get(identity)
        return {
            "serial": manager._inference_sync_serial,
            "engines": [actor_identity(engine) for engine in manager.rollout_engines if engine is not None],
            "urls": urls,
        }
    if completed["serial"] != manager._inference_sync_serial:
        raise RuntimeError("Stale inference weight update completion")
    manager._inference_pending_weights.update(completed.get("unconfirmed_engines", ()))
    manager._inference_pending_weights.difference_update(completed["engines"])
    manager._inference_weight_updating = False
    return completed
