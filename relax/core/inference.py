# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Deploy inference ingress independently of GPU-owning services."""

from typing import Any

from ray import serve

from relax.components.inference_gateway import InferenceGateway
from relax.core.node_group_affinity import with_control_plane_affinity


def deploy_inference_gateway(role: str, config: Any, runtime_env: dict | None = None) -> Any:
    options = with_control_plane_affinity(config, {"num_cpus": 1, "num_gpus": 0, "runtime_env": runtime_env or {}})
    application = InferenceGateway.options(ray_actor_options=options).bind(role)
    return serve.run(application, name=f"inference_{role}", route_prefix=f"/{role}")


def bind_inference_gateway(handle: Any, sources: list[dict[str, Any]], backend: Any = None) -> None:
    handle.rebind_sources.remote(sources, backend).result(timeout_s=30)


def quiesce_inference_gateway(handle: Any) -> None:
    if handle is not None:
        handle.quiesce.remote().result(timeout_s=10)
