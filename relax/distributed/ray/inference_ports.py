# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Allocate ingress and distributed ports using actual worker locations."""

from typing import Any

import ray


def allocate_inference_ports(
    engines: list[tuple[int, Any]],
    *,
    nodes_per_engine: int,
    base_port: int,
    dp_size: int = 1,
    rank_offset: int = 0,
    worker_type: str = "regular",
    max_port: int = 65535,
) -> tuple[dict, dict]:
    cursors: dict[str, int] = {}
    addresses: dict[int, dict] = {}
    for rank, engine in sorted(engines):
        host, _ = ray.get(engine._get_current_node_ip_and_free_port.remote(), timeout=60)

        def port(count: int = 1) -> int:
            _, value = ray.get(
                engine._get_current_node_ip_and_free_port.remote(
                    start_port=cursors.get(host, base_port), consecutive=count, max_port=max_port
                ),
                timeout=60,
            )
            cursors[host] = value + count
            return value

        info = addresses.setdefault(rank, {})
        info.update(host=host, port=port(), nccl_port=port())
        if worker_type == "prefill":
            info["disaggregation_bootstrap_port"] = port()
        if (rank - rank_offset) % nodes_per_engine == 0:
            ingress = f"[{host}]" if ":" in host and not host.startswith("[") else host
            dist_addr = f"{ingress}:{port(30 + dp_size)}"
            for follower in range(rank, rank + nodes_per_engine):
                addresses.setdefault(follower, {})["dist_init_addr"] = dist_addr
    if any("dist_init_addr" not in addresses[rank] for rank, _ in engines):
        raise ValueError("Recovery must allocate every node of a logical inference replica")
    return addresses, cursors
