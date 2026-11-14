# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Build a version consumer inside the existing node-local SGLang Ray actor."""

import socket
from pathlib import Path

from .budget import NodeBudget
from .consumer import VersionConsumer
from .runtime import ConsumptionRunner
from .session import InstallSession
from .sglang_loader import SGLangHTTPGroup
from .store import VersionStore, digest, load_json
from .transport import SharedFSTransport


def create_actor_consumer(actor) -> ConsumptionRunner:
    config = load_json(Path(actor.args.weight_consumer_config).read_bytes())
    directory = Path(config["directory"]) / f"engine-{actor.rank}"
    node_directory = Path(config["directory"]) / ("node-" + digest(socket.gethostname().encode())[:16])
    options = actor._weight_version_server_args
    cache = VersionStore(
        node_directory / "cache",
        disk_budget=config["cache_budget"],
        max_weight_bytes=config["max_weight_bytes"],
        canonical_blocks=config.get("canonical_blocks", True),
    )
    transport_config = config["transport"]
    if transport_config["type"] == "sharedfs":
        source = VersionStore(
            transport_config["store"],
            disk_budget=transport_config["disk_budget"],
            max_weight_bytes=cache.max_weight_bytes,
        )
        if transport_config.get("repair_owner"):
            raise ValueError("Remote repair owners require the TCP delivery extension")
        transport = SharedFSTransport(source)
    else:
        raise ValueError("Transport must be sharedfs")
    group = SGLangHTTPGroup(
        options.url(),
        options.model_path,
        tp_size=options.tp_size,
        pp_size=options.pp_size,
        audit_directory=directory / "audit",
        api_key=options.api_key,
    )
    session = InstallSession(
        directory / "installation",
        group,
        node_install_lock=None if config.get("node_budget") else node_directory / ".installation",
    )
    budget_config = config.get("node_budget")
    budget = None
    if budget_config:
        budget = NodeBudget(node_directory / "resources", budget_config["limits"], budget_config["resident"])
    return ConsumptionRunner(
        VersionConsumer(
            transport,
            cache,
            node_directory / "checkpoints",
            checkpoint_budget=config["checkpoint_budget"],
            delivery_stage=("X" if socket.gethostname() == config["first_target_host"] else "T")
            if config.get("first_target_host")
            else config.get("delivery_stage", "X"),
        ),
        session,
        f"{config['consumer_prefix']}-{actor.rank}",
        config["stream"],
        start_version=config.get("start_version", 0),
        mode=config.get("mode", "all_versions"),
        operation_timeout=config.get("operation_timeout", 1800),
        node_budget=budget,
        operation_resources=budget_config["operation"] if budget_config else None,
    )
