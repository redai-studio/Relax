# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Bounded standalone SGLang checkpoint consumer; no training process is
needed."""

import math
import sys
import time
from pathlib import Path

from relax.utils.logging_utils import get_logger

from .budget import NodeBudget
from .consumer import VersionConsumer
from .runtime import ConsumptionRunner
from .session import InstallSession
from .sglang_loader import EmbeddedSGLangGroup
from .sglang_plugin import create_engine
from .store import VersionStore, canonical, load_json
from .transport import SharedFSTransport


logger = get_logger(__name__)


def consume_versions(cache: VersionStore, configuration: str | Path) -> None:
    config = load_json(Path(configuration).read_bytes())
    start, end = config["start_version"], config["end_version"]
    if type(start) is not int or type(end) is not int or not 0 <= start <= end:
        raise ValueError("Set a finite start_version/end_version range")
    wait_seconds = float(config.get("wait_seconds", 0))
    if not math.isfinite(wait_seconds) or wait_seconds < 0:
        raise ValueError("wait_seconds must be finite and nonnegative")
    model = Path(config["model_path"]).resolve()
    if not (model / "config.json").is_file():
        raise ValueError("model_path must be a prepared local engine startup checkpoint")
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
    consumer = VersionConsumer(
        transport,
        cache,
        config["checkpoint_root"],
        checkpoint_budget=config["checkpoint_budget"],
        delivery_stage=config.get("delivery_stage", "X"),
    )
    options = dict(config.get("engine", {}))
    options.update(model_path=str(model), tp_size=config.get("tp_size", 1), load_format="safetensors")
    node_directory = Path(config.get("node_directory", cache.root.parent))
    budget_config = config.get("node_budget")
    budget = None
    if budget_config:
        budget = NodeBudget(node_directory / "resources", budget_config["limits"], budget_config["resident"])
    engine = create_engine(**options)
    try:
        group = EmbeddedSGLangGroup(
            engine, tp_size=options["tp_size"], audit_directory=Path(config["session_directory"]) / "audit"
        )
        session = InstallSession(
            config["session_directory"], group, node_install_lock=None if budget else node_directory / ".installation"
        )
        runner = ConsumptionRunner(
            consumer,
            session,
            config["consumer"],
            config["stream"],
            start_version=start,
            mode=config.get("mode", "all_versions"),
            operation_timeout=config.get("operation_timeout", 1800),
            node_budget=budget,
            operation_resources=budget_config["operation"] if budget_config else None,
        )
        cursor = runner.transport.consumer_status(runner.name)["cursor"]
        # A fresh engine must restore its previous ACTIVE version before taking new work.
        next_version = max(start, cursor)
        targets = [end] if runner.mode == "latest" else range(next_version, end + 1)
        for version in targets:
            deadline = time.monotonic() + wait_seconds
            next_notice = 0.0
            while True:
                available = transport.versions(runner.stream, after=version - 1, limit=1)
                if available and available[0]["version"] == version:
                    break
                if available:
                    raise ValueError(
                        f"Version {version} was collected; retained history starts at {available[0]['version']}"
                    )
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Version {version} is not committed; rerun to resume the same subscription")
                if time.monotonic() >= next_notice:
                    logger.info(
                        "Waiting for stream %s version %s; %.1fs remain. Check the publisher if no version appears.",
                        runner.stream,
                        version,
                        max(0, deadline - time.monotonic()),
                    )
                    next_notice = time.monotonic() + 30
                time.sleep(min(0.25, max(0, deadline - time.monotonic())))
            receipt = runner.apply(version, full=config.get("full", False))
            sys.stdout.buffer.write(canonical(receipt) + b"\n")
            sys.stdout.buffer.flush()
    finally:
        engine.shutdown()
