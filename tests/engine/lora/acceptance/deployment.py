# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Create the real services and shared evidence context for one isolated
acceptance task."""

from __future__ import annotations

import asyncio
import importlib
import json
import time
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import httpx

from relax.engine.lora.cli import service_url
from relax.utils.logging_utils import get_logger

from . import support
from .processes import baseline, inference_profile


logger = get_logger(__name__)
TASKS = ("publication", "sessions", "capacity", "numerical_cache", "performance", "export")


async def check_managed_profiles(ctx) -> None:
    state = ctx.report["initial"]
    ready = state["versions"][state["default"]["version_id"]]["ready"]
    profiles = ctx.report["engine_profiles"] = {}
    for engine_id in state["serving_engines"]:
        response = await ctx.client.get(ready[engine_id]["engine"]["endpoint"] + "/server_info")
        response.raise_for_status()
        profiles[engine_id] = inference_profile(
            response.json(), deterministic=ctx.config.get("deterministic", False), cold=False
        )


async def execute(config: dict, cluster, report: dict) -> None:
    from argparse import Namespace

    from transformers import AutoTokenizer

    from relax.agentic.pipeline.runtime import RuntimeDomain

    module = importlib.import_module(f"{__package__}.test_{config['task']}")
    runtime = train_runtime = None
    streams, native_sessions = [], []
    async with httpx.AsyncClient(timeout=httpx.Timeout(10, read=120), trust_env=False) as client:
        try:
            runtime = await RuntimeDomain.connect(
                Namespace(rollout_global_dataset=True), "eval", concurrency=2 * len(config["prompts"]) + 2
            )
            train_runtime = await RuntimeDomain.connect(Namespace(rollout_global_dataset=True), "train", concurrency=1)
            ctx = SimpleNamespace(
                config=config,
                report=report,
                client=client,
                manager=cluster.manager,
                runtime=runtime,
                train_runtime=train_runtime,
                streams=streams,
                native_sessions=native_sessions,
                native_owner="acceptance-" + uuid4().hex,
                root=Path(config["barrier_directory"]),
                test_control=Path(config["test_control_file"]),
                tokenizer=AutoTokenizer.from_pretrained(config["model_path"], local_files_only=True),
                prompts=config["prompts"],
                diagnostic_ids=config["diagnostic_token_ids"],
                versions=config["versions"],
                atol=config["atol"],
                rtol=config["rtol"],
                base=service_url(cluster.rollout_url, "rollout"),
            )
            checks = ("steady_state_overhead",) if config.get("overhead_profile") else module.CHECKS
            report["checks"] = {name: {"status": "NOT_RUN"} for name in checks}
            support.configure_test(ctx)
            report["initial"] = await support.request(ctx, "GET", "/lora/versions")
            if report["initial"]["capacity"] != 2 or report["initial"]["default"]["version_id"] != "A":
                raise ValueError("each task requires a fresh capacity-two cohort with A published")
            await check_managed_profiles(ctx)
            await asyncio.wait_for(module.run(ctx), config["timeout_seconds"])
            missing = [name for name in checks if report["checks"][name]["status"] != "PASS"]
            if missing:
                raise AssertionError(f"task did not produce required evidence: {missing}")
            report["final"] = await support.request(ctx, "GET", "/lora/versions")
        finally:
            support.write_test_control(Path(config["test_control_file"]), {})

            async def cleanup():
                outcomes = await asyncio.gather(
                    *(domain.drop_group(stream) for domain, stream in streams),
                    *(
                        cluster.manager.lora_control.remote(
                            "close", {"owner_epoch": ctx.native_owner, "session_id": bound["session_id"]}
                        )
                        for bound in native_sessions
                    ),
                    return_exceptions=True,
                )
                for domain in (train_runtime, runtime):
                    if domain is not None:
                        await domain.shutdown()
                errors = [str(item) for item in outcomes if isinstance(item, BaseException)]
                if errors:
                    raise RuntimeError("Session cleanup failed: " + "; ".join(errors))

            try:
                await asyncio.wait_for(cleanup(), 120)
                report["session_cleanup"] = "CLOSE_ACCEPTED"
            except BaseException as error:
                report["session_cleanup"] = f"UNKNOWN: {error}"
                raise


def worker(config: dict) -> None:
    from .cluster import start_cluster

    output = Path(config["output"])
    report = {
        "status": "FAIL",
        "task": config["task"],
        "profile": config,
        "started_at": time.time(),
        "checks": {},
        "samples": [],
        "events": [],
        "numerical": [],
        "processes": [],
    }
    try:
        # Blocking Ray/Serve startup happens outside the async HTTP loop.
        with start_cluster(config) as cluster, ExitStack() as stack:
            report["deployment"] = cluster.wiring
            config.update(
                ray_address=cluster.ray_address, ray_namespace=cluster.ray_namespace, rollout_url=cluster.rollout_url
            )
            artifacts = json.loads(Path(config["artifacts_file"]).read_text())
            config["baselines"] = {}
            if config["task"] not in ("export", "performance"):
                for name, gpu in zip(("A", "B"), config["gpus"], strict=True):
                    config["baselines"][name] = stack.enter_context(
                        baseline(
                            config["model_path"],
                            artifacts["versions"][name]["path"],
                            name,
                            gpu["uuid"],
                            output / f"baseline-{name}.log",
                            False,
                            report["processes"],
                            deterministic=config.get("deterministic", False),
                            metrics=config.get("native_metrics", False),
                        )
                    )
            if config["task"] == "performance":
                config["performance_baselines"] = [
                    stack.enter_context(
                        baseline(
                            config["model_path"],
                            artifacts["versions"]["A"]["path"],
                            "A",
                            gpu["uuid"],
                            output / f"ordinary-{index}.log",
                            False,
                            report["processes"],
                            deterministic=config.get("deterministic", False),
                            metrics=config.get("native_metrics", False),
                        )
                    )
                    for index, gpu in enumerate(config["gpus"])
                ]
                for item in config["performance_baselines"]:
                    item["hardware"] = "same selected GPU, TP=1, BF16, Triton, memory fraction 0.2, graph enabled"
            asyncio.run(execute(config, cluster, report))
        report["status"] = "PASS"
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        # Reference engines log outside deployment.log; retain their first
        # useful evidence when HTTP only reports that the server disconnected.
        report["process_log_tails"] = {
            item["log"]: Path(item["log"]).read_text(errors="replace").splitlines()[-120:]
            for item in report["processes"]
            if Path(item["log"]).is_file()
        }
        raise
    finally:
        report["finished_at"] = time.time()
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        logger.info("%s: %s; report %s", config["task"], report["status"], output / "report.json")
