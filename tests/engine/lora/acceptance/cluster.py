# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Own a local, real Relax deployment for GPU acceptance.

Uses the production startup handoff from RolloutManager to SessionShard. No
Session, publication manager, engine, or HTTP endpoint is simulated. The full
training Controller is outside this inference acceptance profile.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4


@dataclass
class Cluster:
    manager: Any
    rollout_url: str
    ray_address: str
    ray_namespace: str
    wiring: dict
    close: Callable[[], None]

    def __enter__(self) -> "Cluster":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def _visible_gpus() -> list[int]:
    """Normalize UUIDs without expanding the caller's selected GPU set."""
    selected = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(selected) != 2 or len(set(selected)) != 2 or any(not value.strip() for value in selected):
        raise ValueError("acceptance requires exactly two explicit CUDA_VISIBLE_DEVICES entries")
    output = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    ).stdout
    devices = {uuid.strip(): int(index) for index, uuid in (line.split(",", 1) for line in output.splitlines())}
    indices = [devices[value.strip()] if value.strip().startswith("GPU-") else int(value) for value in selected]
    if len(set(indices)) != 2 or any(index not in devices.values() for index in indices):
        raise ValueError("selected GPU identities do not resolve to two physical devices")
    # SGLangEngine._to_local_gpu_id currently accepts integer visibility only.
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, indices))
    return indices


def _arguments(config: dict, output: Path) -> argparse.Namespace:
    from relax.backends.sglang.arguments import add_sglang_arguments, validate_args
    from relax.utils.arguments import get_slime_extra_args_provider

    parser = get_slime_extra_args_provider()(argparse.ArgumentParser(add_help=False))
    args = parser.parse_args([])
    sglang_parser = add_sglang_arguments(argparse.ArgumentParser(add_help=False))
    vars(args).update(vars(sglang_parser.parse_args([])))
    prompts = output / "prompts.jsonl"
    prompts.write_text("".join(json.dumps({"prompt": prompt}) + "\n" for prompt in config["prompts"]))
    values = {
        "hf_checkpoint": str(Path(config["model_path"]).resolve()),
        "model_source": None,
        "train_backend": "megatron",
        "debug_train_only": False,
        "debug_rollout_only": False,
        "load_debug_rollout_data": None,
        "fp16": False,
        "bf16": True,
        "load": None,
        "save": None,
        "use_agentic_rollout": True,
        "rollout_function_path": "relax.agentic.rollout.generate_rollout",
        "eval_function_path": "relax.agentic.rollout.generate_rollout",
        "apply_chat_template": False,
        "resource": {"actor": [1, 1]},
        "lora_publication_config": str(Path(config["publication_config"]).resolve()),
        "lora_adapter_mode": True,
        "lora_rank": int(config.get("lora_rank", 8)),
        "lora_target_modules": list(config.get("lora_target_modules", ["q_proj", "v_proj"])),
        "rollout_num_gpus": 2,
        "rollout_num_gpus_per_engine": 1,
        "num_gpus_per_node": 2,
        "actor_num_nodes": 1,
        "actor_num_gpus_per_node": 0,
        # No training service or mutable DCS weight transfer is required here.
        "fully_async": False,
        "hybrid": False,
        "colocate": False,
        "offload_train": False,
        "offload_rollout": False,
        "agentic_program_admission": False,
        "use_fault_tolerance": False,
        "num_rollout": 1,
        "num_epoch": None,
        "rollout_global_dataset": True,
        "prompt_data": str(prompts),
        "input_key": "prompt",
        "rollout_shuffle": False,
        "rollout_batch_size": 1,
        "over_sampling_batch_size": 1,
        "global_batch_size": 1,
        "num_iters_per_train_update": 1,
        "agentic_concurrency": 2 * len(config["prompts"]) + 2,
        "agentic_session_lifecycle": bool(config.get("session_lifecycle", False)),
        "agent_command": config["agent_command"],
        "agent_cwd": str(Path(__file__).resolve().parents[4]),
        "agent_env": [],
        "agent_timeout": float(config.get("timeout_seconds", 600)),
        "encode_max_workers": 2,
        "mm_processor_pool_size": 0,
        "apply_chat_template_kwargs": {"enable_thinking": False},
        "use_wandb": False,
        "use_tensorboard": False,
        "use_clearml": False,
        "sglang_server_concurrency": 32,
        "sglang_tensor_parallel_size": 1,
        "sglang_mem_fraction_static": float(config.get("mem_fraction", 0.2)),
        "sglang_max_total_tokens": 8192,
        "sglang_dtype": "bfloat16",
        "sglang_lora_backend": "triton",
        "sglang_context_length": int(config.get("context_length", 4096)),
        "sglang_max_running_requests": 32,
        "sglang_cuda_graph_max_bs_decode": 32,
        "sglang_disable_cuda_graph": False,
        "sglang_disable_decode_cuda_graph": False,
        "sglang_disable_radix_cache": False,
        "sglang_enable_lora_overlap_loading": False,
        "sglang_enable_metrics": bool(config.get("native_metrics", False)),
        "sglang_enable_session_radix_cache": bool(config.get("session_lifecycle", False)),
        "sglang_radix_eviction_policy": "priority" if config.get("session_lifecycle") else "lru",
        "rollout_engine_init_timeout": float(config.get("startup_timeout_seconds", 1200)),
    }
    if config.get("deterministic", False):
        values.update(sglang_enable_deterministic_inference=True, sglang_attention_backend="triton")
    vars(args).update(values)
    validate_args(args)
    return args


def start_cluster(config: dict) -> Cluster:
    """Start only a fresh local Ray; close owns only resources created here.

    Must run in a private subprocess before importing CUDA libraries. The
    parent process owns its process group as a final cleanup boundary.
    """
    gpu_indices = _visible_gpus()
    with socket.socket() as probe:
        try:
            probe.bind(("0.0.0.0", 8000))
        except OSError as error:
            raise RuntimeError("Serve port 8000 is occupied; refusing to replace another deployment") from error
    import ray
    import transfer_queue as tq
    from omegaconf import OmegaConf
    from ray import serve
    from ray.util.placement_group import placement_group, remove_placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    from relax.agentic.runner.ipc import _wait_for_launcher, launcher_socket_path
    from relax.agentic.session.service import deploy_agentic_chat_api_services, shutdown_agentic_chat_api_services
    from relax.components.rollout import Rollout
    from relax.distributed.ray.placement_group import InfoActor
    from relax.distributed.ray.rollout import stop_launched_routers
    from relax.engine.rollout.data_source import RolloutDataSource
    from relax.utils.data.identity_window_sampler import IdentityWindowSampler
    from relax.utils.utils import get_serve_url

    if ray.is_initialized():
        raise RuntimeError("acceptance must not attach to an already initialized Ray runtime")
    output = Path(config["output"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    args = _arguments(config, output)
    namespace = "no7-" + uuid4().hex
    os.environ["RELAX_LAUNCHER_NAMESPACE"] = namespace
    runtime_env = {
        "env_vars": {
            "OMP_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
            **{
                name: os.environ[name]
                for name in ("NO7_TEST_CONTROL_FILE", "PYTHONPATH", "RELAX_LAUNCHER_NAMESPACE")
                if name in os.environ
            },
        }
    }
    cpu_count = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    if cpu_count < 32:
        raise RuntimeError("real Agentic deployment has 16 ingress replicas; at least 32 available CPUs are required")
    cpu_budget = min(32, cpu_count)
    stack = ExitStack()
    try:
        temporary = tempfile.mkdtemp(prefix="no7-ray-")
        context = ray.init(
            address="local",
            namespace=namespace,
            num_cpus=cpu_budget,
            num_gpus=2,
            include_dashboard=False,
            _temp_dir=temporary,
            runtime_env=runtime_env,
        )
        stack.callback(ray.shutdown)
        serve.start(http_options={"host": "0.0.0.0", "port": 8000})
        stack.callback(serve.shutdown)
        stack.callback(stop_launched_routers)
        tq_config = OmegaConf.create(
            {
                "controller": {
                    "sampler": IdentityWindowSampler(dp_size=1, placement="sequential"),
                    "polling_mode": args.polling_mode,
                },
                "backend": {"SimpleStorage": {"num_data_storage_units": 1}},
            },
            flags={"allow_objects": True},
        )
        args.tq_config = tq.init(conf=tq_config) or tq_config
        source = ray.remote(RolloutDataSource).options(num_cpus=1).remote(args)
        stack.callback(ray.kill, source)
        pg = placement_group([{"GPU": 1, "CPU": 1}] * 2, strategy="PACK")
        stack.callback(remove_placement_group, pg)
        ray.get(pg.ready(), timeout=60)
        identities = []
        for bundle in range(2):
            actor = InfoActor.options(
                num_cpus=0.1,
                num_gpus=0.1,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg, placement_group_bundle_index=bundle
                ),
            ).remote()
            try:
                _ip, identity = ray.get(actor.get_ip_and_gpu_id.remote(), timeout=60)
                identities.append(int(identity))
            finally:
                ray.kill(actor)

        launcher_log = stack.enter_context((output / "agent-launcher.log").open("w"))
        launcher = subprocess.Popen(
            [sys.executable, "-m", "relax.agentic.runner.ipc", "--launcher-socket", launcher_socket_path()],
            stdin=subprocess.DEVNULL,
            stdout=launcher_log,
            stderr=subprocess.STDOUT,
        )

        def stop_launcher():
            if launcher.poll() is None:
                launcher.terminate()
                try:
                    launcher.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    launcher.kill()
                    launcher.wait(timeout=10)

        stack.callback(stop_launcher)
        _wait_for_launcher(launcher_socket_path())
        manager = None

        def stop_deployment():
            try:
                shutdown_agentic_chat_api_services()
            finally:
                if manager is not None:
                    try:
                        ray.get(manager.dispose.remote(), timeout=90)
                    finally:
                        ray.kill(manager, no_restart=True)

        stack.callback(stop_deployment)
        deploy_agentic_chat_api_services(config=args, runtime_env=runtime_env)
        rollout = serve.run(
            Rollout.options(ray_actor_options={"num_cpus": 1, "runtime_env": runtime_env}).bind(
                None, (pg, [0, 1], identities), args, data_source=source, runtime_env=runtime_env
            ),
            name="rollout",
            route_prefix="/rollout",
        )
        manager = rollout.get_rollout_manager.remote().result(timeout_s=60)
        wiring = {
            "controller_startup_test": False,
            "production_shard_configuration": True,
            "gpu_indices": gpu_indices,
            "ray_gpu_ids": identities,
            "ray_cpu_budget": cpu_budget,
            "ray_temp_dir": temporary,
            "fixed_cohort_only": True,
        }
        (output / "deployment.json").write_text(json.dumps(wiring, indent=2))
        return Cluster(
            manager=manager,
            rollout_url=get_serve_url("/rollout"),
            ray_address=context.address_info["address"],
            ray_namespace=namespace,
            wiring=wiring,
            close=stack.close,
        )
    except BaseException:
        stack.close()
        raise
