# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Real production RolloutManager, gateway and router on two GPU nodes."""

import os

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("INFERENCE_RUN_GPU_TESTS") != "1",
    reason="Requires an explicitly enabled two-node GPU cluster and shared model",
)


@pytest.mark.parametrize("topology", ["pp", "pd"])
def test_production_manager_gateway_direct_and_topology(topology):
    import argparse
    import asyncio
    import copy
    import dataclasses
    import json
    import socket
    import tempfile
    import time
    from pathlib import Path

    import httpx
    import ray
    import requests
    import transfer_queue as tq
    import yaml
    from fastapi import FastAPI, Request
    from omegaconf import OmegaConf
    from ray import serve
    from sglang.srt.server_args import ServerArgs

    from relax.backends.sglang.arguments import add_sglang_router_arguments
    from relax.components.inference_gateway import InferenceGateway
    from relax.core.service import create_placement_group
    from relax.distributed.ray.rollout import RolloutManager, stop_launched_routers
    from relax.inference.placement import PlacementPlanner
    from relax.inference.routing import ModelUnavailable, select_endpoint
    from relax.utils.inference_client import InferenceClient

    for key in ("RAY_ADDRESS", "MODEL_DIR", "INFERENCE_SHARED_TEST_DIR"):
        assert os.environ.get(key), f"Missing GPU test configuration: {key}"
    ray.init(address=os.environ["RAY_ADDRESS"], ignore_reinit_error=True)
    nodes = [n for n in ray.nodes() if n["Alive"] and n["Resources"].get("GPU", 0)]
    assert len(nodes) >= 2 and all(n["Resources"]["GPU"] == 1 for n in nodes)
    # The fixture consumes two nodes; leave additional homogeneous nodes idle
    # so a larger allocation can validate the same PP/PD topology.
    nodes = nodes[:2]

    args = add_sglang_router_arguments(argparse.ArgumentParser()).parse_args([])
    values = {
        "hf_checkpoint": os.environ["MODEL_DIR"],
        "seed": 1,
        "num_gpus_per_node": 1,
        "rollout_num_gpus": 2,
        "rollout_num_gpus_per_engine": 2 if topology == "pp" else 1,
        "sglang_pp_size": 2 if topology == "pp" else 1,
        "sglang_dp_size": 1,
        "sglang_ep_size": 1,
        "resource": {"rollout": [1, 2]},
        "colocate": False,
        "offload_rollout": True,
        "fully_async": False,
        "fp16": False,
        "rollout_external": False,
        "use_rollout_routing_replay": False,
        "debug_train_only": False,
        "debug_rollout_only": True,
        "enable_affinity": False,
        "use_agentic_rollout": False,
        "use_fault_tolerance": False,
        "ci_test": False,
        "wandb_run_id": None,
        "use_metrics_service": False,
        "custom_reward_post_process_path": None,
        "custom_convert_samples_to_train_data_path": None,
        "rollout_function_path": "relax.engine.rollout.sglang_rollout.generate_rollout",
        "eval_function_path": "relax.engine.rollout.sglang_rollout.generate_rollout",
        "sglang_hf_checkpoint": None,
        "sglang_server_concurrency": 8,
        "use_distributed_post": False,
        "use_slime_router": False,
        "rollout_engine_init_timeout": 600,
    }
    for key, value in values.items():
        setattr(args, key, value)
    overrides = {
        "attention_backend": "triton",
        "mem_fraction_static": 0.5,
        "context_length": 1024,
        "max_total_tokens": 2048,
        "enable_weights_cpu_backup": True,
        "disable_cuda_graph": True,
    }
    fields = {field.name for field in dataclasses.fields(ServerArgs)}
    if "cuda_graph_backend_prefill" in fields:
        overrides["cuda_graph_backend_prefill"] = "disabled"
    if topology == "pd":
        overrides["disaggregation_transfer_backend"] = os.environ.get("INFERENCE_PD_BACKEND", "mooncake")
    shared = Path(tempfile.mkdtemp(prefix=f"topology-{topology}-", dir=os.environ["INFERENCE_SHARED_TEST_DIR"]))
    groups = [
        {"worker_type": kind, "num_gpus": width, "num_gpus_per_engine": width, "overrides": overrides}
        for kind, width in ([("regular", 2)] if topology == "pp" else [("prefill", 1), ("decode", 1)])
    ]
    config_path = shared / "sglang.yaml"
    config_path.write_text(yaml.safe_dump({"sglang": [{"name": "default", "engine_groups": groups}]}))
    args.sglang_config = str(config_path)
    PlacementPlanner.apply(args)
    args.tq_config = OmegaConf.create(
        {
            "backend": {"storage_backend": "SimpleStorage", "SimpleStorage": {"num_data_storage_units": 1}},
            "metrics": {"enabled": False},
        }
    )

    @ray.remote(num_cpus=1, num_gpus=0)
    class Runtime:
        def __init__(self, config, pg):
            # Run the complete production constructor and methods. The wrapper
            # adds only test-owned router cleanup, never substitutes an engine.
            self.manager = RolloutManager.__ray_metadata__.modified_class(config, pg, None)

        def get_inference_snapshot(self):
            return self.manager.get_inference_snapshot()

        async def offload(self):
            await self.manager.offload()

        async def onload(self, tags=None):
            await self.manager.onload(tags)

        def close(self):
            engines = [
                engine
                for server in self.manager.servers.values()
                for group in server.engine_groups
                for engine in group.all_engines
                if engine is not None
            ]
            self.manager.dispose()
            stop_launched_routers()
            for engine in engines:
                ray.kill(engine)
            ray.kill(self.manager.rollout_engine_lock)
            ray.kill(self.manager._weight_sync_lock)

    app = FastAPI()

    @serve.deployment(ray_actor_options={"num_cpus": 1, "num_gpus": 0})
    @serve.ingress(app)
    class Gateway:
        def __init__(self, runtime):
            self.gateway = InferenceGateway("rollout", {"default": runtime})

        @app.get("/engines")
        async def engines(self):
            return await self.gateway.discovery()

        @app.post("/generate")
        async def generate(self, request: Request):
            return await self.gateway.forward(request, "generate")

    pg = None
    runtime = None
    try:
        tq.init(args.tq_config)
        pg = create_placement_group(2, node_group_affinity=False)
        runtime = Runtime.remote(args, pg)
        snapshot = ray.get(runtime.get_inference_snapshot.remote(), timeout=660)
        model = snapshot["models"]["default"]
        assert model["state"] == "ready"
        assert len(model["engines"]) == (1 if topology == "pp" else 2)
        if topology == "pd":
            assert all(not engine["direct_eligible"] for engine in model["engines"])
            assert select_endpoint(snapshot, "default") == model["router_url"]
            without_router = copy.deepcopy(snapshot)
            without_router["models"]["default"]["router_url"] = None
            with pytest.raises(ModelUnavailable):
                select_endpoint(without_router, "default")
        infos = []
        for engine in model["engines"]:
            response = requests.get(f"{engine['base_url']}/get_server_info", timeout=30)
            response.raise_for_status()
            info = response.json()
            infos.append(info)
            assert info["pp_size"] == (2 if topology == "pp" else 1)
            assert info["tp_size"] == 1
        (shared / "server-info.json").write_text(json.dumps(infos, indent=2))

        deadline = time.monotonic() + 90
        workers = None
        while time.monotonic() < deadline:
            response = requests.get(f"{model['router_url']}/workers", timeout=10)
            response.raise_for_status()
            workers = response.json()
            entries = workers.get("workers", []) if isinstance(workers, dict) else workers
            if len(entries) == len(model["engines"]):
                break
            time.sleep(1)
        else:
            pytest.fail(f"Router did not register the expected workers: {workers}")
        (shared / "router-workers.json").write_text(json.dumps(workers, indent=2))
        with socket.socket() as listener:
            listener.bind(("", 0))
            http_port = listener.getsockname()[1]
        serve.start(http_options={"host": "0.0.0.0", "port": http_port})
        serve.run(Gateway.bind(runtime), name="topology-validation", route_prefix="/inference")
        url = f"http://127.0.0.1:{http_port}/inference"
        payload = {
            "text": "Continue counting: one, two, three,",
            "sampling_params": {"temperature": 0, "max_new_tokens": 8, "ignore_eos": True},
            "return_logprob": True,
        }

        async def verify_routes():
            async with InferenceClient(url, direct=False) as gateway, InferenceClient(url, direct=True) as direct:
                before = await gateway.generate(payload)
                other = await direct.generate(payload)
                assert before["text"] == other["text"]
                assert len(before["meta_info"]["output_token_logprobs"]) == 8
                assert len(other["meta_info"]["output_token_logprobs"]) == 8
                await runtime.offload.remote()
                await runtime.offload.remote()
                with pytest.raises(ModelUnavailable):
                    await direct.generate(payload)
                async with httpx.AsyncClient() as raw:
                    response = await raw.post(f"{url}/generate", json=payload)
                    assert response.status_code == 503
                await runtime.onload.remote(["weights"])
                with pytest.raises(ModelUnavailable):
                    await direct.generate(payload)
                await runtime.onload.remote(["kv_cache", "cuda_graph"])
                await runtime.onload.remote()
                after = await gateway.generate(payload)
                assert after["text"] == before["text"]
                assert [row[0] for row in after["meta_info"]["output_token_logprobs"]] == pytest.approx(
                    [row[0] for row in before["meta_info"]["output_token_logprobs"]], abs=2e-3
                )

        asyncio.run(verify_routes())
        assert ray.util.placement_group_table(pg[0])["state"] == "CREATED"
    finally:
        serve.shutdown()
        try:
            if runtime is not None:
                ray.get(runtime.close.remote(), timeout=180)
                ray.kill(runtime)
        finally:
            if pg is not None:
                ray.util.remove_placement_group(pg[0])
            tq.close()
