# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Opt-in two-node engine/lifecycle smoke test; run inside the training image.

Requires explicit INFERENCE_RUN_GPU_TESTS=1, RAY_ADDRESS, MODEL_DIR (a single
HF checkpoint), and INFERENCE_GPUS_PER_NODE. The model must support TP equal to
twice that node width. This does not substitute for a full OPD training run.
"""

import os
from types import SimpleNamespace

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("INFERENCE_RUN_GPU_TESTS") != "1",
    reason="Requires explicit opt-in and an available two-node GPU cluster with a shared model checkpoint",
)


def _gpu_pool_setup(*, pp_size=1):
    import dataclasses

    import ray
    from sglang.srt.server_args import ServerArgs

    from relax.backends.sglang.sglang_engine import SGLangEngine
    from relax.core.service import create_placement_group
    from relax.distributed.ray.inference_ports import allocate_inference_ports
    from relax.distributed.ray.multi_engine_manager import MultiEngineManager
    from relax.distributed.ray.utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST
    from relax.inference.placement import ModelPlacement

    required = ["RAY_ADDRESS", "MODEL_DIR", "INFERENCE_GPUS_PER_NODE"]
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        pytest.fail(f"Explicit GPU validation is missing configuration: {missing}")
    node_width = int(os.environ["INFERENCE_GPUS_PER_NODE"])
    width = 2 * node_width
    ray.init(address=os.environ["RAY_ADDRESS"], ignore_reinit_error=True)
    gpu_nodes = [node for node in ray.nodes() if node["Alive"] and node["Resources"].get("GPU", 0) > 0]
    assert len(gpu_nodes) >= 2
    assert all(node["Resources"]["GPU"] == node_width for node in gpu_nodes), "Use a homogeneous test cluster"
    engine_config = {
        "enable_weights_cpu_backup": True,
        "max_total_tokens": 2048,
        "context_length": 1024,
        "mem_fraction_static": 0.5,
        "attention_backend": os.environ.get("INFERENCE_ATTENTION_BACKEND", "triton"),
        "pp_size": pp_size,
    }
    fields = {field.name for field in dataclasses.fields(ServerArgs)}
    graph_limit = "cuda_graph_max_bs_decode" if "cuda_graph_max_bs_decode" in fields else "cuda_graph_max_bs"
    engine_config[graph_limit] = 4
    args = SimpleNamespace(
        hf_checkpoint=os.environ["MODEL_DIR"],
        genrm_model_path=os.environ["MODEL_DIR"],
        seed=1,
        num_gpus_per_node=node_width,
        rollout_num_gpus_per_engine=width,
        genrm_num_gpus_per_engine=width,
        offload_rollout=True,
        sglang_pp_size=pp_size,
        sglang_dp_size=1,
        sglang_ep_size=1,
        sglang_router_ip=None,
        sglang_router_port=None,
        use_rollout_routing_replay=False,
        fp16=False,
        rollout_external=False,
        fully_async=False,
        genrm_engine_config=engine_config,
        _inference_placement=(
            ModelPlacement("rollout", "default", "validation", 0, width, width, "test", "inference"),
        ),
    )
    pg = create_placement_group(width, node_group_affinity=False)

    class Pool(MultiEngineManager):
        num_gpu_per_engine = node_width

        def __init__(self, args, role, skip_init=False):
            self.role = role
            self.peer_pools = []
            super().__init__(
                args,
                num_slots=2,
                nodes_per_engine=2,
                engine_actor_cls=SGLangEngine,
                role=role,
                skip_init=skip_init,
            )
            if skip_init:
                self._onloaded = False
                self.inference.state = "sleeping"

        def set_peers(self, peers):
            self.peer_pools = peers

        def onload(self, tags=None):
            for peer in self.peer_pools:
                snapshot = ray.get(peer.get_inference_snapshot.remote())
                assert snapshot["models"]["default"]["state"] == "sleeping", snapshot
            return super().onload(tags)

        def _resolve_placement(self, rank):
            return pg, False, rank * node_width

        def _ray_resource_kwargs(self, rank):
            return {"num_cpus": 0.2, "num_gpus": 0.2}

        def _build_engine_env_vars(self):
            return {
                **dict.fromkeys(NOSET_VISIBLE_DEVICES_ENV_VARS_LIST, "1"),
                "SGLANG_MEMORY_SAVER_CUDA_GRAPH": "true",
                "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "false",
                "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE": "false",
            }

        def _build_engine_ctor_kwargs(self, rank):
            return {
                "role": self.role,
                "num_gpus_per_engine": width,
                "sglang_overrides": engine_config,
            }

        def _build_engine_init_kwargs(self, rank, address):
            return {**address, "skip_dcs_registration": True, "skip_router_registration": True}

        def _allocate_engine_addr_and_ports(self, *, new_engines):
            base_port = {"rollout": 30000, "teacher": 32000, "genrm": 34000}[self.role]
            return allocate_inference_ports(new_engines, nodes_per_engine=2, base_port=base_port)[0]

    return args, pg, Pool


@pytest.mark.parametrize("role", ["rollout", "genrm", "teacher"])
def test_multinode_shared_engine_lifecycle_and_pg_ownership(role):
    _verify_multinode_shared_engine_lifecycle(role, pp_size=1)


@pytest.mark.parametrize("role", ["rollout", "genrm", "teacher"])
def test_multinode_pipeline_parallel_lifecycle_and_pg_ownership(role):
    _verify_multinode_shared_engine_lifecycle(role, pp_size=2)


def _verify_multinode_shared_engine_lifecycle(role, *, pp_size):
    import ray
    import requests

    from relax.inference.routing import ModelUnavailable, select_endpoint

    args, pg, Pool = _gpu_pool_setup(pp_size=pp_size)

    pool = None
    try:
        pool = Pool(args, role)
        snapshot = pool.get_inference_snapshot()
        assert len(snapshot["models"]["default"]["engines"]) == 1
        assert ray.get(pool.all_engines[1].get_url.remote()) is None
        endpoint = select_endpoint(snapshot, "default")

        def generate():
            response = requests.post(
                f"{endpoint}/generate",
                json={
                    "text": "Return the word ready.",
                    "sampling_params": {"temperature": 0, "max_new_tokens": 4},
                    "return_logprob": True,
                },
                timeout=180,
            )
            response.raise_for_status()
            result = response.json()
            assert result["meta_info"]["output_token_logprobs"]
            return result

        before = generate()
        pool.offload()
        pool.offload()
        with pytest.raises(ModelUnavailable):
            select_endpoint(pool.get_inference_snapshot(), "default")
        pool.onload(["weights"])
        with pytest.raises(ModelUnavailable):
            select_endpoint(pool.get_inference_snapshot(), "default")
        pool.onload(["kv_cache", "cuda_graph"])
        pool.onload()
        after = generate()
        assert after["text"] == before["text"], "Weights must survive offload/onload"
        before_probs = before["meta_info"]["output_token_logprobs"]
        after_probs = after["meta_info"]["output_token_logprobs"]
        assert [row[1] for row in after_probs] == [row[1] for row in before_probs]
        assert [row[0] for row in after_probs] == pytest.approx([row[0] for row in before_probs], abs=2e-3)
        if role != "rollout":
            with pytest.raises(Exception, match="Static inference models"):
                ray.get(pool.engines[0].register_dcs.remote())
        pool.shutdown()
        pool.shutdown()
        assert ray.util.placement_group_table(pg[0])["state"] == "CREATED"
    finally:
        try:
            if pool is not None:
                pool.shutdown()
        finally:
            ray.util.remove_placement_group(pg[0])


async def score_gpu_judge(args, samples):
    """Custom reward hook that requires a real GenRM response, without
    fallbacks."""
    import math

    import httpx

    from relax.inference.routing import select_endpoint

    pools = args._test_pools
    for role in ("rollout", "teacher"):
        snapshot = await pools[role].get_inference_snapshot.remote()
        assert snapshot["models"]["default"]["state"] == "sleeping", snapshot
    snapshot = await pools["genrm"].get_inference_snapshot.remote()
    endpoint = select_endpoint(snapshot, "default")
    rewards = []
    async with httpx.AsyncClient(timeout=180) as client:
        for sample in samples:
            assert sample.teacher_log_probs and len(sample.teacher_log_probs) == sample.response_length
            response = await client.post(
                f"{endpoint}/generate",
                json={
                    "text": f"Evaluate this answer: {sample.response}",
                    "sampling_params": {"temperature": 0, "max_new_tokens": 2},
                    "return_logprob": True,
                },
            )
            response.raise_for_status()
            rows = response.json()["meta_info"]["output_token_logprobs"]
            assert rows
            reward = sum(row[0] for row in rows) / len(rows)
            assert math.isfinite(reward)
            rewards.append(reward)
    args._test_events.append("genrm_scored")
    return rewards


def test_multinode_cleanup_rpc_failure_keeps_gpu_lease_and_pg():
    import asyncio

    import ray
    import requests

    from relax.inference.lifecycle import LifecycleCoordinator
    from relax.inference.routing import ModelUnavailable, select_endpoint

    args, pg, Pool = _gpu_pool_setup()
    pool = None
    outage = True

    @ray.remote(num_cpus=0)
    def unavailable():
        raise ConnectionError("Injected inference control RPC outage")

    class InterruptedHandle:
        def __init__(self, handle):
            self.handle = handle

        def __getattr__(self, name):
            if outage and name in {"release_memory_occupation", "shutdown"}:
                return SimpleNamespace(remote=lambda **kwargs: unavailable.remote())
            return getattr(self.handle, name)

    try:
        pool = Pool(args, "teacher")
        endpoint = select_endpoint(pool.get_inference_snapshot(), "default")
        pool.all_engines = [InterruptedHandle(engine) for engine in pool.all_engines]
        coordinator = LifecycleCoordinator()

        async def idle():
            return None

        async def offload():
            pool.offload()

        async def blocked_next_phase():
            pytest.fail("Another model acquired GPUs with unconfirmed cleanup")

        async def fail_phase():
            with pytest.raises(RuntimeError, match="cleanup unconfirmed"):
                await coordinator.run_phase("teacher", idle, offload, idle)
            assert coordinator.phase == "teacher"
            with pytest.raises(RuntimeError, match="lease retained"):
                await coordinator.run_phase("genrm", blocked_next_phase, idle, idle)

        asyncio.run(fail_phase())
        assert pool.inference.state == "failed"
        assert pool._cleanup_pending == {0, 1}
        assert all(engine is not None for engine in pool.all_engines)
        assert ray.util.placement_group_table(pg[0])["state"] == "CREATED"
        with pytest.raises(ModelUnavailable):
            select_endpoint(pool.get_inference_snapshot(), "default")
        with pytest.raises(RuntimeError, match="cleanup pending"):
            pool.onload()
        # A failed control RPC really can leave the original GPU model alive.
        response = requests.post(
            f"{endpoint}/generate",
            json={"text": "Count: one, two,", "sampling_params": {"temperature": 0, "max_new_tokens": 2}},
            timeout=180,
        )
        response.raise_for_status()
        assert response.json()["meta_info"]["completion_tokens"] > 0
        outage = False
        pool.all_engines = [engine.handle for engine in pool.all_engines]
        pool.offload()
        assert not pool._cleanup_pending
        assert pool.all_engines == [None, None]
        assert pool.inference.state == "sleeping"
        assert ray.util.placement_group_table(pg[0])["state"] == "CREATED"
    finally:
        outage = False
        if pool is not None:
            pool.all_engines = [
                engine.handle if isinstance(engine, InterruptedHandle) else engine for engine in pool.all_engines
            ]
            pool.shutdown()
        ray.util.remove_placement_group(pg[0])


def test_multinode_real_engine_actor_loss_recovers_and_republishes():
    """Kill one owned Ray engine actor and rebuild the logical TP replica.

    This is intentionally separate from the RPC-outage test above.  A dead
    actor cannot answer ``shutdown``; the manager must acknowledge that Ray has
    already reclaimed the process, release its owned placement group, and
    publish a replacement only after every slot in the logical engine is
    healthy again.
    """

    import ray
    import requests

    from relax.inference.routing import select_endpoint

    args, pg, Pool = _gpu_pool_setup()
    pool = None
    try:
        pool = Pool(args, "teacher")
        victim = pool.all_engines[0]
        ray.kill(victim, no_restart=True)

        dead = pool._fanout("release_memory_occupation")
        assert dead == [0]
        pool._retire_engines(dead)
        assert pool.all_engines == [None, None]
        assert not pool._cleanup_pending

        rebuilt = pool.recover()
        assert rebuilt == {0, 1}
        snapshot = pool.get_inference_snapshot()
        assert snapshot["models"]["default"]["state"] == "ready"
        endpoint = select_endpoint(snapshot, "default")
        response = requests.post(
            f"{endpoint}/generate",
            json={"text": "Return the word ready.", "sampling_params": {"temperature": 0, "max_new_tokens": 4}},
            timeout=180,
        )
        response.raise_for_status()
        assert response.json()["meta_info"]["completion_tokens"] > 0
    finally:
        if pool is not None:
            pool.shutdown()
        ray.util.remove_placement_group(pg[0])


def test_multinode_deferred_teacher_writeback_before_training_publish(monkeypatch):
    import asyncio
    import math
    import socket
    import sys

    import ray
    import requests
    from ray import serve
    from transformers import AutoTokenizer

    from relax.components.inference_gateway import deploy_teacher_gateway
    from relax.distributed.ray import rollout as rollout_module
    from relax.engine.rollout.on_policy_distillation import OpdManager
    from relax.inference.lifecycle import LifecycleCoordinator, run_deferred_batch
    from relax.inference.routing import select_endpoint
    from relax.utils.types import Sample
    from relax.utils.utils import CURRENT_ROLLOUT_BATCH, transfer_batch_to_data_system

    args, pg, Pool = _gpu_pool_setup()
    pools = {}
    published = []
    events = []
    try:
        RemotePool = ray.remote(num_cpus=0.25, concurrency_groups={"discovery": 1})(Pool)
        for role in ("rollout", "teacher", "genrm"):
            pools[role] = RemotePool.remote(args, role, skip_init=role != "rollout")
        snapshots = ray.get([pool.get_inference_snapshot.remote() for pool in pools.values()])
        assert [s["models"]["default"]["state"] for s in snapshots] == ["ready", "sleeping", "sleeping"]
        assert all(
            engine["base_url"] is None and not engine["direct_eligible"]
            for snapshot in snapshots[1:]
            for engine in snapshot["models"]["default"]["engines"]
        )
        ray.get(
            [
                pool.set_peers.remote([other for key, other in pools.items() if key != role])
                for role, pool in pools.items()
            ]
        )

        endpoint = select_endpoint(snapshots[0], "default")
        tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint)
        samples = []
        for length in (2, 4):
            prompt = "Continue counting: one, two, three,"
            prompt_ids = tokenizer.encode(prompt)
            response = requests.post(
                f"{endpoint}/generate",
                json={
                    "input_ids": prompt_ids,
                    "sampling_params": {"temperature": 0, "max_new_tokens": length, "ignore_eos": True},
                    "return_logprob": True,
                },
                timeout=180,
            )
            response.raise_for_status()
            result = response.json()
            rows = result["meta_info"]["output_token_logprobs"]
            assert len(rows) == length
            samples.append(
                Sample(
                    index=7,
                    group_index=length,
                    prompt=prompt,
                    tokens=prompt_ids + [row[1] for row in rows],
                    response=result["text"],
                    response_length=length,
                    rollout_log_probs=[row[0] for row in rows],
                    status=Sample.Status.COMPLETED,
                )
            )

        # A dedicated test cluster owns Serve; its HTTP listener uses a free port.
        with socket.socket() as sock:
            sock.bind(("", 0))
            http_port = sock.getsockname()[1]
        serve.start(http_options={"host": "0.0.0.0", "port": http_port})
        args.enable_affinity = False
        deploy_teacher_gateway({"default": pools["teacher"]}, args)
        args.opd_teacher_gateway_url = f"http://127.0.0.1:{http_port}/teacher"
        args.use_opd = True
        args.opd_type = "sglang"
        args.opd_token_selection = "student_sampled"
        args.opd_teacher_defer = True
        args.opd_teacher_timeout_s = 180
        args.opd_teacher_connector_limit = 4
        args._inference_teacher_managers = {"default": pools["teacher"]}
        args.defer_reward_to_post_process = True
        args.custom_rm_path = f"{__name__}.score_gpu_judge"
        args.custom_reward_post_process_path = None
        args.custom_convert_samples_to_train_data_path = None
        args.group_rm = False
        args.reward_key = None
        args.reward_max_concurrency = 2
        args.advantage_estimator = "grpo"
        args.rewards_normalization = False
        args.n_samples_per_prompt = 1
        args.multimodal_keys = None
        args.debug_train_only = False
        args._test_pools = pools
        args._test_events = events

        async def rollout_on():
            await pools["rollout"].onload.remote()

        async def rollout_off():
            await pools["rollout"].offload.remote()

        facade = SimpleNamespace(
            onload=rollout_on,
            offload=rollout_off,
            lifecycle_coordinator=LifecycleCoordinator(),
            _inference_genrm_managers=[pools["genrm"]],
        )
        monkeypatch.setattr(rollout_module, "_LOCAL_ROLLOUT_MANAGER", facade)

        class TrainingSink:
            async def async_put(self, *, data, partition_id, custom_meta, is_last):
                assert events == ["genrm_scored"]
                for pool in pools.values():
                    snapshot = await pool.get_inference_snapshot.remote()
                    assert snapshot["models"]["default"]["state"] == "sleeping", snapshot
                assert partition_id == "train_0" and is_last
                assert [row["sample_index"] for row in custom_meta] == [7, 7]
                for ordinal, sample in enumerate(samples):
                    assert len(sample.teacher_log_probs) == sample.response_length
                    assert all(math.isfinite(value) for value in sample.teacher_log_probs)
                    assert data["teacher_log_probs"][ordinal].tolist() == pytest.approx(sample.teacher_log_probs)
                    # Identical checkpoints must score the sampled tokens consistently.
                    assert sample.teacher_log_probs == pytest.approx(sample.rollout_log_probs, abs=2e-2)
                    assert data["rewards"][ordinal].item() == pytest.approx(sample.reward)
                published.append(data)

        async def publish():
            await transfer_batch_to_data_system(
                args, [[sample] for sample in samples], 0, 0, TrainingSink(), is_last=True
            )

        asyncio.run(run_deferred_batch(args, samples, OpdManager(args), None, publish))
        assert len(published) == 1
        assert facade.lifecycle_coordinator.phase is None
        assert ray.util.placement_group_table(pg[0])["state"] == "CREATED"
    finally:
        had_failure = sys.exc_info()[0] is not None
        cleanup_errors = []
        try:
            serve.shutdown()
        except Exception as exc:
            cleanup_errors.append(exc)
        for pool in pools.values():
            try:
                ray.get(pool.shutdown.remote(), timeout=180)
            except Exception as exc:
                cleanup_errors.append(exc)
            try:
                ray.kill(pool)
            except Exception as exc:
                cleanup_errors.append(exc)
        try:
            assert ray.util.placement_group_table(pg[0])["state"] == "CREATED"
        except Exception as exc:
            cleanup_errors.append(exc)
        finally:
            try:
                ray.util.remove_placement_group(pg[0])
            finally:
                CURRENT_ROLLOUT_BATCH.clear()
        if cleanup_errors and not had_failure:
            raise RuntimeError("Deferred GPU test cleanup failed") from cleanup_errors[0]
