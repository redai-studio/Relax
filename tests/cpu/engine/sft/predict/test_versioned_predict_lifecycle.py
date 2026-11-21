import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
import ray

from relax.engine.rewards import dapo_genrm
from relax.engine.sft.predict import runner


@pytest.fixture
def trainer(monkeypatch):
    initialize = pytest.importorskip("relax.backends.megatron.initialize")
    actor = Mock(args=SimpleNamespace(offload_train=True, offload_rollout=True))
    actor.genrm_manager = []
    monkeypatch.setattr(initialize, "is_megatron_main_rank", lambda: True)
    monkeypatch.setattr(runner, "get_gloo_group", lambda: None)
    monkeypatch.setattr(runner, "get_serve_url", lambda _: "http://rollout")
    monkeypatch.setattr(runner, "compute_rollout_step", lambda args, step: step)
    monkeypatch.setattr(runner, "tracking_utils", Mock())
    dist = Mock()
    dist.get_world_size.return_value = 1
    dist.all_gather_object.side_effect = lambda out, value, **kwargs: out.__setitem__(0, value)
    monkeypatch.setattr(runner, "dist", dist)
    return actor


def test_partial_resume_settles_and_releases_before_training_wakes(monkeypatch, trainer):
    rollout_module = pytest.importorskip("relax.distributed.ray.rollout")
    manager_type = rollout_module.RolloutManager.__ray_metadata__.modified_class
    manager = object.__new__(manager_type)
    manager.args = SimpleNamespace(offload_rollout=True, weight_version_config="source.json")
    manager.status = "onload"
    manager.health_monitoring_pause = Mock()
    manager.consume_weight_version = AsyncMock()
    rollout = SimpleNamespace(
        _logger=Mock(),
        healthy=Mock(),
        config=SimpleNamespace(loss_type="sft", offload_rollout=True),
        rollout_manager=SimpleNamespace(
            run_predict=SimpleNamespace(remote=manager.run_predict), offload=SimpleNamespace(remote=manager.offload)
        ),
    )
    resident = set()
    events = []
    release_slow = Event()

    def gather(refs):
        return [ref.result() for ref in refs]

    def wait_all(refs, *, num_returns, fetch_local):
        assert num_returns == len(refs)
        release_slow.set()
        for ref in refs:
            ref.exception(timeout=10)
        return refs, []

    monkeypatch.setattr(ray, "get", gather)
    monkeypatch.setattr(ray, "wait", wait_all)
    with ThreadPoolExecutor() as pool:

        def resume(name):
            if name == "broken":
                raise RuntimeError("resume failed")
            if name == "late":
                assert release_slow.wait(10)
            resident.add(name)
            events.append("resume-" + name)

        def release(name):
            resident.discard(name)
            events.append("release-" + name)

        def group(names):
            return SimpleNamespace(
                onload=lambda tags: [pool.submit(resume, name) for name in names],
                offload=lambda: [pool.submit(release, name) for name in names],
            )

        manager.servers = {
            "a": rollout_module.RolloutServer([group(["first"])]),
            "b": rollout_module.RolloutServer([group(["broken", "late"])]),
        }
        asyncio.run(manager.offload())
        events.clear()

        def request(url, *, params, timeout):
            payload = asyncio.run(runner.handle_predict(rollout, **params))
            return Mock(json=lambda: payload)

        monkeypatch.setattr(runner.requests, "get", request)
        trainer.wake_up.side_effect = lambda: events.append("wake")
        try:
            with pytest.raises(RuntimeError, match="resume failed"):
                runner.run_sft_predict(trainer, 3, published_version={"version": 1, "root": "r"})
            assert not resident
            assert manager.status == "offload"
            assert events.index("resume-late") < events.index("release-first")
            assert events.index("release-late") < events.index("wake")
            manager.consume_weight_version.assert_not_awaited()
        finally:
            release_slow.set()


@pytest.mark.parametrize("deferred", [False, True])
def test_managed_judge_scores_installed_weights_before_training_wakes(monkeypatch, trainer, deferred, tmp_path):
    from ray import serve

    from relax.engine.sft.predict import loop
    from relax.utils.types import Sample

    rollout_type = pytest.importorskip("relax.distributed.ray.rollout").RolloutManager.__ray_metadata__.modified_class
    args = trainer.args
    args.loss_type, args.rm_type, args.custom_rm_path = "sft", "dapo-genrm", None
    args.defer_reward_to_post_process = deferred
    args.save, args.weight_version_config = str(tmp_path), "source.json"
    args._genrm_instances_resolved = {"default": {}}
    events = []
    active = False

    def change(ready):
        nonlocal active
        active = ready
        events.append("judge-on" if ready else "judge-off")

    trainer.genrm_manager = [
        SimpleNamespace(
            onload=SimpleNamespace(remote=lambda: change(True)), offload=SimpleNamespace(remote=lambda: change(False))
        )
    ]
    monkeypatch.setattr(ray, "get", lambda refs: refs)
    service = SimpleNamespace(
        onload=SimpleNamespace(remote=AsyncMock(side_effect=lambda: change(True))),
        offload=SimpleNamespace(remote=AsyncMock(side_effect=lambda: change(False))),
    )
    monkeypatch.setattr(serve, "get_app_handle", lambda name: service if name == "genrm" else None)

    async def install(version):
        events.append("install")
        return [{"active_version": {"version": version, "root": "r"}}]

    async def generate(prompts, mm, *, samples):
        events.append("generate")
        samples[0].response = "Answer: 2"
        return [samples[0].response]

    async def judge(_):
        events.append("score")
        if not active:
            raise httpx.ConnectError("judge is offloaded")
        return "Judgement: 1"

    sample = Sample(prompt="1+1?", label="2")

    def prompts(config, *, samples):
        samples.append(sample)
        return [("1+1?", "2", None)]

    monkeypatch.setattr(loop, "render_eval_prompts", prompts)
    monkeypatch.setattr(dapo_genrm, "get_genrm_client", lambda: SimpleNamespace(generate=judge))
    manager = SimpleNamespace(
        args=args,
        onload=AsyncMock(),
        consume_weight_version=install,
        generate_predict=generate,
        submit_eval_result=AsyncMock(),
    )
    rollout = SimpleNamespace(
        _logger=Mock(),
        healthy=Mock(),
        config=args,
        rollout_manager=SimpleNamespace(
            run_predict=SimpleNamespace(remote=lambda *a: rollout_type.run_predict(manager, *a)),
            offload=SimpleNamespace(remote=AsyncMock(side_effect=lambda: events.append("rollout-off"))),
        ),
    )

    def request(url, *, params, timeout):
        payload = asyncio.run(runner.handle_predict(rollout, **params))
        return Mock(json=lambda: payload)

    monkeypatch.setattr(runner.requests, "get", request)
    trainer.wake_up.side_effect = lambda: events.append("wake")
    runner.run_sft_predict(trainer, 3, published_version={"version": 1, "root": "r"})
    assert sample.reward["score"] == 1.0
    assert events == ["install", "generate", "judge-on", "score", "judge-off", "rollout-off", "wake"]
    trainer.update_weights.assert_not_called()


def test_versioned_cleanup_failure_reaches_trainer(monkeypatch, trainer):
    rollout = SimpleNamespace(
        _logger=Mock(),
        healthy=Mock(),
        config=SimpleNamespace(loss_type="sft", offload_rollout=True),
        rollout_manager=SimpleNamespace(
            run_predict=SimpleNamespace(remote=AsyncMock(side_effect=RuntimeError("resume failed"))),
            offload=SimpleNamespace(remote=AsyncMock(side_effect=RuntimeError("release failed"))),
        ),
    )

    def request(url, *, params, timeout):
        payload = asyncio.run(runner.handle_predict(rollout, **params))
        return Mock(json=lambda: payload)

    monkeypatch.setattr(runner.requests, "get", request)
    with pytest.raises(RuntimeError, match="resume failed.*release failed"):
        runner.run_sft_predict(trainer, 3, published_version={"version": 1, "root": "r"})
    trainer.wake_up.assert_not_called()


@pytest.mark.parametrize("response", ["timeout", "missing_ack"])
def test_unconfirmed_release_keeps_colocated_training_asleep(monkeypatch, trainer, response):
    def request(*args, **kwargs):
        if response == "timeout":
            raise TimeoutError("prediction request timed out")
        return Mock(json=lambda: {"status": "ok"})

    monkeypatch.setattr(runner.requests, "get", request)
    with pytest.raises(RuntimeError):
        runner.run_sft_predict(trainer, 1, published_version={"version": 0, "root": "r"})
    trainer.wake_up.assert_not_called()


@pytest.mark.parametrize("sync", [False, True])
def test_training_barrier_waits_for_offload_retry(monkeypatch, sync):
    from relax.distributed import coordination

    manager_type = pytest.importorskip("relax.distributed.ray.rollout").RolloutManager.__ray_metadata__.modified_class
    manager = object.__new__(manager_type)
    manager.status = "onload"
    manager.health_monitoring_pause = Mock()
    manager.servers = {"engine": SimpleNamespace(offload=Mock(side_effect=[RuntimeError("device busy"), None]))}
    with pytest.raises(RuntimeError, match="device busy"):
        manager._offload_local()
    barrier = coordination.RolloutOffloadBarrier(
        SimpleNamespace(get_status=SimpleNamespace(remote=manager.get_status))
    )
    monkeypatch.setattr(ray, "get", lambda value: value)
    retry = Mock(side_effect=lambda _: manager._offload_local())
    if sync:
        monkeypatch.setattr(coordination.time, "sleep", retry)
        barrier.wait_offloaded_sync()
    else:
        monkeypatch.setattr(coordination.asyncio, "sleep", AsyncMock(side_effect=retry))
        asyncio.run(barrier.wait_offloaded())
    retry.assert_called_once()
    assert manager.status == "offload"


def test_judge_cleanup_failure_still_releases_rollout(monkeypatch, trainer):
    from ray import serve

    args = trainer.args
    args.loss_type, args.rm_type = "sft", "dapo-genrm"
    args._genrm_instances_resolved = {"default": {}}
    release = AsyncMock(side_effect=RuntimeError("judge still resident"))
    monkeypatch.setattr(serve, "get_app_handle", lambda name: SimpleNamespace(offload=SimpleNamespace(remote=release)))
    rollout_release = AsyncMock()
    rollout = SimpleNamespace(
        _logger=Mock(),
        healthy=Mock(),
        config=args,
        rollout_manager=SimpleNamespace(
            run_predict=SimpleNamespace(remote=AsyncMock(side_effect=ValueError("prediction failed"))),
            offload=SimpleNamespace(remote=rollout_release),
        ),
    )

    def request(url, *, params, timeout):
        payload = asyncio.run(runner.handle_predict(rollout, **params))
        return Mock(json=lambda: payload)

    monkeypatch.setattr(runner.requests, "get", request)
    with pytest.raises(RuntimeError, match="prediction failed.*judge still resident"):
        runner.run_sft_predict(trainer, 1, published_version={"version": 0, "root": "r"})
    release.assert_awaited_once()
    rollout_release.assert_awaited_once()
    trainer.wake_up.assert_not_called()


def test_postprocessor_owned_custom_judge_keeps_its_lifecycle():
    from relax.engine.sft.predict.loop import managed_predict_judge

    args = SimpleNamespace(
        offload_rollout=True,
        _genrm_instances_resolved={"default": {}},
        custom_rm_path="custom.score",
        defer_reward_to_post_process=True,
    )
    assert managed_predict_judge(args) is None
