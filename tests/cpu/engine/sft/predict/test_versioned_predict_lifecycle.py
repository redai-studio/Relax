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
def test_managed_judge_scores_before_training_wakes(monkeypatch, trainer, deferred):
    trainer.args.rm_type = "dapo-genrm"
    trainer.args.defer_reward_to_post_process = deferred
    events = []
    active = deferred  # A custom post-processor owns a deferred judge.

    def change(ready):
        nonlocal active
        active = ready
        events.append("onload" if ready else "offload")

    judge = SimpleNamespace(
        onload=SimpleNamespace(remote=lambda: change(True)), offload=SimpleNamespace(remote=lambda: change(False))
    )
    trainer.genrm_manager = [judge]
    monkeypatch.setattr(ray, "get", lambda refs: refs)

    async def generate(_):
        if not active:
            raise httpx.ConnectError("judge is offloaded")
        return "Judgement: 1"

    monkeypatch.setattr(dapo_genrm, "get_genrm_client", lambda: SimpleNamespace(generate=generate))
    scores = []

    def request(url, *, params, timeout):
        sample = SimpleNamespace(metadata={"question": "1+1?", "label": "2"}, response="Answer: 2")
        result = asyncio.run(dapo_genrm.async_compute_score_genrm(trainer.args, sample))
        scores.append(result["score"])
        events.append("score")
        return Mock(json=lambda: {"status": "ok"})

    monkeypatch.setattr(runner.requests, "get", request)
    trainer.wake_up.side_effect = lambda: events.append("wake")
    runner.run_sft_predict(trainer, 3, published_version={"version": 1, "root": "r"})
    assert scores == [1.0]
    assert events == (["score", "wake"] if deferred else ["onload", "score", "offload", "wake"])
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
    trainer.wake_up.assert_called_once()
