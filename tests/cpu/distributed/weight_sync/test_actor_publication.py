"""Exercise publication routing without starting the distributed actor
runtime."""

import inspect
import sys
import tempfile
import unittest
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from relax.distributed.weight_sync.config import validate_weight_version_arguments
from relax.distributed.weight_sync.store import canonical


def actor_method(name, **globals_):
    pytest.importorskip("megatron.core")
    try:
        from relax.backends.megatron import actor as actor_module
    except (ImportError, AssertionError) as error:
        pytest.skip(f"Megatron actor dependencies unavailable: {error}")
    method = inspect.unwrap(getattr(actor_module.MegatronTrainRayActor, name))

    def invoke(*args, **kwargs):
        with patch.dict(actor_module.__dict__, globals_):
            return method(*args, **kwargs)

    return invoke


class PublicationRoutingTests(unittest.TestCase):
    def test_initial_publication_protects_setup_and_releases_offload_context(self):
        @dataclass
        class Published:
            version: int

        for failure in (None, "setup", "publish"):
            with self.subTest(failure=failure):
                events = []
                active = False

                @contextmanager
                def protected():
                    nonlocal active
                    active = True
                    events.append("enter")
                    try:
                        yield
                    finally:
                        active = False
                        events.append("exit")

                def publish(*args, **kwargs):
                    self.assertTrue(active, "publication allocates temporary device buffers")
                    events.append("publish")
                    if failure == "publish":
                        raise RuntimeError("publication failure")
                    return Published(0)

                def create(*args):
                    self.assertTrue(active, "Bridge setup can allocate device tensors while training is offloaded")
                    events.append("setup")
                    if failure == "setup":
                        raise RuntimeError("setup failure")
                    return SimpleNamespace(publish=publish)

                module = ModuleType("relax.backends.megatron.weight_update.version_integration")
                module.create_version_publisher = create
                actor = SimpleNamespace(
                    args=SimpleNamespace(offload_train=True, distributed_timeout_minutes=3),
                    _train_state_offloader=SimpleNamespace(disable_during_update=protected),
                    _backup_published_actor=lambda: events.append("backup"),
                )
                method = actor_method(
                    "publish_weight_version",
                    __package__="relax.backends.megatron",
                    reload_process_groups=lambda **kwargs: events.append("reload"),
                    destroy_process_groups=lambda: events.append("destroy"),
                    get_gloo_group=lambda: "source-group",
                )
                with patch.dict(sys.modules, {module.__name__: module}):
                    if failure:
                        with self.assertRaisesRegex(RuntimeError, "failure"):
                            method(actor, "capture-0")
                    else:
                        self.assertEqual(method(actor, "capture-0"), {"version": 0})
                self.assertFalse(active)
                self.assertEqual(events[:3], ["reload", "enter", "setup"])
                self.assertEqual(events[-2:], ["exit", "destroy"])
                self.assertEqual(hasattr(actor, "_version_publisher"), failure != "setup")

    def test_previous_actor_queue_keeps_the_existing_training_contract(self):
        actor = SimpleNamespace(
            args=SimpleNamespace(keep_old_actor=True, update_weights_interval=1), weights_backuper=Mock()
        )
        backup = actor_method("_backup_published_actor")
        backup(actor)
        actor.weights_backuper.copy.assert_called_once_with(src_tag="rollout_actor", dst_tag="old_actor")
        actor.weights_backuper.backup.assert_called_once_with("rollout_actor")
        actor.weights_backuper.reset_mock()
        actor.args.update_weights_interval = 2
        backup(actor)
        actor.weights_backuper.copy.assert_not_called()
        actor.weights_backuper.backup.assert_called_once_with("old_actor")

    def test_offline_consumers_do_not_suppress_async_publication(self):
        actor = SimpleNamespace(
            args=SimpleNamespace(weight_version_config="publisher.json"),
            publish_weight_version=Mock(return_value={"version": 9}),
        )
        update = actor_method("update_weights_fully_async")
        self.assertEqual(update(actor, 9, rollout_only=True, actor_fwd_only=True), {"version": 9})
        actor.publish_weight_version.assert_called_once_with()

    def test_independent_publisher_does_not_wait_for_inference_evaluation(self):
        actor = SimpleNamespace(args=SimpleNamespace(weight_version_config="publisher.json"))
        wait = actor_method("_wait_for_previous_eval")
        # No HTTP, Ray or process-group symbols are supplied: this route must
        # finish without reaching any inference coordination operation.
        self.assertIsNone(wait(actor, before_publication=True))

    def test_final_async_evaluation_finishes_before_shutdown(self):
        actor = SimpleNamespace(args=SimpleNamespace(weight_version_config="publisher.json"))
        requests = Mock()
        requests.get.return_value.json.return_value = {"done": True}
        dist = Mock()
        dist.get_rank.return_value = 0
        wait = actor_method(
            "_wait_for_previous_eval",
            requests=requests,
            dist=dist,
            time=SimpleNamespace(monotonic=lambda: 0),
            get_serve_url=lambda role: "http://rollout",
            get_gloo_group=lambda: "training-group",
        )
        wait(actor)
        requests.get.assert_called_once_with("http://rollout/is_eval_done", timeout=10)
        dist.barrier.assert_called_once_with(group="training-group")

    def test_initial_async_publication_without_forward_recipient(self):
        update = actor_method("update_weights_fully_async")
        for mode in ("true_on_policy_mode", "hybrid"):
            with self.subTest(mode=mode):
                actor = SimpleNamespace(
                    args=SimpleNamespace(weight_version_config="publisher.json", **{mode: True}),
                    publish_weight_version=Mock(return_value={"version": 0}),
                )
                # Controller initialization supplies neither routing flag.
                # It must not enter DCS when there is no forward recipient.
                self.assertEqual(update(actor, 0), {"version": 0})
                actor.publish_weight_version.assert_called_once_with()

    def test_async_publication_retains_training_forward_sync(self):
        client = Mock()
        client.update_weights_for_rollout.return_value = []
        actor = SimpleNamespace(
            args=SimpleNamespace(weight_version_config="publisher.json"),
            publish_weight_version=Mock(return_value={"version": 3}),
            checkpoint_engine_client=client,
        )
        update = actor_method(
            "update_weights_fully_async",
            dist=Mock(),
            get_gloo_group=lambda: "training-group",
            print_memory=Mock(),
            run=lambda result: result,
        )
        update(actor, 3)
        actor.publish_weight_version.assert_called_once_with()
        client.init_process_groups_for_actor_fwd_ref.assert_called_once_with(3)
        client.update_weights_for_rollout.assert_called_once_with(False, True)

    def test_reference_updates_without_a_forward_actor(self):
        client = Mock()
        client.update_weights_for_rollout.return_value = []
        actor = SimpleNamespace(
            args=SimpleNamespace(
                weight_version_config="publisher.json",
                true_on_policy_mode=True,
                ref_update_interval=2,
                resource={"actor": [1, 1], "rollout": [1, 1], "reference": [1, 1]},
            ),
            publish_weight_version=Mock(return_value={"version": 1}),
            checkpoint_engine_client=client,
        )
        update = actor_method(
            "update_weights_fully_async",
            dist=Mock(),
            get_gloo_group=lambda: "training-group",
            print_memory=Mock(),
            run=lambda result: result,
        )
        update(actor, 0, rollout_only=True)
        client.init_process_groups_for_actor_fwd_ref.assert_not_called()
        update(actor, 1, rollout_only=True)
        client.init_process_groups_for_actor_fwd_ref.assert_called_once_with(1)
        client.update_weights_for_rollout.assert_called_once_with(False, True)

    def test_managed_rollouts_require_a_consumer_and_matching_stream(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            publication, consumer = root / "source.json", root / "consumer.json"
            publication.write_bytes(
                canonical(
                    {
                        "store": "source",
                        "stream": "run",
                        "scratch": "scratch",
                        "disk_budget": 1024,
                        "max_weight_bytes": 512,
                    }
                )
            )
            args = SimpleNamespace(
                weight_version_config=str(publication),
                weight_consumer_config=None,
                train_backend="megatron",
                debug_train_only=False,
            )
            with self.assertRaisesRegex(ValueError, "managed rollout"):
                validate_weight_version_arguments(args, offline_training=False)
            validate_weight_version_arguments(args, offline_training=True)
            args.debug_train_only = True
            validate_weight_version_arguments(args, offline_training=False)
            args.debug_train_only = False
            args.weight_consumer_config = str(consumer)
            consumer.write_bytes(canonical({"stream": "another"}))
            with self.assertRaisesRegex(ValueError, "same weight stream"):
                validate_weight_version_arguments(args, offline_training=False)
            consumer.write_bytes(canonical({"stream": "run"}))
            validate_weight_version_arguments(args, offline_training=False)


if __name__ == "__main__":
    unittest.main()
