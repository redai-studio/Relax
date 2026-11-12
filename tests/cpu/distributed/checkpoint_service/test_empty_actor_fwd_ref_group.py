# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise real DCS group/send methods without importing GPU dependencies."""

import ast
import errno
import logging
import socket
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


SOURCE = Path(__file__).resolve().parents[4] / "relax/distributed/checkpoint_service/backends/device_direct.py"


def load_methods(source=None):
    tree = ast.parse(SOURCE.read_text() if source is None else source)
    backend = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DeviceDirectBackend")
    names = {
        "init_process_groups_for_actor_fwd_ref",
        "_update_bucket_weights_from_distributed_for_actor_fwd_ref",
        "update_weights_for_rollout",
        "_named_params_and_buffers",
        "recv_weight",
    }
    methods = [node for node in backend.body if isinstance(node, ast.FunctionDef) and node.name in names]
    for method in methods:
        method.decorator_list = []  # No CUDA imports; execute the real method bodies.
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *methods],
        type_ignores=[],
    )
    namespace = {"timedelta": timedelta, "logger": logging.getLogger(__name__)}
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)
    return namespace


class EmptyActorFwdRefGroupTest(unittest.TestCase):
    def setUp(self):
        self.methods = load_methods()
        self.new_group = Mock(side_effect=lambda **kwargs: object())
        self.wait = Mock()
        self.broadcast = Mock(return_value=SimpleNamespace(wait=self.wait))
        self.methods.update(init_process_group=self.new_group, dist=SimpleNamespace(broadcast=self.broadcast))
        self.tensor = SimpleNamespace(dtype="torch.float32", shape=(2,), data=object())

    def state(self, role="actor", pp_rank=0):
        return SimpleNamespace(
            role_info=SimpleNamespace(role_name=role),
            backend_type="gloo",
            _megatron=SimpleNamespace(
                mpu=SimpleNamespace(
                    get_data_parallel_rank=lambda **kwargs: 0,
                    get_tensor_model_parallel_rank=lambda: 0,
                    get_pipeline_model_parallel_rank=lambda: pp_rank,
                )
            ),
            _model_update_groups_for_actor_fwd_ref=None,
            _skip_empty_actor_fwd_ref_sync=False,
            coordinator_url="http://coordinator",
            http_client=SimpleNamespace(post=Mock(return_value=SimpleNamespace(raise_for_status=Mock()))),
        )

    def topology(self, world_size, port=12345, global_rank=0, pp_size=1):
        return {
            "world_size": world_size,
            "global_rank": global_rank,
            "pp_groups": {f"update_actor_pp_{rank}": f"tcp://127.0.0.1:{port + rank}" for rank in range(pp_size)},
        }

    def initialize(self, state, topology):
        self.methods["init_process_groups_for_actor_fwd_ref"](state, topology)

    def send(self, state):
        self.methods["_update_bucket_weights_from_distributed_for_actor_fwd_ref"](state, [("weight", self.tensor)])

    def test_empty_target_ignores_occupied_port_and_sends_nothing(self):
        with socket.socket() as occupied:
            occupied.bind(("127.0.0.1", 0))
            occupied.listen(1)
            port = occupied.getsockname()[1]

            def create_store(**kwargs):
                with socket.socket() as store:
                    store.bind(("127.0.0.1", port))

            self.new_group.side_effect = create_store
            state = self.state()
            self.initialize(state, self.topology(1, port))
            self.send(state)
            self.new_group.assert_not_called()
            state.http_client.post.assert_not_called()
            self.broadcast.assert_not_called()
            self.assertIsNone(state._model_update_groups_for_actor_fwd_ref)
            self.assertTrue(state._is_pp_src_rank)

    def test_every_pp_source_skips_an_empty_target(self):
        for rank in (0, 1):
            with self.subTest(pp_rank=rank):
                state = self.state(pp_rank=rank)
                self.initialize(state, self.topology(1, pp_size=2))
                self.send(state)
                state.http_client.post.assert_not_called()
        self.new_group.assert_not_called()
        self.broadcast.assert_not_called()

    def test_reference_only_recipient_still_joins_and_receives_source_broadcast(self):
        actor = self.state()
        reference = self.state(role="reference")
        self.initialize(actor, self.topology(2))
        self.initialize(reference, self.topology(2, global_rank=1))
        self.assertEqual([call.kwargs["rank"] for call in self.new_group.call_args_list], [0, 1])
        self.assertEqual([call.kwargs["world_size"] for call in self.new_group.call_args_list], [2, 2])
        self.send(actor)
        actor.http_client.post.assert_called_once_with(
            "http://coordinator/send_weight_meta",
            json={"names": ["weight"], "dtypes": ["float32"], "shapes": [[2]], "group_name": "update_actor_pp_0"},
        )
        self.broadcast.assert_called_once_with(
            self.tensor.data, 0, group=actor._model_update_groups_for_actor_fwd_ref, async_op=True
        )
        self.wait.assert_called_once()

    def test_nonempty_uninitialized_send_never_uses_default_process_group(self):
        state = self.state()
        with self.assertRaisesRegex(RuntimeError, "not initialized"):
            self.send(state)
        state.http_client.post.assert_not_called()
        self.broadcast.assert_not_called()

    def test_nonempty_store_failure_is_not_hidden(self):
        state = self.state()
        failure = OSError(errno.EADDRINUSE, "port in use")
        self.new_group.side_effect = failure
        with self.assertRaises(OSError) as raised:
            self.initialize(state, self.topology(2))
        self.assertIs(raised.exception, failure)
        with self.assertRaisesRegex(RuntimeError, "not initialized"):
            self.send(state)
        state.http_client.post.assert_not_called()
        self.broadcast.assert_not_called()

    def test_only_explicit_world_size_one_is_treated_as_empty(self):
        state = self.state()
        topology = self.topology(2)
        del topology["world_size"]
        self.initialize(state, topology)
        self.new_group.assert_called_once()
        self.assertFalse(state._skip_empty_actor_fwd_ref_sync)

    def test_real_send_and_receive_metadata_flow_across_empty_reference_steps(self):
        metadata = []
        pending_tensors = []
        loaded = []

        class Dtype:
            def __str__(self):
                return "torch.float32"

        dtype = Dtype()

        class CpuTensor:
            def __init__(self, values):
                self.values = list(values)
                self.data = self
                self.dtype = dtype
                self.shape = (len(values),)

        def response(data=None):
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: data)

        def post(url, json):
            self.assertTrue(url.endswith("/send_weight_meta"))
            metadata.append(json)
            return response()

        def get(url, params=None, **kwargs):
            if url.endswith("/recv_weight_meta"):
                return response(list(metadata[params["index"] :]))
            self.assertTrue(url.endswith("/clear_weight_meta"))
            self.assertEqual(pending_tensors, [], "Receiver stopped before consuming all tensor broadcasts")
            metadata.clear()
            return response()

        def broadcast(tensor, *args, **kwargs):
            self.assertIsNotNone(kwargs["group"])
            if "src" in kwargs:  # Real recv_weight uses the src keyword.
                tensor.values[:] = pending_tensors.pop(0)
            else:
                pending_tensors.append(list(tensor.values))
            return SimpleNamespace(wait=lambda: None)

        barrier = Mock()
        self.methods.update(
            torch=SimpleNamespace(
                dtype=Dtype,
                float32=dtype,
                empty=lambda shape, **kwargs: CpuTensor([None] * shape[0]),
                distributed=SimpleNamespace(broadcast=broadcast),
            ),
            dist=SimpleNamespace(broadcast=broadcast, barrier=barrier, get_rank=lambda: 0),
            httpx=SimpleNamespace(Timeout=lambda **kwargs: None, ReadTimeout=TimeoutError),
            get_gloo_group=lambda: "training-gloo",
            tqdm=lambda **kwargs: SimpleNamespace(update=lambda *args: None),
            device_module=SimpleNamespace(empty_cache=lambda: None),
        )
        actor = self.state()
        actor.http_client = SimpleNamespace(post=post)
        actor.args = SimpleNamespace(num_experts=0)
        actor.model = []
        actor.quantization_config = None
        actor._weights_getter = None
        actor._named_params_and_buffers = lambda **kwargs: self.methods["_named_params_and_buffers"](actor, **kwargs)
        actor.weight_version = 0
        actor._use_bridge = actor._lora_merge_mode = actor._lora_adapter_mode = False
        actor._group_name = "slime-pp_0"
        tensor = CpuTensor([1.25, -3.5])
        actor._megatron.named_params_and_buffers = lambda *args, **kwargs: [("weight", tensor)]
        actor._should_sync_tensor = lambda *args: True
        actor._run_expert_pass = lambda **kwargs: None

        def collect(name, param, converted, origin, size, *flags, **kwargs):
            origin.append((name, param))  # Isolate GPU gather/conversion, not the metadata protocol.
            return size

        actor._update_weight_from_distributed = collect
        actor._update_bucket_weights_from_distributed_for_actor_fwd_ref = lambda tensors: self.methods[
            "_update_bucket_weights_from_distributed_for_actor_fwd_ref"
        ](actor, tensors)
        reference = self.state(role="reference")
        reference.args = SimpleNamespace()
        reference.device = "cpu"
        reference.model = []
        reference.http_client = SimpleNamespace(get=get)
        reference._megatron.load_weight = lambda args, model, weights: loaded.append(
            [(name, list(value.values)) for name, value in weights]
        )

        for size in (1, 2, 1, 2):
            self.initialize(actor, self.topology(size))
            barrier.reset_mock()
            # Exercise the complete sender method and real bucket/stop emission;
            # omit unrelated SGLang rollout transport from this protocol test.
            self.methods["update_weights_for_rollout"](actor, rollout_only=False, actor_fwd_only=True)
            if size == 1:
                self.assertEqual(metadata, [], "Empty sync must emit neither tensor metadata nor stop")
                self.assertEqual(pending_tensors, [])
                self.assertEqual(barrier.call_count, 2)  # No actor_fwd/reference completion barrier.
            else:
                self.assertEqual([item["names"] for item in metadata], [["weight"], ["weight_updated_stop"]])
                self.assertEqual(barrier.call_count, 3)
                self.initialize(reference, self.topology(size, global_rank=1))
                self.methods["recv_weight"](reference)
                self.assertEqual(metadata, [])
                self.assertEqual(pending_tensors, [])
        self.assertEqual(loaded, [[("weight", [1.25, -3.5])], [("weight", [1.25, -3.5])]])
        self.assertEqual(self.new_group.call_count, 2)  # One source and one receiver group, both reused.


if __name__ == "__main__":
    unittest.main()
