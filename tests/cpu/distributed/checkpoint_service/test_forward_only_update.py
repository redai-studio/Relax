# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Forward-actor synchronization can run without a rollout communication
group."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest


pytest.importorskip("megatron.core")

from relax.distributed.checkpoint_service.backends import device_direct
from relax.distributed.checkpoint_service.backends.device_direct import DeviceDirectBackend


def test_forward_only_update_without_rollout_group(monkeypatch):
    backend = object.__new__(DeviceDirectBackend)
    backend.weight_version = 0
    backend.args = SimpleNamespace()
    backend.model = []
    backend.quantization_config = None
    backend._weights_getter = None
    backend._use_bridge = False
    backend._lora_merge_mode = False
    backend._lora_adapter_mode = False
    backend._is_pp_src_rank = True
    backend._megatron = Mock()
    backend._megatron.named_params_and_buffers.return_value = []
    backend._run_expert_pass = Mock()
    backend.http_client = Mock()
    backend.coordinator_url = "http://coordinator"
    backend._batch_request = Mock()
    monkeypatch.setattr(device_direct.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(device_direct.dist, "barrier", Mock())
    monkeypatch.setattr(device_direct, "get_gloo_group", lambda: "training")
    monkeypatch.setattr(device_direct.device_module, "empty_cache", Mock())
    monkeypatch.setattr(device_direct, "tqdm", Mock())

    backend.update_weights_for_rollout(actor_fwd_only=True)

    assert backend.weight_version == 1
    backend._run_expert_pass.assert_called_once_with(
        rollout_only=False, actor_fwd_only=True, pbar=device_direct.tqdm.return_value
    )
    backend.http_client.post.assert_called_once_with(
        "http://coordinator/send_weight_meta",
        json={"names": ["weight_updated_stop"], "dtypes": [], "shapes": [], "group_name": "end", "update_id": ""},
    )
    backend._batch_request.assert_not_called()
