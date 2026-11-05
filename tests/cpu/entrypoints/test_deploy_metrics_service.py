# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from unittest.mock import Mock

import pytest

from relax.entrypoints.deploy_metrics_service import create_demo_args
from relax.utils.metrics import service


@pytest.mark.parametrize("enable_wandb", [False, True])
def test_demo_config_initializes_metrics_service(enable_wandb: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    args = create_demo_args()
    if enable_wandb:
        args.use_wandb = True

    # Initialize the replica class beneath Serve's ASGI lifecycle wrapper.
    deployment_class = service.MetricsService.func_or_class
    replica_class = next(cls for cls in deployment_class.__mro__ if "_init_wandb" in cls.__dict__)
    # Patch the replica's bindings so Serve serialization cannot retain the real writers.
    tensorboard_adapter = replica_class.__init__.__globals__["_TensorboardAdapter"]
    monkeypatch.setattr(type(tensorboard_adapter), "_instances", {})
    monkeypatch.setattr(tensorboard_adapter, "_initialize", Mock())
    monkeypatch.setitem(replica_class._init_wandb.__globals__, "wandb", Mock())
    replica = replica_class(healthy=None, pg=None, config=args)

    assert set(replica._adapters) == {"tensorboard"}
    assert replica._timeline_adapter is None
    assert replica._use_wandb is enable_wandb
