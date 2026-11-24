# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Owned local Ray runtimes for tests that exercise real reward actors."""

import tempfile
import uuid
from collections.abc import Iterator

import pytest


@pytest.fixture(scope="module")
def reward_ray_cluster() -> Iterator[None]:
    import ray

    if ray.is_initialized():
        pytest.fail("Reward tests must not reuse or shut down another test's Ray runtime")
    # A short, unique runtime path also isolates Ray's sockets and discovery files.
    with tempfile.TemporaryDirectory(prefix="reward-ray-", dir="/tmp") as runtime_dir:
        try:
            ray.init(
                address="local",
                namespace=f"reward-tests-{uuid.uuid4().hex}",
                _temp_dir=runtime_dir,
                num_cpus=8,
                num_gpus=0,
                include_dashboard=False,
                object_store_memory=256 * 1024 * 1024,
            )
            yield
        finally:
            ray.shutdown()
