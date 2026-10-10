# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib
import logging
import sys
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace

import pytest


@contextmanager
def stubbed_sglang_engine_module(
    monkeypatch: pytest.MonkeyPatch,
    *,
    server_args: ModuleType,
    router_worker_base_url: Callable,
    extra_env: Mapping[str, object] | None = None,
) -> Iterator[ModuleType]:
    """Import the real engine with local dependency stubs and restore its
    cache."""
    with monkeypatch.context() as patch:
        ray = ModuleType("ray")
        ray.get_runtime_context = lambda: SimpleNamespace()
        patch.setitem(sys.modules, "ray", ray)

        sglang_router = ModuleType("sglang_router")
        sglang_router.__version__ = "0.3.2"
        patch.setitem(sys.modules, "sglang_router", sglang_router)

        patch.setitem(sys.modules, "sglang", ModuleType("sglang"))
        patch.setitem(sys.modules, "sglang.srt", ModuleType("sglang.srt"))
        patch.setitem(sys.modules, "sglang.srt.server_args", server_args)
        sglang_utils = ModuleType("sglang.srt.utils")
        sglang_utils.kill_process_tree = lambda _pid: None
        patch.setitem(sys.modules, "sglang.srt.utils", sglang_utils)

        checkpoint_client = ModuleType("relax.distributed.checkpoint_service.client.engine")
        checkpoint_client.create_client = lambda **_kwargs: None
        patch.setitem(sys.modules, checkpoint_client.__name__, checkpoint_client)

        ray_actor = ModuleType("relax.distributed.ray.ray_actor")
        ray_actor.RayActor = object
        patch.setitem(sys.modules, ray_actor.__name__, ray_actor)

        device = ModuleType("relax.utils.device")
        device.get_visible_devices_env_var = lambda: "CUDA_VISIBLE_DEVICES"
        patch.setitem(sys.modules, device.__name__, device)

        async_utils = ModuleType("relax.utils.async_utils")
        async_utils.run = lambda value: value
        patch.setitem(sys.modules, async_utils.__name__, async_utils)

        env = ModuleType("relax.utils.env")
        env.Envs = SimpleNamespace(
            RELAX_SCALE_OUT_MAX_REASON_ITEMS=3,
            RELAX_SCALE_OUT_MAX_REASON_ITEM_LEN=120,
            RELAX_SCALE_OUT_MAX_REASON_TOTAL_LEN=512,
            **(extra_env or {}),
        )
        patch.setitem(sys.modules, env.__name__, env)

        http_utils = ModuleType("relax.utils.http_utils")
        http_utils.get_host_info = lambda: ("worker", "127.0.0.1")
        http_utils.router_worker_base_url = router_worker_base_url
        http_utils.find_available_port = lambda port: port
        patch.setitem(sys.modules, http_utils.__name__, http_utils)

        logging_utils = ModuleType("relax.utils.logging_utils")
        logging_utils.get_logger = logging.getLogger
        patch.setitem(sys.modules, logging_utils.__name__, logging_utils)

        megatron_peft_utils = ModuleType("relax.utils.megatron_peft_utils")
        megatron_peft_utils.convert_megatron_to_sglang_target_modules = lambda value: value
        megatron_peft_utils.is_lora_enabled = lambda _args: False
        patch.setitem(sys.modules, megatron_peft_utils.__name__, megatron_peft_utils)

        parent = importlib.import_module("relax.backends.sglang")
        module_name = "relax.backends.sglang.sglang_engine"
        # Both import styles must resolve to the fresh module during the test
        # and to their original objects afterward, including on import failure.
        # Record missing entries too: importlib adds them without monkeypatch.
        patch.setitem(sys.modules, module_name, None)
        del sys.modules[module_name]
        patch.setattr(parent, "sglang_engine", None, raising=False)
        delattr(parent, "sglang_engine")
        yield importlib.import_module(module_name)
