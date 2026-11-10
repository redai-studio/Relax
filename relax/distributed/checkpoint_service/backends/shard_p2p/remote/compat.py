# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Narrow Qwen3.5 BF16 replica adapter for SGLang's published runtime context.

No process groups are created: a replica only constructs parameters and runs
loaders. Any attempted communication is an error. The real Megatron process
groups stay intact.
"""

import inspect
import time
from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock
from types import SimpleNamespace
from typing import Any, Iterator

import torch


_LAYOUT_LOCK = RLock()


@dataclass(frozen=True)
class RankConfig:
    parallelism: tuple[int, int]
    server_args: Any
    runner_role: str = "target"


class _LayoutGroup(SimpleNamespace):
    def __getattr__(self, name: str) -> Any:
        raise RuntimeError(f"A layout-only SGLang replica cannot use group operation {name}")


class _LayoutProcessGroup:
    def __getattr__(self, name: str) -> Any:
        raise RuntimeError(f"A layout-only SGLang replica cannot use process-group operation {name}")


def _group(rank: int, size: int) -> _LayoutGroup:
    return _LayoutGroup(
        rank=rank,
        rank_in_group=rank,
        world_size=size,
        ranks=list(range(size)),
        is_first_rank=rank == 0,
        is_last_rank=rank == size - 1,
        cpu_group=_LayoutProcessGroup(),
        device_group=_LayoutProcessGroup(),
    )


@contextmanager
def _legacy_groups(values: dict) -> Iterator[None]:
    from importlib import import_module

    state = import_module("sglang.srt.distributed.parallel_state")
    attention = import_module("sglang.srt.layers.dp_attention")
    if getattr(state, "_ENABLE_PDMUX_P_TP", False):
        raise NotImplementedError("Layout-only replicas do not support PD-multiplexed tensor parallel groups")
    groups = {
        "_WORLD": "world",
        "_TP": "tp",
        "_PP": "pp",
        "_ATTN_TP": "attn_tp",
        "_ATTN_CP": "attn_cp",
        "_DCP": "dcp",
        "_MOE_TP": "moe_tp",
        "_MOE_EP": "moe_ep",
        "_MOE_DP": "moe_dp",
    }
    replacements = (
        (state, {name: values[group + "_group"] for name, group in groups.items()}),
        (attention, {"_ATTN_DP_RANK": values["attn_dp_rank"], "_ATTN_DP_SIZE": values["attn_dp_size"]}),
    )
    saved = []
    try:
        for module, fields in replacements:
            for name, value in fields.items():
                if hasattr(module, name):
                    saved.append((module, name, getattr(module, name)))
                    setattr(module, name, value)
        yield
    finally:
        for module, name, value in reversed(saved):
            setattr(module, name, value)


def _validate_parallelism(layout: tuple[int, int]) -> None:
    rank, size = layout
    if type(size) is not int or size <= 0 or type(rank) is not int or not 0 <= rank < size:
        raise NotImplementedError("Replica TP requires valid integer rank coordinates")


@contextmanager
def ParallelismContext(layout: tuple[int, int]) -> Iterator[None]:
    rank, size = layout
    _validate_parallelism(layout)
    from sglang.srt.runtime_context import _PARALLEL_FIELDS, get_parallel

    values = dict(
        world_rank=rank,
        world_size=size,
        tp_rank=rank,
        tp_size=size,
        pp_rank=0,
        pp_size=1,
        dp_rank=None,
        dp_size=1,
        attn_tp_rank=rank,
        attn_tp_size=size,
        attn_dp_rank=0,
        attn_dp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        dcp_rank=0,
        dcp_size=1,
        dcp_enabled=False,
        attn_dcp_rank=0,
        attn_dcp_size=1,
        moe_tp_rank=rank,
        moe_tp_size=size,
        moe_ep_rank=0,
        moe_ep_size=1,
        moe_dp_rank=0,
        moe_dp_size=1,
        launch_world_rank=rank,
        launch_world_size=size,
    )
    for name, group_rank, group_size in (
        ("world", rank, size),
        ("tp", rank, size),
        ("pp", 0, 1),
        ("attn_tp", rank, size),
        ("attn_cp", 0, 1),
        ("dcp", 0, 1),
        ("moe_tp", rank, size),
        ("moe_ep", 0, 1),
        ("moe_dp", 0, 1),
        ("shared_experts_tp", rank, size),
    ):
        values[f"{name}_group"] = _group(group_rank, group_size)
    # Runtime-context fields differ between supported SGLang images. The
    # published override capability is authoritative; never inherit a missing
    # core coordinate from the trainer's live process groups.
    optional = {
        "dp_rank",
        "dp_size",
        "launch_world_rank",
        "launch_world_size",
        "world_rank",
        "world_size",
        "shared_experts_tp_group",
        "dcp_enabled",
    }
    if (
        not isinstance(_PARALLEL_FIELDS, (set, frozenset))
        or (set(values) - optional) - _PARALLEL_FIELDS
        or not {"world_rank", "launch_world_rank"} & _PARALLEL_FIELDS
        or not {"world_size", "launch_world_size"} & _PARALLEL_FIELDS
    ):
        raise NotImplementedError("SGLang runtime context cannot represent the complete target layout")
    # Some installed models still import the legacy getters directly. Both
    # views must describe the same layout; neither may use a trainer group.
    with (
        _LAYOUT_LOCK,
        get_parallel().override(**{name: value for name, value in values.items() if name in _PARALLEL_FIELDS}),
        _legacy_groups(values),
    ):
        yield


def post_load_weights(model: torch.nn.Module) -> None:
    """Run the installed SGLang loader's model-specific post-load hook."""
    from importlib import import_module

    loader = import_module("sglang.srt.model_loader.loader")
    # Older images expose the same native model-fixup hook privately. Keep the
    # native implementation rather than silently omitting derived-cache work.
    process = getattr(loader, "post_load_weights", None) or getattr(loader, "_post_load_weights", None)
    if not callable(process):
        raise NotImplementedError("SGLang loader does not provide a model post-load hook")
    process(model)


def publish_server_args(server_args: Any) -> None:
    from sglang.srt.layers.moe import initialize_moe_config
    from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
    from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
    from sglang.srt.server_args import set_global_server_args_for_scheduler

    set_global_server_args_for_scheduler(server_args)
    for initialize in (initialize_moe_config, initialize_fp8_gemm_config, initialize_fp4_gemm_config):
        signature = inspect.signature(initialize)
        try:
            signature.bind()
        except TypeError:
            signature.bind(server_args)
            initialize(server_args)
        else:
            initialize()


def build_replica(server_fields: dict, tp_rank: int, *, transfer_buffer_device: torch.device | None = None) -> Any:
    from sglang.srt.configs.load_config import LoadConfig
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.model_loader.loader import _initialize_model
    from sglang.srt.server_args import ServerArgs

    from .model_replica import ModelReplica

    args = ServerArgs(**server_fields)
    _validate_parallelism((tp_rank, args.tp_size))
    if args.pp_size != 1 or args.quantization is not None:
        raise NotImplementedError("Variable replica TP requires unquantized expert shard transfer with PP1")
    publish_server_args(args)
    config = RankConfig((tp_rank, args.tp_size), args)
    model_config = ModelConfig.from_server_args(args)
    from relax.backends.sglang.shard_p2p_reload import reload_adapter

    adapter = reload_adapter(model_config)
    started = time.monotonic()
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(model_config.dtype)
        with ParallelismContext(config.parallelism), torch.device("cpu"):
            model = _initialize_model(model_config, LoadConfig(), quant_config=None)
    finally:
        torch.set_default_dtype(old_dtype)
    construction_seconds = time.monotonic() - started
    started = time.monotonic()
    shapes = {name: param.shape for name, param in model.named_parameters()}
    replica = ModelReplica(
        model,
        shapes,
        config,
        postprocess_device=torch.device("cuda", torch.cuda.current_device()),
        transfer_buffer_device=transfer_buffer_device or torch.device("cpu"),
        reload_adapter=adapter,
    )
    replica.cold_profile = dict(
        cpu_model_construction_s=construction_seconds, model_postprocess_s=time.monotonic() - started
    )
    return replica
