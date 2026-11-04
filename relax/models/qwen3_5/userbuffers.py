# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Initialize fixed-shape communication buffers once per training process."""

_KEEPALIVE = None
CAPACITIES = (253952, 258048, 262144)


def initialize():
    global _KEEPALIVE
    import torch
    import torch.distributed as dist
    from transformer_engine.pytorch.module import base

    tex = base.tex
    assert _KEEPALIVE is None
    assert base._ub_communicators is None and not base.is_ub_initialized()
    assert tex.device_supports_multicast() and not tex.ubuf_built_with_mpi()
    assert dist.get_world_size(dist.group.WORLD) in (8, 64)
    world = dist.new_group(backend="gloo")
    size = dist.get_world_size(world)
    domain, _ = dist.new_subgroups_by_enumeration([list(range(i, i + 4)) for i in range(0, size, 4)], backend="gloo")
    helper = tex.CommOverlapHelper(world, domain)
    low, high = tex.get_stream_priority_range()
    base._MIN_STREAM_PRIORITY, base._MAX_STREAM_PRIORITY = low, high
    registry = {}
    objects = []
    for rows in CAPACITIES:
        common = dict(
            use_cublasmp=False,
            num_max_streams=base._NUM_MAX_UB_STREAMS,
            atomic_gemm=False,
            gemm_priority=low,
            comm_priority=high,
        )

        def ring(kind):
            return tex.CommOverlapP2P(
                [rows, 2048],
                torch.bfloat16,
                helper,
                4,
                kind,
                comm_cga_size=1,
                num_comm_sm=1,
                set_sm_margin=False,
                use_ce=True,
                aggregate=False,
                **common,
            )

        ag_fprop = ring(tex.CommOverlapType.AG)
        rs_dgrad = ring(tex.CommOverlapType.RS)
        ag_dgrad = ring(tex.CommOverlapType.AG)
        rs_fprop = tex.CommOverlap(
            [rows, 2048],
            torch.bfloat16,
            helper,
            4,
            comm_type=tex.CommOverlapType.RS,
            num_splits=4,
            comm_cga_size=2,
            num_comm_sm=16,
            set_sm_margin=True,
            rs_overlap_first_gemm=False,
            **common,
        )
        objects.extend([ag_fprop, rs_dgrad, ag_dgrad, rs_fprop])
        for name in ("fc1", "qkv"):
            registry[(f"{name}_s{rows}_fprop", base.UserBufferQuantizationMode.NONE)] = ag_fprop
            registry[(f"{name}_s{rows}_dgrad", base.UserBufferQuantizationMode.NONE)] = rs_dgrad
        for name in ("fc2", "proj"):
            registry[(f"{name}_s{rows}_fprop", base.UserBufferQuantizationMode.NONE)] = rs_fprop
            registry[(f"{name}_s{rows}_dgrad", base.UserBufferQuantizationMode.NONE)] = ag_dgrad
    assert len(registry) == 24 and len({id(v) for v in registry.values()}) == 12
    base._ub_communicators = registry
    base._ub_with_cublasmp = False
    base.layers_atomic_ring_exchange = []
    base._ub_initialized = True
    _KEEPALIVE = (world, domain, helper, objects)
    return dict(
        capacities=list(CAPACITIES),
        registry_keys=len(registry),
        unique_objects=12,
        tp_ranks=dist.get_process_group_ranks(domain),
        bootstrap_count=1,
        padding=False,
        bulk=False,
        quantization="BF16 only",
    )


def close():
    global _KEEPALIVE
    from transformer_engine.pytorch.module import base

    assert _KEEPALIVE is not None
    world, domain, helper, objects = _KEEPALIVE
    base.destroy_ub()
    objects.clear()
    _KEEPALIVE = None
    # Keep bootstrap groups/helper alive until all distinct UB object references are gone.
    del helper, domain, world
