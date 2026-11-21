# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Two training participants agree on FULL publication and local I/O
failures."""

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank, init_file, root):
    from relax.backends.megatron import weight_update
    from relax.backends.megatron.weight_update import hf_weight_iterator_base
    from relax.backends.megatron.weight_update.export_full_checkpoint import export_full_checkpoint
    from relax.distributed.weight_sync.checkpoint import write_checkpoint
    from relax.distributed.weight_sync.store import VersionStore

    dist.init_process_group("gloo", init_method="file://" + init_file, rank=rank, world_size=2)
    values = {"w": torch.arange(16, dtype=torch.bfloat16).reshape(4, 4)}
    args = SimpleNamespace(megatron_to_hf_mode="raw", model_name="test")

    # The exporter deliberately imports Megatron-dependent helpers lazily.
    # Install a lightweight module stub before patch resolution so this CPU
    # test does not depend on Megatron-Core merely to inject its source tensors.
    common = ModuleType("relax.backends.megatron.weight_update.common")
    common.named_params_and_buffers = lambda *args, **kwargs: values.items()
    sys.modules[common.__name__] = common
    weight_update.common = common
    actor = SimpleNamespace(args=args, model=[], hf_config=SimpleNamespace(to_dict=lambda: {"model_type": "test"}))

    class Iterator:
        def get_hf_weight_chunks_checked(self, weights, agreement):
            agreement(None)
            yield list(weights.items())

    options = dict(
        store_path=Path(root) / "store",
        stream="weights",
        scratch=Path(root) / f"scratch-{rank}",
        disk_budget=1024**2,
        max_weight_bytes=4096,
    )
    try:
        # Adapter-only synchronization cannot produce an independent FULL
        # checkpoint. Reject it before any iterator/store work, including when
        # only one participant has the unsupported configuration.
        args.megatron_to_hf_mode = "bridge"
        args.lora_rank = 2
        for adapter_mode in (True, rank == 0):
            args.lora_adapter_mode = adapter_mode
            with patch.object(
                hf_weight_iterator_base.HfWeightIteratorBase,
                "create",
                side_effect=AssertionError("adapter mode must fail before iterator construction"),
            ):
                with pytest.raises(RuntimeError, match="does not support LoRA adapter mode"):
                    export_full_checkpoint(actor, dist.group.WORLD, version=0, **options)
            assert not options["store_path"].exists()
            assert not options["scratch"].exists()
        args.megatron_to_hf_mode = "raw"
        args.lora_rank = 0
        args.lora_adapter_mode = False

        with patch.object(hf_weight_iterator_base.HfWeightIteratorBase, "create", return_value=Iterator()):
            first = export_full_checkpoint(actor, dist.group.WORLD, version=0, **options)

            def fail_create(*args, **kwargs):
                if rank == 0:
                    raise RuntimeError("injected iterator initialization failure")
                return Iterator()

            with patch.object(hf_weight_iterator_base.HfWeightIteratorBase, "create", side_effect=fail_create):
                with pytest.raises(RuntimeError, match="injected iterator initialization failure"):
                    export_full_checkpoint(actor, dist.group.WORLD, version=1, **options)

            def fail_write(*a, **kw):
                raise OSError("injected FULL source write failure")

            with patch(
                "relax.backends.megatron.weight_update.export_full_checkpoint.snapshot_tensors", side_effect=fail_write
            ):
                with pytest.raises(RuntimeError, match="injected FULL source write failure"):
                    export_full_checkpoint(actor, dist.group.WORLD, version=1, **options)
            values["w"].add_(1)
            second = export_full_checkpoint(actor, dist.group.WORLD, version=1, **options)
            assert first["root"] != second["root"]
            if rank == 0:
                from safetensors.torch import load_file

                store = VersionStore(options["store_path"], disk_budget=1024**2, max_weight_bytes=4096)
                assert store.manifest("weights", 1)["kind"] == "full"
                target = write_checkpoint(store, "weights", 1, Path(root) / "checkpoint")
                assert torch.equal(
                    load_file(target / "model.safetensors")["w"].view(torch.uint8), values["w"].view(torch.uint8)
                )
            with pytest.raises(ValueError, match="different FULL exports"):
                export_full_checkpoint(actor, dist.group.WORLD, version=rank + 2, **options)

            actor.hf_config.to_dict = lambda: {"model_type": "test", "rank": rank}
            with pytest.raises(ValueError, match="different FULL exports"):
                export_full_checkpoint(actor, dist.group.WORLD, version=3, **options)
            actor.hf_config.to_dict = lambda: {"model_type": "test"}
            args.megatron_to_hf_mode = "raw" if rank == 0 else "bridge"
            with pytest.raises(ValueError, match="different FULL exports"):
                export_full_checkpoint(actor, dist.group.WORLD, version=4, **options)
            args.megatron_to_hf_mode = "raw"

            # Converter code identity must be checked before iterator setup can
            # enter model-parallel collectives.
            import relax.backends.megatron.weight_update.export_full_checkpoint as exporter
            from relax.backends.megatron.weight_update.hf_weight_iterator_base import HfWeightIteratorBase

            original_revision = exporter._converter_revision
            exporter._converter_revision = lambda: f"rank-{rank}-converter"
            try:
                with patch.object(
                    HfWeightIteratorBase,
                    "create",
                    side_effect=AssertionError("iterator setup must not run on converter mismatch"),
                ):
                    with pytest.raises(ValueError, match="different weight converters"):
                        export_full_checkpoint(actor, dist.group.WORLD, version=5, **options)
            finally:
                exporter._converter_revision = original_revision

            # Exercise the real Direct iterator constructor: rank-local metadata
            # discovery must agree before its PP all-gather, not only after the
            # constructor returns to the FULL exporter.
            import types

            megatron = types.ModuleType("megatron")
            core = types.ModuleType("megatron.core")
            mpu = types.ModuleType("megatron.core.mpu")
            mpu.get_pipeline_model_parallel_world_size = lambda: 2
            mpu.get_expert_model_parallel_world_size = lambda: 1
            mpu.get_pipeline_model_parallel_group = lambda: dist.group.WORLD
            core.mpu = mpu
            megatron.core = core
            sys.modules.update({"megatron": megatron, "megatron.core": core, "megatron.core.mpu": mpu})
            sglang = types.ModuleType("relax.backends.megatron.sglang")
            sglang.monkey_patch_torch_reductions = lambda: None
            conversion = types.ModuleType("relax.backends.megatron.weight_conversion")
            conversion.convert_to_hf = lambda *args, **kwargs: []
            sys.modules[sglang.__name__] = sglang
            sys.modules[conversion.__name__] = conversion
            common.all_gather_params_async = lambda *args, **kwargs: []
            module = __import__(
                "relax.backends.megatron.weight_update.hf_weight_iterator_direct",
                fromlist=["HfWeightIteratorDirect"],
            )
            module.get_gloo_group = lambda: dist.group.WORLD

            from relax.distributed.weight_sync.source_io import agree_errors

            def agree_metadata(error):
                failures = agree_errors(error, dist.group.WORLD)
                if any(failures):
                    raise RuntimeError(f"Direct metadata initialization failed: {failures}")

            def local_metadata(*args, **kwargs):
                if rank == 0:
                    raise RuntimeError("injected Direct metadata discovery failure")
                return values.items()

            module.named_params_and_buffers = local_metadata
            with pytest.raises(RuntimeError, match="injected Direct metadata discovery failure"):
                module.HfWeightIteratorDirect(
                    args=SimpleNamespace(update_weight_buffer_size=4096, mtp_num_layers=None),
                    model=[],
                    model_name="test",
                    quantization_config=None,
                    publication_agreement=agree_metadata,
                )

            # Identical names/shapes/dtypes are not sufficient: ranks must also
            # agree on TP layout metadata before the subsequent gather schedule.
            mpu.get_pipeline_model_parallel_world_size = lambda: 1
            mpu.get_expert_model_parallel_world_size = lambda: 1
            layout_param = torch.nn.Parameter(torch.ones(4, 4))
            layout_param.tensor_model_parallel = bool(rank)
            module.named_params_and_buffers = lambda *args, **kwargs: [("weight", layout_param)]
            with pytest.raises(RuntimeError, match="Direct metadata initialization failed"):
                module._get_megatron_local_param_infos(
                    SimpleNamespace(mtp_num_layers=None), [], agreement=agree_metadata
                )
    finally:
        dist.destroy_process_group()


def test_two_rank_full_export_write_failure_retry_and_identity(tmp_path):
    mp.spawn(_worker, args=(str(tmp_path / "init"), str(tmp_path)), nprocs=2, join=True)
