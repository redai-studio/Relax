import json
from types import SimpleNamespace

import torch
import torch.distributed as dist
from experiments.evidence import capture_rng, payload_digest
from experiments.native_runner import (
    CacheAudit,
    LocalActor,
    initial_state,
    native_args,
    restore_initial,
)

from relax.models.condition_cache import ConditionFeatureCache


def test_cache_audit_checks_actual_fresh_read_and_stays_bounded():
    cache = ConditionFeatureCache(4096, max_entries=2)
    audit = CacheAudit(cache)
    cache.put("one", torch.tensor([1.0]))
    cache.put("two", torch.tensor([2.0]))
    assert torch.equal(cache.get("one"), torch.tensor([1.0]))
    cache.put("three", torch.tensor([3.0]))
    assert len(audit.expected) <= 2
    assert audit.result()["checked_hits"] == 1
    cache.close()
    assert cache.stats()["bytes"] == 0


def test_initial_restore_replays_all_rng_and_fp32_parameters():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    trainer = SimpleNamespace(scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0), step=0)
    state = initial_state(model, optimizer, trainer, {"fixed": True})
    model(torch.randn(1, 2)).sum().backward()
    optimizer.step()
    assert optimizer.state
    restore_initial(state, model, optimizer, trainer, {"fixed": True})
    assert not optimizer.state
    assert payload_digest(capture_rng()) == payload_digest(state["rng"])
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, state["parameters"][name])


def test_actual_native_trainer_constructs_and_updates_cpu_dtensor_without_ray(tmp_path, monkeypatch):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Replicate, Shard, distribute_tensor

    import relax.backends.fsdp.sft as native
    import relax.utils.distributed_utils as distributed_utils
    from relax.backends.fsdp.sft import FSDPSFTTrainer
    from relax.utils.distributed_utils import init_gloo_group

    metadata = tmp_path / "rows.json"
    metadata.write_text(json.dumps([{"prompt": "edit", "image": "a.png", "edit_image": ["b.png"]}] * 8))
    options = SimpleNamespace(
        seed=20261030,
        adapter="test.Adapter",
        model_path=str(tmp_path),
        metadata=str(metadata),
        data_root=str(tmp_path),
        max_pixels=1048576,
        mode="correctness",
        warmup=16,
        measure=64,
    )
    args = native_args(options, 8)

    class Adapter:
        family = "cpu_contract_only"
        lora_target_modules = ["linear"]
        lora_task_type = "FEATURE_EXTRACTION"
        ready = False

        def on_device_ready(self, _model, device):
            assert device.type == "cpu"
            self.ready = True

        def prepare_sample(self, row, root, config, index=0):
            return {"index": index, "x": torch.tensor([float(index + 1)])}

    dist.init_process_group("gloo", init_method=(tmp_path / "rendezvous").as_uri(), rank=0, world_size=1)
    try:
        control = init_gloo_group()
        mesh = init_device_mesh("cpu", (1,))

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(distribute_tensor(torch.ones(1), mesh, [Shard(0)]))

            def forward(self, data):
                value = distribute_tensor(data["x"], mesh, [Replicate()])
                return (self.weight * value).square().sum().full_tensor()

        model = Model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        adapter = Adapter()
        actor = LocalActor(args, adapter, model, optimizer, dist.group.WORLD, control, device=torch.device("cpu"))
        trainer = FSDPSFTTrainer(actor)
        actor.trainer = trainer
        assert adapter.ready and trainer.sampler.batch(0) == actor._read_train_partition(0)["sample_id"].tolist()
        assert trainer.dataset[0]["index"] == 0
        assert trainer.contract["master_dtype"] == "fp32"
        values = []
        monkeypatch.setattr(native, "_report_sft_metrics", lambda _args, data: values.append(data))
        monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _device: 0)
        monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _device: 0)
        trainer.train(0)
        assert trainer.step == 1 and len(values) == 1
        assert values[0]["train/loss"] == sum(i * i for i in range(1, 9)) / 8
        assert model.weight.to_local().item() < 1.0
        assert optimizer.state
        trainer.close()
    finally:
        dist.destroy_process_group()
        distributed_utils.GLOO_GROUP = None
