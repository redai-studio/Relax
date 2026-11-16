# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
import sys
from types import SimpleNamespace

import torch

from relax.utils import record_function_tags as tags


def test_profile_module_ranges_preserve_autograd_and_restore_owner(monkeypatch):
    """Profiling shared/MoK modules must not insert graph nodes or alter
    gradients."""
    monkeypatch.setattr(tags, "_node_backward_enabled", True)
    for kind in ("SharedExpert", "MoK"):
        module = torch.nn.Linear(4, 3, bias=False, dtype=torch.float64)
        reference = torch.nn.Linear(4, 3, bias=False, dtype=torch.float64)
        reference.load_state_dict(module.state_dict())
        x = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
        expected_x = x.detach().clone().requires_grad_()
        expected = reference(expected_x)
        expected.square().sum().backward()
        tags._wrap_forward(module, kind, "L7")
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
            actual = module(x)
            assert type(actual.grad_fn) is type(expected.grad_fn)
            actual.square().sum().backward()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(x.grad, expected_x.grad, rtol=0, atol=0)
        torch.testing.assert_close(module.weight.grad, reference.weight.grad, rtol=0, atol=0)
        assert tags._moe_owner.get() == "unknown"
        names = {event.key for event in profiler.key_averages()}
        assert f"[RCD]{kind}fwd|L7" in names
        assert f"[RCD]{kind}bwd|L7" in names


def test_overlap_ranges_follow_cached_outputs_without_changing_gradients(monkeypatch):
    class CachedShared(torch.nn.Module):
        def produce(self, x):
            self.cached = x.square()

        def consume(self):
            result = self.cached.sum()
            self.cached = None
            return result

    monkeypatch.setattr(tags, "_node_backward_enabled", True)
    module = CachedShared()
    tags._wrap_forward(module, "SharedExpert", "L7|produce", method="produce", cached_outputs=("cached",))
    tags._wrap_forward(module, "SharedExpert", "L7|consume", method="consume", cached_outputs=("cached",))
    x = torch.randn(4, requires_grad=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
        assert module.produce(x) is None
        module.consume().backward()
    torch.testing.assert_close(x.grad, 2 * x.detach(), rtol=0, atol=0)
    assert module.cached is None
    names = {event.key for event in profiler.key_averages()}
    assert "[RCD]SharedExpertbwd|L7|produce" in names
    assert "[RCD]SharedExpertbwd|L7|consume" in names


def test_mok_schedule_diagnostics_accept_keywords_and_variable_expert_counts(monkeypatch, tmp_path):
    def schedule(workspace, config, counts):
        return SimpleNamespace(num_tokens=counts.sum().reshape(1), tokens_per_expert=counts)

    functional = SimpleNamespace(build_schedule=schedule, forward=lambda: None, backward=lambda: None)
    monkeypatch.setitem(sys.modules, "mok", SimpleNamespace(functional=functional))
    monkeypatch.setattr(tags, "_mok_schedule_records", [])
    tags._wrap_mok_diagnostics()
    workspace = SimpleNamespace(num_local_tokens=512, ep_size=8, topk=2)
    config = SimpleNamespace(macrobatch_size=4, minibatch_size=2, fwd_num_comm_sms=2, bwd_num_comm_sms=2)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]):
        first = functional.build_schedule(workspace=workspace, config=config, counts=torch.tensor([2, 3]))
        second = functional.build_schedule(workspace, config=config, counts=torch.tensor([1, 3, 4]))
    torch.testing.assert_close(first.tokens_per_expert, torch.tensor([2, 3]))
    torch.testing.assert_close(second.tokens_per_expert, torch.tensor([1, 3, 4]))
    tags.export_profile_diagnostics(str(tmp_path), "worker")
    exported = json.loads((tmp_path / "worker.mok-schedules.json").read_text())
    assert exported["read_after_capture"]
    assert [row["tokens_per_expert"] for row in exported["records"]] == [[2, 3], [1, 3, 4]]
    assert [row["scheduled_routed_rows"] for row in exported["records"]] == [5, 8]
    assert [row["macrobatches"] for row in exported["records"]] == [2, 2]
    assert not tags._mok_schedule_records
