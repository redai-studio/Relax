# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU-only tests for model-owned transient state at the colocate sleep
boundary."""

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from relax.backends.megatron.weight_update import train_offload


class _Owner(nn.Module):
    def __init__(self, name: str, events: list[str]) -> None:
        super().__init__()
        self.name = name
        self.events = events
        self.release_calls = 0
        self.runtime = SimpleNamespace(
            pre=torch.ones(2, requires_grad=True),
            latent={0: torch.ones(2, requires_grad=True)},
            input_ids=torch.tensor([1, 2]),
            topk=torch.tensor([0, 1]),
            candidates=torch.tensor([True, False]),
        )
        # Frozen CPU tables are ordinary attributes; hash constants are buffers.
        self.table = torch.ones(2, 32, dtype=torch.uint8)
        self.register_buffer("token_map", torch.tensor([0, 1, 2]), persistent=False)
        self.weight = nn.Parameter(torch.ones(2))
        self.decoder = nn.Module()
        self.decoder.runtime = self.runtime

    def _release_transient_state(self) -> None:
        self.events.append(self.name)
        self.release_calls += 1
        self.runtime.pre = None
        self.runtime.latent.clear()
        self.runtime.input_ids = None
        self.runtime.topk = None
        self.runtime.candidates = None


class _Strategy:
    def __init__(self, events: list[str], model: list[nn.Module] | None = None) -> None:
        self.events = events
        self.model = model

    def offload(self) -> None:
        for chunk in self.model or ():
            for module in chunk.modules():
                if isinstance(module, _Owner):
                    assert module.runtime.pre is None and not module.runtime.latent
                    assert module.runtime.topk is None and module.runtime.candidates is None
        self.events.append("offload")

    def reload(self) -> None:
        self.events.append("reload")

    def disable_during_update(self) -> Any:
        return nullcontext()


def _offloader(
    monkeypatch: pytest.MonkeyPatch,
    model: list[nn.Module] | None,
    events: list[str],
    *,
    tms: bool = False,
    enabled: bool = True,
) -> train_offload.MegatronTrainStateOffloader:
    monkeypatch.setattr(train_offload, "torch_memory_saver_preloaded", lambda: tms)
    monkeypatch.setattr(
        train_offload.MegatronTrainStateOffloader, "_init_torch_memory_saver", staticmethod(lambda args: True)
    )
    monkeypatch.setattr(train_offload, "_TmsOffloadStrategy", lambda: _Strategy(events, model))
    monkeypatch.setattr(
        train_offload, "_SelectiveOffloadStrategy", lambda model, optimizer, args: _Strategy(events, model)
    )
    return train_offload.MegatronTrainStateOffloader(
        model,
        None,
        SimpleNamespace(offload_train=enabled, selective_offload=not tms),
    )


@pytest.mark.parametrize("tms", [False, True])
def test_transient_state_is_cleared_before_selective_or_tms_offload(
    monkeypatch: pytest.MonkeyPatch, tms: bool
) -> None:
    events: list[str] = []
    first, second = _Owner("first", events), _Owner("second", events)
    first_table, first_map, first_weight = first.table, first.token_map, first.weight
    second_table = second.table
    offloader = _offloader(monkeypatch, [first, second], events, tms=tms)
    offloader.offload()
    assert events == ["first", "second", "offload"]
    assert offloader.uses_tms is tms
    assert first.release_calls == 1 and second.release_calls == 1
    assert first.decoder.runtime is first.runtime and first.runtime.pre is None
    assert first.table is first_table and second.table is second_table
    assert first.token_map is first_map and first.weight is first_weight
    assert first.table.device.type == "cpu" and first.table.dtype == torch.uint8
    assert set(first.state_dict()) == {"weight"}


def test_disabled_offload_does_not_clear_transient_state_or_run_strategy(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    owner = _Owner("owner", events)
    initial_pre = owner.runtime.pre
    offloader = _offloader(monkeypatch, [owner], events, enabled=False)
    offloader.offload()
    offloader.reload()
    assert events == [] and owner.release_calls == 0
    assert owner.runtime.pre is initial_pre and owner.runtime.latent


def test_other_models_without_release_hook_keep_existing_offload_behavior(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    model = nn.Sequential(nn.Linear(2, 2), nn.ReLU())
    initial = model[0].weight
    offloader = _offloader(monkeypatch, [model], events)
    offloader.offload()
    offloader.reload()
    assert events == ["offload", "reload"] and model[0].weight is initial


def test_reload_does_not_clear_newly_rebuilt_transient_state(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    owner = _Owner("owner", events)
    offloader = _offloader(monkeypatch, [owner], events)
    offloader.offload()
    rebuilt = torch.ones(3)
    owner.runtime.pre = rebuilt
    offloader.reload()
    assert owner.runtime.pre is rebuilt and owner.release_calls == 1
    assert events == ["owner", "offload", "reload"]


def test_release_failure_does_not_silently_offload_live_transient_state(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    model = nn.Module()

    def fail() -> None:
        raise RuntimeError("release failed")

    model._release_transient_state = fail
    offloader = _offloader(monkeypatch, [model], events)
    with pytest.raises(RuntimeError, match="release failed"):
        offloader.offload()
    assert events == []
