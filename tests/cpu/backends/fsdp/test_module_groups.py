# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Adapter-owned FSDP communication groups preserve module ownership."""

import pytest
import torch

from relax.backends.fsdp.runtime import fsdp2_wrap, resolve_module_groups


def _model():
    model = torch.nn.Module()
    model.left = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
    model.right = torch.nn.Linear(4, 4).requires_grad_(False)
    return model


def test_module_groups_keep_frozen_parameters_and_checkpoint_names(monkeypatch):
    model = _model()
    names = list(model.state_dict())
    model._fsdp_module_groups = ((model.left[0], model.left[1]), (model.right,))
    groups = resolve_module_groups(model)
    assert groups == [[model.left[0], model.left[1]], [model.right]]
    assert not any(parameter.requires_grad for parameter in groups[1][0].parameters())
    wrapped = []
    monkeypatch.setattr("torch.distributed.fsdp.fully_shard", lambda module, **kwargs: wrapped.append(module))
    fsdp2_wrap(model)
    assert wrapped == [*groups, model]
    assert list(model.state_dict()) == names


@pytest.mark.parametrize("case", ["duplicate", "ancestor", "shared", "foreign", "root", "fine", "empty"])
def test_module_groups_reject_ambiguous_ownership(case):
    model = _model()
    if case == "duplicate":
        groups = ((model.left[0],), (model.left[0],))
    elif case == "ancestor":
        groups = ((model.left,), (model.left[0],))
    elif case == "shared":
        model.right.weight = model.left[0].weight
        groups = ((model.left[0],), (model.right,))
    elif case == "foreign":
        groups = ((torch.nn.Linear(4, 4),),)
    elif case == "root":
        groups = ((model,),)
    elif case == "fine":
        model.left[0]._fsdp_fine_grained = True
        groups = ((model.left,),)
    else:
        groups = ((),)
    model._fsdp_module_groups = groups
    with pytest.raises(ValueError, match="FSDP"):
        resolve_module_groups(model)


def test_module_groups_default_preserves_existing_policy():
    model = _model()
    assert resolve_module_groups(model) == []


def test_native_group_hooks_wait_for_all_members_after_repeated_calls():
    from torch.distributed.fsdp._fully_shard._fsdp_state import _register_group_forward_hooks

    first, last = torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)
    pending, completed = set(), []

    def pre(module, args, kwargs):
        return args, kwargs

    def post(module, args, output):
        completed.append(module)
        return output

    handles = _register_group_forward_hooks([first, last], pre, post, pending, lambda output: output)
    try:
        for _ in range(2):
            value = first(torch.ones(1, 4))
            value = first(value)
            assert pending == {last}
            assert not completed
            last(value).sum().backward()
            assert completed == [last]
            assert not pending
            completed.clear()
    finally:
        handles.remove()
