# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import pytest

from relax.utils import megatron_bridge_utils


@pytest.mark.parametrize("with_backup", [False, True])
@pytest.mark.parametrize("with_exclusion", [False, True])
def test_megatron_bridge_adapter_wrapper_preserves_export_options(with_backup: bool, with_exclusion: bool) -> None:
    owner, model = object(), object()
    calls = []
    tasks = {"retained_layer": []}

    def build_tasks(self, megatron_model, *, exclude_adapter_base_prefixes=None):
        calls.append((self, megatron_model, exclude_adapter_base_prefixes))
        return tasks

    wrapped = megatron_bridge_utils._make_adapter_splice_wrapper(build_tasks)
    excluded = ("excluded_layer",) if with_exclusion else None
    options = {"exclude_adapter_base_prefixes": excluded} if with_exclusion else {}
    token = megatron_bridge_utils._adapter_splice_weights.set({} if with_backup else None)
    try:
        assert wrapped(owner, model, **options) is tasks
        assert calls == [(owner, model, excluded)]
    finally:
        megatron_bridge_utils._adapter_splice_weights.reset(token)
