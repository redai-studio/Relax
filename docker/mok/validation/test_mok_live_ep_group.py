from types import SimpleNamespace

import pytest
import torch.distributed as dist
from megatron.core.transformer.moe.megakernel.mok.runtime import _live_ep_group

from relax.utils.reloadable_process_group import ReloadableProcessGroup


def test_registered_group_is_resolved_again_after_reload(tmp_path):
    dist.init_process_group("gloo", init_method=f"file://{tmp_path}/store", rank=0, world_size=1)
    replacement = None
    try:
        original = dist.group.WORLD
        wrapper = ReloadableProcessGroup(original, [0])
        with pytest.raises(RuntimeError, match="name not set"):
            _ = wrapper.group_name
        assert _live_ep_group(wrapper) is original
        assert _live_ep_group(wrapper).group_name == original.group_name
        replacement = dist.new_group([0], backend="gloo")
        wrapper.group = replacement
        assert _live_ep_group(wrapper) is replacement
        assert _live_ep_group(wrapper).group_name != original.group_name
        wrapper.group = None
        with pytest.raises(RuntimeError, match="asleep"):
            _live_ep_group(wrapper)
    finally:
        if replacement is not None:
            dist.destroy_process_group(replacement)
        dist.destroy_process_group()


def test_unwrapped_group_is_preserved():
    group = SimpleNamespace(group_name="registered")
    assert _live_ep_group(group) is group
