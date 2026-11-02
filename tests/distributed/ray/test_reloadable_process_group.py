# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Check process-group selection with real CPU groups and Dynamo."""

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("mode", ["resident", "offload", "unspecified"])
def test_reloadable_process_group_resident_compile_and_offload_compatibility(mode, tmp_path):
    # Isolate the global torch.distributed patches, and extract the real code
    # without importing optional CUDA memory-management dependencies.
    source = Path(__file__).resolve().parents[3] / "relax/utils/reloadable_process_group.py"
    program = r"""
import ast
import contextlib
import os
import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

source, rendezvous, mode = sys.argv[1:]
tree = ast.parse(Path(source).read_text())
names = {"monkey_patch_torch_dist", "ReloadableProcessGroup"}
nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
namespace = dict(torch=torch, dist=dist, os=os, timedelta=timedelta,
                 old_new_group_dict={}, _wrap_low_level_call=contextlib.nullcontext,
                 logger=SimpleNamespace(info=lambda *a, **kw: None))
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=0, world_size=1)
try:
    native = dist.group.WORLD
    # Exercise the multi-rank NCCL selection branch using a real CPU group;
    # group creation itself is substituted because this is a CPU regression.
    dist.new_group = lambda *args, **kwargs: native
    args = SimpleNamespace(offload_train=mode == "offload") if mode != "unspecified" else None
    namespace["monkey_patch_torch_dist"](args)
    group = dist.new_group(ranks=[0, 1], backend="nccl")
    if mode == "resident":
        assert group is native
        assert group.group_name == native.group_name
        def query(x, process_group):
            return x + process_group.rank()
        result = torch.compile(query, backend="eager")(torch.ones(1), group)
        torch.testing.assert_close(result, torch.ones(1))
        # Compile a collective through the retained communication wrapper.
        def reduce(x, process_group):
            dist.all_reduce(x, group=process_group)
            return x
        value = torch.compile(reduce, backend="eager")(torch.ones(1), group)
        torch.testing.assert_close(value, torch.ones(1))
    else:
        assert isinstance(group, namespace["ReloadableProcessGroup"])
        assert group.group is native
        assert group.group_info["ranks"] == [0, 1]
        value = torch.ones(1)
        dist.all_reduce(value, group=group)
        torch.testing.assert_close(value, torch.ones(1))
finally:
    dist.destroy_process_group()
"""
    result = subprocess.run(
        [sys.executable, "-c", program, str(source), str(tmp_path / "rendezvous"), mode],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
