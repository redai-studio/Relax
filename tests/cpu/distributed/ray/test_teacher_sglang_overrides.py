# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import sys
from types import ModuleType, SimpleNamespace

from relax.utils.opd.opd_utils import (
    build_teacher_engine_args,
    build_teacher_overrides,
    teacher_sglang_parse_args,
)


def test_teacher_mem_fraction_is_projected_to_engine_args():
    args = SimpleNamespace(
        teacher_hf_checkpoint="/teacher",
        teacher_sglang_mem_fraction_static=0.73,
        sglang_mem_fraction_static=0.42,
    )

    overrides = build_teacher_overrides(args, colocate_sync=True)
    engine_args = build_teacher_engine_args(args, overrides)

    assert overrides["model_path"] == "/teacher"
    assert overrides["load_format"] == "auto"
    assert overrides["enable_memory_saver"] is True
    assert overrides["mem_fraction_static"] == 0.73
    assert engine_args is not args
    assert engine_args.sglang_mem_fraction_static == 0.73
    assert args.sglang_mem_fraction_static == 0.42


def test_teacher_sglang_parse_args_only_keeps_explicit_overrides(monkeypatch):
    server_args = ModuleType("sglang.srt.server_args")

    class _ServerArgs:
        @staticmethod
        def add_cli_args(parser):
            parser.add_argument("--model-path", default="/default-model")
            parser.add_argument("--mem-fraction-static", type=float, default=0.5)

    server_args.ServerArgs = _ServerArgs
    monkeypatch.setitem(sys.modules, "sglang", ModuleType("sglang"))
    monkeypatch.setitem(sys.modules, "sglang.srt", ModuleType("sglang.srt"))
    monkeypatch.setitem(sys.modules, "sglang.srt.server_args", server_args)
    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "--teacher-sglang-mem-fraction-static", "0.73"],
    )

    args = teacher_sglang_parse_args()

    assert args.teacher_sglang_mem_fraction_static == 0.73
    assert not hasattr(args, "teacher_sglang_model_path")
