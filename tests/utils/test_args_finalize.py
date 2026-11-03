# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""参数收尾阶段（``args_finalize``）的单元测试。

收尾阶段是"声明 → 收尾 → 校验"三段里的中间一段，本文件只验证它的两条契约：

1. 只补 **缺失** 的字段，任何已存在的值都不覆盖（幂等）；
2. 子解析器不可用（可选依赖缺失，或测试替身未提供声明 API）时安全跳过，不阻断解析。
"""

import sys
from argparse import Namespace
from types import ModuleType

from relax.utils import args_finalize


def _install_sglang_stub(monkeypatch, **attributes):
    """把 ``relax.backends.sglang.arguments`` 换成替身。

    同时覆盖 ``sys.modules`` 与包属性：``from package import name`` 会先查包属性， 只改
    ``sys.modules`` 不保证生效。
    """
    stub = ModuleType("relax.backends.sglang.arguments")
    for name, value in attributes.items():
        setattr(stub, name, value)
    monkeypatch.setitem(sys.modules, "relax.backends.sglang.arguments", stub)

    package = sys.modules.get("relax.backends.sglang")
    if package is not None:
        monkeypatch.setattr(package, "arguments", stub, raising=False)
    return stub


def test_seed_runtime_derived_fills_missing_only():
    args = Namespace(is_vl_model=True)
    args_finalize.seed_runtime_derived(args)

    assert args.is_vl_model is True, "已存在的值不得被覆盖"
    assert args.uses_unsplit_forward is False
    assert args.use_critic is False
    assert set(args_finalize.RUNTIME_DERIVED_DEFAULTS) <= set(vars(args))


def test_finalize_args_is_idempotent():
    args = Namespace()
    args_finalize.finalize_args(args)
    snapshot = dict(vars(args))
    args_finalize.finalize_args(args)
    assert vars(args) == snapshot


def test_finalize_args_only_fills_missing_subparser_fields(monkeypatch):
    def add_sglang_arguments(parser):
        parser.add_argument("--sglang-dp-size", type=int, default=1)
        parser.add_argument("--sglang-pp-size", type=int, default=2)
        return parser

    _install_sglang_stub(monkeypatch, add_sglang_arguments=add_sglang_arguments)
    args = Namespace(sglang_dp_size=8)

    args_finalize.finalize_args(args)

    assert args.sglang_dp_size == 8, "显式解析出来的值不得被声明默认值覆盖"
    assert args.sglang_pp_size == 2


def test_finalize_args_skips_stub_without_declaration_api(monkeypatch):
    """替身只提供解析入口时跳过补全，而不是抛 AttributeError。"""
    _install_sglang_stub(monkeypatch, sglang_parse_args=lambda: None, validate_args=lambda args: args)

    args = Namespace()
    args_finalize.finalize_args(args)

    # 只留下运行期派生属性，没有任何 sglang 字段。
    assert set(vars(args)) == set(args_finalize.RUNTIME_DERIVED_DEFAULTS)
