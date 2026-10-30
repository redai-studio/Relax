# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Contract tests between ``megatron_adapter`` and the installed Megatron: an
upgrade that renames a timer, changes how the timers are called, or moves a
block, a config field or a function the adapter relies on fails here instead of
silently costing a training run its measurements.

Megatron's source is read, not imported, so neither torch nor a GPU is needed.
Every check also runs on a copy of the source with one such change, to show
that it notices.
"""

import ast
import importlib.util
import shutil
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import pytest

from relax.utils.straggler import megatron_adapter as adapter
from relax.utils.straggler.core.stats import DP_REDUCTION_PHASE


try:
    _SPEC = importlib.util.find_spec("megatron.core")
except ModuleNotFoundError:
    _SPEC = None
if _SPEC is None or not _SPEC.submodule_search_locations:
    pytest.skip("Megatron is not installed", allow_module_level=True)
_ROOT = Path(next(iter(_SPEC.submodule_search_locations)))

_TIMED = frozenset(adapter.MEGATRON_TIMER_SEGMENTS) | adapter.IGNORED_TIMER_NAMES
_GRAD_SYNC_FILE = "distributed/finalize_model_grads.py"

# Functions the adapter calls: file, name, the parameters it passes by position
# (in order) and by keyword.
_FUNCTIONS = (
    (
        "pipeline_parallel/schedules.py",
        "get_schedule_table",
        ("num_microbatches", "num_model_chunks", "microbatch_group_size_per_vp_stage"),
        (),
    ),
    ("utils.py", "get_model_config", ("model",), ()),
    ("parallel_state.py", "get_data_parallel_rank", (), ("with_context_parallel",)),
    ("parallel_state.py", "get_tensor_model_parallel_rank", (), ()),
    ("parallel_state.py", "get_pipeline_model_parallel_rank", (), ()),
    ("parallel_state.py", "get_context_parallel_rank", (), ()),
    ("parallel_state.py", "get_expert_model_parallel_rank", (), ()),
    ("parallel_state.py", "is_pipeline_last_stage", (), ()),
)
# Model config fields the schedule table reads, and the classes that declare them.
_CONFIG_FIELDS = ("overlap_moe_expert_parallel_comm", "microbatch_group_size_per_vp_stage")
_CONFIG_CLASSES = (
    ("model_parallel_config.py", "ModelParallelConfig"),
    ("transformer/transformer_config.py", "TransformerConfig"),
)
# For each object a MODULE_SPECS path walks through (by its path from the
# transformer layer), the classes that set its attributes or define its methods.
_OWNERS = {
    "": ("transformer/transformer_layer.py", ("TransformerLayer",)),
    "mlp": ("transformer/moe/moe_layer.py", ("BaseMoELayer", "MoELayer")),
    "mlp.token_dispatcher": ("transformer/moe/token_dispatcher.py", ("MoETokenDispatcher",)),
}
# The class a spec's ``when`` stands for.
_WHEN = {"MoE": ("transformer/moe/moe_layer.py", "MoELayer")}
_PLACEHOLDER_FILE = "transformer/identity_op.py"


@dataclass(frozen=True)
class _TimerCall:
    path: str
    line: int
    name: str | None  # None: not a string literal
    positional: int  # after the name
    keywords: frozenset[str]
    method: str | None  # what is called on the result; None: something else is done with it
    method_positional: int
    method_keywords: frozenset[str]

    @property
    def where(self):
        return f"{self.path}:{self.line}"


def _is_timers(node):
    return (isinstance(node, ast.Name) and node.id == "timers") or (
        isinstance(node, ast.Attribute) and node.attr == "timers"
    )


def _keywords(call):
    return frozenset(keyword.arg or "**" for keyword in call.keywords)


@cache
def _scan(root):
    """Every ``timers(...)`` call under ``root`` in file and line order, and
    every call of another method on the timers object itself."""
    calls, other_methods = [], []
    for path in sorted(root.rglob("*.py")):
        text = path.read_text()
        if "timers" not in text:
            continue
        relative = path.relative_to(root).as_posix()
        tree = ast.parse(text)
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Attribute) and _is_timers(node.func.value):
                other_methods.append(f"{relative}:{node.lineno}: Timers.{node.func.attr}()")
            if not _is_timers(node.func):
                continue
            first = node.args[0] if node.args else None
            name = first.value if isinstance(first, ast.Constant) and isinstance(first.value, str) else None
            parent = parents.get(node)
            method_call = parents.get(parent)
            if isinstance(parent, ast.Attribute) and isinstance(method_call, ast.Call) and method_call.func is parent:
                method, method_positional, method_keywords = parent.attr, len(method_call.args), _keywords(method_call)
            else:
                method, method_positional, method_keywords = None, 0, frozenset()
            calls.append(
                _TimerCall(
                    relative,
                    node.lineno,
                    name,
                    max(len(node.args) - 1, 0),
                    _keywords(node),
                    method,
                    method_positional,
                    method_keywords,
                )
            )
    return sorted(calls, key=lambda call: (call.path, call.line)), other_methods


def _timer_problems(root):
    calls, other_methods = _scan(root)
    problems = [f"{where}, which the profiler's timers do not support" for where in other_methods]
    for call in calls:
        if call.name is None:
            problems.append(f"{call.where}: a timer name that is not a string literal")
        elif call.name not in _TIMED:
            problems.append(
                f"{call.where}: timer {call.name!r} is neither in MEGATRON_TIMER_SEGMENTS nor in IGNORED_TIMER_NAMES"
            )
        if call.positional > 1 or not call.keywords <= {"log_level"}:
            problems.append(f"{call.where}: timers({call.name!r}, ...) passes arguments the profiler does not take")
        if call.method not in ("start", "stop"):
            problems.append(f"{call.where}: timer {call.name!r} is used other than by start() / stop()")
        elif call.method_positional > 1 or not call.method_keywords <= {"barrier"}:
            problems.append(f"{call.where}: {call.method}() gets arguments the profiler's timers do not take")
    called = {call.name for call in calls}
    problems += [
        f"timer {name!r} is mapped or ignored, but Megatron no longer calls it" for name in sorted(_TIMED - called)
    ]
    for phase, names in adapter.REQUIRED_TIMER_NAMES.items():
        problems += [f"timer {name!r}, required for {phase}, is not called" for name in sorted(names - called)]
    grad_syncs = {call.name for call in calls if call.path == _GRAD_SYNC_FILE}
    if grad_syncs != set(adapter.GRAD_SYNC_TIMERS):
        problems.append(
            f"{_GRAD_SYNC_FILE} times {sorted(grad_syncs)}, GRAD_SYNC_TIMERS has {adapter.GRAD_SYNC_TIMERS}"
        )
    return problems


def _tree(root, relative):
    path = root / relative
    return ast.parse(path.read_text()) if path.exists() else ast.Module(body=[], type_ignores=[])


def _top_level(tree, kind, name):
    return next((node for node in tree.body if isinstance(node, kind) and node.name == name), None)


def _api_problems(root):
    problems = []
    for relative, name, positional, keywords in _FUNCTIONS:
        function = _top_level(_tree(root, relative), ast.FunctionDef, name)
        if function is None:
            problems.append(f"{relative} has no function {name}")
            continue
        arguments = function.args
        params = [arg.arg for arg in arguments.posonlyargs + arguments.args]
        required = len(params) - len(arguments.defaults)
        by_keyword = set(params[len(positional) :]) | {arg.arg for arg in arguments.kwonlyargs}
        if (
            params[: len(positional)] != list(positional)
            or required > len(positional)
            or not set(keywords) <= by_keyword
        ):
            problems.append(f"{relative}: {name}({', '.join(params)}) no longer takes {positional} and {keywords}")
    init = _tree(root, "__init__.py")
    binds_mpu = any(
        isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "mpu" for target in node.targets)
        for node in init.body
    ) or any(
        alias.asname == "mpu"
        for node in init.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    )
    if not binds_mpu:
        problems.append("megatron.core no longer exports mpu")
    fields = set()
    for relative, class_name in _CONFIG_CLASSES:
        config = _top_level(_tree(root, relative), ast.ClassDef, class_name)
        if config is not None:
            fields |= {node.target.id for node in config.body if isinstance(node, ast.AnnAssign)}
    problems += [f"the model config has no field {field}" for field in _CONFIG_FIELDS if field not in fields]
    return problems


def _self_attributes(class_def):
    return {
        target.attr
        for node in ast.walk(class_def)
        for target in (
            node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        )
        if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self"
    }


def _owner_classes(root, owner_path):
    relative, class_names = _OWNERS[owner_path]
    tree = _tree(root, relative)
    return [cls for cls in (_top_level(tree, ast.ClassDef, name) for name in class_names) if cls is not None]


def _module_problems(root):
    problems = []
    layer = _owner_classes(root, "")
    for spec in adapter.MODULE_SPECS:
        owner_path, _, attribute = spec.path.rpartition(".")
        if owner_path not in _OWNERS:
            problems.append(f"spec {spec.name!r}: no Megatron class known for {owner_path!r}, add it to _OWNERS")
            continue
        if not any(attribute in _self_attributes(cls) for cls in _owner_classes(root, owner_path)):
            problems.append(f"spec {spec.name!r}: {_OWNERS[owner_path][1]} no longer set {attribute!r}")
        if spec.method is not None:
            methods = {
                node.name
                for cls in _owner_classes(root, spec.path)
                for node in cls.body
                if isinstance(node, ast.FunctionDef)
            }
            if spec.method not in methods:
                problems.append(f"spec {spec.name!r}: {_OWNERS[spec.path][1]} no longer define {spec.method}()")
        if spec.when is not None:
            relative, class_name = _WHEN[spec.when]
            if spec.when not in class_name or _top_level(_tree(root, relative), ast.ClassDef, class_name) is None:
                problems.append(f"spec {spec.name!r}: {relative} has no class {class_name}")
    for attribute in ("layer_number", "is_mtp_layer"):
        if not any(attribute in _self_attributes(cls) for cls in layer):
            problems.append(f"TransformerLayer no longer sets {attribute!r}, which layer_number() reads")
    placeholders = _tree(root, _PLACEHOLDER_FILE)
    problems += [
        f"{_PLACEHOLDER_FILE} has no class {name}"
        for name in sorted(adapter.PLACEHOLDERS)
        if _top_level(placeholders, ast.ClassDef, name) is None
    ]
    return problems


def _contract_problems(root):
    return _timer_problems(root) + _api_problems(root) + _module_problems(root)


def _source_files():
    calls, _ = _scan(_ROOT)
    files = {call.path for call in calls} | {"__init__.py", _PLACEHOLDER_FILE}
    files |= {relative for relative, *_ in _FUNCTIONS} | {relative for relative, _ in _CONFIG_CLASSES}
    files |= {relative for relative, _ in _OWNERS.values()} | {relative for relative, _ in _WHEN.values()}
    return sorted(files)


def _copy(tmp_path, relative=None, old=None, new=None):
    """The source files the checks read, with ``old`` replaced by ``new`` in
    ``relative``."""
    for name in _source_files():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(_ROOT / name, target)
    if relative is not None:
        path = tmp_path / relative
        text = path.read_text()
        assert old in text, f"{relative} no longer contains {old!r}: update this test case"
        path.write_text(text.replace(old, new))
    return tmp_path


@pytest.mark.parametrize("check", [_timer_problems, _api_problems, _module_problems], ids=lambda check: check.__name__)
def test_the_installed_megatron_matches_the_adapter(check):
    assert check(_ROOT) == []


def test_the_adapters_own_tables_agree():
    assert adapter.GRAD_SYNC_TIMERS[DP_REDUCTION_PHASE] == "all-grads-sync", "the trace aligns DP peers on it"
    assert adapter.REQUIRED_TIMER_NAMES["train"] <= set(adapter.MEGATRON_TIMER_SEGMENTS)
    assert adapter.REQUIRED_TIMER_NAMES["forward_only"] <= set(adapter.FORWARD_ONLY_TIMER_SEGMENTS)
    assert set(adapter.REQUIRED_TIMER_NAMES) == set(adapter.phase_timers(None))
    assert not adapter.IGNORED_TIMER_NAMES & set(adapter.MEGATRON_TIMER_SEGMENTS)


def test_an_unchanged_copy_of_the_source_passes(tmp_path):
    assert _contract_problems(_copy(tmp_path)) == []


@pytest.mark.parametrize(
    "relative, old, new, expected",
    [
        (_GRAD_SYNC_FILE, "'all-grads-sync'", "'all-grads-sync-v2'", "'all-grads-sync-v2' is neither"),
        (_GRAD_SYNC_FILE, "'all-grads-sync'", "'all-grads-sync-v2'", "'all-grads-sync' is mapped or ignored"),
        (
            "pipeline_parallel/schedules.py",
            "config.timers('forward-compute', log_level=2).start()",
            "config.timers('forward-compute', log_level=2, sync=True).start()",
            "passes arguments the profiler does not take",
        ),
        (
            _GRAD_SYNC_FILE,
            "config.timers('all-grads-sync').stop()",
            "config.timers('all-grads-sync').stop(sync=True)",
            "stop() gets arguments",
        ),
        (
            "pipeline_parallel/schedules.py",
            "config.timers('forward-compute').stop()",
            "config.timers('forward-compute').elapsed()",
            "used other than by start() / stop()",
        ),
        (
            _GRAD_SYNC_FILE,
            "config.timers('all-grads-sync').stop()",
            "config.timers('all-grads-sync').stop(); config.timers.log(['all-grads-sync'])",
            "Timers.log()",
        ),
        (
            "pipeline_parallel/schedules.py",
            "def get_schedule_table(num_microbatches, num_model_chunks, microbatch_group_size_per_vp_stage)",
            "def get_schedule_table(num_microbatches, microbatch_group_size_per_vp_stage, num_model_chunks)",
            "get_schedule_table(",
        ),
        (
            "parallel_state.py",
            "def get_data_parallel_rank(with_context_parallel=False",
            "def get_data_parallel_rank(with_cp=False",
            "get_data_parallel_rank(",
        ),
        (
            "model_parallel_config.py",
            "overlap_moe_expert_parallel_comm: bool",
            "overlap_moe_ep_comm: bool",
            "overlap_moe_expert_parallel_comm",
        ),
        ("__init__.py", "mpu = parallel_state", "", "mpu"),
        ("transformer/moe/moe_layer.py", "self.token_dispatcher", "self.dispatcher", "'token_dispatcher'"),
        ("transformer/moe/token_dispatcher.py", "def token_combine(", "def combine_tokens(", "token_combine()"),
        ("transformer/transformer_layer.py", "self.is_mtp_layer", "self.mtp_layer", "'is_mtp_layer'"),
        (_PLACEHOLDER_FILE, "class IdentityFuncOp(", "class IdentityFn(", "IdentityFuncOp"),
    ],
)
def test_a_changed_megatron_fails_the_contract(tmp_path, relative, old, new, expected):
    problems = _contract_problems(_copy(tmp_path, relative, old, new))
    assert any(expected in problem for problem in problems), problems
