# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Shorten two SGLang control polls without bypassing their readiness checks.

Compatibility patch for SGLang versions with one-second polling. Keep the
original function objects so FastAPI routes and decorators retain their
targets. Never replace the shared asyncio.sleep implementation.
"""

import ast
import asyncio
import inspect
import textwrap
from types import CodeType, FunctionType

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


async def _abort_and_wait(manager, interval: float) -> None:
    """Keep the native one-second abort retry while checking for drain more
    often."""
    clock = asyncio.get_running_loop().time
    started = clock()
    manager.abort_request(abort_all=True)
    abort_count = 1
    retry_at = clock() + 1.0
    while await manager.model_update_lock.is_locked():
        remaining = retry_at - clock()
        if remaining <= 0:
            manager.abort_request(abort_all=True)
            abort_count += 1
            remaining = 1.0
            retry_at = clock() + remaining
        await asyncio.sleep(min(interval, remaining))
    logger.info("SGLang abort-pause drained in %.3fs (%d abort sends)", clock() - started, abort_count)


def _replace_abort_loop(node: ast.AsyncFunctionDef, native_interval: float, interval: float) -> None:
    # Only replace the known SGLang abort branch. Leave the surrounding pause
    # condition and every other mode in the upstream function.
    native = ast.parse(
        f"""while True:
    self.abort_request(abort_all=True)
    is_locked = await self.model_update_lock.is_locked()
    if not is_locked:
        break
    await asyncio.sleep({native_interval!r})"""
    ).body[0]
    for condition in (item for item in ast.walk(node) if isinstance(item, ast.If)):
        for branch in (condition.body, condition.orelse):
            if len(branch) == 1 and ast.dump(branch[0]) == ast.dump(native):
                branch[:] = ast.parse(
                    "from relax.backends.sglang.control_polling_patch import _abort_and_wait\n"
                    f"await _abort_and_wait(self, {interval!r})"
                ).body
                return
    raise RuntimeError("Unsupported SGLang abort-pause loop")


def _compile_poll_interval_patch(function: FunctionType, interval: float, *, abort_pause: bool = False) -> CodeType:
    """Build replacement code with a shorter sleep; do not modify the function
    yet."""
    if function.__code__.co_freevars:
        raise RuntimeError(f"Unsupported SGLang closure: {function.__qualname__}")
    source, start = inspect.getsourcelines(function)
    module = ast.parse(textwrap.dedent("".join(source)))
    node = module.body[0]
    if not isinstance(node, ast.AsyncFunctionDef) or node.name != function.__name__:
        raise RuntimeError("Expected the original SGLang async function")
    polls = [
        item.value
        for item in ast.walk(node)
        if isinstance(item, ast.Await)
        and isinstance(item.value, ast.Call)
        and isinstance(item.value.func, ast.Attribute)
        and isinstance(item.value.func.value, ast.Name)
        and item.value.func.value.id == "asyncio"
        and item.value.func.attr == "sleep"
    ]
    if (
        len(polls) != 1
        or len(polls[0].args) != 1
        or polls[0].keywords
        or not isinstance(polls[0].args[0], ast.Constant)
        or polls[0].args[0].value != 1
    ):
        raise RuntimeError(f"Unsupported SGLang polling implementation: {function.__qualname__}")
    expected_locals = function.__code__.co_varnames
    if abort_pause:
        _replace_abort_loop(node, polls[0].args[0].value, interval)
        expected_locals = tuple("_abort_and_wait" if name == "is_locked" else name for name in expected_locals)
    else:
        polls[0].args[0] = ast.copy_location(ast.Constant(interval), polls[0].args[0])
    node.decorator_list = []
    ast.increment_lineno(module, start - 1)
    compiled = compile(ast.fix_missing_locations(module), function.__code__.co_filename, "exec")
    code = next(value for value in compiled.co_consts if isinstance(value, CodeType) and value.co_name == node.name)
    original = function.__code__
    for field in (
        "co_argcount",
        "co_posonlyargcount",
        "co_kwonlyargcount",
        "co_freevars",
        "co_cellvars",
    ):
        if getattr(code, field) != getattr(original, field):
            raise RuntimeError(f"SGLang control signature changed: {field}")
    if code.co_varnames != expected_locals:
        raise RuntimeError("SGLang control locals changed")
    metadata = {"co_flags": original.co_flags}
    if hasattr(original, "co_qualname"):
        metadata["co_qualname"] = original.co_qualname
    return code.replace(**metadata)


def patch_control_poll_interval(interval: float, *, tokenizer_worker_num: int | None = 1) -> None:
    """Patch compatible SGLang control waits; otherwise keep the existing
    behavior."""
    if interval == 1.0:
        return
    if not 0.001 <= interval < 1.0:
        raise ValueError("SGLang control polling interval must be between 0.001 and 1 second")
    if tokenizer_worker_num != 1:
        logger.warning("Skipping control polling patch: requires one tokenizer worker")
        return
    try:
        from sglang.srt.entrypoints import http_server
        from sglang.srt.managers.tokenizer_manager import TokenizerManager

        health = inspect.unwrap(http_server.health_generate)
        pause = inspect.unwrap(TokenizerManager.pause_generation)
    except (ImportError, AttributeError, OSError, TypeError, ValueError) as error:
        logger.warning("Skipping control polling patch; keeping existing behavior: %s", error)
        return
    functions = (health, pause)
    if all(getattr(function, "_relax_control_poll_interval", None) == interval for function in functions):
        return
    if any(hasattr(function, "_relax_control_poll_interval") for function in functions):
        raise RuntimeError("SGLang control polling was already patched with a different interval")
    try:
        endpoints = [
            inspect.unwrap(route.endpoint)
            for route in http_server.app.routes
            if getattr(route, "path", None) in ("/health", "/health_generate")
        ]
        if len(endpoints) != 2 or any(endpoint is not health for endpoint in endpoints):
            raise RuntimeError("Unexpected SGLang health routes; refusing polling patch")
        # Compile both functions before changing either; a failed probe leaves
        # routes, abort behavior and the native polling intervals untouched.
        codes = [
            _compile_poll_interval_patch(health, interval),
            _compile_poll_interval_patch(pause, interval, abort_pause=True),
        ]
    except (AttributeError, OSError, RuntimeError, SyntaxError, TypeError, ValueError) as error:
        logger.warning("Skipping control polling patch; keeping existing behavior: %s", error)
        return
    for function, code in zip(functions, codes, strict=True):
        function.__code__ = code
        function._relax_control_poll_interval = interval
    logger.info("SGLang control polling interval: %.3fs (abort retry: 1s)", interval)
