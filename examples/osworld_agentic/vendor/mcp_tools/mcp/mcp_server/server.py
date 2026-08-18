import mcp.types as types
from fastmcp import FastMCP
import json
import inspect
import argparse
from functools import wraps

from tools.package.code import CodeTools
try:
    from tools.package.google_chrome import BrowserTools
except ImportError as _e:
    import logging as _logging
    _logging.warning(f"google_chrome tools unavailable: {_e}")
    BrowserTools = None
from tools.package.libreoffice_calc import CalcTools
from tools.package.libreoffice_impress import ImpressTools
from tools.package.libreoffice_writer import WriterTools
from tools.package.vlc import VLCTools
# from tools.package.multi_apps_bb83 import PresentationToolsUNO
try:
    from tools.package.libreoffice_impress_ours import PresentationToolsUNO
except ImportError as _e:
    import logging as _logging
    _logging.warning(f"libreoffice_impress_ours tools unavailable: {_e}")
    PresentationToolsUNO = None
# from tools.package.multi_apps_d9b7 import EmailTools
from tools.package.os_ours import UnifiedTools
# from tools.package.computer_use_split import ComputerUseTools
# from tools.package.computer_use import ComputerUseTools

from tools.package.code_ours import VSCodeTools


try:
    from tools.package.impress_calc_ljt import CalcToolsPlus
except ImportError as _e:
    import logging as _logging
    _logging.warning(f"impress_calc_ljt tools unavailable: {_e}")
    CalcToolsPlus = None
try:
    from tools.package.system_ljt import SystemTools
except ImportError as _e:
    import logging as _logging
    _logging.warning(f"system_ljt tools unavailable: {_e}")
    SystemTools = None

from fastmcp.tools.tool import Tool

_STATE_SENTINEL = object()
_TOOL_STATE_ATTRS = ("ret", "fs_ret", "file_ret")


# meta_tools = {
#     ('tools/apis/code.json', CodeTools),
#     ('tools/apis/google_chrome.json', BrowserTools),
#     # ('tools/apis/libreoffice_calc.json', CalcTools),
#     # ('tools/apis/libreoffice_impress.json', ImpressTools),
#     # ('tools/apis/libreoffice_writer.json', WriterTools),
#     ('tools/apis/vlc.json', VLCTools),
#     ('tools/apis/test.json', TestTools)
# }

meta_tools = {
    CodeTools: 'code',
    **(({BrowserTools: 'google_chrome'}) if BrowserTools is not None else {}),
    CalcTools: 'libreoffice_calc',
    ImpressTools: 'libreoffice_impress',
    WriterTools: 'libreoffice_writer',
    VLCTools: 'vlc',
    # TestTools: 'test',
    # ComputerUseTools: 'computer_use',
    # ComputerUseTools: 'computer_use_split',

    # jhr
    **({PresentationToolsUNO: 'libreoffice_impress2'} if PresentationToolsUNO is not None else {}),
    UnifiedTools: 'os',


    **({CalcToolsPlus: 'libreoffice_calc2'} if CalcToolsPlus is not None else {}),
    **({SystemTools: 'os'} if SystemTools is not None else {}),
    VSCodeTools: 'code2'
}


def format_func_name(target_cls, method_name):
    # return method_name
    return f'{meta_tools[target_cls]}.{method_name}'


def _reset_tool_state(target_cls):
    for attr_name in _TOOL_STATE_ATTRS:
        if hasattr(target_cls, attr_name):
            setattr(target_cls, attr_name, _STATE_SENTINEL)


def _extract_tool_state(target_cls):
    for attr_name in _TOOL_STATE_ATTRS:
        if not hasattr(target_cls, attr_name):
            continue
        value = getattr(target_cls, attr_name)
        if value is _STATE_SENTINEL or value in ("", None):
            continue
        return value
    return None


def _looks_like_error(value):
    if not isinstance(value, str):
        return False
    normalized = value.strip().lower()
    if not normalized:
        return False
    error_prefixes = (
        "error",
        "failed",
        "failure",
        "unexpected error",
        "unhandled exception",
        "traceback",
        "exception",
        "❌",
    )
    error_contains = (
        " not found",
        "no save location",
        "invalid ",
        "unable to ",
        "failed to ",
        "could not ",
        "unsupported ",
        "out of range",
        "does not exist",
        "already exists",
        "returned false",
    )
    return normalized.startswith(error_prefixes) or any(fragment in normalized for fragment in error_contains)


def _json_safe(value):
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return str(value)


def _normalize_result_field(value):
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return _json_safe(value)
    return {"value": _json_safe(value)}


def _normalize_error_message(value):
    if value in (None, ""):
        return None
    if isinstance(value, str):
        return value
    return json.dumps(_json_safe(value), ensure_ascii=False)


def _build_response(success, result=None, error_message=None):
    normalized_success = bool(success)
    normalized_error = None if normalized_success else _normalize_error_message(error_message)
    if not normalized_success and normalized_error is None:
        normalized_error = "Tool execution failed."
    return {
        "success": normalized_success,
        "result": _normalize_result_field(result),
        "error_message": normalized_error,
    }


def _normalize_tool_output(raw_result, target_cls, tool_name, exc=None):
    state_value = _extract_tool_state(target_cls)

    if exc is not None:
        return _build_response(
            False,
            None,
            f"{exc.__class__.__name__}: {exc}"
        )

    if isinstance(raw_result, dict) and {
        "success", "result", "error_message"
    }.issubset(raw_result.keys()):
        return _build_response(
            raw_result.get("success"),
            raw_result.get("result"),
            raw_result.get("error_message"),
        )

    if isinstance(raw_result, dict) and "success" in raw_result:
        error_message = raw_result.get("error_message")
        if not error_message and not raw_result.get("success"):
            for key in ("error", "stderr", "message"):
                if raw_result.get(key):
                    error_message = raw_result.get(key)
                    break
        return _build_response(
            raw_result.get("success"),
            raw_result.get("result", raw_result),
            error_message,
        )

    if isinstance(raw_result, bool):
        if raw_result:
            return _build_response(True, state_value, None)
        return _build_response(
            False,
            None,
            state_value or f"{tool_name} returned False."
        )

    if raw_result is None:
        if _looks_like_error(state_value):
            return _build_response(False, None, state_value)
        return _build_response(True, state_value, None)

    if isinstance(raw_result, str):
        if _looks_like_error(raw_result):
            return _build_response(False, None, raw_result)
        return _build_response(True, raw_result, None)

    if _looks_like_error(state_value):
        return _build_response(False, None, state_value)

    return _build_response(True, raw_result, None)


def _wrap_tool_function(func, target_cls, formatted_func_name):
    signature = inspect.signature(func)

    if inspect.iscoroutinefunction(func):
        @wraps(func)
        async def wrapped(*args, **kwargs):
            _reset_tool_state(target_cls)
            try:
                raw_result = await func(*args, **kwargs)
            except Exception as exc:
                return _normalize_tool_output(None, target_cls, formatted_func_name, exc=exc)
            return _normalize_tool_output(raw_result, target_cls, formatted_func_name)
    else:
        @wraps(func)
        def wrapped(*args, **kwargs):
            _reset_tool_state(target_cls)
            try:
                raw_result = func(*args, **kwargs)
            except Exception as exc:
                return _normalize_tool_output(None, target_cls, formatted_func_name, exc=exc)
            return _normalize_tool_output(raw_result, target_cls, formatted_func_name)

    wrapped.__signature__ = signature
    return wrapped


def register_tools_from_json(mcp_instance: FastMCP, tools_json, target_cls):
    tools = []

    for entry in tools_json:
        fn_info = entry.get("function", {})
        full_method_name = fn_info["name"]
        # In the JSON, name has the form "CodeTools.method"
        method_name = full_method_name.split(".")[-1]

        # Fetch this method from the class
        func = getattr(target_cls, method_name, None)
        if not func:
            print(f"Warning: method {method_name} not found in {target_cls}")
            # import time
            # time.sleep(1000)
            continue
        formatted_func_name = format_func_name(target_cls, method_name)
        wrapped_func = _wrap_tool_function(func, target_cls, formatted_func_name)

        # # Special-case initialization for the libreoffice classes
        # libreoffice_tools = ['libreoffice_calc', 'libreoffice_impress', 'libreoffice_writer']
        # if meta_tools[target_cls] in libreoffice_tools:
        #     func = target_cls.ensure_initialized(func)
        #     if not func:
        #         print(f"Warning: method {method_name} not found in {target_cls} with `ensure_initialized`")
        #         continue

        # Pass the parameters and description from the JSON to the Tool object
        mcp_instance.add_tool(
            Tool.from_function(
                wrapped_func,
                name=formatted_func_name
            ))
        tools.append(types.Tool(
            # name=fn_info['name'],
            name=formatted_func_name,
            description=fn_info['description'],
            inputSchema=fn_info['parameters'],
        ))

    return tools


def init_server(server_name):
    mcp = FastMCP("OSWorld")

    tools = []
    for target_cls, name in meta_tools.items():
        apis_path = f'tools/apis/{name}.json'
        with open(apis_path, 'r', encoding='utf-8') as f:
            apis = json.load(f)
        _tools = register_tools_from_json(mcp, apis, target_cls)
        tools.extend(_tools)

    tools = sorted(tools, key=lambda x: x.name)


    async def list_tools() -> list[types.Tool]:
        return tools

    mcp._mcp_server.list_tools()(list_tools)

    return mcp


def parse_args():
    parser = argparse.ArgumentParser(description="Run OSWorld MCP server")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run MCP in debug mode (no HTTP transport)"
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    mcp = init_server("OSWorld")

    if args.debug:
        # Debug mode: run with default transport (likely stdio)
        print("Running in debug mode...")
        mcp.run()
    else:
        # Normal HTTP mode
        mcp.run(
            transport='http',
            host='0.0.0.0',
            port=9292,
            # path='mcp'
        )
