"""owl_parser.py — adapted from OSWorld-MCP owl_agent.py

Parses Qwen NousFnCall-style responses (<thinking>/<tool_call>/<conclusion>)
into action dicts and converts them to pyautogui code strings.

Two public functions:
- parse_action_fncall_id(text, image_path, height, width, model_name)
    Extracts thinking/conclusion/tool_call from raw model output, JSON-decodes
    the tool call, and (for computer_use) converts coordinates abs_resized→abs_origin.
    Returns list[dict{thought, conclusion, action_type, action_inputs, text}].

- parsing_response_to_pyautogui_code(responses, image_height, image_width, input_swap=True)
    Turns the parsed dicts into pyautogui code strings (or {action_type, parameters}
    dict for MCP tool calls).
"""
from __future__ import annotations

import json
import re

from .coordinate_resize import convert_point_format, update_image_size_


def escape_single_quotes(text):
    pattern = r"(?<!\\)'"
    return re.sub(pattern, r"\\'", text)


def parse_action_fncall_id(text, image_path, height, width, model_name):
    thought = ""
    if "<thinking>" in text and "</thinking>" in text:
        thought = text.split("<thinking>")[-1].split("</thinking>")[0]
    elif "<thinking>" in text:
        thought = text.split("<thinking>")[1]

    conclusion = ""
    if "<conclusion>" in text and "</conclusion>" in text:
        conclusion = text.split("<conclusion>")[-1].split("</conclusion>")[0]
    elif "<conclusion>" in text:
        conclusion = text.split("<conclusion>")[1]
    if conclusion == "" and thought != "":
        conclusion = thought

    if "<tool_call>" in text and "</tool_call>" in text:
        all_tool_calls = re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)
        if len(all_tool_calls) >= 2:
            # Detect mouse_move + left_click_drag(coordinate only) pattern.
            # Pure-GUI format: move to start, then dragTo(end). owl_parser needs both
            # coordinates in one call. Merge them so the drag executes correctly.
            try:
                prev_json = json.loads(all_tool_calls[-2].strip())
                last_json = json.loads(all_tool_calls[-1].strip())
                prev_args = prev_json.get('arguments', {k: v for k, v in prev_json.items() if k != 'name'})
                last_args = last_json.get('arguments', {k: v for k, v in last_json.items() if k != 'name'})
                if (prev_args.get('action') == 'mouse_move'
                        and last_args.get('action') == 'left_click_drag'
                        and 'coordinate2' not in last_args
                        and 'coordinate' in prev_args):
                    merged = {'name': 'computer_use', 'arguments': {
                        'action': 'left_click_drag',
                        'coordinate': prev_args['coordinate'],
                        'coordinate2': last_args['coordinate'],
                    }}
                    action = json.dumps(merged)
                else:
                    action = all_tool_calls[-1].strip()
            except Exception:
                action = all_tool_calls[-1].strip()
        else:
            action = all_tool_calls[-1].strip()
    else:
        action = text.split("<tool_call>")[-1].split("<conclusion>")[0]
        action = action[:action.rfind('}') + 1]

    action_str = action.strip('\n')
    # JSON doesn't support Python hex literals (0xFF0000); convert to decimal before parsing
    action_str = re.sub(r'\b0x([0-9a-fA-F]+)\b', lambda m: str(int(m.group(1), 16)), action_str)

    # If the model emitted a conclusion but no tool_call (common after a successful MCP call
    # when the model believes the task is done), synthesise a terminate action so the step
    # does not parse_failed. Treat any conclusion as task-success; the evaluator is
    # authoritative — a false DONE just ends the episode a step early, which is correct.
    if not action_str and conclusion:
        return [{
            "thought": thought,
            "conclusion": conclusion,
            "action_type": "finished",
            "action_inputs": {},
            "text": text,
        }]

    # Accept BOTH tool-call shapes:
    #   nested  (Qwen owl / OpenAI fn-call): {"name": "...", "arguments": {...}}
    #   flat    (Qwen3-VL-Instruct native): {"name": "...", "action": "...", "coordinate": [...]}
    # ToolCUA's evaluation showed Instruct-tuned models emit the flat shape ~92%
    # of the time even when prompted with the nested example. Hard-requiring
    # ['arguments'] caused 67/73 parse_failed on our Instruct base eval.
    _parsed = json.loads(action_str)
    if isinstance(_parsed, dict) and 'arguments' in _parsed and isinstance(_parsed['arguments'], dict):
        action_json = _parsed['arguments']
    else:
        action_json = {k: v for k, v in _parsed.items() if k != 'name'} if isinstance(_parsed, dict) else _parsed

    if "computer_use" in action:
        current_image_ele = update_image_size_({'image': "None", 'width': width, 'height': height})

        if action_json['action'] == "key":
            action_type = 'hotkey'
            keys = action_json['keys']
            keys_str = ""
            for key in keys:
                keys_str += " " + key
            action_inputs = {"hotkey": keys_str}
        elif action_json['action'] == "type":
            action_type = "type"
            if 'clear' not in action_json:
                action_json['clear'] = 0
            if 'enter' not in action_json:
                action_json['enter'] = 0
            action_inputs = {'content': action_json['text'], 'clear': int(action_json['clear']), 'enter': int(action_json['enter'])}
        elif action_json['action'] == "mouse_move":
            action_type = "hover"
            x, y = convert_point_format([action_json['coordinate'][0], action_json['coordinate'][1]], current_image_ele, src_format='abs_resized', tgt_format='abs_origin', model_name=model_name)
            action_inputs = {'start_box': [x, y]}
        elif action_json['action'] == "left_click_drag" or action_json['action'] == "drag":
            action_type = "drag"
            start_coord = action_json.get('coordinate') or action_json.get('startCoordinate') or action_json.get('coordinate2', [0, 0])
            x, y = convert_point_format([start_coord[0], start_coord[1]], current_image_ele, src_format='abs_resized', tgt_format='abs_origin', model_name=model_name)
            # coordinate2 may be absent when the model uses pure-GUI two-step semantics
            # (mouse_move to start, then left_click_drag to end with only coordinate).
            # Fall back to start_coord so the action degrades to a zero-length drag (click)
            # instead of crashing.
            end_coord = action_json.get('coordinate2') or action_json.get('endCoordinate') or start_coord
            x2, y2 = convert_point_format([end_coord[0], end_coord[1]], current_image_ele, src_format='abs_resized', tgt_format='abs_origin', model_name=model_name)
            action_inputs = {'start_box': [x, y], 'end_box': [x2, y2]}
        elif action_json['action'] == "left_click" or action_json['action'] == "click":
            action_type = "click"
            x, y = convert_point_format([action_json['coordinate'][0], action_json['coordinate'][1]], current_image_ele, src_format='abs_resized', tgt_format='abs_origin', model_name=model_name)
            action_inputs = {'start_box': [x, y]}
        elif action_json['action'] == "right_click":
            action_type = "right_single"
            x, y = convert_point_format([action_json['coordinate'][0], action_json['coordinate'][1]], current_image_ele, src_format='abs_resized', tgt_format='abs_origin', model_name=model_name)
            action_inputs = {'start_box': [x, y]}
        elif action_json['action'] == "double_click":
            action_type = "left_double"
            x, y = convert_point_format([action_json['coordinate'][0], action_json['coordinate'][1]], current_image_ele, src_format='abs_resized', tgt_format='abs_origin', model_name=model_name)
            action_inputs = {'start_box': [x, y]}
        elif action_json['action'] == "scroll":
            action_type = "scroll"
            action_inputs = {'pixels': action_json['pixels']}
        elif action_json['action'] == "terminate":
            if action_json['status'] == 'success':
                action_type = "finished"
            else:
                action_type = "fail"
            action_inputs = {}
        elif action_json['action'] == "wait":
            action_type = "wait"
            action_inputs = {'time': action_json['time'] if 'time' in action_json else 1}
        elif action_json['action'] in ("function", "function_call"):
            # Model mistakenly wrapped an MCP call inside computer_use, e.g.:
            # {"name": "computer_use", "arguments": {"action": "function_call",
            #   "function": "libreoffice_writer.set_default_font", "arguments": {...}}}
            # Recover: treat the value of "function" as the real MCP tool name.
            fn = action_json.get('function') or action_json.get('function_call', '')
            if fn:
                action_type = fn
                action_inputs = action_json.get('arguments', {}) or {}
            else:
                action_type = action_json.get('action', 'unknown')
                action_inputs = {k: v for k, v in action_json.items() if k != 'action'}
        else:
            action_type = action_json.get('action', 'unknown')
            action_inputs = {k: v for k, v in action_json.items() if k != 'action'}

    else:
        parsed_action = json.loads(action_str)
        action_type = parsed_action['name']
        action_inputs = parsed_action['arguments']

    return [{
        "thought": thought,
        "conclusion": conclusion,
        "action_type": action_type,
        "action_inputs": action_inputs,
        "text": text,
    }]


def parsing_response_to_pyautogui_code(responses, image_height, image_width, input_swap=True):
    pyautogui_code = "import pyautogui\nimport time\n"
    if isinstance(responses, dict):
        responses = [responses]
    for response_id, response in enumerate(responses):
        if response_id > 0:
            pyautogui_code += "\ntime.sleep(3)\n"

        action_type = response.get("action_type")
        action_inputs = response.get("action_inputs", {})

        if action_type == "hotkey":
            hotkey = action_inputs.get("key", "") or action_inputs.get("hotkey", "")
            if hotkey:
                keys = hotkey.split()
                pyautogui_code += f"\npyautogui.hotkey({', '.join([repr(k) for k in keys])})"

        elif action_type == "type":
            content = action_inputs.get("content", "")
            content = escape_single_quotes(content)
            if content:
                if input_swap:
                    pyautogui_code += "\nimport pyperclip"
                    pyautogui_code += f"\npyperclip.copy('{content.strip()}')"
                    pyautogui_code += "\npyautogui.hotkey('ctrl', 'v')"
                    pyautogui_code += "\ntime.sleep(0.5)\n"
                    if content.endswith("\n") or content.endswith("\\n"):
                        pyautogui_code += "\npyautogui.press('enter')"
                else:
                    if action_inputs.get('clear') == 1:
                        pyautogui_code += "\npyautogui.hotkey('ctrl', 'a')"
                        pyautogui_code += "\npyautogui.press('backspace')"
                    pyautogui_code += f"\npyautogui.write('{content.strip()}', interval=0.1)"
                    pyautogui_code += "\ntime.sleep(0.5)\n"
                    if content.endswith("\n") or content.endswith("\\n") or action_inputs.get('enter') == 1:
                        pyautogui_code += "\npyautogui.press('enter')"

        elif action_type in ["drag", "select"]:
            start_box = action_inputs.get("start_box")
            end_box = action_inputs.get("end_box")
            if start_box and end_box:
                sx, sy = start_box
                ex, ey = end_box
                pyautogui_code += (
                    f"\npyautogui.moveTo({sx}, {sy})\n"
                    f"\npyautogui.dragTo({ex}, {ey}, duration=1.0)\n"
                )

        elif action_type == "scroll":
            pixels = action_inputs.get("pixels")
            pyautogui_code += f"\npyautogui.scroll({pixels})"

        elif action_type in ["click", "left_single", "left_double", "right_single", "hover"]:
            start_box = action_inputs.get("start_box")
            if start_box:
                if isinstance(start_box, (list, tuple)):
                    box = list(start_box)
                else:
                    box = eval(str(start_box))
                if len(box) == 4:
                    x1, y1, x2, y2 = box
                    x = (x1 + x2) / 2
                    y = (y1 + y2) / 2
                elif len(box) == 2:
                    x, y = box
                else:
                    x = y = 0

                if action_type in ("click", "left_single"):
                    pyautogui_code += f"\npyautogui.click({x}, {y}, button='left')"
                elif action_type == "left_double":
                    pyautogui_code += f"\npyautogui.doubleClick({x}, {y}, button='left')"
                elif action_type == "right_single":
                    pyautogui_code += f"\npyautogui.click({x}, {y}, button='right')"
                elif action_type == "hover":
                    pyautogui_code += f"\npyautogui.moveTo({x}, {y})"

        elif action_type == "finished":
            return "DONE"

        elif action_type == "fail":
            return "FAIL"

        elif action_type == "wait":
            pyautogui_code += f"\ntime.sleep({action_inputs.get('time', 1)})"

        else:
            return {'action_type': action_type, 'parameters': action_inputs}

    return pyautogui_code
