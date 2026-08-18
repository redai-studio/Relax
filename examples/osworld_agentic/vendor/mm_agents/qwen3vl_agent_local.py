import base64
import json
import logging
import os
import time
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import copy
import httpx
from PIL import Image

from mm_agents.utils.qwen_vl_utils import smart_resize


from mm_agents.opencua.prompts import (
    REWARD_SYSTEM_PROMPT_V1_L2,
    REWARD_INSTRUTION_TEMPLATE,
    REWARD_INSTRUTION_TEMPLATE_POST,
)

logger = logging.getLogger("desktopenv.qwen3vl_agent_local")


def encode_image(image_content: bytes) -> str:
    return base64.b64encode(image_content).decode("utf-8")


def process_image(image_bytes: bytes) -> str:
    """
    Process an image for Qwen VL models (thinking variant).
    Uses a tighter resize cap consistent with the thinking DUN agent.
    """
    image = Image.open(BytesIO(image_bytes))
    width, height = image.size

    resized_height, resized_width = smart_resize(
        height=height,
        width=width,
        factor=32,
        max_pixels=16 * 16 * 4 * 12800,
    )

    image = image.resize((resized_width, resized_height))

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    processed_bytes = buffer.getvalue()

    return base64.b64encode(processed_bytes).decode("utf-8")


def convert_message_with_img(messages: List[Dict[str, Any]], images: List[str]) -> List[Dict[str, Any]]:

    msgs_no_img: List[Dict[str, Any]] = []
    images_i = 0
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content")

        if isinstance(content, list):
            new_content = []
            for part in content:
                t = part.get("type")
                if t in ("image_url", "image"):
                    
                    if images_i >= len(images):
                        raise RuntimeError(
                            f"convert_message_with_img: need >= {images_i+1} images, but only {len(images)}. "
                            f"(role={role}, images_i={images_i})"
                        )
                    img_val = images[images_i]
                    new_content.append({"type": "image", "image": img_val})
                    images_i += 1
                elif t == "text":
                    new_content.append(part)
            msgs_no_img.append({"role": role, "content": new_content})
        else:

            msgs_no_img.append(m)

    return msgs_no_img


class Qwen3VLAgentLocal:
    def __init__(
        self,
        platform: str = "ubuntu",
        model: str = "qwen3-vl",
        max_steps: int = 100,
        max_image_history_length: int = 3,
        max_reward_image_history_length: int = 1,
        max_tokens: int = 32768,
        top_p: float = 0.9,
        temperature: float = 0.0,
        action_space: str = "pyautogui",
        observation_type: str = "screenshot",
        coordinate_type: str = "relative",  # "relative" | "absolute" | "qwen25"
        example_result_dir: Optional[str] = None,
        add_thought_prefix: bool = False,
    ):
        self.platform = platform
        self.model = model
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.temperature = temperature
        self.action_space = action_space
        self.observation_type = observation_type
        self.max_image_history_length = max_image_history_length
        self.max_reward_image_history_length = max_reward_image_history_length
        self.coordinate_type = coordinate_type
        self.example_result_dir = example_result_dir or os.getcwd()

        assert action_space in ["pyautogui", "hybrid"], "Invalid action space"
        assert observation_type in ["screenshot"], "Invalid observation type"
        assert coordinate_type in ["relative", "absolute", "qwen25"], "Invalid coordinate type"


        self.actions: List[str] = []
        self.observations: List[Dict[str, Any]] = []
        self.responses: List[str] = []
        self.screenshots: List[str] = []

        self.trajectory: List[Dict[str, Any]] = []
        self.reward_trajectory: List[Dict[str, Any]] = []

        self._input_toks_per_step: List[int] = []
        self._output_toks_per_step: List[int] = []
        # raw responses + errors from each retry of the current step; reset
        # again at the start of every predict(). Persisted into trajectory on
        # retry exhaustion so we never silently drop a step.
        self._failed_attempts: List[Dict[str, Any]] = []

    def reset(self, _logger=None):
        global logger
        logger = _logger if _logger is not None else logging.getLogger("desktopenv.qwen3vl_agent_local")

        self.actions = []
        self.observations = []
        self.responses = []
        self.screenshots = []
        self.trajectory = []
        self.reward_trajectory = []
        self._input_toks_per_step = []
        self._output_toks_per_step = []
        self._failed_attempts = []

    def _scale_scroll_for_windows(self, code: str, factor: int = 50) -> str:

        if self.platform.lower() != "windows":
            return code

        import re

        pattern_pos = re.compile(r"(pyautogui\.scroll\()\s*([-+]?\d+)\s*\)")
        return pattern_pos.sub(lambda m: f"{m.group(1)}{int(m.group(2)) * factor})", code)

    def predict(self, instruction: str, obs: Dict, **kwargs) -> Tuple[str, List[str], Dict]:

        if "step_idx" in kwargs:
            logger.info(f"========= {self.model} Step {kwargs['step_idx']} =======")
        else:
            logger.info(f"========================== {self.model} ===================================")
        logger.info(f"Instruction:\n{instruction}")

        step_index = len(self.actions)
        screenshot_bytes: bytes = obs["screenshot"]

        img0 = Image.open(BytesIO(screenshot_bytes))
        original_width, original_height = img0.size

        processed_image_b64 = process_image(screenshot_bytes)
        processed_img = Image.open(BytesIO(base64.b64decode(processed_image_b64)))
        processed_width, processed_height = processed_img.size

        description_prompt_lines = [
            "Use a mouse and keyboard to interact with a computer, and take screenshots.",
            "* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.",
            "* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.",
            (
                f"* The screen's resolution is {processed_width}x{processed_height}."
                if self.coordinate_type in ("absolute", "qwen25")
                else "* The screen's resolution is 1000x1000."
            ),
            "* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.",
            "* If you tried clicking on a program or link but it failed to load even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.",
            "* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.",
        ]
        description_prompt = "\n".join(description_prompt_lines)

        action_description_prompt = """
* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.
* `type`: Type a string of text on the keyboard.
* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.
* `left_click`: Click the left mouse button at a specified (x, y) pixel coordinate on the screen.
* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.
* `right_click`: Click the right mouse button at a specified (x, y) pixel coordinate on the screen.
* `middle_click`: Click the middle mouse button at a specified (x, y) pixel coordinate on the screen.
* `double_click`: Double-click the left mouse button at a specified (x, y) pixel coordinate on the screen.
* `triple_click`: Triple-click the left mouse button at a specified (x, y) pixel coordinate on the screen (simulated as double-click since it's the closest action).
* `scroll`: Performs a scroll of the mouse scroll wheel.
* `hscroll`: Performs a horizontal scroll (mapped to regular scroll).
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and report its completion status.
        """

        tools_def = {
            "type": "function",
            "function": {
                "name_for_human": "computer_use",
                "name": "computer_use",
                "description": description_prompt,
                "parameters": {
                    "properties": {
                        "action": {
                            "description": action_description_prompt,
                            "enum": [
                                "key",
                                "type",
                                "mouse_move",
                                "left_click",
                                "left_click_drag",
                                "right_click",
                                "middle_click",
                                "double_click",
                                "scroll",
                                "wait",
                                "terminate",
                            ],
                            "type": "string",
                        },
                        "keys": {"description": "Required only by `action=key`.", "type": "array"},
                        "text": {"description": "Required only by `action=type`.", "type": "string"},
                        "coordinate": {"description": "The x,y coordinates for mouse actions.", "type": "array"},
                        "pixels": {"description": "The amount of scrolling.", "type": "number"},
                        "time": {"description": "The seconds to wait.", "type": "number"},
                        "status": {
                            "description": "The status of the task.",
                            "type": "string",
                            "enum": ["success", "failure"],
                        },
                    },
                    "required": ["action"],
                    "type": "object",
                },
                "args_format": "Format the arguments as a JSON object.",
            },
        }

        system_prompt = (
            """# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
"""
            + json.dumps(tools_def)
            + """
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

# Response format

Response format for every step:
1) Action: a short imperative describing what to do in the UI.
2) A single <tool_call>...</tool_call> block containing only the JSON: {"name": <function-name>, "arguments": <args-json-object>}.

Rules:
- Output exactly in the order: Action, <tool_call>.
- Be brief: one sentence for Action.
- Do not output anything else outside those parts.
- If finishing, use action=terminate in the tool call."""
        )

        prev_lines = []
        for i, a in enumerate(self.actions):
            prev_lines.append(f"Step {i+1}: {a}")
        previous_actions_str = "\n".join(prev_lines) if prev_lines else "None"

        instruction_prompt = f"""
Please generate the next move according to the UI screenshot, instruction and previous actions.

Instruction: {instruction}

Previous actions:
{previous_actions_str}"""

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
        ]
        image_traj: List[str] = []

        start_i = max(0, step_index - self.max_image_history_length + 1)
        for i in range(start_i, step_index):

            img_url = f"data:image/png;base64,{self.screenshots[i]}"
            messages.append(
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": img_url}}]}
            )
            messages.append(
                {"role": "assistant", "content": [{"type": "text", "text": self.responses[i]}]}
            )
            image_traj.append(os.path.join(self.example_result_dir, f"step_{i}.png"))

        curr_img_url = f"data:image/png;base64,{processed_image_b64}"
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": curr_img_url}},
                    {"type": "text", "text": instruction_prompt},
                ],
            }
        )
        image_traj.append(os.path.join(self.example_result_dir, f"step_{step_index}.png"))

        reward_user_content: List[Dict[str, Any]] = []
        reward_image_traj: List[str] = []

        rstart_i = max(0, step_index - self.max_reward_image_history_length + 1)

        prev_lines = []
        for i in range(rstart_i):
            a = self.actions[i]
            prev_lines.append(f"Step {i+1}: {a}")
        previous_reward_actions_str = "\n".join(prev_lines) if prev_lines else "None"
        reward_user_content.append({"type": "text", "text": f"Previous Actions:\n {previous_reward_actions_str}"})

        for i in range(rstart_i, step_index):
            reward_user_content.append({"type": "text", "text": "Image of environment:\n"})
            reward_user_content.append({"type": "image", "image": "<image>"})
            reward_image_traj.append(os.path.join(self.example_result_dir, f"step_{i}.png"))
            reward_user_content.append(
                {"type": "text", "text": f"\nAction of agent:\nStep {i+1}:\n{self.actions[i]}\n"}
            )

        reward_user_content.append({"type": "text", "text": "Agent's current observation:\n"})
        reward_user_content.append({"type": "image", "image": "<image>"})
        reward_image_traj.append(os.path.join(self.example_result_dir, f"step_{step_index}.png"))

        base_reward_messages = [{"role": "system", "content": REWARD_SYSTEM_PROMPT_V1_L2}]
        base_reward_user_content = copy.deepcopy(reward_user_content)
        base_reward_image_traj = list(reward_image_traj)

        max_retry = 5
        retry_count = 0
        response: str = ""
        low_level_instruction: str = ""
        pyautogui_actions: List[str] = []
        other_info: Dict[str, Any] = {}
        self._failed_attempts = []

        while retry_count < max_retry:
            try:
                response = self.call_llm(
                    {
                        "model": self.model,
                        "messages": messages,
                        "max_tokens": self.max_tokens,
                        "top_p": self.top_p,
                        "temperature": self.temperature if retry_count == 0 else max(0.2, self.temperature),
                    },
                    self.model,
                )

                reward_messages = copy.deepcopy(base_reward_messages)
                reward_user_content = copy.deepcopy(base_reward_user_content)
                reward_image_traj = list(base_reward_image_traj)

                reward_user_content.append(
                    {"type": "text", "text": "\n" + REWARD_INSTRUTION_TEMPLATE.format(instruction=instruction, response=response)}
                )
                reward_messages.append({"role": "user", "content": reward_user_content})

                logger.info(f"Qwen3VL Output:\n{response}")
                if not response:
                    raise ValueError("Empty response from model")

                low_level_instruction, pyautogui_actions, other_info = self.parse_response(
                    response=response,
                    original_width=original_width,
                    original_height=original_height,
                    processed_width=processed_width,
                    processed_height=processed_height,
                )

                if (not pyautogui_actions) or any(a is None or a == "" for a in pyautogui_actions):
                    raise ValueError(f"Parsed empty actions from response: {response[:500]}")
                
                self.trajectory.append(
                    {
                        "messages": convert_message_with_img(messages, image_traj),
                        "response": response,
                        "step_index": step_index,
                        "original_screen_size": [original_width, original_height],
                        "processed_screen_size": [processed_width, processed_height],
                    }
                )
                self.reward_trajectory.append(
                    {
                        "reward_messages": convert_message_with_img(reward_messages, reward_image_traj),
                        "step_index": step_index,
                    }
                )

                break

            except Exception as e:
                logger.error(f"Error during predict (retry {retry_count+1}/{max_retry}): {e}")
                self._failed_attempts.append({
                    "retry": retry_count,
                    "response": response,
                    "error": str(e),
                })
                retry_count += 1
                if retry_count >= max_retry:
                    # Persist failed step so eval/training can inspect why
                    # parsing collapsed instead of silently dropping the
                    # trajectory entry.
                    self.trajectory.append({
                        "messages": convert_message_with_img(messages, image_traj),
                        "response": response,
                        "all_retry_responses": list(self._failed_attempts),
                        "step_index": step_index,
                        "original_screen_size": [original_width, original_height],
                        "processed_screen_size": [processed_width, processed_height],
                        "action_type": "parse_failed",
                        "parse_error": str(e),
                    })
                    return str(e), ["FAIL"], {"error": str(e), "parse_failed": True}
                continue

        pyautogui_actions = [self._scale_scroll_for_windows(code) for code in pyautogui_actions]

        logger.info(f"Low level instruction: {low_level_instruction}")
        logger.info(f"Pyautogui actions: {pyautogui_actions}")

        self.observations.append(obs)
        self.actions.append(low_level_instruction)
        self.responses.append(response)
        self.screenshots.append(processed_image_b64)

        current_step = len(self.actions)
        if current_step >= self.max_steps and not any("DONE" == x or "FAIL" == x for x in pyautogui_actions):
            logger.warning(f"Reached maximum steps {self.max_steps}. Forcing termination.")
            low_level_instruction = "Fail the task because reaching the maximum step limit."
            pyautogui_actions = ["FAIL"]
            other_info["code"] = "FAIL"

        return response, pyautogui_actions, other_info

    def parse_response(
        self,
        response: str,
        original_width: int,
        original_height: int,
        processed_width: Optional[int] = None,
        processed_height: Optional[int] = None,
    ) -> Tuple[str, List[str], Dict[str, Any]]:
        """
        Aligned with OpenCUAAgentLocal: returns (action_text, [pyautogui_code...], other_info)
        """
        low_level_instruction = ""
        pyautogui_code: List[str] = []
        other: Dict[str, Any] = {"raw_response": response, "tool_calls": []}

        if response is None or not response.strip():
            return low_level_instruction, pyautogui_code, other

        def adjust_coordinates(x: float, y: float) -> Tuple[int, int]:

            if self.coordinate_type in ("absolute", "qwen25"):
                if processed_width and processed_height:
                    x_scale = original_width / processed_width
                    y_scale = original_height / processed_height
                    return int(x * x_scale), int(y * y_scale)
                return int(x), int(y)

            # relative (0..999)
            x_scale = original_width / 999
            y_scale = original_height / 999
            return int(x * x_scale), int(y * y_scale)

        def process_tool_call(json_str: str) -> None:
            try:
                tool_call = json.loads(json_str)
                other["tool_calls"].append(tool_call)

                if tool_call.get("name") != "computer_use":
                    return
                args = tool_call.get("arguments", {})
                action = args.get("action")

                # --- mouse actions ---
                if action == "left_click":
                    if "coordinate" in args:
                        x, y = args["coordinate"]
                        adj_x, adj_y = adjust_coordinates(float(x), float(y))
                        pyautogui_code.append(f"pyautogui.click({adj_x}, {adj_y})")
                    else:
                        pyautogui_code.append("pyautogui.click()")

                elif action == "right_click":
                    if "coordinate" in args:
                        x, y = args["coordinate"]
                        adj_x, adj_y = adjust_coordinates(float(x), float(y))
                        pyautogui_code.append(f"pyautogui.rightClick({adj_x}, {adj_y})")
                    else:
                        pyautogui_code.append("pyautogui.rightClick()")

                elif action == "middle_click":
                    if "coordinate" in args:
                        x, y = args["coordinate"]
                        adj_x, adj_y = adjust_coordinates(float(x), float(y))
                        pyautogui_code.append(f"pyautogui.middleClick({adj_x}, {adj_y})")
                    else:
                        pyautogui_code.append("pyautogui.middleClick()")

                elif action == "double_click":
                    if "coordinate" in args:
                        x, y = args["coordinate"]
                        adj_x, adj_y = adjust_coordinates(float(x), float(y))
                        pyautogui_code.append(f"pyautogui.doubleClick({adj_x}, {adj_y})")
                    else:
                        pyautogui_code.append("pyautogui.doubleClick()")

                elif action == "mouse_move":
                    if "coordinate" in args:
                        x, y = args["coordinate"]
                        adj_x, adj_y = adjust_coordinates(float(x), float(y))
                        pyautogui_code.append(f"pyautogui.moveTo({adj_x}, {adj_y})")
                    else:
                        pyautogui_code.append("pyautogui.moveTo(0, 0)")

                elif action == "left_click_drag":
                    if "coordinate" in args:
                        x, y = args["coordinate"]
                        adj_x, adj_y = adjust_coordinates(float(x), float(y))
                        duration = args.get("duration", 0.5)
                        pyautogui_code.append(f"pyautogui.dragTo({adj_x}, {adj_y}, duration={duration})")
                    else:
                        pyautogui_code.append("pyautogui.dragTo(0, 0)")

                # --- keyboard ---
                elif action == "type":
                    text = args.get("text", "")
                    
                    text = str(text).replace("\\", "\\\\").replace("'", "\\'")
                    pyautogui_code.append(f"pyautogui.typewrite('{text}')")

                elif action == "key":
                    keys = args.get("keys", [])
                    if not isinstance(keys, list):
                        keys = [keys]
                    keys = [str(k).strip() for k in keys if k is not None]
                    keys_str = ", ".join([f"'{k}'" for k in keys])
                    if len(keys) > 1:
                        pyautogui_code.append(f"pyautogui.hotkey({keys_str})")
                    elif len(keys) == 1:
                        pyautogui_code.append(f"pyautogui.press({keys_str})")

                # --- scroll / wait / terminate ---
                elif action == "scroll":
                    pixels = args.get("pixels", 0)
                    pyautogui_code.append(f"pyautogui.scroll({int(pixels)})")

                elif action == "wait":
                    pyautogui_code.append("WAIT")

                elif action == "terminate":
                    status = (args.get("status") or "success").lower()
                    pyautogui_code.append("DONE" if status == "success" else "FAIL")

            except Exception as e:
                logger.error(f"Failed to parse tool call: {e}")

        # ---- parse response text ----
        lines = response.split("\n")
        inside_tool_call = False
        current_tool_call: List[str] = []

        for raw in lines:
            line = raw.strip()
            if not line:
                continue

            if line.lower().startswith("action:"):
                if not low_level_instruction:
                    low_level_instruction = line.split(":", 1)[-1].strip()
                continue

            if line.startswith("<tool_call>"):
                inside_tool_call = True
                continue

            if line.startswith("</tool_call>"):
                inside_tool_call = False
                if current_tool_call:
                    process_tool_call("\n".join(current_tool_call))
                    current_tool_call = []
                continue

            if inside_tool_call:
                current_tool_call.append(line)
                continue

            if line.startswith("{") and line.endswith("}"):
                try:
                    obj = json.loads(line)
                    if "name" in obj and "arguments" in obj:
                        process_tool_call(line)
                except Exception:
                    pass

        if current_tool_call:
            process_tool_call("\n".join(current_tool_call))

        if not low_level_instruction and pyautogui_code:
            low_level_instruction = "Execute the tool call"

        other["action"] = low_level_instruction
        other["code"] = pyautogui_code
        return low_level_instruction, pyautogui_code, other

    def call_llm(self, payload: Dict[str, Any], model: str) -> str:

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('QWEN3VL_API_KEY') or os.environ.get('OPENCUA_API_KEY','dummy')}",
        }

        local_eps = os.getenv("QWEN3VL_LOCAL_ENDPOINTS") or os.getenv("OPENCUA_LOCAL_ENDPOINTS")
        local_ep = os.getenv("QWEN3VL_LOCAL_ENDPOINT") or os.getenv("OPENCUA_LOCAL_ENDPOINT")

        bases: List[str] = []
        if local_eps:
            eps = [e.strip() for e in local_eps.split(",") if e.strip()]
            if eps:
                idx = os.getpid() % len(eps)  
                bases = [eps[idx]] + eps[idx + 1 :] + eps[:idx]
        elif local_ep:
            bases = [local_ep.strip()]

        if not bases:
            looks_like_path = ("/" in str(self.model)) or str(self.model).startswith((".", "/"))
            if looks_like_path:
                raise RuntimeError(
                    "QWEN3VL_LOCAL_ENDPOINTS/QWEN3VL_LOCAL_ENDPOINT (or OPENCUA_LOCAL_ENDPOINTS/OPENCUA_LOCAL_ENDPOINT) not set, "
                    "and model looks like a local path. Refusing to fallback to any paid/remote endpoint."
                )
            raise RuntimeError("No local endpoint configured for Qwen3VLAgentLocal.")

        logger.info(f"call_llm bases={bases} (eps={local_eps!r}, ep={local_ep!r})")

        max_total_attempts = 20
        backoff_sec = 5
        model_ctx_limit = 32768

        for attempt_i in range(max_total_attempts):
            for base in bases:
                url = f"{base.rstrip('/')}/v1/chat/completions"
                try:
                    resp = httpx.post(url, headers=headers, json=payload, timeout=3600, verify=False)
                except Exception as ex:
                    logger.error(f"HTTP error calling {url}: {ex}")
                    continue

                if resp.status_code == 400 and "maximum context length" in resp.text:
                    import re as _re
                    m = _re.search(r"contains at least (\d+) input tokens", resp.text)
                    if m:
                        input_toks = int(m.group(1))
                        new_max = max(1024, model_ctx_limit - input_toks - 64)
                        logger.warning(
                            f"Context overflow: {input_toks} input + {payload.get('max_tokens')} requested > {model_ctx_limit}. "
                            f"Reducing max_tokens to {new_max}."
                        )
                        payload["max_tokens"] = new_max
                        break
                    logger.error(f"LLM HTTP 400 context overflow from {url}: {resp.text[:500]}")
                    continue

                if resp.status_code != 200:
                    logger.error(f"LLM HTTP {resp.status_code} from {url}: {resp.text[:500]}")
                    continue

                data = resp.json()
                choice = (data.get("choices") or [{}])[0]
                finish_reason = choice.get("finish_reason")
                content = (choice.get("message") or {}).get("content", "")
                if finish_reason == "stop":
                    usage = data.get("usage") or {}
                    self._input_toks_per_step.append(int(usage.get("prompt_tokens", 0)))
                    self._output_toks_per_step.append(int(usage.get("completion_tokens", 0)))
                    return content
                elif finish_reason == "length" and content:
                    logger.warning(
                        f"LLM output truncated (finish_reason=length) from {url}; using partial response ({len(content)} chars)."
                    )
                    usage = data.get("usage") or {}
                    self._input_toks_per_step.append(int(usage.get("prompt_tokens", 0)))
                    self._output_toks_per_step.append(int(usage.get("completion_tokens", 0)))
                    return content
                else:
                    logger.error(
                        f"LLM did not finish properly from {url} (finish_reason={finish_reason}); will retry."
                    )

            time.sleep(backoff_sec)

        raise RuntimeError(f"LLM call failed after {max_total_attempts} attempts across endpoints: {bases}")

    def record_step_outcome(self, next_obs: Dict):

        if not self.reward_trajectory:
            logger.warning("record_step_outcome called but reward_trajectory is empty.")
            return

        last = self.reward_trajectory[-1]
        step_index = last.get("step_index")
        if step_index is None:
            step_index = max(0, len(self.actions) - 1)

        next_image_path = os.path.join(self.example_result_dir, f"step_{step_index + 1}.png")

        reward_messages = last.get("reward_messages", [])
        if not reward_messages:
            logger.warning("Last reward_trajectory has empty reward_messages.")
            return

        last_msg = reward_messages[-1]
        content = last_msg.get("content")
        if not isinstance(content, list):
            logger.warning("Last reward user message content is not a list; cannot append next obs.")
            return

        next_obs_text = "\nNext observation after executing this action:\n"
        content.append({"type": "text", "text": next_obs_text})
        content.append({"type": "image", "image": next_image_path})
        content.append({"type": "text", "text": REWARD_INSTRUTION_TEMPLATE_POST})

        last["next_obs_path"] = next_image_path
        if self.trajectory:
            self.trajectory[-1]["next_obs_path"] = next_image_path

    def summarize(self, result) -> str:

        out_dir = self.example_result_dir
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "trajectory.json")

        payload = {
            "meta": {
                "result": result,
                "input_toks_per_step": self._input_toks_per_step,
                "output_toks_per_step": self._output_toks_per_step,
                "total_input_toks": sum(self._input_toks_per_step),
                "total_output_toks": sum(self._output_toks_per_step),
            },
            "trajectory": self.trajectory,
            "reward_trajectory": self.reward_trajectory,
        }

        tmp_path = out_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, out_path)
        return out_path


