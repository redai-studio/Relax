"""
hybrid_agent_local.py

Hybrid GUI + MCP agent that uses the NousFnCall protocol (Qwen training
distribution) for tool invocation:
  <thinking>...</thinking>
  <tool_call>{"name": ..., "arguments": {...}}</tool_call>
  <conclusion>...</conclusion>

When action_space="hybrid", routes between:
  - computer_use tool (Qwen's ComputerUse) → pyautogui code string
  - MCP tool calls (libreoffice_calc.*, etc.) → "MCP:<tool>:<json>" string

Preserves the RL trajectory schema consumed by lib_run_single_hybrid,
reward/osworld_rl_reward.py, and sample/osworld_reward_rollout_api.py.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import logging
import os
import sys
import time
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from mm_agents.qwen3vl_agent_local import (
    Qwen3VLAgentLocal,
    convert_message_with_img,
    encode_image,
    process_image,
)
from mm_agents.owl_parser import parse_action_fncall_id, parsing_response_to_pyautogui_code
from mm_agents.agent_function_call import ComputerUse
from mm_agents.coordinate_resize import update_image_size_

# Add repo root to path so `agents`, `prompts`, `environment` are importable
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from agents.tool_retriever import ToolRetriever
from prompts.policy_hybrid import build_system_text, build_user_prompt_text

logger = None


# Action types emitted by parse_action_fncall_id that map to GUI (computer_use)
_GUI_ACTION_TYPES = frozenset({
    "hotkey", "type", "click", "left_single", "left_double",
    "right_single", "hover", "scroll", "drag", "select",
    "wait", "finished", "fail",
})


# Keyword → canonical app for instruction-based fallback when xprop can't pin
# down the active window (e.g. fresh Start Center, unrecognised WM_CLASS).
# Order matters: more specific apps first.
_APP_KEYWORDS = [
    ("libreoffice_calc", [
        "spreadsheet", "workbook", "csv", "pivot", "cell ", "cells",
        " sheet", "sheets", " column", "columns", " row ", " rows",
        "libreoffice calc", "excel",
    ]),
    ("libreoffice_impress", [
        "slide", "slides", "presentation", "impress", "powerpoint", "pptx",
    ]),
    ("libreoffice_writer", [
        "paragraph", "subscript", "superscript", "strikethrough", "underline",
        "document", "writer", "page break", " font", "word doc", ".docx", ".doc ",
    ]),
    ("vlc", [
        "vlc", "playlist", "video player", "media player", "audio track",
    ]),
    ("code", [
        "vs code", "vscode", "visual studio code", "extension", "editor settings",
    ]),
    ("google_chrome", [
        "chrome", "firefox", "browser", "website", "url ", "bookmark",
        "browser tab", "web page",
    ]),
    ("thunderbird", ["thunderbird", "email", "inbox"]),
    ("os", ["terminal", "shell", "bash"]),
]


def _infer_app_from_instruction(instruction):
    """Best-effort active app inference from natural-language instruction.

    Used as a fallback when xprop-based detection fails. Returns None if no
    keyword matches — caller falls back to BM25 over the full registry.
    """
    if not instruction:
        return None
    s = " " + instruction.lower() + " "
    for app, kws in _APP_KEYWORDS:
        if any(k in s for k in kws):
            return app
    return None


def _registry_tool_to_qwen_function(tool: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a ToolRetriever output dict to a qwen_agent function schema.

    Output shape matches what NousFnCallPrompt expects:
        {"name", "description", "parameters": {"type", "properties", "required"}}
    """
    name = tool["name"]
    app = tool.get("app", "")
    qualified_name = f"{app}.{name}" if app and "." not in name else name

    params = tool.get("params", {}) or {}
    properties: Dict[str, Any] = {}
    required: List[str] = []
    for pname, pspec in params.items():
        if isinstance(pspec, dict):
            prop: Dict[str, Any] = {
                "type": pspec.get("type", "string"),
                "description": pspec.get("description", ""),
            }
            if "enum" in pspec:
                prop["enum"] = pspec["enum"]
            if "items" in pspec:
                prop["items"] = pspec["items"]
            properties[pname] = prop
            if "default" not in pspec:
                required.append(pname)
        else:
            properties[pname] = {"type": "string"}
            required.append(pname)

    return {
        "name": qualified_name,
        "description": tool.get("description", ""),
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


class HybridAgentLocal(Qwen3VLAgentLocal):
    """Hybrid GUI+MCP agent (owl-style NousFnCall protocol)."""

    def __init__(
        self,
        *,
        tools_registry_path: Optional[str] = None,
        tool_retriever_backend: str = "bm25",
        tool_retriever_top_k: int = 18,
        mcp_host: str = "localhost",
        mcp_base_port: int = 9292,
        use_old_sys_prompt: bool = False,
        context_policy: str = "baseline",
        # Max consecutive frames that skip_on_no_change may omit before forcing a
        # full screenshot to re-align vision (safety valve against long no-change
        # stretches). Only consulted when context_policy == "skip_on_no_change".
        max_consecutive_skips: int = 2,
        # Sampling-distribution correction knob. PPO/GRPO assumes rollouts are
        # sampled from the raw policy; repetition_penalty distorts that while the
        # recomputed old_logp pretends on-policy. TRAIN rollouts should pass 1.0
        # (together with top_p=1.0); EVAL keeps the 1.05 default — there it guards
        # greedy decoding against repetition loops and never feeds training.
        repetition_penalty: float = 1.05,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._repetition_penalty = float(repetition_penalty)

        # When True: bare "You are a helpful assistant." (no MCP preference hint).
        # When False: soft MCP-preference prompt that still allows GUI fallback.
        self._use_old_sys_prompt = use_old_sys_prompt

        # Context management policy:
        #   "baseline"            — every step gets a screenshot (default, current behavior)
        #   "skip_on_mcp_success" — skip screenshot when previous MCP call succeeded (lossy)
        #   "skip_on_no_change"   — skip screenshot when byte-identical to the previous
        #                           frame (lossless; model already saw that state)
        self._context_policy = context_policy
        self._max_consecutive_skips = int(max_consecutive_skips)

        self._hybrid = self.action_space == "hybrid"

        # Per-step metadata for trajectory.json
        self._step_action_types: List[str] = []
        self._step_exec_oks: List[bool] = []
        self._step_tools_available: List[bool] = []
        self._step_exec_results: List[Dict[str, Any]] = []

        # Parsed action dicts for reward message construction
        self._step_parsed_actions: List[Dict[str, Any]] = []

        if self._hybrid:
            # Relax the assertion from parent that only allows "pyautogui"
            self.action_space = "hybrid"
            registry = tools_registry_path or os.path.join(
                _REPO_ROOT, "tools", "tools_registry.json"
            )
            self._tool_retriever = ToolRetriever(
                registry_path=registry,
                top_k=tool_retriever_top_k,
                backend=tool_retriever_backend,
            )
            self._mcp_host = mcp_host
            self._mcp_base_port = mcp_base_port
        else:
            self._tool_retriever = None

    # ── reset ────────────────────────────────────────────────────────────

    def reset(self, _logger=None):
        global logger
        logger = (
            _logger
            if _logger is not None
            else logging.getLogger("desktopenv.hybrid_agent_local")
        )
        self.actions = []
        self.observations = []
        self.responses = []
        self.screenshots = []
        # Parallel to self.screenshots: md5 of each stored processed b64 frame,
        # used by skip_on_no_change for O(1) pixel-identity comparison.
        self._screenshot_hashes = []
        self.trajectory = []
        self.reward_trajectory = []

        self._step_action_types = []
        self._step_exec_oks = []
        self._step_tools_available = []
        self._step_exec_results = []
        self._step_parsed_actions = []
        # exec result strings keyed by step idx; "" for GUI steps
        self._history_exec_feedback: List[str] = []
        # one-line conclusions for owl-style history string
        self._history_conclusions: List[str] = []
        # active window title at each step (from DesktopEnv.get_app_state).
        # Injected as "Action Response: {app_info}" in next step's prompt so
        # model can see the current file / sheet / slide and avoid index errors.
        self._history_app_infos: List[str] = []
        self._input_toks_per_step = []
        self._output_toks_per_step = []
        # raw responses + errors from each retry of the current step; reset
        # again at the start of every predict(). Persisted into trajectory on
        # retry exhaustion so we never silently drop a step.
        self._failed_attempts: List[Dict[str, Any]] = []

    # ── predict ──────────────────────────────────────────────────────────

    def predict(
        self, instruction: str, obs: Dict, **kwargs
    ) -> Tuple[str, List[str], Dict]:
        if not self._hybrid:
            return super().predict(instruction, obs, **kwargs)

        if "step_idx" in kwargs:
            logger.info(f"========= HybridAgent Step {kwargs['step_idx']} =======")
        logger.info(f"Instruction:\n{instruction}")

        step_index = len(self.actions)
        screenshot_bytes: bytes = obs["screenshot"]

        img0 = Image.open(BytesIO(screenshot_bytes))
        original_width, original_height = img0.size

        processed_image_b64 = process_image(screenshot_bytes)
        processed_img = Image.open(BytesIO(base64.b64decode(processed_image_b64)))
        processed_width, processed_height = processed_img.size

        # Prefer live MCP server tool list (populated by lib_run_single._enrich_obs_with_mcp_tools).
        # Falls back to local ToolRetriever when env didn't enrich (e.g. unit tests).
        detected_app = obs.get("cur_app") if isinstance(obs, dict) else None
        cur_app = detected_app or _infer_app_from_instruction(instruction)
        logger.info(f"[cur_app] detected={detected_app!r} inferred_used={cur_app!r}")
        live_tools = obs.get("tool_list") if isinstance(obs, dict) else None
        if live_tools is not None:
            # Already in {name, description, parameters} JSON-Schema shape from
            # OsworldMcpClient.list_tools — passes straight through NousFnCallPrompt.
            # Use `is not None` (not truthiness) so that [] from apps with no MCP
            # tools (chrome, gimp, thunderbird) suppresses the BM25 fallback.
            retrieved_tools = live_tools
            tools_from_live = True
        else:
            retrieved_tools = (
                self._tool_retriever.retrieve(instruction, app_hint=cur_app)
                if self._tool_retriever else []
            )
            tools_from_live = False
        tools_available = len(retrieved_tools) > 0
        logger.info(f"[tool_list] n={len(retrieved_tools)} source={'live_mcp' if tools_from_live else 'local_registry'}")

        # Build messages using owl-style NousFnCall protocol
        messages, image_traj = self._build_messages_owl(
            instruction, step_index, retrieved_tools,
            processed_image_b64, original_width, original_height,
            processed_width, processed_height,
        )

        # Build base reward context (action-specific part appended after parsing)
        reward_user_content, reward_image_traj = self._build_base_reward_context(
            instruction, step_index, processed_image_b64
        )

        max_retry = 5
        retry_count = 0
        response = ""
        parsed_action: Dict[str, Any] = {}
        pyautogui_actions: List[str] = []
        action_type = "gui"
        self._failed_attempts = []

        while retry_count < max_retry:
            try:
                response = self.call_llm(
                    {
                        "model": self.model,
                        "messages": messages,
                        # 4096 gives the Thinking variant enough budget for
                        "max_tokens": self.max_tokens,
                        "top_p": self.top_p,
                        "temperature": self.temperature if retry_count == 0 else max(0.2, self.temperature),
                        # vLLM /v1/chat/completions accepts this at the top level.
                        # 1.05 for eval (anti-repetition guard, matches OSWorld-MCP);
                        # 1.0 for train rollouts (on-policy sampling, see __init__).
                        "repetition_penalty": self._repetition_penalty,
                        # Stop after the first complete tool call. The system prompt
                        # mandates a single <tool_call>; stopping here prevents the
                        # temp=0 degeneration where the model emits hundreds of
                        # repeated <tool_call> blocks until max_tokens is exhausted
                        # (which left no budget to ever emit `terminate`).
                        # include_stop_str_in_output keeps the closing tag so
                        # owl_parser's `"</tool_call>" in text` check still matches.
                        "stop": ["</tool_call>"],
                        "include_stop_str_in_output": True,
                    },
                    self.model,
                )
                logger.info(f"HybridAgent Output:\n{response}")
                if not response:
                    raise ValueError("Empty response from model")

                # Parse with owl-style protocol
                # owl_parser's convert_point_format routes coordinate conversion by
                # substring-matching model_name ("qwen3"/"qwen2" → ×1920/1000 grid
                # scaling). RL checkpoints are served under paths like
                # runs/<proj>/ckpt/optimized with no such substring, so every
                # post-training eval/rollout silently fell into the wrong branch —
                # all clicks landed off-target, held-out collapsed to ~10% with
                # frac_hit_cap 80%+ while base-path evals stayed healthy (the real
                # e6–e25 "step-1 cliff"). This agent is qwen3vl-only: pin the
                # parser routing explicitly, independent of the serving path.
                _parser_model_name = (
                    self.model
                    if ("qwen3" in self.model.lower() or "qwen2" in self.model.lower())
                    else "qwen3-vl"
                )
                actions_raw = parse_action_fncall_id(
                    response, image_path=None,
                    height=original_height, width=original_width,
                    model_name=_parser_model_name,
                )
                if not actions_raw:
                    raise ValueError("Parser returned no actions")

                primary = actions_raw[0]
                a_type_raw = primary.get("action_type", "")
                thought = (primary.get("thought", "") or "").strip()
                conclusion = (primary.get("conclusion", "") or "").strip()
                action_inputs = primary.get("action_inputs", {}) or {}

                if a_type_raw in _GUI_ACTION_TYPES:
                    action_type = "gui"
                    code = parsing_response_to_pyautogui_code(
                        [primary],
                        image_height=original_height,
                        image_width=original_width,
                        input_swap=False,  # use pyautogui.write() not pyperclip+ctrl+v; clipboard paste is rejected by Chrome/VSCode/terminal fields
                    )
                    if isinstance(code, str):
                        pyautogui_actions = [code]
                    else:
                        # parser returned dict for unrecognised GUI type — treat as fail
                        pyautogui_actions = ["FAIL"]
                    parsed_action = {
                        "action_type": "gui",
                        "action": pyautogui_actions[0],
                        "thinking": thought,
                        "conclusion": conclusion,
                        "raw_action_type": a_type_raw,
                        "raw_action_inputs": action_inputs,
                    }
                else:
                    # MCP tool call — a_type_raw is the (possibly app-qualified) tool name
                    action_type = "mcp"
                    tool_name = a_type_raw
                    if "." not in tool_name:
                        tool_to_app = {}
                        for t in retrieved_tools:
                            full = t.get("name", "")
                            if "." in full:
                                # Live MCP tool: derive app from name prefix.
                                prefix, _, suffix = full.partition(".")
                                tool_to_app[suffix] = prefix
                                tool_to_app[full] = prefix
                            else:
                                tool_to_app[full] = t.get("app", "")
                        app = tool_to_app.get(tool_name, "")
                        if app:
                            tool_name = f"{app}.{tool_name}"
                    pyautogui_actions = [f"MCP:{tool_name}:{json.dumps(action_inputs)}"]
                    parsed_action = {
                        "action_type": "mcp",
                        "tool_name": tool_name,
                        "params": action_inputs,
                        "thinking": thought,
                        "conclusion": conclusion,
                    }

                if not pyautogui_actions or all(not a for a in pyautogui_actions):
                    raise ValueError(f"Empty actions from response: {response[:500]}")

                # Finalize reward messages with action context
                reward_messages = self._finalize_reward_messages(
                    instruction, step_index, action_type, parsed_action,
                    reward_user_content, reward_image_traj, retrieved_tools,
                )

                self.trajectory.append({
                    "messages": convert_message_with_img(messages, image_traj),
                    "response": response,
                    "step_index": step_index,
                    "original_screen_size": [original_width, original_height],
                    "processed_screen_size": [processed_width, processed_height],
                    "action_type": action_type,
                    "parsed_action": parsed_action,
                })
                self.reward_trajectory.append({
                    "reward_messages": convert_message_with_img(reward_messages, reward_image_traj),
                    "step_index": step_index,
                    "action_type": action_type,
                })

                self._step_parsed_actions.append(parsed_action)
                self._step_tools_available.append(tools_available)

                break

            except Exception as e:
                import traceback
                logger.error(f"Error during predict (retry {retry_count+1}/{max_retry}): {e}\n{traceback.format_exc()}")
                self._failed_attempts.append({
                    "retry": retry_count,
                    "response": response,
                    "error": str(e),
                })
                retry_count += 1
                if retry_count >= max_retry:
                    # Persist the failed step so eval/training can inspect why
                    # parsing collapsed (e.g. model lost <tool_call> format).
                    # Without this we silently drop the trajectory entry and
                    # lose all evidence of the model's raw output.
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
                    # Keep per-step aggregate lists in sync so summarize() /
                    # monitoring see "parse_failed" for this step.
                    # record_exec_result() skips its append when it sees
                    # this state (guard added below).
                    self._step_action_types.append("parse_failed")
                    self._step_exec_oks.append(False)
                    self._step_exec_results.append({"ok": False, "error": str(e), "parse_failed": True})
                    self._step_tools_available.append(tools_available)
                    self._step_parsed_actions.append({"action_type": "parse_failed", "parse_error": str(e)})
                    self._history_conclusions.append("")
                    self._history_app_infos.append((obs.get("app_info") if isinstance(obs, dict) else None) or "")
                    self._history_exec_feedback.append("")
                    return str(e), ["FAIL"], {"error": str(e), "parse_failed": True}
                continue

        other_info = {
            "action": response.split("\n")[0] if response else "",
            "action_type": parsed_action.get("action_type", "gui"),
            "parsed_action": parsed_action,
            "tools_available": tools_available,
            "num_tools_retrieved": len(retrieved_tools),
        }

        self.observations.append(obs)
        self.actions.append(response.split("\n")[0] if response else "")
        self.responses.append(response)
        self.screenshots.append(processed_image_b64)
        self._screenshot_hashes.append(self._img_hash(processed_image_b64))
        # Record conclusion for owl-style history (used by next step's history_str).
        # owl_parser.parse_action_fncall_id already falls back to thought when
        # conclusion is empty (owl_parser.py:43-44); double-truncating to 200
        # chars here just produces incoherent history snippets.
        self._history_conclusions.append(parsed_action.get("conclusion", "").strip())
        # Record app_info (active window title) for next step's prompt — model
        # uses this to identify current file / sheet / slide context.
        self._history_app_infos.append((obs.get("app_info") if isinstance(obs, dict) else None) or "")

        current_step = len(self.actions)

        # NOTE: a rollout-side loop-break (force FAIL after N consecutive identical
        # actions) was tried in e10 run1 and REMOVED — it misfired on 81% of
        # held-out trajectories (normal GUI retries like re-clicking a dock icon
        # look identical) and, combined with the length penalty, taught the model
        # to "fail fast by getting stuck". never-terminate is now handled by the
        # </tool_call> stop sequence + reward-side length/repeat penalties instead.
        if current_step >= self.max_steps and not any(
            a in ("DONE", "FAIL") for a in pyautogui_actions
        ):
            logger.warning(f"Reached maximum steps {self.max_steps}. Forcing termination.")
            pyautogui_actions = ["FAIL"]

        return response, pyautogui_actions, other_info

    # ── owl-style messages assembly ──────────────────────────────────────

    @staticmethod
    def _img_hash(b64: str) -> str:
        """md5 of a processed b64 frame. process_image is deterministic (fixed
        resize + PNG re-encode), so equal b64 strings ⟺ pixel-identical frames;
        this hash is just an O(1) comparison key (and a future pHash hook)."""
        return hashlib.md5(b64.encode("utf-8")).hexdigest()

    def _build_messages_owl(
        self,
        instruction: str,
        step_index: int,
        retrieved_tools: List[Dict[str, Any]],
        current_image_b64: str,
        original_width: int,
        original_height: int,
        processed_width: int,
        processed_height: int,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Build (messages, image_traj) using the exact pure-GUI system prompt format.

        Same <tools> / <tool_call> grammar the model was trained on — MCP tools are
        just additional entries in the same <tools> block alongside computer_use.
        No NousFnCallPrompt, no <thinking>/<conclusion> overhead.

        Layout:
          system: pure-GUI format system prompt (computer_use + MCP tools)
          user:   [optional "Tool result: ..." text] [screenshot_i]   ← past step i
          asst:   [raw response_i]
          ...
          user:   [optional "Tool result: ..." text] [screenshot_current] [instruction_prompt]
        """
        if "claude" in self.model.lower():
            resized_w, resized_h = 1280, 720
        else:
            resized_w, resized_h = 1000, 1000

        # ── Build system prompt identical to qwen3vl_agent_local.py ─────────
        description_prompt = "\n".join([
            "Use a mouse and keyboard to interact with a computer, and take screenshots.",
            "* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.",
            "* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.",
            f"* The screen's resolution is {resized_w}x{resized_h}.",
            "* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.",
            "* If you tried clicking on a program or link but it failed to load even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.",
            "* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.",
        ])

        action_description_prompt = """
* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.
* `type`: Type a string of text on the keyboard.
* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.
* `left_click`: Click the left mouse button at a specified (x, y) pixel coordinate on the screen.
* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.
* `right_click`: Click the right mouse button at a specified (x, y) pixel coordinate on the screen.
* `middle_click`: Click the middle mouse button at a specified (x, y) pixel coordinate on the screen.
* `double_click`: Double-click the left mouse button at a specified (x, y) pixel coordinate on the screen.
* `scroll`: Performs a scroll of the mouse scroll wheel.
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and report its completion status."""

        computer_use_def = {
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
                                "key", "type", "mouse_move", "left_click",
                                "left_click_drag", "right_click", "middle_click",
                                "double_click", "scroll", "wait", "terminate",
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

        # MCP tools in same format
        all_tool_defs = [computer_use_def]
        for t in retrieved_tools:
            if "parameters" in t and "params" not in t:
                mcp_def = {"type": "function", "function": dict(t)}
            else:
                fn = _registry_tool_to_qwen_function(t)
                mcp_def = {"type": "function", "function": fn}
            mcp_def["function"].setdefault("args_format", "Format the arguments as a JSON object.")
            all_tool_defs.append(mcp_def)

        # ── Tool-doc enhancement gate (OSWORLD_TOOL_DOC_ENHANCE=1, default off) ──
        # Prompt-side seeding of correct tool usage (see tool_doc_hints.py header).
        # Must run BEFORE tools_block serialization so hints land in <tools>.
        _doc_enhance = os.environ.get("OSWORLD_TOOL_DOC_ENHANCE") == "1"
        if _doc_enhance:
            from mm_agents.tool_doc_hints import enhance_tool_defs
            _n_enh = enhance_tool_defs(all_tool_defs)
            logger.info(f"[tool_doc_enhance] enhanced {_n_enh} tool descriptions")

        tools_block = "\n".join(json.dumps(td) for td in all_tool_defs)
        # MCP preference hint: only shown when MCP tools are actually available.
        mcp_hint = ""
        if len(all_tool_defs) > 1:  # more than just computer_use
            mcp_hint = (
                "\n**Tool preference**: When a listed MCP tool can directly accomplish the "
                "current step (e.g., libreoffice_writer.set_line_spacing for a spacing task), "
                "call it by name instead of using computer_use GUI actions. "
                "Call MCP tools as: {\"name\": \"tool_name\", \"arguments\": {...}}. "
                "Only use computer_use when no MCP tool fits.\n"
            )
            if _doc_enhance:
                from mm_agents.tool_doc_hints import GENERIC_DISCIPLINE
                mcp_hint = mcp_hint + GENERIC_DISCIPLINE

        system_prompt = (
            "# Tools\n\n"
            "You may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n"
            "<tools>\n"
            + tools_block
            + "\n</tools>\n"
            + mcp_hint
            + "\nFor each function call, return a json object with function name and arguments "
            "within <tool_call></tool_call> XML tags:\n"
            "<tool_call>\n"
            '{"name": <function-name>, "arguments": <args-json-object>}\n'
            "</tool_call>\n\n"
            "# Response format\n\n"
            "Response format for every step:\n"
            "1) Action: a short imperative for what to do next (MCP tool call or GUI interaction).\n"
            "2) A single <tool_call>...</tool_call> block containing only the JSON: "
            '{"name": <function-name>, "arguments": <args-json-object>}.\n\n'
            "Rules:\n"
            "- Output exactly in the order: Action, <tool_call>.\n"
            "- Be brief: one sentence for Action.\n"
            "- Do not output anything else outside those parts.\n"
            "- If finishing, use action=terminate in the tool call."
        )

        system_msg = {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        }

        # Build previous_actions_str from conclusions — identical structure to pure GUI
        # but uses conclusion text (meaningful summaries) instead of raw response first line.
        prev_lines = []
        for i, conclusion in enumerate(self._history_conclusions):
            prev_lines.append(f"Step {i+1}: {conclusion}" if conclusion else f"Step {i+1}: (action taken)")
        previous_actions_str = "\n".join(prev_lines) if prev_lines else "None"

        instruction_prompt = (
            f"\nPlease generate the next move according to the UI screenshot, instruction and previous actions.\n\n"
            f"Instruction: {instruction}\n\n"
            f"Previous actions:\n"
            f"{previous_actions_str}"
        )

        messages: List[Dict[str, Any]] = [system_msg]
        image_traj: List[str] = []

        start_i = max(0, step_index - self.max_image_history_length + 1)

        # Carry over MCP feedback from the step just before the image window
        # so the first in-window user message can still show the tool result.
        pending_mcp_feedback = ""
        if start_i > 0 and (start_i - 1) < len(self._history_exec_feedback):
            pending_mcp_feedback = self._history_exec_feedback[start_i - 1]

        # Running count of consecutive frames omitted by skip_on_no_change within
        # the rendered window; reset whenever a real image is emitted. Caps at
        # self._max_consecutive_skips to force periodic visual re-alignment.
        consec_skip = 0

        for i in range(start_i, step_index):
            # user turn: [optional tool result] + screenshot (or placeholder)
            user_content: List[Dict[str, Any]] = []
            if pending_mcp_feedback:
                fb = pending_mcp_feedback
                if len(fb) > 1500:
                    fb = fb[:1500] + "...(truncated)"
                user_content.append({"type": "text", "text": f"Tool result: {fb}\n"})
                pending_mcp_feedback = ""

            # skip_on_mcp_success: omit screenshot when step i-1 was a successful MCP call
            _skip_hist = (
                self._context_policy == "skip_on_mcp_success"
                and i > 0
                and (i - 1) < len(self._step_action_types)
                and self._step_action_types[i - 1] == "mcp"
                and self._step_exec_oks[i - 1]
            )
            # skip_on_no_change: omit screenshot when byte-identical to the frame just
            # before it (lossless). Compares against screenshots[i-1] regardless of
            # whether that frame was itself rendered/skipped (pixel-equality is
            # transitive), so it stays orthogonal to the image-history window.
            _skip_nochange_hist = (
                self._context_policy == "skip_on_no_change"
                and i > 0
                and self._screenshot_hashes[i] == self._screenshot_hashes[i - 1]
                and consec_skip < self._max_consecutive_skips
            )
            if _skip_hist:
                user_content.append({
                    "type": "text",
                    "text": "[Screenshot omitted: previous MCP tool call succeeded]\n",
                })
            elif _skip_nochange_hist:
                user_content.append({
                    "type": "text",
                    "text": "[Screenshot omitted: identical to previous frame]\n",
                })
                consec_skip += 1
            else:
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{self.screenshots[i]}"},
                })
                image_traj.append(os.path.join(self.example_result_dir, f"step_{i}.png"))
                consec_skip = 0

            messages.append({"role": "user", "content": user_content})

            # assistant turn: raw model response
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": self.responses[i]}],
            })

            # Queue MCP feedback (non-empty only for MCP steps) for next user message
            if i < len(self._history_exec_feedback):
                pending_mcp_feedback = self._history_exec_feedback[i]

        # Current user message: [optional tool result] + screenshot (or placeholder) + instruction
        curr_user_content: List[Dict[str, Any]] = []
        if pending_mcp_feedback:
            fb = pending_mcp_feedback
            if len(fb) > 1500:
                fb = fb[:1500] + "...(truncated)"
            curr_user_content.append({"type": "text", "text": f"Tool result: {fb}\n"})

        # skip_on_mcp_success: omit current screenshot when last action was successful MCP
        _skip_curr = (
            self._context_policy == "skip_on_mcp_success"
            and self._step_action_types
            and self._step_action_types[-1] == "mcp"
            and self._step_exec_oks[-1]
        )
        # skip_on_no_change: omit current screenshot when byte-identical to the last
        # stored frame. The current frame is not yet appended to self.screenshots at
        # this point, so screenshots[-1]/_screenshot_hashes[-1] is the previous frame.
        _skip_nochange_curr = (
            self._context_policy == "skip_on_no_change"
            and step_index > 0
            and len(self._screenshot_hashes) > 0
            and self._img_hash(current_image_b64) == self._screenshot_hashes[-1]
            and consec_skip < self._max_consecutive_skips
        )
        if _skip_curr:
            curr_user_content.append({
                "type": "text",
                "text": "[Screenshot omitted: previous MCP tool call succeeded]\n",
            })
        elif _skip_nochange_curr:
            curr_user_content.append({
                "type": "text",
                "text": "[Screenshot omitted: identical to previous frame]\n",
            })
        else:
            curr_user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{current_image_b64}"},
            })
            image_traj.append(os.path.join(self.example_result_dir, f"step_{step_index}.png"))

        curr_user_content.append({"type": "text", "text": instruction_prompt})
        messages.append({"role": "user", "content": curr_user_content})

        return messages, image_traj

    def _build_history_str(self) -> str:
        """Build owl-style history string from conclusions + exec results."""
        if not self._history_conclusions:
            return "No history. This is the first step."
        parts: List[str] = []
        for idx, conclusion in enumerate(self._history_conclusions):
            if conclusion:
                parts.append(f"Step {idx + 1}: {conclusion}")
            if idx < len(self._history_exec_feedback) and self._history_exec_feedback[idx]:
                parts.append(f"Tool call result: {self._history_exec_feedback[idx]}")
        return "\n".join(parts) if parts else "No history. This is the first step."

    # ── execution result recording ──────────────────────────────────────

    def record_exec_result(self, action_type: str, exec_ok: bool, exec_result: Dict[str, Any]):
        """Called by run_single_example_hybrid after action execution.

        - Updates per-step metadata (action_types, exec_oks, exec_results).
        - For MCP steps, stores a compact result string for owl-style history.
        - Enriches the latest reward_trajectory entry with execution context.
        """
        # parse_failed steps are fully recorded inside predict()'s except
        # branch (it appends to all _step_* lists before returning FAIL).
        # The runner still calls us once after env.step("FAIL") — skip the
        # second append so the per-step lists stay in sync.
        if (
            self._step_action_types
            and self._step_action_types[-1] == "parse_failed"
            and len(self._step_exec_oks) == len(self._step_action_types)
            and len(self._step_exec_results) == len(self._step_action_types)
        ):
            return
        self._step_action_types.append(action_type)
        self._step_exec_oks.append(exec_ok)
        self._step_exec_results.append(exec_result)

        if action_type == "mcp":
            # Match OSWorld-MCP: feed the raw str(CallToolResult) back as the
            # model's "Action Response" / history slot. lib_run_single
            # _execute_mcp captured this in exec_result['result'] (up to 3000
            # chars). Strip the JSON wrapper so the model sees
            #   CallToolResult(content=[TextContent(text='...')], is_error=False)
            # rather than {"ok": true, "result": "CallToolResult(...)"}.
            raw = exec_result if isinstance(exec_result, dict) else {}
            if exec_ok:
                feedback = raw.get("result") or ""
            else:
                feedback = f"ERROR: {raw.get('error', '')}"
            # Gate (OSWORLD_TOOL_DOC_ENHANCE=1): flag silent no-op results
            # ("0 replacements" + success:true) so the model doesn't treat
            # execution success as goal progress. Agent-side only.
            if os.environ.get("OSWORLD_TOOL_DOC_ENHANCE") == "1":
                from mm_agents.tool_doc_hints import zero_effect_warning
                feedback = zero_effect_warning(feedback)
            self._history_exec_feedback.append(feedback)
        else:
            self._history_exec_feedback.append("")

        if not self.reward_trajectory:
            return

        last = self.reward_trajectory[-1]
        reward_messages = last.get("reward_messages", [])
        if not reward_messages:
            return

        last_msg = reward_messages[-1]
        content = last_msg.get("content")
        if not isinstance(content, list):
            return

        exec_context = self._build_exec_result_context(action_type, exec_ok, exec_result)
        if exec_context:
            content.append({"type": "text", "text": exec_context})

    def _build_exec_result_context(
        self, action_type: str, exec_ok: bool, exec_result: Dict[str, Any],
    ) -> str:
        """Format execution results for the reward model based on action type."""
        raw = exec_result if isinstance(exec_result, dict) else {}

        if action_type == "mcp":
            result_json = json.dumps(raw, indent=2, ensure_ascii=False, default=str)
            if len(result_json) > 2000:
                result_json = result_json[:2000] + "\n...(truncated)"
            return (
                f"\nTOOL RETURN VALUE:\n{result_json}\n\n"
                f"Evaluate:\n"
                f"1. Did the tool call succeed (ok={exec_ok})?\n"
                f"2. Does the return value indicate the intended operation was performed?\n"
                f"3. Was MCP the right choice for this step?\n"
                f"NOTE: Screenshots may show no visible change — tool operations "
                f"often modify data without visual effect.\n"
            )
        else:
            return (
                f"\nEvaluate:\n"
                f"1. Did the GUI action produce a visible, meaningful change on screen?\n"
                f"2. Was GUI the right choice, or was an MCP tool available for this step?\n"
            )

    # ── record_step_outcome override ────────────────────────────────────

    def record_step_outcome(self, next_obs: Dict):
        """Phase 3: append next observation + evaluation instructions to reward msg."""
        if not self.reward_trajectory:
            if logger:
                logger.warning("record_step_outcome called but reward_trajectory is empty.")
            return

        last = self.reward_trajectory[-1]
        step_index = last.get("step_index")
        if step_index is None:
            step_index = max(0, len(self.actions) - 1)

        next_image_path = os.path.join(self.example_result_dir, f"step_{step_index + 1}.png")
        action_type = last.get("action_type", "gui")

        reward_messages = last.get("reward_messages", [])
        if not reward_messages:
            return
        last_msg = reward_messages[-1]
        content = last_msg.get("content")
        if not isinstance(content, list):
            return

        content.append({"type": "text", "text": "\nNext observation after executing this action:\n"})
        content.append({"type": "image", "image": next_image_path})

        post_template = self._get_hybrid_post_template(action_type)
        content.append({"type": "text", "text": post_template})

        last["next_obs_path"] = next_image_path
        if self.trajectory:
            self.trajectory[-1]["next_obs_path"] = next_image_path

    def _get_hybrid_post_template(self, action_type: str) -> str:
        """Return action-type-aware evaluation instructions."""
        base = (
            "\nEvaluate ONLY the single most recent step using the information above.\n"
            "Use the next observation AFTER executing this step to judge whether the action took effect.\n\n"
        )
        if action_type == "mcp":
            specific = (
                "For MCP tool calls: the screenshot may NOT change even on success — many tools "
                "modify data layers without visual effect. Weigh the TOOL RETURN VALUE more heavily "
                "than the screenshot diff. A successful tool call with correct parameters that advances "
                "the task goal deserves +1 even if the screenshots look identical.\n\n"
            )
        else:
            specific = (
                "For GUI actions: compare the two screenshots carefully. Did this action produce a visible, "
                "meaningful change that moves toward the task goal?\n\n"
            )
        scoring = (
            "Assign +1 if: the step is relevant, the action type choice was reasonable, "
            "and the result indicates progress.\n"
            "Assign -1 if: the step is incorrect, irrelevant, uses a clearly suboptimal action type, "
            "or shows no progress.\n\n"
            "Think carefully, then provide your reasoning and put the final score in \\boxed{}.\n"
        )
        return base + specific + scoring

    # ── summarize ───────────────────────────────────────────────────────

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
            "action_types": self._step_action_types,
            "exec_oks": self._step_exec_oks,
            "tools_available": self._step_tools_available,
        }

        tmp_path = out_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, out_path)
        return out_path

    # ── reward message construction (unchanged from previous version) ────

    def _build_base_reward_context(
        self,
        instruction: str,
        step_index: int,
        current_screenshot_b64: str,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Build the reward user content up to current observation (action appended later)."""
        reward_user_content: List[Dict[str, Any]] = []
        reward_image_traj: List[str] = []

        rstart_i = max(0, step_index - self.max_reward_image_history_length + 1)

        prev_lines = []
        for i in range(rstart_i):
            a = self.actions[i]
            prev_lines.append(f"Step {i+1}: {a}")
        previous_reward_actions_str = "\n".join(prev_lines) if prev_lines else "None"
        reward_user_content.append(
            {"type": "text", "text": f"Previous Actions:\n {previous_reward_actions_str}"}
        )

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

        return reward_user_content, reward_image_traj

    def _finalize_reward_messages(
        self,
        instruction: str,
        step_index: int,
        action_type: str,
        parsed_action: Dict[str, Any],
        base_user_content: List[Dict[str, Any]],
        base_image_traj: List[str],
        available_tools: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Phase 1: base context + action description. Phases 2/3 added later."""
        from mm_agents.opencua.prompts import REWARD_SYSTEM_PROMPT_V1_L2

        reward_user_content = copy.deepcopy(base_user_content)

        tools_summary = ", ".join(t.get("name", "") for t in (available_tools or [])[:10]) or "none"
        action_context = self._build_action_context_for_reward(
            instruction, step_index, action_type, parsed_action, tools_summary,
        )
        reward_user_content.append({"type": "text", "text": action_context})

        reward_messages = [{"role": "system", "content": REWARD_SYSTEM_PROMPT_V1_L2}]
        reward_messages.append({"role": "user", "content": reward_user_content})
        return reward_messages

    def _build_action_context_for_reward(
        self,
        instruction: str,
        step_index: int,
        action_type: str,
        parsed_action: Dict[str, Any],
        tools_summary: str,
    ) -> str:
        """Build the action description block for reward evaluation."""
        header = (
            f"\nYou are evaluating the most recent step of an AI agent.\n"
            f"Objective: {instruction}\n"
            f"Step: {step_index + 1}\n"
            f"Action type chosen: {action_type}\n"
        )

        if action_type == "mcp":
            tool_name = parsed_action.get("tool_name", "")
            params = json.dumps(parsed_action.get("params", {}), indent=2)
            detail = (
                f"MCP TOOL CALLED: {tool_name}\n"
                f"PARAMS: {params}\n"
            )
        else:
            action_str = parsed_action.get("action", "")
            detail = (
                f"GUI ACTION: {action_str}\n"
                f"Available MCP tools: {tools_summary}\n"
            )

        return header + detail
