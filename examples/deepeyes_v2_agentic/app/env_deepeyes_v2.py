# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""DeepEyes V2 environment adapter for the agentic stack.

Exposes three async tool handlers consumed by ``app.agent``. Sandbox lifecycle
and image collection mirror xide's ``examples/deepeyes_v2/env_deepeyes_v2.py``
verbatim against the ported ``app.sandboxes`` API; only the outer
``BaseInteractionEnv`` shell, ``_mt_stats`` instrumentation, ``_log_trace``
gating, and ``_build_obs`` are dropped. Helpers that did not move with the core
flow (``_fix_python_indentation`` / autopep8 cleanup) are intentionally skipped
— the agent loop in Task 3 already runs the raw LLM code as-is.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import timedelta
from io import BytesIO
from math import ceil
from typing import Any, Optional

from app.prompt import (
    INITIALIZATION_CODE_TEMPLATE,
    RELAX_IMAGES_DIR,
    RELAX_INPUTS_DIR,
    RETURN_CODE_PROMPT,
    RETURN_SEARCH_PROMPT,
)
from app.sandboxes.base import BaseSandboxSession, SandboxCapability
from app.sandboxes.exceptions import SandboxError
from app.sandboxes.executor import SandboxExecutor
from app.search_utils import image_search, search
from PIL import Image, UnidentifiedImageError


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tag patterns (single-source-of-truth, also covered by test_env_extractors.py)
# ---------------------------------------------------------------------------
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
CODE_BLOCK_RE = re.compile(r"<code>(.*?)</code>", re.DOTALL)
PYTHON_FENCE_RE = re.compile(r"```python\s*\n(.*?)\n```", re.DOTALL)
TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

SUPPORTED_TOOL_NAMES = {"search", "image_search"}
MAX_IMAGES_PER_ROUND = 10
MAX_STDOUT_CHARS = 8000
MAX_STDERR_CHARS = 4000
MIN_IMAGE_DIM = 28
MAX_IMAGE_PIXELS = 1280 * 28 * 28


@dataclass
class ToolObs:
    body_text: str
    images: list[Image.Image] = field(default_factory=list)
    done: bool = False
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Pure extractors
# ---------------------------------------------------------------------------
def extract_answer(text: str) -> Optional[str]:
    m = ANSWER_RE.search(text)
    return m.group(1).strip() if m else None


def extract_code(text: str) -> Optional[str]:
    block = CODE_BLOCK_RE.search(text)
    if block:
        fence = PYTHON_FENCE_RE.search(block.group(1))
        if fence:
            return fence.group(1).strip()
    # Unified-schema fallback: <tool_call>{"name":"python_exec","arguments":{"code":"..."}}</tool_call>
    tc = extract_tool_call(text)
    if tc and tc.get("name") == "python_exec":
        args = tc.get("arguments") or {}
        code = args.get("code") if isinstance(args, dict) else None
        if isinstance(code, str) and code.strip():
            return code.strip()
    return None


def extract_tool_call(text: str) -> Optional[dict[str, Any]]:
    """Return ``{"name": str, "arguments": Any | None}`` or ``None``.

    Picks the last ``<tool_call>`` block (xide also iterated last-match) and
    parses its JSON body. Returns ``None`` if no block is present, the JSON is
    malformed, or the payload lacks a string ``name``.
    """
    matches = list(TOOL_CALL_RE.finditer(text))
    if not matches:
        return None
    try:
        payload = json.loads(matches[-1].group(1).strip())
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    name = payload.get("name")
    if not isinstance(name, str):
        return None
    return {"name": name, "arguments": payload.get("arguments")}


# ---------------------------------------------------------------------------
# Image helpers (lifted from xide, identical behaviour)
# ---------------------------------------------------------------------------
def _pil_to_b64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def encode_image_data_uri(image: Image.Image) -> str:
    """Encode a PIL image as a data:image/png;base64,...

    URI for OpenAI image_url content parts.
    """
    return f"data:image/png;base64,{_pil_to_b64(image)}"


def _maybe_resize_image(img: Image.Image) -> Image.Image:
    """Cap pixels + enforce min side / max aspect ratio for Qwen-VL ViT."""
    h, w = img.height, img.width
    if max(h, w) / max(min(h, w), 1) > 200:
        max_v, min_v = max(h, w), max(min(h, w), 1)
        old_scale = max_v / min_v
        max_ratio = min(150, old_scale / 2)
        target_max = int(min_v * max_ratio)
        if h > w:
            nh, nw = target_max, int(w * old_scale / max_ratio)
        else:
            nw, nh = target_max, int(h * old_scale / max_ratio)
        img = img.resize((max(nw, 1), max(nh, 1)), Image.LANCZOS)
        h, w = img.height, img.width
    if h * w > MAX_IMAGE_PIXELS:
        beta = (h * w / MAX_IMAGE_PIXELS) ** 0.5
        img = img.resize((max(1, int(w / beta)), max(1, int(h / beta))), Image.LANCZOS)
        h, w = img.height, img.width
    if min(h, w) >= MIN_IMAGE_DIM:
        return img
    ratio = MIN_IMAGE_DIM / min(h, w)
    return img.resize((ceil(w * ratio), ceil(h * ratio)), Image.LANCZOS)


def _decode_and_resize_image(raw: bytes) -> Optional[Image.Image]:
    try:
        img = Image.open(BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        logger.warning(f"[deepeyes-v2] decode image error (image dropped): {exc}")
        return None
    return _maybe_resize_image(img)


def _clip_chars(text: str, max_chars: int, label: str) -> str:
    if len(text) <= max_chars:
        return text
    dropped = len(text) - max_chars
    return text[:max_chars] + (
        f"\n...[truncated {dropped} {label} chars by env_deepeyes_v2 to keep "
        f"multi-turn context bounded; reduce print volume in your next code]"
    )


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------
class DeepEyesV2Env:
    """Async tool handlers for one trajectory.

    Construction is cheap (no network). Sandbox provisioning happens lazily in
    :meth:`exec_code` so a trajectory that answers in the first turn never
    pays the apptainer launch cost.
    """

    def __init__(
        self,
        *,
        data_index: str,
        sandbox_executor: SandboxExecutor,
        image: Optional[Image.Image],
        code_timeout_s: int = 200,
        ensure_sandbox_timeout_s: int = 240,
    ) -> None:
        self.data_index = data_index
        self._executor = sandbox_executor
        self._origin_image = image
        self._code_timeout_s = code_timeout_s
        self._ensure_sandbox_timeout_s = ensure_sandbox_timeout_s
        self._session: Optional[BaseSandboxSession] = None
        self._session_ctx = None
        self._initialized = False
        self._init_lock = asyncio.Lock()

    # ---- public tool handlers ---------------------------------------------

    async def exec_code(self, response_text: str) -> ToolObs:
        code = extract_code(response_text)
        if not code:
            return ToolObs(
                body_text=(
                    "Error: <tool_call> was received but the 'code' field is empty or missing. "
                    "Put your Python code as a non-empty string in arguments.code, e.g. "
                    '<tool_call>{"name": "python_exec", "arguments": {"code": "print(image_1.size)"}}</tool_call>.'
                ),
                done=False,
                error="code_extract_failed",
            )
        try:
            await asyncio.wait_for(self._ensure_sandbox(), timeout=self._ensure_sandbox_timeout_s)
        except asyncio.TimeoutError:
            logger.warning(f"[deepeyes-v2] {self.data_index} ensure_sandbox timed out")
            return ToolObs(
                body_text="Error: sandbox startup timed out; skip tool execution and continue reasoning or answer directly.",
                done=False,
                error="ensure_sandbox_timeout",
            )
        except SandboxError as exc:
            logger.warning(f"[deepeyes-v2] {self.data_index} sandbox provision failed: {exc}")
            return ToolObs(
                body_text="Error: sandbox unavailable; skip tool execution and continue reasoning or answer directly.",
                done=False,
                error=f"sandbox_provision_failed:{type(exc).__name__}",
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception(f"[deepeyes-v2] {self.data_index} ensure_sandbox raised")
            return ToolObs(
                body_text="Error: sandbox unavailable; skip tool execution and continue reasoning or answer directly.",
                done=False,
                error=f"sandbox_provision_failed:{type(exc).__name__}",
            )

        result = await self._run_code_in_sandbox(code)
        if result is None:
            return ToolObs(
                body_text="Error: code execution failed; check your code or continue reasoning without the tool.",
                done=False,
                error="sandbox_exec_failed",
            )
        images = result["images"][:MAX_IMAGES_PER_ROUND]
        marker = "Images:\n" + "<image>" * len(images) if images else ""
        body = RETURN_CODE_PROMPT.format(
            stdout=result["stdout"],
            stderr=result["stderr"],
            image=marker,
        ).strip()
        return ToolObs(body_text=body, images=images, done=False, error=None)

    async def exec_tool(self, response_text: str) -> ToolObs:
        parsed = extract_tool_call(response_text)
        if parsed is None:
            return ToolObs(
                body_text=(
                    "Error: <tool_call> block could not be parsed. Emit a single <tool_call>{...}</tool_call> "
                    "containing valid JSON with a 'name' field (e.g. 'search' or 'image_search') and optional 'arguments'."
                ),
                done=False,
                error="tool_call_extract_failed",
            )
        name = parsed["name"]
        args = parsed["arguments"]
        if name not in SUPPORTED_TOOL_NAMES:
            return ToolObs(
                body_text=f"Error: invalid tool call name: {name}.",
                done=False,
                error="invalid_tool_name",
            )
        if name == "image_search" and args is not None:
            return ToolObs(
                body_text=f"Error: invalid tool call parameters for image search: {args}.",
                done=False,
                error="invalid_image_search_args",
            )
        try:
            result = await asyncio.to_thread(self._dispatch_search, name, args)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[deepeyes-v2] {self.data_index} search failed: {exc}")
            return ToolObs(
                body_text=f"Error: {exc} for {self.data_index}",
                done=False,
                error=str(exc),
            )
        if not result or result.get("status") != "success":
            return ToolObs(
                body_text="Error: search returned no result; continue reasoning or answer directly.",
                done=False,
                error="search_failed",
            )
        images = result.get("images", [])[:MAX_IMAGES_PER_ROUND]
        body = RETURN_SEARCH_PROMPT.format(search_result=result.get("result", "")).strip()
        return ToolObs(body_text=body, images=images, done=False, error=None)

    async def close(self) -> None:
        ctx = self._session_ctx
        if ctx is None:
            return
        self._session_ctx = None
        self._session = None
        self._initialized = False
        try:
            await ctx.__aexit__(None, None, None)
        except Exception as exc:  # noqa: BLE001
            # Container leak risk: surface as ERROR so SRE dashboards see it.
            logger.error(f"[deepeyes-v2] {self.data_index} session close failed (container may leak): {exc}")

    # ---- internals --------------------------------------------------------

    async def _ensure_sandbox(self) -> None:
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            session_ctx = self._executor.acquire_session(
                metadata={
                    "rollout": "deepeyes_v2",
                    "data_index": str(self.data_index)[:64],
                },
                require=frozenset({SandboxCapability.STATEFUL_KERNEL, SandboxCapability.FILE_IO}),
            )
            session: BaseSandboxSession = await session_ctx.__aenter__()
            try:
                await session.wait_ready(timedelta(seconds=self._ensure_sandbox_timeout_s))
                if self._origin_image is None:
                    init_code = (
                        f'import os\nos.makedirs("{RELAX_IMAGES_DIR}", exist_ok=True)\n'
                        f'os.makedirs("{RELAX_INPUTS_DIR}", exist_ok=True)\n'
                        f'os.chdir("{RELAX_INPUTS_DIR}")\n'
                        f'image_1 = None\nprint("relax sandbox initialized; no image_1")\n'
                    )
                else:
                    b64 = await asyncio.to_thread(_pil_to_b64, self._origin_image)
                    init_code = INITIALIZATION_CODE_TEMPLATE.format(
                        base64_image=b64,
                        relax_img_dir=RELAX_IMAGES_DIR,
                        relax_inputs_dir=RELAX_INPUTS_DIR,
                    )
                result = await session.run_code(init_code, timeout=self._code_timeout_s)
                if result.status != "success":
                    raise SandboxError(f"sandbox initialization {result.status}")
            except BaseException:
                # Clean up the half-built session before re-raising; mirrors
                # xide's bootstrap. Logged in close() if the exit itself fails.
                try:
                    await session_ctx.__aexit__(None, None, None)
                except Exception as exit_exc:  # noqa: BLE001
                    logger.error(
                        f"[deepeyes-v2] {self.data_index} partial-init session exit "
                        f"failed (container may leak): {exit_exc}"
                    )
                raise
            self._session = session
            self._session_ctx = session_ctx
            self._initialized = True

    async def _run_code_in_sandbox(self, code: str) -> Optional[dict]:
        session = self._session
        assert session is not None, "_run_code_in_sandbox called before _ensure_sandbox"
        try:
            result = await session.run_code(code, timeout=self._code_timeout_s)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[deepeyes-v2] {self.data_index} sandbox.run_code failed: {exc!r}")
            return None
        if result.status == "timeout":
            await self.close()
            return {"stdout": "", "stderr": "Execution timed out.", "status": "error", "images": []}

        stdout = _clip_chars(result.stdout, MAX_STDOUT_CHARS, "stdout")
        stderr = _clip_chars(result.stderr, MAX_STDERR_CHARS, "stderr")

        try:
            entries = await session.list_files(RELAX_IMAGES_DIR)
        except (SandboxError, asyncio.TimeoutError) as exc:
            entries = []
            logger.warning(f"[deepeyes-v2] {self.data_index} list_files failed: {exc}")

        img_entries = [e for e in entries if e.path.lower().endswith((".png", ".jpg", ".jpeg"))]
        img_paths = [e.path for e in img_entries[:MAX_IMAGES_PER_ROUND]]
        images: list[Image.Image] = []
        if img_paths:
            raws = await asyncio.gather(
                *(session.read_bytes(p) for p in img_paths),
                return_exceptions=True,
            )
            for path, raw in zip(img_paths, raws):
                if isinstance(raw, BaseException):
                    logger.warning(f"[deepeyes-v2] read image {path} failed: {raw}")
            decoded = await asyncio.gather(
                *(
                    asyncio.to_thread(_decode_and_resize_image, raw)
                    for raw in raws
                    if not isinstance(raw, BaseException)
                )
            )
            for img in decoded:
                if img is not None:
                    images.append(img)

        if entries:
            try:
                await session.delete_files([e.path for e in entries])
            except (SandboxError, asyncio.TimeoutError) as exc:
                logger.warning(f"[deepeyes-v2] {self.data_index} delete_files failed: {exc}")

        return {"status": result.status, "stdout": stdout, "stderr": stderr, "images": images}

    def _dispatch_search(self, tool_name: str, tool_args: Any) -> dict:
        """Sync helper invoked via ``asyncio.to_thread``.

        Mirrors xide's
        ``_request_search``: turns the raw search/image_search payloads into
        a uniform ``{status, result, images}`` dict the caller renders.
        """
        if tool_name == "image_search":
            result = image_search(tool_args, self.data_index)
            if result == "Error":
                return {"status": "error", "result": "Error", "images": []}
            images: list[Image.Image] = []
            snippets: list[str] = []
            try:
                titles = result["tool_returned_web_title"]
                paths = result["cached_images_path"]
                for idx, (title, path) in enumerate(zip(titles, paths)):
                    img = Image.open(path)
                    images.append(_maybe_resize_image(img))
                    snippets.append(f"{idx + 1}. <image>\n[{title}] \n")
                content = (
                    f"A Google image search for the image found {len(snippets)} results:"
                    f"\n\n## Web Results\n" + "\n\n".join(snippets)
                )
                return {"status": "success", "result": content, "images": images}
            except (FileNotFoundError, UnidentifiedImageError, OSError, KeyError) as exc:
                return {
                    "status": "error",
                    "result": f"{exc} No results found for the image. Try with text search or direct output the answer.",
                    "images": [],
                }

        # tool_name == "search"
        query = tool_args["query"] if isinstance(tool_args, dict) and "query" in tool_args else str(tool_args)
        result = search(query)
        if result == "Error":
            return {"status": "error", "result": "Error", "images": []}
        snippets: list[str] = []
        try:
            for idx, page in enumerate(result["data"]):
                date_published = ""
                if page.get("date") is not None:
                    date_published = "\nDate published: " + page["date"]
                snippet = ""
                if page.get("snippet") is not None:
                    snippet = "\n" + page["snippet"]
                snippets.append(f"{idx + 1}. [{page['title']}]({page['link']}){date_published}{snippet}")
            content = (
                f"A Google search for '{query}' found {len(snippets)} results:"
                f"\n\n## Web Results\n" + "\n\n".join(snippets)
            )
        except (KeyError, TypeError) as exc:
            return {
                "status": "error",
                "result": f"{exc} No results found for '{query}'. Try with a more general query.",
                "images": [],
            }
        return {"status": "success", "result": content, "images": []}
