# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Shared multimodal HTTP client and hashes for benchmark tools."""

import base64
import hashlib
import io
from functools import lru_cache
from pathlib import Path

from common import normalized_text


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def question_hash(question: str) -> str:
    return sha256(normalized_text(question.replace("<image>", "")).encode())


def image_hash(payload: bytes) -> str:
    from PIL import Image

    with Image.open(io.BytesIO(payload)) as source:
        image = source.convert("RGB")
        return sha256(str(image.size).encode() + image.tobytes())


def inference_chat_kwargs(processor) -> dict[str, bool]:
    template = processor.chat_template
    if type(processor).__name__ == "Qwen3VLProcessor" and isinstance(template, str):
        if not any(marker in template for marker in ("<think>", "enable_thinking", "reasoning_content")):
            return {}
    return {"enable_thinking": True, "preserve_thinking": True}


def preserve_history_template(processor) -> str:
    if not inference_chat_kwargs(processor):
        return processor.chat_template
    from relax.engine.sft.dataset.qwen_chat_template_patch import try_patch_qwen_chat_template
    from relax.engine.sft.dataset.sample import CanonicalMessage, CanonicalSample

    sample = CanonicalSample(
        messages=[CanonicalMessage(role="assistant", content="", learn=False)],
        metadata={"source_dataset": "benchmark", "row_index": 0},
    )
    patched = try_patch_qwen_chat_template(sample, processor.chat_template, {"preserve_thinking": True})
    if patched is None:
        raise ValueError("Expected a supported Qwen history template")
    processor.chat_template = patched.template
    processor.tokenizer.chat_template = patched.template
    return patched.template


class HTTPRunner:
    def __init__(self, processor_path: Path, client, model: str) -> None:
        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(processor_path, local_files_only=True)
        self.template_sha256 = sha256(preserve_history_template(self.processor).encode())
        self.chat_template_kwargs = inference_chat_kwargs(self.processor)
        self.client, self.model = client, model
        self.requests: list[dict] = []
        self.error: str | None = None
        self.last_response: str | None = None

    def infer(self, contents, system_instruction=None, cfg=None) -> dict:
        from PIL import Image

        hf_messages = [{"role": "system", "content": system_instruction}]
        messages = [{"role": "system", "content": system_instruction}]
        for item in contents:
            parts, hf_parts = [], []
            for part in item.get("parts", []):
                if "text" in part:
                    parts.append({"type": "text", "text": part["text"]})
                    hf_parts.append(parts[-1])
                else:
                    if "inline_data" in part:
                        encoded = part["inline_data"]["data"]
                        payload = base64.b64decode(encoded.split("base64,", 1)[-1], validate=True)
                    elif "image_url" in part:
                        payload = url_image_bytes(part["image_url"]["url"])
                    else:
                        raise ValueError("Unsupported model-visible content part")
                    with Image.open(io.BytesIO(payload)) as image:
                        mime = Image.MIME[image.format]
                        hf_parts.append({"type": "image", "image": image.convert("RGB")})
                    encoded = base64.b64encode(payload).decode()
                    parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}})
            role = "assistant" if item["role"] == "model" else "user"
            hf_messages.append({"role": role, "content": hf_parts})
            messages.append({"role": role, "content": parts})
        inputs = self.processor.apply_chat_template(
            hf_messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            **self.chat_template_kwargs,
        )
        tokens = int(inputs["input_ids"].shape[-1])
        # Reserve one terminal token; SGLang rejects a request at the exact context boundary.
        remaining = 32768 - tokens - 1
        if remaining <= 0:
            self.error = "context_limit"
            raise ValueError("Complete history exceeds the 32K context budget; no history was truncated")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            max_tokens=remaining,
            extra_body={"chat_template_kwargs": self.chat_template_kwargs},
        )
        if response.usage is not None and response.usage.prompt_tokens != tokens:
            self.error = "token_budget_mismatch"
            raise ValueError(
                "Local and server prompt token counts differ; verify the shared template and image processing"
            )
        choice = response.choices[0]
        text = choice.message.content or ""
        reasoning = getattr(choice.message, "reasoning_content", None)
        if reasoning:
            text = "<think>" + reasoning + "</think>" + text
        self.last_response = text
        self.requests.append(
            {
                "input_tokens": tokens,
                "max_tokens": remaining,
                "finish_reason": choice.finish_reason,
                "usage": response.usage.model_dump() if response.usage else None,
            }
        )
        if choice.finish_reason == "length":
            self.error = "context_limit"
            raise ValueError("Generation exhausted the 32K context budget")
        if choice.finish_reason != "stop":
            self.error = "server_finish_error"
            raise ValueError(f"Model server ended the response with finish_reason={choice.finish_reason!r}")
        return {
            "candidates": [{"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": "STOP"}],
            "modelVersion": response.model,
            "usageMetadata": response.usage.model_dump() if response.usage else {},
        }


@lru_cache(maxsize=128)
def url_image_bytes(url: str) -> bytes:
    import requests

    if url.startswith("data:"):
        return base64.b64decode(url.split("base64,", 1)[1], validate=True)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.content


if __name__ == "__main__":
    raise SystemExit("Run vstar_benchmark.py to evaluate V*Bench.")
