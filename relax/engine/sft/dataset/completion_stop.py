# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Preserve the model's stop-token supervision on completed SFT replies."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from relax.engine.sft.dataset.sample import CanonicalSample


def _stop_ids(value: Any) -> tuple[int, ...]:
    values = value if isinstance(value, (list, tuple)) else (value,)
    return tuple(dict.fromkeys(token_id for token_id in values if type(token_id) is int and token_id >= 0))


def _special_token_id(tokenizer: Any, token: str) -> int | None:
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if not callable(convert):
        return None
    try:
        token_id = convert(token)
    except KeyError:
        return None
    if type(token_id) is int and token_id >= 0 and token_id != getattr(tokenizer, "unk_token_id", None):
        return token_id
    return None


def _checkpoint_config(tokenizer: Any, filename: str) -> dict[str, Any]:
    model = getattr(tokenizer, "name_or_path", None)
    if not isinstance(model, str) or not model:
        return {}
    path = Path(model) / filename
    if not path.is_file():
        if Path(model).is_dir():
            return {}
        from transformers.utils import cached_file

        cached = cached_file(
            model,
            filename,
            revision=getattr(tokenizer, "init_kwargs", {}).get("_commit_hash"),
            local_files_only=True,
            _raise_exceptions_for_missing_entries=False,
            _raise_exceptions_for_connection_errors=False,
        )
        if cached is None:
            return {}
        path = Path(cached)
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class SFTCompletionStop:
    accepts: tuple[int, ...]
    append_id: int | None
    tool_call_boundary: tuple[int, int] | None = None

    @classmethod
    def from_tokenizer(cls, tokenizer: Any) -> "SFTCompletionStop":
        generation = _checkpoint_config(tokenizer, "generation_config.json")
        config = _checkpoint_config(tokenizer, "config.json")
        model_ids = _stop_ids(config.get("eos_token_id")) or _stop_ids(
            (config.get("text_config") or {}).get("eos_token_id")
        )
        tokenizer_ids = _stop_ids(getattr(tokenizer, "eos_token_id", None))
        accepted = _stop_ids(generation.get("eos_token_id")) or model_ids or tokenizer_ids
        # An explicit single model EOS identifies the completion token even
        # when generation additionally stops on role headers or padding.
        preferred = model_ids if len(model_ids) == 1 else tokenizer_ids
        append_id = next((token_id for token_id in preferred if token_id in accepted), None)
        if len(accepted) == 1:
            append_id = accepted[0]
        # GLM closes each call with </tool_call> and ends the group with the
        # observation token, which the model declares as a generation stop.
        tool_handoff_id = _special_token_id(tokenizer, "<|observation|>")
        tool_call_boundary = None
        if tool_handoff_id is not None and tool_handoff_id in accepted:
            tool_call_end_id = _special_token_id(tokenizer, "</tool_call>")
            if tool_call_end_id is not None:
                tool_call_boundary = (tool_call_end_id, tool_handoff_id)
        return cls(accepted, append_id, tool_call_boundary)

    def ensure_supervised(
        self,
        sample: CanonicalSample,
        tokens: torch.Tensor,
        loss_mask: torch.Tensor,
        tokenizer: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.accepts or not sample.messages or tokens.numel() < 2:
            return tokens, loss_mask
        last = sample.messages[-1]
        if last.role != "assistant" or not last.learn:
            return tokens, loss_mask

        append_id = self.append_id
        tool_call_end_id = None
        if last.tool_calls:
            if self.tool_call_boundary is None:
                return tokens, loss_mask
            tool_call_end_id, append_id = self.tool_call_boundary
        else:
            content = last.content
            if not content:
                return tokens, loss_mask
            has_text = (
                bool(content.strip())
                if isinstance(content, str)
                else any(part.get("type") == "text" and part.get("text", "").strip() for part in content)
            )
            if not has_text:
                return tokens, loss_mask

        # Chat templates may emit formatting whitespace after their stop token.
        decode = getattr(tokenizer, "decode", None)
        for index in range(len(tokens) - 1, 0, -1):
            token_id = int(tokens[index])
            if token_id in self.accepts:
                if not loss_mask[index]:
                    loss_mask = loss_mask.clone()
                    loss_mask[index] = 1
                return tokens, loss_mask
            if decode is None or not decode([token_id], skip_special_tokens=False).isspace():
                break

        if tool_call_end_id is not None and token_id != tool_call_end_id:
            raise ValueError(
                f"SFT sample {sample.metadata['row_index']} must end in a closed tool call "
                "before adding the observation handoff."
            )
        if append_id is None:
            raise ValueError(
                f"SFT sample {sample.metadata['row_index']} has no final stop token and the checkpoint "
                f"does not identify a single completion EOS among {self.accepts}. "
                "Use a chat template that emits the model's assistant stop token."
            )
        return torch.cat((tokens, tokens.new_tensor([append_id]))), torch.cat((loss_mask, loss_mask.new_ones(1)))
