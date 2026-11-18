# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Opt-in, per-sample Judge checkpoints for the standard SGLang rollout."""

import asyncio
import hashlib
import inspect
import json
from argparse import Namespace
from collections.abc import Callable
from typing import Any

import httpx
import ray

from relax.utils.http_utils import _post, is_expected_sglang_499
from relax.utils.logging_utils import get_logger
from relax.utils.misc import load_function
from relax.utils.types import Sample


logger = get_logger(__name__)
_CHECKPOINT_KEY = "_partial_genrm"


class PartialGenRMAborted(asyncio.CancelledError):
    """Unfinished scoring; must not be converted to a reward by except
    Exception."""


class PartialGenRM:
    def __init__(self, args: Namespace, state: Any) -> None:
        self.args = args
        self.state = state
        self.tokenizers: dict[str, Any] = {}
        self.engine_urls: dict[str, list[str]] = {}
        self.next_engine: dict[str, int] = {}
        self.active: dict[str, int] = {}
        self._client: httpx.AsyncClient | None = None

    @classmethod
    def from_args(cls, args: Namespace, state: Any) -> "PartialGenRM | None":
        """An explicit callback parameter opts an async custom reward in."""
        if not getattr(args, "partial_rollout", False) or not getattr(args, "custom_rm_path", None):
            return None
        reward = load_function(args.custom_rm_path)
        parameter = inspect.signature(reward).parameters.get("genrm_generate")
        if parameter is None:
            return None
        if parameter.kind not in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
            raise ValueError("genrm_generate must accept a keyword argument")
        if parameter.default is not None:
            raise ValueError("genrm_generate must default to None for evaluation and non-partial rollout")
        if not inspect.iscoroutinefunction(reward):
            raise ValueError("Partial GenRM requires an async custom reward")
        if not (getattr(args, "genrm_model_path", None) or getattr(args, "_genrm_instances_resolved", None)):
            raise ValueError("Partial GenRM requires a configured local GenRM")
        if (
            getattr(args, "group_rm", False)
            or getattr(args, "custom_generate_function_path", None)
            or getattr(args, "defer_reward_to_post_process", False)
            or getattr(reward, "supports_async_eval", False)
        ):
            raise ValueError("Partial GenRM requires inline per-sample rewards and the default generation function")
        return cls(args, state)

    def reset_step(self) -> None:
        self.engine_urls.clear()
        self.next_engine.clear()

    async def aclose(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            try:
                await client.aclose()
            except Exception as error:
                logger.warning(f"Failed to close partial GenRM HTTP client: {error}")

    @staticmethod
    def restore(sample: Sample) -> bool:
        checkpoint = sample.metadata.get(_CHECKPOINT_KEY)
        if checkpoint is None:
            return False
        sample.status = Sample.Status(checkpoint["policy_status"])
        return True

    @staticmethod
    def clear(sample: Sample) -> None:
        sample.metadata.pop(_CHECKPOINT_KEY, None)

    def bind(self, sample: Sample) -> Callable:
        cursor = 0
        busy = False
        checkpoint = sample.metadata.get(_CHECKPOINT_KEY, {})

        async def generate(
            messages: list[dict], sampling_params: dict | None = None, route_key: str | None = None
        ) -> str:
            nonlocal cursor, busy
            if busy:
                raise RuntimeError("Partial GenRM calls for one sample must be awaited sequentially")
            busy = True
            try:
                if not checkpoint:
                    checkpoint.update(policy_status=sample.status.value, calls=[])
                    sample.metadata[_CHECKPOINT_KEY] = checkpoint
                index = cursor
                cursor += 1
                return await self._generate(checkpoint["calls"], index, messages, sampling_params, route_key)
            finally:
                busy = False

        return generate

    def _resolve_instance(self, route_key: str | None) -> tuple[str, dict]:
        instances = getattr(self.args, "_genrm_instances_resolved", None) or {
            "__default__": {
                "model_path": self.args.genrm_model_path,
                "sampling_config": getattr(self.args, "genrm_sampling_config", None) or {},
            }
        }
        key = next(iter(instances)) if route_key is None and len(instances) == 1 else route_key or "__default__"
        if key not in instances:
            raise ValueError(f"Unknown GenRM route_key={key!r}; available={list(instances)}")
        return key, instances[key]

    def _get_tokenizer(self, key: str, spec: dict) -> Any:
        if key not in self.tokenizers:
            from relax.utils.data.processing_utils import load_tokenizer

            self.tokenizers[key] = load_tokenizer(spec["model_path"], trust_remote_code=True)
        return self.tokenizers[key]

    async def _get_engine_urls(self, key: str) -> list[str]:
        if key not in self.engine_urls:
            instances = getattr(self.args, "_genrm_instances_resolved", None) or {"__default__": {}}
            name = "relax_genrm_manager" if list(instances) == ["__default__"] else f"relax_genrm_manager_{key}"
            manager = ray.get_actor(name)
            hosts_ports = await manager.get_engine_hosts_ports.remote()
            urls = [f"http://{host}:{port}" for host, port in hosts_ports]
            if not urls:
                raise RuntimeError(f"No GenRM engines available for route_key={key!r}")
            self.engine_urls[key] = urls
        return self.engine_urls[key]

    def _check_abort(self) -> None:
        if self.state.abort_event.is_set():
            raise PartialGenRMAborted

    async def _generate(
        self,
        calls: list[dict],
        index: int,
        messages: list[dict],
        sampling_params: dict | None,
        route_key: str | None,
    ) -> str:
        key, spec = self._resolve_instance(route_key)
        config = spec.get("sampling_config") or {}
        params = {
            "temperature": config.get("temperature", 0.2),
            "top_p": config.get("top_p", 1.0),
            "top_k": config.get("top_k", -1),
            "max_new_tokens": config.get("max_response_len", 1024),
        }
        params.update(sampling_params or {})
        if params.get("stop") or params.get("stop_regex"):
            raise ValueError(
                "Partial GenRM does not support stop/stop_regex across rollout steps; use EOS or stop_token_ids"
            )
        fingerprint = hashlib.sha256(json.dumps([key, messages, params], sort_keys=True).encode()).hexdigest()
        if index < len(calls):
            call = calls[index]
            if call["fingerprint"] != fingerprint:
                raise ValueError(
                    "Resumed GenRM calls must keep the same order, route, messages and sampling parameters"
                )
        else:
            call = {"fingerprint": fingerprint, "output_ids": [], "response": "", "terminal": False}
            calls.append(call)
        if call["terminal"]:
            return call["response"].strip()
        self._check_abort()
        tokenizer = self._get_tokenizer(key, spec)
        if "prompt_ids" not in call:
            input_ids = await asyncio.to_thread(
                tokenizer.apply_chat_template,
                messages,
                tokenize=True,
                add_generation_prompt=True,
                **(config.get("chat_template_kwargs") or {}),
            )
            call["prompt_ids"] = list(input_ids if isinstance(input_ids, list) else input_ids["input_ids"])
        params["max_new_tokens"] = max(0, params["max_new_tokens"] - len(call["output_ids"]))
        if params["max_new_tokens"] == 0:
            call["terminal"] = True
            return call["response"].strip()
        urls = await self._get_engine_urls(key)
        engine_index = self.next_engine.get(key, 0) % len(urls)
        self.next_engine[key] = engine_index + 1
        url = urls[engine_index]
        payload = {
            "input_ids": call["prompt_ids"] + call["output_ids"],
            "sampling_params": params,
            "return_logprob": True,
        }
        self._check_abort()
        if self._client is None:
            concurrency = getattr(self.args, "reward_max_concurrency", None) or 64
            self._client = httpx.AsyncClient(limits=httpx.Limits(max_connections=concurrency), timeout=None)
        self.active[url] = self.active.get(url, 0) + 1
        try:
            try:
                # Judge traffic must not consume the policy HTTP pool or the
                # distributed poster actors' execution slots.
                output = await _post(self._client, f"{url}/generate", payload, max_retries=1)
            except httpx.HTTPStatusError as error:
                if is_expected_sglang_499(error):
                    raise PartialGenRMAborted from error
                raise
        finally:
            self.active[url] -= 1
            if not self.active[url]:
                del self.active[url]
        meta = output.get("meta_info", {})
        token_ids = output.get("output_ids")
        if token_ids is None:
            token_ids = [item[1] for item in meta.get("output_token_logprobs", [])]
        text = output.get("text", "")
        if text and not token_ids:
            raise RuntimeError("Partial GenRM requires output token IDs to checkpoint generated text")
        resumed = bool(call["output_ids"])
        call["output_ids"].extend(token_ids)
        finish_reason = meta.get("finish_reason") or {}
        reason = finish_reason.get("type")
        stop_ids = set()
        if not params.get("ignore_eos", False):
            stop_ids.update(params.get("stop_token_ids") or [])
            for attribute in ("eos_token_id", "eos_token_ids", "additional_stop_token_ids"):
                value = getattr(tokenizer, attribute, None)
                stop_ids.update([value] if isinstance(value, int) else value or [])
        if reason == "abort" and token_ids and token_ids[-1] in stop_ids:
            reason = "stop"
            finish_reason = {"type": "stop", "matched": token_ids[-1]}
        # A request-local decoded chunk can end in a partial UTF-8 character.
        # On resume, decode the complete prefix rather than concatenate text.
        if resumed:
            ids = call["output_ids"]
            matched = finish_reason.get("matched")
            if not params.get("no_stop_trim", False) and isinstance(matched, int) and ids and ids[-1] == matched:
                ids = ids[:-1]
            text = await asyncio.to_thread(
                tokenizer.decode,
                ids,
                skip_special_tokens=params.get("skip_special_tokens", True),
                spaces_between_special_tokens=params.get("spaces_between_special_tokens", True),
                clean_up_tokenization_spaces=False,
            )
            if isinstance(matched, str) and matched:
                before, stop, _ = text.partition(matched)
                text = before + stop if params.get("no_stop_trim", False) else before
        call["response"] = text
        if reason in {"stop", "length"}:
            call["terminal"] = True
            return call["response"].strip()
        if reason == "abort":
            raise PartialGenRMAborted
        raise RuntimeError(f"Unexpected GenRM finish_reason={reason!r}")

    async def abort_engines(self) -> None:
        # Repeat until drained: abort_all may reach a worker before an in-flight
        # HTTP generate has reached its scheduler. No new training requests can
        # enter after the rollout abort event is set.
        async with httpx.AsyncClient(timeout=10) as control_client:
            while self.active:
                urls = list(self.active)
                results = await asyncio.gather(
                    *(
                        _post(control_client, f"{url}/abort_request", {"abort_all": True}, max_retries=1)
                        for url in urls
                    ),
                    return_exceptions=True,
                )
                for url, result in zip(urls, results, strict=True):
                    if isinstance(result, BaseException):
                        logger.warning(f"Failed to abort GenRM worker at {url}: {result}")
                if self.active:
                    await asyncio.sleep(0.5)
