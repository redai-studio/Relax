# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Author-code ReTool trajectories through Relax's native SGLang router."""

import functools
import math
import os
import re
import time
from typing import Any

from examples.retool.sandbox import SandboxClient


VERSION = "author_code_stateless_v1"
DIRECT_PREFIX = (
    "Solve the following problem step by step. No external tools are available; "
    "solve using your own reasoning. The last part of your response should be "
    "<answer>\\boxed{ANSWER}</answer>.\n\n*user question:*\n"
)
PREFIX = (
    "Solve the following problem step by step. You now have the ability to selectively write executable Python code "
    "to enhance your reasoning process. The Python code will be executed by an external sandbox, and the output "
    "(wrapped in `<interpreter>output_str</interpreter>`) can be returned to aid your reasoning and help you arrive "
    "at the final answer. The Python code should be complete scripts, including necessary imports.\n"
    "Each code snippet is wrapped with `<code>\n```python\ncode snippet\n```\n</code>`.\n"
    "Each call is a fresh stateless Python3 program; explicitly print results. No network, arbitrary files, "
    "subprocesses or threads;30seconds/1GiB per call. Do not invent interpreter outputs. "
    "End the assistant turn immediately after </code> and use the actual returned output to continue.\n"
    "The last part of your response should be <answer>\\boxed{ANSWER}</answer>.\n\n*user question:*\n"
)


def parse_code(text: str) -> str:
    if text.count("<code>") != 1 or text.count("</code>") != 1:
        raise ValueError("Exactly one complete code block per assistant turn")
    if "<interpreter>" in text or "</interpreter>" in text or not text.rstrip().endswith("</code>"):
        raise ValueError("Model cannot author an observation or continue past its tool boundary")
    block = text[text.index("<code>") + 6 : text.index("</code>")]
    match = re.fullmatch(r"\s*```python\s*\n(.*?)\n```\s*", block, re.DOTALL)
    if not match or not match.group(1).strip():
        raise ValueError("Require nonempty complete Python fenced source")
    return match.group(1)


def boundary(parameters: dict) -> dict:
    result = dict(parameters)
    stops = result.get("stop") or []
    if isinstance(stops, str):
        stops = [stops]
    result["stop"] = list(dict.fromkeys([*stops, "</code>"]))
    result["no_stop_trim"] = True
    return result


async def feedback_for_turn(text: str, client: Any) -> tuple[dict, str | None]:
    try:
        code = parse_code(text)
    except ValueError as exc:
        return {"status": "invalid_call", "stdout": "", "stderr": str(exc)}, None
    if len(code.encode("utf-8")) > 65536:
        return {
            "status": "invalid_call",
            "stdout": "",
            "stderr": "Python source exceeds sandbox 65536-byte limit",
        }, None
    # Only this NEW model-emitted code is executed, exactly once. Infrastructure
    # exceptions propagate rather than becoming a model failure reward.
    feedback = await client.execute(code)
    if feedback.get("status") not in {"ok", "code_error", "timeout"}:
        raise RuntimeError("Unexpected sandbox infrastructure response")
    return feedback, code


async def run_episode(
    args: Any,
    sample: Any,
    sampling_params: dict,
    *,
    tokenizer: Any,
    post: Any,
    client: Any,
    statuses: Any,
    evaluation: bool = False,
) -> Any:
    if sample.response_length or sample.response or sample.loss_mask or sample.rollout_log_probs:
        raise ValueError("ReTool requires fresh trajectories; never replay stored code")
    if getattr(sample.status, "value", None) != "pending":
        raise ValueError("ReTool requires a pending sample")
    if sample.metadata.get("protocol_id") != VERSION:
        raise ValueError("Author-tag rollout is not the historical JSON protocol")
    prompt_ids = tokenizer.encode(sample.prompt, add_special_tokens=False)
    if len(prompt_ids) > (
        (getattr(args, "eval_max_prompt_len", None) if evaluation else None) or args.rollout_max_prompt_len
    ):
        raise ValueError("Full author-protocol prompt exceeds the fixed prompt budget")
    sample.tokens = list(prompt_ids)
    sample.rollout_tokens = list(prompt_ids)
    sample.loss_mask = []
    sample.rollout_log_probs = []
    meta = sample.metadata
    tool_enabled = meta.get("tool_enabled", True)
    meta.update(retool_final_answer="", tool_feedback_counts={}, author_tool_events=[], tool_calls=0)
    total_budget = sampling_params["max_new_tokens"]
    if type(total_budget) is not int or total_budget < 1 or args.max_turns < 1:
        raise ValueError("Positive response budget and turn count required")
    context = getattr(args, "eval_max_context_len", None) if evaluation else None
    context = context or args.rollout_max_context_len
    try:
        for turn in range(args.max_turns):
            room = min(total_budget - sample.response_length, context - len(sample.tokens))
            if room <= 0:
                sample.status = statuses.TRUNCATED
                meta["retool_stop_reason"] = "token_budget"
                break
            output = await post(
                {
                    "input_ids": list(sample.rollout_tokens),
                    "sampling_params": boundary({**sampling_params, "max_new_tokens": room})
                    if tool_enabled
                    else {**sampling_params, "max_new_tokens": room},
                    "return_logprob": True,
                }
            )
            info = output["meta_info"]
            records = info.get("output_token_logprobs")
            finish = info["finish_reason"]["type"]
            if finish == "abort" and not records:
                sample.status = statuses.ABORTED
                break
            if not records or len(records) > room:
                raise RuntimeError("Need actual bounded engine token IDs and logprobs")
            ids = []
            probs = []
            for record in records:
                prob, token = record[:2]
                if type(token) is not int or not 0 <= token < 151665 or not math.isfinite(prob):
                    raise RuntimeError("Invalid generated token/logprob")
                ids.append(token)
                probs.append(float(prob))
            sample.tokens.extend(ids)
            sample.rollout_tokens.extend(ids)
            sample.loss_mask.extend([1] * len(ids))
            sample.rollout_log_probs.extend(probs)
            sample.response_length += len(ids)
            text = tokenizer.decode(ids, skip_special_tokens=False)
            meta.update(retool_assistant_turns=turn + 1, num_turns=turn + 1 + meta["tool_calls"])
            if finish == "abort":
                sample.status = statuses.ABORTED
                break
            if finish == "length":
                sample.status = statuses.TRUNCATED
                if not tool_enabled or "<code>" not in text:
                    meta["retool_final_answer"] = text
                meta["retool_stop_reason"] = "engine_length"
                break
            if finish not in {"stop", "eos"}:
                raise RuntimeError("Unknown engine finish reason")
            if not tool_enabled or ("<code>" not in text and "</code>" not in text):
                meta["retool_final_answer"] = text
                meta["retool_stop_reason"] = "final_answer"
                sample.status = statuses.COMPLETED
                break
            if turn + 1 == args.max_turns or sample.response_length >= total_budget or len(sample.tokens) >= context:
                sample.status = statuses.TRUNCATED
                meta["retool_stop_reason"] = "max_turns_or_budget"
                break
            tool_started = time.perf_counter()
            feedback, code = await feedback_for_turn(text, client)
            tool_seconds = time.perf_counter() - tool_started
            status = feedback["status"]
            counts = meta["tool_feedback_counts"]
            counts[status] = counts.get(status, 0) + 1
            meta["tool_calls"] += 1
            meta["author_tool_events"].append(
                dict(
                    submission=meta["tool_calls"],
                    exact_executed_code=code,
                    feedback=feedback,
                    new_online_execution=code is not None,
                    wall_seconds=tool_seconds,
                )
            )
            observation = feedback["stdout"] if status == "ok" else feedback["stderr"] or status
            suffix = tokenizer.encode(
                "\n<interpreter>\n" + observation + "\n</interpreter>\n", add_special_tokens=False
            )
            room = min(total_budget - sample.response_length, context - len(sample.tokens))
            if len(suffix) >= room:
                sample.status = statuses.TRUNCATED
                meta["retool_stop_reason"] = "observation_budget"
                break
            sample.tokens.extend(suffix)
            sample.rollout_tokens.extend(suffix)
            sample.loss_mask.extend([0] * len(suffix))
            sample.rollout_log_probs.extend([0.0] * len(suffix))
            sample.response_length += len(suffix)
        else:
            sample.status = statuses.TRUNCATED
            meta["retool_stop_reason"] = "max_turns"
        sample.response = tokenizer.decode(sample.tokens[len(prompt_ids) :], skip_special_tokens=False)
        if not (
            len(sample.tokens) - len(prompt_ids)
            == sample.response_length
            == len(sample.loss_mask)
            == len(sample.rollout_log_probs)
        ):
            raise RuntimeError("ReTool token/mask/logprob alignment failure")
        return sample
    finally:
        await client.close()


@functools.lru_cache(maxsize=1)
def logit_processor() -> str:
    from sglang.srt.sampling.custom_logit_processor import DisallowedTokensLogitsProcessor

    return DisallowedTokensLogitsProcessor.to_str()


async def generate(args: Any, sample: Any, sampling_params: dict, evaluation: bool = False) -> Any:
    from examples.retool.vocab import MODEL_VOCAB, TOKENIZER_LIMIT
    from relax.engine.rollout.sglang_rollout import GenerateState
    from relax.utils.types import Sample

    state = GenerateState(args)
    sample.metadata.setdefault("protocol_id", VERSION)
    if evaluation and not sample.metadata.get("tool_enabled", True):
        sample.prompt = state.tokenizer.apply_chat_template(
            [{"role": "user", "content": DIRECT_PREFIX + sample.metadata["question"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
    turn = 0

    async def post(payload: dict) -> dict:
        nonlocal turn
        params = payload["sampling_params"]
        params["custom_params"] = {"token_ids": list(range(TOKENIZER_LIMIT, MODEL_VOCAB))}
        payload["custom_logit_processor"] = logit_processor()
        params["sampling_seed"] = args.rollout_seed + sample.index * 1009 + turn * 1000003
        turn += 1
        result = await state.post_generate(
            f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate", payload
        )
        version = result["meta_info"].get("weight_version")
        if not evaluation:
            if (
                not str(version).isdigit()
                or int(version) + (args.start_rollout_id or 0) != sample.metadata["policy_outer"] + 1
            ):
                raise RuntimeError("Rollout policy version differs from the current training batch")
        sample.weight_versions.append(str(version))
        return result

    # Disabled eval never constructs an execution client or submits code.
    client = (
        SandboxClient(os.environ["SANDBOX_FUSION_URL"]) if sample.metadata.get("tool_enabled", True) else _NoTools()
    )
    return await run_episode(
        args,
        sample,
        sampling_params,
        tokenizer=state.tokenizer,
        post=post,
        client=client,
        statuses=Sample.Status,
        evaluation=evaluation,
    )


class _NoTools:
    async def close(self) -> None:
        pass


generate.manages_inference_permit = True
