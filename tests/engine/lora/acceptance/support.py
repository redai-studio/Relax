# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import asyncio
import json
import time
from uuid import uuid4

from .assertions import compare, sample_scores, scores


def write_test_control(path, values):
    """Replace controls atomically while scheduler processes may read them."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(values))
    temporary.replace(path)


def configure_test(ctx, **values):
    write_test_control(
        ctx.test_control,
        {"directory": str(ctx.root), "capture_prepares": ctx.config.get("capture_prepares", False), **values},
    )


async def request(ctx, method, path, payload=None):
    response = await ctx.client.request(method, ctx.base + path, json=payload)
    response.raise_for_status()
    result = response.json()
    if result.get("error_code"):
        raise RuntimeError(result)
    return result


async def wait_state(ctx, operation, wanted):
    deadline = time.monotonic() + ctx.config["timeout_seconds"]
    while True:
        state = await request(ctx, "GET", "/lora/publications/" + operation)
        ctx.report["events"].append({"time": time.time(), **state})
        if state["state"] == wanted:
            return state
        if state["state"] in {"ABORTED", "RETIRED"} and wanted != state["state"]:
            raise AssertionError(state)
        if time.monotonic() >= deadline:
            raise TimeoutError(state)
        await asyncio.sleep(0.05)


async def publish(ctx, version, retry_of=None):
    submitted = await request(
        ctx, "POST", "/lora/publications", {"version_id": version, "request_id": uuid4().hex, "retry_of": retry_of}
    )
    return await wait_state(ctx, submitted["operation_id"], "PUBLISHED")


async def start_case(ctx, label, index, *, domain=None, wait_first=True):
    from relax.agentic.pipeline import GroupInput
    from relax.utils.types import Sample

    domain = domain or ctx.runtime
    max_tokens = (
        ctx.config.get("resume_tokens", 512) if domain is ctx.train_runtime else ctx.config.get("max_tokens", 32)
    )
    directory = ctx.root / f"{label}-{index}"
    directory.mkdir()
    sample = Sample(
        prompt=[{"role": "user", "content": ctx.prompts[index]}],
        group_index=index,
        index=0,
        metadata={
            "lora_verification": {
                "directory": str(directory),
                "timeout_seconds": ctx.config["timeout_seconds"],
                "max_tokens": max_tokens,
            }
        },
    )
    sample.sampling_params = {
        "temperature": 0.0,
        "max_new_tokens": max_tokens,
        "ignore_eos": domain is ctx.train_runtime,
    }
    stream = await domain.prepare_group(GroupInput(group_id=f"no7:{ctx.root.name}:{label}:{index}", samples=[sample]))
    if stream is None:
        raise AssertionError("Agent failed before the first request")
    ctx.streams.append((domain, stream))
    await domain.lease_group(stream, rollout_id=0)
    deadline = time.monotonic() + ctx.config["timeout_seconds"]
    while wait_first and not (directory / "turn-0.json").exists():
        if time.monotonic() > deadline:
            raise TimeoutError(f"agent did not complete first turn: {directory}")
        await asyncio.sleep(0.05)
    return stream, directory


async def native_bind(ctx):
    sid = "diagnostic-" + uuid4().hex
    bound = await ctx.manager.lora_control.remote("bind", {"owner_epoch": ctx.native_owner, "session_id": sid})
    if bound.get("error_code"):
        raise RuntimeError(bound)
    bound["session_id"] = sid
    ctx.native_sessions.append(bound)
    response = await ctx.client.post(
        bound["engine"]["endpoint"] + "/lora_attempt_status", json=attempt_payload(ctx, bound, uuid4().hex)
    )
    response.raise_for_status()
    status = response.json()
    bound["dp_rank"] = status["dp_rank"]
    bound["dp_size"] = status["dp_size"]
    bound["execution_workers"] = status["execution_workers"]
    return bound


async def native_bindings(ctx):
    """Cover every real engine/attention-DP route without forcing business
    routing."""
    state = await request(ctx, "GET", "/lora/versions")
    engines = set(state["serving_engines"])
    selected, extras, expected = {}, [], set()
    for _ in range(ctx.config.get("max_diagnostic_bindings", 1000)):
        bound = await native_bind(ctx)
        key = (bound["dp_rank"], bound["engine"]["engine_id"])
        expected.update((rank, key[1]) for rank in range(bound["dp_size"]))
        if key in selected:
            extras.append(bound)
        else:
            selected[key] = bound
        if {engine for _, engine in expected} == engines and expected <= selected.keys():
            break
    else:
        raise AssertionError("could not cover every engine/DP route within the binding budget")
    for bound in extras:
        await close_native(ctx, bound)
    return [selected[key] for key in sorted(selected)]


async def close_native(ctx, bound):
    result = await ctx.manager.lora_control.remote(
        "close", {"owner_epoch": ctx.native_owner, "session_id": bound["session_id"]}
    )
    if not result.get("accepted"):
        raise RuntimeError(result)


async def diagnostic(ctx, bound, ids, *, tokens=1, rid=None, extra_key=None):
    identity, engine = bound["binding"], bound["engine"]
    body = {
        "input_ids": ids,
        "rid": rid or uuid4().hex,
        "lora_path": bound["native_lora_id"],
        "return_logprob": True,
        "logprob_start_len": -1,
        "token_ids_logprob": ctx.diagnostic_ids,
        "sampling_params": {"temperature": 0, "max_new_tokens": tokens, "ignore_eos": tokens > 1},
        "lora_binding": {
            "cohort_id": identity["cohort_id"],
            "engine_boot_id": engine["boot_id"],
            "digest": identity["digest"],
            "native_lora_id": bound["native_lora_id"],
            "owner_epoch": ctx.native_owner,
            "session_id": bound["session_id"],
        },
    }
    if extra_key is not None:
        body["extra_key"] = extra_key
    result = await ctx.client.post(engine["endpoint"] + "/generate", json=body)
    result.raise_for_status()
    output = result.json()
    actual = output["meta_info"]["lora_adapter"]
    if actual["native_lora_id"] != bound["native_lora_id"] or actual["adapter_digest"] != identity["digest"]:
        raise AssertionError("native diagnostic used another adapter")
    if actual.get("engine_boot_id") != engine["boot_id"]:
        raise AssertionError("native diagnostic engine changed")
    actual.update(
        adapter_version_id=identity["version_id"],
        publication_id=identity["publication_id"],
        source_train_step=identity["source_train_step"],
    )
    deadline = time.monotonic() + ctx.config["timeout_seconds"]
    while True:
        status = await ctx.client.post(
            engine["endpoint"] + "/lora_attempt_status", json={**body["lora_binding"], "rid": body["rid"]}
        )
        status.raise_for_status()
        drained = status.json()
        if drained["state"] == "REQUEST_FINISHED":
            if drained["native_lora_id"] != bound["native_lora_id"] or drained["rid"] != body["rid"]:
                raise AssertionError("attempt completion identity mismatch")
            output["verification_completion"] = drained
            break
        if time.monotonic() > deadline:
            raise TimeoutError("diagnostic request did not leave the native scheduler")
        await asyncio.sleep(0.01)
    return output


async def reference_decode(ctx, version, ids, *, tokens=1):
    """Start an independent decode in a fresh cache namespace."""
    baseline = ctx.config["baselines"][version]
    if baseline.get("cache_enabled") is not True:
        raise AssertionError("decode reference requires cache enabled, with an isolated cold prefix")
    body = {
        "input_ids": ids,
        "lora_path": baseline["lora_path"],
        "extra_key": "no7-reference-" + uuid4().hex,
        "return_logprob": True,
        "token_ids_logprob": ctx.diagnostic_ids,
        "logprob_start_len": -1,
        "sampling_params": {"temperature": 0, "max_new_tokens": tokens, "ignore_eos": tokens > 1},
    }
    evidence = {"version": version, "request": body}
    ctx.report.setdefault("references", []).append(evidence)
    result = await ctx.client.post(
        baseline["url"].rstrip("/") + "/generate",
        json=body,
    )
    result.raise_for_status()
    output = result.json()
    evidence["response"] = output
    if output["meta_info"].get("cached_tokens") != 0:
        raise AssertionError("independent cold baseline already contains this diagnostic prefix")
    if len(output["output_ids"]) != tokens:
        raise AssertionError("independent decode ended before the requested token count")
    return output


async def cold_diagnostic(ctx, version, ids):
    return scores(await reference_decode(ctx, version, ids))


async def wait_file(ctx, path):
    deadline = time.monotonic() + ctx.config["timeout_seconds"]
    while not path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError("test hook did not execute: " + str(path))
        await asyncio.sleep(0.005)
    return json.loads(path.read_text())


def attempt_payload(ctx, bound, rid):
    identity = bound["binding"]
    return {
        "cohort_id": identity["cohort_id"],
        "engine_boot_id": bound["engine"]["boot_id"],
        "digest": identity["digest"],
        "native_lora_id": bound["native_lora_id"],
        "owner_epoch": ctx.native_owner,
        "session_id": bound["session_id"],
        "rid": rid,
    }


async def score_export(ctx, export, version):
    """Replay each Session on independently loaded weights with the same
    decode/continuation path.

    Bulk input scoring is not a decode reference. Each repetition starts in a
    fresh native cache namespace, shared only by that repetition's attempts; no
    managed-engine cache is flushed.
    """
    if export is None or not export.samples:
        raise AssertionError("real Session produced no exported samples")
    baseline = ctx.config["baselines"][version]
    if baseline.get("cache_enabled") is not True:
        raise AssertionError("Session replay requires a cache-enabled independent baseline")
    for sample in export.samples:
        sample_index = len(ctx.report["samples"])
        ctx.report["samples"].append(
            {
                "tokens": sample.tokens,
                "rollout_tokens": getattr(sample, "rollout_tokens", None),
                "response_length": sample.response_length,
                "loss_mask": sample.loss_mask,
                "logprobs": sample.rollout_log_probs,
                "metadata": sample.metadata,
            }
        )
        actual = sample_scores(sample)
        identity = sample.metadata.get("lora_adapter", {})
        attempts = sample.metadata.get("lora_attempts", [])
        if identity.get("version_id") != version or not attempts:
            raise AssertionError("sample lost adapter provenance")
        if any(
            item["adapter_version_id"] != version or item["adapter_digest"] != identity["digest"] for item in attempts
        ):
            raise AssertionError("mixed adapter identity inside one Session")
        responses = []
        evidence = {
            "sample_index": sample_index,
            "version": version,
            "method": "independent_cached_decode_v1",
            "actual": actual,
            "baseline_repeats": responses,
            "baseline_attempts": [],
            "status": "INCOMPLETE",
            "atol": ctx.atol,
            "rtol": ctx.rtol,
        }
        ctx.report["numerical"].append(evidence)
        try:
            rollout_tokens = getattr(sample, "rollout_tokens", None)
            if rollout_tokens != sample.tokens:
                raise AssertionError("Session replay requires identical model-visible and exported token sequences")
            positions, previous_end = [], 0
            for attempt in attempts:
                start, end = attempt["token_start"], attempt["token_end"]
                if (
                    type(start) is not int
                    or type(end) is not int
                    or not 0 < start <= end <= len(sample.tokens)
                    or start < previous_end
                ):
                    raise AssertionError("invalid or overlapping adapter attempt spans")
                positions.extend(range(start, end))
                previous_end = end
            if set(positions) != actual.keys():
                raise AssertionError("adapter attempt spans do not cover exactly the scored response tokens")
            for _ in range(2):
                # extra_key is a native radix namespace, not a changed prompt.
                # Never reuse a prior repeat's cached full continuation.
                cache_key = "no7-reference-" + uuid4().hex
                trace, expected = [], {}
                evidence["baseline_attempts"].append(trace)
                for attempt in attempts:
                    start, end = attempt["token_start"], attempt["token_end"]
                    if start == end:
                        trace.append({"token_start": start, "token_end": end, "status": "NO_OUTPUT"})
                        continue
                    body = {
                        "input_ids": sample.tokens[:start],
                        "lora_path": baseline["lora_path"],
                        "extra_key": cache_key,
                        "return_logprob": True,
                        "logprob_start_len": -1,
                        "sampling_params": {"max_new_tokens": end - start, "temperature": 0, "ignore_eos": True},
                    }
                    record = {"token_start": start, "token_end": end, "request": body}
                    trace.append(record)
                    result = await ctx.client.post(baseline["url"].rstrip("/") + "/generate", json=body)
                    result.raise_for_status()
                    output = result.json()
                    record["response"] = output
                    if not expected and output["meta_info"].get("cached_tokens") != 0:
                        raise AssertionError("independent Session replay did not start with a cold prefix")
                    if output["output_ids"] != sample.tokens[start:end]:
                        raise AssertionError(f"baseline generated different tokens for attempt {start}:{end}")
                    entries = output["meta_info"]["output_token_logprobs"]
                    if len(entries) != end - start or [row[1] for row in entries] != sample.tokens[start:end]:
                        raise AssertionError("baseline output logprobs do not align with exported tokens")
                    expected.update({start + i: float(row[0]) for i, row in enumerate(entries)})
                responses.append(expected)
            evidence["baseline"] = responses[0]
            evidence["max_error"] = max(abs(actual[key] - responses[0][key]) for key in actual)
            compare(responses[1], responses[0], atol=ctx.atol, rtol=ctx.rtol)
            compare(actual, responses[0], atol=ctx.atol, rtol=ctx.rtol)
        except Exception as error:
            evidence.update(status="FAIL", error=str(error))
            evidence["worst_positions"] = (
                [
                    {
                        "position": key,
                        "token_id": sample.tokens[key],
                        "actual": actual[key],
                        "baseline": responses[0][key],
                        "baseline_repeat": responses[1][key],
                        "absolute_error": abs(actual[key] - responses[0][key]),
                        "allowed_error": ctx.atol + ctx.rtol * abs(responses[0][key]),
                    }
                    for key in sorted(actual, key=lambda key: abs(actual[key] - responses[0][key]), reverse=True)[:8]
                ]
                if len(responses) == 2
                else []
            )
            raise AssertionError(f"Session sample {sample_index}, adapter {version}: {error}") from error
        evidence["status"] = "PASS"
