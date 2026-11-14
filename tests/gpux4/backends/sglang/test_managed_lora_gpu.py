# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Opt-in PR2 GPU acceptance using local Qwen3-0.6B or Qwen3.5-0.8B.

Select two GPUs with CUDA_VISIBLE_DEVICES and set RELAX_LORA_TEST_MODEL to a
local checkpoint of either model. Run this module with the PR2-patched SGLang
on PYTHONPATH (or installed). Each test launches two dedicated engines and
stops only its own processes. No training, model download or cluster job is
started. Model identity, adapters, prompts, ports and context length are
prepared here.
"""

import asyncio
import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np
import pytest
import torch

from relax.backends.sglang.lora_client import SGLangLoRAClient
from relax.engine.lora_versions.artifact import write_adapter_artifact
from relax.engine.lora_versions.identity import AdapterSnapshot, snapshot_adapter
from relax.engine.lora_versions.manager import PublicationError, SessionKey, VersionState
from relax.engine.lora_versions.publisher import LoRAPublisher
from relax.utils.http_utils import find_available_port
from relax.utils.megatron_peft_utils import build_hf_peft_config_dict
from relax.utils.opd.opd_main_worker import LogprobResponse, build_prefill_payload_base


pytestmark = pytest.mark.skipif(
    not os.environ.get("RELAX_LORA_TEST_MODEL"),
    reason="Set RELAX_LORA_TEST_MODEL and select two CUDA devices to explicitly enable GPU acceptance.",
)
CONTEXT_LENGTH = 8192
LONG_TOKENS = 4096
ATOL = RTOL = 1e-3


@dataclass
class LoRAFixture:
    model_path: Path
    old: AdapterSnapshot
    new: AdapterSnapshot
    probe_ids: list[int]
    score_prefix: list[int]
    score_suffix: list[int]


@pytest.fixture(scope="module")
def lora_fixture() -> LoRAFixture:
    from transformers import AutoTokenizer

    model_path = Path(os.environ["RELAX_LORA_TEST_MODEL"]).resolve()
    config_bytes = (model_path / "config.json").read_bytes()
    model_config = json.loads(config_bytes)
    config = model_config.get("text_config", model_config)
    profile = (model_config["model_type"], config["hidden_size"], config["num_hidden_layers"])
    if profile == ("qwen3", 1024, 28):
        block = "self_attn"
        modules = {
            "q_proj": config["num_attention_heads"] * config["head_dim"],
            "v_proj": config["num_key_value_heads"] * config["head_dim"],
        }
    elif profile == ("qwen3_5", 1024, 24):
        # Both linear- and full-attention layers have the same dense MLP.
        block = "mlp"
        modules = {"gate_proj": config["intermediate_size"], "up_proj": config["intermediate_size"]}
    else:
        raise ValueError("This acceptance fixture supports Qwen3-0.6B and Qwen3.5-0.8B only")
    weights = sorted(model_path.glob("*.safetensors"))
    assert weights, "A local safetensors base checkpoint is required"
    digest = hashlib.sha256(config_bytes)
    for path in weights:
        digest.update(path.name.encode())
        weight_digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                weight_digest.update(chunk)
        digest.update(weight_digest.digest())
    base_fingerprint = digest.hexdigest()
    rng = torch.Generator().manual_seed(42)
    tensors = {}
    for layer in range(config["num_hidden_layers"]):
        for module, output_size in modules.items():
            prefix = f"model.layers.{layer}.{block}.{module}"
            tensors[f"{prefix}.lora_A.weight"] = (
                torch.randn(8, config["hidden_size"], generator=rng, dtype=torch.bfloat16) * 0.04
            )
            tensors[f"{prefix}.lora_B.weight"] = (
                torch.randn(output_size, 8, generator=rng, dtype=torch.bfloat16) * 0.04
            )
    peft = build_hf_peft_config_dict(lora_rank=8, lora_alpha=16, target_modules=list(modules), lora_dropout=0.0)
    old = snapshot_adapter(1, tensors, peft, base_fingerprint=base_fingerprint)
    opposite = {name: -tensor if ".lora_B." in name else tensor for name, tensor in tensors.items()}
    new = snapshot_adapter(2, opposite, peft, base_fingerprint=base_fingerprint)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    return LoRAFixture(
        model_path,
        old,
        new,
        tokenizer.encode("Continue counting integers, separated by commas: 1, 2, 3,"),
        tokenizer.encode("Explain how rain forms from water vapor. " * 32),
        tokenizer.encode(
            "Warm air rises and cools. Water vapor condenses into droplets. The droplets combine and fall as rain.",
            add_special_tokens=False,
        ),
    )


@pytest.fixture
async def gpu_engines(lora_fixture, tmp_path, request):
    # Fail before launching anything if the selected environment lacks the patch.
    importlib.import_module("sglang.srt.lora.relax_control")
    from sglang.srt.utils import kill_process_tree

    selected = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    assert len(selected) == 2 and all(selected) and len(set(selected)) == 2, (
        "Explicitly select two GPUs with CUDA_VISIBLE_DEVICES"
    )
    assert torch.cuda.device_count() == 2, "The test requires exactly two visible CUDA devices"
    request.node.user_properties.extend(
        [
            ("base_fingerprint", json.loads(lora_fixture.old.manifest_json)["base_fingerprint"]),
            ("adapter_a_digest", lora_fixture.old.identity.digest),
            ("adapter_b_digest", lora_fixture.new.identity.digest),
            ("context_length", CONTEXT_LENGTH),
        ]
    )
    clients = {}
    async with AsyncExitStack() as stack:
        for gpu in range(2):
            port = find_available_port(30000 + gpu * 2000)
            url = f"http://127.0.0.1:{port}"
            log_path = tmp_path / f"engine-{gpu}.log"
            log = stack.enter_context(log_path.open("w"))
            env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
            command = [
                sys.executable,
                "-m",
                "sglang.launch_server",
                "--model-path",
                str(lora_fixture.model_path),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--base-gpu-id",
                str(gpu),
                "--tp-size",
                "1",
                "--dp-size",
                "1",
                "--pp-size",
                "1",
                "--tokenizer-worker-num",
                "1",
                "--dtype",
                "bfloat16",
                "--enable-lora",
                "--max-lora-rank",
                "8",
                "--max-loras-per-batch",
                "3",
                "--lora-target-modules",
                *lora_fixture.old.config["target_modules"],
                "--lora-backend",
                "triton",
                "--context-length",
                str(CONTEXT_LENGTH),
                "--mem-fraction-static",
                "0.5",
                "--max-running-requests",
                "8",
                "--cuda-graph-max-bs-decode",
                "8",
                "--attention-backend",
                "triton",
                "--incremental-streaming-output",
                "--random-seed",
                "11",
            ]
            process = stack.enter_context(subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT))
            stack.callback(kill_process_tree, process.pid, wait_timeout=30)
            http = await stack.enter_async_context(httpx.AsyncClient(base_url=url, timeout=120, trust_env=False))
            deadline = time.monotonic() + 300
            while True:
                assert process.poll() is None, f"Engine exited during startup; see {log_path}"
                try:
                    health = await http.get("/health", timeout=2)
                    if health.status_code == 200:
                        break
                except httpx.TransportError:
                    pass  # Connection refusal is expected while the child initializes.
                assert time.monotonic() < deadline, f"Engine startup timed out; see {log_path}"
                await asyncio.sleep(0.2)
            clients[str(gpu)] = SGLangLoRAClient(http)
            inventory = await clients[str(gpu)].inventory()
            assert not inventory["adapters"] and not inventory["fences"]
        yield clients


async def corrupt_lora_manifest(request):
    if request.url.path == "/relax/lora/load":
        body = json.loads(request.content)
        body["relax_manifest"] += " "
        encoded = json.dumps(body).encode()
        request.stream = httpx.ByteStream(encoded)
        request.headers["content-length"] = str(len(encoded))


@pytest.mark.asyncio
async def test_two_engine_publication_rejection_and_reclamation(lora_fixture, gpu_engines):
    old, new = lora_fixture.old, lora_fixture.new
    clients = gpu_engines
    publisher = await LoRAPublisher.connect(clients, lora_fixture.probe_ids)
    manager = publisher.manager
    await publisher.publish(old)
    assert manager.serving_engines == ("0", "1")
    manager.register_shard("test", "test-instance")
    session = SessionKey("test", "test-instance", 0)
    assert manager.bind(session) == old.identity
    for client in clients.values():
        response = await client.http.post(
            "/generate",
            json={
                "input_ids": [lora_fixture.probe_ids[0]] * (CONTEXT_LENGTH + 1),
                "lora_path": old.identity.lora_name,
                "sampling_params": {"max_new_tokens": 1},
            },
        )
        assert response.status_code == 400, response.text[:300]
        await wait_requests_zero(client, old.identity.lora_name)

    # Corrupt one real receiver; the other engine can already have loaded B.
    clients["1"].http.event_hooks["request"].append(corrupt_lora_manifest)
    try:
        with pytest.raises(PublicationError, match="REPLICA_INVALID"):
            await publisher.publish(new)
    finally:
        clients["1"].http.event_hooks["request"].remove(corrupt_lora_manifest)
    assert manager.default_version == 1
    for _ in range(100):
        assert not await publisher.reconcile()
        if manager.publication(2).state == VersionState.ABORTED:
            break
        await asyncio.sleep(0.05)
    assert manager.publication(2).state == VersionState.ABORTED
    assert manager.occupied_versions == 1
    for client in clients.values():
        assert sum(row["lora_id"] is not None for row in (await client.inventory())["adapters"]) == 1

    await publisher.publish(new)
    assert manager.default_version == 2 and manager.publication(1).state == VersionState.RETIRED
    await asyncio.gather(
        *(client.probe(old.identity.lora_name, lora_fixture.probe_ids) for client in clients.values())
    )
    with pytest.raises(PublicationError, match="CAPACITY_EXHAUSTED"):
        await publisher.publish(
            snapshot_adapter(
                3, new.copy_tensors(), new.config, base_fingerprint=json.loads(new.manifest_json)["base_fingerprint"]
            )
        )
    manager.release(session)
    for _ in range(100):
        assert not await publisher.reconcile()
        if manager.publication(1).state == VersionState.RECLAIMED:
            break
        await asyncio.sleep(0.05)
    assert manager.publication(1).state == VersionState.RECLAIMED
    for client in clients.values():
        response = await client.http.post(
            "/generate",
            json={
                "input_ids": lora_fixture.probe_ids,
                "lora_path": old.identity.lora_name,
                "sampling_params": {"max_new_tokens": 1},
            },
        )
        assert response.status_code == 409, response.text[:300]
        await client.probe(new.identity.lora_name, lora_fixture.probe_ids)
        assert sum(row["lora_id"] is not None for row in (await client.inventory())["adapters"]) == 1


async def wait_requests_zero(client, name):
    deadline = time.monotonic() + 10
    while True:
        row = next(row for row in (await client.inventory())["adapters"] if row["lora_name"] == name)
        if row["requests"] == 0:
            return
        assert time.monotonic() < deadline, f"Request references did not drain: {name}"
        await asyncio.sleep(0.05)


async def stream_old_request(client, name, input_ids, started, tokens):
    async with client.http.stream(
        "POST",
        "/generate",
        json={
            "input_ids": input_ids,
            "lora_path": name,
            "stream": True,
            "sampling_params": {"temperature": 0, "max_new_tokens": LONG_TOKENS, "ignore_eos": True},
        },
    ) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            assert "error" not in chunk, chunk
            assert (chunk.get("meta_info", {}).get("finish_reason") or {}).get("type") != "abort", chunk
            tokens.extend(chunk.get("output_ids", []))
            if len(tokens) >= 8:
                started.set()
    assert len(tokens) == LONG_TOKENS
    return tokens


@pytest.mark.asyncio
async def test_stop_does_not_block_new_load_and_stale_unload_is_rejected(lora_fixture, gpu_engines, record_property):
    old, new = lora_fixture.old, lora_fixture.new
    for engine, client in gpu_engines.items():
        instance = (await client.inventory())["instance_id"]
        loaded = await client.load(old, instance, 1)
        await client.probe(old.identity.lora_name, lora_fixture.probe_ids)
        started = asyncio.Event()
        tokens = []
        running = asyncio.create_task(
            stream_old_request(client, old.identity.lora_name, lora_fixture.probe_ids, started, tokens)
        )
        try:
            await asyncio.wait_for(started.wait(), timeout=30)
            stopped = await client.control("stop", instance, old.identity.lora_name, 1, loaded["lora_id"])
            assert stopped["requests"] > 0 and stopped["admission_closed"]
            before_load = time.monotonic()
            fresh = await client.load(new, instance, 2)
            assert fresh["phase"] == "loaded" and fresh["pinned"]
            await client.probe(new.identity.lora_name, lora_fixture.probe_ids)
            record_property(f"engine_{engine}_load_and_probe_seconds", time.monotonic() - before_load)
            record_property(f"engine_{engine}_old_tokens_at_publication", len(tokens))
            assert not running.done(), "Loading B waited for A to finish, or the overlap fixture was too short"
            with pytest.raises(httpx.HTTPStatusError) as error:
                await client.control("unload", instance, old.identity.lora_name, 1, loaded["lora_id"])
            assert error.value.response.status_code == 409
            await running
        finally:
            if not running.done():
                running.cancel()
            await asyncio.gather(running, return_exceptions=True)
        await wait_requests_zero(client, old.identity.lora_name)
        absent = await client.control("unload", instance, old.identity.lora_name, 1, loaded["lora_id"])
        assert absent["phase"] == "absent"
        replacement = await client.load(old, instance, 3)
        assert replacement["lora_id"] != loaded["lora_id"]
        with pytest.raises(httpx.HTTPStatusError) as error:
            await client.control("unload", instance, old.identity.lora_name, 3, loaded["lora_id"])
        assert error.value.response.status_code == 409
        assert error.value.response.json()["error"] == "LORA_ID_MISMATCH"
        await client.probe(old.identity.lora_name, lora_fixture.probe_ids)
        await client.probe(new.identity.lora_name, lora_fixture.probe_ids)


async def score_suffix(client, name, fixture):
    ids = fixture.score_prefix + fixture.score_suffix
    payload = build_prefill_payload_base(ids, len(fixture.score_prefix) - 1)
    payload.update(lora_path=name, return_prompt_token_ids=True)
    response = await client.http.post("/generate", json=payload)
    response.raise_for_status()
    parsed = LogprobResponse(response.json())
    parsed.validate_native_prefill(ids, len(fixture.score_suffix))
    return parsed.base_logprobs_1d()[1:], parsed.meta["cached_tokens"]


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["shm", "file"])
async def test_numerical_baselines_kv_isolation_and_slot_reuse(
    lora_fixture, gpu_engines, record_property, transport, tmp_path
):
    old, new = lora_fixture.old, lora_fixture.new
    sources = (
        [write_adapter_artifact(adapter, tmp_path / "artifacts") for adapter in (old, new)]
        if transport == "file"
        else [old, new]
    )
    record_property("transport", transport)
    record_property("atol", ATOL)
    record_property("rtol", RTOL)
    record_property("score_input_ids", json.dumps(lora_fixture.score_prefix + lora_fixture.score_suffix))
    record_property("score_response_length", len(lora_fixture.score_suffix))
    left, right = gpu_engines.values()
    left_id = (await left.inventory())["instance_id"]
    right_id = (await right.inventory())["instance_id"]
    old_receipt = await left.load(sources[0], left_id, 1)
    await right.load(sources[1], right_id, 1)
    baselines = []
    for label, client, adapter in [("a", left, old), ("b", right, new)]:
        # Compare equal cache histories: hybrid models may compute a cached
        # suffix differently from a full prefill even without any adapter.
        cold, cached = await score_suffix(client, adapter.identity.lora_name, lora_fixture)
        assert cached == 0
        warm, cached = await score_suffix(client, adapter.identity.lora_name, lora_fixture)
        assert cached > 0, "Same-version requests must actually exercise the prefix cache"
        repeated, cached = await score_suffix(client, adapter.identity.lora_name, lora_fixture)
        assert cached > 0
        np.testing.assert_allclose(repeated, warm, atol=ATOL, rtol=RTOL)
        # Only reset isolated reference engines, before versions coexist.
        response = await client.http.post("/flush_cache")
        response.raise_for_status()
        repeated, cached = await score_suffix(client, adapter.identity.lora_name, lora_fixture)
        assert cached == 0
        np.testing.assert_allclose(repeated, cold, atol=ATOL, rtol=RTOL)
        repeated, cached = await score_suffix(client, adapter.identity.lora_name, lora_fixture)
        assert cached > 0
        np.testing.assert_allclose(repeated, warm, atol=ATOL, rtol=RTOL)
        record_property(f"baseline_{label}_warm_logprobs", json.dumps(warm.tolist()))
        record_property(f"baseline_{label}_cold_warm_difference", float(np.max(np.abs(cold - warm))))
        baselines.append((cold, warm))
    (baseline_a, warm_a), (baseline_b, warm_b) = baselines
    record_property("baseline_a_logprobs", json.dumps(baseline_a.tolist()))
    record_property("baseline_b_logprobs", json.dumps(baseline_b.tolist()))
    separation = float(np.median(np.abs(baseline_a - baseline_b)))
    record_property("baseline_separation", separation)
    assert separation >= 100 * ATOL, "A/B fixture is insufficiently distinguishable; do not loosen numerical tolerance"
    await left.load(sources[1], left_id, 2)
    await right.load(sources[0], right_id, 2)
    errors = []
    for client, adapter, baseline, warm in [(left, new, baseline_b, warm_b), (right, old, baseline_a, warm_a)]:
        actual, cached = await score_suffix(client, adapter.identity.lora_name, lora_fixture)
        record_property(f"version_{adapter.identity.version_id}_cross_logprobs", json.dumps(actual.tolist()))
        record_property(f"version_{adapter.identity.version_id}_cross_cached_tokens", cached)
        assert cached == 0, "The first request for a new lora_id reused another adapter's KV"
        np.testing.assert_allclose(actual, baseline, atol=ATOL, rtol=RTOL)
        errors.append(float(np.max(np.abs(actual - baseline))))
        actual, cached = await score_suffix(client, adapter.identity.lora_name, lora_fixture)
        assert cached > 0
        np.testing.assert_allclose(actual, warm, atol=ATOL, rtol=RTOL)
        errors.append(float(np.max(np.abs(actual - warm))))
    # A new co-resident version must not alter an existing version's cached results.
    for client, adapter, warm in [(left, old, warm_a), (right, new, warm_b)]:
        actual, cached = await score_suffix(client, adapter.identity.lora_name, lora_fixture)
        assert cached > 0
        np.testing.assert_allclose(actual, warm, atol=ATOL, rtol=RTOL)
        errors.append(float(np.max(np.abs(actual - warm))))
    await left.control("stop", left_id, old.identity.lora_name, 1, old_receipt["lora_id"])
    await wait_requests_zero(left, old.identity.lora_name)
    await left.control("unload", left_id, old.identity.lora_name, 1, old_receipt["lora_id"])
    third = snapshot_adapter(
        3, new.copy_tensors(), new.config, base_fingerprint=json.loads(new.manifest_json)["base_fingerprint"]
    )
    third_source = write_adapter_artifact(third, tmp_path / "artifacts") if transport == "file" else third
    await left.load(third_source, left_id, 3)
    actual, cached = await score_suffix(left, third.identity.lora_name, lora_fixture)
    record_property("reused_slot_logprobs", json.dumps(actual.tolist()))
    record_property("reused_slot_cached_tokens", cached)
    assert cached == 0
    np.testing.assert_allclose(actual, baseline_b, atol=ATOL, rtol=RTOL)
    errors.append(float(np.max(np.abs(actual - baseline_b))))
    actual, cached = await score_suffix(left, third.identity.lora_name, lora_fixture)
    assert cached > 0
    np.testing.assert_allclose(actual, warm_b, atol=ATOL, rtol=RTOL)
    errors.append(float(np.max(np.abs(actual - warm_b))))
    record_property("maximum_cross_version_error", max(errors))
