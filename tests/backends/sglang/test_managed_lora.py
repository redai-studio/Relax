# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU regressions executing the actual SGLang patch, with GPU RPCs
replaced."""

import ast
import asyncio
import base64
import binascii
import copy
import dataclasses
import io
import json
import pickle
import sys
import typing
import uuid
from collections import Counter, OrderedDict
from enum import IntEnum
from functools import total_ordering
from multiprocessing.reduction import ForkingPickler
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import msgspec
import pytest
import torch

from relax.backends.sglang.lora_client import SGLangLoRAClient
from relax.engine.lora_versions.artifact import AdapterArtifact, write_adapter_artifact
from relax.engine.lora_versions.identity import snapshot_adapter, tensor_fingerprint
from relax.engine.lora_versions.manager import PublicationError, SessionKey, VersionState
from relax.engine.lora_versions.publisher import LoRAPublisher

from .source_utils import sglang_source_path


def definitions(relative, names, namespace):
    path = sglang_source_path(relative)
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    assert len(nodes) == len(names), f"missing definitions in {path}: {names}"
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


@pytest.fixture
def runtime(monkeypatch):
    # Load real registry/counter/locks without importing CUDA kernels or server bootstrap.
    sglang_source_path("srt/lora/relax_control.py")
    ns = {name: getattr(typing, name) for name in ("Dict", "List", "Optional", "Union", "Callable")}
    ns.update(
        asyncio=asyncio,
        OrderedDict=OrderedDict,
        msgspec=msgspec,
        fields=msgspec.structs.fields,
        uuid4=uuid.uuid4,
        uuid5=uuid.uuid5,
        NAMESPACE_URL=uuid.NAMESPACE_URL,
        dataclass=dataclasses.dataclass,
    )
    definitions("srt/utils/aio_rwlock.py", ["RWLock", "_ReaderLock", "_WriterLock"], ns)
    definitions("srt/utils/common.py", ["ConcurrentCounter"], ns)
    definitions("srt/lora/lora_registry.py", ["LoRARef", "LoRARegistry", "LoRARequestLease"], ns)
    registry = ModuleType("sglang.srt.lora.lora_registry")
    registry.__dict__.update(ns)
    monkeypatch.setitem(sys.modules, registry.__name__, registry)

    utils = ModuleType("sglang.srt.utils")
    ns.update(
        io=io,
        ForkingPickler=ForkingPickler,
        pybase64=base64,
        pickle=pickle,
        binascii=binascii,
        SerializedTensorPayload=typing.Union[str, bytes, bytearray, memoryview],
    )
    definitions(
        "srt/utils/common.py",
        [
            "MultiprocessingSerializer",
            "SafeUnpickler",
            "_looks_like_pickle_payload",
            "normalize_serialized_named_tensor_payload",
            "normalize_serialized_named_tensor_payloads",
        ],
        ns,
    )
    utils.MultiprocessingSerializer = ns["MultiprocessingSerializer"]
    utils.normalize_serialized_named_tensor_payloads = ns["normalize_serialized_named_tensor_payloads"]
    monkeypatch.setitem(sys.modules, utils.__name__, utils)

    control_mixin = ModuleType("sglang.srt.managers.tokenizer_control_mixin")
    control_mixin.__dict__.update(List=list, LoRAUpdateOutput=SimpleNamespace)
    definitions("srt/managers/tokenizer_control_mixin.py", ["_merge_lora_update_results"], control_mixin.__dict__)
    monkeypatch.setitem(sys.modules, control_mixin.__name__, control_mixin)
    io_struct = ModuleType("sglang.srt.managers.io_struct")
    io_struct.UnloadLoRAAdapterReqInput = SimpleNamespace
    io_struct.LoRAUpdateOutput = SimpleNamespace
    monkeypatch.setitem(sys.modules, io_struct.__name__, io_struct)

    control = ModuleType("sglang.srt.lora.relax_control")
    monkeypatch.setitem(sys.modules, control.__name__, control)
    path = sglang_source_path("srt/lora/relax_control.py")
    exec(compile(path.read_text(), str(path), "exec"), control.__dict__)
    return SimpleNamespace(**ns, control=control, utils=utils)


@pytest.fixture
def shared_memory(runtime):
    """Use the production serializer, or explicitly skip if the OS denies
    IPC."""
    from torch.multiprocessing import get_sharing_strategy, set_sharing_strategy

    previous = get_sharing_strategy()
    try:
        set_sharing_strategy("file_system")
        tensors = {"probe": torch.zeros(1)}
        runtime.utils.MultiprocessingSerializer.serialize(tensors, output_str=True)
    except RuntimeError as exc:
        if "Operation not permitted" not in str(exc) and "Permission denied" not in str(exc):
            raise
        pytest.skip("OS/sandbox denies production torch shared-memory IPC; no serialization fallback")
    finally:
        set_sharing_strategy(previous)


def snapshot(version=1):
    return snapshot_adapter(
        version,
        {"layer.lora_A.weight": torch.arange(12).reshape(3, 4).float()},
        {"r": 3, "target_modules": ["q_proj"]},
        base_fingerprint="base-content-sha",
    )


@pytest.mark.asyncio
async def test_file_publication_waits_for_every_replica_and_retains_artifacts(runtime, tmp_path):
    engines = [Engine(runtime), Engine(runtime)]
    publisher = await LoRAPublisher.connect({str(i): e.client for i, e in enumerate(engines)}, [1])
    old = write_adapter_artifact(snapshot(), tmp_path)
    new = write_adapter_artifact(snapshot(2), tmp_path)
    await publisher.publish(old)
    for engine in engines:
        engine.entered.clear()
    engines[1].resume.clear()
    task = asyncio.create_task(publisher.publish(new))
    await engines[1].entered.wait()
    assert publisher.manager.default_version == 1
    assert not task.done()
    engines[1].resume.set()
    assert (await task).state == VersionState.PUBLISHED
    assert publisher.manager.default_version == 2
    await publisher.reconcile()
    assert publisher.manager.publication(1).state == VersionState.RECLAIMED
    assert old.path.is_dir() and new.path.is_dir()
    assert all(not engine.client._payloads for engine in engines)


@pytest.mark.asyncio
async def test_file_missing_on_one_receiver_aborts_without_changing_default(runtime, tmp_path):
    engines = [Engine(runtime), Engine(runtime)]
    publisher = await LoRAPublisher.connect({str(i): e.client for i, e in enumerate(engines)}, [1])
    await publisher.publish(write_adapter_artifact(snapshot(), tmp_path))
    new = write_adapter_artifact(snapshot(2), tmp_path)
    mounted = False

    async def inaccessible_mount(request):
        if request.url.path == "/relax/lora/load" and not mounted:
            body = json.loads(request.content)
            body["relax_artifact_path"] = str(tmp_path / "not-mounted-on-this-node")
            request = httpx.Request(request.method, request.url, json=body)
        return await engines[1].http(request)

    await engines[1].client.http.aclose()
    engines[1].client.http = httpx.AsyncClient(
        base_url="http://engine", transport=httpx.MockTransport(inaccessible_mount)
    )
    with pytest.raises(PublicationError, match="REPLICA_INVALID"):
        await publisher.publish(new)
    assert publisher.manager.default_version == 1
    assert publisher.manager.publication(2).state == VersionState.ABORTED
    assert all(len(e.physical) == 1 for e in engines)
    assert new.path.is_dir()
    mounted = True
    assert (await publisher.publish(new)).state == VersionState.PUBLISHED


@pytest.mark.parametrize("damage", [None, "bytes", "identity", "missing", "two_sources"])
def test_file_worker_validates_received_bytes_before_model_mutation(runtime, tmp_path, damage):
    from safetensors.torch import save_file

    frozen = snapshot()
    artifact = write_adapter_artifact(frozen, tmp_path)
    req = SimpleNamespace(
        lora_name=frozen.identity.lora_name,
        relax_artifact_path=str(artifact.path),
        relax_attempt_id=1,
        relax_manifest=artifact.manifest_json,
        relax_digest=artifact.identity.digest,
        config_dict=artifact.config,
        serialized_named_tensors=[],
        load_format=None,
        added_tokens_config=None,
        expected_checksums=None,
        to_ref=lambda: object(),
    )
    if damage == "bytes":
        tensors = frozen.copy_tensors()
        tensors[next(iter(tensors))].add_(1)
        save_file(tensors, str(artifact.path / "adapter_model.safetensors"))
    elif damage == "identity":
        req.lora_name = "relax_lora@99." + artifact.identity.digest[:12]
    elif damage == "missing":
        (artifact.path / "manifest.json").unlink()
    elif damage == "two_sources":
        req.serialized_named_tensors = [b"unexpected"]
    path = sglang_source_path("srt/managers/tp_worker.py")
    cls = next(n for n in ast.parse(path.read_text()).body if getattr(n, "name", None) == "BaseTpWorker")
    method = next(n for n in cls.body if getattr(n, "name", None) == "load_lora_adapter_from_tensors")
    ns = {"LoadLoRAAdapterFromTensorsReqInput": object}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), ns)
    install = Mock(return_value=SimpleNamespace(success=True))
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(load_lora_adapter_from_tensors=install),
        _deserialize_own_rank=Mock(side_effect=AssertionError("file loading must not deserialize SHM")),
    )
    if damage is not None:
        with pytest.raises((ValueError, FileNotFoundError)):
            ns[method.name](worker, req)
        install.assert_not_called()
    else:
        result = ns[method.name](worker, req)
        assert result.fingerprints == frozen.fingerprints
        received = install.call_args.args[1]
        for name, tensor in frozen.copy_tensors().items():
            assert torch.equal(received[name], tensor)
        # The installed tensors retain owned bytes, independent of file changes.
        save_file(
            {name: tensor + 1 for name, tensor in received.items()}, str(artifact.path / "adapter_model.safetensors")
        )
        for name, tensor in frozen.copy_tensors().items():
            assert torch.equal(received[name], tensor)


def test_file_source_survives_http_validation_and_scheduler_ipc(runtime, tmp_path):
    from pydantic import TypeAdapter

    helper = {"__name__": "snapshot_schema_helpers"}
    path = sglang_source_path("srt/utils/msgspec_utils.py")
    exec(compile(path.read_text(), str(path), "exec"), helper)
    ns = {
        **vars(runtime),
        "Any": typing.Any,
        "Annotated": typing.Annotated,
        "Base64Bytes": helper["Base64Bytes"],
        "msgspec_struct_pydantic_core_schema": helper["msgspec_struct_pydantic_core_schema"],
    }
    definitions("srt/managers/io_struct.py", ["BaseReq", "LoadLoRAAdapterFromTensorsReqInput"], ns)
    request_type = ns["LoadLoRAAdapterFromTensorsReqInput"]
    artifact = write_adapter_artifact(snapshot(), tmp_path)
    payload = {
        "lora_name": artifact.identity.lora_name,
        "config_dict": artifact.config,
        "serialized_named_tensors": [],
        "relax_artifact_path": str(artifact.path),
        "relax_instance_id": "instance",
        "relax_attempt_id": 1,
        "relax_manifest": artifact.manifest_json,
        "relax_digest": artifact.identity.digest,
        "pinned": True,
    }
    parsed = TypeAdapter(request_type).validate_json(json.dumps(payload))
    # The control layer assigns this ID before sending the request to workers.
    parsed.lora_id = runtime.LoRARef(lora_name=parsed.lora_name, lora_path="__tensor__", pinned=True).lora_id
    decoded = msgspec.msgpack.decode(msgspec.msgpack.encode(parsed), type=request_type)
    assert decoded.relax_artifact_path == str(artifact.path)
    assert decoded.serialized_named_tensors == []
    assert decoded.relax_manifest == artifact.manifest_json
    assert decoded.to_ref().pinned


@pytest.fixture(params=["shm", "file"])
def publication_source(request, tmp_path, runtime, monkeypatch):
    if request.param == "shm":
        request.getfixturevalue("shared_memory")
        return snapshot
    monkeypatch.setattr(
        runtime.utils.MultiprocessingSerializer,
        "serialize",
        Mock(side_effect=AssertionError("file publication must not use shared memory")),
    )
    return lambda version=1: write_adapter_artifact(snapshot(version), tmp_path)


class Engine:
    """Only the transport/GPU work is fake; control and reference tracking are
    real."""

    def __init__(self, runtime):
        self.runtime = runtime
        self.server_args = SimpleNamespace(
            enable_lora=True,
            tp_size=1,
            dp_size=1,
            pp_size=1,
            disaggregation_mode="null",
            speculative_algorithm=None,
            hicache_storage_backend=None,
            tokenizer_worker_num=1,
            max_loras_per_batch=3,
        )
        self.lora_registry = runtime.LoRARegistry()
        self.lora_update_lock = asyncio.Lock()
        self.relax_lora = runtime.control.RelaxLoRAControl(self)
        self.physical = {}
        self.entered = asyncio.Event()
        self.resume = asyncio.Event()
        self.resume.set()
        self.fail_load = False
        self.lose_load_reply = False
        self.fail_probe = False
        self.lose_unload_reply = False
        self.unreachable = False
        self.unloads = 0
        self.loads = 0
        self.client = SGLangLoRAClient(
            httpx.AsyncClient(base_url="http://engine", transport=httpx.MockTransport(self.http))
        )

    def auto_create_handle_loop(self):
        pass

    async def update_lora_adapter_communicator(self, req):
        if hasattr(req, "serialized_named_tensors"):
            self.loads += 1
            self.entered.set()
            await self.resume.wait()
            if getattr(req, "relax_artifact_path", None):
                tensors = self.runtime.control.read_artifact_tensors(req)
            else:
                tensors = self.runtime.utils.MultiprocessingSerializer.deserialize(req.serialized_named_tensors[0])
            fingerprints = self.runtime.control.verify_snapshot(req, tensors)
            self.physical[req.lora_id] = tensors
            return [
                SimpleNamespace(
                    success=not self.fail_load,
                    error_message="injected" if self.fail_load else None,
                    fingerprints=fingerprints,
                    loaded_adapters={},
                )
            ]
        self.unloads += 1
        self.physical.pop(req.lora_id, None)
        if self.lose_unload_reply:
            self.lose_unload_reply = False
            raise TimeoutError("lost worker unload ACK")
        return [SimpleNamespace(success=True, error_message=None, fingerprints=None, loaded_adapters={})]

    async def http(self, request):
        if self.unreachable:
            raise httpx.ConnectError("engine unreachable")
        try:
            if request.url.path == "/relax/lora/inventory":
                result = self.relax_lora.inventory()
            elif request.url.path == "/relax/lora/load":
                payload = json.loads(request.content)
                payload.setdefault("load_format", None)
                payload.setdefault("added_tokens_config", None)
                result = await self.relax_lora.load(SimpleNamespace(**payload))
                if self.lose_load_reply:
                    raise httpx.ReadTimeout("lost load reply")
            elif request.url.path == "/relax/lora/control":
                result = await self.relax_lora.control(json.loads(request.content))
            else:
                body = json.loads(request.content)
                assert body["sampling_params"] == {"temperature": 0, "max_new_tokens": 1, "ignore_eos": True}
                lora_id = await self.lora_registry.acquire(body["lora_path"])
                assert lora_id in self.physical
                await self.lora_registry.release(lora_id)
                result = {"meta_info": {"completion_tokens": 0 if self.fail_probe else 1}}
            return httpx.Response(200, json=result)
        except ValueError as exc:
            return httpx.Response(409, json={"error": str(exc)})


@pytest.mark.asyncio
async def test_publish_all_replicas_then_drain_session_and_requests(runtime, publication_source):
    snapshot = publication_source
    engines = [Engine(runtime), Engine(runtime)]
    publisher = await LoRAPublisher.connect({str(i): e.client for i, e in enumerate(engines)}, [1])
    manager = publisher.manager
    await publisher.publish(snapshot(1))
    manager.register_shard("s", "shard-instance")
    session = SessionKey("s", "shard-instance", 0)
    old = manager.bind(session)
    request_id = await engines[0].lora_registry.acquire(old.lora_name)
    await publisher.publish(snapshot(2))
    assert manager.default_version == 2
    assert manager.publication(1).state == VersionState.RETIRED
    assert await publisher.reconcile() == {}
    assert all(e.unloads == 0 for e in engines)
    manager.release(session)
    await publisher.reconcile()
    assert manager.publication(1).state == VersionState.DRAINING
    assert engines[0].unloads == 0 and engines[1].unloads == 1
    await engines[0].lora_registry.release(request_id)
    await publisher.reconcile()
    assert manager.publication(1).state == VersionState.RECLAIMED
    assert manager.occupied_versions == 1
    assert all(e.loads == 2 and e.unloads == 1 for e in engines)
    assert len(engines[0].client._payloads) == (0 if isinstance(publisher.snapshots[2], AdapterArtifact) else 1)
    await publisher.publish(snapshot(2))
    assert all(e.loads == 2 for e in engines)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["fail_load", "fail_probe", "lose_load_reply"])
async def test_failed_publication_keeps_default_and_cleans_every_target(runtime, publication_source, failure):
    snapshot = publication_source
    engines = [Engine(runtime), Engine(runtime)]
    publisher = await LoRAPublisher.connect({str(i): e.client for i, e in enumerate(engines)}, [1])
    await publisher.publish(snapshot(1))
    setattr(engines[1], failure, True)
    with pytest.raises((PublicationError, ValueError, httpx.ReadTimeout)):
        await publisher.publish(snapshot(2))
    assert publisher.manager.default_version == 1
    assert publisher.manager.publication(2).state == VersionState.ABORTED
    assert all(e.unloads == 1 and len(e.physical) == 1 for e in engines)


@pytest.mark.asyncio
async def test_cancel_during_load_fences_late_completion_and_retry(runtime, publication_source):
    snapshot = publication_source
    engine = Engine(runtime)
    publisher = await LoRAPublisher.connect({"e": engine.client}, [1])
    engine.resume.clear()
    task = asyncio.create_task(publisher.publish(snapshot()))
    await engine.entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert publisher.manager.publication(1).state == VersionState.ABORTING
    assert publisher.manager.occupied_versions == 1
    assert len(engine.client._payloads) == (0 if isinstance(publisher.snapshots[1], AdapterArtifact) else 1)
    engine.resume.set()
    record = next(iter(engine.relax_lora.records.values()))
    await record.task
    assert engine.lora_registry.num_registered_loras == 0
    await publisher.reconcile()
    assert publisher.manager.publication(1).state == VersionState.ABORTED
    assert engine.physical == {} and engine.client._payloads == {}
    result = await publisher.publish(snapshot())
    assert result.attempt_id == 2 and result.state == VersionState.PUBLISHED
    with pytest.raises(httpx.HTTPStatusError):
        await engine.client.load(snapshot(), engine.relax_lora.instance_id, 1)
    assert engine.loads == 2


@pytest.mark.asyncio
async def test_unreachable_and_lost_unload_ack_hold_capacity_until_reconciled(runtime, publication_source):
    snapshot = publication_source
    engine = Engine(runtime)
    publisher = await LoRAPublisher.connect({"e": engine.client}, [1])
    await publisher.publish(snapshot(1))
    await publisher.publish(snapshot(2))
    engine.unreachable = True
    assert await publisher.reconcile()
    with pytest.raises(PublicationError, match="CAPACITY_EXHAUSTED"):
        await publisher.publish(snapshot(3))
    engine.unreachable = False
    engine.lose_unload_reply = True
    assert await publisher.reconcile()
    assert publisher.manager.occupied_versions == 2
    await publisher.reconcile()
    assert publisher.manager.occupied_versions == 1
    assert engine.unloads == 2


@pytest.mark.asyncio
async def test_instance_replacement_recovers_before_serving_and_rejects_stale_control(runtime, publication_source):
    snapshot = publication_source
    old_engine, new_engine = Engine(runtime), Engine(runtime)
    publisher = await LoRAPublisher.connect({"e": old_engine.client}, [1])
    await publisher.publish(snapshot())
    old_id, new_id = old_engine.relax_lora.instance_id, new_engine.relax_lora.instance_id
    new_engine.resume.clear()
    task = asyncio.create_task(publisher.replace_engine("e", old_id, new_id, client=new_engine.client))
    await new_engine.entered.wait()
    assert publisher.manager.serving_engines == ()
    with pytest.raises(httpx.HTTPStatusError):
        await new_engine.client.control("fence", old_id, snapshot().identity.lora_name, 1)
    new_engine.resume.set()
    await task
    assert publisher.manager.serving_engines == ("e",)


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_load", [False, True])
async def test_replacement_waits_for_publication_and_its_failure_cleanup(runtime, tmp_path, fail_load):
    old, new = Engine(runtime), Engine(runtime)
    publisher = await LoRAPublisher.connect({"e": old.client}, [1])
    await publisher.publish(write_adapter_artifact(snapshot(), tmp_path))
    old.entered.clear()
    old.resume.clear()
    old.fail_load = fail_load
    publishing = asyncio.create_task(publisher.publish(write_adapter_artifact(snapshot(2), tmp_path)))
    await old.entered.wait()
    replacing = asyncio.create_task(
        publisher.replace_engine("e", old.relax_lora.instance_id, new.relax_lora.instance_id, client=new.client)
    )
    await asyncio.sleep(0)
    assert not replacing.done()
    assert publisher.engines["e"] is old.client
    assert publisher.manager.default_version == 1
    old.resume.set()
    if fail_load:
        with pytest.raises(PublicationError, match="REPLICA_INVALID"):
            await asyncio.wait_for(publishing, 5)
    else:
        await asyncio.wait_for(publishing, 5)
    await asyncio.wait_for(replacing, 5)
    assert publisher.manager.default_version == (1 if fail_load else 2)
    assert publisher.engines["e"] is new.client
    assert publisher.manager.serving_engines == ("e",)
    assert new.loads == 1
    await publisher.reconcile()
    assert publisher.manager.occupied_versions == 1


@pytest.mark.asyncio
async def test_publication_waits_for_replacement_recovery(runtime, tmp_path):
    old, new = Engine(runtime), Engine(runtime)
    publisher = await LoRAPublisher.connect({"e": old.client}, [1])
    await publisher.publish(write_adapter_artifact(snapshot(), tmp_path))
    new.resume.clear()
    replacing = asyncio.create_task(
        publisher.replace_engine("e", old.relax_lora.instance_id, new.relax_lora.instance_id, client=new.client)
    )
    await new.entered.wait()
    publishing = asyncio.create_task(publisher.publish(write_adapter_artifact(snapshot(2), tmp_path)))
    await asyncio.sleep(0)
    assert not publishing.done()
    assert publisher.manager.default_version == 1
    assert publisher.manager.occupied_versions == 1
    assert publisher.manager.serving_engines == ()
    new.resume.set()
    await asyncio.wait_for(asyncio.gather(replacing, publishing), 5)
    assert publisher.manager.default_version == 2
    assert publisher.manager.serving_engines == ("e",)
    assert old.loads == 1 and new.loads == 2


@pytest.mark.asyncio
async def test_replacement_waits_for_inflight_cleanup(runtime, tmp_path, monkeypatch):
    old, new = Engine(runtime), Engine(runtime)
    publisher = await LoRAPublisher.connect({"e": old.client}, [1])
    await publisher.publish(write_adapter_artifact(snapshot(), tmp_path))
    await publisher.publish(write_adapter_artifact(snapshot(2), tmp_path))
    entered, resume = asyncio.Event(), asyncio.Event()
    control = old.client.control

    async def blocked_control(*args, **kwargs):
        entered.set()
        await resume.wait()
        return await control(*args, **kwargs)

    monkeypatch.setattr(old.client, "control", blocked_control)
    cleaning = asyncio.create_task(publisher.reconcile())
    await entered.wait()
    replacing = asyncio.create_task(
        publisher.replace_engine("e", old.relax_lora.instance_id, new.relax_lora.instance_id, client=new.client)
    )
    await asyncio.sleep(0)
    assert not replacing.done()
    assert publisher.engines["e"] is old.client
    assert publisher.manager.engine_instance("e") == old.relax_lora.instance_id
    resume.set()
    assert await asyncio.wait_for(cleaning, 5) == {}
    await asyncio.wait_for(replacing, 5)
    assert publisher.manager.publication(1).state is VersionState.RECLAIMED
    assert publisher.manager.serving_engines == ("e",)
    assert new.loads == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, "fail_load", "fail_probe"])
async def test_session_release_during_recovery_preserves_cleanup_and_real_errors(runtime, tmp_path, failure):
    old, new = Engine(runtime), Engine(runtime)
    publisher = await LoRAPublisher.connect({"e": old.client}, [1])
    manager = publisher.manager
    await publisher.publish(write_adapter_artifact(snapshot(), tmp_path))
    manager.register_shard("s", "shard")
    session = SessionKey("s", "shard", 0)
    manager.bind(session)
    await publisher.publish(write_adapter_artifact(snapshot(2), tmp_path))
    new.resume.clear()
    if failure:
        setattr(new, failure, True)
    replacing = asyncio.create_task(
        publisher.replace_engine("e", old.relax_lora.instance_id, new.relax_lora.instance_id, client=new.client)
    )
    await new.entered.wait()
    manager.release(session)
    cleaning = asyncio.create_task(publisher.reconcile())
    await asyncio.sleep(0)
    assert manager.publication(1).state is VersionState.DRAINING
    assert not cleaning.done()
    assert new.unloads == 0
    new.resume.set()
    if failure:
        with pytest.raises(ValueError, match="REPLICA_INVALID|probe did not produce"):
            await asyncio.wait_for(replacing, 5)
        assert manager.serving_engines == ()
        assert new.loads == 1
    else:
        await asyncio.wait_for(replacing, 5)
        assert manager.serving_engines == ("e",)
        assert new.loads == 2
    assert await asyncio.wait_for(cleaning, 5) == {}
    assert manager.default_version == 2
    assert manager.publication(1).state is VersionState.RECLAIMED
    assert new.unloads == 1


@pytest.mark.asyncio
async def test_cancelled_replacement_retries_without_losing_payloads_or_locks(runtime, tmp_path):
    old, new = Engine(runtime), Engine(runtime)
    publisher = await LoRAPublisher.connect({"e": old.client}, [1])
    await publisher.publish(write_adapter_artifact(snapshot(), tmp_path))
    old_id, new_id = old.relax_lora.instance_id, new.relax_lora.instance_id
    # Retention belongs to the old client even when the endpoint changes.
    old.client._payloads[(old_id, "old", 1)] = ({}, "owned")
    new.resume.clear()
    replacing = asyncio.create_task(publisher.replace_engine("e", old_id, new_id, client=new.client))
    await new.entered.wait()
    assert old.client._payloads == {}
    replacing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await replacing
    assert publisher.manager.serving_engines == ()
    assert await asyncio.wait_for(publisher.reconcile(), 5) == {}
    new.client._payloads[(new_id, "retained", 1)] = ({}, "owned")
    new.resume.set()
    await asyncio.wait_for(publisher.replace_engine("e", old_id, new_id, client=new.client), 5)
    assert publisher.manager.serving_engines == ("e",)
    assert new.loads == 1
    assert (new_id, "retained", 1) in new.client._payloads
    with pytest.raises(PublicationError, match="ENGINE_INSTANCE_MISMATCH"):
        await publisher.replace_engine("e", old_id, new_id, client=old.client)
    assert publisher.engines["e"] is new.client


@pytest.mark.asyncio
async def test_stale_replacement_cannot_change_client_without_snapshots(runtime):
    old, new, stale = Engine(runtime), Engine(runtime), Engine(runtime)
    publisher = await LoRAPublisher.connect({"e": old.client}, [1])
    old_id, new_id = old.relax_lora.instance_id, new.relax_lora.instance_id
    await publisher.replace_engine("e", old_id, new_id, client=new.client)
    with pytest.raises(PublicationError, match="ENGINE_INSTANCE_MISMATCH"):
        await publisher.replace_engine("e", old_id, stale.relax_lora.instance_id, client=stale.client)
    assert publisher.manager.engine_instance("e") == new_id
    assert publisher.engines["e"] is new.client


@pytest.mark.asyncio
async def test_same_endpoint_replacement_retries_probe_without_reloading(runtime, tmp_path):
    old, new = Engine(runtime), Engine(runtime)
    publisher = await LoRAPublisher.connect({"e": old.client}, [1])
    await publisher.publish(write_adapter_artifact(snapshot(), tmp_path))
    await old.client.http.aclose()
    old.client.http = httpx.AsyncClient(base_url="http://engine", transport=httpx.MockTransport(new.http))
    old_id, new_id = old.relax_lora.instance_id, new.relax_lora.instance_id
    new.fail_probe = True
    with pytest.raises(ValueError, match="probe did not produce"):
        await publisher.replace_engine("e", old_id, new_id)
    assert publisher.manager.serving_engines == ()
    new.fail_probe = False
    await publisher.replace_engine("e", old_id, new_id)
    assert publisher.engines["e"] is old.client
    assert publisher.manager.serving_engines == ("e",)
    assert new.loads == 1


@pytest.mark.parametrize("corruption", ["bytes", "dtype", "shape", "config"])
def test_receiver_validates_actual_bytes_metadata_and_fingerprint_parity(runtime, corruption):
    frozen = snapshot()
    tensors = frozen.copy_tensors()
    req = SimpleNamespace(
        relax_manifest=frozen.manifest_json, relax_digest=frozen.identity.digest, config_dict=frozen.config
    )
    assert runtime.control.verify_snapshot(req, tensors) == frozen.fingerprints
    value = next(iter(tensors.values()))
    assert int(runtime.control.tensor_fingerprint(value)) == int(tensor_fingerprint(value))
    if corruption == "bytes":
        value[0, 0] += 1
    elif corruption == "dtype":
        tensors[next(iter(tensors))] = value.to(torch.int32)
    elif corruption == "shape":
        tensors[next(iter(tensors))] = value.reshape(2, 6)
    else:
        req.config_dict = {"r": 99}
    with pytest.raises(ValueError, match="mismatch"):
        runtime.control.verify_snapshot(req, tensors)


@pytest.mark.asyncio
async def test_registry_acquire_linearizes_before_stop_and_cancellation_rolls_back(runtime):
    registry = runtime.LoRARegistry()
    ref = runtime.LoRARef(lora_name="a", lora_path="__tensor__", pinned=True)
    await registry.register(ref)
    counter = registry._counters[ref.lora_id]
    await counter._condition.acquire()
    acquire = asyncio.create_task(registry.acquire(["a", "a"]))
    await asyncio.sleep(0)
    stop = asyncio.create_task(registry.unregister("a"))
    await asyncio.sleep(0)
    assert not stop.done()
    counter._condition.release()
    assert await acquire == [ref.lora_id, ref.lora_id]
    await stop
    assert registry.request_count(ref.lora_id) == 2
    await registry.release([ref.lora_id, ref.lora_id])
    assert registry.request_count(ref.lora_id) == 0
    await registry.register(ref)
    ref2 = runtime.LoRARef(lora_name="b", lora_path="__tensor__")
    await registry.register(ref2)
    await registry._counters[ref2.lora_id]._condition.acquire()
    acquire = asyncio.create_task(registry.acquire(["a", "b"]))
    await asyncio.sleep(0)
    acquire.cancel()
    with pytest.raises(asyncio.CancelledError):
        await acquire
    assert registry.request_count(ref.lora_id) == 0
    registry._counters[ref2.lora_id]._condition.release()


def tokenizer_methods(runtime):
    path = sglang_source_path("srt/managers/tokenizer_manager.py")
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TokenizerManager")
    async_send = any(isinstance(n, ast.AsyncFunctionDef) and n.name == "_send_one_request" for n in cls.body)
    names = {
        "_remove_req_state",
        "_release_lora_lease",
        "_handle_abort_req",
        "_handle_batch_output",
        "_init_child_req_state",
        "_resolve_lora_path",
        "_logical_rids",
        "generate_request",
        "_validate_and_resolve_lora",
        "_discard_pending_req_states",
        "_release_req_states_on_failure",
        "_wait_for_request_unpaused",
        "_raise_if_request_aborted",
        "_handle_batch_request",
        "_collect_batch_responses",
        "_init_req_state",
        "_register_child_rid",
        "_raise_if_logical_rid_aborted",
        "_raise_if_logical_request_aborted",
    }
    cls.body = [n for n in cls.body if getattr(n, "name", None) in names]
    cls.bases = []
    ns = {
        **vars(runtime),
        "copy": copy,
        "ReqState": object,
        "Optional": typing.Optional,
        "Union": typing.Union,
        "List": list,
        "Dict": dict,
        "GenerateReqInput": object,
        "EmbeddingReqInput": object,
        "fastapi": SimpleNamespace(Request=object, HTTPException=RuntimeError),
        "AbortReq": object,
        "BatchStrOutput": type("BatchStrOutput", (), {}),
        "BatchTokenIDOutput": type("BatchTokenIDOutput", (), {}),
        "BatchEmbeddingOutput": type("BatchEmbeddingOutput", (), {}),
        "unwrap_from_pickle": lambda x: x,
        "is_health_check_generate_req": lambda _: False,
        "logger": Mock(),
        "HEALTH_CHECK_RID_PREFIX": "health-",
    }
    # Future annotations keep unrelated response type annotations out of this CPU fixture.
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), ns)
    manager = ns["TokenizerManager"]()
    manager._test_async_send = async_send
    manager.rid_to_state = {}
    manager._lora_draining_requests = {}
    manager._lora_release_tasks = set()
    manager.child_rid_to_logical_rid = {}
    manager.logical_rid_to_child_rids = {}
    manager.server_args = SimpleNamespace(batch_notify_size=8)
    # SGLang 0.5.20 reads request configuration through runtime-context accessors.
    ns.update(
        get_lora=lambda: manager.server_args,
        get_serving=lambda: manager.server_args,
        get_disagg=lambda: manager.server_args,
    )
    manager.lora_registry = runtime.LoRARegistry()
    return manager


@pytest.mark.asyncio
@pytest.mark.parametrize("dispatched,terminal", [(False, "none"), (True, "abort"), (True, "batch")])
async def test_request_lease_waits_for_scheduler_terminal_and_releases_once(runtime, dispatched, terminal):
    manager = tokenizer_methods(runtime)
    ref = runtime.LoRARef(lora_name="a", lora_path="__tensor__")
    await manager.lora_registry.register(ref)
    await manager.lora_registry.acquire("a")
    state = SimpleNamespace(
        lora_lease=runtime.LoRARequestLease(ref.lora_id), dispatched=dispatched, finished=False, lifecycle_id=object()
    )
    manager.rid_to_state["rid"] = state
    manager._remove_req_state("rid")
    await asyncio.sleep(0)
    assert manager.lora_registry.request_count(ref.lora_id) == int(dispatched)
    if terminal == "abort":
        manager._handle_abort_req(SimpleNamespace(rid="rid"))
        manager._handle_abort_req(SimpleNamespace(rid="rid"))
    elif terminal == "batch":
        output = SimpleNamespace(time_stats=None, rids=["rid"], finished_reasons=[None])
        await manager._handle_batch_output(output)
        assert manager.lora_registry.request_count(ref.lora_id) == 1
        output.finished_reasons = [{"type": "length"}]
        await manager._handle_batch_output(output)
        await manager._handle_batch_output(output)
    await asyncio.gather(*manager._lora_release_tasks)
    assert manager.lora_registry.request_count(ref.lora_id) == 0
    assert not manager._lora_draining_requests


@pytest.mark.asyncio
async def test_parallel_children_share_one_reference_until_last_completion(runtime):
    manager = tokenizer_methods(runtime)
    ref = runtime.LoRARef(lora_name="a", lora_path="__tensor__")
    await manager.lora_registry.register(ref)
    await manager.lora_registry.acquire("a")
    parent = SimpleNamespace(
        lora_lease=runtime.LoRARequestLease(ref.lora_id), dispatched=False, finished=False, lifecycle_id=object()
    )
    manager.rid_to_state["parent"] = parent
    manager._raise_if_logical_rid_aborted = lambda _: None
    manager._raise_if_request_aborted = lambda _: None
    manager._init_req_state = lambda obj, request=None, lifecycle_id=None: manager.rid_to_state.update(
        {obj.rid: SimpleNamespace(lora_lease=None, dispatched=True, finished=False, lifecycle_id=lifecycle_id)}
    )
    manager._register_child_rid = lambda *_: None
    for rid in ["child1", "child2"]:
        manager._init_child_req_state("parent", SimpleNamespace(rid=rid))
    manager._remove_req_state("parent")
    manager.rid_to_state["child1"].finished = True
    manager._remove_req_state("child1")
    assert manager.lora_registry.request_count(ref.lora_id) == 1
    manager._remove_req_state("child2")
    assert manager.lora_registry.request_count(ref.lora_id) == 1
    manager._handle_abort_req(SimpleNamespace(rid="child2"))
    await asyncio.gather(*manager._lora_release_tasks)
    assert manager.lora_registry.request_count(ref.lora_id) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("dispatched", [False, True])
async def test_request_failure_releases_only_after_scheduler_terminal(runtime, dispatched):
    manager = tokenizer_methods(runtime)
    ref = runtime.LoRARef(lora_name="a", lora_path="__tensor__")
    await manager.lora_registry.register(ref)
    obj = SimpleNamespace(
        rid="request",
        is_single=True,
        lora_path="a",
        max_thinking_tokens=None,
        routed_dp_rank=None,
        return_prompt_token_ids=False,
        return_logprob=False,
        stream=False,
        normalize_batch_and_arguments=lambda: None,
    )
    state = SimpleNamespace(
        obj=obj,
        lora_lease=None,
        dispatched=False,
        finished=False,
        abort_requested=False,
        lifecycle_id=object(),
        time_stats=Mock(),
        output_ids=[],
        get_text=lambda: "",
        prompt_token_ids=None,
        out_list=[],
        event=asyncio.Event(),
    )
    manager.auto_create_handle_loop = lambda: None
    manager._set_default_priority = lambda _: None
    manager._init_req_state = lambda *_: manager.rid_to_state.update({obj.rid: state}) or {obj.rid: state.lifecycle_id}
    manager.server_args.language_only = False
    manager.server_args.max_loaded_loras = None
    manager.enable_lora = True
    manager.request_logger = Mock()
    manager.tokenizer = None
    manager.is_pause_cond = asyncio.Condition()
    manager.is_pause = False
    manager.model_update_lock = runtime.RWLock()
    manager._raise_if_logical_request_aborted = lambda _: None
    manager._raise_if_logical_rid_aborted = lambda _: None
    manager.encoder_dispatch_ready = {}
    manager.abort_request = Mock()
    manager.config_value = lambda _: "0"

    async def tokenize(_):
        assert manager.lora_registry.request_count(ref.lora_id) == 1
        if not dispatched:
            raise ValueError("input length exceeds context length")
        return SimpleNamespace(rid=obj.rid)

    def dispatch(_):
        state.dispatched = True
        raise asyncio.CancelledError()

    manager._tokenize_one_request = tokenize
    manager._send_one_request = AsyncMock(side_effect=dispatch) if manager._test_async_send else dispatch
    with pytest.raises(asyncio.CancelledError if dispatched else ValueError):
        await manager.generate_request(obj).__anext__()
    if dispatched:
        assert manager.lora_registry.request_count(ref.lora_id) == 1
        terminal = SimpleNamespace(rid=obj.rid, abort_message=None, finished_reason=None, weight_versions=None)
        manager._handle_abort_req(terminal)
        manager._handle_abort_req(terminal)
    await asyncio.gather(*manager._lora_release_tasks)
    assert manager.lora_registry.request_count(ref.lora_id) == 0
    assert manager.rid_to_state == {}
    assert not manager._lora_draining_requests


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rid,lora_paths", [(None, ["a", "b"]), ("batch", ["a", "b"]), (["p0", "p1"], ["a", "b"]), (None, ["a", None])]
)
async def test_parallel_batch_keeps_lease_through_prefix_and_all_children(runtime, monkeypatch, rid, lora_paths):
    manager = tokenizer_methods(runtime)
    refs = {name: runtime.LoRARef(lora_name=name, lora_path="__tensor__") for name in ("a", "b")}
    for ref in refs.values():
        await manager.lora_registry.register(ref)
    # Execute the real request normalizer and __getitem__; autogenerated RIDs
    # differ between SGLang versions and must not create unused parent leases.
    path = sglang_source_path("srt/managers/io_struct.py")
    cls = next(node for node in ast.parse(path.read_text()).body if getattr(node, "name", None) == "GenerateReqInput")
    module = ModuleType("managed_request_fixture")
    module.__dict__.update(
        dataclass=dataclasses.dataclass,
        field=dataclasses.field,
        uuid=uuid,
        Counter=Counter,
        IntEnum=IntEnum,
        total_ordering=total_ordering,
    )
    monkeypatch.setitem(sys.modules, module.__name__, module)
    helpers = []
    for source, name in (
        ("srt/model_executor/forward_batch_info.py", "CaptureHiddenMode"),
        ("srt/managers/schedule_batch.py", "get_return_hidden_states_mode"),
    ):
        helpers.extend(
            node
            for node in ast.parse(sglang_source_path(source).read_text()).body
            if getattr(node, "name", None) == name
        )
    code = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *helpers, cls],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(code), str(path), "exec"), module.__dict__)
    obj = module.GenerateReqInput(input_ids=[[1, 2], [3, 4]], sampling_params={"n": 3}, lora_path=lora_paths, rid=rid)
    namespace = manager._init_req_state.__func__.__globals__
    namespace["APIServerReqTimeStats"] = lambda **_: Mock()
    namespace["ReqState"] = lambda out, finished, event, request, stats, **kwargs: SimpleNamespace(
        out_list=out,
        finished=finished,
        event=event,
        obj=request,
        time_stats=stats,
        lora_lease=None,
        dispatched=False,
        abort_requested=False,
        lifecycle_id=kwargs.get("lifecycle_id"),
    )
    manager.enable_trace = False
    manager.disaggregation_mode = "null"
    manager.enable_lora = True
    manager.server_args.language_only = False
    manager.server_args.max_loaded_loras = None
    manager.server_args.enable_strict_thinking = False
    manager.auto_create_handle_loop = lambda: None
    manager._set_default_priority = lambda _: None
    manager.request_logger = Mock()
    manager.tokenizer = None
    manager.is_pause = False
    manager.is_pause_cond = asyncio.Condition()
    manager.model_update_lock = runtime.RWLock()
    manager.encoder_dispatch_ready = {}
    manager.abort_request = Mock()
    expected = {name: int(name in lora_paths) for name in refs}

    async def tokenize(prompt):
        assert len(manager.rid_to_state) == 2
        assert {name: manager.lora_registry.request_count(ref.lora_id) for name, ref in refs.items()} == expected
        assert prompt.lora_id == (refs[prompt.lora_path].lora_id if prompt.lora_path else None)
        return SimpleNamespace(rid=prompt.rid, sampling_params=SimpleNamespace(), stream=False)

    manager._tokenize_one_request = tokenize
    dispatched = []

    def dispatch(request):
        assert {name: manager.lora_registry.request_count(ref.lora_id) for name, ref in refs.items()} == expected
        manager.rid_to_state[request.rid].dispatched = True
        dispatched.append(request.rid)

    manager._send_one_request = AsyncMock(side_effect=dispatch) if manager._test_async_send else dispatch

    async def response(request, http_request):
        manager.rid_to_state[request.rid].finished = True
        manager._remove_req_state(request.rid)
        await asyncio.sleep(0)
        yield {"meta_info": {"id": request.rid}}

    manager._wait_one_response = response
    outputs = [value async for value in manager.generate_request(obj)]
    await asyncio.gather(*manager._lora_release_tasks)
    assert len(dispatched) == 8 and len(set(dispatched)) == 8
    assert len(outputs) == 1 and len(outputs[0]) == 6
    assert all(manager.lora_registry.request_count(ref.lora_id) == 0 for ref in refs.values())
    assert not manager.rid_to_state and not manager._lora_draining_requests


@pytest.mark.asyncio
async def test_stop_compares_id_and_does_not_hold_update_lock_while_draining(runtime, shared_memory):
    engine = Engine(runtime)
    frozen = snapshot()
    instance = engine.relax_lora.instance_id
    loaded = await engine.client.load(frozen, instance, 1)
    lora_id = await engine.lora_registry.acquire(frozen.identity.lora_name)
    with pytest.raises(httpx.HTTPStatusError):
        await engine.client.control("stop", instance, frozen.identity.lora_name, 1, "stale-id")
    assert engine.lora_registry.num_registered_loras == 1
    row = await engine.client.control("stop", instance, frozen.identity.lora_name, 1, lora_id)
    assert row["requests"] == 1 and row["admission_closed"] and row["attempt_fenced"]
    # Loading B must finish even though A still has a running request.
    await asyncio.wait_for(engine.client.load(snapshot(2), instance, 2), timeout=1)
    assert engine.loads == 2
    with pytest.raises(ValueError):
        await engine.lora_registry.acquire(frozen.identity.lora_name)
    with pytest.raises(httpx.HTTPStatusError):
        await engine.client.control("unload", instance, frozen.identity.lora_name, 1, loaded["lora_id"])
    await engine.lora_registry.release(lora_id)
    await engine.client.control("unload", instance, frozen.identity.lora_name, 1, lora_id)
    await engine.client.control("unload", instance, frozen.identity.lora_name, 1, lora_id)
    assert engine.unloads == 1


@pytest.mark.asyncio
async def test_managed_name_never_implicitly_reloads(runtime):
    import fastapi

    manager = tokenizer_methods(runtime)
    manager._resolve_lora_path.__func__.__globals__["fastapi"] = fastapi
    manager.server_args.max_loaded_loras = None
    manager.lora_ref_cache = {snapshot().identity.lora_name: runtime.LoRARef(lora_path="__tensor__")}
    manager.load_lora_adapter = Mock(side_effect=AssertionError("implicit load must not run"))
    with pytest.raises(fastapi.HTTPException) as error:
        await manager._resolve_lora_path(SimpleNamespace(lora_path=snapshot().identity.lora_name))
    assert error.value.status_code == 409
    assert error.value.detail == "LORA_VERSION_RETIRED"
    manager.load_lora_adapter.assert_not_called()


@pytest.mark.parametrize(
    ("failure", "tp_size"),
    [(None, 1), (None, 2), ("other_rank", 2), ("fingerprint", 2), ("local_exception", 2)],
)
def test_tp_collective_uses_parallel_state_and_checks_all_ranks(runtime, failure, tp_size):
    path = sglang_source_path("srt/managers/scheduler.py")
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Scheduler")
    method = next(node for node in cls.body if getattr(node, "name", None) == "_managed_lora_update")
    group = object()
    local = SimpleNamespace(success=True, error_message=None, fingerprints={"w": 1})
    remote = SimpleNamespace(success=failure != "other_rank", error_message="rank 1 failed", fingerprints={"w": 1})
    if failure == "fingerprint":
        remote.fingerprints = {"w": 2}

    def gather(results, value, **kwargs):
        assert kwargs["group"] is group
        assert len(results) == tp_size
        results[:] = [value] + [remote] * (tp_size - 1)

    operation = Mock(return_value=local)
    if failure == "local_exception":
        operation.side_effect = ValueError("receiver validation failed")
    ns = {"torch": SimpleNamespace(distributed=SimpleNamespace(all_gather_object=Mock(side_effect=gather)))}
    ns.update(dataclass=dataclasses.dataclass, Optional=typing.Optional)
    definitions("srt/distributed/parallel_state_wrapper.py", ["ParallelState"], ns)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), ns)
    scheduler = SimpleNamespace(ps=ns["ParallelState"].trivial(tp_size=tp_size), tp_cpu_group=group)
    result = ns["_managed_lora_update"](scheduler, operation, object())
    assert result.success is (failure is None)
    ns["torch"].distributed.all_gather_object.assert_called_once()


@pytest.mark.parametrize("partial", [False, True])
def test_worker_unload_cleans_partial_state_and_retries_without_double_decrement(runtime, partial):
    path = sglang_source_path("srt/lora/lora_manager.py")
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LoRAManager")
    method = next(node for node in cls.body if getattr(node, "name", None) == "_unload_lora_adapter")
    ns = {"LoRARef": runtime.LoRARef, "LoRAUpdateOutput": SimpleNamespace}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), ns)
    ref = runtime.LoRARef(lora_name=snapshot().identity.lora_name, lora_path="__tensor__", pinned=True)
    slots = {ref.lora_id: 0}
    backend = SimpleNamespace(
        configs={ref.lora_id: object()},
        loras={ref.lora_id: object()},
        lora_refs={} if partial else {ref.lora_id: ref},
        num_pinned_loras=0 if partial else 1,
        pending_lora_load_events={},
        memory_pool=SimpleNamespace(remove_lora=lambda key: slots.pop(key, None)),
        _notify_lora_slots_updated=Mock(),
        create_lora_update_result=lambda **kw: SimpleNamespace(**kw),
    )
    unload = ns["_unload_lora_adapter"]
    assert unload(backend, ref).success
    assert unload(backend, ref).success
    assert backend.num_pinned_loras == 0
    assert not backend.configs and not backend.loras and not backend.lora_refs and not slots
    backend._notify_lora_slots_updated.assert_called_once_with({0})


@pytest.mark.asyncio
async def test_parallel_sampling_acquires_once_per_logical_prompt(runtime):
    manager = tokenizer_methods(runtime)
    manager.server_args.max_loaded_loras = None
    refs = [runtime.LoRARef(lora_name=name, lora_path="__tensor__") for name in ("a", "b")]
    for ref in refs:
        await manager.lora_registry.register(ref)
    path = sglang_source_path("srt/managers/io_struct.py")
    cls = next(node for node in ast.parse(path.read_text()).body if getattr(node, "name", None) == "GenerateReqInput")
    normalizer = next(node for node in cls.body if getattr(node, "name", None) == "_normalize_lora_paths")
    ns = {}
    exec(compile(ast.Module(body=[normalizer], type_ignores=[]), str(path), "exec"), ns)
    obj = SimpleNamespace(
        is_single=False, batch_size=2, rid=["prompt1", "prompt2"], lora_path=["a", "b"], parallel_sample_num=3
    )
    ns["_normalize_lora_paths"](obj, 6)
    assert len(obj.lora_path) == 6
    for rid in obj.rid:
        manager.rid_to_state[rid] = SimpleNamespace(
            lora_lease=None, dispatched=False, finished=False, lifecycle_id=object()
        )
    await manager._resolve_lora_path(obj)
    assert [manager.lora_registry.request_count(ref.lora_id) for ref in refs] == [1, 1]
    for rid in obj.rid:
        manager._remove_req_state(rid)
    await asyncio.gather(*manager._lora_release_tasks)
    assert [manager.lora_registry.request_count(ref.lora_id) for ref in refs] == [0, 0]


@pytest.mark.parametrize("model_type", ["qwen3", "qwen3_5"])
def test_gpu_fixture_derives_identity_and_reproducible_adapters(tmp_path, monkeypatch, model_type):
    from safetensors.torch import save_file
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    from .test_managed_lora_gpu import lora_fixture

    # Only fixture preparation runs here: no server, CUDA allocation or inference.
    config = {
        "model_type": "qwen3",
        "hidden_size": 1024,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
    }
    if model_type == "qwen3_5":
        config = {
            "model_type": model_type,
            "text_config": {"hidden_size": 1024, "num_hidden_layers": 24, "intermediate_size": 3584},
        }
    (tmp_path / "config.json").write_text(json.dumps(config))
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "Continue": 1}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]").save_pretrained(tmp_path)
    save_file({"fixture": torch.zeros(1)}, str(tmp_path / "model.safetensors"))
    monkeypatch.setenv("RELAX_LORA_TEST_MODEL", str(tmp_path))
    first = lora_fixture.__wrapped__()
    a, b = first.old.copy_tensors(), first.new.copy_tensors()
    if model_type == "qwen3":
        assert len(a) == 112
        assert a["model.layers.0.self_attn.q_proj.lora_B.weight"].shape == (2048, 8)
        assert a["model.layers.0.self_attn.v_proj.lora_B.weight"].shape == (1024, 8)
    else:
        assert len(a) == 96
        assert first.old.config["target_modules"] == ["gate_proj", "up_proj"]
        for layer in range(24):
            for module in ("gate_proj", "up_proj"):
                assert a[f"model.layers.{layer}.mlp.{module}.lora_A.weight"].shape == (8, 1024)
                assert a[f"model.layers.{layer}.mlp.{module}.lora_B.weight"].shape == (3584, 8)
    for name in a:
        assert torch.equal(b[name], -a[name] if ".lora_B." in name else a[name])
    assert first.probe_ids and first.score_prefix and first.score_suffix
    assert first.old.identity.digest != first.new.identity.digest
    save_file({"fixture": torch.ones(1)}, str(tmp_path / "model.safetensors"))
    second = lora_fixture.__wrapped__()
    assert first.old.fingerprints == second.old.fingerprints
    assert first.old.identity.digest != second.old.identity.digest


@pytest.mark.asyncio
async def test_gpu_fault_injection_changes_wire_manifest():
    from .test_managed_lora_gpu import corrupt_lora_manifest

    async def receiver(scope, receive, send):
        wire = b""
        while True:
            message = await receive()
            wire += message.get("body", b"")
            if not message.get("more_body", False):
                break
        assert json.loads(wire)["relax_manifest"] == "{} "
        assert int(dict(scope["headers"])[b"content-length"]) == len(wire)
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    async with httpx.AsyncClient(
        base_url="http://engine",
        transport=httpx.ASGITransport(app=receiver),
        event_hooks={"request": [corrupt_lora_manifest]},
    ) as client:
        response = await client.post("/relax/lora/load", json={"relax_manifest": "{}"})
        assert response.status_code == 200
