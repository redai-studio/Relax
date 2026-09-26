# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Isolated GPU validation only; never add this directory to production
PYTHONPATH.

With NO7_TEST_CONTROL_FILE set, Python startup installs hooks after the real
SGLang modules load. No HTTP fault-injection API or inference wrapper is added.
The control file selects exact request IDs; empty controls leave requests
intact. K/V capture deliberately synchronizes only selected calibration
requests, which must run outside all performance windows.
"""

import functools
import importlib.abc
import importlib.machinery
import json
import os
import sys
from pathlib import Path


CONTROL = os.environ.get("NO7_TEST_CONTROL_FILE")


def control():
    if not CONTROL:
        return {}
    try:
        return json.loads(Path(CONTROL).read_text())
    except FileNotFoundError:
        return {}


def record(directory, rid, kind, value):
    path = Path(directory) / f"{rid}.{kind}.json"
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value))
    temporary.replace(path)


def worker_id(rid, rank):
    return rid if rank == 0 else f"{rid}.rank-{rank}"


def install_req(module):
    original = module.Req.__init__

    @functools.wraps(original)
    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        config = control()
        alias = config.get("cache_alias", {}).get(self.rid)
        if alias is None:
            return
        if self.lora_id != alias["actual_uid"] or not self.extra_key.endswith(self.lora_id):
            raise AssertionError("negative control does not match the actual native adapter")
        # Change only KV lookup identity. Keep actual LoRA uid/config/weights B.
        self.extra_key = self.extra_key[: -len(self.lora_id)] + alias["cached_uid"]
        record(
            config["directory"],
            self.rid,
            "alias",
            {"actual_uid": self.lora_id, "cached_uid": alias["cached_uid"], "extra_key": self.extra_key},
        )

    module.Req.__init__ = init


def install_cache(module):
    # Support radix-enabled references and explicit no-cache diagnostics.
    # Capture before either cache frees the request's KV locations.
    cls = getattr(module, "RadixCache", None) or module.ChunkCache
    original = cls.cache_finished_req

    @functools.wraps(original)
    def finished(self, req, *args, **kwargs):
        config = control()
        capture = config.get("capture_kv", {}).get(req.rid)
        if capture is not None:
            import torch
            from sglang.srt.runtime_context import get_parallel

            # The registered sensitivity slice belongs to rank zero. Other
            # shards must not race to overwrite it with different K/V values.
            if worker_rank(get_parallel()) != 0:
                return original(self, req, *args, **kwargs)

            positions = capture["positions"]
            length = kwargs["kv_len_to_handle"]
            if not positions or min(positions) < 0 or max(positions) >= min(length, len(req.origin_input_ids)):
                raise AssertionError("KV calibration positions are outside computed prefix")
            locations = self.req_to_token_pool.req_to_token[req.req_pool_idx]
            indices = locations[torch.tensor(positions, device=locations.device)].to(torch.int64)
            pool = self.token_to_kv_pool_allocator.get_kvcache()
            values = {}
            for name, getter in (("k", pool.get_key_buffer), ("v", pool.get_value_buffer)):
                selected = getter(capture["layer"]).index_select(0, indices).reshape(-1)
                elements = capture["elements"]
                if not elements or min(elements) < 0 or max(elements) >= selected.numel():
                    raise AssertionError("KV calibration elements are outside the registered slice")
                values[name] = selected[elements].float().cpu().tolist()
            record(
                config["directory"],
                req.rid,
                "kv",
                {
                    "native_lora_id": req.lora_id,
                    "layer": capture["layer"],
                    "positions": positions,
                    "elements": capture["elements"],
                    "input_ids": list(req.origin_input_ids),
                    **values,
                },
            )
        return original(self, req, *args, **kwargs)

    cls.cache_finished_req = finished


def install_control(module):
    import asyncio

    # Use the actual defining mixin; SGLang may split control methods by release.
    classes = [
        value
        for value in vars(module).values()
        if isinstance(value, type) and "prepare_lora_publication" in value.__dict__
    ]
    if len(classes) != 1:
        raise RuntimeError("cannot locate native publication mixin for validation")
    cls = classes[0]
    original = cls.prepare_lora_publication
    reject = cls._reject_legacy_lora_mutation

    @functools.wraps(reject)
    def audited_reject(self, operation):
        config = control()
        if config.get("audit_mutations"):
            path = Path(config["directory"]) / f"mutations-{os.getpid()}.jsonl"
            with path.open("a") as sink:
                sink.write(json.dumps({"operation": operation}) + "\n")
        return reject(self, operation)

    cls._reject_legacy_lora_mutation = audited_reject

    @functools.wraps(original)
    async def prepare(self, payload):
        config = control()
        if config.get("capture_prepares"):
            path = Path(config["directory"]) / f"prepares-{os.getpid()}.jsonl"
            with path.open("a") as sink:
                sink.write(
                    json.dumps(
                        {
                            "path": payload["path"],
                            "native_lora_id": payload["native_lora_id"],
                            "boot": self.lora_version_control.owner[1],
                        }
                    )
                    + "\n"
                )
        failure = config.get("fail_prepare", {})
        if failure.get("engine_boot_id") == self.lora_version_control.owner[1]:
            self._publication_version(payload, create=True)
            raise ValueError("NO7_INJECTED_PREPARE_FAILURE")
        hold = config.get("hold_prepare", {})
        if hold.get("engine_boot_id") != self.lora_version_control.owner[1]:
            return await original(self, payload)
        version = self._publication_version(payload, create=True)
        tasks = getattr(self, "_no7_held_prepares", None)
        if tasks is None:
            self._no7_held_prepares = tasks = {}
        if payload["native_lora_id"] not in tasks:

            async def owned():
                while Path(hold["barrier"]).exists():
                    await asyncio.sleep(0.02)
                return await original(self, payload)

            task = asyncio.create_task(owned())
            task.add_done_callback(lambda done: done.exception() if not done.cancelled() else None)
            tasks[payload["native_lora_id"]] = task
        return self._publication_result(payload, version)

    cls.prepare_lora_publication = prepare

    bind_original = cls.bind_lora_publication_request

    @functools.wraps(bind_original)
    def bind(self, obj):
        bind_original(self, obj)
        config = control()
        identity = getattr(obj, "_lora_execution", None)
        if config.get("capture_attempt") and identity is not None and identity.kind == "business":
            from dataclasses import asdict

            path = Path(config["directory"]) / "train.attempt.json"
            if not path.exists():
                record(config["directory"], "train", "attempt", asdict(identity))

    cls.bind_lora_publication_request = bind


def install_communicator(module):
    original = module.FanOutCommunicator.handle_recv

    @functools.wraps(original)
    def receive(self, reply):
        config = control()
        if self._mode == "owned" and config.get("replay_old_ack"):
            previous = getattr(self, "_no7_previous_ack", None)
            if (
                previous is not None
                and self._active_key is not None
                and self._correlation_key(previous) != self._active_key
            ):
                before = (list(self._result_values or []), self._result_event.is_set())
                original(self, previous)
                after = (list(self._result_values or []), self._result_event.is_set())
                if before != after:
                    raise AssertionError("late ACK advanced a different native control operation")
                record(
                    config["directory"],
                    "control",
                    "late-ack",
                    {
                        "old": repr(self._correlation_key(previous)),
                        "active": repr(self._active_key),
                        "unchanged": True,
                    },
                )
            self._no7_previous_ack = reply
        return original(self, reply)

    module.FanOutCommunicator.handle_recv = receive


def worker_rank(ps):
    return getattr(ps, "pp_rank", 0) * getattr(ps, "tp_size", 1) + getattr(ps, "tp_rank", 0)


def execution_stream(scheduler, cuda):
    if getattr(scheduler, "enable_overlap", True) or getattr(getattr(scheduler, "ps", None), "pp_size", 1) > 1:
        return scheduler.forward_stream
    return cuda.current_stream()


def install_execution(module):
    begin_original = module.begin_execution_batch

    @functools.wraps(begin_original)
    def begin(scheduler, batch):
        selected = begin_original(scheduler, batch)
        config = control()
        rank = worker_rank(getattr(scheduler, "ps", None))
        if not config:
            return selected
        requests = batch.reqs
        delayed = config.get("delay_last_use", {})
        for req in requests:
            if req.rid == delayed.get("rid"):
                scheduler._no7_delayed_use = (req.rid, req.lora_id)
        stale = config.get("stale_slot")
        if stale and any(item.rid == stale["rid"] for item in requests):
            import torch

            manager = scheduler.tp_worker.model_runner.lora_manager
            pool = manager.memory_pool
            slot = pool.uid_to_buffer_id[stale["actual_uid"]]
            saved = pool._no7_saved_slots[stale["retired_uid"]]
            if saved["slot"] != slot:
                raise AssertionError("negative control did not reuse the retired physical slot")
            views = slot_views(pool, slot)
            with torch.cuda.stream(execution_stream(scheduler, torch.cuda)):
                scheduler._no7_restore_slot = (slot, [value.clone() for value in views])
                for value, old in zip(views, saved["weights"], strict=True):
                    value.copy_(old)
                manager._notify_lora_slots_updated({slot})
            record(
                config["directory"],
                worker_id(stale["rid"], rank),
                "stale-slot",
                {**stale, "slot": slot, "worker_rank": rank},
            )
        if not config.get("capture_mixed_batches"):
            return selected
        members = sorted({r.lora_id for r in requests if getattr(r, "lora_request_kind", None) == "business"})
        if len(members) > 1:
            manager = scheduler.tp_worker.model_runner.lora_manager
            is_verify = getattr(batch.forward_mode, "is_target_verify", lambda: False)()
            phase = "decode" if batch.forward_mode.is_decode() or is_verify else "prefill"
            scheduler._no7_batch_sequence = getattr(scheduler, "_no7_batch_sequence", 0) + 1
            for item in requests:
                if getattr(item, "lora_request_kind", None) != "business":
                    continue
                uid = item.lora_id
                proof_id = worker_id(item.rid, rank)
                path = Path(config["directory"]) / f"{proof_id}.mixed.json"
                proof = json.loads(path.read_text()) if path.exists() else {}
                if phase in proof:
                    continue
                adapter = manager.configs[uid]
                proof[phase] = {
                    "batch_id": scheduler._no7_batch_sequence,
                    "members": members,
                    "native_lora_id": uid,
                    "slot": manager.memory_pool.uid_to_buffer_id[uid],
                    "rank": adapter.r,
                    "scaling": adapter.lora_alpha / adapter.r,
                    "worker_rank": rank,
                }
                record(config["directory"], proof_id, "mixed", proof)
        return selected

    module.begin_execution_batch = begin
    original = module.finish_execution_batch

    @functools.wraps(original)
    def finish(scheduler, records, event=None):
        restore = getattr(scheduler, "_no7_restore_slot", None)
        if restore is not None:
            import torch

            manager = scheduler.tp_worker.model_runner.lora_manager
            slot, weights = restore
            with torch.cuda.stream(execution_stream(scheduler, torch.cuda)):
                for value, saved in zip(slot_views(manager.memory_pool, slot), weights, strict=True):
                    value.copy_(saved)
                manager._notify_lora_slots_updated({slot})
                # PP supplied the forward event before this test-only restore.
                # Include the restoration before allowing the slot to retire.
                event = torch.cuda.Event()
                event.record(stream=execution_stream(scheduler, torch.cuda))
            del scheduler._no7_restore_slot
        original(scheduler, records, event) if event is not None else original(scheduler, records)
        config = control()
        delayed = config.get("delay_last_use", {})
        if worker_rank(getattr(scheduler, "ps", None)) != delayed.get("worker_rank", 0):
            return
        delayed_use = getattr(scheduler, "_no7_delayed_use", None)
        if delayed_use is None:
            return
        rid, uid = delayed_use
        for version in records:
            if version.identity.native_lora_id != uid:
                continue
            import torch

            # Independent real CUDA event models a final instance reader.
            # It is consumed by retirement, not by per-request rank ACKs.
            first = not hasattr(version, "_no7_stream")
            stream = torch.cuda.Stream() if first else version._no7_stream
            event = torch.cuda.Event()
            with torch.cuda.stream(stream):
                stream.wait_event(version.last_use_event)
                if first:
                    torch.cuda._sleep(int(delayed["cycles"]))
                event.record(stream)
            version.last_use_event = event
            version._no7_stream = stream
            if first:
                record(config["directory"], rid, "last-use", {"pending": not event.query(), "uid": uid})

    module.finish_execution_batch = finish

    retire_original = module.poll_version_retirements

    @functools.wraps(retire_original)
    def retire(scheduler, requests=None):
        config = control()
        delayed_use = getattr(scheduler, "_no7_delayed_use", None)
        if config and delayed_use is not None:
            rid, uid = delayed_use
            for version in scheduler.lora_retirements.values():
                if version.identity.native_lora_id == uid and hasattr(version, "_no7_stream"):
                    pending = not version.last_use_event.query()
                    record(
                        config["directory"],
                        rid,
                        "retire-wait" if pending else "last-use-done",
                        {
                            "native_lora_id": uid,
                            "pending": pending,
                            "state": version.state,
                            "actual_unload_count": version.actual_unload_count,
                        },
                    )
        return retire_original(scheduler, requests)

    module.poll_version_retirements = retire


def slot_views(pool, slot):
    # Dense attention fixture only; each list contains one tensor per layer.
    return [
        layer[slot] for buffers in (pool.A_buffer, pool.B_buffer) for layers in buffers.values() for layer in layers
    ]


def install_pool(module):
    classes = [
        value for value in vars(module).values() if isinstance(value, type) and "begin_remove_lora" in value.__dict__
    ]
    if len(classes) != 1:
        raise RuntimeError("cannot locate native LoRA memory pool")
    cls = classes[0]
    original = cls.begin_remove_lora

    @functools.wraps(original)
    def begin(self, uid, stream, after_clear=None):
        config = control()
        if uid in config.get("capture_retired_slots", []) and uid not in self.clearing_loras:
            import torch

            saved = getattr(self, "_no7_saved_slots", {})
            if uid not in saved:
                slot = self.uid_to_buffer_id[uid]
                with torch.cuda.stream(stream):
                    saved[uid] = {"slot": slot, "weights": [value.clone() for value in slot_views(self, slot)]}
                self._no7_saved_slots = saved
        delayed = config.get("delay_clear", {})
        if uid != delayed.get("uid") or uid in self.clearing_loras:
            return original(self, uid, stream, after_clear)
        from sglang.srt.runtime_context import get_parallel

        if worker_rank(get_parallel()) != delayed.get("worker_rank", 0):
            return original(self, uid, stream, after_clear)

        def clear(slot):
            import torch

            if after_clear is not None:
                after_clear(slot)
            torch.cuda._sleep(int(delayed["cycles"]))

        result = original(self, uid, stream, clear)
        event = self.clearing_loras.get(uid)
        record(
            config["directory"],
            uid,
            "clear",
            {"pending": event is not None and not event.query(), "slot": result, "uid": uid},
        )
        return result

    cls.begin_remove_lora = begin


def install_graph(module):
    original = module.DecodeCudaGraphRunner.execute

    @functools.wraps(original)
    def execute(self, forward_batch, *args, **kwargs):
        output = original(self, forward_batch, *args, **kwargs)
        config = control()
        if config.get("capture_graph"):
            for uid in set(forward_batch.lora_ids or ()) - {None}:
                proof_id = f"{os.getpid()}.{uid}"
                if (Path(config["directory"]) / f"{proof_id}.graph.json").exists():
                    continue
                record(
                    config["directory"],
                    proof_id,
                    "graph",
                    {
                        "replayed": True,
                        "native_lora_id": uid,
                        "worker_rank": worker_rank(getattr(getattr(self, "model_runner", None), "ps", None)),
                    },
                )
        return output

    module.DecodeCudaGraphRunner.execute = execute


HOOKS = {
    "sglang.srt.model_executor.runner.decode_cuda_graph_runner": install_graph,
    "sglang.srt.managers.schedule_batch": install_req,
    "sglang.srt.managers.communicator": install_communicator,
    "sglang.srt.lora.version_control": install_execution,
    "sglang.srt.lora.mem_pool": install_pool,
    "sglang.srt.mem_cache.radix_cache": install_cache,
    "sglang.srt.mem_cache.chunk_cache": install_cache,
    "sglang.srt.managers.tokenizer_control_mixin": install_control,
}


class Loader(importlib.abc.Loader):
    def __init__(self, delegate, hook):
        self.delegate, self.hook = delegate, hook

    def create_module(self, spec):
        return self.delegate.create_module(spec)

    def exec_module(self, module):
        self.delegate.exec_module(module)
        self.hook(module)


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        hook = HOOKS.get(fullname)
        if hook is None:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is not None and spec.loader is not None:
            spec.loader = Loader(spec.loader, hook)
        return spec


if CONTROL:
    sys.meta_path.insert(0, Finder())
