# Copyright (c) 2026 Relax Authors. All Rights Reserved.

# These tests poke SGLang's staged LoRA publication internals (op=begin/bucket/end/
# unload) from the Relax side; they belong to the SGLang patch in docker/patch and are
# kept here so the docker patch stays runtime-only. Move them upstream with the patch if
# it is ever submitted to SGLang.

"""Unit tests for the staged LoRA publication protocol
(op=begin/bucket/end/unload).

No GPU and no real broadcast: the receive step is stubbed out, so what is under
test is the state machine and, more importantly, the *verdicts* it reports --
which kind of failure leaves the rank provably absent and which one makes it
unknown.
"""

import asyncio
import hashlib
import unittest
from types import SimpleNamespace

import pytest
import torch


pytest.importorskip("sglang.srt.lora.utils", exc_type=ImportError)
pytest.importorskip("sglang.srt.managers.io_struct", exc_type=ImportError)
pytest.importorskip("sglang.srt.managers.tokenizer_control_mixin", exc_type=ImportError)
pytest.importorskip("sglang.srt.model_executor.model_runner", exc_type=ImportError)

from sglang.srt.lora.utils import verify_lora_tensor_checksums
from sglang.srt.managers.io_struct import (
    LoRAUpdateOutput,
    UpdateLoRAFromDistributedReqInput,
)
from sglang.srt.managers.tokenizer_control_mixin import (
    TokenizerControlMixin,
    _merge_lora_update_results,
)
from sglang.srt.model_executor.model_runner import ModelRunner


ADAPTER_CONFIG = {"target_modules": ["q_proj"], "r": 8, "lora_alpha": 16}


def _checksums(tensors):
    """RFC 9.4 manifest entry, written out literally so the test is an
    independent statement of the format rather than a call into the code under
    test."""
    out = {}
    for name, tensor in tensors.items():
        frozen = tensor.detach().cpu().contiguous()
        hasher = hashlib.sha256()
        hasher.update(str(frozen.dtype).replace("torch.", "").encode())
        hasher.update(str(list(frozen.shape)).encode())
        hasher.update(frozen.view(torch.uint8).numpy().tobytes())
        out[name] = hasher.hexdigest()
    return out


class _FakeLoRAManager:
    """Records what the staged protocol asked of the real load/unload paths."""

    def __init__(self, load_success=True):
        self.base_hf_config = SimpleNamespace(vocab_size=1000)
        self.load_success = load_success
        self.validated = []
        self.loaded = []
        self.unloaded_names = []

    def validate_new_adapter(self, adapter, lora_ref):
        self.validated.append((lora_ref.lora_name, adapter.r))

    def load_lora_adapter_from_tensors(self, lora_ref, tensors, config_dict, added_tokens_config=None):
        self.loaded.append((lora_ref, dict(tensors)))
        if not self.load_success:
            return LoRAUpdateOutput(success=False, error_message="boom")
        return LoRAUpdateOutput(success=True)

    def unload_lora_adapter_by_name(self, lora_name):
        self.unloaded_names.append(lora_name)
        return LoRAUpdateOutput(success=True)


class _StagedRank:
    """One rank of the engine side, with the collective stubbed out."""

    def __init__(self, load_success=True):
        self.lora_manager = _FakeLoRAManager(load_success=load_success)
        self.ps = SimpleNamespace(tp_rank=0)
        self.rank = ModelRunner.__new__(ModelRunner)
        self.rank.lora_manager = self.lora_manager
        self.rank.ps = self.ps
        self.rank._prepare_staged_lora = lambda lora_id: None
        self.receive_calls = []
        self.rank._receive_lora_buckets = self._receive

    def _receive(self, names, dtypes, shapes, group_name, bucket_sizes):
        # One deterministic tensor per announced name, so a replayed bucket is
        # byte-identical to the original and the manifest can be computed up front.
        self.receive_calls.append(tuple(names))
        return {name: torch.full((2,), float(index + 1), dtype=torch.float32) for index, name in enumerate(names)}

    def begin(self, name, attempt, config=None, lora_id="id-1"):
        return self.rank.stage_lora_begin(
            UpdateLoRAFromDistributedReqInput(
                op="begin",
                protocol_version=2,
                lora_name=name,
                attempt_id=attempt,
                lora_id=lora_id,
                config_dict=ADAPTER_CONFIG if config is None else config,
                pinned=True,
            )
        )

    def bucket(self, name, attempt, index, bucket_names=("a",), lora_id="id-1"):
        # Tensors are recomputed exactly as _receive will, so the test can build the
        # sender's manifest from the same deterministic content.
        tensors = {
            bucket_name: torch.full((2,), float(i + 1), dtype=torch.float32)
            for i, bucket_name in enumerate(bucket_names)
        }
        result = self.rank.stage_lora_bucket(
            UpdateLoRAFromDistributedReqInput(
                op="bucket",
                lora_name=name,
                attempt_id=attempt,
                bucket_index=index,
                lora_id=lora_id,
                names=list(bucket_names),
                dtypes=["float32"] * len(bucket_names),
                shapes=[[2]] * len(bucket_names),
                bucket_sizes=[len(bucket_names)],
            )
        )
        return result, tensors

    def end(self, name, attempt, expected, lora_id="id-1"):
        return self.rank.stage_lora_end(
            UpdateLoRAFromDistributedReqInput(
                op="end",
                lora_name=name,
                attempt_id=attempt,
                lora_id=lora_id,
                config_dict=ADAPTER_CONFIG,
                expected_checksums=expected,
            )
        )

    def unload(self, name, attempt=None):
        return self.rank.unload_lora_from_distributed(
            UpdateLoRAFromDistributedReqInput(op="unload", lora_name=name, attempt_id=attempt, reason="test")
        )


class TestChecksumVerification(unittest.TestCase):
    def _tensors(self):
        return {
            "a": torch.arange(4, dtype=torch.float32),
            "b": torch.zeros(2, 2, dtype=torch.bfloat16),
        }

    def test_accepts_a_matching_manifest(self):
        tensors = self._tensors()
        verify_lora_tensor_checksums(tensors, _checksums(tensors))

    def test_relax_sender_manifest_is_accepted_verbatim(self):
        # Regression: the engine hashed the bytes alone while Relax hashed
        # dtype+shape+bytes, so every real publication was rejected as a value
        # mismatch even though each side's own tests passed.
        from relax.distributed.checkpoint_service.lora_publication import tensor_manifest_hash

        tensors = self._tensors()
        manifest = {name: tensor_manifest_hash(t) for name, t in tensors.items()}
        self.assertEqual(manifest, _checksums(tensors))
        verify_lora_tensor_checksums(tensors, manifest)

    def test_value_drift_is_detected(self):
        tensors = self._tensors()
        expected = _checksums(tensors)
        tensors["a"] = tensors["a"] + 1
        with self.assertRaises(RuntimeError) as ctx:
            verify_lora_tensor_checksums(tensors, expected)
        self.assertIn("MISMATCH", str(ctx.exception))

    def test_missing_and_extra_tensors_are_detected(self):
        tensors = self._tensors()
        expected = _checksums(tensors)
        with self.assertRaises(RuntimeError):
            verify_lora_tensor_checksums({"a": tensors["a"]}, expected)
        with self.assertRaises(RuntimeError):
            verify_lora_tensor_checksums({**tensors, "c": tensors["a"]}, expected)


class TestStagedStateMachine(unittest.TestCase):
    def test_happy_path_loads_the_manifested_tensors(self):
        rank = _StagedRank()
        self.assertTrue(rank.begin("v1", 1).success)
        self.assertEqual(rank.lora_manager.validated, [("v1", 8)])

        result, tensors = rank.bucket("v1", 1, 0)
        self.assertTrue(result.success)
        self.assertEqual(rank.rank.staged_lora_key, ("v1", 1))

        self.assertTrue(rank.end("v1", 1, _checksums(tensors)).success)
        ((lora_ref, loaded),) = rank.lora_manager.loaded
        self.assertEqual(lora_ref.lora_name, "v1")
        self.assertEqual(lora_ref.lora_id, "id-1")
        # READY_LOCAL: the candidate is gone but must not be revivable.
        self.assertIsNone(rank.rank.staged_lora_key)
        self.assertEqual(rank.rank.staged_lora_terminals[("v1", 1)], "READY_LOCAL")

    def test_buckets_must_arrive_in_bucket_order(self):
        rank = _StagedRank()
        rank.begin("v1", 1)
        _, first = rank.bucket("v1", 1, 0, bucket_names=("a",))
        _, second = rank.bucket("v1", 1, 1, bucket_names=("b",))
        self.assertEqual(sorted(rank.rank.staged_lora_buckets), [0, 1])
        self.assertTrue(rank.end("v1", 1, _checksums({**first, **second})).success)
        ((_, loaded),) = rank.lora_manager.loaded
        self.assertEqual(sorted(loaded), ["a", "b"])

    def test_duplicate_bucket_is_drained_and_stays_idempotent(self):
        rank = _StagedRank()
        rank.begin("v1", 1)
        rank.bucket("v1", 1, 0)
        stashed = rank.rank.staged_lora_buckets[0]
        result, _ = rank.bucket("v1", 1, 0)
        self.assertTrue(result.success)
        self.assertIs(rank.rank.staged_lora_buckets[0], stashed)
        # The protocol-2 sender never broadcasts the same attempt/bucket twice.
        self.assertEqual(len(rank.receive_calls), 1)

    def test_stale_bucket_participates_then_reports(self):
        rank = _StagedRank()
        rank.begin("v1", 1)
        result, _ = rank.bucket("v1", 2, 0)
        self.assertFalse(result.success)
        self.assertIsNot(result.clean, False)
        self.assertEqual(len(rank.receive_calls), 0)
        self.assertEqual(rank.rank.staged_lora_buckets, {})

    def test_terminal_attempt_is_never_revived(self):
        rank = _StagedRank()
        rank.begin("v1", 1)
        _, tensors = rank.bucket("v1", 1, 0)
        rank.end("v1", 1, _checksums(tensors))

        # Same-attempt late RPCs are answered from the terminal state instead of
        # reviving the candidate: drained buckets and a repeated End are idempotent
        # success, and nothing is loaded twice.
        late_bucket, _ = rank.bucket("v1", 1, 0)
        self.assertTrue(late_bucket.success)
        self.assertIsNone(rank.rank.staged_lora_key)
        self.assertTrue(rank.end("v1", 1, _checksums(tensors)).success)
        self.assertTrue(rank.begin("v1", 1).success)
        self.assertEqual(len(rank.lora_manager.loaded), 1)

    def test_manifest_mismatch_is_clean_and_aborts_the_attempt(self):
        rank = _StagedRank()
        rank.begin("v1", 1)
        _, tensors = rank.bucket("v1", 1, 0)
        wrong = {"a": "0" * 64}
        result = rank.end("v1", 1, wrong)
        self.assertFalse(result.success)
        self.assertIsNot(result.clean, False)
        self.assertEqual(rank.lora_manager.loaded, [])
        self.assertEqual(rank.rank.staged_lora_terminals[("v1", 1)], "ABORTED")
        self.assertFalse(rank.end("v1", 1, _checksums(tensors)).success)

    def test_end_without_manifest_is_refused_before_loading(self):
        rank = _StagedRank()
        rank.begin("v1", 1)
        rank.bucket("v1", 1, 0)
        result = rank.end("v1", 1, None)
        self.assertFalse(result.success)
        self.assertEqual(rank.lora_manager.loaded, [])

    def test_load_failure_reports_an_unknown_rank(self):
        rank = _StagedRank(load_success=False)
        rank.begin("v1", 1)
        _, tensors = rank.bucket("v1", 1, 0)
        result = rank.end("v1", 1, _checksums(tensors))
        self.assertFalse(result.success)
        self.assertIs(result.clean, False)

    def test_begin_refuses_a_second_candidate_and_bad_config(self):
        rank = _StagedRank()
        self.assertFalse(rank.begin("v1", 1, config={}).success)
        self.assertIsNone(rank.rank.staged_lora_key)

        self.assertTrue(rank.begin("v1", 1).success)
        self.assertFalse(rank.begin("v2", 1).success)
        # Duplicate Begin of the same attempt is idempotent and keeps the stash.
        rank.bucket("v1", 1, 0)
        self.assertTrue(rank.begin("v1", 1).success)
        self.assertEqual(sorted(rank.rank.staged_lora_buckets), [0])

    def test_new_attempt_starts_empty_but_only_after_cleanup(self):
        rank = _StagedRank()
        rank.begin("v1", 1)
        rank.bucket("v1", 1, 0)
        # A full retry (new attempt_id) is only legal once the publisher cleaned the
        # previous attempt back to absent; without that cleanup the engine refuses
        # rather than silently mixing two attempts' buckets.
        self.assertFalse(rank.begin("v1", 2).success)
        self.assertTrue(rank.unload("v1").success)
        self.assertTrue(rank.begin("v1", 2).success)
        self.assertEqual(rank.rank.staged_lora_key, ("v1", 2))
        self.assertEqual(rank.rank.staged_lora_buckets, {})

    def test_unload_discards_the_candidate_and_is_idempotent(self):
        rank = _StagedRank()
        rank.begin("v1", 1)
        rank.bucket("v1", 1, 0)
        self.assertTrue(rank.unload("v1", 1).success)
        self.assertIsNone(rank.rank.staged_lora_key)
        self.assertEqual(rank.rank.staged_lora_buckets, {})
        self.assertTrue(rank.unload("v1", 1).success)
        self.assertEqual(rank.lora_manager.unloaded_names, ["v1"])

    def test_stale_unload_cannot_strip_the_current_candidate(self):
        """A late cleanup of attempt 1 must not discard attempt 2's stash."""

        rank = _StagedRank()
        rank.begin("v1", 1)
        rank.bucket("v1", 1, 0)
        self.assertTrue(rank.unload("v1", 1).success)
        self.assertTrue(rank.begin("v1", 2).success)
        rank.bucket("v1", 2, 0)

        self.assertTrue(rank.unload("v1", 1).success)  # duplicated old cleanup
        self.assertEqual(rank.rank.staged_lora_key, ("v1", 2))
        self.assertEqual(sorted(rank.rank.staged_lora_buckets), [0])
        self.assertEqual(rank.lora_manager.unloaded_names, ["v1"])

    def test_stale_unload_cannot_unload_a_newer_loaded_instance(self):
        """Same name, new attempt: the stale cleanup must not reach the
        weights."""

        rank = _StagedRank()
        rank.begin("v1", 1)
        _, tensors = rank.bucket("v1", 1, 0)
        self.assertTrue(rank.end("v1", 1, _checksums(tensors)).success)
        self.assertTrue(rank.unload("v1", 1).success)  # attempt 1 cleaned up
        self.assertEqual(rank.lora_manager.unloaded_names, ["v1"])

        # The retry reuses the immutable name, so only the attempt tells them apart.
        self.assertTrue(rank.begin("v1", 2).success)
        _, tensors = rank.bucket("v1", 2, 0)
        self.assertTrue(rank.end("v1", 2, _checksums(tensors)).success)

        self.assertTrue(rank.unload("v1", 1).success)  # duplicated old cleanup
        self.assertEqual(rank.lora_manager.unloaded_names, ["v1"])

        # The current attempt's cleanup still unloads, and reclaim (no attempt_id: a
        # published version never gets another attempt) stays name-scoped.
        self.assertTrue(rank.unload("v1", 2).success)
        self.assertTrue(rank.unload("v1").success)
        self.assertEqual(rank.lora_manager.unloaded_names, ["v1", "v1"])

    def test_stale_end_and_bucket_stay_drained_after_a_stale_unload(self):
        """The fence must not resurrect an attempt either."""

        rank = _StagedRank()
        rank.begin("v1", 1)
        rank.bucket("v1", 1, 0)
        rank.unload("v1", 1)
        _, tensors = rank.bucket("v1", 1, 0)  # duplicate of the discarded attempt
        result = rank.end("v1", 1, _checksums(tensors))

        self.assertFalse(result.success)
        self.assertIn("ABORTED", result.error_message)
        self.assertEqual(rank.lora_manager.loaded, [])


class TestTokenizerStagedBookkeeping(unittest.TestCase):
    def test_merge_keeps_an_unknown_rank_unknown(self):
        clean = LoRAUpdateOutput(success=False, error_message="refused")
        unknown = LoRAUpdateOutput(success=False, error_message="dirty", clean=False)
        merged = _merge_lora_update_results([clean, unknown])
        self.assertFalse(merged.success)
        self.assertIs(merged.clean, False)
        self.assertIn("refused", merged.error_message)
        self.assertIn("dirty", merged.error_message)
        self.assertIsNone(_merge_lora_update_results([clean]).clean)

    def test_destructive_updates_are_refused_while_a_candidate_is_in_flight(self):
        owner = SimpleNamespace(staged_lora_publication=None)
        self.assertIsNone(TokenizerControlMixin._reject_if_staged_lora_pending(owner, "update"))
        owner.staged_lora_publication = SimpleNamespace(lora_name="v1", attempt_id=7)
        refusal = TokenizerControlMixin._reject_if_staged_lora_pending(owner, "update")
        self.assertIn("v1", refusal)
        self.assertIn("7", refusal)

    def test_unload_is_fenced_by_the_attempt_id(self):
        """A late cleanup of a superseded attempt must not touch the
        registry."""

        class _Registry:
            def __init__(self):
                self.unregistered = []

            def get_all_adapters(self):
                return {"v1": object()}

            async def unregister(self, lora_name):
                self.unregistered.append(lora_name)
                return "id-v1"

            async def wait_for_unload(self, lora_id):
                return None

        class _Tokenizer(TokenizerControlMixin):
            def __init__(self):
                self.lora_update_lock = asyncio.Lock()
                self.lora_publication_attempts = {"v1": 2}
                self.staged_lora_publication = SimpleNamespace(
                    lora_name="v1", attempt_id=2, lora_ref=SimpleNamespace()
                )
                self.lora_registry = _Registry()
                self.fanned_out = []

            async def update_lora_adapter_communicator(self, obj):
                self.fanned_out.append(obj.op)
                return [LoRAUpdateOutput(success=True)]

            async def unload(self, attempt):
                return await TokenizerControlMixin._update_lora_from_distributed_unload(
                    self,
                    UpdateLoRAFromDistributedReqInput(op="unload", lora_name="v1", attempt_id=attempt, reason="test"),
                )

        async def run():
            tokenizer = _Tokenizer()
            assert (await tokenizer.unload(1)).success
            # Nothing happened: attempt 2 still owns the name on this engine.
            assert tokenizer.fanned_out == []
            assert tokenizer.lora_registry.unregistered == []
            assert tokenizer.staged_lora_publication.attempt_id == 2

            assert (await tokenizer.unload(2)).success
            assert tokenizer.fanned_out == ["unload"]
            assert tokenizer.lora_registry.unregistered == ["v1"]
            assert tokenizer.staged_lora_publication is None
            # The name is gone, so the fence entry goes with it.
            assert tokenizer.lora_publication_attempts == {"v1": 2}

            assert (await tokenizer.unload(None)).success  # reclaim: name-scoped
            assert tokenizer.fanned_out == ["unload", "unload"]

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()


class _ProtocolRegistry:
    def __init__(self):
        self.adapters = {}

    def get_all_adapters(self):
        return self.adapters

    async def register(self, ref):
        self.adapters[ref.lora_name] = ref

    async def unregister(self, name):
        return self.adapters.pop(name).lora_id

    async def wait_for_unload(self, lora_id):
        pass


class _ProtocolTokenizer(TokenizerControlMixin):
    """Real control methods and worker state machine, with only transport
    mocked."""

    def __init__(self):
        self.server_args = SimpleNamespace(enable_lora=True, dp_size=1)
        self.lora_update_lock = asyncio.Lock()
        self.model_update_lock = SimpleNamespace(reader_lock=asyncio.Lock())
        self.staged_lora_publication = None
        self.lora_publication_attempts = {}
        self.lora_registry = _ProtocolRegistry()
        self.worker = _StagedRank()
        self.operations = []
        self._ensure_lora_publication_state()

    def auto_create_handle_loop(self):
        pass

    async def update_lora_adapter_communicator(self, obj):
        self.operations.append(obj.op)
        method = {
            "begin": self.worker.rank.stage_lora_begin,
            "bucket": self.worker.rank.stage_lora_bucket,
            "end": self.worker.rank.stage_lora_end,
            "unload": self.worker.rank.unload_lora_from_distributed,
        }[obj.op]
        return [method(obj)]

    async def control(self, op, attempt=1, **kwargs):
        from relax.distributed.checkpoint_service.lora_publication import materialize_adapter_snapshot

        snapshot = materialize_adapter_snapshot(ADAPTER_CONFIG, {"a": torch.ones(2)})
        payload = dict(
            op=op,
            protocol_version=2,
            version_id=1,
            digest=snapshot.digest,
            lora_name="relax_policy_lora@test-1-" + snapshot.digest[:16],
            attempt_id=attempt,
            config_dict=ADAPTER_CONFIG,
            pinned=True,
            expected_checksums=snapshot.manifest,
            names=["a"],
            dtypes=["float32"],
            shapes=[[2]],
            bucket_sizes=[1],
            bucket_index=0,
            engine_incarnations=[self.lora_engine_incarnation],
        )
        payload.update(kwargs)
        return await self.update_lora_from_distributed(UpdateLoRAFromDistributedReqInput(**payload))


def test_protocol_cleanup_before_begin_and_after_other_attempts_is_final():
    async def run():
        engine = _ProtocolTokenizer()
        assert (await engine.control("unload")).success
        assert not (await engine.control("begin")).success
        assert engine.worker.rank.staged_lora_key is None
        assert (await engine.control("begin", attempt=2)).success
        assert (await engine.control("bucket", attempt=2)).success
        assert (await engine.control("end", attempt=2)).success
        assert (await engine.control("unload", attempt=1)).success
        assert len(engine.lora_registry.adapters) == 1
        assert (await engine.control("unload", attempt=2)).success
        assert not (await engine.control("begin", attempt=1)).success
        assert not (await engine.control("begin", attempt=2)).success
        assert engine.worker.lora_manager.unloaded_names.count(next(iter(engine.lora_publication_attempts))) == 2

    asyncio.run(run())


def test_protocol_duplicate_begin_bucket_end_do_not_repeat_transport_or_load():
    async def run():
        engine = _ProtocolTokenizer()
        first = await engine.control("begin")
        second = await engine.control("begin")
        assert first.success and second.success
        assert first.publication_receipt == second.publication_receipt
        assert engine.operations == ["begin"]
        replies = await asyncio.gather(engine.control("bucket"), engine.control("bucket"))
        assert all(reply.success for reply in replies)
        assert engine.operations.count("bucket") == 1
        assert len(engine.worker.receive_calls) == 1
        assert (await engine.control("end")).success
        assert (await engine.control("end")).success
        assert len(engine.worker.lora_manager.loaded) == 1
        assert (await engine.control("status")).publication_receipt["state"] == "READY_LOCAL"

    asyncio.run(run())


def test_protocol_legacy_sender_and_managed_bypass_are_rejected():
    from sglang.srt.managers.io_struct import UnloadLoRAAdapterReqInput

    async def run():
        engine = _ProtocolTokenizer()
        assert not (await engine.control("begin", protocol_version=1)).success
        assert engine.operations == []
        assert (await engine.control("begin")).success
        assert (await engine.control("bucket")).success
        ready = await engine.control("end")
        name = ready.publication_receipt["lora_name"]
        assert not (await engine.unload_lora_adapter(UnloadLoRAAdapterReqInput(lora_name=name))).success
        assert not (await engine.control("legacy")).success
        assert name in engine.lora_registry.adapters
        assert engine.worker.lora_manager.unloaded_names == []

    asyncio.run(run())


def test_protocol_incarnation_change_is_not_ready():
    async def run():
        engine = _ProtocolTokenizer()
        prepared = await engine.control("begin")
        assert (await engine.control("bucket")).success
        assert (await engine.control("end")).success
        receipt = prepared.publication_receipt
        engine.lora_engine_incarnation = "new-process"
        result = await engine.control("status", engine_incarnations=[receipt["engine_incarnation"]])
        assert not result.success and result.clean is False

    asyncio.run(run())


def test_sender_reentry_and_duplicate_control_keep_collective_counts_matched():
    import pytest

    from relax.agentic.session.lora_version import LoRAVersionError, LoRAVersionRegistry
    from relax.distributed.checkpoint_service.lora_publication import (
        EngineReply,
        LoRAPublisher,
        materialize_adapter_snapshot,
    )

    registry = LoRAVersionRegistry(deployment_epoch="test")
    engines = {name: _ProtocolTokenizer() for name in ("engine0", "engine1")}
    sent = []
    received = {name: [] for name in engines}
    snapshot = materialize_adapter_snapshot(ADAPTER_CONFIG, {"a": torch.ones(2), "b": torch.ones(2)})
    for name, engine in engines.items():

        def receive(names, dtypes, shapes, group_name, bucket_sizes, engine_name=name):
            ordinal = len(received[engine_name])
            assert ordinal < len(sent), "receiver entered an unmatched collective"
            assert tuple(names) == sent[ordinal]
            received[engine_name].append(tuple(names))
            return {key: snapshot.tensors[key].clone() for key in names}

        engine.worker.rank._receive_lora_buckets = receive

    def fire(endpoint, payload):
        return {name: dict(payload) for name in engines}

    def collect(pending):
        replies = {}
        for name, payload in pending.items():

            async def deliver():
                req = UpdateLoRAFromDistributedReqInput(**payload)
                result = await engines[name].update_lora_from_distributed(req)
                if payload["op"] == "bucket":
                    duplicate = await engines[name].update_lora_from_distributed(
                        UpdateLoRAFromDistributedReqInput(**payload)
                    )
                    assert duplicate.success
                return result

            result = asyncio.run(deliver())
            replies[name] = EngineReply(result.success, receipt=result.publication_receipt)
        return replies

    def broadcast(names, index):
        before = list(sent)
        with pytest.raises(LoRAVersionError) as error:
            publisher.publish(snapshot, [1, 1], version_id=1)
        assert error.value.code == "PUBLICATION_IN_PROGRESS"
        assert sent == before
        sent.append(tuple(names))

    publisher = LoRAPublisher(fire=fire, collect=collect, broadcast=broadcast, registry=registry)
    assert publisher.publish(snapshot, [1, 1], version_id=1).status == "PUBLISHED"
    assert sent == [("a",), ("b",)]
    assert received == {"engine0": sent, "engine1": sent}


def test_cancelled_unload_waiter_cannot_bypass_native_request_refs():
    async def run():
        engine = _ProtocolTokenizer()
        assert (await engine.control("begin")).success
        assert (await engine.control("bucket")).success
        assert (await engine.control("end")).success
        entered = asyncio.Event()
        drained = asyncio.Event()

        async def wait_for_unload(lora_id):
            entered.set()
            await drained.wait()

        engine.lora_registry.wait_for_unload = wait_for_unload
        first = asyncio.create_task(engine.control("unload"))
        await entered.wait()
        first.cancel()
        try:
            await first
        except asyncio.CancelledError:
            pass
        second = asyncio.create_task(engine.control("unload"))
        await asyncio.sleep(0)
        assert not second.done()
        assert engine.worker.lora_manager.unloaded_names == []
        drained.set()
        assert (await second).success
        assert len(engine.worker.lora_manager.unloaded_names) == 1

    asyncio.run(run())


def test_cancelled_bucket_waiter_keeps_one_engine_owned_receive():
    async def run():
        engine = _ProtocolTokenizer()
        assert (await engine.control("begin")).success
        entered = asyncio.Event()
        broadcast = asyncio.Event()
        communicator = engine.update_lora_adapter_communicator

        async def delayed_receive(obj):
            if obj.op == "bucket":
                entered.set()
                await broadcast.wait()
            return await communicator(obj)

        engine.update_lora_adapter_communicator = delayed_receive
        first = asyncio.create_task(engine.control("bucket"))
        await entered.wait()
        first.cancel()
        try:
            await first
        except asyncio.CancelledError:
            pass
        second = asyncio.create_task(engine.control("bucket"))
        await asyncio.sleep(0)
        assert not second.done()
        broadcast.set()
        assert (await second).success
        assert engine.operations.count("bucket") == 1
        assert len(engine.worker.receive_calls) == 1
        assert (await engine.control("end")).success

    asyncio.run(run())


def test_confirmed_unload_compacts_controls_and_rejects_late_publication():
    async def run():
        engine = _ProtocolTokenizer()
        for attempt in range(1, 6):
            assert (await engine.control("begin", attempt)).success
            assert (await engine.control("bucket", attempt)).success
            assert (await engine.control("end", attempt)).success
            assert engine.lora_publications
            assert len(engine.lora_publication_controls) == 3
            assert (await engine.control("unload", attempt)).success
            await asyncio.sleep(0)
            assert not engine.lora_publications
            assert not engine.lora_publication_controls
            assert not engine.lora_publication_unloads
            before = list(engine.operations)
            assert (await engine.control("unload", attempt)).success
            for op in ("begin", "bucket", "end"):
                assert not (await engine.control(op, attempt)).success
            assert engine.operations == before
        assert len(engine.worker.lora_manager.unloaded_names) == 5

    asyncio.run(run())


def test_unconfirmed_unload_retains_full_publication_record():
    async def run():
        engine = _ProtocolTokenizer()
        assert (await engine.control("begin")).success
        assert (await engine.control("bucket")).success
        assert (await engine.control("end")).success
        original = engine.update_lora_adapter_communicator

        async def fail_unload(obj):
            if obj.op == "unload":
                return [LoRAUpdateOutput(success=False, clean=False, error_message="unknown unload")]
            return await original(obj)

        engine.update_lora_adapter_communicator = fail_unload
        assert not (await engine.control("unload")).success
        assert engine.lora_publications
        assert engine.lora_publication_controls
        assert engine.lora_publication_unloads
        assert not engine.lora_publication_unloaded

    asyncio.run(run())


def test_receiver_incarnation_change_cannot_complete_source_collective():
    import pytest

    from relax.agentic.session.lora_version import LoRAVersionRegistry
    from relax.distributed.checkpoint_service.lora_publication import (
        EngineReply,
        LoRAPublicationError,
        LoRAPublisher,
        materialize_adapter_snapshot,
    )
    from tests.agentic.lora_helpers import commit_ready

    registry = LoRAVersionRegistry(deployment_epoch="test")
    old = registry.allocate("a" * 64, version_id=1)
    commit_ready(registry, old.version_id, old.attempt_id)
    engines = {name: _ProtocolTokenizer() for name in ("engine0", "engine1")}
    pending_bucket = {}

    def fire(endpoint, payload):
        pending = {name: dict(payload) for name in engines}
        if payload["op"] == "bucket":
            engines["engine1"].lora_engine_incarnation = "restarted-engine"
            pending_bucket.update(pending)
        return pending

    def collect(pending):
        replies = {}
        for name, payload in pending.items():
            result = asyncio.run(
                engines[name].update_lora_from_distributed(UpdateLoRAFromDistributedReqInput(**payload))
            )
            replies[name] = EngineReply(
                result.success, ambiguous=result.clean is False, receipt=result.publication_receipt
            )
        return replies

    def broadcast(names, index):
        # Real receiver validation runs here. CPU receives record participation;
        # the source must not complete when any receiver refused before entry.
        collect(pending_bucket)
        counts = [len(engine.worker.receive_calls) for engine in engines.values()]
        assert counts == [1, 0]
        raise RuntimeError("collective cannot complete with a missing receiver")

    publisher = LoRAPublisher(fire=fire, collect=collect, broadcast=broadcast, registry=registry)
    snapshot = materialize_adapter_snapshot(ADAPTER_CONFIG, {"a": torch.ones(2)})
    with pytest.raises(LoRAPublicationError) as error:
        publisher.publish(snapshot, [1], version_id=2)
    assert error.value.kind == "FATAL"
    assert registry.default_version == old.version_id
    assert registry.status().capacity_owning == 2
    assert all("unload" not in engine.operations for engine in engines.values())
