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

import torch
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
    return {
        name: hashlib.sha256(
            tensor.detach().cpu().contiguous().flatten().view(torch.uint8).numpy().tobytes()
        ).hexdigest()
        for name, tensor in tensors.items()
    }


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
        self.assertEqual(rank.rank.staged_lora_terminal, ("v1", 1, "READY_LOCAL"))

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
        # Participation is unconditional: the collective happened for the duplicate.
        self.assertEqual(len(rank.receive_calls), 2)

    def test_stale_bucket_participates_then_reports(self):
        rank = _StagedRank()
        rank.begin("v1", 1)
        result, _ = rank.bucket("v1", 2, 0)
        self.assertFalse(result.success)
        self.assertIsNot(result.clean, False)
        self.assertEqual(len(rank.receive_calls), 1)
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
        self.assertEqual(rank.rank.staged_lora_terminal, ("v1", 1, "ABORTED"))
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
        self.assertEqual(rank.lora_manager.unloaded_names, ["v1", "v1"])

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
        self.assertEqual(rank.lora_manager.unloaded_names, ["v1", "v1", "v1"])

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

        class _Tokenizer:
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
            assert tokenizer.lora_publication_attempts == {}

            assert (await tokenizer.unload(None)).success  # reclaim: name-scoped
            assert tokenizer.fanned_out == ["unload", "unload"]

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
