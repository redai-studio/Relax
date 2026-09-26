# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Golden tests for RFC paired-bootstrap and artifact semantics."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from relax.engine.sft.eval.acceptance import (
    encoded_pair_id,
    paired_bootstrap,
    preference_eval_chunk_sizes,
    preference_eval_local_batch_sizes,
    record_probe_contract,
    write_pair_artifacts,
)


@pytest.mark.parametrize(
    ("global_batch_size", "expected"),
    [
        (1, [1] * 512),
        (30, [30] * 17 + [2]),
        (32, [32] * 16),
        (512, [512]),
        (513, [512]),
    ],
)
def test_preference_eval_chunks_preserve_all_512_unique_pairs(global_batch_size, expected):
    sizes = preference_eval_chunk_sizes(512, global_batch_size)
    assert sizes == expected
    assert sum(sizes) == 512
    assert max(sizes) <= global_batch_size


def test_preference_eval_partial_chunk_uses_actual_per_rank_batch_size():
    assert preference_eval_local_batch_sizes(512, 30, dp_size=2) == [15] * 17 + [1]
    with pytest.raises(ValueError, match="divisible by data-parallel size"):
        preference_eval_local_batch_sizes(512, 31, dp_size=2)


def test_paired_bootstrap_matches_pcg64_float64_golden_fixture():
    result = paired_bootstrap([1.0, 0.0, 1.0, 0.5])
    assert result["point_estimate"] == 0.625
    assert result["lower_95"] == 0.25
    assert result["indices_sha256"] == "dc7ec9501aead17b5115aad49b487302aed95c384d6c2aadcf67b56a849ef53f"
    assert result["replicates_sha256"] == "52f63cf64c72be2904c2911052092af8fc2cd2f05444fa17898c79a1dc22e174"
    assert result["passes_lower_bound_gt_0_50"] is False


def _pair(pair_id: str, token: int):
    return SimpleNamespace(
        pair_id=pair_id,
        chosen_tokens=np.asarray([1, token]),
        rejected_tokens=np.asarray([1, token + 1]),
        chosen_loss_mask=np.asarray([0, 1]),
        rejected_loss_mask=np.asarray([0, 1]),
        chosen_score_position=1,
        rejected_score_position=1,
    )


def test_pair_artifacts_preserve_original_ids_and_require_identical_final_plan(tmp_path):
    pairs = [_pair(f"pair-{index}", index) for index in range(4)]
    record_probe_contract(str(tmp_path), "reward_model", 0, pairs, expected_pair_count=4)
    rows = [
        {
            "encoded_pair_id": encoded_pair_id(pair.pair_id),
            "chosen_score": float(index + 1),
            "rejected_score": 0.0,
            "pair_loss": 0.1,
        }
        for index, pair in enumerate(pairs)
    ]
    plan = [
        {"rank": 0, "chunk": 0, "batch": 0, "microbatch": 0, "encoded_pair_ids": [r["encoded_pair_id"] for r in rows]}
    ]
    write_pair_artifacts(str(tmp_path), "reward_model", 0, rows, plan)
    record_probe_contract(str(tmp_path), "reward_model", 4, pairs, expected_pair_count=4)
    write_pair_artifacts(str(tmp_path), "reward_model", 4, rows, plan)

    pair_path = tmp_path / "preference_eval" / "reward_model-step-0000004-pairs.jsonl"
    written = [json.loads(line) for line in pair_path.read_text(encoding="utf-8").splitlines()]
    assert [row["pair_id"] for row in written] == [pair.pair_id for pair in pairs]
    assert set(written[0]) == {"pair_id", "chosen_score", "rejected_score", "pair_loss"}

    changed_plan = [{**plan[0], "encoded_pair_ids": list(reversed(plan[0]["encoded_pair_ids"]))}]
    with pytest.raises(RuntimeError, match="batch plan changed"):
        write_pair_artifacts(str(tmp_path), "reward_model", 4, rows, changed_plan)


def test_probe_contract_rejects_any_count_other_than_frozen_512(tmp_path):
    with pytest.raises(RuntimeError, match="exactly 512 probe pairs"):
        record_probe_contract(str(tmp_path), "reward_model", 0, [_pair("pair-0", 0)])


@pytest.mark.parametrize("save_path", [None, ""])
def test_probe_contract_rejects_nonfrozen_count_without_artifact_output(save_path):
    with pytest.raises(RuntimeError, match="exactly 512 probe pairs"):
        record_probe_contract(save_path, "reward_model", 0, [_pair("pair-0", 0)])


def test_probe_contract_accepts_frozen_count_without_artifact_output():
    pairs = [_pair(f"pair-{index}", index) for index in range(512)]
    assert record_probe_contract(None, "reward_model", 0, pairs) is None
