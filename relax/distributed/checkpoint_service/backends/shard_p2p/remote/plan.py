# Copyright (c) 2026 Relax Authors. All Rights Reserved.
from collections import defaultdict

from ..plan import fingerprint


class ReceiptValidator:
    """Pin a validated static plan, while checking every version's real
    receipts."""

    def __init__(self, plan: dict) -> None:
        expected = defaultdict(set)
        for write in plan["writes"]:
            expected[write["source_rank"]].add(write["id"])
        self.expected: dict[int, frozenset[str]] = {rank: frozenset(ids) for rank, ids in expected.items()}
        self.plan_hash: str = fingerprint(plan)

    def validate(self, receipts: dict[int, dict], generation: str, version: int) -> None:
        if set(receipts) != set(self.expected):
            raise RuntimeError("Remote shard update is missing a source receipt")
        for rank, ids in self.expected.items():
            receipt = receipts[rank]
            completed = receipt["writes"]
            if (
                receipt["source_rank"] != rank
                or receipt["version"] != version
                or receipt["generation"] != generation
                or receipt["plan_hash"] != self.plan_hash
                or set(completed) != ids
                or len(completed) != len(ids)
            ):
                raise RuntimeError("Remote shard update has a stale or incomplete source receipt")
