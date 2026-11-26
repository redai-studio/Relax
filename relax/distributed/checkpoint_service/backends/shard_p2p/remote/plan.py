# Copyright (c) 2026 Relax Authors. All Rights Reserved.
from collections import defaultdict

from ..plan import fingerprint


PACKED_TARGET_ENCODING = "packed_rectangles_v1"


def packed_target_regions(writes: list[dict], sizes: dict[str, int]) -> dict[str, dict]:
    """Pack a validated post-replica plan without changing logical write IDs.

    Source rank zero and the receiver independently derive the same wire
    offsets. The full logical plan remains responsible for disjoint coverage.
    """
    result = {name: {} for name in sizes}
    cursors = dict.fromkeys(sizes, 0)
    seen = set()
    for write in sorted(writes, key=lambda item: (item["param"], item["id"])):
        name, identity = write["param"], write["id"]
        fields = {key: write[key] for key in ("dst_offset", "dst_stride", "rows", "row_bytes")}
        if (
            name not in sizes
            or not isinstance(identity, str)
            or not identity
            or identity in seen
            or any(type(value) is not int for value in fields.values())
            or any(fields[key] % 2 for key in ("dst_offset", "dst_stride", "row_bytes"))
            or fields["rows"] <= 0
            or fields["row_bytes"] <= 0
            or fields["dst_offset"] < 0
            or fields["dst_stride"] < fields["row_bytes"]
            or fields["dst_offset"] + (fields["rows"] - 1) * fields["dst_stride"] + fields["row_bytes"] > sizes[name]
        ):
            raise ValueError("Packed remote rectangle exceeds its logical target")
        size = fields["rows"] * fields["row_bytes"]
        if cursors[name] + size > sizes[name]:
            raise ValueError("Packed remote rectangles exceed target capacity")
        result[name][identity] = dict(offset=cursors[name], **fields)
        cursors[name] += size
        seen.add(identity)
    if any(not regions for regions in result.values()):
        raise ValueError("Packed remote target inventory is incomplete")
    return result


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
