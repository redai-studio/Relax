# Copyright (c) 2026 Relax Authors. All Rights Reserved.


def index_receipts(values: list[dict]) -> dict[int, dict]:
    """Preserve originals and reject duplicate owners before indexing."""
    if not isinstance(values, list) or any(
        not isinstance(value, dict) or type(value.get("source_rank")) is not int for value in values
    ):
        raise RuntimeError("Malformed RPC source receipts")
    receipts = {value["source_rank"]: value for value in values}
    if len(receipts) != len(values):
        raise RuntimeError("Duplicate RPC source receipt")
    return receipts
