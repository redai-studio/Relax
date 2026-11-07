# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Prepare pinned text-replay inputs on Kaggle, without model weights.

By default only tokenizer files are fetched; an existing private Dataset can
provide the corpus. --fetch-corpus additionally obtains the one pinned public
SWE-Gym parquet. All bytes stay under /tmp and are excluded from result
outputs. Its transfer-budget checks need an outer subprocess deadline for a
strict wall limit, because urllib timeouts apply to individual blocking network
operations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any


TOKENIZER_ID = "Qwen/Qwen3-0.6B"
# Public reproducibility identifiers: https://huggingface.co/Qwen/Qwen3-0.6B.
# Scoped annotations identify reviewed public hashes rather than credentials.
TOKENIZER_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"  # gitleaks:allow
TOKENIZER_FILES = {
    "config.json": "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd",
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",  # gitleaks:allow
    "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",  # gitleaks:allow
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
}
SWE_SOURCE: dict[str, Any] = {
    "repo": "SWE-Gym/OpenHands-SFT-Trajectories",
    "revision": "4aaa5a4a4b5861f4799d2336908760c190ac3b17",
    "path": "data/train.success.oss-00000-of-00001.parquet",
    "file": "swegym-openhands.parquet",
    "url": (
        "https://huggingface.co/datasets/SWE-Gym/OpenHands-SFT-Trajectories/"
        "resolve/4aaa5a4a4b5861f4799d2336908760c190ac3b17/"
        "data/train.success.oss-00000-of-00001.parquet"
    ),
    "bytes": 10379409,
    "sha256": "ea4bf37de020e165c5210bedddeef523d8834a89a35a8c65fec24f76f0eae4f1",
    "published_sha256": "ea4bf37de020e165c5210bedddeef523d8834a89a35a8c65fec24f76f0eae4f1",
    "rows": 491,
}


def file_identity(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while block := stream.read(131072):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def fetch(url: str, target: Path, expected: str, deadline: float, size: int | None = None) -> dict[str, Any]:
    if target.is_symlink():
        raise ValueError("Refuse a symlinked input file")
    if target.exists():
        actual_size = target.stat().st_size
        if actual_size > 64 * 1024 * 1024 or (size is not None and actual_size != size):
            raise ValueError("Existing input has an unexpected size")
        actual, actual_size = file_identity(target)
        if actual != expected or (size is not None and actual_size != size):
            raise ValueError("Existing input differs from the pinned bytes; use a fresh cloud scratch root")
        return {"file": target.name, "sha256": actual, "bytes": actual_size, "reused": True}
    temporary = target.with_suffix(target.suffix + ".download")
    if temporary.exists():
        raise ValueError("Stale partial input exists; use a fresh cloud scratch root")
    request = urllib.request.Request(url, headers={"User-Agent": "Relax-Task14-pinned-replay"})
    count = 0
    try:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Pinned input transfer budget exhausted before request")
        with urllib.request.urlopen(request, timeout=min(60, remaining)) as response, temporary.open("xb") as stream:
            while True:
                if time.monotonic() > deadline:
                    raise TimeoutError("Pinned input transfer budget exhausted")
                block = response.read(131072)
                if time.monotonic() > deadline:
                    raise TimeoutError("Pinned input transfer budget exhausted during read")
                if not block:
                    break
                count += len(block)
                if count > 64 * 1024 * 1024:
                    raise ValueError("Unexpected input size; no model weights are allowed")
                stream.write(block)
        actual, actual_size = file_identity(temporary)
        if time.monotonic() > deadline:
            raise TimeoutError("Pinned input transfer budget exhausted before publication")
        if actual != expected or (size is not None and actual_size != size):
            raise ValueError("Downloaded input does not match its pinned SHA256/size")
        temporary.replace(target)
        return {"file": target.name, "sha256": actual, "bytes": actual_size, "reused": False}
    finally:
        temporary.unlink(missing_ok=True)


def save(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.is_symlink() or temporary.exists() or temporary.is_symlink():
        raise ValueError("Refuse a symlinked metadata target or stale partial metadata")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    if sys.platform != "linux" or not Path("/kaggle/working").is_dir():
        raise RuntimeError("Run this input preparation only inside Kaggle; never on the user's computer")
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/tmp/relax-task14-replay-inputs"))
    parser.add_argument("--fetch-corpus", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if root == Path("/tmp") or not root.is_relative_to(Path("/tmp")):
        raise ValueError("Input scratch must be a private child directory of /tmp, outside notebook outputs")
    root.mkdir(parents=True, exist_ok=True)
    tokenizer = root / "tokenizer"
    if tokenizer.is_symlink():
        raise ValueError("Tokenizer scratch must not be a symlink")
    tokenizer.mkdir(exist_ok=True)
    expected_names = set(TOKENIZER_FILES) | {"tokenizer-source.json"}
    if any(path.name not in expected_names for path in tokenizer.iterdir()):
        raise ValueError("Tokenizer scratch contains files outside the finite allowlist")
    deadline = time.monotonic() + 300
    receipts = []
    for name, digest in TOKENIZER_FILES.items():
        url = f"https://huggingface.co/{TOKENIZER_ID}/resolve/{TOKENIZER_REVISION}/{name}"
        receipts.append(fetch(url, tokenizer / name, digest, deadline))
    save(
        tokenizer / "tokenizer-source.json",
        {
            "id": TOKENIZER_ID,
            "revision": TOKENIZER_REVISION,
            "files": TOKENIZER_FILES,
        },
    )
    corpus = None
    if args.fetch_corpus:
        corpus = root / "corpus"
        if corpus.is_symlink():
            raise ValueError("Corpus scratch must not be a symlink")
        corpus.mkdir(exist_ok=True)
        receipts.append(
            fetch(SWE_SOURCE["url"], corpus / SWE_SOURCE["file"], SWE_SOURCE["sha256"], deadline, SWE_SOURCE["bytes"])
        )
        save(
            corpus / "provenance.json",
            {
                "files": [SWE_SOURCE],
                "purpose": "Pinned text-only Task14 replay",
                "licenses": {SWE_SOURCE["repo"]: "MIT"},
                "original_token_ids_or_logprobs_verified": False,
                "tool_commands_executed": False,
            },
        )
    result = {
        "tokenizer": str(tokenizer),
        "corpus": str(corpus) if corpus else None,
        "receipts": receipts,
        "tokenizer_license": "Apache-2.0",
        "model_weights_downloaded": False,
        "corpus_fetched": args.fetch_corpus,
        "output_data_outside_notebook_results": True,
        "interpretation": "Reconstructed tokenizer IDs; synthetic logprobs remain declared by the replay benchmark",
    }
    save(root / "preparation.json", result)
    sys.stdout.write(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
