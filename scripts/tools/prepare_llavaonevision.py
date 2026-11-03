# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Prepare OneVision SFT data or sample valid rows from its local cache.

Run ``python -m scripts.tools.prepare_llavaonevision prepare --output-dir DIR``
to download a pinned snapshot and prepare resumable SFT shards with extracted
images. READY.json is published only after all source shards complete.

Run ``python -m scripts.tools.prepare_llavaonevision sample --data-dir DIR``
to select exactly --count valid rows from cached Parquet shards and embed images
as data URIs. Sampling is uniform over valid cached rows, not the full dataset.
"""

import argparse
import base64
import fcntl
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)
REPO_ID = "mvp-lab/LLaVA-OneVision-1.5-Instruct-Data"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def normalize_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    messages = []
    for turn in row["conversations"]:
        role = turn.get("role") or turn.get("from")
        role = {"human": "user", "gpt": "assistant"}.get(role, role)
        content = turn.get("content")
        if content is None:
            content = turn.get("value")
        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            raise ValueError("Invalid conversation role or content")
        messages.append({"role": role, "content": content})
    if not any(m["role"] == "assistant" and m["content"].strip() for m in messages):
        raise ValueError("No assistant response")
    return messages


def convert_shard(source: Path, relative: Path, root: Path) -> dict[str, Any]:
    import pyarrow.parquet as pq

    output = root / "sft/train" / relative.with_suffix(".jsonl")
    stats_path = root / "sft/stats" / relative.with_suffix(".json")
    if output.exists() and stats_path.exists():
        return json.loads(stats_path.read_text())
    output.parent.mkdir(parents=True, exist_ok=True)
    rejects = root / "sft/rejects" / relative.with_suffix(".jsonl")
    rejects.parent.mkdir(parents=True, exist_ok=True)
    image_dir = root / "sft/images" / relative.with_suffix("")
    temporary = output.with_suffix(".jsonl.tmp")
    stats = {"source": str(relative), "rows": 0, "accepted": 0, "rejected": 0, "images": 0}
    parquet = pq.ParquetFile(source)
    created_directories: set[Path] = set()
    with temporary.open("w") as target, rejects.open("w") as rejected:
        # Read one row group at a time: nested list structs in this repository
        # can fail with Arrow's iter_batches across chunk boundaries.
        for group in range(parquet.num_row_groups):
            for row in parquet.read_row_group(group).to_pylist():
                index = stats["rows"]
                stats["rows"] += 1
                try:
                    messages = normalize_messages(row)
                    media = row.get("image")
                    media = media if isinstance(media, list) else [media] if media else []
                    media = [m for m in media if m is not None]
                    markers = sum(m["content"].count("<image>") for m in messages)
                    if markers != len(media):
                        raise ValueError(f"Image markers {markers} != images {len(media)}")
                    payloads = []
                    for item in media:
                        payload = item.get("bytes") if isinstance(item, dict) else item
                        if not isinstance(payload, bytes) or not payload:
                            raise ValueError("Image has no embedded bytes; external paths are not portable")
                        payloads.append(payload)
                    paths = []
                    for image_index, payload in enumerate(payloads):
                        # Bounded directory sizes; raw image encoding is preserved.
                        directory = image_dir / f"{index // 1000:06d}"
                        if directory not in created_directories:
                            directory.mkdir(parents=True, exist_ok=True)
                            created_directories.add(directory)
                        path = directory / f"{index:09d}-{image_index}.image"
                        # The JSONL shard is published atomically only after all images finish.
                        # Interrupted shards are rewritten on resume, so per-image rename is unnecessary.
                        path.write_bytes(payload)
                        paths.append(str(path))
                    result = {
                        "messages": messages,
                        "images": paths,
                        "metadata": {
                            "id": row.get("id"),
                            "data_source": row.get("data_source"),
                            "source_shard": str(relative),
                            "source_row": index,
                        },
                    }
                    target.write(json.dumps(result, ensure_ascii=False) + "\n")
                    stats["accepted"] += 1
                    stats["images"] += len(paths)
                except (KeyError, TypeError, ValueError) as exc:
                    rejected.write(json.dumps({"row": index, "id": row.get("id"), "reason": str(exc)}) + "\n")
                    stats["rejected"] += 1
    if stats["rows"] != parquet.metadata.num_rows:
        raise RuntimeError(f"Row count mismatch for {relative}")
    temporary.replace(output)
    write_json(stats_path, stats)
    return stats


def prepare(root: Path, workers: int, revision: str | None, subsets: list[str] | None) -> None:
    from huggingface_hub import HfApi, hf_hub_download

    manifest_path = root / "source_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if revision and revision != manifest["revision"]:
            raise ValueError("Output directory already pins another revision")
    else:
        info = HfApi().dataset_info(REPO_ID, revision=revision, files_metadata=True)
        manifest = {
            "repo_id": REPO_ID,
            "revision": info.sha,
            "files": [{"path": f.rfilename, "size": f.size} for f in info.siblings],
        }
        write_json(manifest_path, manifest)
    files = [f for f in manifest["files"] if f["path"].endswith(".parquet")]
    if subsets:
        available = {Path(f["path"]).parts[0] for f in files}
        if set(subsets) - available:
            raise ValueError(f"Unknown subsets: {set(subsets) - available}")
        files = [f for f in files if Path(f["path"]).parts[0] in subsets]
    # Small shards first so schema and compatibility checks can finish early.
    files.sort(key=lambda f: f["size"])
    hf_hub_download(REPO_ID, "README.md", repo_type="dataset", revision=manifest["revision"], local_dir=root / "raw")
    logger.info(
        "Preparing %d shards, %.3f TB, revision %s",
        len(files),
        sum(f["size"] for f in files) / 1e12,
        manifest["revision"],
    )

    def download(item: dict[str, Any]) -> Path:
        for attempt in range(5):
            try:
                source = hf_hub_download(
                    REPO_ID, item["path"], repo_type="dataset", revision=manifest["revision"], local_dir=root / "raw"
                )
                if Path(source).stat().st_size != item["size"]:
                    raise RuntimeError(f"Source size mismatch: {item['path']}")
                logger.info("Downloaded %s (%d bytes)", item["path"], item["size"])
                return Path(source)
            except Exception:
                if attempt == 4:
                    raise
                logger.exception("Retry %d for %s", attempt + 1, item["path"])
                time.sleep(min(2**attempt, 16))
        raise AssertionError("Unreachable")

    totals = {"shards": 0, "rows": 0, "accepted": 0, "rejected": 0, "images": 0}
    failures = []
    # Keep network transfers moving while conversion waits on shared-filesystem I/O.
    # Futures hold paths only; downloaded image bytes are not queued in RAM.
    with (
        ThreadPoolExecutor(max_workers=workers) as downloader,
        ThreadPoolExecutor(max_workers=workers) as executor,
    ):
        downloads = {item["path"]: downloader.submit(download, item) for item in files}

        def process(item: dict[str, Any]) -> dict[str, Any]:
            source = downloads[item["path"]].result()
            return convert_shard(source, Path(item["path"]), root)

        futures = {executor.submit(process, item): item["path"] for item in files}
        for future in as_completed(futures):
            try:
                stats = future.result()
                totals["shards"] += 1
                for name in ("rows", "accepted", "rejected", "images"):
                    totals[name] += stats[name]
                logger.info(
                    "Completed %d/%d %s: accepted=%d rejected=%d",
                    totals["shards"],
                    len(files),
                    stats["source"],
                    stats["accepted"],
                    stats["rejected"],
                )
            except Exception as exc:
                logger.exception("Failed shard %s", futures[future])
                failures.append({"source": futures[future], "error": str(exc)})
            write_json(root / "progress.json", {**totals, "expected_shards": len(files), "failures": failures})
    if failures:
        raise RuntimeError(f"{len(failures)} shards failed; rerun the same command to resume")
    # Partial subset runs are useful for validation, but must not mark full data ready.
    name = "SUBSET_READY.json" if subsets else "READY.json"
    write_json(root / "sft" / name, {**totals, "revision": manifest["revision"], "subsets": subsets})
    logger.info("Preparation complete: %s", totals)


def normalize_row(row: dict[str, Any]) -> tuple[list[dict[str, str]], list[bytes]]:
    messages = normalize_messages(row)
    media = row.get("image")
    media = media if isinstance(media, list) else [media] if media else []
    media = [item for item in media if item is not None]
    if sum(message["content"].count("<image>") for message in messages) != len(media):
        raise ValueError("Image marker count does not match image count")
    payloads = [item.get("bytes") if isinstance(item, dict) else item for item in media]
    if any(not isinstance(payload, bytes) or not payload for payload in payloads):
        raise ValueError("Missing embedded image bytes")
    return messages, payloads


def scan_shard(path: Path) -> dict[str, Any]:
    import pyarrow.parquet as pq

    valid, rejected, index = [], 0, 0
    parquet = pq.ParquetFile(path)
    for group in range(parquet.num_row_groups):
        for row in parquet.read_row_group(group).to_pylist():
            try:
                normalize_row(row)
                valid.append(index)
            except (KeyError, TypeError, ValueError):
                rejected += 1
            index += 1
    return {"path": str(path), "valid": valid, "rejected": rejected}


def convert_selected(source: Path, indices: list[int], raw: Path, output: Path) -> dict[str, int]:
    import pyarrow.parquet as pq

    relative = source.relative_to(raw).with_suffix(".jsonl")
    destination = output / "train" / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    # A published shard belongs to the immutable selection.json plan. Validate
    # its count before reuse; only unfinished .tmp shards need rewriting.
    if destination.exists():
        rows = images = 0
        with destination.open() as existing:
            for line in existing:
                row = json.loads(line)
                rows += 1
                images += len(row["images"])
        if rows != len(indices):
            raise RuntimeError(f"Existing shard count mismatch: {destination}")
        return {"rows": rows, "images": images}
    partial = destination.with_suffix(".jsonl.tmp")
    selected = set(indices)
    offset, written, images = 0, 0, 0
    parquet = pq.ParquetFile(source)
    with partial.open("w", buffering=8 * 1024 * 1024) as target:
        for group in range(parquet.num_row_groups):
            count = parquet.metadata.row_group(group).num_rows
            if not any(offset <= index < offset + count for index in selected):
                offset += count
                continue
            for index, row in enumerate(parquet.read_row_group(group).to_pylist(), start=offset):
                if index not in selected:
                    continue
                messages, payloads = normalize_row(row)
                # Relax load_image accepts data:image/* URIs and detects encoding via PIL.
                media = ["data:image/octet-stream;base64," + base64.b64encode(p).decode("ascii") for p in payloads]
                result = {
                    "messages": messages,
                    "images": media,
                    "metadata": {
                        "id": row.get("id"),
                        "data_source": row.get("data_source"),
                        "source_shard": str(source.relative_to(raw)),
                        "source_row": index,
                    },
                }
                target.write(json.dumps(result, ensure_ascii=False) + "\n")
                written += 1
                images += len(media)
            offset += count
    if written != len(selected):
        raise RuntimeError(f"Selection count mismatch for {source}: {written} != {len(selected)}")
    partial.replace(destination)
    return {"rows": written, "images": images}


def sample(root: Path, count: int, seed: int, workers: int) -> Path:
    import numpy as np

    raw = root / "raw"
    output = root / f"sft-{count}"
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / "source_manifest.json").read_text())
    plan_path = output / "selection.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        if plan["seed"] != seed or plan["count"] != count or plan["revision"] != manifest["revision"]:
            raise ValueError("Existing selection has different parameters")
    else:
        files = [
            raw / f["path"]
            for f in manifest["files"]
            if f["path"].endswith(".parquet")
            and (raw / f["path"]).is_file()
            and (raw / f["path"]).stat().st_size == f["size"]
        ]
        scans = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(scan_shard, path) for path in files]
            for future in as_completed(futures):
                scans.append(future.result())
                if len(scans) % 20 == 0:
                    logger.info("Scanned %d/%d shards", len(scans), len(files))
        scans.sort(key=lambda item: item["path"])
        total = sum(len(item["valid"]) for item in scans)
        if total < count:
            raise ValueError(f"Cached data has only {total} valid rows; requested {count}")
        selected = np.sort(np.random.default_rng(seed).choice(total, size=count, replace=False))
        selection, offset = [], 0
        for item in scans:
            length = len(item["valid"])
            first, last = np.searchsorted(selected, [offset, offset + length])
            indices = [item["valid"][int(index) - offset] for index in selected[first:last]]
            if indices:
                selection.append({"path": item["path"], "indices": indices})
            offset += length
        plan = {
            "count": count,
            "seed": seed,
            "revision": manifest["revision"],
            "population_valid": total,
            "population_rejected": sum(item["rejected"] for item in scans),
            "population_shards": len(scans),
            "selection": selection,
        }
        write_json(plan_path, plan)
    totals = {"rows": 0, "images": 0, "shards": 0}
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as executor:
        futures = [
            executor.submit(convert_selected, Path(item["path"]), item["indices"], raw, output)
            for item in plan["selection"]
        ]
        for future in as_completed(futures):
            stats = future.result()
            totals["rows"] += stats["rows"]
            totals["images"] += stats["images"]
            totals["shards"] += 1
            write_json(output / "progress.json", {**totals, "target_rows": count})
            logger.info("Prepared %d/%d rows (%d shards)", totals["rows"], count, totals["shards"])
    if totals["rows"] != count:
        raise RuntimeError(f"Final count mismatch: {totals['rows']} != {count}")
    write_json(
        output / "READY.json",
        {
            **totals,
            **{k: v for k, v in plan.items() if k != "selection"},
            "sampling_scope": "locally cached shards",
            "image_storage": "inline data URI",
        },
    )
    logger.info("Ready: %s (%d rows)", output, count)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare", help="Download and convert OneVision shards")
    prepare_parser.add_argument("--output-dir", type=Path, required=True)
    prepare_parser.add_argument("--workers", type=int, default=8)
    prepare_parser.add_argument("--revision")
    prepare_parser.add_argument("--subsets", nargs="+", help="Optional exact source directory names for a smoke run")
    sample_parser = commands.add_parser("sample", help="Sample valid rows from locally cached shards")
    sample_parser.add_argument("--data-dir", type=Path, required=True)
    sample_parser.add_argument("--count", type=int, default=1_000_000)
    sample_parser.add_argument("--seed", type=int, default=1234)
    sample_parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.command == "sample" and args.count < 1:
        parser.error("--count must be positive")
    root = (args.output_dir if args.command == "prepare" else args.data_dir).resolve()
    if args.command == "prepare":
        root.mkdir(parents=True, exist_ok=True)
    with (root / ".prepare.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.command == "prepare":
            prepare(root, args.workers, args.revision, args.subsets)
        else:
            sample(root, args.count, args.seed, args.workers)


if __name__ == "__main__":
    main()
