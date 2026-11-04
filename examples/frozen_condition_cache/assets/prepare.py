#!/usr/bin/env python3
"""Download pinned real assets, verify every byte, extract real edit pairs.

Run with the kit venv, before the paid GPU campaign. No large weights are
bundled in this kit. Re-running resumes incomplete HTTP downloads and checks
existing complete files; no Hugging Face credentials are required.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".partial")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def identity(files):
    mapping = {f["path"]: {"sha256": f["sha256"], "size": f["size"]} for f in files}
    return hashlib.sha256(json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def download(file, root):
    destination = root / file["path"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size == file["size"]:
        if sha256_file(destination) == file["sha256"]:
            print(f"VERIFIED {file['path']}", flush=True)
            return
        raise ValueError(f"Existing complete file has wrong SHA-256: {destination}")
    partial = destination.with_name(destination.name + ".partial")
    if partial.exists() and partial.stat().st_size > file["size"]:
        raise ValueError(f"Oversize partial file: {partial}")
    for attempt in range(4):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset < file["size"]:
                headers = {"User-Agent": "Relax17-reproducible-assets/1"}
                if offset:
                    headers["Range"] = f"bytes={offset}-"
                request = urllib.request.Request(file["url"], headers=headers)
                with urllib.request.urlopen(request, timeout=120) as response:
                    if offset and response.status == 206:
                        content_range = response.headers.get("Content-Range", "")
                        if not content_range.startswith(f"bytes {offset}-"):
                            raise ValueError("Incorrect resume Content-Range")
                        mode = "ab"
                    elif response.status == 200:
                        offset, mode = 0, "wb"
                    else:
                        raise ValueError(f"Unexpected download status {response.status}")
                    with partial.open(mode) as output:
                        for block in iter(lambda: response.read(4 * 1024 * 1024), b""):
                            offset += len(block)
                            if offset > file["size"]:
                                raise ValueError(f"Download exceeds locked size: {file['path']}")
                            output.write(block)
            if partial.stat().st_size != file["size"]:
                raise IOError("Truncated download")
            if sha256_file(partial) != file["sha256"]:
                # Preserve mismatched bytes for diagnosis; never label them ready.
                raise ValueError(f"SHA-256 mismatch: {partial}")
            partial.replace(destination)
            print(f"DOWNLOADED+VERIFIED {file['path']}", flush=True)
            return
        except (OSError, urllib.error.URLError) as error:
            if attempt == 3:
                raise
            print(f"RETRY {file['path']}: {type(error).__name__}", flush=True)
            time.sleep(min(2**attempt, 8))


def prepare_dataset(spec, source_root, data_root):
    import pyarrow.parquet as parquet
    from PIL import Image

    ready = data_root / "data-ready.json"
    expected_identity = identity(spec["files"])
    if ready.is_file():
        result = json.loads(ready.read_text())
        if result.get("source_identity") != expected_identity:
            raise ValueError("Existing data export belongs to another source lock")
        for subset, count in [("high128", 128), ("low1024", 1024), ("debug8", 8)]:
            rows = json.loads((data_root / f"{subset}.json").read_text())
            if len(rows) != count or sha256_file(data_root / f"{subset}.json") != result["metadata_sha256"][subset]:
                raise ValueError("Existing data metadata changed")
        # A ready data tree must still contain every byte identified by its receipt.
        for image in result["images"]:
            path = data_root / image["path"]
            if not path.is_file() or path.stat().st_size != image["size"] or sha256_file(path) != image["sha256"]:
                raise ValueError(f"Existing extracted image changed: {path}")
        return result
    candidates = []
    shard_rows = []
    for shard in spec["selection"]["train_shards"]:
        file = parquet.ParquetFile(source_root / shard)
        count = file.metadata.num_rows
        shard_rows.append({"path": shard, "num_rows": count})
        for row in range(count):
            rank = hashlib.sha256(f"magicbrush|1701|{shard}|{row}".encode()).hexdigest()
            candidates.append((rank, shard, row))
    if len(candidates) != 1384:
        raise ValueError(f"Pinned eight-shard row count changed: {len(candidates)}")
    selection = sorted(candidates)[:1160]
    by_shard = {}
    for index, (_, shard, row) in enumerate(selection):
        subset = "high128" if index < 128 else "low1024" if index < 1152 else "debug8"
        by_shard.setdefault(shard, {})[row] = (index, subset)
    records = {}
    image_manifest = []
    provenance = []
    (data_root / "images").mkdir(parents=True, exist_ok=True)
    for shard, selected in sorted(by_shard.items()):
        row_offset = 0
        # Reading small Arrow record batches keeps extraction bounded; images are
        # the exact public encoded bytes, without resize/re-encode or mask use.
        file = parquet.ParquetFile(source_root / shard)
        for batch in file.iter_batches(
            batch_size=8, columns=["img_id", "turn_index", "instruction", "source_img", "target_img"]
        ):
            for local, row in enumerate(batch.to_pylist()):
                row_number = row_offset + local
                if row_number not in selected:
                    continue
                index, subset = selected[row_number]
                record = {"prompt": row["instruction"]}
                trace = {
                    "selection_index": index,
                    "subset": subset,
                    "shard": shard,
                    "row": row_number,
                    "img_id": row["img_id"],
                    "turn_index": row["turn_index"],
                    "instruction": row["instruction"],
                }
                if not isinstance(record["prompt"], str) or not record["prompt"].strip():
                    raise ValueError("Blank/non-string instruction in fixed data selection")
                for column, target_key in [("source_img", "edit_image"), ("target_img", "image")]:
                    value = row[column]
                    content = value.get("bytes") if isinstance(value, dict) else None
                    if not isinstance(content, bytes) or not content:
                        raise ValueError(
                            f"Image bytes absent in {shard}:{row_number}:{column}; no unpinned URL fallback"
                        )
                    digest = hashlib.sha256(content).hexdigest()
                    with Image.open(io.BytesIO(content)) as image:
                        width, height, mode, encoding = *image.size, image.mode, image.format
                    extension = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}.get(encoding, ".image")
                    relative = f"images/{index:04d}-{column}-{digest[:16]}{extension}"
                    path = data_root / relative
                    temporary = path.with_name(path.name + ".partial")
                    temporary.write_bytes(content)
                    temporary.replace(path)
                    record[target_key] = [relative] if target_key == "edit_image" else relative
                    entry = {
                        "path": relative,
                        "size": len(content),
                        "sha256": digest,
                        "width": width,
                        "height": height,
                        "mode": mode,
                        "encoding": encoding,
                        "shard": shard,
                        "row": row_number,
                        "column": column,
                    }
                    image_manifest.append(entry)
                    trace[column] = relative
                records[index] = record
                provenance.append(trace)
            row_offset += batch.num_rows
        print(f"EXTRACTED {shard}", flush=True)
    if sorted(records) != list(range(1160)):
        raise ValueError("Incomplete image extraction; prepared.json will not be created")
    hashes = {}
    for subset, (start, end) in [("high128", (0, 128)), ("low1024", (128, 1152)), ("debug8", (1152, 1160))]:
        path = data_root / f"{subset}.json"
        atomic_json(path, [records[index] for index in range(start, end)])
        hashes[subset] = sha256_file(path)
    result = {
        "schema_version": 1,
        "state": "READY",
        "source_repo": spec["repo"],
        "source_revision": spec["revision"],
        "source_identity": expected_identity,
        "license": spec["license"],
        "source_shards": shard_rows,
        "subset_sizes": {"high128": 128, "low1024": 1024, "debug8": 8},
        "subsets_disjoint": True,
        "row_ranking": spec["selection"]["row_ranking"],
        "metadata_sha256": hashes,
        "column_mapping": {"instruction": "prompt", "source_img": "edit_image[0]", "target_img": "image"},
        "image_bytes_reencoded": False,
        "images": sorted(image_manifest, key=lambda item: item["path"]),
        "rows": sorted(provenance, key=lambda item: item["selection_index"]),
    }
    atomic_json(ready, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 9))
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock_path = ROOT / "manifest.json"
    lock = json.loads(lock_path.read_text())
    # Disk estimate includes retained public Parquet, both actual models, and
    # image export. Refuse to start large downloads on an obviously small disk.
    import shutil

    required = sum(model["total_bytes"] for model in lock["models"].values()) + lock["data"]["total_bytes"]
    existing = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    if shutil.disk_usage(output).free < max(0, required - existing) + 12 * 1024**3:
        raise RuntimeError("Insufficient asset disk space; budget at least 110GB plus experiment evidence/cache")
    atomic_json(output / "asset-lock.json", lock)
    jobs = []
    for name, model in lock["models"].items():
        root = output / "models" / name
        jobs.extend((file, root) for file in model["files"])
    source_root = output / "sources" / "magicbrush"
    jobs.extend((file, source_root) for file in lock["data"]["files"])
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(download, file, root) for file, root in jobs]
        for future in futures:
            future.result()
    model_results = {}
    for name, model in lock["models"].items():
        model_root = output / "models" / name
        receipt = {
            "repo": model["repo"],
            "revision": model["revision"],
            "identity": identity(model["files"]),
            "files": model["files"],
            "verified_all_files": True,
        }
        atomic_json(model_root / "asset-receipt.json", receipt)
        model_results[name] = {
            "path": str(model_root),
            "identity": receipt["identity"],
            "revision": model["revision"],
            "receipt": str(model_root / "asset-receipt.json"),
        }
    data_root = output / "data"
    data_root.mkdir(exist_ok=True)
    data = prepare_dataset(lock["data"], source_root, data_root)
    result = {
        "schema_version": 1,
        "state": "READY",
        "models": model_results,
        "data": {
            "root": str(data_root),
            "high128": str(data_root / "high128.json"),
            "low1024": str(data_root / "low1024.json"),
            "debug8": str(data_root / "debug8.json"),
            "identity": data["source_identity"],
            "receipt": str(data_root / "data-ready.json"),
        },
        "input_manifest_sha256": sha256_file(lock_path),
        "all_assets_downloaded_and_verified": True,
    }
    atomic_json(output / "prepared.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
