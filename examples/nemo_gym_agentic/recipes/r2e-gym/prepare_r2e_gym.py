# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Prepare R2E-Gym recipe rows for the NeMo Gym SWE agent."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any


DEFAULT_DATASET = "R2E-Gym/R2E-Gym-Lite"
DEFAULT_SPLIT = "train"
DOCKER_IMAGE_PATTERN = re.compile(r"^(?P<owner>[^/]+)/(?P<repo>[^/:]+)_final:(?P<commit>[0-9a-fA-F]+)$")


def instance_id_from_docker_image(docker_image: str) -> str:
    match = DOCKER_IMAGE_PATTERN.fullmatch(docker_image)
    if match is None:
        raise ValueError(f"R2E-Gym docker_image must look like '<owner>/<repo>_final:<commit>', got {docker_image!r}")
    return f"{match.group('owner')}__{match.group('repo')}-{match.group('commit')}"


def sif_name_from_docker_image(docker_image: str) -> str:
    match = DOCKER_IMAGE_PATTERN.fullmatch(docker_image)
    if match is None:
        instance_id_from_docker_image(docker_image)
        raise AssertionError("unreachable")
    return f"{match.group('repo').lower()}_final_{match.group('commit')}.sif"


def convert_r2e_row(
    row: dict[str, Any],
    *,
    dataset_name: str = DEFAULT_DATASET,
    split: str = DEFAULT_SPLIT,
    agent_name: str = "swe_agents",
) -> dict[str, Any]:
    required = ("repo_name", "docker_image", "commit_hash", "problem_statement", "parsed_commit_content")
    missing = [key for key in required if not isinstance(row.get(key), str) or not row[key]]
    if missing:
        raise ValueError(f"R2E-Gym row has missing or invalid fields: {', '.join(missing)}")

    docker_image = row["docker_image"]
    instance_id = instance_id_from_docker_image(docker_image)
    image_commit = docker_image.rsplit(":", maxsplit=1)[1]
    if image_commit != row["commit_hash"]:
        raise ValueError(f"docker_image commit {image_commit!r} does not match commit_hash {row['commit_hash']!r}")

    try:
        parsed_commit = json.loads(row["parsed_commit_content"])
    except json.JSONDecodeError as exc:
        raise ValueError("R2E-Gym parsed_commit_content must contain valid JSON") from exc
    if not isinstance(parsed_commit, dict):
        raise ValueError("R2E-Gym parsed_commit_content must contain a JSON object")
    base_commit = parsed_commit.get("old_commit_hash") or f"{row['commit_hash']}^"
    if not isinstance(base_commit, str):
        raise ValueError("R2E-Gym parsed_commit_content.old_commit_hash must be a string")

    instance_dict = dict(row)
    instance_dict.update(
        {
            "instance_id": instance_id,
            "repo": row["repo_name"],
            "version": row["commit_hash"],
            "base_commit": base_commit,
        }
    )
    metadata = {
        "instance_id": instance_id,
        "base_commit": base_commit,
        "dataset_name": dataset_name,
        "split": split,
        "problem_statement": row["problem_statement"],
        "instance_dict": json.dumps(instance_dict, ensure_ascii=False, separators=(",", ":")),
    }
    return {
        "responses_create_params": {
            "input": [],
            "metadata": metadata,
            "model": "model",
            "temperature": 0.7,
            "top_p": 0.8,
            "max_output_tokens": 32768,
        },
        "agent_ref": {"type": "responses_api_agents", "name": agent_name},
        "instance_id": instance_id,
        "repo_name": row.get("repo_name"),
        "docker_image": docker_image,
        "commit_hash": row["commit_hash"],
        "problem_statement": row["problem_statement"],
    }


def prepare_rows(
    rows: Iterable[dict[str, Any]],
    output_path: Path,
    manifest_path: Path,
    *,
    limit: int,
    dataset_name: str,
    split: str,
    agent_name: str,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as output, manifest_path.open("w", encoding="utf-8") as manifest:
        for row in rows:
            converted = convert_r2e_row(
                row,
                dataset_name=dataset_name,
                split=split,
                agent_name=agent_name,
            )
            output.write(json.dumps(converted, ensure_ascii=False, separators=(",", ":")) + "\n")
            manifest.write(
                json.dumps(
                    {
                        "instance_id": converted["instance_id"],
                        "docker_image": converted["docker_image"],
                        "sif_name": sif_name_from_docker_image(converted["docker_image"]),
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
            count += 1
            if count >= limit:
                break
    if count == 0:
        raise ValueError("The selected R2E-Gym split did not produce any rows")
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--agent-name", default="swe_agents")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit < 1:
        raise ValueError("--limit must be greater than zero")

    from datasets import load_dataset

    rows = load_dataset(args.dataset, split=args.split, streaming=True)
    count = prepare_rows(
        rows,
        args.output,
        args.manifest,
        limit=args.limit,
        dataset_name=args.dataset,
        split=args.split,
        agent_name=args.agent_name,
    )
    print(f"Prepared {count} R2E-Gym row(s) in {args.output}")
    print(f"Wrote Apptainer image manifest to {args.manifest}")


if __name__ == "__main__":
    main()
