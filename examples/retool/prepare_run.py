# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Download the pinned ReTool training inputs and prepare native datasets."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


MODEL = ("Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28")
AUTHOR = ("JoeYing/ReTool-SFT", "74943ce52f389f16926702302fcff6255875cbb2")
DAPO = ("BytedTsinghua-SIA/DAPO-Math-17k", "65877096c24ffa7abc4e4fa5edb95cf3413a5674")


def prepare(root: Path) -> None:
    download_env = os.environ.copy()
    if proxy := download_env.pop("RETOOL_DOWNLOAD_PROXY", None):
        download_env.update(http_proxy=proxy, https_proxy=proxy, HTTP_PROXY=proxy, HTTPS_PROXY=proxy)

    def download(source: tuple[str, str], kind: str, target: Path, files: list[str]) -> None:
        subprocess.run(
            [sys.executable, "-m", __spec__.name, "--download", *source, kind, str(target), *files],
            env=download_env,
            check=True,
        )

    def build(module: str, *args: str) -> None:
        subprocess.run([sys.executable, "-m", f"examples.retool.{module}", *args], check=True)

    model = root / "model"
    data = root / "data"
    author_file = "train_2000.parquet"
    dapo_file = "data/dapo-math-17k.parquet"
    download(MODEL, "model", model, ["*.json", "*.safetensors", "*.txt"])
    download(AUTHOR, "dataset", data / "author", [author_file])
    download(DAPO, "dataset", data / "dapo", [dapo_file])
    build(
        "prepare_sft",
        "--parquet",
        str(data / "author" / author_file),
        "--model",
        str(model),
        "--out",
        str(data / "sft"),
    )
    build(
        "prepare_data",
        "train",
        "--input",
        str(data / "dapo" / dapo_file),
        "--model",
        str(model),
        "--out",
        str(data / "train.jsonl"),
    )
    with (root / "inputs.json").open("x") as stream:
        json.dump(dict(model=MODEL, sft=AUTHOR, rl=DAPO), stream, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?")
    parser.add_argument("--download", nargs="+")
    args = parser.parse_args()
    if args.download:
        from huggingface_hub import snapshot_download

        repo, revision, kind, target, *patterns = args.download
        snapshot_download(repo_id=repo, revision=revision, repo_type=kind, local_dir=target, allow_patterns=patterns)
    elif args.root:
        prepare(args.root.resolve())
    else:
        parser.error("Specify the output root")


if __name__ == "__main__":
    main()
