# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Download pinned public data/config/tokenizers; model weights are
excluded."""

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download


MODEL_REVISIONS = {
    "Qwen3.5-27B": "fc05daec18b0a78c049392ed2e771dde82bdf654",
    "Qwen3.5-35B-A3B": "59d61f3ce65a6d9863b86d2e96597125219dc754",
}
DATA_REVISION = "5ea611a9fde449e7df8e72aefe698996a9882390"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for model, revision in MODEL_REVISIONS.items():
        for filename in (
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
            "vocab.json",
            "merges.txt",
            "model.safetensors.index.json",
        ):
            hf_hub_download(f"Qwen/{model}", filename, revision=revision, local_dir=args.output / model)
    hf_hub_download(
        "unsloth/OpenMathReasoning-mini",
        "data/cot-00000-of-00001.parquet",
        repo_type="dataset",
        revision=DATA_REVISION,
        local_dir=args.output / "OpenMathReasoning-mini",
    )
    (args.output / "revisions.json").write_text(
        json.dumps({"models": MODEL_REVISIONS, "data": DATA_REVISION}, indent=2) + "\n"
    )
    print(f"Pinned data/config/tokenizers downloaded to {args.output}; no model weights downloaded")


if __name__ == "__main__":
    main()
