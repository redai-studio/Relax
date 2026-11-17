# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Download pinned public data/config/tokenizers; model weights are
excluded."""

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files


MODEL_REVISIONS = {
    "Qwen3.5-27B": "fc05daec18b0a78c049392ed2e771dde82bdf654",
    "Qwen3.5-35B-A3B": "59d61f3ce65a6d9863b86d2e96597125219dc754",
}
DATA_REVISION = "5ea611a9fde449e7df8e72aefe698996a9882390"
FOUR_GPU_MODELS = {
    "Qwen3.5-4B": ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"),
    "OLMoE-1B-7B-0924": ("allenai/OLMoE-1B-7B-0924", "6d84c48581ece794365f2b8e9cfb043c68ade9c5"),
}
OLMOE_SFT_TEMPLATE = (
    "{{ bos_token if bos_token is not none else '' }}"
    "{% for message in messages %}"
    "{% if message['role'] == 'assistant' %}"
    "{% generation %}{{ message['content'] }}{{ eos_token }}{% endgeneration %}"
    "{% else %}{{ message['content'] }}{{ '\\n\\n' }}{% endif %}"
    "{% endfor %}"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=("eight", "four"), default="eight")
    args = parser.parse_args()
    models = (
        FOUR_GPU_MODELS
        if args.profile == "four"
        else {model: (f"Qwen/{model}", revision) for model, revision in MODEL_REVISIONS.items()}
    )
    for model, (repository, revision) in models.items():
        available = set(list_repo_files(repository, revision=revision))
        for filename in (
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
            "vocab.json",
            "merges.txt",
            "model.safetensors.index.json",
            "special_tokens_map.json",
        ):
            if filename in available:
                hf_hub_download(repository, filename, revision=revision, local_dir=args.output / model)
        if model == "OLMoE-1B-7B-0924":
            template = args.output / model / "chat_template.jinja"
            if template.exists() and template.read_text() != OLMOE_SFT_TEMPLATE:
                raise ValueError("Refusing to overwrite a different local OLMoE SFT template")
            template.write_text(OLMOE_SFT_TEMPLATE)
            print("Installed the explicit local OLMoE prompt/answer SFT template; workload manifests hash this file")
    hf_hub_download(
        "unsloth/OpenMathReasoning-mini",
        "data/cot-00000-of-00001.parquet",
        repo_type="dataset",
        revision=DATA_REVISION,
        local_dir=args.output / "OpenMathReasoning-mini",
    )
    (args.output / "revisions.json").write_text(
        json.dumps(
            {
                "models": {name: revision for name, (_, revision) in models.items()},
                "data": DATA_REVISION,
                "profile": args.profile,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Pinned data/config/tokenizers downloaded to {args.output}; no model weights downloaded")


if __name__ == "__main__":
    main()
