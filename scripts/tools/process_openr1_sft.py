# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Convert the multimodal-open-r1 RL parquet into the SFT messages layout.

process_openr1.py emits the GRPO layout (prompt / image<binary> / label). The
SFT loader needs the assistant turn *inside* the messages list, because
--label-key only accepts a plain-string prompt column
(relax/engine/sft/dataset/streaming.py:279-283) while this dataset's prompt is
a list of chat turns. So the label is appended as the assistant turn here and
--label-key is left unset at launch.

Images are written out as files and referenced by path rather than embedded,
which keeps the parquet small enough to stream and lets the processor pool mmap
them. Row i becomes images/<i:07d>.<ext>, ext taken from the decoded format.
"""

import argparse
import io
from pathlib import Path

import pandas as pd
from PIL import Image


def _normalize_think(label: str) -> str:
    """Put the <think> body on its own lines, matching the R1 response
    style."""
    return label.replace("<think>", "<think>\n", 1).replace("</think>", "\n</think>", 1)


def convert_row(row, index: int, image_dir: Path, path_prefix: str):
    images = []
    for offset, blob in enumerate(row["image"]):
        payload = blob["bytes"] if isinstance(blob, dict) else blob
        with Image.open(io.BytesIO(payload)) as img:
            ext = (img.format or "JPEG").lower()
            ext = "jpg" if ext == "jpeg" else ext
            name = f"{index:07d}.{ext}" if offset == 0 else f"{index:07d}_{offset}.{ext}"
            (image_dir / name).write_bytes(payload)
        images.append(f"{path_prefix}{name}")

    messages = [{"role": turn["role"], "content": turn["content"]} for turn in row["prompt"]]
    messages.append({"role": "assistant", "content": _normalize_think(row["label"])})
    return {"messages": messages, "images": images}


def convert_dataset(input_file: str, output_file: str, image_dir: str, absolute: bool) -> None:
    image_path = Path(image_dir)
    image_path.mkdir(parents=True, exist_ok=True)
    prefix = f"{image_path.resolve()}/" if absolute else f"{image_path.name}/"

    df = pd.read_parquet(input_file)
    converted = [convert_row(row, i, image_path, prefix) for i, (_, row) in enumerate(df.iterrows())]
    pd.DataFrame(converted).to_parquet(output_file, index=False)
    print(len(df), len(converted), image_path.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert the open-r1 RL parquet to the SFT messages layout")
    parser.add_argument("--input-file", required=True, help="parquet produced by process_openr1.py")
    parser.add_argument("--output-file", required=True, help="destination SFT parquet")
    parser.add_argument("--image-dir", required=True, help="directory the extracted images are written to")
    parser.add_argument(
        "--relative-image-paths",
        action="store_true",
        help="Store images/<name> instead of an absolute path. Absolute is the default because the "
        "SFT loader resolves paths from the worker's cwd, not from the parquet location.",
    )
    args = parser.parse_args()
    convert_dataset(args.input_file, args.output_file, args.image_dir, not args.relative_image_paths)
