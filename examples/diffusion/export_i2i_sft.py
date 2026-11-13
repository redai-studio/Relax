# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Export a committed image SFT checkpoint through its model adapter."""

import argparse
import json
from pathlib import Path

from relax.utils.misc import load_function


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads((args.checkpoint / "adapter_contract.json").read_text())
    adapter = load_function(contract["adapter_path"])()
    adapter.export_checkpoint(args.checkpoint, args.output, contract)


if __name__ == "__main__":
    main()
