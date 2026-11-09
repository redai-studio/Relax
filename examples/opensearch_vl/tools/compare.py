# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Compare common-evaluator summaries, rejecting different evaluation protocols."""

import argparse
import json
from pathlib import Path

from common import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--relax", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = {name: json.loads(getattr(args, name).read_text()) for name in ("base", "relax")}
    if any(run["protocol"] != runs["base"]["protocol"] for run in runs.values()):
        raise ValueError("Model runs used different test data, processors or evaluation settings")
    if any(run["metrics"]["failures"] for run in runs.values()):
        raise ValueError("Cannot accept incomplete evaluation")
    keys = runs["base"]["metrics"]
    deltas = {key: runs["relax"]["metrics"][key] - runs["base"]["metrics"][key]
              for key in keys if isinstance(runs["relax"]["metrics"][key], (int, float))
              and isinstance(runs["base"]["metrics"][key], (int, float))}
    write_json(args.output, {"runs": runs, "relax_minus_base": deltas,
                             "status": "measured; assess numerical and task differences, not a framework pass certificate"})


if __name__ == "__main__":
    main()
