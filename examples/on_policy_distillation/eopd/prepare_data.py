"""Prepare MATH train + MATH-500 eval jsonl for the EOPD example.

Output records: {"prompt": [{"role": "user", "content": ...}], "label": str}
Prompt template follows EOPD (arXiv 2603.07079) / verl math_dataset.py: problem
+ " Let's think step by step and output the final answer within \\boxed{}."
"""

import argparse
import glob
import json
import os
import sys

import pandas as pd


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from relax.engine.rewards.math_dapo_utils import last_boxed_only_string, remove_boxed  # noqa: E402


INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."


def extract_solution(sol: str):
    boxed = last_boxed_only_string(sol)
    if boxed is None:
        return None
    try:
        return remove_boxed(boxed)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--math-dir", required=True, help="local snapshot of DigitalLearningGmbH/MATH-lighteval")
    ap.add_argument("--math500-jsonl", required=True, help="HuggingFaceH4/MATH-500 test.jsonl")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(a.math_dir, "data", "train-*.parquet")))
    assert files, f"no train parquet under {a.math_dir}"
    n_ok = n_skip = 0
    with open(os.path.join(a.out_dir, "math_train.jsonl"), "w") as f:
        for fp in files:
            df = pd.read_parquet(fp)
            for _, r in df.iterrows():
                label = extract_solution(r["solution"])
                if not label:
                    n_skip += 1
                    continue
                rec = {"prompt": [{"role": "user", "content": r["problem"] + " " + INSTRUCTION}], "label": str(label)}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_ok += 1
    print("math_train.jsonl:", n_ok, "rows,", n_skip, "skipped (no boxed answer)")

    n = 0
    with open(os.path.join(a.out_dir, "math500_test.jsonl"), "w") as f:
        for line in open(a.math500_jsonl):
            r = json.loads(line)
            rec = {
                "prompt": [{"role": "user", "content": r["problem"] + " " + INSTRUCTION}],
                "label": str(r["answer"]),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print("math500_test.jsonl:", n, "rows")


if __name__ == "__main__":
    main()
