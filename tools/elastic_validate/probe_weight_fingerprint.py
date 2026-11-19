#!/usr/bin/env python3
"""Compare rollout engines using fixed-prompt input-token logprobs.

This is a behavioral probe, not a checksum of every weight tensor.

Usage (run inside the training cluster's network namespace):

    python probe_weight_fingerprint.py --engine-urls http://h1:30001,http://h2:30002 \
        --model /path/to/Qwen3-4B --out /tmp/fingerprint_t0.json

Compare two runs:

    python probe_weight_fingerprint.py --compare /tmp/fingerprint_t0.json /tmp/fingerprint_t1.json
"""

import argparse
import json
import math
import sys

import requests


PROBES = [
    "The quick brown fox jumps over the lazy dog. In a groundbreaking discovery, scientists announced today that",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left =",
    "解下列方程：已知 x^2 - 5x + 6 = 0，那么 x 的值为多少？我们一步一步分析，首先对方程进行因式分解，得到",
]


def fingerprint_engine(url: str, tokenizer, timeout: float = 120.0) -> dict:
    """Collect the per-token logprob fingerprint from one engine."""
    per_prompt = []
    for text in PROBES:
        input_ids = tokenizer(text, return_tensors=None)["input_ids"]
        payload = {
            "input_ids": input_ids,
            "sampling_params": {"max_new_tokens": 1, "temperature": 0},
            "return_logprob": True,
            "logprob_start_len": 0,
        }
        resp = requests.post(f"{url}/generate", json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        # meta_info.input_token_logprobs: [[logprob, token_id, text], ...];
        # the first position carries logprob=None.
        token_lps = data["meta_info"]["input_token_logprobs"]
        per_prompt.append([lp for lp, _tid, _text in token_lps if lp is not None])
    return {"url": url, "prompts": per_prompt}


def compare(fp_a: dict, fp_b: dict) -> int:
    """Compare two fingerprint files; exit code 0 = match within tolerance."""
    worst = 0.0
    worst_where = None
    prompts_a, prompts_b = fp_a["prompts"], fp_b["prompts"]
    if not prompts_a or len(prompts_a) != len(prompts_b):
        print("Empty or unequal prompt sets — NOT comparable")
        return 2
    for i, (pa, pb) in enumerate(zip(prompts_a, prompts_b, strict=True)):
        if not pa or len(pa) != len(pb):
            print(f"PROMPT {i}: length mismatch {len(pa)} vs {len(pb)} — NOT comparable")
            return 2
        for j, (xa, xb) in enumerate(zip(pa, pb, strict=True)):
            if not math.isfinite(xa) or not math.isfinite(xb):
                print(f"PROMPT {i}, token {j}: nonfinite logprob — NOT comparable")
                return 2
            d = abs(xa - xb)
            if d > worst:
                worst = d
                worst_where = (i, j, xa, xb)
    print(f"engines: {fp_a['url']}  vs  {fp_b['url']}")
    print(f"max |Δlogprob| = {worst:.6f} at {worst_where}")
    TOL = 0.05
    if worst <= TOL:
        print(f"MATCH (tolerance {TOL})")
        return 0
    print(f"MISMATCH (tolerance {TOL}) — logprob probes differ")
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine-urls", help="comma-separated engine base URLs")
    ap.add_argument("--model", default="/path/to/Qwen3-4B", help="tokenizer path")
    ap.add_argument("--out", help="write fingerprints to this json file")
    ap.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"), help="compare two fingerprint files")
    args = ap.parse_args()

    if args.compare:
        fp_a = json.load(open(args.compare[0]))
        fp_b = json.load(open(args.compare[1]))
        rc = 0
        engines_a = fp_a["engines"]
        engines_b = fp_b["engines"]
        if not engines_a or len(engines_a) != len(engines_b):
            print(f"engine count differs: {len(engines_a)} vs {len(engines_b)}")
            return 2
        # Engines are collected in startup order (engine 0..N); ports are
        # assigned per run, so match by index, not by port.
        for i, (a, b) in enumerate(zip(engines_a, engines_b)):
            print(f"--- engine index {i} ---")
            rc |= compare(a, b)
        sys.exit(rc)

    if not args.engine_urls:
        ap.error("--engine-urls required (or use --compare)")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    engines = []
    for url in args.engine_urls.split(","):
        url = url.strip().rstrip("/")
        print(f"probing {url} ...", flush=True)
        engines.append(fingerprint_engine(url, tokenizer))
    result = {"engines": engines}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f)
        print(f"written: {args.out}")
    else:
        print(json.dumps(result)[:500])


if __name__ == "__main__":
    sys.exit(main())
