# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""One-rollout comparison of DAPO scores at a44ba4c and the current revision."""

import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from .math_dapo_utils import compute_score, normalize_final_answer


def _golden_score(response, label):
    matches = re.findall(r"(?i)Answer\s*:\s*([^\n]+)", response[-300:])
    pred = normalize_final_answer(matches[-1] if matches else "[INVALID]")
    truth = str(int(float(normalize_final_answer(label))))
    return 1.0 if pred == truth else -1.0


def _hash_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fingerprint(data_dir, model):
    paths = {
        "train": data_dir / "dapo-math-17k/dapo-math-17k.jsonl",
        "aime": data_dir / "aime-2024/aime-2024.jsonl",
    }
    shards = sorted(model.glob("*.safetensors"))
    if not shards:
        raise RuntimeError("no model weight shards found")
    paths.update({f"model/{p.name}": p for p in shards})
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json"):
        path = model / name
        if path.is_file():
            paths[f"model/{name}"] = path
    print("CE_DAPO_INPUTS " + json.dumps({k: _hash_file(p) for k, p in paths.items()}, sort_keys=True), flush=True)


def replay(args, samples, phase, rollout_id):
    counts = Counter({k: 0 for k in ("paired", "old_correct", "new_correct", "up", "down", "missing", "mismatch", "errors")})
    groups = defaultdict(list)
    inputs = []
    for sample in samples:
        try:
            old = _golden_score(sample.response, sample.label)
            new = compute_score(sample.response, sample.label)["score"]
            native = sample.reward
            if isinstance(native, dict):
                native = native.get(args.eval_reward_key or args.reward_key if phase != "train" else args.reward_key)
            if native not in (-1, 1):
                counts["missing"] += 1
            elif native != new:
                counts["mismatch"] += 1
            counts["paired"] += 1
            counts["old_correct"] += old == 1
            counts["new_correct"] += new == 1
            counts["up"] += old < new
            counts["down"] += old > new
            if phase != "train" and isinstance(sample.index, int):
                groups[sample.index // args.n_samples_per_eval_prompt].append((old == 1, new == 1))
            prompt_length = len(sample.tokens) - sample.response_length
            if prompt_length > 0:
                inputs.append(hashlib.sha256(json.dumps(sample.tokens[:prompt_length]).encode()).hexdigest())
        except (ValueError, TypeError, KeyError, OverflowError):
            counts["errors"] += 1
    result = dict(counts, phase=phase, rollout_id=rollout_id, samples=len(samples), fingerprinted=len(inputs))
    result["input_multiset_sha256"] = hashlib.sha256("".join(sorted(inputs)).encode()).hexdigest()
    if counts["paired"]:
        result["reward_delta"] = 2 * (counts["up"] - counts["down"]) / counts["paired"]
    if groups:
        complete = [g for g in groups.values() if len(g) == args.n_samples_per_eval_prompt]
        result["complete_groups"] = len(complete)
        result["old_pass_at_k_count"] = sum(any(a for a, _ in g) for g in complete)
        result["new_pass_at_k_count"] = sum(any(b for _, b in g) for g in complete)
    print("CE_DAPO_REPLAY " + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    fingerprint(Path(sys.argv[1]), Path(sys.argv[2]))
