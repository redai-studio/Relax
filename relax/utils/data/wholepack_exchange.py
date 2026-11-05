# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Optional CPU whole-pack exchanges for the reviewed TP4 geometry."""

import collections
import json
import time

from .wholepack_vector_search import optimize


def refine(bins, capacity):
    if capacity != 262144 or len(bins) != 16:
        raise ValueError("wholepack exchange requires DP16 capacity262144")
    started = time.monotonic()
    k = len(bins[0])

    def measure(groups):
        weights = [10 + int(max(g[w]["T"] for g in groups) > capacity) for w in range(k)]
        a = [
            sum(max(groups[d][w]["Q"], groups[d + 1][w]["Q"]) * weights[w] for w in range(k)) for d in range(0, 16, 2)
        ]
        b = [sum(max(groups[d][w]["T"], groups[d + 1][w]["T"]) for w in range(k)) for d in range(0, 16, 2)]
        return dict(A=a, B=b, Amax=max(a), Bmax=max(b), long_signature=weights)

    def signature(groups):
        return collections.Counter(json.dumps(w, sort_keys=True) for g in groups for w in g)

    original = measure(bins)
    # Existing valid fallback layouts can contain multiple oversize singletons.
    if any(sum(w["oversize"] for w in g) > 1 for g in bins):
        return bins, dict(
            original=original,
            candidate=original,
            accepted=0,
            trials=0,
            fallback_reason="multiple_oversize_in_existing_bin",
            seconds=time.monotonic() - started,
        )
    # Candidate tensor scales cubically in K; keep CPU planning bounded.
    if k > 16:
        return bins, dict(
            original=original,
            candidate=original,
            accepted=0,
            trials=0,
            fallback_reason="wave_count_above_16",
            seconds=time.monotonic() - started,
        )
    result, candidate, accepted, trials = optimize(bins, measure, 3)
    assert signature(result) == signature(bins)
    assert all(len(g) == k for g in result)
    assert candidate["long_signature"] == original["long_signature"]
    assert candidate["Amax"] <= original["Amax"] and candidate["Bmax"] <= original["Bmax"]
    identities = [i for g in result for w in g if not w["dummy"] for i in w["ids"]]
    assert len(identities) == len(set(identities)) == 1600
    assert all(sum(w["oversize"] for w in g) <= 1 for g in result)
    assert all(any(not r["dummy"] and r["ids"] == w["ids"] for r in g) for g in result for w in g if w["dummy"])
    return result, dict(
        original=original, candidate=candidate, accepted=accepted, trials=trials, seconds=time.monotonic() - started
    )
