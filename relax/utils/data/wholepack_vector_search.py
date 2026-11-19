# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU-only vectorized evaluation of the existing whole-pack exchange
neighborhood."""

import itertools

import numpy as np


def optimize(old, measure, phases=3):
    orig = measure(old)
    D = len(old)
    K = len(old[0])
    assert D > 0 and D % 2 == 0 and all(len(g) == K for g in old)
    packs = [w for g in old for w in g]
    P = len(packs)
    priority = np.empty(P, dtype=np.int64)
    for n, i in enumerate(
        sorted(
            range(P),
            key=lambda i: (
                -(10 * packs[i]["Q"] / orig["Amax"] + packs[i]["T"] / orig["Bmax"]),
                packs[i]["ids"],
                packs[i]["dummy"],
            ),
        )
    ):
        priority[i] = n
    Q = np.asarray([w["Q"] for w in packs], dtype=np.int64)
    T = np.asarray([w["T"] for w in packs], dtype=np.int64)
    oversized = np.asarray([w["oversize"] for w in packs], dtype=bool)
    excluded = {tuple(w["ids"]) for w in packs if w["dummy"]}
    movable = np.asarray([not w["dummy"] and tuple(w["ids"]) not in excluded for w in packs], dtype=bool)
    assignments = np.arange(P, dtype=np.int64).reshape(D, K)
    current = orig
    accepted = trials = 0
    score = lambda m: max(x / orig["Amax"] + y / orig["Bmax"] for x, y in zip(m["A"], m["B"]))
    # Bound coordinates, sorting scratch, and candidate tensors together.
    chunk_size = max(1, (64 * 1024 * 1024) // (K * 512 + 1024))
    for phase in range(phases):
        best = None
        bestscore = score(current)
        over = (T[assignments] > 262144).astype(np.int64)
        over_count = over.sum(axis=0)
        domain_scores = np.asarray(current["A"]) / orig["Amax"] + np.asarray(current["B"]) / orig["Bmax"]
        top_domains = np.argsort(domain_scores)[-3:]
        row_oversize = oversized[assignments].sum(axis=1)
        counted_trials = bool((row_oversize <= 1).all())
        if counted_trials:
            movable_rows = movable[assignments]
            normal_counts = [int(x) for x in (movable_rows & ~oversized[assignments]).sum(axis=1)]
            over_counts = [int(x) for x in (movable_rows & oversized[assignments]).sum(axis=1)]
            normal_sum, over_sum = sum(normal_counts), sum(over_counts)
            empty_normal_sum = sum(n for n, over in zip(normal_counts, row_oversize) if over == 0)
            trials += (normal_sum * normal_sum - sum(n * n for n in normal_counts)) // 2
            trials += (over_sum * over_sum - sum(n * n for n in over_counts)) // 2
            trials += over_sum * empty_normal_sum
            required = np.flatnonzero(domain_scores >= bestscore - 1e-12).tolist()
            if len(required) > 2:
                break
            if not required:
                pairs = itertools.combinations(range(D), 2)
            elif len(required) == 2:
                pairs = sorted(
                    (min(a, b), max(a, b))
                    for a in (2 * required[0], 2 * required[0] + 1)
                    for b in (2 * required[1], 2 * required[1] + 1)
                )
            else:
                required_ranks = (2 * required[0], 2 * required[0] + 1)
                pairs = (
                    (a, b)
                    for a in range(D)
                    for b in (range(a + 1, D) if a in required_ranks else required_ranks)
                    if b > a
                )
        else:
            # Retain the original behavior for direct callers with abnormal rows.
            pairs = itertools.combinations(range(D), 2)
        coordinates = ((a, b, i, j) for a, b in pairs for i in range(K) for j in range(K))
        while True:
            batch = list(itertools.islice(coordinates, chunk_size))
            if not batch:
                break
            coords = np.asarray(batch, dtype=np.int64)
            a, b, i, j = coords.T
            left = assignments[a, i]
            right = assignments[b, j]
            valid = movable[left] & movable[right]
            aa = assignments[a].copy()
            bb = assignments[b].copy()
            idx = np.arange(len(coords))
            aa[idx, i] = right
            bb[idx, j] = left
            valid &= (oversized[aa].sum(axis=1) <= 1) & (oversized[bb].sum(axis=1) <= 1)
            selected = np.flatnonzero(valid)
            if not counted_trials:
                trials += len(selected)
            if not len(selected):
                continue
            # Any untouched domain is a lower bound on the candidate score.
            # Surviving candidates must retain the original wave weights.
            untouched_bound = np.full(len(selected), -np.inf)
            for domain in top_domains:
                keep = (a[selected] // 2 != domain) & (b[selected] // 2 != domain)
                np.maximum(untouched_bound, np.where(keep, domain_scores[domain], -np.inf), out=untouched_bound)
            selected = selected[untouched_bound < bestscore - 1e-12]
            if not len(selected):
                continue
            aa = aa[selected]
            bb = bb[selected]
            ca = a[selected]
            cb = b[selected]
            aa = np.take_along_axis(aa, np.argsort(priority[aa], axis=1), axis=1)
            bb = np.take_along_axis(bb, np.argsort(priority[bb], axis=1), axis=1)
            same_domain = (ca // 2 == cb // 2)[:, None]
            ta, tb, qa, qb = T[aa], T[bb], Q[aa], Q[bb]
            counts = over_count - over[ca] - over[cb] + (ta > 262144) + (tb > 262144)
            weights = 10 + (counts > 0).astype(np.int64)
            # Each swap changes at most two EP domains. Same-domain swaps must
            # pair both new rows rather than either row's original neighbour.
            qa_peer = np.where(same_domain, qb, Q[assignments[ca ^ 1]])
            qb_peer = np.where(same_domain, qa, Q[assignments[cb ^ 1]])
            ta_peer = np.where(same_domain, tb, T[assignments[ca ^ 1]])
            tb_peer = np.where(same_domain, ta, T[assignments[cb ^ 1]])
            new_a = (np.maximum(qa, qa_peer) * weights).sum(axis=1)
            new_b = (np.maximum(qb, qb_peer) * weights).sum(axis=1)
            padded_a = np.maximum(ta, ta_peer).sum(axis=1)
            padded_b = np.maximum(tb, tb_peer).sum(axis=1)
            allowed = (
                (weights == np.asarray(orig["long_signature"])).all(axis=1)
                & (new_a <= current["Amax"])
                & (new_b <= current["Amax"])
                & (padded_a <= current["Bmax"])
                & (padded_b <= current["Bmax"])
            )
            unchanged = np.full(len(selected), -np.inf)
            for domain in top_domains:
                keep = (ca // 2 != domain) & (cb // 2 != domain)
                np.maximum(unchanged, np.where(keep, domain_scores[domain], -np.inf), out=unchanged)
            scores = np.maximum(
                unchanged,
                np.maximum(
                    new_a / orig["Amax"] + padded_a / orig["Bmax"], new_b / orig["Amax"] + padded_b / orig["Bmax"]
                ),
            )
            # Preserve original scan order and its 1e-12 improvement threshold.
            for n in np.flatnonzero(allowed):
                if scores[n] < bestscore - 1e-12:
                    best = (
                        int(ca[n]),
                        int(cb[n]),
                        aa[n].copy(),
                        bb[n].copy(),
                        new_a[n],
                        new_b[n],
                        padded_a[n],
                        padded_b[n],
                    )
                    bestscore = float(scores[n])
        if best is None:
            break
        a, b, aa, bb, new_a, new_b, padded_a, padded_b = best
        assignments = assignments.copy()
        assignments[a], assignments[b] = aa, bb
        best_a, best_b = np.asarray(current["A"]).copy(), np.asarray(current["B"]).copy()
        best_a[a // 2], best_a[b // 2] = new_a, new_b
        best_b[a // 2], best_b[b // 2] = padded_a, padded_b
        bins = [[packs[int(k)] for k in row] for row in assignments]
        current = measure(bins)
        assert current["A"] == best_a.tolist() and current["B"] == best_b.tolist()
        accepted += 1
    return [[packs[int(k)] for k in row] for row in assignments], current, accepted, trials
