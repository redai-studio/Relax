# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU-only vectorized evaluation of the existing whole-pack exchange
neighborhood."""

import itertools

import numpy as np


def optimize(old, measure, phases=3):
    orig = measure(old)
    D = len(old)
    K = len(old[0])
    assert D == 16 and all(len(g) == K for g in old)
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
    coords = np.asarray(
        [(a, b, i, j) for a, b in itertools.combinations(range(D), 2) for i in range(K) for j in range(K)],
        dtype=np.int64,
    )
    a, b, i, j = coords.T
    score = lambda m: max(x / orig["Amax"] + y / orig["Bmax"] for x, y in zip(m["A"], m["B"]))
    for phase in range(phases):
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
        trials += len(selected)
        if not len(selected):
            break
        aa = aa[selected]
        bb = bb[selected]
        ca = a[selected]
        cb = b[selected]
        aa = np.take_along_axis(aa, np.argsort(priority[aa], axis=1), axis=1)
        bb = np.take_along_axis(bb, np.argsort(priority[bb], axis=1), axis=1)
        candidates = np.broadcast_to(assignments, (len(selected), D, K)).copy()
        idx = np.arange(len(selected))
        candidates[idx, ca] = aa
        candidates[idx, cb] = bb
        tq = T[candidates]
        qq = Q[candidates]
        weights = 10 + (tq.max(axis=1) > 262144).astype(np.int64)
        A = (qq.reshape(-1, D // 2, 2, K).max(axis=2) * weights[:, None, :]).sum(axis=2)
        B = tq.reshape(-1, D // 2, 2, K).max(axis=2).sum(axis=2)
        allowed = (
            (weights == np.asarray(orig["long_signature"])).all(axis=1)
            & (A.max(axis=1) <= current["Amax"])
            & (B.max(axis=1) <= current["Bmax"])
        )
        scores = (A / orig["Amax"] + B / orig["Bmax"]).max(axis=1)
        best = None
        bestscore = score(current)
        # Preserve original scan order and its 1e-12 improvement threshold.
        for n in np.flatnonzero(allowed):
            if scores[n] < bestscore - 1e-12:
                best = int(n)
                bestscore = float(scores[n])
        if best is None:
            break
        assignments = candidates[best]
        bins = [[packs[int(k)] for k in row] for row in assignments]
        current = measure(bins)
        assert current["A"] == A[best].tolist() and current["B"] == B[best].tolist()
        accepted += 1
    return [[packs[int(k)] for k in row] for row in assignments], current, accepted, trials
