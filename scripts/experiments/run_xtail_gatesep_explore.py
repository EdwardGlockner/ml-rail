"""Exploration after the separability test (NOT pre-registered; informs the method's pre-registration).

1. Does the logistic gate generalize to domains it was not fit on? Fit on validation images of half the
   domains, score test images of the other half (both halves, summed; then the reverse split).
2. Can a two-signal AND rule (two thresholds, chosen on validation) match the logistic gate?

Usage (after run_xtail_gatesep.py):
    python scripts/experiments/run_xtail_gatesep_explore.py
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from scripts.experiments.run_xtail_gatesep import OUT, SIGNALS, apply, best_threshold, design  # noqa: E402

V = dict(np.load(OUT / "gate_separability_val.npz"))
T = dict(np.load(OUT / "gate_separability_test.npz"))


def sub(D, mask):
    return {k: v[mask] for k, v in D.items()}


def logistic_gain(Vfit, Tapply):
    mu, sd = design(Vfit).mean(0), design(Vfit).std(0) + 1e-8
    lr = LogisticRegression(max_iter=2000).fit((design(Vfit) - mu) / sd, Vfit["label"] > 0, sample_weight=Vfit["w"])
    sv = lr.predict_proba((design(Vfit) - mu) / sd)[:, 1]
    st = lr.predict_proba((design(Tapply) - mu) / sd)[:, 1]
    g, tau, d = best_threshold(sv, Vfit["label"], Vfit["w"])
    return apply(st, Tapply["label"], Tapply["w"], tau, d)


print("1. logistic gate across domains (fit on validation of one half, applied to test of the other half)")
rng = np.random.default_rng(0)
for split_name, A in (("order halves 0-4 | 5-9", set(range(5))), ("even | odd", set(range(0, 10, 2))),
                      ("random", set(rng.choice(10, 5, replace=False).tolist()))):
    inA_v, inA_t = np.isin(V["domain"], list(A)), np.isin(T["domain"], list(A))
    g = logistic_gain(sub(V, inA_v), sub(T, ~inA_t)) + logistic_gain(sub(V, ~inA_v), sub(T, inA_t))
    print(f"   {split_name:24s} held-out-domain test gain {g:+.2f} Average")
print(f"   (in-domain reference, all domains: {logistic_gain(V, T):+.2f})")

print("2. two-signal AND rules (each threshold from a quantile grid, chosen on validation)")
qs = np.linspace(0, 0.995, 200)
for a, b in itertools.combinations(SIGNALS, 2):
    ta, tb = np.quantile(V[a], qs), np.quantile(V[b], qs)
    lw = V["label"] * V["w"]
    best = (0.0, None, None)
    for x in ta:
        ma = V[a] >= x
        if (lw[ma]).sum() <= best[0] and ma.mean() < 0.01:
            continue
        for y in tb:
            g = lw[ma & (V[b] >= y)].sum()
            if g > best[0]:
                best = (float(g), x, y)
    gt = float((T["label"] * T["w"])[(T[a] >= best[1]) & (T[b] >= best[2])].sum()) if best[1] is not None else 0.0
    print(f"   {a} AND {b}: val {best[0]:+.2f} -> test {gt:+.2f}")
