"""Two checks of the rescue gate on the held-out seeds 6-10 (X-TAIL order I):

1. Sensitivity: change in Transfer/Average/Last over ML-RAIL for every (q, m) of the selection grid.
2. Acceptance rate: after all ten domains are learned, the fraction of each domain's own test images whose
   max similarity to that domain's stored training features clears tau_d, against the nominal 1 - q.

Usage:
    python scripts/experiments/run_xtail_gate_sensitivity.py
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from data.real.xtail import XTAIL_ROOT  # noqa: E402
from evaluation import xtail_protocol as xp  # noqa: E402
from evaluation.ml_rail import BLOCKS, blocks_feature, load_with_extra, make_ml_rail  # noqa: E402

QS, MASSES, SEEDS = (0.25, 0.5), (0.05, 0.1), (6, 7, 8, 9, 10)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def acceptance(data, feat, seed, q):
    m = make_ml_rail()
    for t, d in enumerate(data.values()):
        idx = xp.fewshot(d, seed, 16, t)
        m.add(feat(d, "train", idx), d["train_labels"][idx] + d["offset"])
    rates = {}
    y = torch.as_tensor(m.y, device=m.X.device)
    for name, d in data.items():
        Xd = m.X[(y >= d["offset"]) & (y < d["offset"] + d["n_classes"])]
        K = m._k(Xd, Xd)
        K[K >= 1 - 1e-6] = -1
        tau = float(torch.quantile(K.max(1).values.cpu(), q))
        Xt = torch.as_tensor(feat(d, "test"), device=xp.DEV)
        s = torch.cat([m._k(Xt[i:i + 2048], Xd).max(1).values.cpu() for i in range(0, len(Xt), 2048)])
        rates[name] = float((s >= tau).float().mean())
    return rates


def main():
    out = XTAIL_ROOT / "results" / "gate_sensitivity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    data = load_with_extra("single")
    feat = blocks_feature(BLOCKS)
    zs = xp.zero_shot_probs(data)
    res = json.load(open(out)) if out.exists() else {"sensitivity": [], "acceptance": {}}  # resume
    for seed in SEEDS:
        if sum(r["seed"] == seed for r in res["sensitivity"]) == len(QS) * len(MASSES):
            continue
        res["sensitivity"] = [r for r in res["sensitivity"] if r["seed"] != seed]
        base = xp.metrics(xp.run_stream(data, make_ml_rail(), seed, feat=feat, fusion_weight=0.8, zs=zs))
        for q, mm in itertools.product(QS, MASSES):
            t0 = time.time()
            g = xp.metrics(xp.run_stream(data, make_ml_rail(), seed, feat=feat, fusion_weight=0.8, zs=zs,
                                         gate=xp.DomainThresholdGate(q, mm)))
            res["sensitivity"].append({"seed": seed, "q": q, "min_mass": mm,
                                       **{f"delta_{k}": g[k] - base[k] for k in ("transfer", "average", "last")}})
            json.dump(res, open(out, "w"), indent=1)
            r = res["sensitivity"][-1]
            log(f"seed {seed} q={q} m={mm}: dT {r['delta_transfer']:+.2f} dA {r['delta_average']:+.2f} "
                f"dL {r['delta_last']:+.2f} [{time.time() - t0:.0f}s]")
    for q in QS:
        if str(q) in res["acceptance"]:
            continue
        rates = [acceptance(data, feat, s, q) for s in SEEDS]
        res["acceptance"][str(q)] = {n: float(np.mean([r[n] for r in rates])) for n in data}
        json.dump(res, open(out, "w"), indent=1)
        a = res["acceptance"][str(q)]
        log(f"acceptance at q={q} (nominal {1 - q:.2f}): mean {np.mean(list(a.values())):.3f}, "
            + ", ".join(f"{n} {v:.2f}" for n, v in a.items()))


if __name__ == "__main__":
    main()
