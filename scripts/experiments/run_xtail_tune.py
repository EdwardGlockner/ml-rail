"""Choose the training-free method's configuration on
X-TAIL *validation* splits only (CoOp val; <= 10 images per class), never on test.

Readout: RAIL's kernel ridge (Dual form, RBF) with RAIL's fusion rule. Search: which feature blocks
(global, intermediate class tokens, mean patch token -- each unit-normalized, concatenation rescaled
to unit length), kernel gamma, ridge lambda, fusion weight. Criterion: Average + Last on validation
(Transfer is preserved by the fusion rule). Seed 100 for the 16-shot draws (test seeds are 0-4).

Usage:
    python scripts/experiments/run_xtail_tune.py
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

from evaluation import xtail_protocol as xp  # noqa: E402

OUT = REPO_ROOT / "data" / "generated" / "xtail"
BLOCK_SETS = {
    "G": ("global",),
    "G+P": ("global", "patch_mean"),
    "G+L10": ("global", "cls10"),
    "G+L8+L10": ("global", "cls8", "cls10"),
    "G+L10+P": ("global", "cls10", "patch_mean"),
    "G+L8+L10+P": ("global", "cls8", "cls10", "patch_mean"),
    "G+L6+L8+L10+P": ("global", "cls6", "cls8", "cls10", "patch_mean"),
    "all": ("global", "cls4", "cls6", "cls8", "cls10", "patch_mean"),
}
GAMMAS = (0.05, 0.1, 0.2, 0.4)
LAMS = (1e-3, 1e-2, 1e-1)
WEIGHTS = (0.6, 0.7, 0.8, 0.9, 1.0)
SEED = 100


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def block_feature(blocks):
    def f(d, split, idx=None):
        X = np.concatenate([d[f"{split}_{b}"] if idx is None else d[f"{split}_{b}"][idx] for b in blocks], axis=1)
        return (X / np.sqrt(len(blocks))).astype(np.float32)
    return f


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = xp.load_features()
    zs = xp.zero_shot_probs(data, split="val")
    results = []
    for (bname, blocks), gamma, lam in itertools.product(BLOCK_SETS.items(), GAMMAS, LAMS):
        t0 = time.time()
        tables = xp.run_stream(data, xp.DualRAIL(gamma=gamma, lam=lam, dtype=torch.float32), SEED,
                               feat=block_feature(blocks), fusion_weight=list(WEIGHTS), zs=zs, split="val")
        for w, table in tables.items():
            m = xp.metrics(table)
            results.append({"blocks": bname, "gamma": gamma, "lam": lam, "w": w, **m})
        best = max((r for r in results if r["blocks"] == bname and r["gamma"] == gamma and r["lam"] == lam),
                   key=lambda r: r["average"] + r["last"])
        log(f"{bname:14s} gamma {gamma:<5} lam {lam:<6}: best w {best['w']} -> average {best['average']:.2f} last {best['last']:.2f}  [{time.time() - t0:.0f}s]")
        with open(OUT / "tune_val.json", "w") as f:
            json.dump(results, f, indent=1)
    top = sorted(results, key=lambda r: -(r["average"] + r["last"]))[:10]
    rail = [r for r in results if r["blocks"] == "G" and r["gamma"] == 0.1 and r["lam"] == 1e-3 and r["w"] == 0.8]
    log("RAIL's own setting on validation: " + json.dumps(rail))
    log("top 10 on validation:\n" + "\n".join(json.dumps(r) for r in top))


if __name__ == "__main__":
    main()
