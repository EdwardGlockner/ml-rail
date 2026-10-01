"""Does the validation-chosen configuration change if validation is
capped at 4 images per class (CoOp's few-shot convention, min(shots, 4)) instead of 10?

Usage:
    python scripts/run_xtail_tune_val4.py
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from evaluation import xtail_protocol as xp  # noqa: E402
from scripts.run_xtail_tune import BLOCK_SETS, SEED, WEIGHTS, block_feature  # noqa: E402

OUT = REPO_ROOT / "data" / "generated" / "xtail"


def cap_val(data, cap=4, seed=11):
    rng = np.random.default_rng(seed)
    for d in data.values():
        y = d["val_labels"]
        keep = np.sort(np.concatenate([rng.permutation(np.where(y == c)[0])[:cap] for c in np.unique(y)]))
        for k in [k for k in d if k.startswith("val_")]:
            d[k] = d[k][keep]
    return data


def main():
    data = cap_val(xp.load_features())
    zs = xp.zero_shot_probs(data, split="val")
    res = []
    for bname, gamma in itertools.product(("G", "G+P", "G+L10", "G+L10+P", "G+L8+L10+P"), (0.1, 0.2, 0.4)):
        tables = xp.run_stream(data, xp.DualRAIL(gamma=gamma, lam=1e-3, dtype=torch.float32), SEED,
                               feat=block_feature(BLOCK_SETS[bname]), fusion_weight=list(WEIGHTS), zs=zs, split="val")
        for w, t in tables.items():
            res.append({"blocks": bname, "gamma": gamma, "lam": 1e-3, "w": w, **xp.metrics(t)})
        print(bname, gamma, flush=True)
    json.dump(res, open(OUT / "tune_val4.json", "w"), indent=1)
    top = sorted(res, key=lambda r: -(r["average"] + r["last"]))[:5]
    print("top 5 with 4 val/class:\n" + "\n".join(json.dumps(r) for r in top))


if __name__ == "__main__":
    main()
