"""ML-RAIL tuned under RAIL's own protocol: feature blocks, kernel width gamma and ridge lambda are chosen on
the first domain's validation split only (RAIL: "minimizing the regression error on the validation set of
the first domain, without access to future domains"), then evaluated on the held-out seeds 6-10. Two
selections: over all block sets, and gamma and lambda only, with ML-RAIL's five blocks fixed.

The gates' settings cannot be tuned this way (measuring their effect on Transfer needs domains not yet
learned), so the gate rows (rescue gate; rescue and reverse gates) use the settings selected on all
domains' validation data; see the README.

Usage:
    python scripts/experiments/run_xtail_first_domain_tuning.py
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
from evaluation.ml_rail import BLOCKS, blocks_feature, load_with_extra  # noqa: E402

BLOCK_SETS = {"global": ("global",), "three_blocks": ("global", "cls10", "patch_mean"), "ml_rail": BLOCKS}
GAMMAS, LAMS = (0.05, 0.1, 0.2, 0.4, 0.8), (1e-4, 1e-3, 1e-2, 1e-1)
VAL_SEED, TEST_SEEDS = 100, (6, 7, 8, 9, 10)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def first_domain_grid(data):
    name = next(iter(data))
    d = data[name]
    idx = xp.fewshot(d, VAL_SEED, 16, 0)
    y_val = d["val_labels"]
    rows = []
    for (bname, blocks), gamma, lam in itertools.product(BLOCK_SETS.items(), GAMMAS, LAMS):
        feat = blocks_feature(blocks)
        m = xp.DualRAIL(gamma=gamma, lam=lam, dtype=torch.float64)
        m.add(feat(d, "train", idx), d["train_labels"][idx])
        s = np.asarray(m.scores(feat(d, "val")), dtype=np.float64)
        target = np.eye(d["n_classes"])[y_val]
        rows.append({"blocks": bname, "gamma": gamma, "lam": lam, "val_mse": float(((s - target) ** 2).mean()),
                     "val_acc": float((s.argmax(1) == y_val).mean() * 100)})
    return name, rows


def main():
    out = XTAIL_ROOT / "results" / "first_domain_tuning.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    data = load_with_extra("single")
    name, grid = first_domain_grid(data)
    best = min(grid, key=lambda r: r["val_mse"])
    best_blocks = min((r for r in grid if r["blocks"] == "ml_rail"), key=lambda r: r["val_mse"])
    log(f"first domain ({name}) selection by validation regression error: {best}; five blocks: {best_blocks}")
    zs = xp.zero_shot_probs(data)
    results = {"first_domain": name, "grid": grid, "selected": best, "selected_five_blocks": best_blocks, "test": []}
    if out.exists():  # resume
        results["test"] = json.load(open(out)).get("test", [])
    for prefix, sel in (("ml_rail_first_domain", best), ("ml_rail_blocks_first_domain_params", best_blocks)):
        feat = blocks_feature(BLOCK_SETS[sel["blocks"]])
        for seed in TEST_SEEDS:
            for gname, gate in ((prefix, lambda: "rail"), (prefix + "_gate", lambda: xp.DomainThresholdGate(0.25, 0.1)),
                                (prefix + "_gate_reverse", lambda: xp.DomainThresholdGate(0.25, 0.1, reverse_q=0.01))):
                if any(r["method"] == gname and r["seed"] == seed for r in results["test"]):
                    continue
                t0 = time.time()
                table = xp.run_stream(data, xp.DualRAIL(gamma=sel["gamma"], lam=sel["lam"], dtype=torch.float32), seed,
                                      feat=feat, fusion_weight=0.8, zs=zs, gate=gate())
                results["test"].append({"method": gname, "seed": seed, **xp.metrics(table), "table": table.tolist()})
                json.dump(results, open(out, "w"), indent=1)
                m = results["test"][-1]
                log(f"seed {seed} {gname}: {m['transfer']:.2f} / {m['average']:.2f} / {m['last']:.2f} [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
