"""Improve method A on X-TAIL, choosing everything on validation
(CoOp val, seed 100, as in an earlier experiment).

Stage 1: A1 (fusion calibration: softmax temperature on the adapter's scores) x Z1 (single prompt vs.
prompt ensemble), on A's features and kernel.
Stage 2: A2 (flip-averaged training features; optionally test-time flips), A3 (layer GeM blocks),
A4 (sum of per-block RBF kernels), on top of stage 1's best.

Usage:
    python scripts/experiments/run_xtail_improve.py --stage 1
"""
from __future__ import annotations

import argparse
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
from evaluation import xtail_variants as xv  # noqa: E402
from evaluation.ml_rail import blocks_feature, l2, load_with_extra  # noqa: E402,F401
from scripts.experiments.run_xtail_tune import SEED, block_feature  # noqa: E402

OUT = REPO_ROOT / "data" / "generated" / "xtail"
A_BLOCKS = ("global", "cls10", "patch_mean")
WEIGHTS = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
TEMPS = (None, 0.01, 0.02, 0.05, 0.1, 0.2)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def evaluate(data, zs, make_method, feat, tag, results):
    t0 = time.time()
    tables = xp.run_stream(data, make_method(), SEED, feat=feat, fusion_weight=list(WEIGHTS), zs=zs, split="val")
    for w, t in tables.items():
        results.append({"tag": tag, "w": w, **xp.metrics(t)})
    best = max((r for r in results if r["tag"] == tag), key=lambda r: r["average"] + r["last"])
    log(f"{tag}: best w {best['w']} -> transfer {best['transfer']:.2f} average {best['average']:.2f} last {best['last']:.2f} [{time.time() - t0:.0f}s]")


def stage1():
    results = []
    for prompts in ("single", "ensemble"):
        data = xp.load_features()
        if prompts == "ensemble":
            xp.ensemble_text(data)
        zs = xp.zero_shot_probs(data, split="val")
        for T in TEMPS:
            def make(T=T):
                m = xp.DualRAIL(gamma=0.2, lam=1e-3, dtype=torch.float32)
                return m if T is None else xv.Calibrated(m, T)
            evaluate(data, zs, make, block_feature(A_BLOCKS), f"{prompts}_T{T}", results)
        json.dump(results, open(OUT / "improve_stage1_val.json", "w"), indent=1)
    top = sorted(results, key=lambda r: -(r["average"] + r["last"]))[:8]
    log("top 8 on validation:\n" + "\n".join(json.dumps(r) for r in top))


def stage2():
    s1 = json.load(open(OUT / "improve_stage1_val.json"))
    best1 = max(s1, key=lambda r: r["average"] + r["last"])
    prompts, T = best1["tag"].split("_T")
    T = None if T == "None" else float(T)
    log(f"stage-1 best on validation: prompts {prompts}, temperature {T}, w {best1['w']}")
    data = load_with_extra(prompts)
    zs = xp.zero_shot_probs(data, split="val")
    results = []

    def run(tag, blocks, gamma, flip_train=False, flip_test=False, sum_kernel=False):
        feat = blocks_feature(blocks, flip_train, flip_test)
        dims = [512 if b in ("global", "patch_mean") else 768 for b in blocks]
        edges = np.cumsum([0] + dims)
        slices = [(int(edges[i]), int(edges[i + 1])) for i in range(len(dims))]

        def make():
            m = (xv.SumKernelDual(slices, gamma=gamma, lam=1e-3) if sum_kernel
                 else xp.DualRAIL(gamma=gamma, lam=1e-3, dtype=torch.float32))
            return m if T is None else xv.Calibrated(m, T)
        evaluate(data, zs, make, feat, tag, results)
        json.dump(results, open(OUT / "improve_stage2_val.json", "w"), indent=1)

    base = ("global", "cls10", "patch_mean")
    for gamma in (0.2, 0.4):
        run(f"A_g{gamma}", base, gamma)
        run(f"A2train_g{gamma}", base, gamma, flip_train=True)
        run(f"A2traintest_g{gamma}", base, gamma, flip_train=True, flip_test=True)
        run(f"A3gem10_g{gamma}", base + ("gem10",), gamma)
        run(f"A3gem8gem10_g{gamma}", base + ("gem8", "gem10"), gamma)
        run(f"A2A3_g{gamma}", base + ("gem10",), gamma, flip_train=True, flip_test=True)
    for gamma in (0.5, 1.0, 2.0):  # per-block kernels on unit blocks: wider gamma range
        run(f"A4sum_g{gamma}", base, gamma, sum_kernel=True)
    top = sorted(results, key=lambda r: -(r["average"] + r["last"]))[:8]
    log("top 8 on validation:\n" + "\n".join(json.dumps(r) for r in top))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage", type=int, default=1)
    a = p.parse_args()
    {1: stage1, 2: stage2}[a.stage]()
