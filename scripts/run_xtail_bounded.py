"""Phase 1 kill test (validation, X-TAIL 10 domains, seed 100, RAIL's
prompt): bounded-memory ML-RAIL (random Fourier features, width D) vs. dual ML-RAIL (all training features
stored), both with RAIL's gate and fusion. Pass: some D <= 16000 within 0.5 on Average and Last.

Usage:
    python scripts/run_xtail_bounded.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from evaluation import xtail_protocol as xp  # noqa: E402
from scripts.run_xtail_improve import blocks_feature, load_with_extra  # noqa: E402

OUT = REPO_ROOT / "data" / "generated" / "xtail"
APRIME = ("global", "cls10", "patch_mean", "gem8", "gem10")
WIDTHS = tuple(int(w) for w in __import__("os").environ.get("XTAIL_WIDTHS", "4000,8000,16000").split(","))
SEED = 100


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    data = load_with_extra("single")
    zs = xp.zero_shot_probs(data, split="val")
    feat = blocks_feature(APRIME)
    in_dim = feat(data["aircraft"], "train", [0]).shape[1]
    results = []
    runs = [] if __import__("os").environ.get("XTAIL_SKIP_DUAL") else [("dual", lambda: xp.DualRAIL(gamma=0.2, lam=1e-3, dtype=torch.float32))]
    prev = OUT / "bounded_val.json"
    if prev.exists():  # keep earlier results (the run is resumed width by width)
        results = [r for r in json.load(open(prev)) if r["model"] not in {n for n, _ in runs} | {f"rff{D}" for D in WIDTHS}]
    runs += [(f"rff{D}", lambda D=D: xp.RFFRidge(in_dim, width=D, gamma=0.2, lam=1e-3, seed=SEED)) for D in WIDTHS]
    # pre-registered fallback: ReLU random projection, ridge strength on validation
    for D in [int(w) for w in __import__("os").environ.get("XTAIL_RELU_WIDTHS", "").split(",") if w]:
        for lam in (1e-2, 1e-1, 1.0):
            runs.append((f"relu{D}_lam{lam}", lambda D=D, lam=lam: xp.RFFRidge(in_dim, width=D, lam=lam, seed=SEED, feature_map="relu")))
    for name, make in runs:
        t0 = time.time()
        m = make()
        table = xp.run_stream(data, m, SEED, feat=feat, fusion_weight=0.8, zs=zs, split="val")
        r = {"model": name, **xp.metrics(table), "seconds": time.time() - t0,
             "memory_numbers": m.memory_numbers() if hasattr(m, "memory_numbers") else int(m.X.numel() + m.alpha.numel())}
        results.append(r)
        json.dump(results, open(OUT / "bounded_val.json", "w"), indent=1)
        log(f"{name}: {r['transfer']:.2f} / {r['average']:.2f} / {r['last']:.2f}   memory {r['memory_numbers'] / 1e6:.0f}M numbers  [{r['seconds']:.0f}s]")
        del m
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
