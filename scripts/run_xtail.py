"""Run single methods on the X-TAIL protocol (cached CLIP ViT-B/16 features) and save the 10x10 tables;
used for the RAIL replication. The reported results come from scripts/evaluate.py.

Published reference (RAIL, arXiv:2406.18868v3, Table 1, X-TAIL 16-shot, order I; per-domain Last row):
  Primal-RAIL  Transfer 62.4  Average 70.7  Last 79.1
  Dual-RAIL    Transfer 62.4  Average 71.9  Last 82.4
Usage:
    python scripts/run_xtail.py --methods zeroshot,primal,dual --seeds 0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import os  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from data.real.xtail import DOMAINS  # noqa: E402
from evaluation import xtail_protocol as xp  # noqa: E402
from evaluation.ml_rail import BLOCKS, EXTRA, blocks_feature, make_ml_rail  # noqa: E402

OUT = REPO_ROOT / "data" / "generated" / "xtail"
PUBLISHED = {
    "primal": {"transfer": 62.4, "average": 70.7, "last": 79.1,
               "last_row": [41.7, 94.0, 66.0, 86.4, 97.2, 82.4, 93.1, 83.6, 75.0, 71.3]},
    "dual": {"transfer": 62.4, "average": 71.9, "last": 82.4,
             "last_row": [45.3, 94.2, 69.0, 87.0, 97.2, 87.2, 93.0, 92.4, 82.5, 76.3]},
}

# kernel-solve precision: float64 as RAIL (default); XTAIL_DTYPE=float32 is much lighter on a consumer GPU
# and within 0.05 points of float64 (results file names get a _fp32 suffix)
DTYPE = torch.float32 if os.environ.get("XTAIL_DTYPE") == "float32" else torch.float64

METHODS = {  # name -> (factory, features, fusion weight)
    "zeroshot": (lambda: xp.ZeroShot(), "global", 0.8),
    "primal": (lambda: xp.PrimalRAIL(width=15000, reg=0.1), "global", 0.8),
    "dual": (lambda: xp.DualRAIL(gamma=0.1, lam=0.001, dtype=DTYPE), "global", 0.8),
    "mlrail": (lambda: make_ml_rail(DTYPE), BLOCKS, 0.8),
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(methods, seeds, domains, prompts="single", shots=16, tag=""):
    OUT.mkdir(parents=True, exist_ok=True)
    data = xp.load_features(domains)
    if prompts == "ensemble":
        xp.ensemble_text(data)
    elif prompts != "single":  # other text heads (evaluation/xtail_prompts.py), e.g. "t1b+generic"
        from evaluation.xtail_prompts import build_text
        build_text(data, prompts)
    if any(not isinstance(METHODS[m][1], str) for m in methods):
        for n, d in data.items():
            z = np.load(EXTRA / f"{n}.npz")
            d.update({k: z[k] for k in z.files if k.endswith(("gem8", "gem10"))})
    zs = xp.zero_shot_probs(data)
    suffix = {"single": "", "ensemble": "_ens"}.get(prompts, "_" + prompts.replace("+", "_")) + tag
    for seed in seeds:
        for name in methods:
            t0 = time.time()
            factory, feat, w = METHODS[name]
            feat = feat if isinstance(feat, str) else blocks_feature(feat)
            table = xp.run_stream(data, factory(), seed, feat=feat, fusion_weight=w, zs=zs, shots=shots)
            m = xp.metrics(table)
            res = {"method": name, "seed": seed, "domains": list(domains), "shots": shots, "metrics": m, "table": table.tolist()}
            res["prompts"] = prompts
            with open(OUT / f"{name}{suffix}_seed{seed}.json", "w") as f:
                json.dump(res, f, indent=2)
            ref = PUBLISHED.get(name)
            log(f"{name} seed {seed}: transfer {m['transfer']:.1f} average {m['average']:.1f} last {m['last']:.1f}"
                + (f"   (published {ref['transfer']} / {ref['average']} / {ref['last']})" if ref else "") + f"  [{time.time() - t0:.0f}s]")
            log("   last row: " + " ".join(f"{v:.1f}" for v in table[-1])
                + ("\n              published: " + " ".join(f"{v:.1f}" for v in ref["last_row"]) if ref else ""))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--methods", default="zeroshot,primal,dual")
    p.add_argument("--seeds", default="0")
    p.add_argument("--domains", default=",".join(DOMAINS))
    p.add_argument("--prompts", default="single",
                   choices=("single", "ensemble", "generic", "t1a", "t1b", "t1a+generic", "t1b+generic"))
    p.add_argument("--order", default="I", choices=("I", "II", "11"),
                   help="I: X-TAIL order I (default, uses --domains); II: RAIL's order II; 11: the 11-domain variant with CIFAR100")
    p.add_argument("--shots", type=int, default=16)
    a = p.parse_args()
    from data.real.xtail import DOMAINS_11, ORDER_II
    domains = {"I": a.domains.split(","), "II": list(ORDER_II), "11": list(DOMAINS_11)}[a.order]
    tag = (("" if a.order == "I" else f"_order{a.order}") + ("" if a.shots == 16 else f"_{a.shots}shot")
           + ("_fp32" if DTYPE == torch.float32 else ""))
    main(a.methods.split(","), [int(s) for s in a.seeds.split(",")], domains, a.prompts, a.shots, tag)
