"""Does another backbone want a different kernel width? Validation check (sampling seed 100, order I) of
gamma for Dual-RAIL (final embedding) and ML-RAIL (five blocks) on a backbone cache from
scripts/xtail_encode_backbone.py, with lambda 1e-3 and fusion weight 0.8 as for ViT-B/16. The reported
backbone results keep ViT-B/16's settings; this shows whether that choice is far from what validation
on the new backbone would pick.

Usage:
    python scripts/experiments/run_xtail_backbone_gamma.py [--backbone vitl14]
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

import torch  # noqa: E402

from data.real.xtail import XTAIL_ROOT  # noqa: E402
from evaluation import xtail_protocol as xp  # noqa: E402
from evaluation.ml_rail import BLOCKS, blocks_feature, load_with_extra  # noqa: E402

GAMMAS = (0.05, 0.1, 0.2, 0.4, 0.8)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(backbone):
    out = XTAIL_ROOT / "results" / f"backbone_gamma_{backbone}.json"
    res = json.load(open(out)) if out.exists() else {}
    data = load_with_extra("single", backbone=backbone)
    zs = xp.zero_shot_probs(data, split="val")
    for name, feat in (("rail_dual", "global"), ("ml_rail", blocks_feature(BLOCKS))):
        for g in GAMMAS:
            key = f"{name}_gamma{g}"
            if key in res:
                continue
            table = xp.run_stream(data, xp.DualRAIL(gamma=g, lam=1e-3, dtype=torch.float32), 100, feat=feat,
                                  fusion_weight=0.8, zs=zs, split="val")
            res[key] = xp.metrics(table)
            json.dump(res, open(out, "w"), indent=1)
            m = res[key]
            log(f"{backbone} val {key}: {m['transfer']:.2f} / {m['average']:.2f} / {m['last']:.2f}")
    for name, frozen in (("rail_dual", 0.1), ("ml_rail", 0.2)):
        best = max(GAMMAS, key=lambda g: res[f"{name}_gamma{g}"]["average"])
        log(f"{name}: validation picks gamma {best} (Average {res[f'{name}_gamma{best}']['average']:.2f}); "
            f"frozen gamma {frozen} gives {res[f'{name}_gamma{frozen}']['average']:.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="vitl14")
    main(p.parse_args().backbone)
