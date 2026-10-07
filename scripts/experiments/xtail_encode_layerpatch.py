"""Spatial readout, phase 2 (a): pooled intermediate-layer patch tokens
for the fine-grained X-TAIL domains -- mean and GeM (p = 3, of the non-negative part) of the
post-layernorm patch tokens at layers 8 and 10, on exactly the images cached by
`scripts/experiments/xtail_encode_patches.py`.

Usage:
    python scripts/experiments/xtail_encode_layerpatch.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from data.real.xtail import load_domain  # noqa: E402
from data.real.xtail import XTAIL_ROOT  # noqa: E402
from scripts.xtail_encode import MODEL, Images  # noqa: E402
from scripts.experiments.xtail_encode_patches import FINE, OUT as PATCH_DIR  # noqa: E402

OUT = (XTAIL_ROOT / "features/vitb16_layerpatch")
LAYERS = (8, 10)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


@torch.no_grad()
def encode(zs, loader):
    vm = zs.model.vision_model
    out = {f"{k}{l}": [] for l in LAYERS for k in ("mean", "gem")}
    for batch in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            o = vm(pixel_values=zs._preprocess_images(batch.numpy()), output_hidden_states=True)
        for l in LAYERS:
            p = vm.post_layernorm(o.hidden_states[l][:, 1:]).float()
            out[f"mean{l}"].append(p.mean(1).cpu())
            out[f"gem{l}"].append(p.clamp_min(0).pow(3).mean(1).pow(1 / 3).cpu())
    return {k: torch.cat(v).numpy().astype(np.float32) for k, v in out.items()}


def main():
    from features.clip_zero_shot import ClipZeroShotClassifier
    OUT.mkdir(parents=True, exist_ok=True)
    zs = ClipZeroShotClassifier(MODEL)
    for name in FINE:
        path = OUT / f"{name}.npz"
        if path.exists():
            continue
        dom = load_domain(name)
        idx_cache = np.load(PATCH_DIR / f"{name}.npz")
        data = {}
        for split in ("train", "val", "test"):
            idx = idx_cache[f"{split}_index"]
            feats = encode(zs, DataLoader(Images(dom, split, idx), batch_size=64, num_workers=3))
            data.update({f"{split}_{k}": v for k, v in feats.items()})
            data[f"{split}_index"] = idx
            log(f"{name} {split}: {len(idx)} images")
        np.savez(path, **data)


if __name__ == "__main__":
    main()
