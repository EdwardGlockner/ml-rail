"""Cache CLIP ViT-B/16 patch tokens for the
fine-grained X-TAIL domains, for the spatial-readout diagnostics.

Per domain: the same train-pool and validation indices as `scripts/xtail_encode.py`, plus a fixed
test subset of up to 25 images per class (diagnostics only -- final results use the full test
splits). Stores the projected, L2-normalized final-layer patch tokens (196 x 512, float16) and the
global embedding, in the 14 x 14 grid order.

Usage:
    python scripts/xtail_encode_patches.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from data.real.xtail import load_domain  # noqa: E402
from scripts.xtail_encode import MODEL, OUT as FEAT_DIR, Images, pool_indices  # noqa: E402

FINE = ("aircraft", "oxford_flowers", "food101", "oxford_pets", "stanford_cars")
OUT = Path("D:/xtail_data/features/vitb16_patches")
TEST_CAP = 25


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


@torch.no_grad()
def encode(zs, loader):
    vm, proj = zs.model.vision_model, zs.model.visual_projection
    glob, patches = [], []
    for batch in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            o = vm(pixel_values=zs._preprocess_images(batch.numpy()))
            g = proj(o.pooler_output).float()
            p = proj(vm.post_layernorm(o.last_hidden_state[:, 1:])).float()
        glob.append((g / g.norm(dim=-1, keepdim=True)).cpu())
        patches.append((p / p.norm(dim=-1, keepdim=True)).half().cpu())
    return torch.cat(glob).numpy(), torch.cat(patches).numpy()


def main():
    from features.clip_zero_shot import ClipZeroShotClassifier
    OUT.mkdir(parents=True, exist_ok=True)
    zs = ClipZeroShotClassifier(MODEL)
    for name in FINE:
        path = OUT / f"{name}.npz"
        if path.exists():
            continue
        dom = load_domain(name)
        main_cache = np.load(FEAT_DIR / f"{name}.npz", allow_pickle=True)
        data = {}
        for split in ("train", "val", "test"):
            labels = np.array([lab for _, lab in dom.items[split]])
            idx = main_cache[f"{split}_index"] if split != "test" else pool_indices(labels, TEST_CAP, seed=7)
            g, p = encode(zs, DataLoader(Images(dom, split, idx), batch_size=64, num_workers=3))
            data.update({f"{split}_global": g, f"{split}_patches": p, f"{split}_labels": labels[idx], f"{split}_index": idx})
            log(f"{name} {split}: {len(idx)} images")
        np.savez(path, **data)


if __name__ == "__main__":
    main()
