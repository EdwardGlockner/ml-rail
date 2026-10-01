"""Extra frozen features for every X-TAIL domain, on exactly the
images in the main cache (train pool, validation, full test) -- no new images.

  flip_global, flip_cls10, flip_patch_mean   features of the horizontally flipped image (A2)
  gem8, gem10                                GeM (p = 3, non-negative part) of post-layernorm patch
                                             tokens at layers 8 and 10 of the original image (A3)

Usage:
    python scripts/xtail_encode_extra.py [--domains ...]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from data.real.xtail import DOMAINS, load_domain  # noqa: E402
from scripts.xtail_encode import MODEL, OUT as FEAT_DIR, WORKERS, Images  # noqa: E402

OUT = Path("D:/xtail_data/features/vitb16_extra")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def norm(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


@torch.no_grad()
def encode(zs, loader):
    vm, proj = zs.model.vision_model, zs.model.visual_projection
    out = {k: [] for k in ("flip_global", "flip_cls10", "flip_patch_mean", "gem8", "gem10")}
    for batch in loader:
        x = zs._preprocess_images(batch.numpy())
        with torch.autocast("cuda", dtype=torch.float16):
            o = vm(pixel_values=x, output_hidden_states=True)
            f = vm(pixel_values=torch.flip(x, dims=[3]), output_hidden_states=True)
        for l in (8, 10):
            p = vm.post_layernorm(o.hidden_states[l][:, 1:]).float()
            out[f"gem{l}"].append(p.clamp_min(0).pow(3).mean(1).pow(1 / 3).cpu())
        out["flip_global"].append(norm(proj(f.pooler_output).float()).cpu())
        out["flip_cls10"].append(norm(vm.post_layernorm(f.hidden_states[10][:, 0]).float()).cpu())
        out["flip_patch_mean"].append(norm(norm(proj(vm.post_layernorm(f.last_hidden_state[:, 1:])).float()).mean(1)).cpu())
    return {k: torch.cat(v).numpy().astype(np.float32) for k, v in out.items()}


def main(domains):
    from features.clip_zero_shot import ClipZeroShotClassifier
    OUT.mkdir(parents=True, exist_ok=True)
    zs = ClipZeroShotClassifier(MODEL)
    for name in domains:
        path = OUT / f"{name}.npz"
        if path.exists():
            continue
        dom = load_domain(name)
        cache = np.load(FEAT_DIR / f"{name}.npz", allow_pickle=True)
        data = {}
        for split in ("train", "val", "test"):
            idx = cache[f"{split}_index"]
            t0 = time.time()
            feats = encode(zs, DataLoader(Images(dom, split, idx), batch_size=96, num_workers=WORKERS))
            data.update({f"{split}_{k}": v for k, v in feats.items()})
            log(f"{name} {split}: {len(idx)} images, {len(idx) / (time.time() - t0):.0f} img/s")
        np.savez(path, **data)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--domains", default=",".join(DOMAINS))
    main(p.parse_args().domains.split(","))
