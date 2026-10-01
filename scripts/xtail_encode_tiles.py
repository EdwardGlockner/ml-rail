"""B3: high-resolution detail without spatial decomposition, for the
fine-grained X-TAIL domains. Each image (same indices as `scripts/xtail_encode_patches.py`) is
resized (short side 448, bicubic), center-cropped to 448, and split into four 224 tiles; each tile
is encoded as an ordinary image. Stored: the mean of the four tiles' global embeddings
(`tiles_global`, re-normalized) and of their patch means (`tiles_patch_mean`).

Usage:
    python scripts/xtail_encode_tiles.py
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
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from data.real.xtail import clip_preprocess, load_domain  # noqa: E402
from scripts.xtail_encode import MODEL  # noqa: E402
from scripts.xtail_encode_patches import FINE, OUT as PATCH_DIR  # noqa: E402

OUT = Path("D:/xtail_data/features/vitb16_tiles")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Tiles(Dataset):
    def __init__(self, dom, split, idx):
        self.dom, self.split, self.idx = dom, split, idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        big = clip_preprocess(self.dom.load(self.split, int(self.idx[i])), size=448)
        tiles = [big[r:r + 224, c:c + 224] for r in (0, 224) for c in (0, 224)]
        return torch.from_numpy(np.stack(tiles).copy())  # (4, 224, 224, 3)


def norm(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


@torch.no_grad()
def encode(zs, loader):
    vm, proj = zs.model.vision_model, zs.model.visual_projection
    g_out, p_out = [], []
    for batch in loader:
        b = batch.shape[0]
        x = zs._preprocess_images(batch.reshape(-1, 224, 224, 3).numpy())
        with torch.autocast("cuda", dtype=torch.float16):
            o = vm(pixel_values=x)
            g = norm(proj(o.pooler_output).float())
            p = norm(norm(proj(vm.post_layernorm(o.last_hidden_state[:, 1:])).float()).mean(1))
        g_out.append(norm(g.reshape(b, 4, -1).mean(1)).cpu())
        p_out.append(norm(p.reshape(b, 4, -1).mean(1)).cpu())
    return torch.cat(g_out).numpy().astype(np.float32), torch.cat(p_out).numpy().astype(np.float32)


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
            g, p = encode(zs, DataLoader(Tiles(dom, split, idx), batch_size=24, num_workers=3))
            data.update({f"{split}_tiles_global": g, f"{split}_tiles_patch_mean": p, f"{split}_index": idx})
            log(f"{name} {split}: {len(idx)} images")
        np.savez(path, **data)


if __name__ == "__main__":
    main()
