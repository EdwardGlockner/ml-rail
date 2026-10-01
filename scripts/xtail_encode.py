"""Encode the X-TAIL domains once with CLIP ViT-B/16 (the benchmark's backbone) and cache features.

Per domain: a fixed random pool of up to 80 training images per class (every 16-shot draw for our
seeds is taken from it -- still a uniformly random 16-shot sample of the CoOp train split), up to 10
validation images per class (hyperparameter selection), and the full test split. Features: global
embedding (projected, L2-normalized), class tokens of layers 4/6/8/10 (post-layernorm, normalized),
mean of the projected patch tokens (normalized), and class-name text embeddings with RAIL's prompt
"a photo of a {}.". CLIP-standard preprocessing (bicubic short-side resize + center crop).

Usage:
    python scripts/xtail_encode.py [--domains aircraft,caltech101,...]
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
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from data.real.xtail import DOMAINS, clip_preprocess, load_domain  # noqa: E402

OUT = Path("D:/xtail_data/features/vitb16")
MODEL = "openai/clip-vit-base-patch16"
LAYERS = (4, 6, 8, 10)
CAP = {"train": 80, "val": 10, "test": None}
PROMPT = "a photo of a {}."
# 4 logical CPUs; "gentle mode" (XTAIL_WORKERS=1) keeps the machine usable while encoding
WORKERS = int(__import__("os").environ.get("XTAIL_WORKERS", 3))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pool_indices(labels, cap, seed=2026):
    if cap is None:
        return np.arange(len(labels))
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    keep = [rng.permutation(np.where(labels == c)[0])[:cap] for c in np.unique(labels)]
    return np.sort(np.concatenate(keep))


class Images(Dataset):
    def __init__(self, dom, split, idx):
        self.dom, self.split, self.idx = dom, split, idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        return torch.from_numpy(np.array(clip_preprocess(self.dom.load(self.split, int(self.idx[i])))))


@torch.no_grad()
def encode(zs, loader):
    vm, proj = zs.model.vision_model, zs.model.visual_projection
    out = {k: [] for k in ("global", "patch_mean", *[f"cls{l}" for l in LAYERS])}

    def norm(x):
        return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    for batch in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            o = vm(pixel_values=zs._preprocess_images(batch.numpy()), output_hidden_states=True)
            g = proj(o.pooler_output).float()
            p = norm(proj(vm.post_layernorm(o.last_hidden_state[:, 1:])).float())
        out["global"].append(norm(g).cpu())
        out["patch_mean"].append(norm(p.mean(1)).cpu())
        for l in LAYERS:
            out[f"cls{l}"].append(norm(vm.post_layernorm(o.hidden_states[l][:, 0]).float()).cpu())
    return {k: torch.cat(v).numpy().astype(np.float32) for k, v in out.items()}


def main(domains):
    from features.clip_zero_shot import ClipZeroShotClassifier
    OUT.mkdir(parents=True, exist_ok=True)
    zs = ClipZeroShotClassifier(MODEL)
    for name in domains:
        path = OUT / f"{name}.npz"
        if path.exists():
            log(f"{name}: cached")
            continue
        dom = load_domain(name)
        data = {"classnames": np.array(dom.classnames, dtype=object),
                "text": zs.encode_text(dom.classnames, PROMPT).numpy().astype(np.float32)}
        for split in ("train", "val", "test"):
            labels = np.array([lab for _, lab in dom.items[split]])
            idx = pool_indices(labels, CAP[split])
            t0 = time.time()
            feats = encode(zs, DataLoader(Images(dom, split, idx), batch_size=128, num_workers=WORKERS))
            for k, v in feats.items():
                data[f"{split}_{k}"] = v
            data[f"{split}_labels"], data[f"{split}_index"] = labels[idx], idx
            log(f"{name} {split}: {len(idx)} images, {len(idx) / (time.time() - t0):.0f} img/s")
        np.savez(path, **data)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--domains", default=",".join(DOMAINS))
    main(p.parse_args().domains.split(","))
