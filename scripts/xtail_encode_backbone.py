"""Encode the X-TAIL domains with another CLIP backbone, on exactly the images in the ViT-B/16 cache
(same training pools, validation subsets and full test splits), with every ML-RAIL block in one pass.

Blocks are taken at the same relative depths as for ViT-B/16 (12 layers): the class token at 10/12 of the
depth and GeM-pooled patch tokens at 8/12 and 10/12; for ViT-L/14 (24 layers) that is layers 20 and 16.
They are stored under the ViT-B/16 block names (cls10, gem8, gem10), so the rest of the code is unchanged;
`layers` in each file records the actual layers. Also stored: the projected global embedding, the mean
projected patch token, and the class-name text embeddings with RAIL's prompt.

Usage:
    python scripts/xtail_encode_backbone.py [--model openai/clip-vit-large-patch14] [--tag vitl14]
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

from data.real.xtail import DOMAINS_11, XTAIL_ROOT, load_domain  # noqa: E402
from scripts.xtail_encode import OUT as B16_DIR, PROMPT, WORKERS, Images  # noqa: E402


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def norm(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


@torch.no_grad()
def encode(zs, loader, cls_layer, gem_layers):
    vm, proj = zs.model.vision_model, zs.model.visual_projection
    out = {k: [] for k in ("global", "patch_mean", "cls10", "gem8", "gem10")}
    for batch in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            o = vm(pixel_values=zs._preprocess_images(batch.numpy()), output_hidden_states=True)
            g = proj(o.pooler_output).float()
            p = norm(proj(vm.post_layernorm(o.last_hidden_state[:, 1:])).float())
        out["global"].append(norm(g).cpu())
        out["patch_mean"].append(norm(p.mean(1)).cpu())
        out["cls10"].append(norm(vm.post_layernorm(o.hidden_states[cls_layer][:, 0]).float()).cpu())
        for name, layer in zip(("gem8", "gem10"), gem_layers):
            t = vm.post_layernorm(o.hidden_states[layer][:, 1:]).float()
            out[name].append(t.clamp_min(0).pow(3).mean(1).pow(1 / 3).cpu())
    return {k: torch.cat(v).numpy().astype(np.float32) for k, v in out.items()}


def main(model, tag, domains, batch_size):
    import os

    from features.clip_zero_shot import ClipZeroShotClassifier
    if torch.cuda.is_available() and os.environ.get("XTAIL_GPU_FRACTION"):  # low-load setting, as elsewhere
        torch.cuda.set_per_process_memory_fraction(float(os.environ["XTAIL_GPU_FRACTION"]))
    out_dir = XTAIL_ROOT / "features" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    zs = ClipZeroShotClassifier(model)
    depth = zs.model.vision_model.config.num_hidden_layers
    cls_layer, gem_layers = round(depth * 10 / 12), (round(depth * 8 / 12), round(depth * 10 / 12))
    log(f"{model}: {depth} layers; class token at layer {cls_layer}, GeM at layers {gem_layers}")
    for name in domains:
        path = out_dir / f"{name}.npz"
        if path.exists():
            log(f"{name}: cached")
            continue
        dom = load_domain(name)
        ref = np.load(B16_DIR / f"{name}.npz", allow_pickle=True)
        data = {"classnames": ref["classnames"], "layers": np.array([cls_layer, *gem_layers]),
                "text": zs.encode_text(list(ref["classnames"]), PROMPT).numpy().astype(np.float32)}
        for split in ("train", "val", "test"):
            idx = ref[f"{split}_index"]
            t0 = time.time()
            feats = encode(zs, DataLoader(Images(dom, split, idx), batch_size=batch_size, num_workers=WORKERS),
                           cls_layer, gem_layers)
            data.update({f"{split}_{k}": v for k, v in feats.items()})
            data[f"{split}_labels"], data[f"{split}_index"] = ref[f"{split}_labels"], idx
            log(f"{name} {split}: {len(idx)} images, {len(idx) / (time.time() - t0):.0f} img/s")
        np.savez(path, **data)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="openai/clip-vit-large-patch14")
    p.add_argument("--tag", default="vitl14")
    p.add_argument("--domains", default=",".join(DOMAINS_11))
    p.add_argument("--batch-size", type=int, default=32)
    a = p.parse_args()
    main(a.model, a.tag, a.domains.split(","), a.batch_size)
