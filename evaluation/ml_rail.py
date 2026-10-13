"""ML-RAIL: RAIL's dual (kernel ridge) adapter on multi-layer frozen CLIP ViT-B/16 features.

Five blocks from one forward pass -- the final embedding, the layer-10 class token, the mean patch token
and GeM-pooled patch tokens of layers 8 and 10 -- are each L2-normalized, concatenated and rescaled to
unit length. The multi-layer idea follows LayUP (Ahrens et al.); RAIL is Xu et al., NeurIPS 2024.
"""
from __future__ import annotations

import numpy as np
import torch

from data.real.xtail import DOMAINS, FEATURES
from evaluation import xtail_protocol as xp

BLOCKS = ("global", "cls10", "patch_mean", "gem8", "gem10")
DIMS = [512 if b in ("global", "patch_mean") else 768 for b in BLOCKS]
_EDGES = np.cumsum([0] + DIMS)
SLICES = [(int(_EDGES[i]), int(_EDGES[i + 1])) for i in range(len(DIMS))]
EXTRA = FEATURES / "vitb16_extra"  # flipped-view and GeM features (scripts/xtail_encode_extra.py)


def l2(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def load_with_extra(prompts="single", domains=DOMAINS, backbone="vitb16"):
    """Cached features of every domain, including the extra blocks; `prompts="ensemble"` swaps in the
    prompt-ensemble text head. Another `backbone` loads FEATURES/<backbone> (scripts/xtail_encode_backbone.py),
    whose files hold every block."""
    if backbone != "vitb16":
        return xp.load_features(domains, FEATURES / backbone)
    data = xp.load_features(domains)
    if prompts == "ensemble":
        xp.ensemble_text(data)
    for n, d in data.items():
        z = np.load(EXTRA / f"{n}.npz")
        d.update({k: z[k] for k in z.files})
    return data


def block_slices(data, blocks=BLOCKS):
    """Column range of each block in blocks_feature(blocks)'s output, from the cached feature widths."""
    d = next(iter(data.values()))
    edges = np.cumsum([0] + [d[f"train_{b}"].shape[1] for b in blocks])
    return [(int(edges[i]), int(edges[i + 1])) for i in range(len(blocks))]


def blocks_feature(blocks=BLOCKS, flip_train=False, flip_test=False):
    """Concatenation of unit-normalized blocks (rescaled to unit length). With flips, a block that has a
    flipped counterpart is replaced by the normalized mean of original and flipped features."""
    def f(d, split, idx=None):
        out = []
        for b in blocks:
            x = d[f"{split}_{b}"]
            use_flip = (flip_train if split == "train" else flip_test) and f"{split}_flip_{b}" in d
            if use_flip:
                x = l2(x) + l2(d[f"{split}_flip_{b}"])
            out.append(l2(x if idx is None else x[idx]))
        return (np.concatenate(out, 1) / np.sqrt(len(out))).astype(np.float32)
    return f


def make_ml_rail(dtype=torch.float32):
    """ML-RAIL's adapter: RBF kernel ridge with gamma 0.2 and lambda 1e-3 (chosen on validation)."""
    return xp.DualRAIL(gamma=0.2, lam=1e-3, dtype=dtype)
