"""The X-TAIL protocol (Xu et al., NeurIPS 2024), reimplemented on cached CLIP features.

After learning domains 0..t, every domain's test set is classified over the union of *all* domains'
class names (seen and unseen), giving a 10 x 10 accuracy table acc[t, d]. Metrics exactly as in
RAIL's code: Transfer = mean over columns d >= 1 of mean_{t < d} acc[t, d]; Average = mean over
columns of the column mean over all rows; Last = mean of the last row.

RAIL's fusion: zero-shot probabilities p = softmax(100 cos(x, text)) over all classes; if the
zero-shot top-1 class is one already learned, p <- (1 - w) p + w * adapter scores (zero-padded
for unlearned classes); otherwise p is kept (so unseen-domain zero-shot accuracy is preserved).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from data.real.xtail import DOMAINS

FEATS = Path("D:/xtail_data/features/vitb16")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
# "Gentle mode": cap this process's share of GPU memory (e.g. XTAIL_GPU_FRACTION=0.6) so the
# machine stays usable; unset = no cap.
if DEV == "cuda" and __import__("os").environ.get("XTAIL_GPU_FRACTION"):
    torch.cuda.set_per_process_memory_fraction(float(__import__("os").environ["XTAIL_GPU_FRACTION"]))
# Gentle mode, continued: XTAIL_SOLVE=cpu moves kernel-ridge solves to the CPU (2 threads), so the GPU
# only builds kernel chunks (needed when other programs hold GPU memory).
SOLVE_ON_CPU = __import__("os").environ.get("XTAIL_SOLVE") == "cpu"
if SOLVE_ON_CPU:
    torch.set_num_threads(2)


def load_features(domains=DOMAINS, feat_dir=FEATS):
    data, offset = {}, 0
    for name in domains:
        z = np.load(feat_dir / f"{name}.npz", allow_pickle=True)
        d = {k: z[k] for k in z.files}
        d["offset"] = offset
        d["n_classes"] = len(d["classnames"])
        offset += d["n_classes"]
        data[name] = d
    return data


def fewshot(d, seed, shots=16, domain_index=0):
    """Indices into the train pool: `shots` per class, drawn uniformly (with replacement only if a
    class has fewer images, as RAIL's CoOp loader does)."""
    rng = np.random.default_rng(seed * 1000 + domain_index)
    y = d["train_labels"]
    idx = []
    for c in range(d["n_classes"]):
        where = np.where(y == c)[0]
        idx.append(rng.choice(where, size=shots, replace=len(where) < shots))
    return np.concatenate(idx)


def metrics(table):
    n = table.shape[0]
    transfer = np.mean([table[:d, d].mean() for d in range(1, n)])
    return {"transfer": float(transfer), "average": float(table.mean(0).mean()), "last": float(table[-1].mean())}


def zero_shot_probs(data, scale=100.0, split="test"):
    text = torch.from_numpy(np.concatenate([data[n]["text"] for n in data])).to(DEV)
    out = {}
    for n, d in data.items():
        x = torch.from_numpy(d[f"{split}_global"]).to(DEV)
        out[n] = torch.softmax(scale * x @ text.T, dim=1).cpu()
    return out


def run_stream(data, method, seed, feat="global", fusion_weight=0.8, shots=16, zs=None, split="test"):
    """method: object with .add(X, y_global) and .scores(X) -> (N, n_learned) numpy/tensor.
    `feat`: which cached feature (or callable d, split, idx -> array) the method sees.
    `split`: "test" for reported results, "val" for choosing hyperparameters (CoOp val splits)."""
    names = list(data)
    zs = zs if zs is not None else zero_shot_probs(data, split=split)
    weights = list(fusion_weight) if isinstance(fusion_weight, (list, tuple)) else [fusion_weight]
    tables = {w: np.zeros((len(names), len(names))) for w in weights}
    getf = feat if callable(feat) else (lambda d, split, idx=None: d[f"{split}_{feat}"] if idx is None else d[f"{split}_{feat}"][idx])
    learned = 0
    for t, n in enumerate(names):
        d = data[n]
        idx = fewshot(d, seed, shots, t)
        method.add(getf(d, "train", idx), d["train_labels"][idx] + d["offset"])
        learned += d["n_classes"]
        for j, m in enumerate(names):
            e = data[m]
            mask = zs[m].argmax(1) < learned
            pad = None
            if mask.any():
                s = torch.as_tensor(method.scores(getf(e, split)[mask.numpy()]), dtype=torch.float32)
                pad = torch.zeros(int(mask.sum()), zs[m].shape[1])
                pad[:, : s.shape[1]] = s
            for w in weights:
                p = zs[m].clone()
                if pad is not None:
                    p[mask] = (1 - w) * p[mask] + w * pad
                tables[w][t, j] = 100.0 * float((p.argmax(1).numpy() == e[f"{split}_labels"] + e["offset"]).mean())
    return tables if isinstance(fusion_weight, (list, tuple)) else tables[weights[0]]


class Calibrated:
    """A1: wraps a method so its scores are probabilities, softmax(scores / temperature),
    before RAIL's fusion mixes them with the zero-shot probabilities."""

    def __init__(self, method, temperature):
        self.method, self.temperature = method, temperature

    def add(self, X, y):
        self.method.add(X, y)

    def scores(self, X):
        s = torch.as_tensor(self.method.scores(X), dtype=torch.float32)
        return torch.softmax(s / self.temperature, dim=1)


PROMPT_ENSEMBLE = (  # Tip-Adapter's 7 templates + RAIL's single prompt
    "a photo of a {}.", "itap of a {}.", "a bad photo of the {}.", "a origami {}.",
    "a photo of the large {}.", "a {} in a video game.", "art of the {}.", "a photo of the small {}.",
)


def ensemble_text(data, cache=Path("D:/xtail_data/features/vitb16_text_ensemble.npz")):
    """Replace each domain's single-prompt text embedding by the normalized mean over PROMPT_ENSEMBLE."""
    out = dict(np.load(cache)) if cache.exists() else {}
    missing = [n for n in data if n not in out]
    if missing:  # compute only the domains not cached yet (e.g. cifar100 for the 11-domain variant)
        from features.clip_zero_shot import ClipZeroShotClassifier
        zs = ClipZeroShotClassifier("openai/clip-vit-base-patch16")
        for n in missing:
            embs = torch.stack([zs.encode_text(list(data[n]["classnames"]), t) for t in PROMPT_ENSEMBLE]).mean(0)
            out[n] = (embs / embs.norm(dim=-1, keepdim=True)).numpy().astype(np.float32)
        np.savez(cache, **out)
    z = out
    for n, d in data.items():
        d["text"] = z[n]
    return data


class ZeroShot:
    def add(self, X, y):
        pass

    def scores(self, X):
        return np.zeros((len(X), 0))


class PrimalRAIL:
    """Random ReLU expansion (torch nn.Linear default init, no bias) + ridge (lambda) on one-hot
    targets. RAIL's recursive update equals this joint solution exactly."""

    def __init__(self, width=15000, reg=0.1, seed=42, in_dim=512):
        g = torch.Generator().manual_seed(seed)
        bound = 1.0 / np.sqrt(in_dim)  # kaiming_uniform(a=sqrt(5)) of nn.Linear == U(-1/sqrt(fan_in), 1/sqrt(fan_in))
        self.W = ((torch.rand(in_dim, width, generator=g) * 2 - 1) * bound).to(DEV)
        self.reg, self.G, self.C, self.n = reg, torch.zeros(width, width, device=DEV, dtype=torch.float64), None, 0
        self.beta = None

    def _h(self, X):
        return torch.relu(torch.as_tensor(X, dtype=torch.float32, device=DEV) @ self.W)

    def add(self, X, y):
        H = self._h(X).double()
        n_new = int(y.max()) + 1
        C = torch.zeros(H.shape[1], n_new, device=DEV, dtype=torch.float64)
        if self.C is not None:
            C[:, : self.C.shape[1]] = self.C
        C.index_add_(1, torch.as_tensor(y, device=DEV), H.T)
        self.G += H.T @ H
        self.C = C
        self.beta = torch.linalg.solve(self.G + self.reg * torch.eye(self.G.shape[0], device=DEV, dtype=torch.float64), self.C).float()

    def scores(self, X):
        out = [(self._h(X[i : i + 4096]) @ self.beta).cpu() for i in range(0, len(X), 4096)]
        return torch.cat(out)


class DualRAIL:
    """Kernel ridge regression, RBF kernel exp(-gamma |x - x'|^2) over all stored training features."""

    def __init__(self, gamma=0.1, lam=0.001, dtype=torch.float64):
        self.gamma, self.lam, self.dtype = gamma, lam, dtype  # float64 as RAIL; float32 only for fast searches
        self.X, self.y = None, None

    def _k(self, A, B):
        return torch.exp(-self.gamma * torch.cdist(A, B).pow(2))

    def add(self, X, y):
        X = torch.as_tensor(X, dtype=torch.float32, device=DEV)
        self.X = X if self.X is None else torch.cat([self.X, X])
        self.y = y if self.y is None else np.concatenate([self.y, y])
        if SOLVE_ON_CPU:  # gentle mode: build K in GPU chunks into CPU memory, solve on the CPU (same math)
            n = len(self.X)
            K = torch.empty(n, n, dtype=self.dtype)
            for i in range(0, n, 1024):
                K[i : i + 1024] = self._k(self.X[i : i + 1024], self.X).to(self.dtype).cpu()
            K.diagonal().add_(self.lam)
            Y = torch.zeros(n, int(self.y.max()) + 1, dtype=self.dtype)
            Y[torch.arange(n), torch.as_tensor(self.y)] = 1
            self.alpha = torch.linalg.solve(K, Y).float().to(DEV)
            return
        Y = torch.zeros(len(self.y), int(self.y.max()) + 1, device=DEV, dtype=self.dtype)
        Y[torch.arange(len(self.y)), torch.as_tensor(self.y, device=DEV)] = 1
        K = self._k(self.X, self.X).to(self.dtype)
        self.alpha = torch.linalg.solve(K + self.lam * torch.eye(len(K), device=DEV, dtype=self.dtype), Y).float()

    def scores(self, X):
        X = torch.as_tensor(X, dtype=torch.float32, device=DEV)
        return torch.cat([(self._k(X[i : i + 2048], self.X) @ self.alpha).cpu() for i in range(0, len(X), 2048)])


class SumKernelDual(DualRAIL):
    """A4: kernel ridge with the mean of per-block RBF kernels (an RBF on the concatenation
    is instead the *product* of per-block kernels). `slices`: (start, end) columns of each block."""

    def __init__(self, slices, gamma=0.2, lam=1e-3, dtype=torch.float32):
        super().__init__(gamma, lam, dtype)
        self.slices = slices

    def _k(self, A, B):
        return sum(torch.exp(-self.gamma * torch.cdist(A[:, a:b], B[:, a:b]).pow(2)) for a, b in self.slices) / len(self.slices)
