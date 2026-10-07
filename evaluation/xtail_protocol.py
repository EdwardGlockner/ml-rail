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

import os

import numpy as np
import torch

from data.real.xtail import DOMAINS, XTAIL_ROOT

FEATS = (XTAIL_ROOT / "features/vitb16")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
# "Gentle mode": cap this process's share of GPU memory (e.g. XTAIL_GPU_FRACTION=0.6) so the
# machine stays usable; unset = no cap.
if DEV == "cuda" and os.environ.get("XTAIL_GPU_FRACTION"):
    torch.cuda.set_per_process_memory_fraction(float(os.environ["XTAIL_GPU_FRACTION"]))
# Gentle mode, continued: XTAIL_SOLVE=cpu moves kernel-ridge solves to the CPU (2 threads), so the GPU
# only builds kernel chunks (needed when other programs hold GPU memory).
SOLVE_ON_CPU = os.environ.get("XTAIL_SOLVE") == "cpu"
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


def run_stream(data, method, seed, feat="global", fusion_weight=0.8, shots=16, zs=None, split="test", gate="rail", route=False):
    """method: object with .add(X, y_global) and .scores(X) -> (N, n_learned) numpy/tensor.
    `feat`: which cached feature (or callable d, split, idx -> array) the method sees.
    `split`: "test" for reported results, "val" for choosing hyperparameters (CoOp val splits).
    `fusion_weight`: a float, a list (-> dict of tables), or "method" (per-image weight from
    method.fusion_weight_for(zero-shot top-1 class)).
    `gate`: "rail" (zero-shot top-1 among learned classes); "oracle" (the adapter
    applies exactly to images of learned domains); "oracle_domain" (as "oracle", and the prediction of a
    learned-domain image is restricted to its true domain's classes -- an upper bound on any routing);
    looser gates "mass:<tau>" and "topk:<k>" (supersets of RAIL's gate); or a callable
    gate(method, X, zs_probs, learned, learned_domains) -> bool mask, learned_domains =
    [(offset, n_classes), ...].
    `route`: each admitted image's prediction is restricted to the domain of its top
    learned zero-shot class.
    A method with .set_text(text) receives the text embeddings of the learned classes (incl. the new
    domain) before each .add -- never future class names."""
    names = list(data)
    cls_dom = np.concatenate([np.full(data[n]["n_classes"], k) for k, n in enumerate(names)])
    zs = zs if zs is not None else zero_shot_probs(data, split=split)
    weights = list(fusion_weight) if isinstance(fusion_weight, (list, tuple)) else [fusion_weight]
    tables = {w: np.zeros((len(names), len(names))) for w in weights}
    getf = feat if callable(feat) else (lambda d, split, idx=None: d[f"{split}_{feat}"] if idx is None else d[f"{split}_{feat}"][idx])
    text = np.concatenate([data[n]["text"] for n in names])
    learned = 0
    for t, n in enumerate(names):
        d = data[n]
        idx = fewshot(d, seed, shots, t)
        if hasattr(method, "set_text"):
            method.set_text(text[: learned + d["n_classes"]])
        method.add(getf(d, "train", idx), d["train_labels"][idx] + d["offset"])
        learned += d["n_classes"]
        for j, m in enumerate(names):
            e = data[m]
            top1 = zs[m].argmax(1)
            if callable(gate):
                mask = gate(method, getf(e, split), zs[m], learned,
                            [(data[k]["offset"], data[k]["n_classes"]) for k in names[: t + 1]])
            elif gate == "rail":
                mask = top1 < learned
            elif gate.startswith("mass:"):  # RAIL's gate or zero-shot mass on learned classes >= tau
                mask = (top1 < learned) | (zs[m][:, :learned].sum(1) >= float(gate[5:]))
            elif gate.startswith("topk:"):  # Any of the zero-shot top-k classes is learned
                mask = (zs[m].topk(int(gate[5:]), dim=1).indices < learned).any(1)
            elif gate in ("oracle", "oracle_domain"):
                mask = torch.full_like(top1, j <= t, dtype=torch.bool)
            else:
                raise ValueError(f"unknown gate: {gate!r}")
            pad = None
            if mask.any():
                s = torch.as_tensor(method.scores(getf(e, split)[mask.numpy()]), dtype=torch.float32)
                pad = torch.zeros(int(mask.sum()), zs[m].shape[1])
                pad[:, : s.shape[1]] = s
            for w in weights:
                p = zs[m].clone()
                if pad is not None:
                    if isinstance(w, tuple):  # (w, w_r): images the gate rescued get weight w_r
                        wv = torch.full((int(mask.sum()), 1), float(w[0]))
                        wv[gate.last_rescued[mask]] = float(w[1])
                    elif w == "method":
                        wv = torch.as_tensor(method.fusion_weight_for(top1[mask].numpy()), dtype=torch.float32)[:, None]
                    else:
                        wv = w
                    p[mask] = (1 - wv) * p[mask] + wv * pad
                if route and mask.any():  # Restrict admitted images to one learned domain
                    dom = torch.as_tensor(cls_dom[zs[m][mask][:, :learned].argmax(1).numpy()])
                    keep = dom[:, None] == torch.as_tensor(cls_dom)[None, :]
                    p[mask] = torch.where(keep, p[mask], torch.tensor(-1.0))
                if gate == "oracle_domain" and j <= t:
                    p = p[:, e["offset"] : e["offset"] + e["n_classes"]]
                    hit = p.argmax(1).numpy() == e[f"{split}_labels"]
                else:
                    hit = p.argmax(1).numpy() == e[f"{split}_labels"] + e["offset"]
                tables[w][t, j] = 100.0 * float(hit.mean())
    return tables if isinstance(fusion_weight, (list, tuple)) else tables[weights[0]]


PROMPT_ENSEMBLE = (  # Tip-Adapter's 7 templates + RAIL's single prompt
    "a photo of a {}.", "itap of a {}.", "a bad photo of the {}.", "a origami {}.",
    "a photo of the large {}.", "a {} in a video game.", "art of the {}.", "a photo of the small {}.",
)


def ensemble_text(data, cache=(XTAIL_ROOT / "features/vitb16_text_ensemble.npz")):
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
        if SOLVE_ON_CPU:  # low-load mode: build K in GPU chunks into CPU memory, solve on the CPU (same math)
            n = len(self.X)
            K = torch.empty(n, n, dtype=self.dtype)
            for i in range(0, n, 1024):
                K[i : i + 1024] = self._k(self.X[i : i + 1024], self.X).to(self.dtype).cpu()
            K.diagonal().add_(self.lam)
            Y = torch.zeros(n, int(self.y.max()) + 1, dtype=self.dtype)
            Y[torch.arange(n), torch.as_tensor(self.y)] = 1
            self.alpha = torch.linalg.solve(K, self._targets(Y)).float().to(DEV)
            return
        Y = torch.zeros(len(self.y), int(self.y.max()) + 1, device=DEV, dtype=self.dtype)
        Y[torch.arange(len(self.y)), torch.as_tensor(self.y, device=DEV)] = 1
        K = self._k(self.X, self.X).to(self.dtype)
        self.alpha = torch.linalg.solve(K + self.lam * torch.eye(len(K), device=DEV, dtype=self.dtype), self._targets(Y)).float()

    def _targets(self, Y):
        return Y

    def scores(self, X):
        X = torch.as_tensor(X, dtype=torch.float32, device=DEV)
        return torch.cat([(self._k(X[i : i + 2048], self.X) @ self.alpha).cpu() for i in range(0, len(X), 2048)])


class DomainThresholdGate:
    """The rescue gate: RAIL's gate, plus images it rejects that look like a learned domain.

    A rejected image is admitted if its zero-shot mass on learned classes is >= min_mass and, for some
    learned domain d, its max kernel similarity to d's stored training features is >= tau_d. tau_d is
    the q-quantile of the nearest-neighbour similarities among d's own training images (each image
    against the others; exact duplicates from with-replacement sampling excluded). This is an empirical
    per-domain threshold; it carries no coverage guarantee.
    Needs a method with stored features X, labels y and a kernel _k (DualRAIL)."""

    def __init__(self, q, min_mass=0.0, strip=0):
        """strip: number of trailing non-feature columns in the test features (image ids)."""
        self.q, self.min_mass, self.strip, self.tau, self._owner = q, min_mass, strip, {}, None

    def _rows(self, method, off, n):
        y = torch.as_tensor(method.y, device=method.X.device)
        return method.X[(y >= off) & (y < off + n)]

    def __call__(self, method, X, p, learned, domains):
        if self._owner is not method:  # thresholds belong to one model's stored features
            self.tau, self._owner = {}, method
        mask = p.argmax(1) < learned
        rej = (~mask) & (p[:, :learned].sum(1) >= self.min_mass)
        self.last_rescued = torch.zeros_like(mask)  # Which admitted images were rescued
        if not rej.any():
            return mask
        X = X[:, : X.shape[1] - self.strip] if self.strip else X
        Xr = torch.as_tensor(X[rej.numpy()], dtype=torch.float32, device=DEV)
        ok = torch.zeros(len(Xr), dtype=torch.bool)
        for off, n in domains:
            Xd = self._rows(method, off, n)
            if off not in self.tau:
                nn = []
                for i in range(0, len(Xd), 2048):
                    K = method._k(Xd[i : i + 2048], Xd)
                    K[K >= 1 - 1e-6] = -1  # self and exact duplicates
                    nn.append(K.max(1).values.cpu())
                self.tau[off] = float(torch.quantile(torch.cat(nn), self.q))
            s = torch.cat([method._k(Xr[i : i + 2048], Xd).max(1).values.cpu() for i in range(0, len(Xr), 2048)])
            ok |= s >= self.tau[off]
        mask = mask.clone()
        mask[rej] = ok
        self.last_rescued[rej] = ok
        return mask
