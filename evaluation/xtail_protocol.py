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
            else:
                mask = torch.full_like(top1, j <= t, dtype=torch.bool)
            pad = None
            if mask.any():
                s = torch.as_tensor(method.scores(getf(e, split)[mask.numpy()]), dtype=torch.float32)
                pad = torch.zeros(int(mask.sum()), zs[m].shape[1])
                pad[:, : s.shape[1]] = s
            for w in weights:
                p = zs[m].clone()
                if pad is not None:
                    wv = torch.as_tensor(method.fusion_weight_for(top1[mask].numpy()), dtype=torch.float32)[:, None] if w == "method" else w
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


class RFFRidge:
    """Bounded-memory ML-RAIL. Random Fourier features phi(x) = sqrt(2/D) cos(x W + b),
    W ~ N(0, 2 gamma I), b ~ U(0, 2 pi), approximate the RBF kernel exp(-gamma |x - y|^2); ridge on
    them converges to DualRAIL's kernel ridge as D grows. Only G = Phi^T Phi (D x D) and
    Phi^T Y (D x classes) are kept -- no images stored, memory independent of the stream length,
    exact and order-invariant accumulation."""

    def __init__(self, in_dim, width=8000, gamma=0.2, lam=1e-3, seed=0, chunk=2048, feature_map="rff"):
        """feature_map "relu": Primal-RAIL-style ReLU random projection (weights U(-1/sqrt(in_dim),
        1/sqrt(in_dim)), no bias) instead of random Fourier features; gamma is then unused."""
        g = torch.Generator().manual_seed(seed)
        self.feature_map = feature_map
        if feature_map == "relu":
            self.W = ((torch.rand(in_dim, width, generator=g) * 2 - 1) / in_dim ** 0.5).to(DEV)
            self.b = torch.zeros(width).to(DEV)
        else:
            self.W = (torch.randn(in_dim, width, generator=g) * (2 * gamma) ** 0.5).to(DEV)
            self.b = (torch.rand(width, generator=g) * 2 * np.pi).to(DEV)
        self.width, self.lam, self.chunk = width, lam, chunk
        self.G = torch.zeros(width, width, dtype=torch.float64)  # kept on the CPU (gentle on GPU memory)
        self.C = torch.zeros(width, 0, dtype=torch.float64)
        self.beta = None

    def _phi(self, X):
        X = torch.as_tensor(X, dtype=torch.float32, device=DEV)
        if self.feature_map == "relu":
            return torch.relu(X @ self.W)
        return (2.0 / self.width) ** 0.5 * torch.cos(X @ self.W + self.b)

    def add(self, X, y):
        n_cls = max(int(y.max()) + 1, self.C.shape[1])
        if n_cls > self.C.shape[1]:
            self.C = torch.cat([self.C, torch.zeros(self.width, n_cls - self.C.shape[1], dtype=torch.float64)], 1)
        yt = torch.as_tensor(y)
        for i in range(0, len(X), self.chunk):
            H = self._phi(X[i : i + self.chunk])
            for j in range(0, self.width, 4000):  # column blocks: never a full D x D matrix on the GPU
                self.G[:, j : j + 4000] += (H.T @ H[:, j : j + 4000]).double().cpu()
            self.C.index_add_(1, yt[i : i + self.chunk], H.T.double().cpu())
            del H
        A = self.G.float().clone()
        A.diagonal().add_(self.lam)
        self.beta = torch.linalg.solve(A, self.C.float()).to(DEV)

    def scores(self, X):
        return torch.cat([(self._phi(X[i : i + self.chunk]) @ self.beta).cpu() for i in range(0, len(X), self.chunk)])

    def memory_numbers(self):
        return self.G.numel() + self.C.numel() + self.W.numel() + self.b.numel()


class SumKernelDual(DualRAIL):
    """A4: kernel ridge with the mean of per-block RBF kernels (an RBF on the concatenation
    is instead the *product* of per-block kernels). `slices`: (start, end) columns of each block."""

    def __init__(self, slices, gamma=0.2, lam=1e-3, dtype=torch.float32):
        super().__init__(gamma, lam, dtype)
        self.slices = slices

    def _k(self, A, B):
        return sum(torch.exp(-self.gamma * torch.cdist(A[:, a:b], B[:, a:b]).pow(2)) for a, b in self.slices) / len(self.slices)


def learned_zero_shot(X, text, scale=100.0):
    """Zero-shot probabilities over the given (learned) class names only, from the global block, which
    is the first 512 columns of ML-RAIL's feature (unit block / sqrt(n_blocks); renormalized here)."""
    g = X[:, :512]
    return torch.softmax(scale * (g / g.norm(dim=1, keepdim=True)) @ text.T, dim=1)


def krr_loo(evals, Q, Y, lam):
    """Exact leave-one-out predictions of kernel ridge with K = Q diag(evals) Q^T (one
    eigendecomposition serves every lambda): loo_i = (f_i - h_ii y_i) / (1 - h_ii), H = K (K + lam I)^-1.
    Also returns the dual coefficients (K + lam I)^-1 Y."""
    s = evals / (evals + lam)
    QtY = Q.T @ Y
    F = Q @ (s[:, None] * QtY)
    h = (Q * Q) @ s
    return (F - h[:, None] * Y) / (1 - h)[:, None], Q @ (QtY / (evals + lam)[:, None])


class PerDomainKRR:
    """One RBF kernel ridge model per domain (block-diagonal over domains); each domain's
    settings are chosen when it arrives by exact leave-one-out on its own training images:
    gamma and lambda (A1), or per-block bandwidths gamma * m_b by one coordinate-descent sweep (A2,
    `block_mults`), then optionally its fusion weight (LOO accuracy of RAIL's fused prediction, with
    zero-shot over learned class names only). Ties keep the first candidate (put the default first).
    Scores are the per-domain models' outputs concatenated in domain order."""

    def __init__(self, slices, gammas=(0.2,), lams=(1e-3,), block_mults=None, weights=None, default_weight=0.8, scale=100.0):
        self.slices, self.gammas, self.lams = slices, gammas, lams
        self.block_mults, self.weights, self.default_weight, self.scale = block_mults, weights, default_weight, scale
        self.models, self.class_weight, self.chosen, self.text = [], np.zeros(0, np.float32), [], None

    def set_text(self, text):
        self.text = torch.as_tensor(text, dtype=torch.float64)

    def _colscale(self, gamma, mults):
        c = torch.empty(self.slices[-1][1], dtype=torch.float64)
        for (a, b), m in zip(self.slices, mults):
            c[a:b] = (gamma * m) ** 0.5
        return c

    def add(self, X, y):
        X = torch.as_tensor(X, dtype=torch.float64)
        off, n_cls = int(y.min()), int(y.max() - y.min()) + 1
        yl = torch.as_tensor(y - off)
        Y = torch.zeros(len(y), n_cls, dtype=torch.float64)
        Y[torch.arange(len(y)), yl] = 1
        nb = len(self.slices)

        def evaluate(gamma, mults):
            Xs = X * self._colscale(gamma, mults)
            evals, Q = torch.linalg.eigh(torch.exp(-torch.cdist(Xs, Xs).pow(2)))
            best = None
            for lam in self.lams:
                loo, alpha = krr_loo(evals, Q, Y, lam)
                key = (float((loo.argmax(1) == yl).double().mean()), -float(((loo - Y) ** 2).mean()))
                if best is None or key > best[0]:
                    best = (key, lam, loo, alpha)
            return best

        if self.block_mults is None:
            cands = {(g, (1.0,) * nb): evaluate(g, (1.0,) * nb) for g in self.gammas}
        else:  # one coordinate-descent sweep over the blocks' bandwidth multipliers
            g, m = self.gammas[0], [1.0] * nb
            cur = evaluate(g, tuple(m))
            cands = {(g, tuple(m)): cur}
            for b in range(nb):
                for v in self.block_mults:
                    trial = m.copy()
                    trial[b] = v
                    if v == m[b] or not any(trial):
                        continue
                    r = cands[(g, tuple(trial))] = evaluate(g, tuple(trial))
                    if r[0] > cur[0]:
                        cur, m = r, trial
        (gamma, mults), (key, lam, loo, alpha) = max(cands.items(), key=lambda kv: kv[1][0])
        w = self.default_weight
        if self.weights is not None:
            q = learned_zero_shot(X, self.text, self.scale)
            pad = torch.zeros_like(q)
            pad[:, off : off + n_cls] = loo
            target = yl + off
            w = max(sorted(self.weights, key=lambda v: abs(v - self.default_weight)),
                    key=lambda v: float((((1 - v) * q + v * pad).argmax(1) == target).double().mean()))
        c = self._colscale(gamma, mults)
        self.models.append((c.float().to(DEV), (X * c).float().to(DEV), alpha.float().to(DEV)))
        self.class_weight = np.concatenate([self.class_weight, np.full(n_cls, w, np.float32)])
        self.chosen.append({"gamma": gamma, "mults": list(mults), "lam": lam, "w": w, "loo_acc": key[0]})

    def scores(self, X):
        X = torch.as_tensor(X, dtype=torch.float32, device=DEV)
        out = []
        for i in range(0, len(X), 2048):
            xi = X[i : i + 2048]
            out.append(torch.cat([torch.exp(-torch.cdist(xi * c, Xs).pow(2)) @ a for c, Xs, a in self.models], 1).cpu())
        return torch.cat(out)

    def fusion_weight_for(self, cls):
        return self.class_weight[cls]


class JointLOOWeight(DualRAIL):
    """A1w: ML-RAIL's joint kernel ridge (float64 Cholesky on the CPU), plus a per-domain fusion
    weight chosen when the domain arrives by exact leave-one-out *within the joint model* on that
    domain's training images (zero-shot over learned class names only; ties -> nearest the default)."""

    def __init__(self, gamma=0.2, lam=1e-3, weights=(0.5, 0.6, 0.7, 0.8, 0.9, 1.0), default_weight=0.8, scale=100.0):
        super().__init__(gamma, lam, torch.float64)
        self.weights, self.default_weight, self.scale = weights, default_weight, scale
        self.class_weight, self.chosen = np.zeros(0, np.float32), []

    def set_text(self, text):
        self.text = torch.as_tensor(text, dtype=torch.float64)

    def add(self, X, y):
        X = torch.as_tensor(X, dtype=torch.float32, device=DEV)
        n_old = 0 if self.X is None else len(self.X)
        self.X = X if self.X is None else torch.cat([self.X, X])
        self.y = y if self.y is None else np.concatenate([self.y, y])
        n = len(self.X)
        K = torch.empty(n, n, dtype=torch.float64)
        for i in range(0, n, 1024):
            K[i : i + 1024] = self._k(self.X[i : i + 1024], self.X).double().cpu()
        K.diagonal().add_(self.lam)
        L = torch.linalg.cholesky(K)
        del K
        Y = torch.zeros(n, int(self.y.max()) + 1, dtype=torch.float64)
        Y[torch.arange(n), torch.as_tensor(self.y)] = 1
        alpha = torch.cholesky_solve(Y, L)
        self.alpha = alpha.float().to(DEV)
        # LOO on the new rows: f = K alpha = Y - lam alpha; h_ii = 1 - lam [(K + lam I)^-1]_ii
        E = torch.zeros(n, n - n_old, dtype=torch.float64)
        E[torch.arange(n_old, n), torch.arange(n - n_old)] = 1
        inv_diag = torch.cholesky_solve(E, L)[torch.arange(n_old, n), torch.arange(n - n_old)]
        del L, E
        h = 1 - self.lam * inv_diag
        Yn = Y[n_old:]
        loo = ((Yn - self.lam * alpha[n_old:]) - h[:, None] * Yn) / (1 - h)[:, None]
        q = learned_zero_shot(X.double().cpu(), self.text, self.scale)
        target = torch.as_tensor(y)
        w = max(sorted(self.weights, key=lambda v: abs(v - self.default_weight)),
                key=lambda v: float((((1 - v) * q + v * loo).argmax(1) == target).double().mean()))
        n_cls = int(y.max() - y.min()) + 1
        self.class_weight = np.concatenate([self.class_weight, np.full(n_cls, w, np.float32)])
        self.chosen.append({"w": w, "loo_acc": float((loo.argmax(1) == target).double().mean())})

    def fusion_weight_for(self, cls):
        return self.class_weight[cls]


class ConformalGate:
    """RAIL's gate, plus a rescue of rejected images: admitted if, for some learned domain
    d, the max kernel similarity to d's stored training features is >= tau_d, the q-quantile of d's
    own training images' max similarity to the other training images of d (exact duplicates from
    with-replacement sampling excluded), and the zero-shot mass on learned classes is >= min_mass.
    Needs a method with stored features X, labels y and a kernel _k (DualRAIL)."""

    def __init__(self, q, min_mass=0.0):
        self.q, self.min_mass, self.tau = q, min_mass, {}

    def _rows(self, method, off, n):
        y = torch.as_tensor(method.y, device=method.X.device)
        return method.X[(y >= off) & (y < off + n)]

    def __call__(self, method, X, p, learned, domains):
        mask = p.argmax(1) < learned
        rej = (~mask) & (p[:, :learned].sum(1) >= self.min_mass)
        if not rej.any():
            return mask
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
        return mask


class Cascade:
    """Cascade routing: the router (ML-RAIL, joint) picks each image's domain as that of its top learned
    class; the expert's scores for that domain's classes are returned, every other learned domain's
    columns are set to `suppress` (so the fused prediction stays inside the routed domain)."""

    def __init__(self, router, expert, suppress=-10.0):
        self.router, self.expert, self.suppress, self.bounds = router, expert, suppress, []

    def set_text(self, text):
        for m in (self.router, self.expert):
            if hasattr(m, "set_text"):
                m.set_text(text)

    def add(self, X, y):
        self.bounds.append((int(y.min()), int(y.max()) + 1))
        self.router.add(X, y)
        if self.expert is not self.router:
            self.expert.add(X, y)

    def scores(self, X):
        r = torch.as_tensor(self.router.scores(X), dtype=torch.float32)
        e = r if self.expert is self.router else torch.as_tensor(self.expert.scores(X), dtype=torch.float32)
        dom = torch.zeros(r.shape[1], dtype=torch.long)
        for k, (a, b) in enumerate(self.bounds):
            dom[a:b] = k
        routed = dom[r.argmax(1)]
        keep = routed[:, None] == dom[None, :]
        return torch.where(keep, e, torch.tensor(self.suppress))


class GatedResidual(DualRAIL):
    """A4: inside RAIL's gate, kernel ridge fits what zero-shot gets wrong -- targets
    Y - zero-shot probabilities over the learned class names (recomputed as the vocabulary grows) --
    and scores = those zero-shot probabilities + the learned correction."""

    def set_text(self, text):
        self.text = torch.as_tensor(text, dtype=torch.float32, device=DEV)

    def _q(self, X):
        return torch.cat([learned_zero_shot(X[i : i + 4096].to(DEV), self.text) for i in range(0, len(X), 4096)])

    def _targets(self, Y):
        return Y - self._q(self.X).to(Y.device, Y.dtype)

    def scores(self, X):
        X = torch.as_tensor(X, dtype=torch.float32, device=DEV)
        return self._q(X).cpu() + super().scores(X)
