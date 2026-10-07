"""Variants of the X-TAIL adapter and gate that were tested and did not improve on ML-RAIL with the
rescue gate (see docs/RESEARCH_LOG.md). Kept so that every reported experiment can be rerun."""
from __future__ import annotations

import numpy as np
import torch

from evaluation.xtail_protocol import DEV, SOLVE_ON_CPU, DualRAIL  # noqa: F401


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


class SumKernelDual(DualRAIL):
    """A4: kernel ridge with the mean of per-block RBF kernels (an RBF on the concatenation
    is instead the *product* of per-block kernels). `slices`: (start, end) columns of each block."""

    def __init__(self, slices, gamma=0.2, lam=1e-3, dtype=torch.float32):
        super().__init__(gamma, lam, dtype)
        self.slices = slices

    def _k(self, A, B):
        return sum(torch.exp(-self.gamma * torch.cdist(A[:, a:b], B[:, a:b]).pow(2)) for a, b in self.slices) / len(self.slices)


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
        self.G = torch.zeros(width, width, dtype=torch.float64)  # kept on the CPU to save GPU memory
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
