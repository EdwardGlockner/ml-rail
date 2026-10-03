"""Kill test (validation): gate-free residual kernel ridge on a zero-shot
prior, vs. ML-RAIL with RAIL's gate, same seed (100), features (ML-RAIL's blocks) and text head (RAIL's prompt).

    score(x) = c * z(x) + k(x, X) (K + lambda I)^-1 (Y - c * z(X))

z = zero-shot probabilities over all classes; Y one-hot over all classes. Two solves per step,
A^-1 Y and A^-1 Z, give every prior scale c; the gated ML-RAIL baseline uses A^-1 Y on learned columns
with RAIL's fusion (weight 0.8), computed in the same pass.

Usage:
    python scripts/run_xtail_residual.py
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from evaluation import xtail_protocol as xp  # noqa: E402
from scripts.run_xtail_improve import blocks_feature, load_with_extra  # noqa: E402

OUT = REPO_ROOT / "data" / "generated" / "xtail"
GAMMAS = (0.2, 0.5, 1.0, 2.0, 4.0)
LAMS = (1e-3, 1e-2)
CS = (0.5, 1.0, 2.0)
APRIME = ("global", "cls10", "patch_mean", "gem8", "gem10")
SEED = 100


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def kernel(A, B, gamma):
    """RBF kernel, rows of A vs. rows of B, built on the GPU in chunks, returned on the CPU."""
    Bd = torch.from_numpy(B).to(xp.DEV)
    out = torch.empty(len(A), len(B))
    for i in range(0, len(A), 1024):
        a = torch.from_numpy(A[i : i + 1024]).to(xp.DEV)
        out[i : i + 1024] = torch.exp(-gamma * torch.cdist(a, Bd).pow(2)).cpu()
    return out


def main(split="val", w=0.8):
    torch.set_num_threads(2)
    data = load_with_extra("single")
    names = list(data)
    feat = blocks_feature(APRIME)
    text = torch.from_numpy(np.concatenate([data[n]["text"] for n in names]))
    C = len(text)
    shot = {n: xp.fewshot(data[n], SEED, 16, t) for t, n in enumerate(names)}
    F_tr = {n: feat(data[n], "train", shot[n]) for n in names}
    F_ev = {n: feat(data[n], split) for n in names}
    z_tr = {n: torch.softmax(100 * torch.from_numpy(data[n]["train_global"][shot[n]]) @ text.T, 1) for n in names}
    z_ev = {n: torch.softmax(100 * torch.from_numpy(data[n][f"{split}_global"]) @ text.T, 1) for n in names}
    y_ev = {n: data[n][f"{split}_labels"] + data[n]["offset"] for n in names}
    results = []
    for gamma, lam in itertools.product(GAMMAS, LAMS):
        t0 = time.time()
        tables = {("residual", c): np.zeros((10, 10)) for c in CS}
        tables[("gated", None)] = np.zeros((10, 10))
        learned = 0
        for t, n in enumerate(names):
            learned += data[n]["n_classes"]
            X = np.concatenate([F_tr[m] for m in names[: t + 1]])
            y = np.concatenate([data[m]["train_labels"][shot[m]] + data[m]["offset"] for m in names[: t + 1]])
            Y = torch.zeros(len(X), C)
            Y[torch.arange(len(X)), torch.from_numpy(y)] = 1
            Z = torch.cat([z_tr[m] for m in names[: t + 1]])
            K = kernel(X, X, gamma)
            K.diagonal().add_(lam)
            sol = torch.linalg.solve(K, torch.cat([Y, Z], 1))
            aY, aZ = sol[:, :C], sol[:, C:]
            del K, sol
            for j, e in enumerate(names):
                kx = kernel(F_ev[e], X, gamma)
                sY, sZ = kx @ aY, kx @ aZ
                for c in CS:
                    s = c * z_ev[e] + sY - c * sZ
                    tables[("residual", c)][t, j] = 100 * float((s.argmax(1).numpy() == y_ev[e]).mean())
                p = z_ev[e].clone()  # ML-RAIL with RAIL's gate and fusion
                mask = p.argmax(1) < learned
                pad = torch.zeros_like(p)
                pad[:, :learned] = sY[:, :learned]
                p[mask] = (1 - w) * p[mask] + w * pad[mask]
                tables[("gated", None)][t, j] = 100 * float((p.argmax(1).numpy() == y_ev[e]).mean())
        for (kind, c), tab in tables.items():
            results.append({"kind": kind, "gamma": gamma, "lam": lam, "c": c, **xp.metrics(tab)})
        json.dump(results, open(OUT / "residual_val.json", "w"), indent=1)
        r = [x for x in results if x["gamma"] == gamma and x["lam"] == lam]
        best = max((x for x in r if x["kind"] == "residual"), key=lambda x: x["average"] + x["last"])
        gated = next(x for x in r if x["kind"] == "gated")
        log(f"gamma {gamma} lam {lam}: residual best c {best['c']} -> {best['transfer']:.2f} / {best['average']:.2f} / {best['last']:.2f}"
            f" | gated {gated['transfer']:.2f} / {gated['average']:.2f} / {gated['last']:.2f}  [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
