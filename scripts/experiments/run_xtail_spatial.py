"""Phase 1 (rule written before any result): which
spatial pooling of frozen CLIP ViT-B/16 patch tokens best separates fine-grained classes?

Within-domain 16-shot classification on Aircraft, Flowers, Food101, Pets, Cars (each domain's own
classes; no fusion, isolating the representation). Each pooling is concatenated with the global
embedding (blocks unit-normalized, concatenation rescaled to unit length) and read out with RBF
kernel ridge; gamma and lambda chosen on validation per pooling (mean over domains). Test = fixed
subset of <= 25 images per class (diagnostics only).

Usage:
    python scripts/experiments/run_xtail_spatial.py --seeds 0,1,2
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from data.real.xtail import DOMAINS  # noqa: E402
from data.real.xtail import XTAIL_ROOT  # noqa: E402
from evaluation import xtail_protocol as xp  # noqa: E402
from scripts.experiments.xtail_encode_patches import FINE, OUT as PATCH_DIR  # noqa: E402

OUT = REPO_ROOT / "data" / "generated" / "xtail"
GAMMAS = (0.1, 0.2, 0.4, 0.8)
LAMS = (1e-3, 1e-2, 1e-1)
GRID = 14


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def l2(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def regions(n_side):
    edges = np.linspace(0, GRID, n_side + 1).round().astype(int)
    return [(edges[i], edges[i + 1], edges[j], edges[j + 1]) for i in range(n_side) for j in range(n_side)]


def pool(kind, g, P):
    """g: (N, D) global; P: (N, 196, D) patch tokens. Returns a list of (N, D) blocks."""
    P = P.astype(np.float32)
    if kind == "global":
        return []
    if kind == "mean":
        return [P.mean(1)]
    if kind in ("grid2", "grid3"):
        G = P.reshape(len(P), GRID, GRID, -1)
        return [G[:, a:b, c:e].mean((1, 2)) for a, b, c, e in regions(int(kind[-1]))]
    if kind == "attn":
        s = np.einsum("nd,npd->np", g, P) / 0.05
        w = np.exp(s - s.max(1, keepdims=True))
        w /= w.sum(1, keepdims=True)
        return [np.einsum("np,npd->nd", w, P)]
    if kind == "topk":
        s = np.einsum("nd,npd->np", g, P)
        top = np.argsort(-s, axis=1)[:, :16]
        return [np.take_along_axis(P, top[:, :, None], axis=1).mean(1)]
    if kind == "gem":
        return [np.cbrt((np.clip(P, 0, None) ** 3).mean(1))]  # generalized-mean pooling, p = 3
    if kind == "multiscale":
        return pool("mean", g, P) + pool("grid2", g, P)
    raise ValueError(kind)


POOLINGS = ("global", "mean", "grid2", "grid3", "attn", "topk", "gem", "multiscale")


def features(kind, g, P):
    blocks = [g] + pool(kind, g, P)
    return (np.concatenate([l2(b) for b in blocks], 1) / np.sqrt(len(blocks))).astype(np.float32)


def krr_acc(Xtr, ytr, Xev, yev, gamma, lam):
    dev = xp.DEV
    A, B = torch.from_numpy(Xtr).to(dev), torch.from_numpy(Xev).to(dev)
    K = torch.exp(-gamma * torch.cdist(A, A).pow(2))
    Y = torch.nn.functional.one_hot(torch.from_numpy(ytr).to(dev).long()).float()
    alpha = torch.linalg.solve(K + lam * torch.eye(len(K), device=dev), Y)
    pred = (torch.exp(-gamma * torch.cdist(B, A).pow(2)) @ alpha).argmax(1).cpu().numpy()
    return float((pred == yev).mean())


def main(seeds):
    OUT.mkdir(parents=True, exist_ok=True)
    main_cache = xp.load_features(FINE)
    results = {}
    for seed in seeds:
        for name in FINE:
            z = np.load(PATCH_DIR / f"{name}.npz")
            d = main_cache[name]
            idx = xp.fewshot(d, seed, 16, DOMAINS.index(name))  # same draws as the main harness
            tr = (z["train_global"][idx], z["train_patches"][idx], z["train_labels"][idx])
            for kind in POOLINGS:
                Xtr = features(kind, tr[0], tr[1])
                Xva = features(kind, z["val_global"], z["val_patches"])
                Xte = features(kind, z["test_global"], z["test_patches"])
                val = {(g, l): krr_acc(Xtr, tr[2], Xva, z["val_labels"], g, l) for g, l in itertools.product(GAMMAS, LAMS)}
                test = {(g, l): krr_acc(Xtr, tr[2], Xte, z["test_labels"], g, l) for g, l in itertools.product(GAMMAS, LAMS)}
                results.setdefault(str(seed), {}).setdefault(kind, {})[name] = {
                    "val": {f"{g}_{l}": v for (g, l), v in val.items()}, "test": {f"{g}_{l}": v for (g, l), v in test.items()}}
            log(f"seed {seed} {name} done")
        # hyperparameters chosen on validation, per pooling, by mean over domains
        line = []
        for kind in POOLINGS:
            r = results[str(seed)][kind]
            hp = max(r[FINE[0]]["val"], key=lambda k: np.mean([r[n]["val"][k] for n in FINE]))
            accs = {n: r[n]["test"][hp] for n in FINE}
            results[str(seed)][kind]["chosen"] = {"hp": hp, "test": accs, "mean": float(np.mean(list(accs.values())))}
            line.append(f"{kind} {np.mean(list(accs.values())):.4f}")
        log(f"seed {seed}: " + ", ".join(line))
        with open(OUT / "spatial_phase1.json", "w") as f:
            json.dump(results, f, indent=1)


LAYER_DIR = (XTAIL_ROOT / "features/vitb16_layerpatch")
KS = (4, 8, 16)


def kmeans(X, k, seed, iters=25):
    """Plain k-means on the GPU (k-means++ seeding). X: (n, d) float32."""
    g = torch.Generator(device=xp.DEV).manual_seed(seed)
    X = torch.from_numpy(X).to(xp.DEV)
    C = X[torch.randint(len(X), (1,), device=xp.DEV, generator=g)]
    for _ in range(1, k):
        d = torch.cdist(X, C).min(1).values.pow(2)
        C = torch.cat([C, X[torch.multinomial(d / d.sum(), 1, generator=g)]])
    for _ in range(iters):
        a = torch.cdist(X, C).argmin(1)
        C = torch.stack([X[a == j].mean(0) if (a == j).any() else C[j] for j in range(k)])
    return C


def part_blocks(C, P):
    """Per-part mean of each image's patches (k blocks) and the part-count histogram."""
    Pt = torch.from_numpy(P.astype(np.float32)).to(xp.DEV)
    a = torch.cdist(Pt.reshape(-1, Pt.shape[-1]), C).argmin(1).reshape(len(P), -1)
    onehot = torch.nn.functional.one_hot(a, len(C)).float()  # (N, 196, k)
    counts = onehot.sum(1)  # (N, k)
    means = torch.einsum("npk,npd->nkd", onehot, Pt) / counts.clamp_min(1)[:, :, None]
    return [means[:, j].cpu().numpy() for j in range(len(C))], [(counts / P.shape[1]).cpu().numpy()]


def phase2(seeds):
    """Phase 2 (pre-registered in an earlier experiment): intermediate-layer patch tokens; part discovery
    by k-means on the 16-shot training images' patch tokens only."""
    main_cache = xp.load_features(FINE)
    results = {}
    for seed in seeds:
        for name in FINE:
            z, zl = np.load(PATCH_DIR / f"{name}.npz"), np.load(LAYER_DIR / f"{name}.npz")
            d = main_cache[name]
            idx = xp.fewshot(d, seed, 16, DOMAINS.index(name))
            split = {"train": idx, "val": slice(None), "test": slice(None)}
            g = {s: z[f"{s}_global"][split[s]] for s in split}
            P = {s: z[f"{s}_patches"][split[s]] for s in split}
            lay = {s: {k: zl[f"{s}_{k}"][split[s]] for k in ("mean8", "mean10", "gem8", "gem10")} for s in split}
            y = {"train": z["train_labels"][idx], "val": z["val_labels"], "test": z["test_labels"]}

            def assemble(blocks):
                return (np.concatenate([l2(b) for b in blocks], 1) / np.sqrt(len(blocks))).astype(np.float32)

            variants = {"ref_mean": {s: assemble([g[s], P[s].astype(np.float32).mean(1)]) for s in split}}
            for k in ("mean8", "mean10", "gem8", "gem10"):
                variants[f"layer_{k}"] = {s: assemble([g[s], lay[s][k]]) for s in split}
                variants[f"layer_{k}+P"] = {s: assemble([g[s], P[s].astype(np.float32).mean(1), lay[s][k]]) for s in split}
            Ptr = P["train"].astype(np.float32).reshape(-1, P["train"].shape[-1])
            sub = np.random.default_rng(seed).choice(len(Ptr), min(len(Ptr), 200000), replace=False)
            for k in KS:
                C = kmeans(Ptr[sub], k, seed)
                parts = {s: part_blocks(C, P[s]) for s in split}
                variants[f"parts{k}"] = {s: assemble([g[s]] + parts[s][0]) for s in split}
                variants[f"hist{k}"] = {s: assemble([g[s]] + parts[s][1]) for s in split}
            for vname, X in variants.items():
                val = {(gm, lm): krr_acc(X["train"], y["train"], X["val"], y["val"], gm, lm) for gm, lm in itertools.product(GAMMAS, LAMS)}
                test = {(gm, lm): krr_acc(X["train"], y["train"], X["test"], y["test"], gm, lm) for gm, lm in itertools.product(GAMMAS, LAMS)}
                results.setdefault(str(seed), {}).setdefault(vname, {})[name] = {
                    "val": {f"{a}_{b}": v for (a, b), v in val.items()}, "test": {f"{a}_{b}": v for (a, b), v in test.items()}}
            log(f"phase2 seed {seed} {name} done")
        for vname, r in results[str(seed)].items():
            hp = max(r[FINE[0]]["val"], key=lambda key: np.mean([r[n]["val"][key] for n in FINE]))
            accs = {n: r[n]["test"][hp] for n in FINE}
            r["chosen"] = {"hp": hp, "val_mean": float(np.mean([r[n]["val"][hp] for n in FINE])),
                           "test": accs, "mean": float(np.mean(list(accs.values())))}
        log(f"phase2 seed {seed}: " + ", ".join(f"{v} {r['chosen']['mean']:.4f}" for v, r in results[str(seed)].items()))
        with open(OUT / "spatial_phase2.json", "w") as f:
            json.dump(results, f, indent=1)


TILE_DIR = (XTAIL_ROOT / "features/vitb16_tiles")


def phase3(seeds):
    """B3: high-resolution tiles (448 upsampled, four 224 tiles, averaged) -- detail without
    spatial decomposition. Same task, readout, seeds, test subset and bar as Phases 1-2."""
    main_cache = xp.load_features(FINE)
    results = {}
    for seed in seeds:
        for name in FINE:
            z, zt = np.load(PATCH_DIR / f"{name}.npz"), np.load(TILE_DIR / f"{name}.npz")
            idx = xp.fewshot(main_cache[name], seed, 16, DOMAINS.index(name))
            split = {"train": idx, "val": slice(None), "test": slice(None)}
            g = {s: z[f"{s}_global"][split[s]] for s in split}
            pm = {s: z[f"{s}_patches"][split[s]].astype(np.float32).mean(1) for s in split}
            tg = {s: zt[f"{s}_tiles_global"][split[s]] for s in split}
            tp = {s: zt[f"{s}_tiles_patch_mean"][split[s]] for s in split}
            y = {"train": z["train_labels"][idx], "val": z["val_labels"], "test": z["test_labels"]}

            def assemble(blocks):
                return (np.concatenate([l2(b) for b in blocks], 1) / np.sqrt(len(blocks))).astype(np.float32)

            variants = {
                "ref_mean": {s: assemble([g[s], pm[s]]) for s in split},
                "tiles_global": {s: assemble([g[s], tg[s]]) for s in split},
                "tiles_global+P": {s: assemble([g[s], pm[s], tg[s]]) for s in split},
                "tiles_all": {s: assemble([g[s], pm[s], tg[s], tp[s]]) for s in split},
                "multiscale_avg": {s: assemble([l2(g[s]) + l2(tg[s]), pm[s]]) for s in split},
            }
            for vname, X in variants.items():
                val = {(gm, lm): krr_acc(X["train"], y["train"], X["val"], y["val"], gm, lm) for gm, lm in itertools.product(GAMMAS, LAMS)}
                test = {(gm, lm): krr_acc(X["train"], y["train"], X["test"], y["test"], gm, lm) for gm, lm in itertools.product(GAMMAS, LAMS)}
                results.setdefault(str(seed), {}).setdefault(vname, {})[name] = {
                    "val": {f"{a}_{b}": v for (a, b), v in val.items()}, "test": {f"{a}_{b}": v for (a, b), v in test.items()}}
            log(f"phase3 seed {seed} {name} done")
        for vname, r in results[str(seed)].items():
            hp = max(r[FINE[0]]["val"], key=lambda key: np.mean([r[n]["val"][key] for n in FINE]))
            accs = {n: r[n]["test"][hp] for n in FINE}
            r["chosen"] = {"hp": hp, "val_mean": float(np.mean([r[n]["val"][hp] for n in FINE])),
                           "test": accs, "mean": float(np.mean(list(accs.values())))}
        log(f"phase3 seed {seed}: " + ", ".join(f"{v} {r['chosen']['mean']:.4f}" for v, r in results[str(seed)].items()))
        with open(OUT / "spatial_phase3.json", "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--phase", type=int, default=1)
    a = p.parse_args()
    {1: main, 2: phase2, 3: phase3}[a.phase]([int(s) for s in a.seeds.split(",")])
