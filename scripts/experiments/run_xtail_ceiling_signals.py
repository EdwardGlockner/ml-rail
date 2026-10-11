"""Which training-free signal best separates the unseen-domain images the gate wrongly admits from the
learned-domain ones? This is the study behind the reverse gate's layer-10 class-token signal.

`--collect` records, for every image ML-RAIL + rescue gate admits at every step before the last, a
per-domain percentile score u = max over learned domains d of F_d(s_d(x)) for each signal, where s_d is
the image's similarity to domain d's stored images and F_d the empirical CDF of the same similarity among
d's own training images (leave-one-out). It also records what taking the image back would gain, and
whether it comes from a domain not yet learned. Signals:

  nn1        RBF kernel on all five blocks, nearest neighbour (the full-kernel reverse gate: u < 0.01)
  top5       the same kernel, mean of the 5 nearest neighbours
  blk_<b>    cosine on one feature block <b>, nearest neighbour (blk_cls10: the class-token reverse gate)
  cls        kernel similarity to the training images of the adapter's top class
  maha       Mahalanobis distance to the nearest class mean (pooled within-class covariance, shrunk)

`--analyze` (no GPU) reports, on development seeds 1-5: each signal's AUROC for unseen vs. learned among
admitted images; the best Transfer gain each reaches with no loss in Average, with the threshold chosen on
validation and (optimistically) on the seeds themselves; the threshold each of two selection criteria
picks on validation; and where the oracle's gain comes from, per unseen domain. Writes
$XTAIL_ROOT/results/ceiling/analysis.json. Development only.

Usage:
    python scripts/experiments/run_xtail_ceiling_signals.py --collect [--setting order_II] [--runs val:100,test:1,...]
    python scripts/experiments/run_xtail_ceiling_signals.py --analyze
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from data.real.xtail import XTAIL_ROOT  # noqa: E402
from evaluation import xtail_protocol as xp  # noqa: E402
from evaluation.ml_rail import BLOCKS, SLICES, blocks_feature, load_with_extra, make_ml_rail  # noqa: E402
from scripts.experiments.run_xtail_transfer_headroom import SETTINGS  # noqa: E402

OUT = XTAIL_ROOT / "results" / "ceiling"
GATE_Q, GATE_MIN_MASS, W, CH = 0.25, 0.1, 0.8, 1024
SIGNALS = ["nn1", "top5"] + [f"blk_{b}" for b in BLOCKS] + ["cls", "maha"]
DEV_RUNS = ("test:1", "test:2", "test:3", "test:4", "test:5")
NB = len(BLOCKS)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def loo_topk(A, k):
    """Each row's mean RBF similarity to its k nearest other rows (exact duplicates excluded)."""
    out = []
    for i in range(0, len(A), CH):
        K = torch.exp(-0.2 * torch.cdist(A[i:i + CH], A).pow(2))
        K[K >= 1 - 1e-6] = -1
        out.append(K.topk(k, 1).values.mean(1).cpu())
    return torch.cat(out)


def loo_block(A, sl):
    """Each row's max cosine similarity to the other rows in one feature block (duplicates excluded)."""
    out = []
    Ab = A[:, sl[0]:sl[1]] * np.sqrt(NB)  # the blocks are unit vectors scaled by 1/sqrt(NB)
    for i in range(0, len(Ab), CH):
        S = Ab[i:i + CH] @ Ab.T
        S[S >= 1 - 1e-5] = -2
        out.append(S.max(1).values.cpu())
    return torch.cat(out)


def cdf(sorted_vals, x):
    return torch.searchsorted(sorted_vals, x.contiguous(), right=True).float() / len(sorted_vals)


def collect(data, split, seed, shots=16):
    names = list(data)
    feat = blocks_feature(BLOCKS)
    zs = xp.zero_shot_probs(data, split=split)
    m = make_ml_rail()
    n_cls = sum(d["n_classes"] for d in data.values())
    base = np.zeros((len(names),) * 2)
    sizes = np.zeros_like(base)
    rec = {k: [] for k in ["t", "j", "gain", "unseen", "near"] + SIGNALS}
    ref = {k: [] for k in SIGNALS}  # per learned domain: sorted reference values
    spans, offs, learned, n_train = [], [], 0, 0
    test = {n: torch.as_tensor(feat(data[n], split)) for n in names}  # on the CPU; chunks move to the GPU
    for t, n in enumerate(names):
        d = data[n]
        idx = xp.fewshot(d, seed, shots, t)
        m.add(feat(d, "train", idx), d["train_labels"][idx] + d["offset"])
        a, b = n_train, n_train + len(idx)
        spans.append((a, b))
        offs.append((d["offset"], d["n_classes"]))
        n_train, learned = b, learned + d["n_classes"]
        Xd = m.X[a:b]
        ref["nn1"].append(torch.sort(loo_topk(Xd, 1)).values)
        ref["top5"].append(torch.sort(loo_topk(Xd, 5)).values)
        for bi, sl in enumerate(SLICES):
            ref[f"blk_{BLOCKS[bi]}"].append(torch.sort(loo_block(Xd, sl)).values)
        ref["cls"].append(ref["nn1"][-1])
        y_all = torch.as_tensor(m.y, device=xp.DEV)
        Xall = m.X.double()
        means = torch.zeros(learned, Xall.shape[1], dtype=torch.float64, device=xp.DEV).index_add_(0, y_all, Xall)
        means /= torch.bincount(y_all, minlength=learned).clamp(min=1)[:, None]
        R = Xall - means[y_all]
        cov = (R.T @ R) / len(R)
        cov = 0.9 * cov + 0.1 * torch.eye(len(cov), device=xp.DEV, dtype=torch.float64) * cov.diagonal().mean()
        L = torch.linalg.cholesky(torch.linalg.inv(cov))
        Mw = (means @ L).float()

        def maha_min(Z, cls_lo, cls_hi):
            return -torch.cdist((Z.double() @ L).float(), Mw[cls_lo:cls_hi]).min(1).values

        ref["maha"] = [torch.sort(maha_min(m.X[s0:s1], o, o + k).cpu()).values for (s0, s1), (o, k) in zip(spans, offs)]
        tau = torch.tensor([float(torch.quantile(r, GATE_Q)) for r in ref["nn1"]])
        for j, mn in enumerate(names):
            e = data[mn]
            X = test[mn]
            p0 = zs[mn]
            scores, nn1, top5, blk, maha = [], [], [], [[] for _ in BLOCKS], []
            for i in range(0, len(X), CH):
                Xc = X[i:i + CH].to(xp.DEV)
                K = m._k(Xc, m.X)
                scores.append((K @ m.alpha).cpu())
                nn1.append(torch.stack([K[:, s0:s1].max(1).values for s0, s1 in spans], 1).cpu())
                top5.append(torch.stack([K[:, s0:s1].topk(5, 1).values.mean(1) for s0, s1 in spans], 1).cpu())
                for bi, sl in enumerate(SLICES):
                    S = (Xc[:, sl[0]:sl[1]] * np.sqrt(NB)) @ (m.X[:, sl[0]:sl[1]] * np.sqrt(NB)).T
                    blk[bi].append(torch.stack([S[:, s0:s1].max(1).values for s0, s1 in spans], 1).cpu())
                maha.append(torch.stack([maha_min(Xc, o, o + k) for o, k in offs], 1).cpu())
                del K, S, Xc
            scores, nn1, top5, maha = torch.cat(scores), torch.cat(nn1), torch.cat(top5), torch.cat(maha)
            blk = [torch.cat(v) for v in blk]
            mask = (p0.argmax(1) < learned) | ((p0[:, :learned].sum(1) >= GATE_MIN_MASS) & (nn1 >= tau).any(1))
            pad = torch.zeros_like(p0)
            pad[:, :scores.shape[1]] = scores
            p = p0.clone()
            p[mask] = (1 - W) * p0[mask] + W * pad[mask]
            y = torch.as_tensor(e[f"{split}_labels"] + e["offset"])
            pred = p.argmax(1)
            base[t, j] = 100 * float((pred == y).float().mean())
            sizes[t, j] = len(y)
            if learned == n_cls or not mask.any():
                continue
            Xs = X[mask]
            cls_top = torch.as_tensor(scores[mask].argmax(1), device=xp.DEV)
            ycl = torch.as_tensor(m.y, device=xp.DEV)
            ccs = []
            for i in range(0, len(Xs), CH):
                K = m._k(Xs[i:i + CH].to(xp.DEV), m.X)
                K[ycl[None, :] != cls_top[i:i + CH, None]] = -1
                ccs.append(K.max(1).values.cpu())
            ccs = torch.cat(ccs)
            dom_of_cls = torch.as_tensor(np.searchsorted([o for o, _ in offs], cls_top.cpu().numpy(), side="right") - 1)
            F = {"nn1": torch.stack([cdf(ref["nn1"][k], nn1[mask][:, k]) for k in range(t + 1)], 1),
                 "top5": torch.stack([cdf(ref["top5"][k], top5[mask][:, k]) for k in range(t + 1)], 1),
                 "maha": torch.stack([cdf(ref["maha"][k], maha[mask][:, k]) for k in range(t + 1)], 1)}
            for bi, b in enumerate(BLOCKS):
                F[f"blk_{b}"] = torch.stack([cdf(ref[f"blk_{b}"][k], blk[bi][mask][:, k]) for k in range(t + 1)], 1)
            cls_u = torch.zeros(len(ccs))
            for k in dom_of_cls.unique().tolist():
                w = dom_of_cls == k
                cls_u[w] = cdf(ref["cls"][k], ccs[w])
            best_unl = learned + p0[:, learned:].argmax(1)
            gain = ((best_unl == y).int() - (pred == y).int())[mask]
            n_adm = int(mask.sum())
            rec["t"].append(np.full(n_adm, t))
            rec["j"].append(np.full(n_adm, j))
            rec["gain"].append(gain.numpy())
            rec["unseen"].append(np.full(n_adm, j > t))
            rec["near"].append(F["nn1"].argmax(1).numpy())
            for k, v in F.items():
                rec[k].append(v.max(1).values.numpy())
            rec["cls"].append(cls_u.numpy())
        log(f"  {split}:{seed} step {t} ({n})")
    return {k: np.concatenate(v) for k, v in rec.items()} | {"base": base, "sizes": sizes}


def effect(r, take):
    """Change in Transfer and Average (points) from taking back the admitted images in `take`."""
    delta = np.zeros_like(r["base"])
    np.add.at(delta, (r["t"][take], r["j"][take]), r["gain"][take])
    a, b = xp.metrics(r["base"] + 100 * delta / r["sizes"]), xp.metrics(r["base"])
    return np.array([a["transfer"] - b["transfer"], a["average"] - b["average"]])


def auroc(score, y):
    order = np.argsort(score, kind="stable")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    n1, n0 = y.sum(), (~y).sum()
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def load(setting, run):
    prefix = "" if setting == "order_I" else f"{setting}_"
    return dict(np.load(OUT / f"{prefix}{run.replace(':', '_')}.npz"))


def analyze():
    grid = np.round(np.arange(0.001, 0.051, 0.001), 3)
    val, dev = load("order_I", "val:100"), [load("order_I", r) for r in DEV_RUNS]
    mean = lambda rs, f: np.mean([effect(r, f(r)) for r in rs], 0)  # noqa: E731
    out = {"order_I": {"oracle": mean(dev, lambda r: r["unseen"]).tolist(), "signals": {}}}
    log(f"order I, seeds 1-5: oracle dT/dA {np.round(out['order_I']['oracle'], 3)}")
    for s in SIGNALS:
        val_curve = [effect(val, val[s] < x) for x in grid]
        dev_curve = [mean(dev, lambda r, x=x: r[s] < x) for x in grid]
        ok = [k for k, e in enumerate(val_curve) if e[1] >= 0]
        k_old = max(ok, key=lambda k: val_curve[k][0]) if ok else None  # most Transfer, Average >= 0
        k_bal = int(np.argmax([e.sum() for e in val_curve]))  # most Transfer + Average
        okd = [k for k, e in enumerate(dev_curve) if e[1] >= 0]
        k_opt = max(okd, key=lambda k: dev_curve[k][0]) if okd else None
        row = {"auroc": float(np.mean([auroc(1 - r[s], r["unseen"]) for r in dev])),
               "grid": grid.tolist(), "dev_curve": np.round(dev_curve, 4).tolist(),
               "val_most_transfer": None if k_old is None else {"u": float(grid[k_old]), "dev": dev_curve[k_old].tolist()},
               "val_balanced": {"u": float(grid[k_bal]), "dev": dev_curve[k_bal].tolist()},
               "dev_optimistic": None if k_opt is None else {"u": float(grid[k_opt]), "dev": dev_curve[k_opt].tolist()}}
        out["order_I"]["signals"][s] = row
        fmt = lambda v: "-" if v is None else f"u<{v['u']:.3f}: {v['dev'][0]:+.2f} / {v['dev'][1]:+.2f}"  # noqa: E731
        log(f"  {s:15s} AUROC {row['auroc']:.3f} | val most-Transfer {fmt(row['val_most_transfer'])} | "
            f"val balanced {fmt(row['val_balanced'])} | optimistic {fmt(row['dev_optimistic'])}")
    # where the oracle's gain comes from (order I, seeds 1-5 pooled)
    from data.real.xtail import DOMAINS
    cat = {k: np.concatenate([r[k] for r in dev]) for k in ("j", "near", "gain", "unseen", "nn1", "blk_cls10")}
    U = cat["unseen"]
    total = cat["gain"][U].clip(min=0).sum()
    out["order_I"]["by_unseen_domain"] = {}
    for j in np.unique(cat["j"][U]):
        w = U & (cat["j"] == j)
        g = cat["gain"][w].clip(min=0)
        near = np.bincount(cat["near"][w], minlength=len(DOMAINS))
        row = {"admitted": int(w.sum()), "share_of_oracle_gain": float(g.sum() / total),
               "caught_full_kernel": float(g[cat["nn1"][w] < 0.01].sum() / max(g.sum(), 1)),
               "caught_class_token": float(g[cat["blk_cls10"][w] < 0.008].sum() / max(g.sum(), 1)),
               "nearest_learned_domain": DOMAINS[int(near.argmax())], "nearest_share": float(near.max() / w.sum())}
        out["order_I"]["by_unseen_domain"][DOMAINS[j]] = row
        log(f"  {DOMAINS[j]:15s} {row['share_of_oracle_gain']:5.1%} of the oracle gain; caught: full kernel "
            f"{row['caught_full_kernel']:5.1%}, class token {row['caught_class_token']:5.1%}; nearest "
            f"{row['nearest_learned_domain']} ({row['nearest_share']:.0%})")
    # the two reverse gates vs. the oracle in every setting
    for setting in SETTINGS:
        try:
            rs = [load(setting, r) for r in DEV_RUNS]
        except FileNotFoundError:
            continue
        cur = np.array([effect(r, r["nn1"] < 0.01) for r in rs])
        new = np.array([effect(r, r["blk_cls10"] < 0.008) for r in rs])
        orc = np.array([effect(r, r["unseen"]) for r in rs])
        out.setdefault(setting, {}).update({"full_kernel": cur.mean(0).tolist(), "class_token": new.mean(0).tolist(),
                                            "oracle": orc.mean(0).tolist(), "class_token_minus_full_kernel_per_seed": (new - cur).tolist()})
        log(f"{setting}: full kernel {cur.mean(0).round(3)}, class token {new.mean(0).round(3)}, oracle {orc.mean(0).round(3)}")
    json.dump(out, open(OUT / "analysis.json", "w"), indent=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--collect", action="store_true")
    p.add_argument("--analyze", action="store_true")
    p.add_argument("--setting", default="order_I", choices=SETTINGS)
    p.add_argument("--runs", default="val:100," + ",".join(DEV_RUNS))
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.collect:
        domains, shots = SETTINGS[a.setting]
        data = load_with_extra("single", list(domains))
        prefix = "" if a.setting == "order_I" else f"{a.setting}_"
        for r in a.runs.split(","):
            path = OUT / f"{prefix}{r.replace(':', '_')}.npz"
            if path.exists():
                continue
            split, seed = r.split(":")
            t0 = time.time()
            np.savez(path, **collect(data, split, int(seed), shots))
            log(f"{r} done [{time.time() - t0:.0f}s]")
    if a.analyze:
        analyze()


if __name__ == "__main__":
    main()
