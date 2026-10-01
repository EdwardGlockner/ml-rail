"""Screening T1 and T3 on validation: zero-shot accuracy of each text
head over the full X-TAIL label set (what Transfer measures, since RAIL's fusion leaves not-yet-learned
domains on zero-shot), with and without domain-then-class reweighting:

    p'(c | x)  proportional to  p(c | x) * p(dataset(c) | x) ** alpha,
    p(dataset | x) = softmax(100 * cos(x, mean of that dataset's class embeddings)).

Usage:
    python scripts/run_xtail_zeroshot.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from evaluation import xtail_protocol as xp  # noqa: E402
from evaluation.xtail_prompts import build_text  # noqa: E402

OUT = REPO_ROOT / "data" / "generated" / "xtail"
HEADS = ("single", "generic", "t1a", "t1b", "t1a+generic", "t1b+generic")
ALPHAS = (0.0, 0.5, 1.0, 2.0, 4.0)


def zero_shot_accuracy(data, alpha, split="val"):
    names = list(data)
    text = torch.from_numpy(np.concatenate([data[n]["text"] for n in names]))
    owner = torch.cat([torch.full((data[n]["n_classes"],), i) for i, n in enumerate(names)])
    protos = torch.stack([torch.from_numpy(data[n]["text"]).mean(0) for n in names])
    protos = protos / protos.norm(dim=-1, keepdim=True)
    acc = {}
    for n in names:
        x = torch.from_numpy(data[n][f"{split}_global"])
        logp = torch.log_softmax(100 * x @ text.T, dim=1)
        if alpha:
            logp = logp + alpha * torch.log_softmax(100 * x @ protos.T, dim=1)[:, owner]
        pred = logp.argmax(1).numpy()
        acc[n] = 100 * float((pred == data[n][f"{split}_labels"] + data[n]["offset"]).mean())
    return acc


def main():
    results = []
    for head in HEADS:
        data = build_text(xp.load_features(), head)
        for alpha in ALPHAS:
            acc = zero_shot_accuracy(data, alpha)
            names = list(acc)
            row = {"head": head, "alpha": alpha, "transfer_proxy": float(np.mean([acc[n] for n in names[1:]])),
                   "mean_all": float(np.mean(list(acc.values()))), "per_domain": acc}
            results.append(row)
            print(f"{head:12s} alpha {alpha:<4}: transfer proxy {row['transfer_proxy']:.2f}  mean {row['mean_all']:.2f}  "
                  + " ".join(f"{v:.1f}" for v in acc.values()), flush=True)
    json.dump(results, open(OUT / "zeroshot_heads_val.json", "w"), indent=1)


BETAS = (0.0, 0.25, 0.5, 1.0)  # T2 hub-correction strength
THRESHOLDS = (None, 0.3, 0.5, 0.7)  # T4: also route to the adapter if its top score >= threshold
APRIME = ("global", "cls10", "patch_mean", "gem8", "gem10")


def hub_bias(data, names, learned, shot_idx):
    """T2: b_c = mean cos(x, t_c) over the 16-shot training images of learned domains *other than*
    c's own; classes of the only learned domain (step 0) get the mean bias (neutral)."""
    text = torch.from_numpy(np.concatenate([data[n]["text"] for n in names]))
    b = torch.empty(len(text))
    for i, n in enumerate(names):
        others = [m for m in learned if m != n]
        lo, hi = data[n]["offset"], data[n]["offset"] + data[n]["n_classes"]
        if others:
            X = torch.from_numpy(np.concatenate([data[m]["train_global"][shot_idx[m]] for m in others]))
            b[lo:hi] = (X @ text[lo:hi].T).mean(0)
        else:
            b[lo:hi] = float("nan")
    b[torch.isnan(b)] = b[~torch.isnan(b)].mean()
    return b


def stream_grid(data, seed, split="val", w=0.8):
    """One ML-RAIL stream; every (beta, threshold) combination evaluated from the same adapter scores."""
    from scripts.run_xtail_improve import blocks_feature
    names = list(data)
    feat = blocks_feature(APRIME)
    text = torch.from_numpy(np.concatenate([data[n]["text"] for n in names]))
    shot_idx = {n: xp.fewshot(data[n], seed, 16, t) for t, n in enumerate(names)}
    m = xp.DualRAIL(gamma=0.2, lam=1e-3, dtype=torch.float32)
    tables = {(b, th): np.zeros((len(names), len(names))) for b in BETAS for th in THRESHOLDS}
    learned_n = 0
    for t, n in enumerate(names):
        d = data[n]
        m.add(feat(d, "train", shot_idx[n]), d["train_labels"][shot_idx[n]] + d["offset"])
        learned_n += d["n_classes"]
        bias = hub_bias(data, names, names[: t + 1], shot_idx)
        for j, en in enumerate(names):
            e = data[en]
            x = torch.from_numpy(e[f"{split}_global"])
            cos = x @ text.T
            S = torch.as_tensor(m.scores(feat(e, split)), dtype=torch.float32)
            pad = torch.zeros(len(x), len(text))
            pad[:, : S.shape[1]] = S
            y = e[f"{split}_labels"] + e["offset"]
            for b in BETAS:
                p = torch.softmax(100 * (cos - b * bias), dim=1)
                for th in THRESHOLDS:
                    mask = p.argmax(1) < learned_n
                    if th is not None:
                        mask |= S.max(1).values >= th
                    q = p.clone()
                    q[mask] = (1 - w) * p[mask] + w * pad[mask]
                    tables[(b, th)][t, j] = 100 * float((q.argmax(1).numpy() == y).mean())
    return tables


def grid_main(heads=("generic", "t1b+generic"), seed=100):
    from scripts.run_xtail_improve import load_with_extra
    results = []
    for head in heads:
        data = build_text(load_with_extra("single"), head)
        for (b, th), table in stream_grid(data, seed).items():
            results.append({"head": head, "beta": b, "threshold": th, **xp.metrics(table)})
        json.dump(results, open(OUT / "transfer_grid_val.json", "w"), indent=1)
        best = max((r for r in results if r["head"] == head), key=lambda r: r["transfer"] + r["average"])
        print(f"{head}: best {best}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "grid":
        grid_main()
    else:
        main()
