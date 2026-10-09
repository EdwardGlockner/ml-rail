"""How much Transfer is lost to the gate sending images of not-yet-learned domains to the adapter, and how
much a reverse gate could recover.

The adapter can only answer with learned classes, so an image of a domain not yet learned that the gate
admits (its zero-shot top-1 class is learned, or the rescue gate let it in) is wrong for certain. A reverse
gate takes such an image back: if it is far from every learned domain, it is predicted as zero-shot CLIP's
best class among those not yet learned. Variants, all on top of ML-RAIL + rescue gate:

  base         ML-RAIL + rescue gate (q 0.25, m 0.1), the reported method
  oracle       every admitted image of a not-yet-learned domain is taken back: the ceiling of this idea
  reverse_q    take back admitted images whose similarity to every learned domain d is below the
               q-quantile of d's own nearest-neighbour similarities (learned-domain images can be hit too)

Also logs, per run, the share of not-yet-learned-domain images that are admitted and how often zero-shot's
best not-yet-learned class is right for them. Development seeds and validation only. `--setting` runs order II or the 5-shot 11-domain variant; their
results are stored under "<setting>/<split>:<seed>".

Usage:
    python scripts/experiments/run_xtail_transfer_headroom.py [--runs val:100,test:0] [--setting order_II]
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

from data.real.xtail import DOMAINS, DOMAINS_11, ORDER_II, XTAIL_ROOT  # noqa: E402
from evaluation import xtail_protocol as xp  # noqa: E402
from evaluation.ml_rail import BLOCKS, blocks_feature, load_with_extra, make_ml_rail  # noqa: E402

GATE_Q, GATE_MIN_MASS, W = 0.25, 0.1, 0.8
REVERSE_QS = (0.001, 0.01, 0.025, 0.05, 0.1)
SETTINGS = {"order_I": (DOMAINS, 16), "order_II": (ORDER_II, 16), "11_domains_5shot": (DOMAINS_11, 5)}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def nn_similarities(method, Xd):
    """Each training image's max similarity to the other training images of its domain (duplicates excluded)."""
    out = []
    for i in range(0, len(Xd), 2048):
        K = method._k(Xd[i : i + 2048], Xd)
        K[K >= 1 - 1e-6] = -1
        out.append(K.max(1).values.cpu())
    return torch.cat(out)


def run(data, split, seed, shots=16):
    names = list(data)
    feat = blocks_feature(BLOCKS)
    zs = xp.zero_shot_probs(data, split=split)
    m = make_ml_rail()
    variants = ["base", "oracle"] + [f"reverse_{q}" for q in REVERSE_QS]
    tables = {v: np.zeros((len(names), len(names))) for v in variants}
    nn_sims, spans, learned, n_train = [], [], 0, 0
    diag = {"admitted_unseen": [], "acc_admitted_unseen_base": [], "acc_admitted_unseen_best_unlearned": []}
    test_feats = {n: torch.as_tensor(feat(data[n], split), device=xp.DEV) for n in names}
    for t, n in enumerate(names):
        d = data[n]
        idx = xp.fewshot(d, seed, shots, t)
        m.add(feat(d, "train", idx), d["train_labels"][idx] + d["offset"])
        spans.append((n_train, n_train + len(idx)))
        n_train += len(idx)
        learned += d["n_classes"]
        nn_sims.append(nn_similarities(m, m.X[spans[-1][0] : spans[-1][1]]))
        tau = torch.tensor([float(torch.quantile(s, GATE_Q)) for s in nn_sims])
        tau_r = {q: torch.tensor([float(torch.quantile(s, q)) for s in nn_sims]) for q in REVERSE_QS}
        for j, mname in enumerate(names):
            e = data[mname]
            X = test_feats[mname]
            scores, sims = [], []
            for i in range(0, len(X), 2048):
                K = m._k(X[i : i + 2048], m.X)
                scores.append((K @ m.alpha).cpu())
                sims.append(torch.stack([K[:, a:b].max(1).values for a, b in spans], 1).cpu())
            scores, sims = torch.cat(scores), torch.cat(sims)
            p0 = zs[mname]
            top1 = p0.argmax(1)
            mask = (top1 < learned) | ((p0[:, :learned].sum(1) >= GATE_MIN_MASS) & (sims >= tau).any(1))
            p = p0.clone()
            pad = torch.zeros_like(p0)
            pad[:, : scores.shape[1]] = scores
            p[mask] = (1 - W) * p0[mask] + W * pad[mask]
            y = torch.as_tensor(e[f"{split}_labels"] + e["offset"])
            pred = p.argmax(1)
            if learned < p0.shape[1]:
                best_unlearned = learned + p0[:, learned:].argmax(1)
            else:  # last step: every class is learned, nothing can be taken back
                best_unlearned = pred
            tables["base"][t, j] = 100 * float((pred == y).float().mean())
            take = mask & torch.tensor(j > t)
            tables["oracle"][t, j] = 100 * float((torch.where(take, best_unlearned, pred) == y).float().mean())
            for q in REVERSE_QS:
                take = mask & (sims < tau_r[q]).all(1) & torch.tensor(learned < p0.shape[1])
                tables[f"reverse_{q}"][t, j] = 100 * float((torch.where(take, best_unlearned, pred) == y).float().mean())
            if j > t:
                diag["admitted_unseen"].append(float(mask.float().mean()))
                if mask.any():
                    diag["acc_admitted_unseen_base"].append(float((pred[mask] == y[mask]).float().mean()))
                    diag["acc_admitted_unseen_best_unlearned"].append(float((best_unlearned[mask] == y[mask]).float().mean()))
        log(f"  {split}:{seed} step {t} ({n}) done")
    return tables, {k: float(np.mean(v)) for k, v in diag.items()}


def main(runs, setting="order_I"):
    out = XTAIL_ROOT / "results" / "transfer_headroom.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    res = json.load(open(out)) if out.exists() else {}
    domains, shots = SETTINGS[setting]
    data = load_with_extra("single", list(domains))
    for run_id in runs:
        r = run_id if setting == "order_I" else f"{setting}/{run_id}"
        if r in res:
            continue
        split, seed = run_id.split(":")
        t0 = time.time()
        tables, diag = run(data, split, int(seed), shots)
        res[r] = {"diagnostics": diag, **{v: {**xp.metrics(tb), "table": tb.tolist()} for v, tb in tables.items()}}
        json.dump(res, open(out, "w"), indent=1)
        base = res[r]["base"]
        log(f"{r}: base {base['transfer']:.2f} / {base['average']:.2f} / {base['last']:.2f} [{time.time() - t0:.0f}s]; {diag}")
        for v in tables:
            if v != "base":
                log(f"  {v:15s} dT {res[r][v]['transfer'] - base['transfer']:+.3f}  dA {res[r][v]['average'] - base['average']:+.3f}"
                    f"  dL {res[r][v]['last'] - base['last']:+.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="val:100,test:0")
    p.add_argument("--setting", default="order_I", choices=SETTINGS)
    a = p.parse_args()
    main(a.runs.split(","), a.setting)
