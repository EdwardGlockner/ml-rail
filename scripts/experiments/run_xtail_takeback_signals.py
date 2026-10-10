"""Can a better signal take back more of the unseen-domain images the gate wrongly admits?

`--collect` records, for every image ML-RAIL + rescue gate admits at every step before the last, the
signals a take-back rule could use and what taking it back would gain:

  u      max over learned domains d of F_d(s_d(x)), where s_d is the image's max similarity to d's stored
         images and F_d the empirical CDF of d's own nearest-neighbour similarities (reverse gate: u < 0.01)
  mu     zero-shot probability mass on the classes not yet learned
  pl,pu  zero-shot top probability among learned / not-yet-learned classes
  fmax   the adapter's top score
  gain   (best not-yet-learned class correct) - (current prediction correct), in {-1, 0, 1}

`--analyze` evaluates rule families on these records without rerunning anything: each rule is selected
on validation (largest Transfer gain with no loss in Average, as the reverse gate was) and reported on
development seeds 1-5 against the current reverse gate (u < 0.01). Development only.

Usage:
    python scripts/experiments/run_xtail_takeback_signals.py --collect [--runs val:100,test:1,...]
    python scripts/experiments/run_xtail_takeback_signals.py --analyze
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

from data.real.xtail import XTAIL_ROOT  # noqa: E402
from evaluation import xtail_protocol as xp  # noqa: E402
from evaluation.ml_rail import BLOCKS, blocks_feature, load_with_extra, make_ml_rail  # noqa: E402
from scripts.experiments.run_xtail_transfer_headroom import SETTINGS, nn_similarities  # noqa: E402

GATE_Q, GATE_MIN_MASS, W = 0.25, 0.1, 0.8
OUT = XTAIL_ROOT / "results" / "takeback"
DEV_RUNS = ("test:1", "test:2", "test:3", "test:4", "test:5")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def collect(data, split, seed, shots=16):
    names = list(data)
    feat = blocks_feature(BLOCKS)
    zs = xp.zero_shot_probs(data, split=split)
    m = make_ml_rail()
    n_cls = sum(d["n_classes"] for d in data.values())
    base = np.zeros((len(names), len(names)))
    sizes = np.zeros((len(names), len(names)))
    rec = {k: [] for k in ("t", "j", "u", "mu", "pl", "pu", "fmax", "gain")}
    nn_sorted, spans, learned, n_train = [], [], 0, 0
    test_feats = {n: torch.as_tensor(feat(data[n], split), device=xp.DEV) for n in names}
    for t, n in enumerate(names):
        d = data[n]
        idx = xp.fewshot(d, seed, shots, t)
        m.add(feat(d, "train", idx), d["train_labels"][idx] + d["offset"])
        spans.append((n_train, n_train + len(idx)))
        n_train += len(idx)
        learned += d["n_classes"]
        nn_sorted.append(torch.sort(nn_similarities(m, m.X[spans[-1][0] : spans[-1][1]])).values)
        tau = torch.tensor([float(torch.quantile(s, GATE_Q)) for s in nn_sorted])
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
            pad = torch.zeros_like(p0)
            pad[:, : scores.shape[1]] = scores
            p = p0.clone()
            p[mask] = (1 - W) * p0[mask] + W * pad[mask]
            y = torch.as_tensor(e[f"{split}_labels"] + e["offset"])
            pred = p.argmax(1)
            base[t, j] = 100 * float((pred == y).float().mean())
            sizes[t, j] = len(y)
            if learned == n_cls or not mask.any():
                continue
            cdf = torch.stack([torch.searchsorted(s, sims[:, k].contiguous(), right=True).float() / len(s)
                               for k, s in enumerate(nn_sorted)], 1)
            best_unl = learned + p0[:, learned:].argmax(1)
            gain = (best_unl == y).int() - (pred == y).int()
            sel = mask.numpy()
            rec["t"].append(np.full(sel.sum(), t))
            rec["j"].append(np.full(sel.sum(), j))
            rec["u"].append(cdf.max(1).values.numpy()[sel])
            rec["mu"].append(p0[:, learned:].sum(1).numpy()[sel])
            rec["pl"].append(p0[:, :learned].max(1).values.numpy()[sel])
            rec["pu"].append(p0[:, learned:].max(1).values.numpy()[sel])
            rec["fmax"].append(scores.max(1).values.numpy()[sel])
            rec["gain"].append(gain.numpy()[sel])
        log(f"  {split}:{seed} step {t} ({n}) done")
    return {k: np.concatenate(v) for k, v in rec.items()} | {"base": base, "sizes": sizes}


def effect(r, take):
    """Change in Transfer / Average / Last (points) from taking back the admitted images in `take`."""
    delta = np.zeros_like(r["base"])
    np.add.at(delta, (r["t"][take], r["j"][take]), r["gain"][take])
    table = r["base"] + 100 * delta / r["sizes"]
    a, b = xp.metrics(table), xp.metrics(r["base"])
    return {k: a[k] - b[k] for k in a}


def features(r):
    return np.stack([np.log(r["u"] + 1e-4), r["mu"], np.log((r["pu"] + 1e-8) / (r["pl"] + 1e-8)), r["fmax"]], 1)


def rule_families(val):
    """name -> list of (params, function records -> take mask)."""
    fam = {"A: u": [((a,), lambda r, a=a: r["u"] < a) for a in np.linspace(0.002, 0.05, 25)]}
    fam["B: u, mass"] = [((a, b, c), lambda r, a=a, b=b, c=c: (r["u"] < a) | ((r["u"] < b) & (r["mu"] >= c)))
                         for a, b, c in itertools.product((0.005, 0.01, 0.015), (0.02, 0.05, 0.1, 0.2), (0.3, 0.5, 0.7, 0.9))]
    fam["C: u, ratio"] = [((a, b, c), lambda r, a=a, b=b, c=c: (r["u"] < a) | ((r["u"] < b) & (r["pu"] >= c * r["pl"])))
                          for a, b, c in itertools.product((0.005, 0.01, 0.015), (0.02, 0.05, 0.1, 0.2), (0.5, 1.0, 2.0, 4.0))]
    # D: logistic regression on validation records, target "taking back helps" (gain > 0) vs. hurts (gain < 0)
    X, g = features(val).astype(np.float64), val["gain"]
    keep = g != 0
    mu, sd = X[keep].mean(0), X[keep].std(0) + 1e-8
    Z, target = torch.as_tensor((X[keep] - mu) / sd), torch.as_tensor((g[keep] > 0).astype(np.float64))
    w = torch.zeros(Z.shape[1] + 1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w], max_iter=200)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(Z @ w[:-1] + w[-1], target) + 1e-3 * (w[:-1] ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    wv = w.detach().numpy()
    score = lambda r: ((features(r) - mu) / sd) @ wv[:-1] + wv[-1]  # noqa: E731
    fam["D: logistic"] = [((th,), lambda r, th=th: score(r) > th) for th in np.linspace(-1, 4, 51)]
    return fam, {"weights": wv.tolist(), "feature_mean": mu.tolist(), "feature_std": sd.tolist()}


def analyze(setting="order_I"):
    prefix = "" if setting == "order_I" else f"{setting}_"
    load = lambda k: dict(np.load(OUT / f"{prefix}{k.replace(':', '_')}.npz"))  # noqa: E731
    val, dev = load("val:100"), [load(k) for k in DEV_RUNS]
    current = lambda r: r["u"] < 0.01  # noqa: E731
    fam, logistic = rule_families(val)
    cur_dev = [effect(r, current(r)) for r in dev]
    out = {"setting": setting, "current": {"val": effect(val, current(val)),
                                           "dev_mean": {k: float(np.mean([e[k] for e in cur_dev])) for k in ("transfer", "average")}},
           "logistic": logistic, "families": {}}
    log(f"{setting} current (u < 0.01): val dT {out['current']['val']['transfer']:+.3f} dA {out['current']['val']['average']:+.3f}; "
        f"dev dT {out['current']['dev_mean']['transfer']:+.3f} dA {out['current']['dev_mean']['average']:+.3f}")
    for name, rules in fam.items():
        ok = [(effect(val, f(val)), p, f) for p, f in rules]
        ok = [x for x in ok if x[0]["average"] >= 0]
        if not ok:
            continue
        ev, params, f = max(ok, key=lambda x: x[0]["transfer"])
        d = [effect(r, f(r)) for r in dev]
        dT = [x["transfer"] - c["transfer"] for x, c in zip(d, cur_dev)]
        dA = [x["average"] - c["average"] for x, c in zip(d, cur_dev)]
        out["families"][name] = {"params": [float(v) for v in params], "val": ev,
                                 "dev_vs_current": {"transfer": dT, "average": dA}}
        log(f"  {name:12s} {params}: val dT {ev['transfer']:+.3f} dA {ev['average']:+.3f} | dev vs. current: "
            f"dT {np.mean(dT):+.3f} ({min(dT):+.2f}..{max(dT):+.2f}) dA {np.mean(dA):+.3f}")
    json.dump(out, open(OUT / f"{prefix}analysis.json", "w"), indent=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--collect", action="store_true")
    p.add_argument("--analyze", action="store_true")
    p.add_argument("--runs", default="val:100," + ",".join(DEV_RUNS))
    p.add_argument("--setting", default="order_I", choices=SETTINGS)
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    prefix = "" if a.setting == "order_I" else f"{a.setting}_"
    if a.collect:
        domains, shots = SETTINGS[a.setting]
        data = load_with_extra("single", list(domains))
        for r in a.runs.split(","):
            path = OUT / f"{prefix}{r.replace(':', '_')}.npz"
            if path.exists():
                continue
            split, seed = r.split(":")
            np.savez(path, **collect(data, split, int(seed), shots))
    if a.analyze:
        analyze(a.setting)


if __name__ == "__main__":
    main()
