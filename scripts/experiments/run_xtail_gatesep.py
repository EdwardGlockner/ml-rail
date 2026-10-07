"""Can any signal available at test time separate the
images whose admission past RAIL's gate would help (gain) from those it would hurt (loss)?

Each image RAIL's gate rejects at steps t < last is labeled gain / loss / neutral by what admitting it
would do to its prediction (ML-RAIL, fusion 0.8), weighted by its exact effect on Average (1 / its domain's
test-set size). Per signal: weighted AUC, and the net Average gain of the best threshold chosen on
validation (seed 100) applied to test (seed 1). s7 = logistic regression on s1-s6, fit on validation.

Usage (low-load settings, see README):
    python scripts/experiments/run_xtail_gatesep.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from evaluation import xtail_protocol as xp  # noqa: E402
from evaluation import xtail_variants as xv  # noqa: E402
from evaluation.ml_rail import blocks_feature, load_with_extra  # noqa: E402
from evaluation.ml_rail import BLOCKS, make_ml_rail  # noqa: E402

OUT = REPO_ROOT / "data" / "generated" / "xtail"
SIGNALS = ("s1_learned_mass", "s2_top_learned_prob", "s3_adapter_max", "s4_kernel_max", "s5_profile_max", "s6_product_max")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def collect(data, split, seed, feat):
    names = list(data)
    zs = xp.zero_shot_probs(data, split=split)
    text = torch.as_tensor(np.concatenate([data[n]["text"] for n in names]), device=xp.DEV)
    model, learned, rows = make_ml_rail(), 0, []
    for t, n in enumerate(names[:-1]):  # the last step admits everything: nothing to separate
        d = data[n]
        idx = xp.fewshot(d, seed, 16, t)
        model.add(feat(d, "train", idx), d["train_labels"][idx] + d["offset"])
        learned += d["n_classes"]
        text_l = text[:learned]
        q_store = torch.cat([xv.learned_zero_shot(model.X[i : i + 4096], text_l) for i in range(0, len(model.X), 4096)]).sqrt()
        for j, m in enumerate(names):
            e = data[m]
            p = zs[m]
            cand = (p.argmax(1) >= learned).numpy()
            if not cand.any():
                continue
            X = torch.as_tensor(feat(e, split)[cand], device=xp.DEV)
            s = torch.as_tensor(model.scores(X), dtype=torch.float32)
            pc, y = p[cand], torch.as_tensor(e[f"{split}_labels"][cand] + e["offset"])
            fused = 0.2 * pc
            fused[:, :learned] += 0.8 * s
            ok_zs, ok_fused = pc.argmax(1) == y, fused.argmax(1) == y
            label = ok_fused.int() - ok_zs.int()  # +1 gain, -1 loss, 0 neutral
            keep = label != 0
            if not keep.any():
                continue
            Xk = X[keep.to(xp.DEV)]
            q = xv.learned_zero_shot(Xk, text_l).sqrt()
            s4, s5, s6 = [], [], []
            for i in range(0, len(Xk), 512):
                K = model._k(Xk[i : i + 512], model.X)
                BC = q[i : i + 512] @ q_store.T
                s4.append(K.max(1).values.cpu())
                s5.append(BC.max(1).values.cpu())
                s6.append((K * BC).max(1).values.cpu())
            rows.append({
                "label": label[keep].numpy(), "w": np.full(int(keep.sum()), 1.0 / len(cand)),
                "learned_domain": np.full(int(keep.sum()), j <= t), "domain": np.full(int(keep.sum()), j),
                "s1_learned_mass": pc[keep][:, :learned].sum(1).numpy(),
                "s2_top_learned_prob": pc[keep][:, :learned].max(1).values.numpy(),
                "s3_adapter_max": s[keep].max(1).values.numpy(),
                "s4_kernel_max": torch.cat(s4).numpy(), "s5_profile_max": torch.cat(s5).numpy(),
                "s6_product_max": torch.cat(s6).numpy(),
            })
        log(f"{split} step {t}: {sum(len(r['label']) for r in rows)} labeled images so far")
    return {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}


def best_threshold(score, label, w):
    """Admit score >= tau (direction +1) or score <= tau (-1); returns (net gain, tau, direction) maximizing
    the weighted sum of labels admitted, on these data (admitting nothing = 0)."""
    best = (0.0, np.inf, 1)
    for direction in (1, -1):
        order = np.argsort(-direction * score)
        cum = np.cumsum((label * w)[order])
        k = int(np.argmax(cum))
        if cum[k] > best[0]:
            best = (float(cum[k]), float(score[order][k]), direction)
    return best


def apply(score, label, w, tau, direction):
    if not np.isfinite(tau):
        return 0.0
    admit = score >= tau if direction == 1 else score <= tau
    return float((label * w)[admit].sum())


def design(D):
    cols = [D[s] for s in SIGNALS]
    cols += [np.log(np.clip(D[s], 1e-8, None)) for s in SIGNALS if s != "s3_adapter_max"]
    return np.stack(cols, 1)


def main():
    data = load_with_extra("single")
    feat = blocks_feature(BLOCKS)
    V = collect(data, "val", 100, feat)
    T = collect(data, "test", 1, feat)
    np.savez(OUT / "gate_separability_val.npz", **V)
    np.savez(OUT / "gate_separability_test.npz", **T)
    out = {}
    for name, D in (("val", V), ("test", T)):
        lw = D["label"] * D["w"]
        out[f"{name}_oracle_learned_domain"] = float(lw[D["learned_domain"]].sum())
        out[f"{name}_total_gain"] = float(lw[D["label"] > 0].sum())
        out[f"{name}_total_loss"] = float(lw[D["label"] < 0].sum())
        log(f"{name}: oracle (admit learned-domain images) {out[f'{name}_oracle_learned_domain']:+.2f} Average; "
            f"all gains {out[f'{name}_total_gain']:+.2f}, all losses {out[f'{name}_total_loss']:+.2f}; "
            f"{int((D['label'] > 0).sum())} gain / {int((D['label'] < 0).sum())} loss images")
    mu, sd = design(V).mean(0), design(V).std(0) + 1e-8
    lr = LogisticRegression(max_iter=2000).fit((design(V) - mu) / sd, V["label"] > 0, sample_weight=V["w"])
    V["s7_logistic"] = lr.predict_proba((design(V) - mu) / sd)[:, 1]
    T["s7_logistic"] = lr.predict_proba((design(T) - mu) / sd)[:, 1]
    for s in SIGNALS + ("s7_logistic",):
        auc_v = roc_auc_score(V["label"] > 0, V[s], sample_weight=V["w"])
        auc_t = roc_auc_score(T["label"] > 0, T[s], sample_weight=T["w"])
        gain_v, tau, direction = best_threshold(V[s], V["label"], V["w"])
        gain_t = apply(T[s], T["label"], T["w"], tau, direction)
        out[s] = {"auc_val": auc_v, "auc_test": auc_t, "gain_val": gain_v, "tau": tau, "direction": direction, "gain_test": gain_t}
        log(f"{s}: AUC val {auc_v:.3f} test {auc_t:.3f}; best threshold on val {gain_v:+.2f} Average -> test {gain_t:+.2f}")
    json.dump(out, open(OUT / "gate_separability.json", "w"), indent=1)


if __name__ == "__main__":
    main()
