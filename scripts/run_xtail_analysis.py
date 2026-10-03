"""X-TAIL 10 domains, RAIL's single prompt, ML-RAIL features.

--part headroom: ML-RAIL under RAIL's gate (G0), an oracle-membership gate (G1) and an
    oracle-domain gate (G2), validation seed 100 and test seeds 1-5, plus G0's misrouting rates.
--part val: per-domain choices by leave-one-out (C0 control, A1, A2) and the gated
    residual (A4) vs. ML-RAIL, validation seed 100.
--part test --candidates A1,A4 (the registered test rule): seeds 1-5 vs. ML-RAIL.

Usage (gentle mode):
    python scripts/run_xtail_analysis.py --part headroom
"""
from __future__ import annotations

import argparse
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
APRIME = ("global", "cls10", "patch_mean", "gem8", "gem10")
DIMS = [512 if b in ("global", "patch_mean") else 768 for b in APRIME]
EDGES = np.cumsum([0] + DIMS)
SLICES = [(int(EDGES[i]), int(EDGES[i + 1])) for i in range(len(DIMS))]
VAL_SEED, TEST_SEEDS = 100, (1, 2, 3, 4, 5)
WEIGHTS = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def aprime():
    return xp.DualRAIL(gamma=0.2, lam=1e-3, dtype=torch.float32)


CANDIDATES = {  # name -> (model factory, fusion weights evaluated; the first is the registered one)
    "Aprime": (aprime, [0.8]),
    "C0": (lambda: xp.PerDomainKRR(SLICES), [0.8]),
    "A1": (lambda: xp.PerDomainKRR(SLICES, gammas=(0.2, 0.05, 0.1, 0.4, 0.8), lams=(1e-3, 1e-4, 1e-2, 1e-1),
                                   weights=WEIGHTS), ["method", 0.8]),
    "A2": (lambda: xp.PerDomainKRR(SLICES, gammas=(0.2,), lams=(1e-3, 1e-4, 1e-2, 1e-1), block_mults=(0.0, 0.5, 1.0, 2.0),
                                   weights=WEIGHTS), ["method", 0.8]),
    "A1w": (lambda: xp.JointLOOWeight(gamma=0.2, lam=1e-3, weights=WEIGHTS), ["method", 0.8]),
    "A4": (lambda: xp.GatedResidual(gamma=0.2, lam=1e-3, dtype=torch.float32), list(WEIGHTS)),
}


def misrouting(data, zs):
    """RAIL's gate errors after each step t: share of learned-domain images whose zero-shot top-1 is not
    a learned class (they lose the adapter) and of unseen-domain images whose top-1 is (they get blended)."""
    names, learned, missed, captured = list(data), 0, [], []
    for t, n in enumerate(names):
        learned += data[n]["n_classes"]
        for j, m in enumerate(names):
            inside = float((zs[m].argmax(1) < learned).float().mean())
            (missed if j <= t else captured).append(100 * (1 - inside) if j <= t else 100 * inside)
    return {"missed_learned_pct": float(np.mean(missed)), "captured_unseen_pct": float(np.mean(captured))}


def headroom():
    data = load_with_extra("single")
    feat = blocks_feature(APRIME)
    results = []
    for split, seeds in (("val", (VAL_SEED,)), ("test", TEST_SEEDS)):
        zs = xp.zero_shot_probs(data, split=split)
        route = misrouting(data, zs)
        log(f"{split}: RAIL gate misses {route['missed_learned_pct']:.1f}% of learned-domain images, "
            f"captures {route['captured_unseen_pct']:.1f}% of unseen-domain images (mean over cells)")
        for seed in seeds:
            for gate in ("rail", "oracle", "oracle_domain"):
                t0 = time.time()
                table = xp.run_stream(data, aprime(), seed, feat=feat, fusion_weight=0.8, zs=zs, split=split, gate=gate)
                r = {"split": split, "seed": seed, "gate": gate, **xp.metrics(table), **route, "table": table.tolist()}
                results.append(r)
                json.dump(results, open(OUT / "gate_headroom.json", "w"), indent=1)
                log(f"{split} seed {seed} {gate}: {r['transfer']:.2f} / {r['average']:.2f} / {r['last']:.2f} [{time.time() - t0:.0f}s]")
    for split in ("val", "test"):
        for gate in ("oracle", "oracle_domain"):
            diff = [{k: a[k] - b[k] for k in ("transfer", "average", "last")}
                    for a in results if a["split"] == split and a["gate"] == gate
                    for b in results if b["split"] == split and b["gate"] == "rail" and b["seed"] == a["seed"]]
            log(f"{split} {gate} - rail (mean): " + " / ".join(f"{k} {np.mean([d[k] for d in diff]):+.2f}" for k in ("transfer", "average", "last")))


GATES = ("rail",) + tuple(f"mass:{t}" for t in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5)) + tuple(f"topk:{k}" for k in (2, 3, 5))


def gates():
    """Looser zero-shot gates. Validation picks the best Average among gates with Transfer
    within 0.2 of RAIL's; then RAIL vs. that gate on test seeds 1-5 (the registered rule)."""
    data = load_with_extra("single")
    feat = blocks_feature(APRIME)
    path = OUT / "gates.json"
    results = json.load(open(path)) if path.exists() else []

    def run(split, seed, gate, zs):
        if any(r["split"] == split and r["seed"] == seed and r["gate"] == gate for r in results):
            return next(r for r in results if r["split"] == split and r["seed"] == seed and r["gate"] == gate)
        t0 = time.time()
        table = xp.run_stream(data, aprime(), seed, feat=feat, fusion_weight=0.8, zs=zs, split=split, gate=gate)
        r = {"split": split, "seed": seed, "gate": gate, **xp.metrics(table), "table": table.tolist()}
        results.append(r)
        json.dump(results, open(path, "w"), indent=1)
        log(f"{split} seed {seed} {gate}: {r['transfer']:.2f} / {r['average']:.2f} / {r['last']:.2f} [{time.time() - t0:.0f}s]")
        return r

    zs = xp.zero_shot_probs(data, split="val")
    val = {g: run("val", VAL_SEED, g, zs) for g in GATES}
    ok = [g for g in GATES[1:] if val[g]["transfer"] >= val["rail"]["transfer"] - 0.2]
    best = max(ok, key=lambda g: val[g]["average"]) if ok else None
    log(f"validation choice: {best}" + (f" (Average {val[best]['average'] - val['rail']['average']:+.2f})" if best else ""))
    if best is None or val[best]["average"] - val["rail"]["average"] < 0.5:
        log("validation bar not met: no test run")
        return
    zs = xp.zero_shot_probs(data, split="test")
    diffs = []
    for seed in TEST_SEEDS:
        a, b = run("test", seed, "rail", zs), run("test", seed, best, zs)
        diffs.append({k: b[k] - a[k] for k in ("transfer", "average", "last")})
    log("test (gate - rail) per seed: " + "; ".join(" / ".join(f"{d[k]:+.2f}" for k in ("transfer", "average", "last")) for d in diffs))


def run_candidates(names, split, seeds, out_name):
    data = load_with_extra("single")
    feat = blocks_feature(APRIME)
    zs = xp.zero_shot_probs(data, split=split)
    path = OUT / out_name
    results = json.load(open(path)) if path.exists() else []
    done = {(r["name"], r["seed"]) for r in results}
    for seed in seeds:
        for name in names:
            if (name, seed) in done:
                continue
            make, weights = CANDIDATES[name]
            t0 = time.time()
            m = make()
            tables = xp.run_stream(data, m, seed, feat=feat, fusion_weight=list(weights), zs=zs, split=split)
            for w, table in tables.items():
                results.append({"name": name, "seed": seed, "w": w, **xp.metrics(table),
                                "chosen": getattr(m, "chosen", None)})
                log(f"{split} seed {seed} {name} w={w}: {results[-1]['transfer']:.2f} / {results[-1]['average']:.2f} / {results[-1]['last']:.2f}")
            json.dump(results, open(path, "w"), indent=1)
            log(f"  [{time.time() - t0:.0f}s]")
            if getattr(m, "chosen", None):
                log("  per-domain choices: " + "; ".join(f"{n}: " + ",".join(f"{k}={v}" for k, v in c.items())
                                                         for n, c in zip(data, m.chosen)))
            del m
            torch.cuda.empty_cache()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--part", choices=("headroom", "val", "test", "gates"), required=True)
    p.add_argument("--candidates", default="Aprime,C0,A1,A2,A4")
    a = p.parse_args()
    if a.part == "headroom":
        headroom()
    elif a.part == "gates":
        gates()
    elif a.part == "val":
        run_candidates(a.candidates.split(","), "val", (VAL_SEED,), "perdomain_val.json")
    else:
        run_candidates(["Aprime"] + [c for c in a.candidates.split(",") if c != "Aprime"], "test", TEST_SEEDS, "perdomain_test.json")
