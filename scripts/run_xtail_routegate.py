"""X-TAIL 10 domains, RAIL's single prompt, ML-RAIL features.

--part gate-select: per-domain conformal rescue gate. Grid on validation (seed 100) and test seed 1 (for the
    held-out-domain check), then, if (i) and (ii) pass, test seeds 2-5 for RAIL vs. the chosen gate.
--part route: route-then-restrict (step 1 measurement): ML-RAIL with and without routing, validation and
    test seeds 1-5.
--part route-experts: step 2, per-domain leave-one-out models behind the routing (only if step 1 allows).

Usage (gentle mode):
    python scripts/run_xtail_routegate.py --part route
"""
from __future__ import annotations

import argparse
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
from scripts.run_xtail_analysis import APRIME, SLICES, aprime  # noqa: E402

OUT = REPO_ROOT / "data" / "generated" / "xtail"
VAL_SEED, TEST_SEEDS = 100, (1, 2, 3, 4, 5)
QS = (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9)
MASSES = (0.0, 0.01, 0.05, 0.1)
SPLITS = {"halves": set(range(5)), "even_odd": set(range(0, 10, 2)),
          "random": set(np.random.default_rng(0).choice(10, 5, replace=False).tolist())}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Runner:
    def __init__(self, path):
        self.data = load_with_extra("single")
        self.feat = blocks_feature(APRIME)
        self.zs = {s: xp.zero_shot_probs(self.data, split=s) for s in ("val", "test")}
        self.path = path
        self.results = json.load(open(path)) if path.exists() else []

    def run(self, split, seed, name, make_method, gate="rail", route=False):
        for r in self.results:
            if r["split"] == split and r["seed"] == seed and r["name"] == name:
                return r
        t0 = time.time()
        g = gate() if callable(gate) else gate
        table = xp.run_stream(self.data, make_method(), seed, feat=self.feat, fusion_weight=0.8,
                              zs=self.zs[split], split=split, gate=g, route=route)
        r = {"split": split, "seed": seed, "name": name, **xp.metrics(table), "table": table.tolist()}
        self.results.append(r)
        json.dump(self.results, open(self.path, "w"), indent=1)
        log(f"{split} seed {seed} {name}: {r['transfer']:.2f} / {r['average']:.2f} / {r['last']:.2f} [{time.time() - t0:.0f}s]")
        return r


def column_gain(r, base, cols):
    """Contribution of the given columns (domains) to Average, method minus baseline."""
    d = np.array(r["table"]) - np.array(base["table"])
    return float(d[:, sorted(cols)].sum() / 100)


def gate_select():
    R = Runner(OUT / "conformal_gate.json")
    configs = [(q, m) for m, q in itertools.product(MASSES, QS)]
    name = lambda q, m: f"conformal_q{q}_m{m}"  # noqa: E731
    val = {"rail": R.run("val", VAL_SEED, "rail", aprime)}
    t1 = {"rail": R.run("test", 1, "rail", aprime)}
    for q, m in configs:
        val[(q, m)] = R.run("val", VAL_SEED, name(q, m), aprime, gate=lambda q=q, m=m: xp.ConformalGate(q, m))
        t1[(q, m)] = R.run("test", 1, name(q, m), aprime, gate=lambda q=q, m=m: xp.ConformalGate(q, m))
    ok = [c for c in configs if val[c]["transfer"] >= val["rail"]["transfer"] - 0.2]
    best = max(ok, key=lambda c: val[c]["average"]) if ok else None
    gain = val[best]["average"] - val["rail"]["average"] if best else 0.0
    log(f"(i) validation choice {best}: Average {gain:+.2f} (bar +0.5)")
    held = {}
    for sname, A in SPLITS.items():
        B = set(range(10)) - A
        tot = 0.0
        for fit, score in ((A, B), (B, A)):
            c = max(ok, key=lambda c: column_gain(val[c], val["rail"], fit))
            tot += column_gain(t1[c], t1["rail"], score)
        held[sname] = tot
    log("(ii) held-out-domain test gain (bar +0.3 each): " + ", ".join(f"{k} {v:+.2f}" for k, v in held.items()))
    if best is None or gain < 0.5 or min(held.values()) < 0.3:
        log("FAIL at (i) or (ii): no further test seeds")
        return
    diffs = []
    for seed in TEST_SEEDS:
        a = R.run("test", seed, "rail", aprime)
        b = R.run("test", seed, name(*best), aprime, gate=lambda: xp.ConformalGate(*best))
        diffs.append({k: b[k] - a[k] for k in ("transfer", "average", "last")})
    log("(iii) test gate - rail per seed: " + "; ".join(" / ".join(f"{d[k]:+.2f}" for k in ("transfer", "average", "last")) for d in diffs))


def route():
    R = Runner(OUT / "route_restrict.json")
    for split, seeds in (("val", (VAL_SEED,)), ("test", TEST_SEEDS)):
        diffs = []
        for seed in seeds:
            a = R.run(split, seed, "Aprime", aprime)
            b = R.run(split, seed, "Aprime_route", aprime, route=True)
            diffs.append({k: b[k] - a[k] for k in ("transfer", "average", "last")})
        log(f"{split} route - joint (mean): " + " / ".join(f"{k} {np.mean([d[k] for d in diffs]):+.2f}" for k in ("transfer", "average", "last")))


def per_domain():
    return xp.PerDomainKRR(SLICES, gammas=(0.2, 0.05, 0.1, 0.4, 0.8), lams=(1e-3, 1e-4, 1e-2, 1e-1))


def route_experts():
    R = Runner(OUT / "route_restrict.json")
    a = R.run("val", VAL_SEED, "Aprime", aprime)
    b = R.run("val", VAL_SEED, "PerDomainLOO_route", per_domain, route=True)
    c = R.run("val", VAL_SEED, "Aprime_route", aprime, route=True)
    log(f"validation: per-domain LOO + route vs ML-RAIL: Average {b['average'] - a['average']:+.2f}, Last {b['last'] - a['last']:+.2f} "
        f"(ML-RAIL + route: {c['average'] - a['average']:+.2f} / {c['last'] - a['last']:+.2f})")
    if max(b["average"] - a["average"], b["last"] - a["last"]) < 0.5:
        log("validation bar not met: no test run")
        return
    diffs = []
    for seed in TEST_SEEDS:
        x, y = R.run("test", seed, "Aprime", aprime), R.run("test", seed, "PerDomainLOO_route", per_domain, route=True)
        diffs.append({k: y[k] - x[k] for k in ("transfer", "average", "last")})
    log("test per-domain LOO + route - ML-RAIL per seed: " + "; ".join(" / ".join(f"{d[k]:+.2f}" for k in ("transfer", "average", "last")) for d in diffs))


def cascade_self():
    m = aprime()
    return xp.Cascade(m, m)


def cascade_loo():
    return xp.Cascade(aprime(), per_domain())


def cascade():
    """Amendment: cascade routing by ML-RAIL's own top learned class; step 1' (ML-RAIL as its own expert, cost
    <= 0.3), step 2' (per-domain leave-one-out experts)."""
    R = Runner(OUT / "route_restrict.json")
    a = R.run("val", VAL_SEED, "Aprime", aprime)
    c = R.run("val", VAL_SEED, "Cascade_self", cascade_self)
    cost = (c["average"] - a["average"], c["last"] - a["last"])
    log(f"step 1' cascade(ML-RAIL, ML-RAIL) - ML-RAIL: Average {cost[0]:+.2f}, Last {cost[1]:+.2f} (allowed -0.3)")
    if min(cost) < -0.3:
        log("step 1' FAIL")
        return
    b = R.run("val", VAL_SEED, "Cascade_LOO", cascade_loo)
    log(f"step 2' validation cascade(ML-RAIL, per-domain LOO) - ML-RAIL: Average {b['average'] - a['average']:+.2f}, Last {b['last'] - a['last']:+.2f}")
    if max(b["average"] - a["average"], b["last"] - a["last"]) < 0.5:
        log("step 2' validation bar not met: no test run")
        return
    diffs = []
    for seed in TEST_SEEDS:
        x, y = R.run("test", seed, "Aprime", aprime), R.run("test", seed, "Cascade_LOO", cascade_loo)
        diffs.append({k: y[k] - x[k] for k in ("transfer", "average", "last")})
    log("step 2' test cascade - ML-RAIL per seed: " + "; ".join(" / ".join(f"{d[k]:+.2f}" for k in ("transfer", "average", "last")) for d in diffs))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--part", choices=("gate-select", "route", "route-experts", "cascade"), required=True)
    {"gate-select": gate_select, "route": route, "route-experts": route_experts, "cascade": cascade}[p.parse_args().part]()
