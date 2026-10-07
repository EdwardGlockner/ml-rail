"""Routing and gating experiments on X-TAIL (10 domains, RAIL's single prompt, ML-RAIL features).

--part gate-select     rescue gate: grid on validation (seed 100) and test seed 1, held-out-domain
                       check, then test seeds 2-5 against RAIL's gate
--part route           restrict each admitted image to the domain of its top zero-shot class
--part route-experts   per-domain leave-one-out models behind that routing
--part cascade         route by ML-RAIL's own top class; per-domain experts re-score within the domain
--part confirm         the gate (frozen settings) on task order II and the 11-domain 5-shot variant
--part final           separate fusion weight for rescued images; selection and all three settings

Usage (low-load settings, see README):
    python scripts/run_xtail_routegate.py --part final
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

from evaluation import xtail_protocol as xp  # noqa: E402
from evaluation import xtail_variants as xv  # noqa: E402
from evaluation.ml_rail import blocks_feature, load_with_extra  # noqa: E402
from evaluation.ml_rail import BLOCKS, SLICES, make_ml_rail  # noqa: E402

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
        self.feat = blocks_feature(BLOCKS)
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
    val = {"rail": R.run("val", VAL_SEED, "rail", make_ml_rail)}
    t1 = {"rail": R.run("test", 1, "rail", make_ml_rail)}
    for q, m in configs:
        val[(q, m)] = R.run("val", VAL_SEED, name(q, m), make_ml_rail, gate=lambda q=q, m=m: xp.DomainThresholdGate(q, m))
        t1[(q, m)] = R.run("test", 1, name(q, m), make_ml_rail, gate=lambda q=q, m=m: xp.DomainThresholdGate(q, m))
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
        a = R.run("test", seed, "rail", make_ml_rail)
        b = R.run("test", seed, name(*best), make_ml_rail, gate=lambda: xp.DomainThresholdGate(*best))
        diffs.append({k: b[k] - a[k] for k in ("transfer", "average", "last")})
    log("(iii) test gate - rail per seed: " + "; ".join(" / ".join(f"{d[k]:+.2f}" for k in ("transfer", "average", "last")) for d in diffs))


def route():
    R = Runner(OUT / "route_restrict.json")
    for split, seeds in (("val", (VAL_SEED,)), ("test", TEST_SEEDS)):
        diffs = []
        for seed in seeds:
            a = R.run(split, seed, "Aprime", make_ml_rail)
            b = R.run(split, seed, "Aprime_route", make_ml_rail, route=True)
            diffs.append({k: b[k] - a[k] for k in ("transfer", "average", "last")})
        log(f"{split} route - joint (mean): " + " / ".join(f"{k} {np.mean([d[k] for d in diffs]):+.2f}" for k in ("transfer", "average", "last")))


def per_domain():
    return xv.PerDomainKRR(SLICES, gammas=(0.2, 0.05, 0.1, 0.4, 0.8), lams=(1e-3, 1e-4, 1e-2, 1e-1))


def route_experts():
    R = Runner(OUT / "route_restrict.json")
    a = R.run("val", VAL_SEED, "Aprime", make_ml_rail)
    b = R.run("val", VAL_SEED, "PerDomainLOO_route", per_domain, route=True)
    c = R.run("val", VAL_SEED, "Aprime_route", make_ml_rail, route=True)
    log(f"validation: per-domain LOO + route vs ML-RAIL: Average {b['average'] - a['average']:+.2f}, Last {b['last'] - a['last']:+.2f} "
        f"(ML-RAIL + route: {c['average'] - a['average']:+.2f} / {c['last'] - a['last']:+.2f})")
    if max(b["average"] - a["average"], b["last"] - a["last"]) < 0.5:
        log("validation bar not met: no test run")
        return
    diffs = []
    for seed in TEST_SEEDS:
        x, y = R.run("test", seed, "Aprime", make_ml_rail), R.run("test", seed, "PerDomainLOO_route", per_domain, route=True)
        diffs.append({k: y[k] - x[k] for k in ("transfer", "average", "last")})
    log("test per-domain LOO + route - ML-RAIL per seed: " + "; ".join(" / ".join(f"{d[k]:+.2f}" for k in ("transfer", "average", "last")) for d in diffs))


def cascade_self():
    m = make_ml_rail()
    return xv.Cascade(m, m)


def cascade_loo():
    return xv.Cascade(make_ml_rail(), per_domain())


def cascade():
    """Amendment: cascade routing by ML-RAIL's own top learned class; step 1' (ML-RAIL as its own expert, cost
    <= 0.3), step 2' (per-domain leave-one-out experts)."""
    R = Runner(OUT / "route_restrict.json")
    a = R.run("val", VAL_SEED, "Aprime", make_ml_rail)
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
        x, y = R.run("test", seed, "Aprime", make_ml_rail), R.run("test", seed, "Cascade_LOO", cascade_loo)
        diffs.append({k: y[k] - x[k] for k in ("transfer", "average", "last")})
    log("step 2' test cascade - ML-RAIL per seed: " + "; ".join(" / ".join(f"{d[k]:+.2f}" for k in ("transfer", "average", "last")) for d in diffs))


def confirm():
    """The rescue gate (q 0.25, m 0.05, frozen) vs. RAIL's gate on order II and the
    11-domain 5-shot variant, seeds 0-4."""
    from data.real.xtail import DOMAINS_11, ORDER_II
    path = OUT / "conformal_confirm.json"
    results = json.load(open(path)) if path.exists() else []
    for bench, domains, shots in (("orderII", list(ORDER_II), 16), ("11dom_5shot", list(DOMAINS_11), 5)):
        data = load_with_extra("single", domains)
        feat = blocks_feature(BLOCKS)
        zs = xp.zero_shot_probs(data)
        diffs = []
        for seed in range(5):
            row = {}
            for name in ("rail", "conformal"):
                prev = [r for r in results if r["bench"] == bench and r["seed"] == seed and r["name"] == name]
                if prev:
                    row[name] = prev[0]
                    continue
                t0 = time.time()
                gate = xp.DomainThresholdGate(0.25, 0.05) if name == "conformal" else "rail"
                table = xp.run_stream(data, make_ml_rail(), seed, feat=feat, fusion_weight=0.8, shots=shots, zs=zs, gate=gate)
                row[name] = {"bench": bench, "seed": seed, "name": name, **xp.metrics(table), "table": table.tolist()}
                results.append(row[name])
                json.dump(results, open(path, "w"), indent=1)
                log(f"{bench} seed {seed} {name}: {row[name]['transfer']:.2f} / {row[name]['average']:.2f} / {row[name]['last']:.2f} [{time.time() - t0:.0f}s]")
            diffs.append({k: row["conformal"][k] - row["rail"][k] for k in ("transfer", "average", "last")})
        mean = {k: float(np.mean([d[k] for d in diffs])) for k in ("transfer", "average", "last")}
        pos = sum(d["average"] > 0 for d in diffs)
        ok = mean["average"] >= 0.5 and pos >= 4 and mean["transfer"] >= -0.2 and mean["last"] >= -0.2
        log(f"{bench}: conformal - rail mean {mean['transfer']:+.3f} / {mean['average']:+.3f} / {mean['last']:+.3f}, "
            f"Average positive on {pos}/5 -> {'PASS' if ok else 'FAIL'}")


WR = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


def load_bench(domains):
    return load_with_extra("single", domains)


def final():
    """A separate (smaller) fusion weight w_r for images the rescue gate rescues."""
    from data.real.xtail import DOMAINS_11, ORDER_II
    path = OUT / "gate_rescue_weight.json"
    results = json.load(open(path)) if path.exists() else []
    feat = blocks_feature(BLOCKS)

    def run(bench, data, split, seed, q, m, shots=16):
        """Tables for every w_r of one (q, m) gate, cached by name."""
        names = {wr: f"q{q}_m{m}_wr{wr}" for wr in WR}
        have = {r["name"]: r for r in results if r["bench"] == bench and r["split"] == split and r["seed"] == seed}
        if all(n in have for n in names.values()):
            return {wr: have[n] for wr, n in names.items()}
        t0 = time.time()
        zs = xp.zero_shot_probs(data, split=split)
        gate = xp.DomainThresholdGate(q, m)
        tables = xp.run_stream(data, make_ml_rail(), seed, feat=feat, fusion_weight=[(0.8, wr) for wr in WR],
                               shots=shots, zs=zs, split=split, gate=gate)
        out = {}
        for wr in WR:
            r = {"bench": bench, "split": split, "seed": seed, "name": names[wr], **xp.metrics(tables[(0.8, wr)]),
                 "table": tables[(0.8, wr)].tolist()}
            results.append(r)
            out[wr] = r
        json.dump(results, open(path, "w"), indent=1)
        best = max(out.values(), key=lambda r: r["average"])
        log(f"{bench} {split} seed {seed} q{q} m{m}: best w_r {best['name']} {best['transfer']:.2f} / {best['average']:.2f} [{time.time() - t0:.0f}s]")
        return out

    # RAIL's-gate baselines from the earlier runs (same harness, deterministic)
    gate_runs = json.load(open(OUT / "conformal_gate.json"))
    confirm_runs = json.load(open(OUT / "conformal_confirm.json"))
    rail = {("I", "val", VAL_SEED): next(r for r in gate_runs if r["split"] == "val" and r["name"] == "rail")}
    for s in TEST_SEEDS:
        rail[("I", "test", s)] = next(r for r in gate_runs if r["split"] == "test" and r["seed"] == s and r["name"] == "rail")
    for bench in ("orderII", "11dom_5shot"):
        for s in range(5):
            rail[(bench, "test", s)] = next(r for r in confirm_runs if r["bench"] == bench and r["seed"] == s and r["name"] == "rail")

    dataI = load_with_extra("single")
    val, t1 = {}, {}
    for q in (0.25, 0.5):
        for m in (0.05, 0.1):
            for wr, r in run("I", dataI, "val", VAL_SEED, q, m).items():
                val[(q, m, wr)] = r
            for wr, r in run("I", dataI, "test", 1, q, m).items():
                t1[(q, m, wr)] = r
    base_v, base_t1 = rail[("I", "val", VAL_SEED)], rail[("I", "test", 1)]
    ok = [c for c in val if val[c]["transfer"] >= base_v["transfer"] - 0.1]
    if not ok:
        log("no configuration within 0.1 Transfer on validation: FAIL")
        return
    best = max(ok, key=lambda c: val[c]["average"])
    log(f"selection: {best}: validation Transfer {val[best]['transfer'] - base_v['transfer']:+.2f}, "
        f"Average {val[best]['average'] - base_v['average']:+.2f}")
    held = {}
    for sname, A in SPLITS.items():
        B = set(range(10)) - A
        held[sname] = sum(column_gain(t1[max(ok, key=lambda c: column_gain(val[c], base_v, fit))], base_t1, score)
                          for fit, score in ((A, B), (B, A)))
    log("held-out-domain test gain (bar +0.3 each): " + ", ".join(f"{k} {v:+.2f}" for k, v in held.items()))
    q, m, wr = best
    verdicts = {}
    for bench, data, seeds, shots in (("I", dataI, TEST_SEEDS, 16), ("orderII", load_bench(list(ORDER_II)), range(5), 16),
                                      ("11dom_5shot", load_bench(list(DOMAINS_11)), range(5), 5)):
        diffs = []
        for s in seeds:
            r = run(bench, data, "test", s, q, m, shots)[wr]
            b = rail[(bench, "test", s)]
            diffs.append({k: r[k] - b[k] for k in ("transfer", "average", "last")})
        mean = {k: float(np.mean([d[k] for d in diffs])) for k in ("transfer", "average", "last")}
        pos = sum(d["average"] > 0 for d in diffs)
        verdicts[bench] = mean["average"] >= 0.5 and pos >= 4 and mean["transfer"] >= -0.2 and mean["last"] >= -0.2
        log(f"{bench}: gate - rail mean {mean['transfer']:+.3f} / {mean['average']:+.3f} / {mean['last']:+.3f}, "
            f"{pos}/5 positive -> {'PASS' if verdicts[bench] else 'FAIL'}")
    heldout_ok = min(held.values()) >= 0.3
    log(f"GATE RULE: {'PASS' if all(verdicts.values()) and heldout_ok else 'FAIL'} "
        f"(held-out {'ok' if heldout_ok else 'below bar'}; settings {verdicts})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--part", choices=("gate-select", "route", "route-experts", "cascade", "confirm", "final"), required=True)
    {"gate-select": gate_select, "route": route, "route-experts": route_experts, "cascade": cascade,
     "confirm": confirm, "final": final}[p.parse_args().part]()
