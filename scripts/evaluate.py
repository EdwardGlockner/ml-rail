"""Final evaluation on held-out seeds that no selection step has used.

Runs zero-shot CLIP, Dual-RAIL, ML-RAIL, ML-RAIL with the rescue gate, and ML-RAIL with the rescue and
reverse gates (the reverse gate on the full kernel, and on the layer-10 class token as reported) in all
three X-TAIL settings, with every setting frozen: Dual-RAIL uses RAIL's published hyperparameters;
ML-RAIL and the gates use the values selected on validation (see README, "How the result was reached").
The reported numbers are seeds 16-20. Seeds 11-15 are the held-out run of the full-kernel reverse gate,
and seeds 6-10 the first held-out run, before either reverse gate (later extended on order I for the
tuning-protocol table); see scripts/reproduce.py for the exact calls. Results are cached per run in
$XTAIL_ROOT/results/evaluate.json, so the script can be stopped and restarted.

Usage:
    python scripts/evaluate.py                       # all settings and methods, seeds 16-20
    python scripts/evaluate.py --seeds 6 --float64   # precision check on one seed
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

import torch  # noqa: E402

from data.real.xtail import DOMAINS, DOMAINS_11, ORDER_II, XTAIL_ROOT  # noqa: E402
from evaluation import xtail_protocol as xp  # noqa: E402
from evaluation.ml_rail import BLOCKS, SLICES, blocks_feature, load_with_extra  # noqa: E402

SETTINGS = {"order_I": (DOMAINS, 16), "order_II": (ORDER_II, 16), "11_domains_5shot": (DOMAINS_11, 5)}
GATE_Q, GATE_MIN_MASS = 0.25, 0.1  # rescue-gate settings selected on validation
REVERSE_Q = 0.01  # reverse gate, selected on validation (largest q with no loss in Average)
# reverse gate on the layer-10 class token, selected on validation (largest Transfer + Average)
REVERSE_CLS10_Q, CLS10 = 0.008, SLICES[BLOCKS.index("cls10")]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def methods(dtype):
    """name -> (model factory, feature, gate factory)."""
    ml_feat = blocks_feature(BLOCKS)
    return {
        "zero_shot": (xp.ZeroShot, "global", lambda: "rail"),
        "rail_dual": (lambda: xp.DualRAIL(gamma=0.1, lam=1e-3, dtype=dtype), "global", lambda: "rail"),
        "ml_rail": (lambda: xp.DualRAIL(gamma=0.2, lam=1e-3, dtype=dtype), ml_feat, lambda: "rail"),
        "ml_rail_gate": (lambda: xp.DualRAIL(gamma=0.2, lam=1e-3, dtype=dtype), ml_feat,
                         lambda: xp.DomainThresholdGate(GATE_Q, GATE_MIN_MASS)),
        "ml_rail_gate_reverse": (lambda: xp.DualRAIL(gamma=0.2, lam=1e-3, dtype=dtype), ml_feat,
                                 lambda: xp.DomainThresholdGate(GATE_Q, GATE_MIN_MASS, reverse_q=REVERSE_Q)),
        "ml_rail_gate_reverse_cls10": (lambda: xp.DualRAIL(gamma=0.2, lam=1e-3, dtype=dtype), ml_feat,
                                       lambda: xp.DomainThresholdGate(GATE_Q, GATE_MIN_MASS, reverse_q=REVERSE_CLS10_Q,
                                                                      reverse_slice=CLS10)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="16,17,18,19,20")
    p.add_argument("--settings", default=",".join(SETTINGS))
    p.add_argument("--methods", default="zero_shot,rail_dual,ml_rail,ml_rail_gate,ml_rail_gate_reverse,"
                                        "ml_rail_gate_reverse_cls10")
    p.add_argument("--float64", action="store_true", help="float64 kernel solves (default float32)")
    a = p.parse_args()
    dtype = torch.float64 if a.float64 else torch.float32
    path = XTAIL_ROOT / "results" / "evaluate.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    results = json.load(open(path)) if path.exists() else []
    done = {(r["setting"], r["method"], r["seed"], r["dtype"]) for r in results}
    for setting in a.settings.split(","):
        domains, shots = SETTINGS[setting]
        data = load_with_extra("single", list(domains))
        zs = xp.zero_shot_probs(data)
        for seed in (int(s) for s in a.seeds.split(",")):
            for name in a.methods.split(","):
                key = (setting, name, seed, str(dtype).split(".")[-1])
                if key in done:
                    continue
                make, feat, gate = methods(dtype)[name]
                t0 = time.time()
                table = xp.run_stream(data, make(), seed, feat=feat, fusion_weight=0.8, shots=shots, zs=zs, gate=gate())
                results.append({"setting": setting, "method": name, "seed": seed, "dtype": key[3],
                                **xp.metrics(table), "table": table.tolist()})
                json.dump(results, open(path, "w"), indent=1)
                m = results[-1]
                log(f"{setting} seed {seed} {name} ({key[3]}): {m['transfer']:.2f} / {m['average']:.2f} / {m['last']:.2f} "
                    f"[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
