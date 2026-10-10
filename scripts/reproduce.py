"""Reproduce every reported number, table and figure, from downloading the data to the README figures.
Each step caches its output under $XTAIL_ROOT, so an interrupted run continues where it stopped.

  setup        download and verify the datasets              scripts/setup_xtail_data.py
  encode       cache CLIP ViT-B/16 features                  scripts/xtail_encode.py
  encode_extra intermediate-layer features                   scripts/xtail_encode_extra.py
  evaluate     Tables 1-2 on the held-out seeds 11-15        scripts/evaluate.py
  replication  the earlier held-out run, seeds 6-10          scripts/evaluate.py --seeds 6,...,10
  replication_reverse  both gates on seeds 6-10, order I (table 3)
  tuning       Table 3: RAIL's first-domain tuning protocol  scripts/experiments/run_xtail_first_domain_tuning.py
  sensitivity  gate settings and acceptance rates            scripts/experiments/run_xtail_gate_sensitivity.py
  headroom     Transfer headroom and the reverse gate's      scripts/experiments/run_xtail_transfer_headroom.py
  headroom_II  development runs (order I, order II, 5-shot)
  headroom_5shot
  takeback     other take-back signals (negative result)    scripts/experiments/run_xtail_takeback_signals.py
  plot         figures, tables, results/xtail_results.json   scripts/plot_xtail_figures.py

Usage:
    python scripts/reproduce.py                    # all steps
    python scripts/reproduce.py --from evaluate    # skip the data and encoding steps
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STEPS = {
    "setup": "scripts/setup_xtail_data.py",
    "encode": "scripts/xtail_encode.py",
    "encode_extra": "scripts/xtail_encode_extra.py",
    "evaluate": "scripts/evaluate.py",
    "replication": "scripts/evaluate.py --seeds 6,7,8,9,10 --methods zero_shot,rail_dual,ml_rail,ml_rail_gate",
    "replication_reverse": "scripts/evaluate.py --seeds 6,7,8,9,10 --settings order_I --methods ml_rail_gate_reverse",
    "tuning": "scripts/experiments/run_xtail_first_domain_tuning.py",
    "sensitivity": "scripts/experiments/run_xtail_gate_sensitivity.py",
    "headroom": "scripts/experiments/run_xtail_transfer_headroom.py --runs val:100,test:0,test:1,test:2,test:3,test:4,test:5",
    "headroom_II": "scripts/experiments/run_xtail_transfer_headroom.py --setting order_II --runs test:1,test:2,test:3,test:4,test:5",
    "headroom_5shot": "scripts/experiments/run_xtail_transfer_headroom.py --setting 11_domains_5shot --runs test:1,test:2,test:3,test:4,test:5",
    "takeback": "scripts/experiments/run_xtail_takeback_signals.py --collect --analyze",
    "plot": "scripts/plot_xtail_figures.py",
}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--from", dest="start", default="setup", choices=STEPS)
    p.add_argument("--only", choices=STEPS)
    a = p.parse_args()
    names = [a.only] if a.only else list(STEPS)[list(STEPS).index(a.start):]
    for name in names:
        print(f"== {name}: {STEPS[name]}", flush=True)
        script, *args = STEPS[name].split()
        subprocess.run([sys.executable, str(REPO_ROOT / script), *args], cwd=REPO_ROOT, check=True)
