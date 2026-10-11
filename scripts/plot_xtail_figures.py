"""Figures, tables and results/xtail_results.json for the README, drawn from the final evaluation
(scripts/evaluate.py: seeds 16-20; seeds 11-15 and 6-10 are earlier held-out runs) and the checks in
scripts/experiments/ (run_xtail_first_domain_tuning.py, run_xtail_gate_sensitivity.py,
run_xtail_transfer_headroom.py, run_xtail_ceiling_signals.py). No new experiments, except that the
thresholds in fig. 3 are recomputed from cached features in seconds.

  docs/figures/xtail_method.png          fig. 1  method overview
  docs/figures/xtail_steps.png           fig. 2  accuracy over all domains after each learning step
  docs/figures/xtail_thresholds.png      fig. 3  per-domain thresholds vs. one global threshold
  docs/figures/xtail_gate_per_seed.png   fig. 4  effect of the gates, every held-out seed
  docs/figures/xtail_ceiling.png         fig. 5  take-back signals: separability and trade-off
  docs/figures/xtail_table_main.png      table 1 main results
  docs/figures/xtail_table_settings.png  table 2 task order and shot count
  docs/figures/xtail_table_tuning.png    table 3 effect of the tuning protocol

Usage:
    python scripts/plot_xtail_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from data.real.xtail import XTAIL_ROOT  # noqa: E402

RESULTS = XTAIL_ROOT / "results"
FIG = REPO_ROOT / "docs" / "figures"
SEEDS = (16, 17, 18, 19, 20)  # final run, all methods
EARLIER_SEEDS = (11, 12, 13, 14, 15)  # held-out run of the full-kernel reverse gate
REPLICATION_SEEDS = (6, 7, 8, 9, 10)  # earlier held-out run (both gates on order I only); table 3 uses these
REVERSE = "ml_rail_gate_reverse_cls10"  # the reported reverse gate (layer-10 class token)
SETTINGS = [("order_I", "Order I, 16-shot"), ("order_II", "Order II, 16-shot"), ("11_domains_5shot", "11 domains, 5-shot")]
DOMAIN_LABELS = ["Aircraft", "Caltech101", "DTD", "EuroSAT", "Flowers", "Food101", "MNIST", "Pets", "Cars", "SUN397"]
# Published X-TAIL order I, 16-shot (Transfer, Average, Last), from each paper's main table.
PUBLISHED = [("RAIL (Primal) [1]", "closed form", (62.4, 70.7, 79.1)),
             ("RAIL (Dual) [1]", "closed form", (62.4, 71.9, 82.4)),
             ("MoRAM [2]", "gradient", (63.3, 72.7, 80.9)),
             ("LADA [3]", "gradient", (None, 72.7, 83.1))]
# Best single global similarity threshold, tuned on all ten domains' validation data in the separability
# analysis (scripts/experiments/run_xtail_gatesep_explore.py); shown in fig. 3 for comparison only.
GLOBAL_THRESHOLD = 0.958
GATE_Q = 0.25
# Okabe-Ito colour-blind-safe palette
GREY, BLUE, ORANGE, RED = "#7f7f7f", "#0072B2", "#E69F00", "#D55E00"
INK = "#111111"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.7,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "figure.facecolor": "white",
})


def evaluation(seeds=SEEDS):
    """(setting, method) -> list of runs over `seeds` (float32)."""
    rows = json.load(open(RESULTS / "evaluate.json"))
    out = {}
    for r in rows:
        if r["dtype"] == "float32" and r["seed"] in seeds:
            out.setdefault((r["setting"], r["method"]), []).append(r)
    for runs in out.values():
        runs.sort(key=lambda r: r["seed"])
    return out


def panel_label(ax, s, x=-0.13):
    ax.text(x, 1.02, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")


# ------------------------------------------------------------------ fig. 1

def box(ax, x, y, w, h, title, body="", ours=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=0.8",
                                fc="#eef4fa" if ours else "white", ec=BLUE if ours else INK, lw=0.9))
    ax.text(x + w / 2, y + h - 1.8, title, ha="center", va="top", fontsize=8.5, fontweight="bold", color=INK)
    if body:
        ax.text(x + w / 2, y + h - 5.4, body, ha="center", va="top", fontsize=7.2, color=INK, linespacing=1.3)


def arrow(ax, a, b, ls="-", rad=0.0, color=INK):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=8, color=color, lw=0.8, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def fig_method():
    fig, ax = plt.subplots(figsize=(9.2, 3.9))
    ax.set_xlim(0, 114)
    ax.set_ylim(9, 65.5)
    ax.axis("off")
    box(ax, 0.5, 19, 12.5, 12, "Image", "any domain,\nno domain label")
    box(ax, 17, 17, 14, 16, "CLIP ViT-B/16", "frozen")
    box(ax, 36, 35, 25, 13, "Zero-shot head", 'cos(image, "a photo of a {}.")\nover all class names')
    box(ax, 36, 13, 25, 16, "Multi-layer features", "final embedding, layer-10 class\ntoken, patch mean, pooled\nlayers 8 and 10", ours=True)
    box(ax, 66, 13, 24, 16, "Kernel ridge adapter", "RAIL's closed-form solve\nover all learned domains")
    box(ax, 66, 34, 24, 18, "Rescue and reverse gate", "admit: top-1 class learned, or\nmass on learned $\\geq 0.1$ and\nsimilarity to some $d \\geq \\tau_d$\ntake back: class-token\nsimilarity to every $d < \\rho_d$", ours=True)
    box(ax, 66, 54.5, 46, 9.5, r"Per-domain thresholds $\tau_d$ and $\rho_d$",
        "$\\tau_d$: 25th percentile of nearest-neighbour similarities\n$\\rho_d$: 0.8th percentile, in the layer-10 class token", ours=True)
    box(ax, 95, 10, 17, 32, "Prediction", "admitted:\n$0.2\\,p_{zs} + 0.8\\,f(x)$\n\nrejected:\n$p_{zs}$\n\ntaken back:\n$p_{zs}$ over classes\nnot yet learned")
    arrow(ax, (13.4, 25), (16.6, 25))
    arrow(ax, (31.4, 28), (35.6, 41), rad=-0.15)
    arrow(ax, (31.4, 22), (35.6, 21))
    arrow(ax, (61.4, 21), (65.6, 21))
    arrow(ax, (61.4, 41.5), (65.6, 41.5))
    arrow(ax, (78, 29.4), (78, 33.6), ls="--")
    arrow(ax, (78, 54.1), (78, 52.4), ls="--")
    arrow(ax, (90.4, 43), (94.6, 36), rad=-0.15)
    arrow(ax, (90.4, 21), (94.6, 21))
    fig.savefig(FIG / "xtail_method.png")
    plt.close(fig)


# ------------------------------------------------------------------ fig. 2

def fig_steps(ev):
    series = [("Zero-shot CLIP", "zero_shot", GREY, ":"), ("RAIL (Dual), our run", "rail_dual", ORANGE, "--"),
              ("ML-RAIL", "ml_rail", BLUE, "-"), ("ML-RAIL + both gates", REVERSE, RED, "-")]
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    x = np.arange(1, 11)
    for name, key, c, ls in series:
        a = np.array([np.array(r["table"]).mean(1) for r in ev[("order_I", key)]])
        m, sd = a.mean(0), a.std(0, ddof=1)
        ax.plot(x, m, ls, color=c, lw=1.2, marker="o" if ls == "-" else None, ms=2.5, label=name)
        ax.fill_between(x, m - sd, m + sd, color=c, alpha=0.18, lw=0)
    ax.set_xticks(x, DOMAIN_LABELS, rotation=40, ha="right", rotation_mode="anchor", fontsize=7.5)
    ax.set_xlabel("Most recently learned domain (order I)")
    ax.set_ylabel("Accuracy over all 10 domains (%)")
    ax.set_xlim(0.7, 10.3)
    ax.grid(axis="y", color="#dddddd", lw=0.5)
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(FIG / "xtail_steps.png")
    plt.close(fig)


# ------------------------------------------------------------------ fig. 3

def fig_thresholds():
    import torch
    from evaluation import xtail_protocol as xp
    from evaluation.ml_rail import BLOCKS, blocks_feature, load_with_extra
    data, feat = load_with_extra("single"), blocks_feature(BLOCKS)
    sims = []
    for t, n in enumerate(data):
        X = torch.as_tensor(feat(data[n], "train", xp.fewshot(data[n], SEEDS[0], 16, t)), device=xp.DEV)
        K = torch.exp(-0.2 * torch.cdist(X, X).pow(2))
        K[K >= 1 - 1e-6] = -1  # self and exact duplicates
        sims.append(K.max(1).values.cpu().numpy())
    fig, ax = plt.subplots(figsize=(7.0, 2.3))
    ax.boxplot(sims, whis=(5, 95), showfliers=False, widths=0.5, patch_artist=True,
               boxprops={"facecolor": "#f2f2f2", "edgecolor": INK, "lw": 0.7}, medianprops={"color": INK, "lw": 0.7},
               whiskerprops={"color": INK, "lw": 0.7}, capprops={"color": INK, "lw": 0.7})
    ax.scatter(range(1, 11), [np.quantile(s, GATE_Q) for s in sims], marker="D", s=14, color=BLUE, zorder=3,
               label=r"rescue threshold $\tau_d$ (25th percentile)")
    ax.axhline(GLOBAL_THRESHOLD, color=RED, ls="--", lw=0.9, label="best single global threshold (tuned on all domains)")
    ax.set_xticks(range(1, 11), DOMAIN_LABELS)
    ax.set_ylabel("Nearest-neighbour similarity")
    ax.grid(axis="y", color="#dddddd", lw=0.5)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.savefig(FIG / "xtail_thresholds.png")
    plt.close(fig)


# ------------------------------------------------------------------ fig. 4

def fig_per_seed(ev):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
    variants = (("rescue gate", "ml_rail_gate", ORANGE, -0.17), ("both gates", REVERSE, BLUE, 0.17))
    for ax, metric, label in ((axes[0], "average", r"$\Delta$ Average vs. ML-RAIL (points)"),
                              (axes[1], "transfer", r"$\Delta$ Transfer vs. ML-RAIL (points)")):
        for i, (b, _) in enumerate(SETTINGS):
            for vname, key, color, off in variants:
                d = [g[metric] - m[metric] for g, m in zip(ev[(b, key)], ev[(b, "ml_rail")])]
                ax.scatter(np.full(len(d), i + off) + np.linspace(-0.07, 0.07, len(d)), d, s=10, color=color, zorder=3,
                           label=vname if i == 0 else None)
                ax.plot([i + off - 0.12, i + off + 0.12], [np.mean(d)] * 2, color=INK, lw=1.0, zorder=4)
        ax.axhline(0, color=INK, lw=0.6)
        if metric == "transfer":
            ax.axhline(-0.2, color=GREY, lw=0.7, ls="--")
            ax.text(-0.45, -0.215, "rescue gate's tolerance for the mean", fontsize=6.5, color=GREY, va="top", ha="left")
        ax.set_xticks(range(3), [lab for _, lab in SETTINGS])
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylabel(label)
        ax.grid(axis="y", color="#dddddd", lw=0.5)
    axes[0].set_ylim(0, 1.5)
    axes[1].set_ylim(-0.45, 0.8)
    axes[0].legend(frameon=False, loc="upper left")
    panel_label(axes[0], "(a)", -0.16)
    panel_label(axes[1], "(b)", -0.16)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(FIG / "xtail_gate_per_seed.png")
    plt.close(fig)


# ------------------------------------------------------------------ fig. 5

SIGNAL_LABELS = {"nn1": "kernel, nearest", "top5": "kernel, top-5 mean", "cls": "kernel, top class",
                 "blk_global": "final embedding", "blk_cls10": "layer-10 class token", "blk_patch_mean": "patch mean",
                 "blk_gem8": "pooled layer 8", "blk_gem10": "pooled layer 10", "maha": "Mahalanobis"}


def fig_ceiling(ceil):
    sig = ceil["order_I"]["signals"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    order = sorted(sig, key=lambda k: sig[k]["auroc"])
    ax.barh(range(len(order)), [sig[k]["auroc"] for k in order], height=0.6,
            color=[RED if k == "blk_cls10" else BLUE if k == "nn1" else "#c9c9c9" for k in order])
    ax.set_yticks(range(len(order)), [SIGNAL_LABELS[k] for k in order], fontsize=7.5)
    ax.set_xlim(0.75, 1.0)
    ax.set_xlabel("AUROC, unseen vs. learned domain")
    ax.grid(axis="x", color="#dddddd", lw=0.5)
    ax = axes[1]
    for k, c, lab in (("nn1", BLUE, "kernel, nearest (reverse gate before)"), ("blk_cls10", RED, "layer-10 class token")):
        curve = np.array(sig[k]["dev_curve"])
        ax.plot(curve[:, 0], curve[:, 1], "-", color=c, lw=1.1, label=lab)
        chosen = sig[k]["val_balanced"]["dev"]
        ax.scatter([chosen[0]], [chosen[1]], color=c, s=22, zorder=4, edgecolor=INK, lw=0.5)
    o = ceil["order_I"]["oracle"]
    ax.scatter([o[0]], [o[1]], marker="*", s=70, color=INK, zorder=4, label="oracle")
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_xlabel(r"$\Delta$ Transfer (points)")
    ax.set_ylabel(r"$\Delta$ Average (points)")
    ax.set_xlim(0, 2.05)
    ax.set_ylim(-0.6, 0.75)
    ax.grid(color="#dddddd", lw=0.5)
    ax.legend(frameon=False, loc="upper left", fontsize=7)
    panel_label(axes[0], "(a)", -0.5)
    panel_label(axes[1], "(b)", -0.16)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(FIG / "xtail_ceiling.png")
    plt.close(fig)


# ------------------------------------------------------------------ tables

def cells(runs, with_std=("average", "last")):
    """(mean, std or None) for Transfer, Average, Last. Transfer is identical across seeds for every method
    that keeps RAIL's gate, because unseen domains are classified by zero-shot CLIP."""
    out = []
    for m in ("transfer", "average", "last"):
        v = np.array([r[m] for r in runs])
        sd = v.std(ddof=1)
        out.append((v.mean(), sd if m in with_std and sd >= 0.005 else None))
    return out


def delta(runs, base):
    """Mean gain in Average and Last over the same-harness Dual-RAIL, as text."""
    da = np.mean([r["average"] for r in runs]) - np.mean([r["average"] for r in base])
    dl = np.mean([r["last"] for r in runs]) - np.mean([r["last"] for r in base])
    return f"{da:+.2f} / {dl:+.2f}".replace("-", "−")


def render_table(path, header, rows, col_w, decimals):
    """Booktabs-style table. rows: dicts with 'cells' (list of str or (mean, std)), optional 'shade',
    'rule_before' and 'group' (rows sharing a group are ranked together). Within a group, the best value
    of each numeric column is bold and the second best underlined."""
    rank = {}
    for c in range(len(header)):
        groups = {}
        for i, r in enumerate(rows):
            v = r["cells"][c]
            if isinstance(v, tuple):
                groups.setdefault(r.get("group", 0), []).append((round(v[0], decimals[i][c]), i))
        for g in groups.values():
            vals = sorted({v for v, _ in g}, reverse=True)
            for v, i in g:
                rank[(i, c)] = vals.index(v) if len(vals) > 1 else None
    row_h, head_h, pad = 0.27, 0.30, 0.06
    W, H = sum(col_w), head_h + row_h * len(rows) + 2 * pad
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    x0 = np.cumsum([0] + col_w)
    top = H - pad
    text_col = [all(not isinstance(r["cells"][c], tuple) for r in rows) and not header[c].startswith("Δ")
                for c in range(len(header))]
    for c, h in enumerate(header):
        ax.text(x0[c] + (0.08 if text_col[c] else col_w[c] / 2), top - head_h / 2, h,
                ha="left" if text_col[c] else "center", va="center", fontsize=9.5, fontweight="bold")
    ax.plot([0, W], [top, top], color=INK, lw=1.1)
    ax.plot([0, W], [top - head_h, top - head_h], color=INK, lw=0.6)
    for i, r in enumerate(rows):
        yt = top - head_h - i * row_h
        yc = yt - row_h / 2
        if r.get("shade"):
            ax.add_patch(plt.Rectangle((0, yt - row_h), W, row_h, color="#e8f1fa", lw=0))
        if r.get("rule_before"):
            ax.plot([0, W], [yt, yt], color=INK, lw=0.4)
        for c, v in enumerate(r["cells"]):
            xc = x0[c] + col_w[c] / 2
            if not isinstance(v, tuple):
                ax.text(x0[c] + 0.08 if text_col[c] else xc, yc, v, ha="left" if text_col[c] else "center", va="center",
                        fontsize=9, fontweight="bold" if (r.get("shade") and c == 0) else "normal",
                        color=INK if v != "–" else GREY)
                continue
            mean, std = v
            k = rank.get((i, c))
            t = ax.text(xc + 0.08, yc, f"{mean:.{decimals[i][c]}f}", ha="right", va="center", fontsize=9.5,
                        fontweight="bold" if k == 0 else "normal")
            if k == 1:
                fig.canvas.draw()
                bb = t.get_window_extent().transformed(ax.transData.inverted())
                ax.plot([bb.x0, bb.x1], [bb.y0 + 0.015, bb.y0 + 0.015], color=INK, lw=0.6)
            if std is not None:
                ax.text(xc + 0.1, yc - 0.01, f"±{std:.2f}", ha="left", va="center", fontsize=7, color=GREY)
    yb = top - head_h - len(rows) * row_h
    ax.plot([0, W], [yb, yb], color=INK, lw=1.1)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


OURS = (("ML-RAIL (ours)", "ml_rail"), ("+ rescue gate (ours)", "ml_rail_gate"),
        ("+ rescue and reverse gates (ours)", REVERSE))


def tables(ev, ev_rep, tuning):
    header = ["Method", "Training", "Transfer", "Average", "Last", "Δ Avg / Last"]
    widths = [2.75, 1.0, 0.85, 1.05, 1.05, 1.25]
    rows, dec = [], []
    for name, training, (t, a, last) in PUBLISHED:
        rows.append({"cells": [name, training, (t, None) if t is not None else "–", (a, None), (last, None), ""], "group": 0})
        dec.append([0, 0, 1, 1, 1, 0])
    base = ev[("order_I", "rail_dual")]
    for j, (name, key) in enumerate((("Zero-shot CLIP", "zero_shot"), ("RAIL (Dual), our run", "rail_dual")) + OURS):
        runs = ev[("order_I", key)]
        c = cells(runs) if key != "zero_shot" else [(m, None) for m, _ in cells(runs, ())]
        rows.append({"cells": [name, "–" if key == "zero_shot" else "closed form"] + c
                     + ["" if key in ("zero_shot", "rail_dual") else delta(runs, base)],
                     "group": 1, "rule_before": j == 0, "shade": key == REVERSE})
        dec.append([0, 0, 2, 2, 2, 0])
    render_table(FIG / "xtail_table_main.png", header, rows, widths, dec)

    header = ["Setting", "Method", "Transfer", "Average", "Last", "Δ Avg / Last"]
    rows, dec = [], []
    for g, (b, label) in enumerate(SETTINGS[1:]):
        base = ev[(b, "rail_dual")]
        for j, (name, key) in enumerate((("Zero-shot CLIP", "zero_shot"), ("RAIL (Dual), our run", "rail_dual")) + OURS):
            runs = ev[(b, key)]
            c = cells(runs) if key != "zero_shot" else [(m, None) for m, _ in cells(runs, ())]
            rows.append({"cells": [label if j == 0 else "", name] + c
                         + ["" if key in ("zero_shot", "rail_dual") else delta(runs, base)],
                         "group": g, "shade": key == REVERSE, "rule_before": g > 0 and j == 0})
            dec.append([0, 0, 2, 2, 2, 0])
    render_table(FIG / "xtail_table_settings.png", header, rows, [1.45, 2.6, 0.85, 1.05, 1.05, 1.25], dec)

    header = ["Hyperparameters chosen on", "Features, γ, λ", "Transfer", "Average", "Last", "Δ Avg / Last"]
    base = ev_rep[("order_I", "rail_dual")]  # the first-domain runs were made on seeds 6-10
    sel = tuning["selected"]
    tuned = lambda m: [r for r in tuning["test"] if r["method"] == m]  # noqa: E731
    spec = [
        ("RAIL's published setting", "final embedding, 0.1, 10⁻³", ev_rep[("order_I", "rail_dual")], False),
        ("all domains' validation", "five blocks, 0.2, 10⁻³", ev_rep[("order_I", "ml_rail")], False),
        ("  + rescue gate", "", ev_rep[("order_I", "ml_rail_gate")], False),
        ("  + rescue and reverse gates", "", ev_rep.get(("order_I", REVERSE), []), True),
        ("first domain's validation", f"final embedding, {sel['gamma']}, {sel['lam']:g}", tuned("ml_rail_first_domain"), False),
        ("  + rescue gate", "", tuned("ml_rail_first_domain_gate"), False),
        ("  + rescue and reverse gates*", "", tuned("ml_rail_first_domain_gate_reverse"), False),
        ("first domain, five blocks fixed", "five blocks, 0.8, 0.1", tuned("ml_rail_blocks_first_domain_params"), False),
        ("  + rescue gate", "", tuned("ml_rail_blocks_first_domain_params_gate"), False),
        ("  + rescue and reverse gates", "", tuned("ml_rail_blocks_first_domain_params_gate_reverse_cls10"), False),
    ]
    rows, dec = [], []
    for j, (name, desc, runs, shade) in enumerate(spec):
        if not runs:
            continue
        rows.append({"cells": [name, desc] + cells(runs, ("average", "last", "transfer")) + ["" if j == 0 else delta(runs, base)],
                     "shade": shade, "rule_before": name.startswith(("first", "all"))})
        dec.append([0, 0, 2, 2, 2, 0])
    render_table(FIG / "xtail_table_tuning.png", header, rows, [2.6, 1.95, 1.1, 1.05, 1.05, 1.25], dec)


def export_results(ev, ev_earlier, ev_rep, tuning, sens, headroom, ceil):
    def runs(rows):
        return [{"seed": r["seed"], **{k: round(r[k], 3) for k in ("transfer", "average", "last")},
                 "table": np.round(np.array(r["table"]), 2).tolist()} for r in rows]
    out = {
        "description": "X-TAIL results (CLIP ViT-B/16, RAIL's single prompt). 'settings': the final run on held-out "
                       "seeds 16-20, where ml_rail_gate_reverse_cls10 is the reported method and ml_rail_gate_reverse "
                       "the earlier full-kernel reverse gate; 'run_seeds_11_15': the held-out run of the full-kernel "
                       "reverse gate; 'replication_seeds_6_10': the first held-out run (no reverse gate). table[t][d] is "
                       "the accuracy (%) on domain d after learning domains 0..t; Transfer/Average/Last as in RAIL's "
                       "code. The tuning protocol and gate sensitivity checks use seeds 6-10; the Transfer headroom "
                       "and take-back signal analyses use development seeds 1-5 and validation.",
        "published_order_I_16shot": {name: dict(zip(("transfer", "average", "last"), v)) for name, _, v in PUBLISHED},
        "settings": {label: {m: runs(ev[(b, m)]) for m in ("zero_shot", "rail_dual", "ml_rail", "ml_rail_gate",
                                                           "ml_rail_gate_reverse", REVERSE)}
                     for b, label in SETTINGS},
        "run_seeds_11_15": {label: {m: runs(ev_earlier[(b, m)]) for m in ("zero_shot", "rail_dual", "ml_rail", "ml_rail_gate",
                                                                          "ml_rail_gate_reverse")}
                            for b, label in SETTINGS},
        "replication_seeds_6_10": {label: {m: runs(ev_rep[(b, m)]) for m in ("zero_shot", "rail_dual", "ml_rail", "ml_rail_gate")}
                                   for b, label in SETTINGS},
        "tuning_protocol_order_I": {"first_domain_selection": tuning["selected"],
                                    "runs": {m: runs([r for r in tuning["test"] if r["method"] == m])
                                             for m in sorted({r["method"] for r in tuning["test"]})}},
        "gate_sensitivity_order_I": sens,
        "transfer_headroom_development": {k: {**{v: {m: round(x[m], 3) for m in ("transfer", "average", "last")}
                                                 for v, x in run.items() if v != "diagnostics"},
                                              "diagnostics": run["diagnostics"]} for k, run in headroom.items()},
        "take_back_signals_development": ceil,
    }
    path = REPO_ROOT / "results" / "xtail_results.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    ev, ev_earlier, ev_rep = evaluation(SEEDS), evaluation(EARLIER_SEEDS), evaluation(REPLICATION_SEEDS)
    tuning = json.load(open(RESULTS / "first_domain_tuning.json"))
    sens = json.load(open(RESULTS / "gate_sensitivity.json"))
    headroom = json.load(open(RESULTS / "transfer_headroom.json"))
    ceil = json.load(open(RESULTS / "ceiling" / "analysis.json"))
    export_results(ev, ev_earlier, ev_rep, tuning, sens, headroom, ceil)
    tables(ev, ev_rep, tuning)
    fig_method()
    fig_steps(ev)
    fig_thresholds()
    fig_per_seed(ev)
    fig_ceiling(ceil)
    print("wrote", *sorted(p.name for p in FIG.glob("xtail_*.png")))
