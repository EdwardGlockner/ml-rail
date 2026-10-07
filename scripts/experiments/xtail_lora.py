"""Per-domain LoRA experts behind the cascade.

Each domain gets a rank-2 LoRA on the q/k/v projections of CLIP ViT-B/16's image encoder, trained only on
its own 16 shots (cross-entropy against the frozen single-prompt text embeddings of its classes). At
test, ML-RAIL routes each admitted image to the domain of its top learned class; that domain's expert
re-scores it within the domain (variant E: expert probabilities; E+A: mean with ML-RAIL's scores).

Stages (all resumable -- every output is saved per seed/domain under $XTAIL_ROOT/lora, and training
checkpoints every epoch and when paused):
    select     epochs E in {5, 10, 20} on Aircraft's validation split, seed 100 (first domain only)
    train      one expert per domain for a seed
    routes     which (image, expert) pairs the cascade will ask for
    encode     expert probabilities for exactly those pairs
    evaluate   ML-RAIL vs. cascade-with-experts, RAIL's gate (the registered comparison) and the rescue gate
    all        select -> seed 100 validation -> registered rule -> test seeds 1-5

Pause:  scripts/experiments/xtail_lora_pause.ps1   (creates $XTAIL_ROOT/lora/PAUSE; the job saves and exits)
Resume: scripts/experiments/xtail_lora_resume.ps1  (removes PAUSE and relaunches with low-load settings; continues where it stopped)

Usage:
    python scripts/experiments/xtail_lora.py --stage all
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torchvision.transforms as T  # noqa: E402
import torchvision.transforms.functional as TF  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from data.real.xtail import clip_preprocess, load_domain  # noqa: E402
from data.real.xtail import XTAIL_ROOT  # noqa: E402
from evaluation import xtail_protocol as xp  # noqa: E402
from evaluation.ml_rail import blocks_feature, load_with_extra  # noqa: E402
from evaluation.ml_rail import BLOCKS, make_ml_rail  # noqa: E402
from scripts.xtail_encode import MODEL, WORKERS  # noqa: E402

ROOT = (XTAIL_ROOT / "lora")
PAUSE = ROOT / "PAUSE"
OUT = REPO_ROOT / "data" / "generated" / "xtail"
DEV = xp.DEV
MEAN = torch.tensor((0.48145466, 0.4578275, 0.40821073), device=DEV).view(1, 3, 1, 1)
STD = torch.tensor((0.26862954, 0.26130258, 0.27577711), device=DEV).view(1, 3, 1, 1)
EPOCH_GRID, BATCH, LR, WD, WARMUP = (5, 10, 20), 32, 2e-4, 0.01, 50
VAL_SEED, TEST_SEEDS = 100, (1, 2, 3, 4, 5)


class Paused(Exception):
    pass


def check_pause():
    if PAUSE.exists():
        raise Paused


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- model

class LoRALinear(nn.Module):
    def __init__(self, base, r=2, alpha=1.0, dropout=0.25):
        super().__init__()
        self.base, self.scale = base, alpha / r
        self.A = nn.Parameter(torch.zeros(r, base.in_features, device=base.weight.device))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + self.scale * (self.drop(x) @ self.A.t().to(x.dtype)) @ self.B.t().to(x.dtype)


_MODEL = None


def clip_model():
    global _MODEL
    if _MODEL is None:
        from transformers import CLIPModel
        m = CLIPModel.from_pretrained(MODEL).to(DEV)
        m.requires_grad_(False)
        for layer in m.vision_model.encoder.layers:
            for name in ("q_proj", "k_proj", "v_proj"):
                setattr(layer.self_attn, name, LoRALinear(getattr(layer.self_attn, name)))
        m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        _MODEL = m
    return _MODEL


def loras(m):
    return [mod for mod in m.modules() if isinstance(mod, LoRALinear)]


def lora_state(m):
    return [(l.A.detach().cpu().clone(), l.B.detach().cpu().clone()) for l in loras(m)]


def set_lora(m, state):
    with torch.no_grad():
        for l, (A, B) in zip(loras(m), state):
            l.A.copy_(A)
            l.B.copy_(B)


def init_lora(m, seed):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for l in loras(m):
            bound = 1.0 / math.sqrt(l.A.shape[1])  # kaiming-uniform(a = sqrt 5), as nn.Linear / LoRA
            l.A.copy_((torch.rand(l.A.shape, generator=g) * 2 - 1) * bound)
            l.B.zero_()


def embed(m, x):
    """x: float (B, 3, 224, 224) in [0, 1] -> L2-normalized projected image embedding."""
    o = m.vision_model(pixel_values=(x - MEAN) / STD)
    f = m.visual_projection(o.pooler_output).float()
    return f / f.norm(dim=-1, keepdim=True)


# ---------------------------------------------------------------- images

class Sized(Dataset):
    def __init__(self, dom, split, img_idx, size):
        self.dom, self.split, self.idx, self.size = dom, split, img_idx, size

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        return torch.from_numpy(np.array(clip_preprocess(self.dom.load(self.split, int(self.idx[i])), self.size)))


def shot_images(name, d, dom, seed, t):
    """uint8 (N, 3, 256, 256) of the domain's 16-shot training images (the same draw as ML-RAIL), cached."""
    path = ROOT / "cache" / f"{name}_seed{seed}_train256.npy"
    if path.exists():
        return torch.from_numpy(np.load(path))
    pos = xp.fewshot(d, seed, 16, t)
    loader = DataLoader(Sized(dom, "train", d["train_index"][pos], 256), batch_size=64, num_workers=WORKERS)
    imgs = torch.cat([b for b in loader]).permute(0, 3, 1, 2).contiguous()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, imgs.numpy())
    return imgs


def augment(batch):
    out = []
    for img in batch:
        i, j, h, w = T.RandomResizedCrop.get_params(img, (0.5, 1.0), (3 / 4, 4 / 3))
        c = TF.resized_crop(img, i, j, h, w, [224, 224], antialias=True)
        out.append(TF.hflip(c) if torch.rand(1).item() < 0.5 else c)
    return torch.stack(out).to(DEV).float() / 255


# ---------------------------------------------------------------- training

def train_expert(name, d, dom, seed, t, epochs, out_path):
    """Trains (or resumes) one domain's expert; saves the final LoRA state to out_path."""
    if out_path.exists():
        return
    m = clip_model()
    imgs = shot_images(name, d, dom, seed, t)
    y = torch.as_tensor(d["train_labels"][xp.fewshot(d, seed, 16, t)], device=DEV)
    text = torch.as_tensor(d["text"], device=DEV)
    params = [p for l in loras(m) for p in (l.A, l.B)]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
    n, spe = len(imgs), math.ceil(len(imgs) / BATCH)
    total = epochs * spe
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / WARMUP) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    scaler = torch.cuda.amp.GradScaler()
    part = out_path.with_suffix(".partial.pt")
    if part.exists():
        ck = torch.load(part)
        set_lora(m, ck["lora"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start = ck["step"]
        log(f"  {name}: resuming at step {start}/{total}")
    else:
        init_lora(m, seed * 100 + t)
        start = 0
    m.train()
    t0 = time.time()

    def save(step):
        torch.save({"lora": lora_state(m), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "step": step}, part)

    for step in range(start, total):
        epoch, k = divmod(step, spe)
        perm = np.random.default_rng([seed, t, epoch]).permutation(n)
        b = perm[k * BATCH : (k + 1) * BATCH]
        torch.manual_seed(seed * 1_000_003 + t * 10_007 + step)
        x = augment(imgs[b])
        with torch.autocast("cuda", dtype=torch.float16):
            f = embed(m, x)
        loss = nn.functional.cross_entropy(100.0 * f @ text.T, y[b])
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        if (step + 1) % spe == 0 or PAUSE.exists():
            save(step + 1)
            check_pause()
    m.eval()
    torch.save(lora_state(m), out_path)
    part.unlink(missing_ok=True)
    log(f"  {name}: expert trained ({epochs} epochs, {n} images) [{time.time() - t0:.0f}s]")


@torch.no_grad()
def expert_probs(name, d, dom, split, positions, lora):
    """Expert probabilities over the domain's classes for the given positions of a cached split."""
    m = clip_model()
    m.eval()
    set_lora(m, lora)
    text = torch.as_tensor(d["text"], device=DEV)
    loader = DataLoader(Sized(dom, split, d[f"{split}_index"][positions], 224), batch_size=128, num_workers=WORKERS)
    out = []
    for batch in loader:
        check_pause()
        with torch.autocast("cuda", dtype=torch.float16):
            f = embed(m, batch.permute(0, 3, 1, 2).to(DEV).float() / 255)
        out.append(torch.softmax(100.0 * f @ text.T, 1).cpu())
    return torch.cat(out).numpy()


# ---------------------------------------------------------------- stages

def select_epochs(data):
    path = OUT / "lora_select.json"
    res = json.load(open(path)) if path.exists() else {}
    d, dom = data["aircraft"], load_domain("aircraft")
    for E in EPOCH_GRID:
        if str(E) in res:
            continue
        ck = ROOT / "select" / f"aircraft_seed{VAL_SEED}_E{E}.pt"
        ck.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        train_expert("aircraft", d, dom, VAL_SEED, 0, E, ck)
        secs = time.time() - t0
        p = expert_probs("aircraft", d, dom, "val", np.arange(len(d["val_labels"])), torch.load(ck))
        acc = float((p.argmax(1) == d["val_labels"]).mean() * 100)
        res[str(E)] = {"val_acc": acc, "seconds": secs}
        json.dump(res, open(path, "w"), indent=1)
        log(f"select: E = {E}: Aircraft validation accuracy {acc:.2f} (expert alone) [{secs:.0f}s]")
    allowed = [E for E in EPOCH_GRID if not (E == 20 and res["20"]["seconds"] > 3600)]
    E = max(allowed, key=lambda E: (res[str(E)]["val_acc"], -E))
    log(f"select: E = {E} (zero-shot Aircraft validation for reference: "
        f"{float((np.asarray(d['val_global']) @ d['text'].T).argmax(1).__eq__(d['val_labels']).mean() * 100):.2f})")
    return E


def train_all(data, seed, E):
    for t, (name, d) in enumerate(data.items()):
        ck = ROOT / f"seed{seed}" / f"{name}.pt"
        ck.parent.mkdir(parents=True, exist_ok=True)
        sel = ROOT / "select" / f"aircraft_seed{VAL_SEED}_E{E}.pt"
        if not ck.exists() and t == 0 and seed == VAL_SEED and sel.exists():  # identical run: reuse
            torch.save(torch.load(sel), ck)
        train_expert(name, d, load_domain(name), seed, t, E, ck)


def gates():
    return {"rail": lambda strip: "rail", "conformal": lambda strip: xp.DomainThresholdGate(0.25, 0.05, strip=strip)}


def routes(data, seed, split, zs, feat):
    """For each gate, the (domain j, position, expert d) pairs the cascade asks for at some step."""
    path = ROOT / f"seed{seed}" / f"routes_{split}.npz"
    if path.exists():
        z = np.load(path)
        return {k: z[k] for k in z.files}
    names = list(data)
    cls_dom = np.concatenate([np.full(data[n]["n_classes"], k) for k, n in enumerate(names)])
    pairs = set()
    for gname, make_gate in gates().items():
        router, gate, learned = make_ml_rail(), make_gate(0), 0
        for t, n in enumerate(names):
            d = data[n]
            idx = xp.fewshot(d, seed, 16, t)
            router.add(feat(d, "train", idx), d["train_labels"][idx] + d["offset"])
            learned += d["n_classes"]
            for j, m in enumerate(names):
                e = data[m]
                X = feat(e, split)
                mask = gate(router, X, zs[m], learned, [(data[k]["offset"], data[k]["n_classes"]) for k in names[: t + 1]]) \
                    if callable(gate) else zs[m].argmax(1) < learned
                if not mask.any():
                    continue
                pos = np.where(mask.numpy())[0]
                r = cls_dom[torch.as_tensor(router.scores(X[pos])).argmax(1).numpy()]
                pairs.update(zip([j] * len(pos), pos.tolist(), r.tolist()))
        log(f"routes seed {seed} {split}: after gate {gname}, {len(pairs)} (image, expert) pairs")
    arr = np.array(sorted(pairs), dtype=np.int64)
    np.savez(path, j=arr[:, 0], pos=arr[:, 1], expert=arr[:, 2])
    return {"j": arr[:, 0], "pos": arr[:, 1], "expert": arr[:, 2]}


def encode_all(data, seed, split, R):
    names = list(data)
    for di, dn in enumerate(names):
        lora = None
        for j, jn in enumerate(names):
            sel = (R["expert"] == di) & (R["j"] == j)
            if not sel.any():
                continue
            path = ROOT / f"seed{seed}" / "probs" / f"{split}_{dn}_on_{jn}.npz"
            if path.exists():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            lora = lora if lora is not None else torch.load(ROOT / f"seed{seed}" / f"{dn}.pt")
            pos = np.sort(R["pos"][sel])
            t0 = time.time()
            p = expert_probs(dn, data[dn], load_domain(jn), split, pos, lora) if dn == jn else \
                _probs_other(data[dn], data[jn], load_domain(jn), split, pos, lora)
            np.savez(path, pos=pos, probs=p.astype(np.float16))
            log(f"  encode seed {seed} {split}: expert {dn} on {jn}: {len(pos)} images [{time.time() - t0:.0f}s]")


@torch.no_grad()
def _probs_other(d_expert, d_images, dom_images, split, positions, lora):
    """Expert of one domain applied to images of another (misrouted images)."""
    m = clip_model()
    m.eval()
    set_lora(m, lora)
    text = torch.as_tensor(d_expert["text"], device=DEV)
    loader = DataLoader(Sized(dom_images, split, d_images[f"{split}_index"][positions], 224), batch_size=128, num_workers=WORKERS)
    out = []
    for batch in loader:
        check_pause()
        with torch.autocast("cuda", dtype=torch.float16):
            f = embed(m, batch.permute(0, 3, 1, 2).to(DEV).float() / 255)
        out.append(torch.softmax(100.0 * f @ text.T, 1).cpu())
    return torch.cat(out).numpy()


class Strip:
    """Hides the two trailing id columns (domain index, position) from the wrapped method."""

    def __init__(self, m, k=2):
        self.m, self.k = m, k

    def add(self, X, y):
        self.m.add(X[:, : -self.k], y)

    def scores(self, X):
        return self.m.scores(X[:, : -self.k])

    def __getattr__(self, a):
        return getattr(self.m, a)


class ExpertCascade:
    """Route by ML-RAIL's top learned class; within the routed domain, the expert's probabilities (mix "E")
    or their mean with ML-RAIL's scores (mix "E+A"); other learned domains suppressed."""

    def __init__(self, router, probs, mix, suppress=-10.0):
        self.router, self.probs, self.mix, self.suppress, self.bounds, self.misses = router, probs, mix, suppress, [], 0

    def __getattr__(self, a):  # the rescue gate reads the router's stored features and kernel
        return getattr(self.router, a)

    def add(self, X, y):
        self.bounds.append((int(y.min()), int(y.max()) + 1))
        self.router.add(X, y)

    def scores(self, X):
        r = torch.as_tensor(self.router.scores(X), dtype=torch.float32)
        dom = torch.zeros(r.shape[1], dtype=torch.long)
        for k, (a, b) in enumerate(self.bounds):
            dom[a:b] = k
        routed = dom[r.argmax(1)].numpy()
        out = torch.full_like(r, self.suppress)
        jcol, pcol = X[:, -2].astype(np.int64), X[:, -1].astype(np.int64)
        for k in np.unique(routed):
            a, b = self.bounds[k]
            for j in np.unique(jcol[routed == k]):
                rows = np.where((routed == k) & (jcol == j))[0]
                pos, P = self.probs.get((k, j), (np.zeros(0, np.int64), None))
                at = np.searchsorted(pos, pcol[rows])
                found = (at < len(pos)) & (pos[np.minimum(at, len(pos) - 1)] == pcol[rows]) if len(pos) else np.zeros(len(rows), bool)
                self.misses += int((~found).sum())
                e = r[rows, a:b].clone()  # fallback (should not happen): ML-RAIL's own scores
                if found.any():
                    e[found] = torch.as_tensor(P[at[found]].astype(np.float32))
                out[rows, a:b] = e if self.mix == "E" else 0.5 * e + 0.5 * r[rows, a:b]
        return out


def evaluate(data, seed, split, zs, feat):
    path = OUT / "lora_experts.json"
    results = json.load(open(path)) if path.exists() else []
    names = list(data)
    probs = {}
    for k, dn in enumerate(names):
        for j, jn in enumerate(names):
            f = ROOT / f"seed{seed}" / "probs" / f"{split}_{dn}_on_{jn}.npz"
            if f.exists():
                z = np.load(f)
                probs[(k, j)] = (z["pos"], z["probs"])
    jidx = {n: j for j, n in enumerate(names)}

    def feat_id(d, s, idx=None):
        x = feat(d, s, idx)
        pos = np.arange(len(x)) if idx is None else np.asarray(idx)
        return np.concatenate([x, np.full((len(x), 1), jidx[d["_name"]], np.float32), pos[:, None].astype(np.float32)], 1)

    out = {}
    for gname, make_gate in gates().items():
        for variant in ("Aprime", "E", "E+A"):
            key = f"{gname}/{variant}"
            prev = [r for r in results if r["seed"] == seed and r["split"] == split and r["name"] == key]
            if prev:
                out[key] = prev[0]
                continue
            method = Strip(make_ml_rail()) if variant == "Aprime" else ExpertCascade(Strip(make_ml_rail()), probs, variant)
            table = xp.run_stream(data, method, seed, feat=feat_id, fusion_weight=0.8, zs=zs, split=split, gate=make_gate(2))
            r = {"seed": seed, "split": split, "name": key, **xp.metrics(table), "table": table.tolist(),
                 "misses": getattr(method, "misses", 0)}
            results.append(r)
            json.dump(results, open(path, "w"), indent=1)
            out[key] = r
            log(f"evaluate seed {seed} {split} {key}: {r['transfer']:.2f} / {r['average']:.2f} / {r['last']:.2f} (lookup misses {r['misses']})")
    return out


def pipeline(data, seed, split, E, zs, feat):
    train_all(data, seed, E)
    R = routes(data, seed, split, zs, feat)
    encode_all(data, seed, split, R)
    return evaluate(data, seed, split, zs, feat)


def run_all():
    data = load_with_extra("single")
    for n, d in data.items():
        d["_name"] = n
    feat = blocks_feature(BLOCKS)
    zs = {s: xp.zero_shot_probs(data, split=s) for s in ("val", "test")}
    E = select_epochs(data)
    v = pipeline(data, VAL_SEED, "val", E, zs["val"], feat)
    base = v["rail/Aprime"]
    best = max(("E", "E+A"), key=lambda k: v[f"rail/{k}"]["average"] + v[f"rail/{k}"]["last"])
    gain = {m: v[f"rail/{best}"][m] - base[m] for m in ("transfer", "average", "last")}
    log(f"validation: chosen variant {best}: " + " / ".join(f"{k} {g:+.2f}" for k, g in gain.items()))
    if max(gain["average"], gain["last"]) < 0.5:
        log("validation bar (+0.5 Average or Last) not met: no test run")
        return
    metric = "average" if gain["average"] >= gain["last"] else "last"
    diffs = []
    for seed in TEST_SEEDS:
        r = pipeline(data, seed, "test", E, zs["test"], feat)
        diffs.append({m: r[f"rail/{best}"][m] - r["rail/Aprime"][m] for m in ("transfer", "average", "last")})
        log(f"test seed {seed}: expert cascade - ML-RAIL: " + " / ".join(f"{diffs[-1][m]:+.2f}" for m in ("transfer", "average", "last")))
    mean = {m: float(np.mean([d[m] for d in diffs])) for m in ("transfer", "average", "last")}
    pos = sum(d[metric] > 0 for d in diffs)
    others_ok = all(mean[m] >= -0.2 for m in ("transfer", "average", "last") if m != metric)
    ok = mean[metric] >= 0.5 and pos >= 4 and others_ok
    log(f"RULE ({metric}): mean " + " / ".join(f"{mean[m]:+.3f}" for m in ("transfer", "average", "last"))
        + f", {pos}/5 positive -> {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("all",), default="all")
    p.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    try:
        run_all()
        log("DONE")
    except Paused:
        log("PAUSED (state saved; run scripts/experiments/xtail_lora_resume.ps1 to continue)")
