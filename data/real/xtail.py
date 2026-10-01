"""X-TAIL benchmark data (Xu et al., NeurIPS 2024 -- RAIL), read exactly as RAIL's loaders read it.

Domains in X-TAIL order I (alphabetical), CoOp train/val/test splits (`split_zhou_*.json`), class names
as RAIL builds them (label order from the split files; FGVC-Aircraft from variants.txt; MNIST as
'number: "0 - zero"' etc.). Data built by `scripts/setup_xtail_data.py` under D:/xtail_data/datasets.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("D:/xtail_data/datasets")
DOMAINS = ("aircraft", "caltech101", "dtd", "eurosat", "oxford_flowers", "food101", "mnist",
           "oxford_pets", "stanford_cars", "sun397")
_COOP = {  # domain -> (folder, split file, image root)
    "caltech101": ("caltech101", "split_zhou_Caltech101.json", "101_ObjectCategories"),
    "dtd": ("dtd", "split_zhou_DescribableTextures.json", "images"),
    "eurosat": ("eurosat", "split_zhou_EuroSAT.json", "2750"),
    "oxford_flowers": ("oxford_flowers", "split_zhou_OxfordFlowers.json", "jpg"),
    "food101": ("food-101", "split_zhou_Food101.json", "images"),
    "oxford_pets": ("oxford_pets", "split_zhou_OxfordPets.json", "images"),
    "stanford_cars": ("stanford_cars", "split_zhou_StanfordCars.json", ""),
    "sun397": ("sun397", "split_zhou_SUN397.json", "SUN397"),
}


@dataclass
class Domain:
    name: str
    classnames: list[str]
    items: dict  # split -> list of (image source, label); source is a path, or an index into `mnist`
    mnist: dict | None = None  # split -> torchvision MNIST dataset (only for the MNIST domain)

    def load(self, split: str, i: int) -> Image.Image:
        src, _ = self.items[split][i]
        if self.mnist is not None:
            return self.mnist[split][src][0].convert("RGB")
        return Image.open(src).convert("RGB")


def _coop(name: str) -> Domain:
    folder, split_file, img_root = _COOP[name]
    split = json.loads((ROOT / folder / split_file).read_text())
    base = ROOT / folder / img_root
    items = {part: [(str(base / p), int(lab)) for p, lab, _ in split[part]] for part in ("train", "val", "test")}
    names = {int(lab): c for p, lab, c in split["train"]}
    return Domain(name, [names[k] for k in sorted(names)], items)


def _aircraft() -> Domain:
    d = ROOT / "fgvc_aircraft"
    classnames = [line.strip() for line in (d / "variants.txt").read_text().splitlines() if line.strip()]
    idx = {c: i for i, c in enumerate(classnames)}

    def read(fname):
        out = []
        for line in (d / fname).read_text().splitlines():
            parts = line.strip().split(" ")
            out.append((str(d / "images" / (parts[0] + ".jpg")), idx[" ".join(parts[1:])]))
        return out

    return Domain("aircraft", classnames, {"train": read("images_variant_train.txt"),
                                           "val": read("images_variant_val.txt"),
                                           "test": read("images_variant_test.txt")})


def _mnist() -> Domain:
    from torchvision import datasets
    tr = datasets.MNIST(str(ROOT / "mnist"), train=True, download=False)
    te = datasets.MNIST(str(ROOT / "mnist"), train=False, download=False)
    # RAIL: classnames 'number: "k"' (underscores -> spaces is a no-op); no separate val split,
    # so we hold out a fixed 10% of the training set as validation (RAIL itself uses none).
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(tr))
    n_val = len(tr) // 10
    tl = tr.targets.numpy()
    items = {"train": [(int(i), int(tl[i])) for i in perm[n_val:]],
             "val": [(int(i), int(tl[i])) for i in perm[:n_val]],
             "test": [(i, int(t)) for i, t in enumerate(te.targets.numpy())]}
    classnames = [f'number: "{c}"' for c in tr.classes]  # exactly RAIL's: 'number: "0 - zero"', ...
    return Domain("mnist", classnames, items, mnist={"train": tr, "val": tr, "test": te})


CIFAR_ROOT = Path(__file__).resolve().parents[2] / "data" / "generated" / "cifar100_raw"


def _cifar100() -> Domain:
    """Only in the 11-domain (MTIL-style) 5-shot X-TAIL variant. RAIL: torchvision class names with
    underscores as spaces; like MNIST, a fixed 10% of the training set is held out as validation."""
    from torchvision import datasets
    tr = datasets.CIFAR100(str(CIFAR_ROOT), train=True, download=False)
    te = datasets.CIFAR100(str(CIFAR_ROOT), train=False, download=False)
    perm = np.random.default_rng(0).permutation(len(tr))
    n_val = len(tr) // 10
    tl = np.asarray(tr.targets)
    items = {"train": [(int(i), int(tl[i])) for i in perm[n_val:]],
             "val": [(int(i), int(tl[i])) for i in perm[:n_val]],
             "test": [(i, int(t)) for i, t in enumerate(te.targets)]}
    return Domain("cifar100", [c.replace("_", " ") for c in tr.classes], items, mnist={"train": tr, "val": tr, "test": te})


DOMAINS_11 = ("aircraft", "caltech101", "cifar100", "dtd", "eurosat", "oxford_flowers", "food101", "mnist",
              "oxford_pets", "stanford_cars", "sun397")  # 5-shot X-TAIL variant (Rank-1 LoRA pool, MoRAM)
ORDER_II = ("stanford_cars", "aircraft", "oxford_pets", "food101", "sun397", "mnist", "oxford_flowers", "dtd",
            "caltech101", "eurosat")  # RAIL Appendix G


def load_domain(name: str) -> Domain:
    if name == "aircraft":
        return _aircraft()
    if name == "mnist":
        return _mnist()
    if name == "cifar100":
        return _cifar100()
    dom = _coop(name)
    if name == "eurosat":  # RAIL's eurosat.py appends this to every class name (no other loader renames)
        dom.classnames = [c + " satellite location" for c in dom.classnames]
    return dom


def clip_preprocess(img: Image.Image, size: int = 224) -> np.ndarray:
    """CLIP's standard eval preprocessing (bicubic resize of the short side, center crop), as a
    uint8 (size, size, 3) array; normalization happens in the encoder."""
    w, h = img.size
    s = size / min(w, h)
    img = img.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BICUBIC)
    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    return np.asarray(img.crop((left, top, left + size, top + size)))
