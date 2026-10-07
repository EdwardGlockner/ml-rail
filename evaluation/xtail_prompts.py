"""Zero-shot text heads for X-TAIL.

Every class belongs to a known dataset (class names come with the benchmark's label set), so a
class's text may use its own dataset's template(s) without any domain hint for test images.

Heads:
  single    RAIL's prompt, "a photo of a {}."                                   (the benchmark default)
  generic   Tip-Adapter's 7 templates + RAIL's prompt, same for every dataset
  t1a       the single per-dataset templates in RAIL's/CoOp's dataset loaders
  t1b       CLIP's published per-dataset prompt ensembles (third_party/clip_prompts.md)
  t1a+generic, t1b+generic   normalized average of the two heads
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
from data.real.xtail import XTAIL_ROOT

GENERIC = ("a photo of a {}.", "itap of a {}.", "a bad photo of the {}.", "a origami {}.",
           "a photo of the large {}.", "a {} in a video game.", "art of the {}.", "a photo of the small {}.")

T1A = {  # RAIL's scenario_datasets/*.py `template` (MNIST: its class names already read 'number: "..."')
    "aircraft": ("a photo of a {}, a type of aircraft.",), "caltech101": ("a photo of a {}.",),
    "dtd": ("{} texture.",), "eurosat": ("a centered satellite photo of {}.",),
    "oxford_flowers": ("a photo of a {}, a type of flower.",), "food101": ("a photo of {}, a type of food.",),
    "mnist": ("a photo of a {}.",), "oxford_pets": ("a photo of a {}, a type of pet.",),
    "stanford_cars": ("a photo of a {}.",), "sun397": ("a photo of a {}.",), "cifar100": ("a photo of a {}.",),
}
_CLIP_SECTION = {"aircraft": "FGVCAircraft", "caltech101": "Caltech101", "dtd": "DescribableTextures",
                 "eurosat": "EuroSAT", "oxford_flowers": "Flowers102", "food101": "Food101",
                 "oxford_pets": "OxfordPets", "stanford_cars": "StanfordCars", "sun397": "SUN397",
                 "cifar100": "CIFAR100"}


def clip_templates(domain: str) -> tuple[str, ...]:
    """CLIP's per-dataset templates; MNIST keeps the generic set (see T1A's note)."""
    if domain not in _CLIP_SECTION:
        return GENERIC
    text = (Path(__file__).resolve().parents[1] / "third_party" / "clip_prompts.md").read_text(encoding="utf-8")
    start = text.index(f"## {_CLIP_SECTION[domain]}")
    block = text[text.index("templates = [", start): text.index("]", text.index("templates = [", start))]
    return tuple(re.findall(r"'([^']*\{\}[^']*)'", block))


def head_templates(head: str, domain: str) -> list[tuple[str, ...]]:
    """Template sets whose (normalized) embeddings are averaged with equal weight per set."""
    parts = {"single": [("a photo of a {}.",)], "generic": [GENERIC], "t1a": [T1A[domain]],
             "t1b": [clip_templates(domain)]}
    if "+" in head:
        a, b = head.split("+")
        return parts[a] + parts[b]
    return parts[head]


def build_text(data: dict, head: str, cache_dir=(XTAIL_ROOT / "features/vitb16_text")) -> dict:
    """Set each domain's `text` to the chosen head's class embeddings (cached per head and domain)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    zs = None
    for name, d in data.items():
        path = cache_dir / f"{head.replace('+', '_')}_{name}.npy"
        if not path.exists():
            if zs is None:
                from features.clip_zero_shot import ClipZeroShotClassifier
                zs = ClipZeroShotClassifier("openai/clip-vit-base-patch16")
            sets = []
            for templates in head_templates(head, name):
                e = torch.stack([zs.encode_text(list(d["classnames"]), t) for t in templates]).mean(0)
                sets.append(e / e.norm(dim=-1, keepdim=True))
            e = torch.stack(sets).mean(0)
            np.save(path, (e / e.norm(dim=-1, keepdim=True)).numpy().astype(np.float32))
        d["text"] = np.load(path)
    return data
