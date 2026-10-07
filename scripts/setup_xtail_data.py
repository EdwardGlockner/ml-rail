"""Build the X-TAIL benchmark data (RAIL, NeurIPS 2024; CoOp dataset layout) under $XTAIL_ROOT/datasets.

Provenance, since several official links are dead (checked 2026-10-01):
  caltech101, dtd, eurosat, oxford_flowers, fgvc_aircraft  torchvision downloads, rearranged into the CoOp layout
  food-101, oxford_pets                                    official URLs (ETH, Oxford VGG)
  stanford_cars   official Stanford links dead; rebuilt from HF Multimodal-Fatima/StanfordCars_{train,test},
                  which keeps the original filenames (00001.jpg ...)
  sun397          official Princeton links dead; rebuilt from HF 1aurent/SUN397 (original sun_*.jpg names and
                  class paths; images stored as RGB, so originally non-JPEG files may differ at pixel level);
                  only images named in the CoOp split are written
  mnist, cifar100 torchvision (cifar100 only for the 11-domain 5-shot variant)
  split files     CoOp's split_zhou_*.json from Google Drive (links in CoOp DATASETS.md)
Finally every path in every split file is checked to exist.

Usage:
    python scripts/setup_xtail_data.py [--only name1,name2]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tarfile
import time
import urllib.request
from pathlib import Path

# same default as data/real/xtail.py: xtail_data/ in the repository, or $XTAIL_ROOT
XTAIL_ROOT = Path(os.environ.get("XTAIL_ROOT", Path(__file__).resolve().parents[1] / "xtail_data"))
ROOT = XTAIL_ROOT / "datasets"
DL = XTAIL_ROOT / "_downloads"

SPLITS = {  # CoOp Google Drive ids
    "caltech101/split_zhou_Caltech101.json": "1hyarUivQE36mY6jSomru6Fjd-JzwcCzN",
    "oxford_pets/split_zhou_OxfordPets.json": "1501r8Ber4nNKvmlFVQZ8SeUHTcdTTEqs",
    "stanford_cars/split_zhou_StanfordCars.json": "1ObCFbaAgVu0I-k_Au-gIUcefirdAuizT",
    "oxford_flowers/split_zhou_OxfordFlowers.json": "1Pp0sRXzZFZq15zVOzKjKBu4A9i01nozT",
    "oxford_flowers/cat_to_name.json": "1AkcxCXeK_RCGCEC_GvmWxjcjaNhu-at0",
    "food-101/split_zhou_Food101.json": "1QK0tGi096I0Ba6kggatX1ee6dJFIcEJl",
    "sun397/split_zhou_SUN397.json": "1y2RD81BYuiyvebdN-JymPfyWYcd8_MUq",
    "dtd/split_zhou_DescribableTextures.json": "1u3_QfB467jqHgNXC00UIzbLZRQCg2S7x",
    "eurosat/split_zhou_EuroSAT.json": "1Ip7yaCWFi0eaOFUGga0lUdVi_DDQth1o",
}
IMAGE_ROOT = {  # split-file paths are relative to these (as in RAIL's loaders)
    "caltech101": "101_ObjectCategories", "oxford_pets": "images", "stanford_cars": "",
    "oxford_flowers": "jpg", "food-101": "images", "sun397": "SUN397", "dtd": "images", "eurosat": "2750",
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fetch(url, dst: Path):
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    log(f"downloading {url}")
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 22)
    tmp.rename(dst)
    return dst


def copytree(src: Path, dst: Path):
    if dst.exists():
        return
    log(f"copying {src} -> {dst}")
    shutil.copytree(src, dst)


def splits():
    import gdown
    for rel, fid in SPLITS.items():
        dst = ROOT / rel
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            gdown.download(id=fid, output=str(dst), quiet=True)
            log(f"split file {rel}: {'ok' if dst.exists() else 'FAILED'}")


def torchvision_datasets():
    """Download five datasets with torchvision and copy them into the CoOp layout RAIL's loaders expect."""
    from torchvision import datasets
    tv = DL / "torchvision"
    if not (ROOT / "caltech101" / "101_ObjectCategories").exists():
        datasets.Caltech101(str(tv / "caltech101_raw"), download=True)
        copytree(tv / "caltech101_raw" / "caltech101" / "101_ObjectCategories", ROOT / "caltech101" / "101_ObjectCategories")
    if not (ROOT / "dtd" / "images").exists():
        datasets.DTD(str(tv / "dtd_raw"), download=True)
        for sub in ("images", "imdb", "labels"):
            copytree(tv / "dtd_raw" / "dtd" / "dtd" / sub, ROOT / "dtd" / sub)
    if not (ROOT / "eurosat" / "2750").exists():
        datasets.EuroSAT(str(tv / "eurosat_raw"), download=True)
        copytree(tv / "eurosat_raw" / "eurosat" / "2750", ROOT / "eurosat" / "2750")
    if not (ROOT / "oxford_flowers" / "jpg").exists():
        datasets.Flowers102(str(tv / "flowers102_raw"), download=True)
        copytree(tv / "flowers102_raw" / "flowers-102" / "jpg", ROOT / "oxford_flowers" / "jpg")
        for f in ("imagelabels.mat", "setid.mat"):
            shutil.copy(tv / "flowers102_raw" / "flowers-102" / f, ROOT / "oxford_flowers" / f)
    if not (ROOT / "fgvc_aircraft" / "images").exists():
        datasets.FGVCAircraft(str(tv / "fgvc_aircraft_raw"), download=True)
        copytree(tv / "fgvc_aircraft_raw" / "fgvc-aircraft-2013b" / "data", ROOT / "fgvc_aircraft")


def food_pets():
    if not (ROOT / "food-101" / "images").exists():
        tar = fetch("https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz", DL / "food-101.tar.gz")
        log("extracting food-101")
        with tarfile.open(tar) as t:
            t.extractall(ROOT)
    for name in ("images", "annotations"):
        if not (ROOT / "oxford_pets" / name).exists():
            tar = fetch(f"https://thor.robots.ox.ac.uk/pets/{name}.tar.gz", DL / f"pets_{name}.tar.gz")
            log(f"extracting pets {name}")
            with tarfile.open(tar) as t:
                t.extractall(ROOT / "oxford_pets")


def hf_parquets(repo, prefix):
    from huggingface_hub import HfApi, hf_hub_download
    files = [f for f in HfApi().list_repo_files(repo, repo_type="dataset") if f.startswith(prefix) and f.endswith(".parquet")]
    for f in sorted(files):
        yield f, hf_hub_download(repo, f, repo_type="dataset", local_dir=str(DL / repo.replace("/", "__")))


def cars():
    import pyarrow.parquet as pq
    for split, repo in (("cars_train", "Multimodal-Fatima/StanfordCars_train"), ("cars_test", "Multimodal-Fatima/StanfordCars_test")):
        out = ROOT / "stanford_cars" / split
        if out.exists() and len(list(out.iterdir())) > 8000:
            continue
        out.mkdir(parents=True, exist_ok=True)
        for f, path in hf_parquets(repo, "data/"):
            t = pq.read_table(path, columns=["image"])
            for img in t.column("image").to_pylist():
                (out / img["path"]).write_bytes(img["bytes"])
            log(f"cars {split}: {f} done")


def sun397():
    import pyarrow.parquet as pq
    split = json.loads((ROOT / "sun397" / "split_zhou_SUN397.json").read_text())
    wanted = {p for part in ("train", "val", "test") for p, _, _ in split[part]}
    base = ROOT / "sun397" / "SUN397"
    have = {str(p.relative_to(base)).replace("\\", "/") for p in base.rglob("*.jpg")} if base.exists() else set()
    if wanted <= have:
        return
    import yaml  # class-label names live in the dataset card
    card = urllib.request.urlopen(urllib.request.Request("https://huggingface.co/datasets/1aurent/SUN397/raw/main/README.md",
                                                         headers={"User-Agent": "Mozilla/5.0"}), timeout=60).read().decode()
    names = yaml.safe_load(card.split("---")[1])["dataset_info"]["features"][1]["dtype"]["class_label"]["names"]
    for f, path in hf_parquets("1aurent/SUN397", "data/"):
        pf = pq.ParquetFile(path)
        n = 0
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg, columns=["image", "label"])
            for img, lab in zip(t.column("image").to_pylist(), t.column("label").to_pylist()):
                rel = names[str(lab)].lstrip("/") + "/" + img["path"]
                if rel in wanted:
                    dst = base / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    dst.write_bytes(img["bytes"])
                    n += 1
        pf.close()  # Windows: the file must be closed before it can be deleted
        Path(path).unlink()  # keep the disk footprint small
        log(f"sun397: {f} -> {n} split images written")


def mnist():
    from torchvision import datasets
    datasets.MNIST(str(ROOT / "mnist"), train=True, download=True)
    datasets.MNIST(str(ROOT / "mnist"), train=False, download=True)


def cifar100():
    from torchvision import datasets
    datasets.CIFAR100(str(ROOT / "cifar100"), train=True, download=True)
    datasets.CIFAR100(str(ROOT / "cifar100"), train=False, download=True)


def verify():
    ok = True
    for rel in SPLITS:
        if "split_zhou" not in rel:
            continue
        name = rel.split("/")[0]
        split = json.loads((ROOT / rel).read_text())
        base = ROOT / name / IMAGE_ROOT[name]
        missing = [p for part in ("train", "val", "test") for p, _, _ in split[part] if not (base / p).exists()]
        counts = {part: len(split[part]) for part in ("train", "val", "test")}
        log(f"verify {name:15s} {counts} classes {len({c for _, _, c in split['train']})} missing {len(missing)}")
        ok &= not missing
    air = ROOT / "fgvc_aircraft"
    n_air = sum(1 for line in (air / "images_variant_test.txt").read_text().splitlines() if (air / "images" / (line.split(" ")[0] + ".jpg")).exists())
    log(f"verify fgvc_aircraft test images present: {n_air}")
    log("ALL SPLITS COMPLETE" if ok else "SOME IMAGES MISSING")


STEPS = {"splits": splits, "torchvision": torchvision_datasets, "food_pets": food_pets, "cars": cars, "sun397": sun397,
         "mnist": mnist, "cifar100": cifar100, "verify": verify}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--only", default=",".join(STEPS))
    for name in p.parse_args().only.split(","):
        log(f"=== {name}")
        STEPS[name]()
