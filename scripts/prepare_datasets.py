"""Dataset Preparation Script for AI Safety Workshop.

Fetches and prepares legitimate, open-licensed samples for:
1. PPE Detection (Construction Site Safety - MIT / CC BY 4.0)
2. Fire & Smoke Detection (Roboflow Universe fire-8 - CC BY 4.0)

Converts raw datasets into canonical YOLO structure under:
datasets/ppe/
datasets/fire_smoke/
"""

import json
import logging
import os
import shutil
import urllib.request
from pathlib import Path
from PIL import Image
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prepare_datasets")

BASE_DIR = Path(__file__).resolve().parent.parent


def fetch_ppe_dataset(max_train: int = 20, max_val: int = 5, max_test: int = 2):
    """Fetch and prepare PPE dataset subset from VoxDroid Construction-Site-Safety repo (MIT License)."""
    raw_dir = BASE_DIR / "datasets" / "ppe" / "raw"
    proc_dir = BASE_DIR / "datasets" / "ppe" / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    # Clean previous processed
    for split in ["train", "val", "test"]:
        (proc_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (proc_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    repo_api = "https://api.github.com/repos/VoxDroid/Construction-Site-Safety-PPE-Detection/contents/Model-Training/Dataset"
    req_headers = {"User-Agent": "AI-Safety-Model-Lab/0.2.0"}

    # Target class mapping:
    # Original: 0: Hardhat, 5: Person, 7: Safety Vest
    # New canonical: 0: person, 1: helmet, 2: vest
    remap = {5: 0, 0: 1, 7: 2}

    splits_target = [
        ("train", "train", max_train),
        ("valid", "val", max_val),
        ("test", "test", max_test),
    ]

    for gh_split, local_split, max_count in splits_target:
        logger.info(f"Fetching PPE {local_split} split (target: {max_count})...")
        list_url = f"{repo_api}/{gh_split}/labels"
        req = urllib.request.Request(list_url, headers=req_headers)
        with urllib.request.urlopen(req) as resp:
            label_files = json.loads(resp.read().decode())

        collected = 0
        for item in label_files:
            if collected >= max_count:
                break
            fname = item["name"]
            stem = fname.rsplit(".", 1)[0]
            # Fetch label
            lbl_raw_url = f"https://raw.githubusercontent.com/VoxDroid/Construction-Site-Safety-PPE-Detection/main/Model-Training/Dataset/{gh_split}/labels/{fname}"
            try:
                with urllib.request.urlopen(lbl_raw_url) as resp:
                    lbl_text = resp.read().decode("utf-8")
            except Exception as e:
                logger.warning(f"Failed to fetch label {fname}: {e}")
                continue

            # Parse lines and check if contains target classes
            new_lines = []
            for line in lbl_text.strip().splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    if cls_id in remap:
                        new_cls = remap[cls_id]
                        new_lines.append(f"{new_cls} " + " ".join(parts[1:]))

            if not new_lines:
                continue  # Skip image with zero target annotations

            # Corresponding image in repo: stem + .jpg
            # Note: in VoxDroid, images are in Model-Training/Dataset/<gh_split>/images/<stem>.jpg
            img_fname = f"{stem}.jpg"
            img_raw_url = f"https://raw.githubusercontent.com/VoxDroid/Construction-Site-Safety-PPE-Detection/main/Model-Training/Dataset/{gh_split}/images/{img_fname}"
            local_img_path = proc_dir / "images" / local_split / img_fname
            local_lbl_path = proc_dir / "labels" / local_split / fname

            try:
                urllib.request.urlretrieve(img_raw_url, local_img_path)
                # Verify image is valid
                with Image.open(local_img_path) as img:
                    img.verify()
            except Exception as e:
                if local_img_path.exists():
                    local_img_path.unlink()
                logger.warning(f"Failed to fetch image {img_fname}: {e}")
                continue

            # Save remapped label
            with open(local_lbl_path, "w", encoding="utf-8") as f:
                f.write("\n".join(new_lines) + "\n")

            collected += 1
            logger.info(f"PPE {local_split}: [{collected}/{max_count}] {img_fname}")

    # Generate canonical data.yaml
    yaml_dict = {
        "path": str(proc_dir.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 3,
        "names": {
            0: "person",
            1: "helmet",
            2: "vest"
        }
    }
    with open(proc_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.dump(yaml_dict, f, sort_keys=False)

    logger.info("PPE dataset preparation complete.")


def fetch_fire_smoke_dataset(max_train: int = 20, max_val: int = 5, max_test: int = 2):
    """Fetch and prepare Fire & Smoke dataset subset from Abonia1 fire-8 repo (CC BY 4.0 License)."""
    raw_dir = BASE_DIR / "datasets" / "fire_smoke" / "raw"
    proc_dir = BASE_DIR / "datasets" / "fire_smoke" / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        (proc_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (proc_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    repo_api = "https://api.github.com/repos/Abonia1/YOLOv8-Fire-and-Smoke-Detection/contents/datasets/fire-8"
    req_headers = {"User-Agent": "AI-Safety-Model-Lab/0.2.0"}

    # Target class mapping:
    # Original: 0: Fire, 1: default, 2: smoke
    # New canonical: 0: fire, 1: smoke
    remap = {0: 0, 2: 1}

    splits_target = [
        ("train", "train", max_train),
        ("valid", "val", max_val),
        ("test", "test", max_test),
    ]

    for gh_split, local_split, max_count in splits_target:
        logger.info(f"Fetching Fire/Smoke {local_split} split (target: {max_count})...")
        list_url = f"{repo_api}/{gh_split}/labels"
        req = urllib.request.Request(list_url, headers=req_headers)
        with urllib.request.urlopen(req) as resp:
            label_files = json.loads(resp.read().decode())

        collected = 0
        for item in label_files:
            if collected >= max_count:
                break
            fname = item["name"]
            stem = fname.rsplit(".", 1)[0]
            # Fetch label
            lbl_raw_url = f"https://raw.githubusercontent.com/Abonia1/YOLOv8-Fire-and-Smoke-Detection/main/datasets/fire-8/{gh_split}/labels/{fname}"
            try:
                with urllib.request.urlopen(lbl_raw_url) as resp:
                    lbl_text = resp.read().decode("utf-8")
            except Exception as e:
                logger.warning(f"Failed to fetch label {fname}: {e}")
                continue

            new_lines = []
            for line in lbl_text.strip().splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    if cls_id in remap:
                        new_cls = remap[cls_id]
                        new_lines.append(f"{new_cls} " + " ".join(parts[1:]))

            if not new_lines:
                continue

            img_fname = f"{stem}.jpg"
            img_raw_url = f"https://raw.githubusercontent.com/Abonia1/YOLOv8-Fire-and-Smoke-Detection/main/datasets/fire-8/{gh_split}/images/{img_fname}"
            local_img_path = proc_dir / "images" / local_split / img_fname
            local_lbl_path = proc_dir / "labels" / local_split / fname

            try:
                urllib.request.urlretrieve(img_raw_url, local_img_path)
                with Image.open(local_img_path) as img:
                    img.verify()
            except Exception as e:
                if local_img_path.exists():
                    local_img_path.unlink()
                logger.warning(f"Failed to fetch image {img_fname}: {e}")
                continue

            with open(local_lbl_path, "w", encoding="utf-8") as f:
                f.write("\n".join(new_lines) + "\n")

            collected += 1
            logger.info(f"Fire/Smoke {local_split}: [{collected}/{max_count}] {img_fname}")

    # Generate canonical data.yaml
    yaml_dict = {
        "path": str(proc_dir.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 2,
        "names": {
            0: "fire",
            1: "smoke"
        }
    }
    with open(proc_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.dump(yaml_dict, f, sort_keys=False)

    logger.info("Fire/Smoke dataset preparation complete.")


if __name__ == "__main__":
    fetch_ppe_dataset(max_train=20, max_val=5, max_test=2)
    fetch_fire_smoke_dataset(max_train=20, max_val=5, max_test=2)
