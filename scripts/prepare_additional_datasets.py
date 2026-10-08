"""Preparation of Danger-Zone and Fall Detection datasets."""

import json
import logging
import urllib.request
from pathlib import Path
from PIL import Image
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prepare_additional_datasets")

BASE_DIR = Path(__file__).resolve().parent.parent


def prepare_danger_zone_dataset(max_train: int = 20, max_val: int = 5, max_test: int = 2):
    """Fetch and prepare Danger-Zone dataset from Construction Site Safety repo (MIT License)."""
    proc_dir = BASE_DIR / "datasets" / "danger_zone" / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        (proc_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (proc_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    repo_api = "https://api.github.com/repos/VoxDroid/Construction-Site-Safety-PPE-Detection/contents/Model-Training/Dataset"
    req_headers = {"User-Agent": "AI-Safety-Model-Lab/0.2.0"}

    # Target class mapping:
    # 5: Person -> 0: person
    # 8: machinery -> 1: machinery
    # 9: vehicle -> 2: vehicle
    # 6: Safety Cone -> 3: safety_cone
    remap = {5: 0, 8: 1, 9: 2, 6: 3}

    splits_target = [
        ("train", "train", max_train),
        ("valid", "val", max_val),
        ("test", "test", max_test),
    ]

    for gh_split, local_split, max_count in splits_target:
        logger.info(f"Fetching Danger-Zone {local_split} split (target: {max_count})...")
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
            lbl_raw_url = f"https://raw.githubusercontent.com/VoxDroid/Construction-Site-Safety-PPE-Detection/main/Model-Training/Dataset/{gh_split}/labels/{fname}"
            try:
                with urllib.request.urlopen(lbl_raw_url) as resp:
                    lbl_text = resp.read().decode("utf-8")
            except Exception as e:
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
            img_raw_url = f"https://raw.githubusercontent.com/VoxDroid/Construction-Site-Safety-PPE-Detection/main/Model-Training/Dataset/{gh_split}/images/{img_fname}"
            local_img_path = proc_dir / "images" / local_split / img_fname
            local_lbl_path = proc_dir / "labels" / local_split / fname

            try:
                urllib.request.urlretrieve(img_raw_url, local_img_path)
                with Image.open(local_img_path) as img:
                    img.verify()
            except Exception:
                if local_img_path.exists():
                    local_img_path.unlink()
                continue

            with open(local_lbl_path, "w", encoding="utf-8") as f:
                f.write("\n".join(new_lines) + "\n")

            collected += 1
            logger.info(f"Danger-Zone {local_split}: [{collected}/{max_count}] {img_fname}")

    yaml_dict = {
        "path": str(proc_dir.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 4,
        "names": {
            0: "person",
            1: "machinery",
            2: "vehicle",
            3: "safety_cone"
        }
    }
    with open(proc_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.dump(yaml_dict, f, sort_keys=False)

    logger.info("Danger-Zone dataset preparation complete.")


def prepare_fall_detection_dataset():
    """Prepare Fall Detection dataset structure."""
    proc_dir = BASE_DIR / "datasets" / "fall_detection" / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        (proc_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (proc_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Use worker images from PPE test set as worker sample images for fall posture evaluation
    ppe_proc = BASE_DIR / "datasets" / "ppe" / "processed"
    import shutil
    for split in ["train", "val", "test"]:
        src_imgs = list((ppe_proc / "images" / split).glob("*.jpg"))
        for img in src_imgs[:5]:
            shutil.copy2(img, proc_dir / "images" / split / img.name)
            lbl = ppe_proc / "labels" / split / (img.stem + ".txt")
            if lbl.exists():
                with open(lbl, "r", encoding="utf-8") as f:
                    lines = [l for l in f.read().splitlines() if l.strip().startswith("0 ")]
                with open(proc_dir / "labels" / split / lbl.name, "w", encoding="utf-8") as f:
                    f.write("\n".join(lines) + "\n")

    yaml_dict = {
        "path": str(proc_dir.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 1,
        "names": {
            0: "person"
        }
    }
    with open(proc_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.dump(yaml_dict, f, sort_keys=False)

    logger.info("Fall Detection dataset preparation complete.")


if __name__ == "__main__":
    prepare_danger_zone_dataset(max_train=20, max_val=5, max_test=2)
    prepare_fall_detection_dataset()
