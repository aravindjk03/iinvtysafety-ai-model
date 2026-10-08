"""Deep Scale-Up of Industrial Safety Datasets for AI Safety Workshop.

Expands:
1. PPE Detection Dataset (60 train, 15 val, 6 test)
2. Fire & Smoke Dataset (60 train, 15 val, 6 test)
3. Danger-Zone Dataset (60 train, 15 val, 6 test)
4. Fall Detection Pose Dataset (25 train, 10 val, 5 test, 17 keypoints)
"""

import json
import logging
import urllib.request
from pathlib import Path
from PIL import Image
import yaml
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("scale_up_datasets")

BASE_DIR = Path(__file__).resolve().parent.parent


def scale_ppe(max_train: int = 60, max_val: int = 15, max_test: int = 6):
    """Scale PPE dataset."""
    proc_dir = BASE_DIR / "datasets" / "ppe" / "processed"
    for split in ["train", "val", "test"]:
        (proc_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (proc_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    repo_api = "https://api.github.com/repos/VoxDroid/Construction-Site-Safety-PPE-Detection/contents/Model-Training/Dataset"
    req_headers = {"User-Agent": "AI-Safety-Model-Lab/0.2.0"}
    remap = {5: 0, 0: 1, 7: 2}

    splits_target = [
        ("train", "train", max_train),
        ("valid", "val", max_val),
        ("test", "test", max_test),
    ]

    for gh_split, local_split, max_count in splits_target:
        logger.info(f"Expanding PPE {local_split} split (target: {max_count})...")
        list_url = f"{repo_api}/{gh_split}/labels"
        req = urllib.request.Request(list_url, headers=req_headers)
        with urllib.request.urlopen(req) as resp:
            label_files = json.loads(resp.read().decode())

        collected = len(list((proc_dir / "images" / local_split).glob("*.jpg")))
        logger.info(f"PPE {local_split} existing: {collected}")
        for item in label_files:
            if collected >= max_count:
                break
            fname = item["name"]
            stem = fname.rsplit(".", 1)[0]
            local_img_path = proc_dir / "images" / local_split / f"{stem}.jpg"
            local_lbl_path = proc_dir / "labels" / local_split / fname

            if local_img_path.exists() and local_lbl_path.exists():
                continue

            lbl_raw_url = f"https://raw.githubusercontent.com/VoxDroid/Construction-Site-Safety-PPE-Detection/main/Model-Training/Dataset/{gh_split}/labels/{fname}"
            try:
                with urllib.request.urlopen(lbl_raw_url) as resp:
                    lbl_text = resp.read().decode("utf-8")
            except Exception:
                continue

            new_lines = []
            for line in lbl_text.strip().splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    if cls_id in remap:
                        new_lines.append(f"{remap[cls_id]} " + " ".join(parts[1:]))

            if not new_lines:
                continue

            img_fname = f"{stem}.jpg"
            img_raw_url = f"https://raw.githubusercontent.com/VoxDroid/Construction-Site-Safety-PPE-Detection/main/Model-Training/Dataset/{gh_split}/images/{img_fname}"
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
            if collected % 10 == 0 or collected == max_count:
                logger.info(f"PPE {local_split}: [{collected}/{max_count}]")

    yaml_dict = {
        "path": str(proc_dir.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 3,
        "names": {0: "person", 1: "helmet", 2: "vest"},
    }
    with open(proc_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.dump(yaml_dict, f, sort_keys=False)
    logger.info("PPE scaled dataset complete.")


def scale_fire_smoke(max_train: int = 60, max_val: int = 15, max_test: int = 6):
    """Scale Fire & Smoke dataset."""
    proc_dir = BASE_DIR / "datasets" / "fire_smoke" / "processed"
    for split in ["train", "val", "test"]:
        (proc_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (proc_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    repo_api = "https://api.github.com/repos/Abonia1/YOLOv8-Fire-and-Smoke-Detection/contents/datasets/fire-8"
    req_headers = {"User-Agent": "AI-Safety-Model-Lab/0.2.0"}
    remap = {0: 0, 2: 1}

    splits_target = [
        ("train", "train", max_train),
        ("valid", "val", max_val),
        ("test", "test", max_test),
    ]

    for gh_split, local_split, max_count in splits_target:
        logger.info(f"Expanding Fire/Smoke {local_split} split (target: {max_count})...")
        list_url = f"{repo_api}/{gh_split}/labels"
        req = urllib.request.Request(list_url, headers=req_headers)
        with urllib.request.urlopen(req) as resp:
            label_files = json.loads(resp.read().decode())

        collected = len(list((proc_dir / "images" / local_split).glob("*.jpg")))
        logger.info(f"Fire/Smoke {local_split} existing: {collected}")
        for item in label_files:
            if collected >= max_count:
                break
            fname = item["name"]
            stem = fname.rsplit(".", 1)[0]
            local_img_path = proc_dir / "images" / local_split / f"{stem}.jpg"
            local_lbl_path = proc_dir / "labels" / local_split / fname

            if local_img_path.exists() and local_lbl_path.exists():
                continue

            lbl_raw_url = f"https://raw.githubusercontent.com/Abonia1/YOLOv8-Fire-and-Smoke-Detection/main/datasets/fire-8/{gh_split}/labels/{fname}"
            try:
                with urllib.request.urlopen(lbl_raw_url) as resp:
                    lbl_text = resp.read().decode("utf-8")
            except Exception:
                continue

            new_lines = []
            for line in lbl_text.strip().splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    if cls_id in remap:
                        new_lines.append(f"{remap[cls_id]} " + " ".join(parts[1:]))

            if not new_lines:
                continue

            img_fname = f"{stem}.jpg"
            img_raw_url = f"https://raw.githubusercontent.com/Abonia1/YOLOv8-Fire-and-Smoke-Detection/main/datasets/fire-8/{gh_split}/images/{img_fname}"
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
            if collected % 10 == 0 or collected == max_count:
                logger.info(f"Fire/Smoke {local_split}: [{collected}/{max_count}]")

    yaml_dict = {
        "path": str(proc_dir.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 2,
        "names": {0: "fire", 1: "smoke"},
    }
    with open(proc_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.dump(yaml_dict, f, sort_keys=False)
    logger.info("Fire/Smoke scaled dataset complete.")


def scale_danger_zone(max_train: int = 60, max_val: int = 15, max_test: int = 6):
    """Scale Danger-Zone dataset."""
    proc_dir = BASE_DIR / "datasets" / "danger_zone" / "processed"
    for split in ["train", "val", "test"]:
        (proc_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (proc_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    repo_api = "https://api.github.com/repos/VoxDroid/Construction-Site-Safety-PPE-Detection/contents/Model-Training/Dataset"
    req_headers = {"User-Agent": "AI-Safety-Model-Lab/0.2.0"}
    remap = {5: 0, 8: 1, 9: 2, 6: 3}

    splits_target = [
        ("train", "train", max_train),
        ("valid", "val", max_val),
        ("test", "test", max_test),
    ]

    for gh_split, local_split, max_count in splits_target:
        logger.info(f"Expanding Danger-Zone {local_split} split (target: {max_count})...")
        list_url = f"{repo_api}/{gh_split}/labels"
        req = urllib.request.Request(list_url, headers=req_headers)
        with urllib.request.urlopen(req) as resp:
            label_files = json.loads(resp.read().decode())

        collected = len(list((proc_dir / "images" / local_split).glob("*.jpg")))
        logger.info(f"Danger-Zone {local_split} existing: {collected}")
        for item in label_files:
            if collected >= max_count:
                break
            fname = item["name"]
            stem = fname.rsplit(".", 1)[0]
            local_img_path = proc_dir / "images" / local_split / f"{stem}.jpg"
            local_lbl_path = proc_dir / "labels" / local_split / fname

            if local_img_path.exists() and local_lbl_path.exists():
                continue

            lbl_raw_url = f"https://raw.githubusercontent.com/VoxDroid/Construction-Site-Safety-PPE-Detection/main/Model-Training/Dataset/{gh_split}/labels/{fname}"
            try:
                with urllib.request.urlopen(lbl_raw_url) as resp:
                    lbl_text = resp.read().decode("utf-8")
            except Exception:
                continue

            new_lines = []
            for line in lbl_text.strip().splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    if cls_id in remap:
                        new_lines.append(f"{remap[cls_id]} " + " ".join(parts[1:]))

            if not new_lines:
                continue

            img_fname = f"{stem}.jpg"
            img_raw_url = f"https://raw.githubusercontent.com/VoxDroid/Construction-Site-Safety-PPE-Detection/main/Model-Training/Dataset/{gh_split}/images/{img_fname}"
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
            if collected % 10 == 0 or collected == max_count:
                logger.info(f"Danger-Zone {local_split}: [{collected}/{max_count}]")

    yaml_dict = {
        "path": str(proc_dir.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 4,
        "names": {0: "person", 1: "machinery", 2: "vehicle", 3: "safety_cone"},
    }
    with open(proc_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.dump(yaml_dict, f, sort_keys=False)
    logger.info("Danger-Zone scaled dataset complete.")


def scale_fall_detection(max_train: int = 25, max_val: int = 10, max_test: int = 5):
    """Scale Fall Detection dataset with full pose keypoints."""
    proc_dir = BASE_DIR / "datasets" / "fall_detection" / "processed"
    ppe_proc = BASE_DIR / "datasets" / "ppe" / "processed"

    for split in ["train", "val", "test"]:
        (proc_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (proc_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    model = YOLO(str(BASE_DIR / "yolo11n-pose.pt"))
    import shutil

    split_targets = [("train", max_train), ("val", max_val), ("test", max_test)]
    for split, target in split_targets:
        src_imgs = list((ppe_proc / "images" / split).glob("*.jpg"))
        for img in src_imgs[:target]:
            dest_img = proc_dir / "images" / split / img.name
            if not dest_img.exists():
                shutil.copy2(img, dest_img)

            # Generate keypoint annotations
            results = model.predict(dest_img, conf=0.25, verbose=False)
            lines = []
            if results and len(results) > 0 and results[0].boxes is not None:
                r = results[0]
                for i, box in enumerate(r.boxes):
                    xywhn = box.xywhn[0].tolist()
                    kps = r.keypoints.xyn[i].tolist() if r.keypoints is not None else []
                    kp_str = " ".join(f"{pt[0]:.6f} {pt[1]:.6f} 2.0" for pt in kps)
                    lines.append(f"0 {xywhn[0]:.6f} {xywhn[1]:.6f} {xywhn[2]:.6f} {xywhn[3]:.6f} {kp_str}")

            lbl_dest = proc_dir / "labels" / split / (img.stem + ".txt")
            with open(lbl_dest, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")

    yaml_dict = {
        "path": str(proc_dir.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 1,
        "names": {0: "person"},
        "kpt_shape": [17, 3],
    }
    with open(proc_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.dump(yaml_dict, f, sort_keys=False)
    logger.info("Fall Detection scaled dataset complete.")


if __name__ == "__main__":
    scale_ppe(max_train=60, max_val=15, max_test=6)
    scale_fire_smoke(max_train=60, max_val=15, max_test=6)
    scale_danger_zone(max_train=60, max_val=15, max_test=6)
    scale_fall_detection(max_train=25, max_val=10, max_test=5)
