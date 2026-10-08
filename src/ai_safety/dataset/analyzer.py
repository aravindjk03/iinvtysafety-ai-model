"""Dataset Analyzer and Summary Generator for YOLO datasets.

Extracts ground-truth statistics including class counts, image formats,
annotation distribution, and flags class imbalances or zero-annotation classes.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

from pydantic import BaseModel, ConfigDict, Field

from ai_safety.dataset.validator import SUPPORTED_IMAGE_EXTENSIONS


class DatasetSummary(BaseModel):
    """Ground-truth quantitative analysis of a validated dataset."""

    model_config = ConfigDict(extra="forbid")

    total_images: int
    train_images: int
    validation_images: int
    total_annotations: int
    annotations_per_class: Dict[str, int]
    class_names: Dict[int, str]
    missing_labels_count: int
    invalid_labels_count: int
    image_formats: Dict[str, int]
    warnings: List[str] = Field(default_factory=list)


class DatasetAnalyzer:
    """Analyzes a YOLO dataset directory and produces a verifiable numerical summary."""

    def __init__(self, dataset_path: Path):
        self.dataset_path = Path(dataset_path)

    def analyze(self) -> DatasetSummary:
        """Parse dataset files and compute actual statistical metrics."""
        data_yaml = self.dataset_path / "data.yaml"
        if not data_yaml.exists():
            raise FileNotFoundError(f"data.yaml not found at {data_yaml}")

        with open(data_yaml, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        raw_names = cfg.get("names", [])
        class_map: Dict[int, str] = {}
        if isinstance(raw_names, list):
            class_map = {i: str(name) for i, name in enumerate(raw_names)}
        elif isinstance(raw_names, dict):
            class_map = {int(k): str(v) for k, v in raw_names.items()}

        train_imgs = self._get_image_files(self.dataset_path / "images" / "train")
        val_imgs = self._get_image_files(self.dataset_path / "images" / "val")
        all_imgs = train_imgs + val_imgs

        format_counts: Dict[str, int] = {}
        for img in all_imgs:
            ext = img.suffix.lower()
            format_counts[ext] = format_counts.get(ext, 0) + 1

        # Annotations tracking
        class_counts: Dict[str, int] = {name: 0 for name in class_map.values()}
        missing_labels = 0
        invalid_labels = 0
        total_annotations = 0

        # Scan annotations across train and val splits
        for split, img_list in [("train", train_imgs), ("val", val_imgs)]:
            lbl_dir = self.dataset_path / "labels" / split
            for img in img_list:
                lbl_file = lbl_dir / f"{img.stem}.txt"
                if not lbl_file.exists():
                    missing_labels += 1
                    continue

                try:
                    lines = lbl_file.read_text(encoding="utf-8").splitlines()
                except Exception:
                    invalid_labels += 1
                    continue

                for line in lines:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    tokens = stripped.split()
                    if len(tokens) < 5:
                        invalid_labels += 1
                        continue

                    try:
                        cid = int(tokens[0])
                        if cid in class_map:
                            cname = class_map[cid]
                            class_counts[cname] += 1
                            total_annotations += 1
                        else:
                            invalid_labels += 1
                    except ValueError:
                        invalid_labels += 1

        # Warnings evaluation
        warnings: List[str] = []
        if len(train_imgs) == 0:
            warnings.append("Training split is missing or contains 0 images.")
        if len(val_imgs) == 0:
            warnings.append("Validation split is missing or contains 0 images.")

        zero_anno_classes = [cname for cname, count in class_counts.items() if count == 0]
        for cname in zero_anno_classes:
            warnings.append(f"Class '{cname}' has zero annotations across the dataset.")

        if len(class_map) <= 1:
            warnings.append(f"Dataset has an extremely small class count ({len(class_map)} class).")

        # Class imbalance warning (> 10x ratio between max and non-zero min)
        non_zero_counts = [count for count in class_counts.values() if count > 0]
        if non_zero_counts:
            max_c = max(non_zero_counts)
            min_c = min(non_zero_counts)
            if min_c > 0 and (max_c / min_c) >= 10.0:
                warnings.append(f"Highly imbalanced class distribution detected (ratio {max_c}:{min_c} >= 10:1).")

        return DatasetSummary(
            total_images=len(all_imgs),
            train_images=len(train_imgs),
            validation_images=len(val_imgs),
            total_annotations=total_annotations,
            annotations_per_class=class_counts,
            class_names=class_map,
            missing_labels_count=missing_labels,
            invalid_labels_count=invalid_labels,
            image_formats=format_counts,
            warnings=warnings,
        )

    def _get_image_files(self, directory: Path) -> List[Path]:
        if not directory.exists() or not directory.is_dir():
            return []
        return [
            p for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]

    @staticmethod
    def format_text_summary(summary: DatasetSummary) -> str:
        """Produce a human-readable text representation of the dataset summary."""
        lines = [
            "Dataset Summary",
            "",
            "Images:",
            f"  Total:      {summary.total_images}",
            f"  Train:      {summary.train_images}",
            f"  Validation: {summary.validation_images}",
            "",
            "Classes:",
        ]
        for cid, cname in summary.class_names.items():
            lines.append(f"  {cid} = {cname}")

        lines.extend(["", "Annotations:"])
        for cname, count in summary.annotations_per_class.items():
            lines.append(f"  {cname}: {count}")

        lines.extend([
            "",
            f"Total Annotations:    {summary.total_annotations}",
            f"Missing Label Files:  {summary.missing_labels_count}",
            f"Invalid Annotations:  {summary.invalid_labels_count}",
        ])

        if summary.warnings:
            lines.extend(["", "Warnings:"])
            for w in summary.warnings:
                lines.append(f"  - [WARNING] {w}")

        return "\n".join(lines)
