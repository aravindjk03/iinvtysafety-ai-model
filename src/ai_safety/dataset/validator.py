"""YOLO Object Detection Dataset Validator.

Performs comprehensive validation of directory layout, data.yaml configuration,
image extensions, label syntax, coordinate normalization, and class mappings.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import yaml

from pydantic import BaseModel, ConfigDict, Field


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ValidationErrorItem(BaseModel):
    """Structured dataset validation error."""

    model_config = ConfigDict(extra="forbid")

    code: str = Field(..., description="Stable error code")
    message: str = Field(..., description="Human-readable description of error")
    file: Optional[str] = Field(None, description="Relative path to offending file")
    line: Optional[int] = Field(None, description="Line number where error occurred")


class ValidationWarningItem(BaseModel):
    """Structured dataset validation warning."""

    model_config = ConfigDict(extra="forbid")

    code: str = Field(..., description="Stable warning code")
    message: str = Field(..., description="Human-readable description of warning")
    file: Optional[str] = Field(None, description="Relative path to offending file")


class DatasetStatistics(BaseModel):
    """Statistical summary of dataset contents."""

    model_config = ConfigDict(extra="forbid")

    train_images: int = 0
    validation_images: int = 0
    train_labels: int = 0
    validation_labels: int = 0
    classes: int = 0


class ValidationResult(BaseModel):
    """Result object returned by DatasetValidator."""

    model_config = ConfigDict(extra="forbid")

    valid: bool
    errors: List[ValidationErrorItem] = Field(default_factory=list)
    warnings: List[ValidationWarningItem] = Field(default_factory=list)
    statistics: DatasetStatistics = Field(default_factory=DatasetStatistics)


class DatasetValidator:
    """Validates YOLO object detection datasets against structural and syntax requirements."""

    def __init__(self, dataset_path: Path):
        self.dataset_path = Path(dataset_path)
        self.kpt_shape: Optional[List[int]] = None
        self.errors: List[ValidationErrorItem] = []
        self.warnings: List[ValidationWarningItem] = []
        self.stats = DatasetStatistics()

    def add_error(self, code: str, message: str, file: Optional[str] = None, line: Optional[int] = None) -> None:
        self.errors.append(ValidationErrorItem(code=code, message=message, file=file, line=line))

    def add_warning(self, code: str, message: str, file: Optional[str] = None) -> None:
        self.warnings.append(ValidationWarningItem(code=code, message=message, file=file))

    def validate(self) -> ValidationResult:
        """Execute full dataset validation."""
        self.errors.clear()
        self.warnings.clear()
        self.stats = DatasetStatistics()

        # 1. Path existence and directory checks
        if not self.dataset_path.exists():
            self.add_error("DATASET_NOT_FOUND", f"Dataset path does not exist: {self.dataset_path}")
            return self._build_result()

        if not self.dataset_path.is_dir():
            self.add_error("NOT_A_DIRECTORY", f"Dataset path is not a directory: {self.dataset_path}")
            return self._build_result()

        # 2. data.yaml validation
        data_yaml_path = self.dataset_path / "data.yaml"
        if not data_yaml_path.exists():
            self.add_error("MISSING_DATA_YAML", "data.yaml configuration file is missing from dataset root.", file="data.yaml")
            return self._build_result()

        yaml_data, class_names = self._parse_data_yaml(data_yaml_path)
        if yaml_data is None:
            return self._build_result()

        num_classes = len(class_names)
        self.stats.classes = num_classes

        # 3. Directory layout checks
        images_dir = self.dataset_path / "images"
        labels_dir = self.dataset_path / "labels"

        if not images_dir.exists() or not images_dir.is_dir():
            self.add_error("MISSING_IMAGES_DIR", "images/ directory is missing.", file="images")
        if not labels_dir.exists() or not labels_dir.is_dir():
            self.add_error("MISSING_LABELS_DIR", "labels/ directory is missing.", file="labels")

        if self.errors:
            return self._build_result()

        train_img_dir = images_dir / "train"
        val_img_dir = images_dir / "val"
        train_lbl_dir = labels_dir / "train"
        val_lbl_dir = labels_dir / "val"

        if not train_img_dir.exists() or not train_img_dir.is_dir():
            self.add_error("MISSING_TRAIN_SPLIT", "images/train/ directory is missing.", file="images/train")
        if not val_img_dir.exists() or not val_img_dir.is_dir():
            self.add_error("MISSING_VAL_SPLIT", "images/val/ directory is missing.", file="images/val")
        if not train_lbl_dir.exists() or not train_lbl_dir.is_dir():
            self.add_error("MISSING_TRAIN_LABELS_DIR", "labels/train/ directory is missing.", file="labels/train")
        if not val_lbl_dir.exists() or not val_lbl_dir.is_dir():
            self.add_error("MISSING_VAL_LABELS_DIR", "labels/val/ directory is missing.", file="labels/val")

        if self.errors:
            return self._build_result()

        # 4. Image and label validation per split
        train_imgs = self._scan_images(train_img_dir, "train")
        val_imgs = self._scan_images(val_img_dir, "val")

        self.stats.train_images = len(train_imgs)
        self.stats.validation_images = len(val_imgs)

        if self.stats.train_images == 0:
            self.add_error("EMPTY_TRAIN_SPLIT", "No image files found in images/train/.", file="images/train")
        if self.stats.validation_images == 0:
            self.add_error("EMPTY_VAL_SPLIT", "No image files found in images/val/.", file="images/val")

        if self.stats.train_images == 0 and self.stats.validation_images == 0:
            self.add_error("EMPTY_DATASET", "Dataset contains zero valid images.")
            return self._build_result()

        # Validate train labels
        train_lbl_count = self._validate_labels(train_imgs, train_lbl_dir, "train", num_classes)
        self.stats.train_labels = train_lbl_count

        # Validate val labels
        val_lbl_count = self._validate_labels(val_imgs, val_lbl_dir, "val", num_classes)
        self.stats.validation_labels = val_lbl_count

        return self._build_result()

    def _parse_data_yaml(self, path: Path) -> Tuple[Optional[Dict[str, Any]], List[str]]:
        """Parse and validate data.yaml contents."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except Exception as e:
            self.add_error("MALFORMED_DATA_YAML", f"data.yaml could not be parsed: {str(e)}", file="data.yaml")
            return None, []

        if not isinstance(data, dict):
            self.add_error("MALFORMED_DATA_YAML", "data.yaml root must be a YAML mapping (dictionary).", file="data.yaml")
            return None, []

        if "names" not in data:
            self.add_error("MISSING_CLASS_NAMES", "data.yaml is missing required 'names' field.", file="data.yaml")
            return None, []

        raw_names = data["names"]
        class_names: List[str] = []

        if isinstance(raw_names, list):
            class_names = [str(n) for n in raw_names]
        elif isinstance(raw_names, dict):
            try:
                sorted_keys = sorted(int(k) for k in raw_names.keys())
                class_names = [str(raw_names[k]) for k in sorted_keys]
            except Exception:
                class_names = [str(v) for v in raw_names.values()]
        else:
            self.add_error("INVALID_CLASS_NAMES", "'names' field in data.yaml must be a list or dict of class names.", file="data.yaml")
            return None, []

        if len(class_names) == 0:
            self.add_error("INVALID_CLASS_COUNT", "Class count must be at least 1 in data.yaml.", file="data.yaml")
            return None, []

        if "nc" in data:
            nc = data["nc"]
            if not isinstance(nc, int) or nc != len(class_names):
                self.add_error("CLASS_COUNT_MISMATCH", f"Class count nc ({nc}) does not match length of names ({len(class_names)}).", file="data.yaml")

        if "kpt_shape" in data:
            self.kpt_shape = data["kpt_shape"]

        return data, class_names

    def _scan_images(self, img_dir: Path, split: str) -> List[Path]:
        """Scan directory for images and flag unsupported extensions."""
        images: List[Path] = []
        for entry in img_dir.iterdir():
            if entry.is_file():
                ext = entry.suffix.lower()
                rel_path = f"images/{split}/{entry.name}"
                if ext in SUPPORTED_IMAGE_EXTENSIONS:
                    images.append(entry)
                else:
                    self.add_error(
                        "UNSUPPORTED_IMAGE_EXTENSION",
                        f"Unsupported image extension '{entry.suffix}'. Supported: {sorted(list(SUPPORTED_IMAGE_EXTENSIONS))}",
                        file=rel_path,
                    )
        return images

    def _validate_labels(
        self,
        images: List[Path],
        label_dir: Path,
        split: str,
        num_classes: int,
    ) -> int:
        """Validate annotation labels for images in split."""
        label_count = 0
        img_stems = {img.stem: img for img in images}

        # Check existing label files
        for lbl_file in label_dir.iterdir():
            if not lbl_file.is_file() or lbl_file.suffix != ".txt":
                continue

            rel_file = f"labels/{split}/{lbl_file.name}"
            label_count += 1

            if lbl_file.stem not in img_stems:
                self.add_warning("ORPHAN_LABEL_FILE", f"Label file exists with no matching image: {lbl_file.name}", file=rel_file)

            # Validate lines in label file
            try:
                content = lbl_file.read_text(encoding="utf-8").strip()
            except Exception as e:
                self.add_error("UNREADABLE_LABEL_FILE", f"Could not read label file: {str(e)}", file=rel_file)
                continue

            if not content:
                self.add_warning("EMPTY_LABEL_FILE", "Label file is empty (contains no annotations).", file=rel_file)
                continue

            for line_no, line in enumerate(content.splitlines(), start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                self._validate_label_line(stripped, rel_file, line_no, num_classes)

        # Check images missing label files
        for stem in img_stems:
            expected_lbl = label_dir / f"{stem}.txt"
            if not expected_lbl.exists():
                self.add_warning(
                    "MISSING_LABEL_FILE",
                    f"Image {img_stems[stem].name} has no matching label file in labels/{split}/",
                    file=f"images/{split}/{img_stems[stem].name}",
                )

        return label_count

    def _validate_label_line(self, line: str, file: str, line_no: int, num_classes: int) -> None:
        """Validate single YOLO line: class_id x_center y_center width height."""
        expected_len = 5
        if self.kpt_shape and isinstance(self.kpt_shape, (list, tuple)) and len(self.kpt_shape) == 2:
            expected_len = 5 + int(self.kpt_shape[0]) * int(self.kpt_shape[1])

        tokens = line.split()
        if len(tokens) != expected_len:
            self.add_error(
                "MALFORMED_BOUNDING_BOX",
                f"YOLO format expects exactly {expected_len} values for configured shape, found {len(tokens)}.",
                file=file,
                line=line_no,
            )
            return

        # 1. Class ID integer
        try:
            class_id = int(tokens[0])
        except ValueError:
            self.add_error(
                "INVALID_CLASS_ID_FORMAT",
                f"Class ID must be an integer, got '{tokens[0]}'.",
                file=file,
                line=line_no,
            )
            return

        if class_id < 0 or class_id >= num_classes:
            self.add_error(
                "INVALID_CLASS_ID",
                f"Class ID {class_id} is outside the configured class range [0, {num_classes - 1}].",
                file=file,
                line=line_no,
            )

        # 2. Coordinates numeric and normalized
        coords: List[float] = []
        for i, val in enumerate(tokens[1:], start=1):
            try:
                coords.append(float(val))
            except ValueError:
                self.add_error(
                    "NON_NUMERIC_COORDINATE",
                    f"Coordinate value #{i} is non-numeric: '{val}'.",
                    file=file,
                    line=line_no,
                )
                return

        x, y, w, h = coords[:4]

        if not (0.0 <= x <= 1.0):
            self.add_error(
                "COORDINATE_OUT_OF_RANGE",
                f"x_center ({x}) must be normalized between 0.0 and 1.0.",
                file=file,
                line=line_no,
            )
        if not (0.0 <= y <= 1.0):
            self.add_error(
                "COORDINATE_OUT_OF_RANGE",
                f"y_center ({y}) must be normalized between 0.0 and 1.0.",
                file=file,
                line=line_no,
            )
        if not (0.0 < w <= 1.0):
            self.add_error(
                "COORDINATE_OUT_OF_RANGE",
                f"width ({w}) must be greater than 0.0 and <= 1.0.",
                file=file,
                line=line_no,
            )
        if not (0.0 < h <= 1.0):
            self.add_error(
                "COORDINATE_OUT_OF_RANGE",
                f"height ({h}) must be greater than 0.0 and <= 1.0.",
                file=file,
                line=line_no,
            )

    def _build_result(self) -> ValidationResult:
        return ValidationResult(
            valid=len(self.errors) == 0,
            errors=self.errors,
            warnings=self.warnings,
            statistics=self.stats,
        )
