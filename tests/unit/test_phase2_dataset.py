"""Unit tests for Phase 2 YOLO Dataset Validator and Analyzer."""

from pathlib import Path
import pytest
import yaml

from ai_safety.dataset.analyzer import DatasetAnalyzer
from ai_safety.dataset.validator import DatasetValidator


@pytest.fixture
def base_dataset(tmp_path: Path) -> Path:
    """Create a minimal valid YOLO dataset directory."""
    ds = tmp_path / "dataset"
    (ds / "images" / "train").mkdir(parents=True, exist_ok=True)
    (ds / "images" / "val").mkdir(parents=True, exist_ok=True)
    (ds / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (ds / "labels" / "val").mkdir(parents=True, exist_ok=True)

    # Valid images
    (ds / "images" / "train" / "img1.jpg").write_bytes(b"\xFF\xD8\xFF\xE0dummy_jpg")
    (ds / "images" / "val" / "img2.jpg").write_bytes(b"\xFF\xD8\xFF\xE0dummy_jpg")

    # Valid labels (class_id x y w h)
    (ds / "labels" / "train" / "img1.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (ds / "labels" / "val" / "img2.txt").write_text("1 0.3 0.3 0.4 0.4\n", encoding="utf-8")

    # Valid data.yaml
    cfg = {
        "names": {0: "helmet", 1: "vest"},
        "nc": 2,
        "train": "images/train",
        "val": "images/val",
    }
    with open(ds / "data.yaml", "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)

    return ds


def test_valid_yolo_dataset(base_dataset: Path):
    """Verify validator succeeds on valid YOLO dataset."""
    validator = DatasetValidator(base_dataset)
    res = validator.validate()
    assert res.valid is True
    assert len(res.errors) == 0
    assert res.statistics.train_images == 1
    assert res.statistics.validation_images == 1
    assert res.statistics.classes == 2


def test_missing_dataset_path(tmp_path: Path):
    """Verify error on non-existent dataset path."""
    non_existent = tmp_path / "does_not_exist"
    validator = DatasetValidator(non_existent)
    res = validator.validate()
    assert res.valid is False
    assert any(e.code == "DATASET_NOT_FOUND" for e in res.errors)


def test_missing_images_dir(base_dataset: Path):
    """Verify error when images directory is missing."""
    import shutil
    shutil.rmtree(base_dataset / "images")
    validator = DatasetValidator(base_dataset)
    res = validator.validate()
    assert res.valid is False
    assert any(e.code == "MISSING_IMAGES_DIR" for e in res.errors)


def test_missing_labels_dir(base_dataset: Path):
    """Verify error when labels directory is missing."""
    import shutil
    shutil.rmtree(base_dataset / "labels")
    validator = DatasetValidator(base_dataset)
    res = validator.validate()
    assert res.valid is False
    assert any(e.code == "MISSING_LABELS_DIR" for e in res.errors)


def test_missing_data_yaml(base_dataset: Path):
    """Verify error when data.yaml is missing."""
    (base_dataset / "data.yaml").unlink()
    validator = DatasetValidator(base_dataset)
    res = validator.validate()
    assert res.valid is False
    assert any(e.code == "MISSING_DATA_YAML" for e in res.errors)


def test_malformed_data_yaml(base_dataset: Path):
    """Verify error when data.yaml contains invalid syntax."""
    (base_dataset / "data.yaml").write_text("names: [unclosed_bracket", encoding="utf-8")
    validator = DatasetValidator(base_dataset)
    res = validator.validate()
    assert res.valid is False
    assert any(e.code == "MALFORMED_DATA_YAML" for e in res.errors)


def test_invalid_class_id(base_dataset: Path):
    """Verify error when label references class ID outside configured range."""
    # Config has classes 0 and 1; reference class 5
    (base_dataset / "labels" / "train" / "img1.txt").write_text("5 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    validator = DatasetValidator(base_dataset)
    res = validator.validate()
    assert res.valid is False
    assert any(e.code == "INVALID_CLASS_ID" for e in res.errors)


def test_malformed_bounding_box(base_dataset: Path):
    """Verify error when bounding box does not have exactly 5 values."""
    # Only 4 tokens
    (base_dataset / "labels" / "train" / "img1.txt").write_text("0 0.5 0.5 0.2\n", encoding="utf-8")
    validator = DatasetValidator(base_dataset)
    res = validator.validate()
    assert res.valid is False
    assert any(e.code == "MALFORMED_BOUNDING_BOX" for e in res.errors)


def test_out_of_range_coordinates(base_dataset: Path):
    """Verify error when coordinates exceed [0.0, 1.0] range."""
    (base_dataset / "labels" / "train" / "img1.txt").write_text("0 1.5 0.5 0.2 0.2\n", encoding="utf-8")
    validator = DatasetValidator(base_dataset)
    res = validator.validate()
    assert res.valid is False
    assert any(e.code == "COORDINATE_OUT_OF_RANGE" for e in res.errors)


def test_empty_dataset(tmp_path: Path):
    """Verify error when dataset has directories but 0 images."""
    ds = tmp_path / "empty_ds"
    (ds / "images" / "train").mkdir(parents=True, exist_ok=True)
    (ds / "images" / "val").mkdir(parents=True, exist_ok=True)
    (ds / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (ds / "labels" / "val").mkdir(parents=True, exist_ok=True)
    with open(ds / "data.yaml", "w", encoding="utf-8") as f:
        yaml.dump({"names": ["a", "b"]}, f)

    validator = DatasetValidator(ds)
    res = validator.validate()
    assert res.valid is False
    assert any(e.code in ["EMPTY_DATASET", "EMPTY_TRAIN_SPLIT"] for e in res.errors)


def test_unsupported_image_extension(base_dataset: Path):
    """Verify error when image uses unsupported file extension."""
    (base_dataset / "images" / "train" / "bad.gif").write_bytes(b"GIF89a")
    validator = DatasetValidator(base_dataset)
    res = validator.validate()
    assert res.valid is False
    assert any(e.code == "UNSUPPORTED_IMAGE_EXTENSION" for e in res.errors)


def test_missing_label_warning(base_dataset: Path):
    """Verify warning when image exists without matching label file."""
    (base_dataset / "images" / "train" / "unlabeled.jpg").write_bytes(b"\xFF\xD8dummy")
    validator = DatasetValidator(base_dataset)
    res = validator.validate()
    assert res.valid is True  # Valid with warning
    assert any(w.code == "MISSING_LABEL_FILE" for w in res.warnings)


def test_class_imbalance_warning(base_dataset: Path):
    """Verify analyzer warns when class distribution is highly imbalanced."""
    # 20 annotations for class 0, 1 annotation for class 1 (ratio 20:1 >= 10:1)
    lines = ["0 0.5 0.5 0.1 0.1" for _ in range(20)]
    (base_dataset / "labels" / "train" / "img1.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    analyzer = DatasetAnalyzer(base_dataset)
    summary = analyzer.analyze()
    assert any("imbalanced" in w.lower() for w in summary.warnings)


def test_dataset_summary_generation(base_dataset: Path):
    """Verify dataset analyzer extracts ground-truth metrics correctly."""
    analyzer = DatasetAnalyzer(base_dataset)
    summary = analyzer.analyze()
    assert summary.total_images == 2
    assert summary.train_images == 1
    assert summary.validation_images == 1
    assert summary.total_annotations == 2
    assert summary.annotations_per_class["helmet"] == 1
    assert summary.annotations_per_class["vest"] == 1
    assert ".jpg" in summary.image_formats
    text_summary = DatasetAnalyzer.format_text_summary(summary)
    assert "helmet: 1" in text_summary
