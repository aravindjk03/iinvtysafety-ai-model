"""Unit tests for Phase 4: Real Fire & Smoke Detection Model integration and verification.

Covers all 15 mandatory verification tests specified in Phase 4 specification.
"""

import json
from pathlib import Path
import pytest
import yaml

from ai_safety.core.exceptions import ModelLoadError, RegistryError
from ai_safety.core.model_metadata import ModelMetadata, ModelOrigin, ModelStatus, TaskType
from ai_safety.core.model_registry import ModelRegistry
from ai_safety.core.prediction import BoundingBox, Detection, PredictionResult
from ai_safety.dataset.analyzer import DatasetAnalyzer
from ai_safety.dataset.validator import DatasetValidator
from ai_safety.inference.predictor import Predictor
from ai_safety.training.config import TrainingConfig
from ai_safety.training.run_manager import TrainingRunManager

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture
def registry():
    return ModelRegistry(registry_file="models/registry.json", repo_root=REPO_ROOT)


@pytest.fixture
def predictor(registry):
    return Predictor(registry=registry)


def test_01_fire_smoke_dataset_metadata_valid():
    """Test 1: Fire/Smoke dataset metadata in data.yaml is valid."""
    yaml_path = REPO_ROOT / "datasets" / "fire_smoke" / "processed" / "data.yaml"
    assert yaml_path.exists()
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert "names" in data
    assert data["nc"] == 2
    assert "fire" in str(data["names"])
    assert "smoke" in str(data["names"])


def test_02_dataset_validation_rejects_invalid_class_ids(tmp_path):
    """Test 2: Dataset validation rejects invalid class IDs."""
    ds_dir = tmp_path / "bad_dataset"
    (ds_dir / "images" / "train").mkdir(parents=True)
    (ds_dir / "images" / "val").mkdir(parents=True)
    (ds_dir / "labels" / "train").mkdir(parents=True)
    (ds_dir / "labels" / "val").mkdir(parents=True)

    with open(ds_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.dump({"names": ["fire", "smoke"], "nc": 2}, f)

    (ds_dir / "images" / "train" / "img1.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 20)
    (ds_dir / "images" / "val" / "img2.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 20)

    # Class ID 99 is invalid for nc=2
    (ds_dir / "labels" / "train" / "img1.txt").write_text("99 0.5 0.5 0.2 0.2\n")
    (ds_dir / "labels" / "val" / "img2.txt").write_text("0 0.5 0.5 0.2 0.2\n")

    val = DatasetValidator(ds_dir)
    res = val.validate()
    assert not res.valid
    assert any(e.code == "INVALID_CLASS_ID" for e in res.errors)


def test_03_dataset_validation_rejects_invalid_bounding_boxes(tmp_path):
    """Test 3: Dataset validation rejects out-of-range coordinates."""
    ds_dir = tmp_path / "bad_bbox"
    (ds_dir / "images" / "train").mkdir(parents=True)
    (ds_dir / "images" / "val").mkdir(parents=True)
    (ds_dir / "labels" / "train").mkdir(parents=True)
    (ds_dir / "labels" / "val").mkdir(parents=True)

    with open(ds_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.dump({"names": ["fire", "smoke"], "nc": 2}, f)

    (ds_dir / "images" / "train" / "img1.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 20)
    (ds_dir / "images" / "val" / "img2.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 20)

    # Coordinate 1.5 > 1.0 (out of normalized range)
    (ds_dir / "labels" / "train" / "img1.txt").write_text("0 1.5 0.5 0.2 0.2\n")
    (ds_dir / "labels" / "val" / "img2.txt").write_text("0 0.5 0.5 0.2 0.2\n")

    val = DatasetValidator(ds_dir)
    res = val.validate()
    assert not res.valid
    assert any(e.code == "COORDINATE_OUT_OF_RANGE" for e in res.errors)


def test_04_dataset_summary_correctly_counts_annotations():
    """Test 4: Dataset summary correctly counts Fire/Smoke annotations."""
    ds_path = REPO_ROOT / "datasets" / "fire_smoke" / "processed"
    analyzer = DatasetAnalyzer(ds_path)
    summary = analyzer.analyze()

    assert summary.total_images >= 25
    assert summary.train_images >= 20
    assert summary.validation_images >= 5
    assert summary.total_annotations >= 25
    assert "fire" in summary.annotations_per_class
    assert "smoke" in summary.annotations_per_class
    assert summary.annotations_per_class["fire"] >= 20
    assert summary.annotations_per_class["smoke"] >= 4


def test_05_training_configuration_accepts_fire_smoke():
    """Test 5: Training configuration accepts Fire/Smoke dataset."""
    cfg = TrainingConfig(
        base_model="fire-smoke-pretrained-base",
        dataset_path="datasets/fire_smoke/processed",
        epochs=5,
        image_size=320,
        batch_size=4,
        device="cpu",
    )
    assert cfg.epochs == 5
    assert cfg.device == "cpu"
    assert cfg.image_size == 320


def test_06_training_run_creates_unique_artifacts():
    """Test 6: Training run creates unique artifacts in training_runs/."""
    run_dirs = list((REPO_ROOT / "training_runs").glob("run-*"))
    assert len(run_dirs) >= 2  # At least PPE and Fire/Smoke runs

    # Find fire smoke run by checking lineage.json
    fs_runs = []
    for rdir in run_dirs:
        lineage_file = rdir / "lineage.json"
        if lineage_file.exists():
            with open(lineage_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("base_model_id") == "fire-smoke-pretrained-base":
                fs_runs.append(rdir)

    assert len(fs_runs) >= 1
    fs_run = fs_runs[0]
    assert (fs_run / "configuration.json").exists()
    assert (fs_run / "dataset_summary.json").exists()
    assert (fs_run / "metrics.json").exists()
    assert (fs_run / "weights" / "best.pt").exists()


def test_07_lineage_records_fire_smoke_info():
    """Test 7: Lineage records Fire/Smoke training information."""
    lineage_file = REPO_ROOT / "models" / "custom" / "fire-smoke-workshop-v1" / "lineage.json"
    assert lineage_file.exists()
    with open(lineage_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["base_model_id"] == "fire-smoke-pretrained-base"
    assert "datasets/fire_smoke" in data["dataset_path"]
    assert data["training_configuration"]["epochs"] == 5
    assert data["framework"] == "ultralytics_yolo"


def test_08_fire_smoke_model_registers_successfully(registry):
    """Test 8: Fire/Smoke model registers successfully."""
    assert registry.model_exists("fire-smoke-workshop-v1")
    model = registry.get_model("fire-smoke-workshop-v1")
    assert model.model_id == "fire-smoke-workshop-v1"
    assert model.status == ModelStatus.READY
    assert model.classes == ["fire", "smoke"]


def test_09_duplicate_registration_handled_correctly(registry):
    """Test 9: Duplicate registration is handled correctly."""
    meta = registry.get_model("fire-smoke-workshop-v1")
    duplicate_meta = meta.model_copy(update={"version": "1.0.0"})
    with pytest.raises(RegistryError):
        registry.register_model(duplicate_meta)


def test_10_predictor_loads_fire_smoke(predictor):
    """Test 10: Predictor loads registered Fire/Smoke model."""
    adapter = predictor.get_adapter("fire-smoke-workshop-v1")
    assert adapter is not None
    assert adapter.metadata.model_id == "fire-smoke-workshop-v1"


def test_11_prediction_schema_valid(predictor):
    """Test 11: Prediction schema is valid."""
    test_img = next((REPO_ROOT / "datasets" / "fire_smoke" / "processed" / "images" / "test").glob("*.jpg"))
    res = predictor.predict("fire-smoke-workshop-v1", str(test_img), confidence_threshold=0.001)

    assert isinstance(res, PredictionResult)
    assert res.model_id == "fire-smoke-workshop-v1"
    assert res.task_type == TaskType.OBJECT_DETECTION
    assert res.processing_time_ms >= 0.0


def test_12_invalid_model_path_fails_safely(tmp_path):
    """Test 12: Invalid model path fails safely with ModelLoadError."""
    temp_reg_file = tmp_path / "temp_registry.json"
    temp_reg = ModelRegistry(registry_file=temp_reg_file, repo_root=REPO_ROOT)
    bad_meta = ModelMetadata(
        model_id="nonexistent-model",
        name="Ghost Model",
        version="1.0.0",
        task_type=TaskType.OBJECT_DETECTION,
        framework="ultralytics_yolo",
        architecture="yolo11n",
        model_path="models/custom/nonexistent/model.pt",
        classes=["fire"],
        input_size=(320, 320),
        confidence_threshold=0.25,
        origin=ModelOrigin.CUSTOM,
    )
    temp_reg.register_model(bad_meta)
    predictor = Predictor(registry=temp_reg)
    with pytest.raises(ModelLoadError):
        predictor.predict("nonexistent-model", "some_img.jpg")


def test_13_confidence_values_within_valid_range(predictor):
    """Test 13: Confidence values are within valid range [0.0, 1.0]."""
    test_img = next((REPO_ROOT / "datasets" / "fire_smoke" / "processed" / "images" / "test").glob("*.jpg"))
    res = predictor.predict("fire-smoke-workshop-v1", str(test_img), confidence_threshold=0.001)
    for det in res.detections:
        assert 0.0 <= det.confidence <= 1.0


def test_14_bounding_boxes_are_valid(predictor):
    """Test 14: Bounding boxes are geometrically valid."""
    test_img = next((REPO_ROOT / "datasets" / "fire_smoke" / "processed" / "images" / "test").glob("*.jpg"))
    res = predictor.predict("fire-smoke-workshop-v1", str(test_img), confidence_threshold=0.001)
    for det in res.detections:
        assert isinstance(det.bounding_box, BoundingBox)
        assert det.bounding_box.x1 <= det.bounding_box.x2
        assert det.bounding_box.y1 <= det.bounding_box.y2
        assert det.bounding_box.width >= 0
        assert det.bounding_box.height >= 0


def test_15_existing_ppe_functionality_still_works(predictor, registry):
    """Test 15: Existing PPE model functionality still works (no regression)."""
    assert registry.model_exists("ppe-workshop-v1")
    ppe_img = next((REPO_ROOT / "datasets" / "ppe" / "processed" / "images" / "test").glob("*.jpg"))
    res = predictor.predict("ppe-workshop-v1", str(ppe_img), confidence_threshold=0.001)
    assert res.model_id == "ppe-workshop-v1"
    assert res.task_type == TaskType.OBJECT_DETECTION
