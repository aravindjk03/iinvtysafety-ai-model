"""Unit tests for Phase 3: Real PPE Detection Model integration and lifecycle."""

from pathlib import Path
import pytest

from ai_safety.core.exceptions import RegistryError
from ai_safety.core.model_metadata import ModelMetadata, ModelOrigin, ModelStatus, TaskType
from ai_safety.core.model_registry import ModelRegistry
from ai_safety.core.prediction import BoundingBox, Detection, PredictionResult
from ai_safety.inference.predictor import Predictor

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture
def registry():
    """Load the project ModelRegistry instance."""
    return ModelRegistry(registry_file="models/registry.json", repo_root=REPO_ROOT)


@pytest.fixture
def predictor(registry):
    """Create a Predictor bound to the registry."""
    return Predictor(registry=registry)


def test_01_ppe_model_registered(registry):
    """Test that ppe-workshop-v1 is registered in the registry."""
    assert registry.model_exists("ppe-workshop-v1")
    model = registry.get_model("ppe-workshop-v1")
    assert model.model_id == "ppe-workshop-v1"
    assert model.status == ModelStatus.READY
    assert model.origin == ModelOrigin.CUSTOM


def test_02_ppe_metadata_valid(registry):
    """Test that PPE metadata conforms to ModelMetadata contract."""
    model = registry.get_model("ppe-workshop-v1")
    assert model.task_type == TaskType.OBJECT_DETECTION
    assert model.framework == "ultralytics_yolo"
    assert model.confidence_threshold == 0.25
    assert model.input_size == (320, 320)
    assert model.base_model == "ppe-pretrained-base"
    assert model.training_run_id is not None


def test_03_ppe_class_schema(registry):
    """Test that PPE model classes strictly match the workshop schema."""
    model = registry.get_model("ppe-workshop-v1")
    expected_classes = ["person", "helmet", "vest"]
    assert model.classes == expected_classes


def test_04_ppe_pretrained_protection(registry):
    """Test that ppe-pretrained-base is protected from overwriting."""
    assert registry.model_exists("ppe-pretrained-base")
    base = registry.get_model("ppe-pretrained-base")
    assert base.origin == ModelOrigin.PRETRAINED

    duplicate_meta = ModelMetadata(
        model_id="ppe-pretrained-base",
        name="Attempted Overwrite",
        version="2.0.0",
        task_type=TaskType.OBJECT_DETECTION,
        framework="ultralytics_yolo",
        architecture="yolo11n",
        model_path="models/pretrained/ppe/model.pt",
        classes=["person"],
        input_size=(320, 320),
        confidence_threshold=0.25,
        origin=ModelOrigin.PRETRAINED,
    )
    with pytest.raises(RegistryError, match="Cannot overwrite protected pretrained model"):
        registry.register_model(duplicate_meta)


def test_05_ppe_lineage_exists():
    """Test that PPE lineage record exists and references base model and dataset."""
    lineage_file = REPO_ROOT / "models" / "custom" / "ppe-workshop-v1" / "lineage.json"
    assert lineage_file.exists()
    import json
    with open(lineage_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["base_model_id"] == "ppe-pretrained-base"
    assert "datasets/ppe" in data["dataset_path"]
    assert data["training_configuration"]["epochs"] == 5
    assert data["training_configuration"]["device"] == "cpu"


def test_06_ppe_metrics_exist():
    """Test that empirical evaluation metrics are recorded for PPE model."""
    metrics_file = REPO_ROOT / "models" / "custom" / "ppe-workshop-v1" / "metrics.json"
    assert metrics_file.exists()
    import json
    with open(metrics_file, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    assert "precision" in metrics
    assert "recall" in metrics
    assert "map50" in metrics
    assert "map50_95" in metrics
    assert metrics["map50"] > 0.0


def test_07_ppe_predictor_loading(predictor):
    """Test that existing Predictor loads the registered PPE model."""
    adapter = predictor.get_adapter("ppe-workshop-v1")
    assert adapter is not None
    assert adapter.metadata.model_id == "ppe-workshop-v1"


def test_08_ppe_prediction_schema(predictor):
    """Test that PPE inference emits normalized PredictionResult schema."""
    test_img = next((REPO_ROOT / "datasets" / "ppe" / "processed" / "images" / "test").glob("*.jpg"))
    result = predictor.predict("ppe-workshop-v1", str(test_img), confidence_threshold=0.001)

    assert isinstance(result, PredictionResult)
    assert result.model_id == "ppe-workshop-v1"
    assert result.task_type == TaskType.OBJECT_DETECTION
    assert result.processing_time_ms >= 0.0
    assert isinstance(result.detections, list)

    for det in result.detections:
        assert isinstance(det, Detection)
        assert det.class_name in ["person", "helmet", "vest"]
        assert 0.0 <= det.confidence <= 1.0
        assert isinstance(det.bounding_box, BoundingBox)
        assert det.bounding_box.x1 <= det.bounding_box.x2
        assert det.bounding_box.y1 <= det.bounding_box.y2


def test_09_ppe_confidence_threshold_filtering(predictor):
    """Test that higher confidence threshold filters out low-confidence detections."""
    test_img = next((REPO_ROOT / "datasets" / "ppe" / "processed" / "images" / "test").glob("*.jpg"))
    res_low = predictor.predict("ppe-workshop-v1", str(test_img), confidence_threshold=0.001)
    res_high = predictor.predict("ppe-workshop-v1", str(test_img), confidence_threshold=0.999)

    assert len(res_high.detections) <= len(res_low.detections)
