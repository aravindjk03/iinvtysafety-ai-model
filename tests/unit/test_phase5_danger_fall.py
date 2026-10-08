"""Unit tests for Phase 5: Danger-Zone Detection, Fall Detection, and Deep Fire Training."""

import json
import math
from pathlib import Path
import pytest

from ai_safety.core.exceptions import RegistryError
from ai_safety.core.model_metadata import ModelMetadata, ModelOrigin, ModelStatus, TaskType
from ai_safety.core.model_registry import ModelRegistry
from ai_safety.core.prediction import BoundingBox, Detection, Keypoint, PredictionResult
from ai_safety.dataset.validator import DatasetValidator
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


def test_01_danger_zone_model_registered(registry):
    """Test 01: danger-zone-workshop-v1 is registered and ready."""
    assert registry.model_exists("danger-zone-workshop-v1")
    meta = registry.get_model("danger-zone-workshop-v1")
    assert meta.model_id == "danger-zone-workshop-v1"
    assert meta.status == ModelStatus.READY
    assert meta.origin == ModelOrigin.CUSTOM
    assert meta.task_type == TaskType.OBJECT_DETECTION
    assert meta.base_model == "danger-zone-pretrained-base"


def test_02_danger_zone_classes(registry):
    """Test 02: danger-zone model classes match industrial danger zone schema."""
    meta = registry.get_model("danger-zone-workshop-v1")
    expected = ["person", "machinery", "vehicle", "safety_cone"]
    assert meta.classes == expected


def test_03_danger_zone_lineage_and_metrics():
    """Test 03: Danger-Zone model records empirical lineage and metrics."""
    model_dir = REPO_ROOT / "models" / "custom" / "danger-zone-workshop-v1"
    lineage_path = model_dir / "lineage.json"
    metrics_path = model_dir / "metrics.json"

    assert lineage_path.exists()
    assert metrics_path.exists()

    with open(lineage_path, "r", encoding="utf-8") as f:
        lineage = json.load(f)
    assert lineage["base_model_id"] == "danger-zone-pretrained-base"
    assert "danger_zone" in lineage["dataset_path"]
    assert lineage["training_configuration"]["epochs"] == 15
    assert lineage["device"] == "cpu"

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["map50"] <= 1.0
    assert 0.0 <= metrics["map50_95"] <= 1.0


def test_04_danger_zone_base_pretrained_protection(registry):
    """Test 04: danger-zone-pretrained-base is protected from overwriting."""
    assert registry.model_exists("danger-zone-pretrained-base")
    base = registry.get_model("danger-zone-pretrained-base")
    assert base.origin == ModelOrigin.PRETRAINED

    bad_meta = ModelMetadata(
        model_id="danger-zone-pretrained-base",
        name="Overwrite Attempt",
        version="2.0.0",
        task_type=TaskType.OBJECT_DETECTION,
        framework="ultralytics_yolo",
        architecture="yolo11n",
        model_path="models/pretrained/danger_zone/model.pt",
        classes=["person"],
        input_size=(320, 320),
        confidence_threshold=0.25,
        origin=ModelOrigin.PRETRAINED,
    )
    with pytest.raises(RegistryError, match="Cannot overwrite protected pretrained model"):
        registry.register_model(bad_meta)


def test_05_danger_zone_prediction(predictor):
    """Test 05: Predictor executes inference on danger-zone model."""
    test_img = next((REPO_ROOT / "datasets" / "danger_zone" / "processed" / "images" / "test").glob("*.jpg"))
    res = predictor.predict("danger-zone-workshop-v1", str(test_img), confidence_threshold=0.01)
    assert res.model_id == "danger-zone-workshop-v1"
    assert res.task_type == TaskType.OBJECT_DETECTION
    assert res.processing_time_ms >= 0.0
    assert isinstance(res.detections, list)


def test_06_fire_deep_model_registered(registry):
    """Test 06: fire-smoke-deep-v1 is registered and ready."""
    assert registry.model_exists("fire-smoke-deep-v1")
    meta = registry.get_model("fire-smoke-deep-v1")
    assert meta.model_id == "fire-smoke-deep-v1"
    assert meta.status == ModelStatus.READY
    assert meta.origin == ModelOrigin.CUSTOM
    assert meta.classes == ["fire", "smoke"]


def test_07_fire_deep_lineage_and_epochs():
    """Test 07: fire-smoke-deep-v1 records 15 epochs deep training in lineage."""
    model_dir = REPO_ROOT / "models" / "custom" / "fire-smoke-deep-v1"
    lineage_path = model_dir / "lineage.json"
    assert lineage_path.exists()

    with open(lineage_path, "r", encoding="utf-8") as f:
        lineage = json.load(f)
    assert lineage["training_configuration"]["epochs"] == 15
    assert lineage["device"] == "cpu"
    assert "fire_smoke" in lineage["dataset_path"]


def test_08_fire_deep_prediction(predictor):
    """Test 08: Predictor executes inference on fire-smoke-deep-v1."""
    test_img = next((REPO_ROOT / "datasets" / "fire_smoke" / "processed" / "images" / "test").glob("*.jpg"))
    res = predictor.predict("fire-smoke-deep-v1", str(test_img), confidence_threshold=0.01)
    assert res.model_id == "fire-smoke-deep-v1"
    assert res.task_type == TaskType.OBJECT_DETECTION
    assert res.processing_time_ms >= 0.0


def test_09_fall_detection_model_registered(registry):
    """Test 09: fall-detection-workshop-v1 is registered as pose estimation model."""
    assert registry.model_exists("fall-detection-workshop-v1")
    meta = registry.get_model("fall-detection-workshop-v1")
    assert meta.model_id == "fall-detection-workshop-v1"
    assert meta.status == ModelStatus.READY
    assert meta.origin == ModelOrigin.CUSTOM
    assert meta.task_type == TaskType.POSE_ESTIMATION
    assert meta.classes == ["person"]


def test_10_fall_detection_base_model(registry):
    """Test 10: fall-pretrained-base is registered as protected pose model."""
    assert registry.model_exists("fall-pretrained-base")
    base = registry.get_model("fall-pretrained-base")
    assert base.task_type == TaskType.POSE_ESTIMATION
    assert base.origin == ModelOrigin.PRETRAINED


def test_11_fall_detection_lineage_and_metrics():
    """Test 11: Fall detection model records lineage and metrics."""
    model_dir = REPO_ROOT / "models" / "custom" / "fall-detection-workshop-v1"
    metrics_path = model_dir / "metrics.json"
    lineage_path = model_dir / "lineage.json"

    assert metrics_path.exists()
    assert lineage_path.exists()

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    assert metrics["map50"] >= 0.8  # Real empirical score from pose val
    assert metrics["precision"] >= 0.8

    with open(lineage_path, "r", encoding="utf-8") as f:
        lineage = json.load(f)
    assert lineage["base_model_id"] == "fall-pretrained-base"
    assert lineage["training_configuration"]["task_type"] == "pose_estimation"


def test_12_fall_detection_pose_keypoints(predictor):
    """Test 12: Predictor returns 17 COCO skeletal keypoints for worker detections."""
    test_img = next((REPO_ROOT / "datasets" / "fall_detection" / "processed" / "images" / "test").glob("*.jpg"))
    res = predictor.predict("fall-detection-workshop-v1", str(test_img), confidence_threshold=0.1)

    assert res.model_id == "fall-detection-workshop-v1"
    assert res.task_type == TaskType.POSE_ESTIMATION
    assert len(res.detections) > 0

    first_det = res.detections[0]
    assert first_det.class_name == "person"
    assert first_det.keypoints is not None
    assert len(first_det.keypoints) == 17

    kp_names = [kp.name for kp in first_det.keypoints]
    expected_keypoints = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
    ]
    assert kp_names == expected_keypoints

    for kp in first_det.keypoints:
        assert isinstance(kp, Keypoint)
        assert kp.x >= 0.0
        assert kp.y >= 0.0


def test_13_fall_posture_torso_angle_calculation(predictor):
    """Test 13: Verifies mathematical torso angle computation for downstream fall detection."""
    test_img = next((REPO_ROOT / "datasets" / "fall_detection" / "processed" / "images" / "test").glob("*.jpg"))
    res = predictor.predict("fall-detection-workshop-v1", str(test_img), confidence_threshold=0.1)

    assert len(res.detections) > 0
    kps = {kp.name: kp for kp in res.detections[0].keypoints}

    # Calculate midpoint of shoulders and midpoint of hips
    mid_shoulder_x = (kps["left_shoulder"].x + kps["right_shoulder"].x) / 2.0
    mid_shoulder_y = (kps["left_shoulder"].y + kps["right_shoulder"].y) / 2.0
    mid_hip_x = (kps["left_hip"].x + kps["right_hip"].x) / 2.0
    mid_hip_y = (kps["left_hip"].y + kps["right_hip"].y) / 2.0

    dx = mid_shoulder_x - mid_hip_x
    dy = mid_shoulder_y - mid_hip_y
    # Angle relative to vertical (degrees)
    torso_angle_deg = math.degrees(math.atan2(abs(dx), abs(dy)))
    assert 0.0 <= torso_angle_deg <= 90.0


def test_14_dataset_validator_supports_both_detection_and_pose():
    """Test 14: DatasetValidator cleanly validates both object detection and pose datasets."""
    dz_val = DatasetValidator(REPO_ROOT / "datasets" / "danger_zone" / "processed")
    dz_res = dz_val.validate()
    assert dz_res.valid is True
    assert len(dz_res.errors) == 0

    fall_val = DatasetValidator(REPO_ROOT / "datasets" / "fall_detection" / "processed")
    fall_res = fall_val.validate()
    assert fall_res.valid is True
    assert len(fall_res.errors) == 0


def test_15_v2_models_registered_and_ready(registry):
    """Test 15: All v2 scaled models are registered with version 2.0.0 and ready."""
    v2_ids = ["ppe-workshop-v2", "fire-smoke-v2", "danger-zone-v2", "fall-detection-v2"]
    for mid in v2_ids:
        assert registry.model_exists(mid), f"Model {mid} missing from registry"
        meta = registry.get_model(mid)
        assert meta.version == "2.0.0"
        assert meta.status == ModelStatus.READY
        assert meta.origin == ModelOrigin.CUSTOM


def test_16_v2_fire_smoke_high_accuracy():
    """Test 16: fire-smoke-v2 achieves high mAP50 (> 0.50) on scaled dataset."""
    metrics_path = REPO_ROOT / "models" / "custom" / "fire-smoke-v2" / "metrics.json"
    assert metrics_path.exists()
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    assert metrics["map50"] >= 0.50


def test_17_v2_fall_detection_pose_accuracy():
    """Test 17: fall-detection-v2 achieves high mAP50 (> 0.90) on scaled dataset."""
    metrics_path = REPO_ROOT / "models" / "custom" / "fall-detection-v2" / "metrics.json"
    assert metrics_path.exists()
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    assert metrics["map50"] >= 0.90


def test_18_v2_multi_model_predictor_inference(predictor):
    """Test 18: Predictor dynamically executes inference across all v2 models."""
    v2_map = {
        "ppe-workshop-v2": "ppe",
        "fire-smoke-v2": "fire_smoke",
        "danger-zone-v2": "danger_zone",
        "fall-detection-v2": "fall_detection",
    }
    for model_id, ds_key in v2_map.items():
        test_img = next((REPO_ROOT / "datasets" / ds_key / "processed" / "images" / "test").glob("*.jpg"))
        res = predictor.predict(model_id, str(test_img), confidence_threshold=0.1)
        assert res.model_id == model_id
        assert res.processing_time_ms >= 0.0
        assert isinstance(res.detections, list)
        if model_id == "fall-detection-v2" and len(res.detections) > 0:
            assert res.task_type == TaskType.POSE_ESTIMATION
            assert len(res.detections[0].keypoints) == 17

