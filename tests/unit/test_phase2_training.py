"""Unit tests for Phase 2 Training Engine, Configuration, Run Management, and Registration."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from pydantic import ValidationError

from ai_safety.core.exceptions import RegistryError
from ai_safety.core.lineage import ModelLineage
from ai_safety.core.model_metadata import ModelOrigin, ModelStatus, TaskType
from ai_safety.core.model_registry import ModelRegistry
from ai_safety.dataset.analyzer import DatasetAnalyzer
from ai_safety.evaluation.evaluator import EvaluationMetrics
from ai_safety.training.config import (
    TrainingConfig,
    get_training_profile,
    resolve_device,
)
from ai_safety.training.run_manager import TrainingRunManager
from ai_safety.training.service import TrainingService


# -----------------------------------------------------------------------------
# Configuration & Hardware Device Tests
# -----------------------------------------------------------------------------

def test_valid_training_config():
    """Verify valid training configuration instantiates correctly."""
    cfg = TrainingConfig(
        base_model="yolo11n",
        dataset_path="data/ppe",
        epochs=15,
        image_size=320,
        batch_size=8,
        device="cpu",
    )
    assert cfg.epochs == 15
    assert cfg.image_size == 320
    assert cfg.batch_size == 8
    assert cfg.device == "cpu"


def test_invalid_epoch():
    """Verify error when epochs <= 0."""
    with pytest.raises(ValidationError):
        TrainingConfig(base_model="yolo11n", dataset_path="data", epochs=0)


def test_invalid_image_size():
    """Verify error when image_size <= 0."""
    with pytest.raises(ValidationError):
        TrainingConfig(base_model="yolo11n", dataset_path="data", image_size=0)


def test_invalid_batch_size():
    """Verify error when batch_size <= 0."""
    with pytest.raises(ValidationError):
        TrainingConfig(base_model="yolo11n", dataset_path="data", batch_size=-4)


def test_invalid_device():
    """Verify error when invalid device name is specified."""
    with pytest.raises(ValidationError):
        TrainingConfig(base_model="yolo11n", dataset_path="data", device="tpu")


def test_cpu_device_resolution():
    """Verify CPU resolution returns cpu."""
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("CPU") == "cpu"


def test_cuda_unavailable_handling():
    """Verify explicit CUDA request fails clearly if CUDA is unavailable (no silent fallback)."""
    with patch("torch.cuda.is_available", return_value=False):
        with pytest.raises(RuntimeError, match="CUDA was explicitly requested"):
            resolve_device("cuda")

        # But AUTO safely falls back to CPU
        assert resolve_device("auto") == "cpu"


def test_training_profile_loading():
    """Verify named profile workshop_cpu loads expected settings."""
    prof = get_training_profile("workshop_cpu")
    assert prof["device"] == "cpu"
    assert prof["epochs"] == 10
    assert prof["image_size"] == 320
    assert prof["batch_size"] == 4

    cfg = TrainingConfig.from_profile("workshop_cpu", base_model="yolo11n", dataset_path="data/test")
    assert cfg.epochs == 10
    assert cfg.device == "cpu"


# -----------------------------------------------------------------------------
# Training Run & Lineage Tests
# -----------------------------------------------------------------------------

def test_training_run_creation_and_unique_id(tmp_path: Path):
    """Verify run manager creates unique run directories with correct layout."""
    mgr = TrainingRunManager(runs_root=tmp_path / "runs")
    cfg = TrainingConfig.from_profile("workshop_cpu", base_model="yolo11n", dataset_path="data")
    summary = DatasetAnalyzer(Path("tests/fixtures/synthetic_dataset")).analyze()

    handle1 = mgr.create_run("yolo11n", "data", cfg, summary)
    handle2 = mgr.create_run("yolo11n", "data", cfg, summary)

    # Unique run IDs
    assert handle1.run_id != handle2.run_id
    assert handle1.run_id.startswith("run-")
    assert handle2.run_id.startswith("run-")

    # Directory layout exists
    assert handle1.run_dir.exists()
    assert (handle1.run_dir / "weights").exists()
    assert (handle1.run_dir / "logs").exists()
    assert (handle1.run_dir / "results").exists()


def test_run_persistence_config_summary_lineage(tmp_path: Path):
    """Verify configuration.json, dataset_summary.json, and lineage.json are persisted."""
    mgr = TrainingRunManager(runs_root=tmp_path / "runs")
    cfg = TrainingConfig.from_profile("workshop_cpu", base_model="yolo11n", dataset_path="data")
    summary = DatasetAnalyzer(Path("tests/fixtures/synthetic_dataset")).analyze()

    handle = mgr.create_run("yolo11n", "data", cfg, summary, base_model_version="1.0.0")

    assert (handle.run_dir / "configuration.json").exists()
    assert (handle.run_dir / "dataset_summary.json").exists()
    assert (handle.run_dir / "lineage.json").exists()

    with open(handle.run_dir / "lineage.json", "r", encoding="utf-8") as f:
        lineage_data = json.load(f)
        assert lineage_data["run_id"] == handle.run_id
        assert lineage_data["base_model_id"] == "yolo11n"
        assert lineage_data["device"] == "cpu"
        assert lineage_data["status"] == "created"


def test_best_model_detection(tmp_path: Path):
    """Verify handle finds best.pt artifact when present."""
    mgr = TrainingRunManager(runs_root=tmp_path / "runs")
    cfg = TrainingConfig.from_profile("workshop_cpu", base_model="yolo11n", dataset_path="data")
    summary = DatasetAnalyzer(Path("tests/fixtures/synthetic_dataset")).analyze()

    handle = mgr.create_run("yolo11n", "data", cfg, summary)
    assert handle.find_best_model() is None

    # Simulate model training saving best.pt
    best_file = handle.weights_dir / "best.pt"
    best_file.write_bytes(b"MOCK_BEST_WEIGHTS")

    found = handle.find_best_model()
    assert found is not None
    assert found == best_file


# -----------------------------------------------------------------------------
# Training Service & Model Registration Tests
# -----------------------------------------------------------------------------

def test_pretrained_base_model_protection(tmp_path: Path):
    """Verify training service never overwrites pretrained models."""
    registry_file = tmp_path / "registry.json"
    registry = ModelRegistry(registry_file=registry_file, repo_root=tmp_path)
    service = TrainingService(
        registry=registry,
        runs_dir=tmp_path / "runs",
        custom_models_dir=tmp_path / "models" / "custom",
        repo_root=tmp_path,
    )

    # Output directory must not target models/pretrained
    assert "models/pretrained" not in str(service.runs_dir).replace("\\", "/")
    assert "models/pretrained" not in str(service.custom_models_dir).replace("\\", "/")


def test_custom_model_registration_and_discovery(tmp_path: Path):
    """Verify training workflow registers custom model into registry and creates folder structure."""
    registry_file = tmp_path / "registry.json"
    registry = ModelRegistry(registry_file=registry_file, repo_root=tmp_path)
    service = TrainingService(
        registry=registry,
        runs_dir=tmp_path / "runs",
        custom_models_dir=tmp_path / "models" / "custom",
        repo_root=tmp_path,
    )

    dataset_path = Path("tests/fixtures/synthetic_dataset").resolve()
    cfg = TrainingConfig.from_profile("workshop_cpu", base_model="yolo11n", dataset_path=str(dataset_path))

    # Mock the heavy Ultralytics execution and evaluation for fast unit testing
    def mock_training(base_weights, data_yaml, config, run_handle):
        best_pt = run_handle.weights_dir / "best.pt"
        best_pt.write_bytes(b"MOCK_YOLO_BEST_WEIGHTS")

    mock_metrics = EvaluationMetrics(
        precision=0.92,
        recall=0.88,
        map50=0.94,
        map50_95=0.76,
    )

    with patch.object(service, "_execute_training", side_effect=mock_training):
        with patch.object(service.evaluator, "evaluate_weights", return_value=mock_metrics):
            res = service.train(
                base_model="yolo11n",
                dataset_path=dataset_path,
                configuration=cfg,
                custom_name="workshop-ppe-detector",
                version="1.0.0",
            )

    # Model registered in ModelRegistry
    assert registry.model_exists(res.custom_model_id)
    retrieved = registry.get_model(res.custom_model_id)
    assert retrieved.origin == ModelOrigin.CUSTOM
    assert retrieved.status == ModelStatus.READY
    assert retrieved.name == "workshop-ppe-detector"
    assert retrieved.version == "1.0.0"

    # Files created under models/custom/<id>/
    custom_dir = res.custom_model_dir
    assert (custom_dir / "model.pt").exists()
    assert (custom_dir / "metadata.json").exists()
    assert (custom_dir / "metrics.json").exists()
    assert (custom_dir / "lineage.json").exists()

    # Metrics matches real values
    with open(custom_dir / "metrics.json", "r", encoding="utf-8") as f:
        m_data = json.load(f)
        assert m_data["map50"] == 0.94


def test_duplicate_version_protection(tmp_path: Path):
    """Verify duplicate version registration raises RegistryError."""
    registry_file = tmp_path / "registry.json"
    registry = ModelRegistry(registry_file=registry_file, repo_root=tmp_path)
    service = TrainingService(
        registry=registry,
        runs_dir=tmp_path / "runs",
        custom_models_dir=tmp_path / "models" / "custom",
        repo_root=tmp_path,
    )

    dataset_path = Path("tests/fixtures/synthetic_dataset").resolve()
    cfg = TrainingConfig.from_profile("workshop_cpu", base_model="yolo11n", dataset_path=str(dataset_path))

    def mock_training(base_weights, data_yaml, config, run_handle):
        best_pt = run_handle.weights_dir / "best.pt"
        best_pt.write_bytes(b"MOCK_YOLO_BEST_WEIGHTS")

    mock_metrics = EvaluationMetrics(precision=0.9, recall=0.85, map50=0.91, map50_95=0.7)

    with patch.object(service, "_execute_training", side_effect=mock_training):
        with patch.object(service.evaluator, "evaluate_weights", return_value=mock_metrics):
            # First registration succeeds
            service.train(
                base_model="yolo11n",
                dataset_path=dataset_path,
                configuration=cfg,
                custom_name="duplicate-test",
                version="1.0.0",
            )

            # Second registration with identical version raises RegistryError
            with pytest.raises(RegistryError, match="already exists"):
                service.train(
                    base_model="yolo11n",
                    dataset_path=dataset_path,
                    configuration=cfg,
                    custom_name="duplicate-test",
                    version="1.0.0",
                )


def test_native_export(tmp_path: Path):
    """Verify native model bundle export contains model, metadata, metrics, and lineage."""
    registry_file = tmp_path / "registry.json"
    registry = ModelRegistry(registry_file=registry_file, repo_root=tmp_path)
    service = TrainingService(
        registry=registry,
        runs_dir=tmp_path / "runs",
        custom_models_dir=tmp_path / "models" / "custom",
        repo_root=tmp_path,
    )

    dataset_path = Path("tests/fixtures/synthetic_dataset").resolve()
    cfg = TrainingConfig.from_profile("workshop_cpu", base_model="yolo11n", dataset_path=str(dataset_path))

    def mock_training(base_weights, data_yaml, config, run_handle):
        best_pt = run_handle.weights_dir / "best.pt"
        best_pt.write_bytes(b"MOCK_YOLO_BEST_WEIGHTS")

    mock_metrics = EvaluationMetrics(precision=0.9, recall=0.85, map50=0.91, map50_95=0.7)

    with patch.object(service, "_execute_training", side_effect=mock_training):
        with patch.object(service.evaluator, "evaluate_weights", return_value=mock_metrics):
            res = service.train(
                base_model="yolo11n",
                dataset_path=dataset_path,
                configuration=cfg,
                custom_name="export-test",
                version="1.0.0",
            )

    export_dest = tmp_path / "exports"
    bundle_path = service.export_native(model_id=res.custom_model_id, export_output_dir=export_dest)

    assert bundle_path.exists()
    assert (bundle_path / "model.pt").exists()
    assert (bundle_path / "metadata.json").exists()
    assert (bundle_path / "metrics.json").exists()
    assert (bundle_path / "lineage.json").exists()
