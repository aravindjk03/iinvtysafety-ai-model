"""Integration test executing the full end-to-end training and registration lifecycle.

Marked as 'integration' so it is excluded from fast unit test runs:
Run via:
    pytest -m integration
"""

from pathlib import Path
import pytest

from ai_safety.core.model_metadata import ModelOrigin, ModelStatus
from ai_safety.core.model_registry import ModelRegistry
from ai_safety.inference.predictor import Predictor
from ai_safety.training.config import TrainingConfig
from ai_safety.training.service import TrainingService


@pytest.mark.integration
def test_full_training_lifecycle_integration(tmp_path: Path):
    """End-to-end test:

    Dataset Validation -> Training Execution -> Best Model Detection ->
    Evaluation -> Model Registration -> Lineage -> Predictor Lookup.

    Uses yolo11n.yaml (initializes architecture locally offline without downloading weights).
    """
    registry_file = tmp_path / "models" / "registry.json"
    registry = ModelRegistry(registry_file=registry_file, repo_root=tmp_path)

    service = TrainingService(
        registry=registry,
        runs_dir=tmp_path / "training_runs",
        custom_models_dir=tmp_path / "models" / "custom",
        repo_root=tmp_path,
    )

    dataset_path = Path("tests/fixtures/synthetic_dataset").resolve()

    # Configure a tiny 1-epoch offline training run on CPU
    config = TrainingConfig(
        base_model="yolo11n.yaml",  # Builds architecture locally offline
        dataset_path=str(dataset_path),
        epochs=1,
        image_size=32,
        batch_size=2,
        device="cpu",
        workers=0,
        project_name="integration_test_model",
    )

    # 1. Execute full training workflow
    result = service.train(
        base_model="yolo11n.yaml",
        dataset_path=dataset_path,
        configuration=config,
        custom_name="Integration Test Model",
        version="1.0.0",
    )

    # 2. Verify Result Structure
    assert result.run_id.startswith("run-")
    assert result.custom_model_id == "integration-test-model-v1-0-0"
    assert result.version == "1.0.0"
    assert result.best_weights_path.exists()
    assert result.custom_model_dir.exists()

    # 3. Verify Artifacts Created
    assert (result.custom_model_dir / "model.pt").exists()
    assert (result.custom_model_dir / "metadata.json").exists()
    assert (result.custom_model_dir / "metrics.json").exists()
    assert (result.custom_model_dir / "lineage.json").exists()

    # 4. Verify Metrics were captured (not fabricated)
    assert 0.0 <= result.metrics.precision <= 1.0
    assert 0.0 <= result.metrics.recall <= 1.0
    assert 0.0 <= result.metrics.map50 <= 1.0

    # 5. Verify Discovery in Phase 1 ModelRegistry
    assert registry.model_exists(result.custom_model_id)
    registered_meta = registry.get_model(result.custom_model_id)
    assert registered_meta.origin == ModelOrigin.CUSTOM
    assert registered_meta.status == ModelStatus.READY
    assert registered_meta.training_run_id == result.run_id

    # 6. Verify Predictor Adapter can load the newly trained custom model
    predictor = Predictor(registry=registry, default_device="cpu", repo_root=tmp_path)
    adapter = predictor.get_adapter(result.custom_model_id)
    adapter.load_model()
    assert adapter.is_loaded is True
    assert adapter.metadata.status == ModelStatus.READY
