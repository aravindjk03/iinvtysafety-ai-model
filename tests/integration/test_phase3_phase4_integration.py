"""Integration tests verifying multi-model architectural capability across Phase 3 and Phase 4.

Validates that ModelRegistry and Predictor seamlessly handle both PPE and Fire/Smoke models.
"""

from pathlib import Path
import pytest

from ai_safety.core.model_metadata import ModelOrigin, ModelStatus, TaskType
from ai_safety.core.model_registry import ModelRegistry
from ai_safety.core.prediction import PredictionResult
from ai_safety.inference.predictor import Predictor

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture
def registry():
    return ModelRegistry(registry_file="models/registry.json", repo_root=REPO_ROOT)


@pytest.fixture
def predictor(registry):
    return Predictor(registry=registry)


@pytest.mark.integration
def test_dual_model_registry_and_predictor_integration(registry, predictor):
    """Verify that ModelRegistry contains both PPE and Fire/Smoke, and Predictor switches seamlessly."""
    # 1. ModelRegistry contains both
    assert registry.model_exists("ppe-workshop-v1")
    assert registry.model_exists("fire-smoke-workshop-v1")

    ppe_meta = registry.get_model("ppe-workshop-v1")
    fs_meta = registry.get_model("fire-smoke-workshop-v1")

    assert ppe_meta.origin == ModelOrigin.CUSTOM
    assert fs_meta.origin == ModelOrigin.CUSTOM
    assert ppe_meta.classes == ["person", "helmet", "vest"]
    assert fs_meta.classes == ["fire", "smoke"]

    # 2. Both retain independent lineage
    ppe_lineage = REPO_ROOT / "models" / "custom" / "ppe-workshop-v1" / "lineage.json"
    fs_lineage = REPO_ROOT / "models" / "custom" / "fire-smoke-workshop-v1" / "lineage.json"
    assert ppe_lineage.exists()
    assert fs_lineage.exists()

    # 3. Same Predictor loads both by model_id and executes inference
    ppe_img = next((REPO_ROOT / "datasets" / "ppe" / "processed" / "images" / "test").glob("*.jpg"))
    fs_img = next((REPO_ROOT / "datasets" / "fire_smoke" / "processed" / "images" / "test").glob("*.jpg"))

    res_ppe = predictor.predict("ppe-workshop-v1", str(ppe_img), confidence_threshold=0.001)
    res_fs = predictor.predict("fire-smoke-workshop-v1", str(fs_img), confidence_threshold=0.001)

    # 4. Normalized contract adherence
    assert isinstance(res_ppe, PredictionResult)
    assert isinstance(res_fs, PredictionResult)

    assert res_ppe.model_id == "ppe-workshop-v1"
    assert res_fs.model_id == "fire-smoke-workshop-v1"

    assert res_ppe.task_type == TaskType.OBJECT_DETECTION
    assert res_fs.task_type == TaskType.OBJECT_DETECTION

    # 5. Switching back and forth without state corruption
    res_ppe_2 = predictor.predict("ppe-workshop-v1", str(ppe_img), confidence_threshold=0.001)
    assert res_ppe_2.model_id == "ppe-workshop-v1"
