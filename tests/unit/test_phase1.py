"""Unit tests for Phase 1 specification of AI Safety Model Lab.

Covers all 14 required verification scenarios:
  TEST 1: Model metadata validation
  TEST 2: Valid model registration
  TEST 3: Duplicate model registration
  TEST 4: Model lookup
  TEST 5: Model listing
  TEST 6: Pretrained model protection
  TEST 7: Prediction schema validation
  TEST 8: Invalid model type handling
  TEST 9: Missing model handling
  TEST 10: Relative path handling
  TEST 11: CPU device selection
  TEST 12: Training interface exists
  TEST 13: Evaluation interface exists
  TEST 14: Export interface exists
"""

from pathlib import Path
import pytest
from pydantic import ValidationError

from ai_safety.adapters.base_adapter import BaseModelAdapter
from ai_safety.adapters.yolo_adapter import YOLOAdapter
from ai_safety.core.exceptions import (
    InvalidInputError,
    ModelLoadError,
    ModelNotFoundError,
    PredictionError,
    RegistryError,
    UnsupportedModelError,
)
from ai_safety.core.model_metadata import (
    ExecutionDevice,
    ModelMetadata,
    ModelOrigin,
    ModelStatus,
    TaskType,
)
from ai_safety.core.model_registry import ModelRegistry
from ai_safety.core.prediction import (
    BoundingBox,
    Detection,
    Keypoint,
    PredictionResult,
)
from ai_safety.evaluation.evaluator import EvaluationMetrics, ModelEvaluator
from ai_safety.export.exporter import ExportConfig, ExportFormat, ModelExporter
from ai_safety.inference.predictor import Predictor
from ai_safety.training.trainer import ModelTrainer, TrainingConfig


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def repo_root(tmp_path: Path) -> Path:
    """Creates a temporary repository root with required directory layout."""
    (tmp_path / "models").mkdir(parents=True, exist_ok=True)
    (tmp_path / "tests" / "fixtures").mkdir(parents=True, exist_ok=True)
    
    # Create tiny dummy fixture strictly for architecture testing
    dummy_file = tmp_path / "tests" / "fixtures" / "dummy_model.pt"
    dummy_file.write_bytes(b"TEST ONLY - MOCK ARCHITECTURE FIXTURE")
    return tmp_path


@pytest.fixture
def sample_metadata() -> ModelMetadata:
    """Returns valid sample metadata for a custom model."""
    return ModelMetadata(
        model_id="custom-ppe-yolo-v1",
        name="Custom PPE YOLO Model",
        version="1.0.0",
        task_type=TaskType.OBJECT_DETECTION,
        framework="ultralytics_yolo",
        architecture="yolo11n",
        model_path="models/custom/ppe_yolo.pt",
        classes=["helmet", "vest", "person"],
        input_size=(640, 640),
        confidence_threshold=0.25,
        origin=ModelOrigin.CUSTOM,
        base_model="yolo11n",
        training_run_id="run_12345",
        status=ModelStatus.REGISTERED,
        training_supported=True,
    )


@pytest.fixture
def pretrained_metadata() -> ModelMetadata:
    """Returns valid sample metadata for a protected pretrained model."""
    return ModelMetadata(
        model_id="pretrained-fire-yolo-v1",
        name="Pretrained Fire & Smoke Detector",
        version="1.0.0",
        task_type=TaskType.OBJECT_DETECTION,
        framework="ultralytics_yolo",
        architecture="yolo11n",
        model_path="models/pretrained/fire_smoke/fire_yolo.pt",
        classes=["fire", "smoke"],
        input_size=(640, 640),
        confidence_threshold=0.30,
        origin=ModelOrigin.PRETRAINED,
        base_model=None,
        training_run_id=None,
        status=ModelStatus.REGISTERED,
        training_supported=False,
    )


# -----------------------------------------------------------------------------
# TEST 1: Model metadata validation
# -----------------------------------------------------------------------------
def test_01_model_metadata_validation(sample_metadata: ModelMetadata):
    """TEST 1: Validate required fields, invalid boundaries, and types."""
    assert sample_metadata.model_id == "custom-ppe-yolo-v1"
    assert sample_metadata.task_type == TaskType.OBJECT_DETECTION
    assert sample_metadata.origin == ModelOrigin.CUSTOM
    assert sample_metadata.classes == ["helmet", "vest", "person"]
    assert sample_metadata.confidence_threshold == 0.25

    # Missing required field
    with pytest.raises(ValidationError):
        ModelMetadata(
            model_id="missing-fields",
            # name missing
            version="1.0.0",
            task_type=TaskType.OBJECT_DETECTION,
            framework="ultralytics_yolo",
            architecture="yolo11n",
            model_path="models/m.pt",
            classes=["a"],
            origin=ModelOrigin.CUSTOM,
        )

    # Invalid confidence threshold out of bounds [0.0, 1.0]
    with pytest.raises(ValidationError):
        sample_metadata.confidence_threshold = 1.5

    with pytest.raises(ValidationError):
        sample_metadata.confidence_threshold = -0.1

    # Empty classes
    with pytest.raises(ValidationError):
        sample_metadata.classes = []


# -----------------------------------------------------------------------------
# TEST 2: Valid model registration
# -----------------------------------------------------------------------------
def test_02_valid_model_registration(repo_root: Path, sample_metadata: ModelMetadata):
    """TEST 2: Verify custom model registers successfully and persists."""
    registry_file = repo_root / "models" / "registry.json"
    registry = ModelRegistry(registry_file=registry_file, repo_root=repo_root)

    assert not registry.model_exists(sample_metadata.model_id)
    registry.register_model(sample_metadata)

    assert registry.model_exists(sample_metadata.model_id)
    retrieved = registry.get_model(sample_metadata.model_id)
    assert retrieved.name == sample_metadata.name

    # Verify reloading from disk
    new_reg = ModelRegistry(registry_file=registry_file, repo_root=repo_root)
    assert new_reg.model_exists(sample_metadata.model_id)


# -----------------------------------------------------------------------------
# TEST 3: Duplicate model registration
# -----------------------------------------------------------------------------
def test_03_duplicate_model_registration(repo_root: Path, sample_metadata: ModelMetadata):
    """TEST 3: Registering duplicate model ID raises RegistryError when overwrite=False."""
    registry = ModelRegistry(registry_file=repo_root / "models" / "registry.json", repo_root=repo_root)
    registry.register_model(sample_metadata)

    # Attempt duplicate registration without overwrite
    with pytest.raises(RegistryError, match="already registered"):
        registry.register_model(sample_metadata, overwrite=False)

    # Allowing overwrite for custom model succeeds
    updated = sample_metadata.model_copy(update={"version": "1.0.1"})
    registry.register_model(updated, overwrite=True)
    assert registry.get_model(sample_metadata.model_id).version == "1.0.1"


# -----------------------------------------------------------------------------
# TEST 4: Model lookup
# -----------------------------------------------------------------------------
def test_04_model_lookup(repo_root: Path, sample_metadata: ModelMetadata):
    """TEST 4: Model lookup succeeds for existing models and raises ModelNotFoundError for missing."""
    registry = ModelRegistry(registry_file=repo_root / "models" / "registry.json", repo_root=repo_root)
    registry.register_model(sample_metadata)

    found = registry.get_model(sample_metadata.model_id)
    assert found.model_id == sample_metadata.model_id

    with pytest.raises(ModelNotFoundError, match="not found in registry"):
        registry.get_model("non-existent-id")


# -----------------------------------------------------------------------------
# TEST 5: Model listing
# -----------------------------------------------------------------------------
def test_05_model_listing(
    repo_root: Path,
    sample_metadata: ModelMetadata,
    pretrained_metadata: ModelMetadata,
):
    """TEST 5: Verify listing models with filtering by origin and task type."""
    registry = ModelRegistry(registry_file=repo_root / "models" / "registry.json", repo_root=repo_root)
    registry.register_model(sample_metadata)
    registry.register_model(pretrained_metadata)

    # Pose estimation model for filtering test
    pose_meta = ModelMetadata(
        model_id="custom-fall-pose-v1",
        name="Fall Pose Estimation",
        version="1.0.0",
        task_type=TaskType.POSE_ESTIMATION,
        framework="ultralytics_yolo",
        architecture="yolo11n-pose",
        model_path="models/custom/fall_pose.pt",
        classes=["person"],
        origin=ModelOrigin.CUSTOM,
    )
    registry.register_model(pose_meta)

    all_models = registry.list_models()
    assert len(all_models) == 3

    pretrained_only = registry.list_models(origin=ModelOrigin.PRETRAINED)
    assert len(pretrained_only) == 1
    assert pretrained_only[0].model_id == pretrained_metadata.model_id

    custom_only = registry.list_models(origin=ModelOrigin.CUSTOM)
    assert len(custom_only) == 2

    pose_only = registry.list_models(task_type=TaskType.POSE_ESTIMATION)
    assert len(pose_only) == 1
    assert pose_only[0].model_id == "custom-fall-pose-v1"


# -----------------------------------------------------------------------------
# TEST 6: Pretrained model protection
# -----------------------------------------------------------------------------
def test_06_pretrained_model_protection(repo_root: Path, pretrained_metadata: ModelMetadata):
    """TEST 6: Pretrained baseline models cannot be overwritten or updated."""
    registry = ModelRegistry(registry_file=repo_root / "models" / "registry.json", repo_root=repo_root)
    registry.register_model(pretrained_metadata)

    # Attempt to overwrite even with overwrite=True must fail
    attacker_copy = pretrained_metadata.model_copy(update={"name": "Hacked Model"})
    with pytest.raises(RegistryError, match="Cannot overwrite protected pretrained model"):
        registry.register_model(attacker_copy, overwrite=True)

    # Attempt to update via update_model must fail
    with pytest.raises(RegistryError, match="Cannot update protected pretrained model"):
        registry.update_model(attacker_copy)


# -----------------------------------------------------------------------------
# TEST 7: Prediction schema validation
# -----------------------------------------------------------------------------
def test_07_prediction_schema_validation():
    """TEST 7: Verify normalized PredictionResult, Detection, BoundingBox, Keypoint schemas."""
    # BoundingBox validation
    box = BoundingBox(x1=10.0, y1=20.0, x2=110.0, y2=220.0)
    assert box.width == 100.0
    assert box.height == 200.0
    assert box.area == 20000.0

    # Inverted coordinates must fail
    with pytest.raises(ValidationError):
        BoundingBox(x1=100.0, y1=20.0, x2=10.0, y2=220.0)

    # Keypoint validation for pose
    kp = Keypoint(id=0, name="nose", x=50.0, y=60.0, confidence=0.95)
    assert kp.confidence == 0.95

    # Detection validation with optional keypoints
    detection = Detection(
        class_id=0,
        class_name="person",
        confidence=0.92,
        bounding_box=box,
        keypoints=[kp],
    )
    assert detection.class_name == "person"
    assert len(detection.keypoints) == 1

    # PredictionResult validation
    result = PredictionResult(
        model_id="test-model",
        task_type=TaskType.OBJECT_DETECTION,
        processing_time_ms=15.4,
        detections=[detection],
    )
    assert result.processing_time_ms == 15.4
    assert len(result.detections) == 1
    assert result.timestamp is not None


# -----------------------------------------------------------------------------
# TEST 8: Invalid model type handling
# -----------------------------------------------------------------------------
def test_08_invalid_model_type_handling(repo_root: Path):
    """TEST 8: Unsupported task types, frameworks, or models are rejected explicitly."""
    # Invalid task type (not in TaskType enum)
    with pytest.raises(ValidationError):
        ModelMetadata(
            model_id="invalid-task-model",
            name="Invalid Task",
            version="1.0.0",
            task_type="unsupported_custom_task",  # type: ignore
            framework="ultralytics_yolo",
            architecture="yolo11n",
            model_path="models/m.pt",
            classes=["test"],
            origin=ModelOrigin.CUSTOM,
        )

    # Unsupported framework in YOLO adapter
    unsupported_fw_meta = ModelMetadata(
        model_id="tf-model",
        name="TF Model",
        version="1.0.0",
        task_type=TaskType.OBJECT_DETECTION,
        framework="tensorflow_keras",
        architecture="resnet50",
        model_path="models/m.pt",
        classes=["test"],
        origin=ModelOrigin.CUSTOM,
    )
    with pytest.raises(UnsupportedModelError, match="does not support framework"):
        YOLOAdapter(metadata=unsupported_fw_meta, repo_root=repo_root)

    # Unsupported framework in Predictor
    registry = ModelRegistry(registry_file=repo_root / "models" / "registry.json", repo_root=repo_root)
    registry.register_model(unsupported_fw_meta)
    predictor = Predictor(registry=registry, repo_root=repo_root)

    with pytest.raises(UnsupportedModelError, match="No adapter available"):
        predictor.predict(model_id="tf-model", input_data="test_input.jpg")


# -----------------------------------------------------------------------------
# TEST 9: Missing model handling
# -----------------------------------------------------------------------------
def test_09_missing_model_handling(repo_root: Path, sample_metadata: ModelMetadata):
    """TEST 9: Missing weights files fail clearly with ModelLoadError; validate returns False."""
    adapter = YOLOAdapter(metadata=sample_metadata, repo_root=repo_root)

    # File does not exist yet on disk
    assert not adapter.validate()

    with pytest.raises(ModelLoadError, match="Model weights file not found"):
        adapter.load_model()

    # Predict before load fails with ModelLoadError
    with pytest.raises(ModelLoadError, match="not loaded"):
        adapter.predict(input_data="image.jpg")


# -----------------------------------------------------------------------------
# TEST 10: Relative path handling
# -----------------------------------------------------------------------------
def test_10_relative_path_handling():
    """TEST 10: Enforce relative paths, normalize separators, reject path traversal."""
    # Absolute paths must be rejected
    with pytest.raises(ValidationError, match="must be relative"):
        ModelMetadata(
            model_id="abs-path-model",
            name="Absolute Path Model",
            version="1.0.0",
            task_type=TaskType.OBJECT_DETECTION,
            framework="ultralytics_yolo",
            architecture="yolo11n",
            model_path="C:/Windows/System32/calc.exe",
            classes=["calc"],
            origin=ModelOrigin.CUSTOM,
        )

    # Path traversal with '..' must be rejected
    with pytest.raises(ValidationError, match="Path traversal detected"):
        ModelMetadata(
            model_id="traversal-model",
            name="Traversal Model",
            version="1.0.0",
            task_type=TaskType.OBJECT_DETECTION,
            framework="ultralytics_yolo",
            architecture="yolo11n",
            model_path="../../outside/model.pt",
            classes=["secret"],
            origin=ModelOrigin.CUSTOM,
        )

    # Valid relative path is normalized
    meta = ModelMetadata(
        model_id="valid-rel-path",
        name="Valid Relative Path Model",
        version="1.0.0",
        task_type=TaskType.OBJECT_DETECTION,
        framework="ultralytics_yolo",
        architecture="yolo11n",
        model_path="models/custom/model.pt",
        classes=["obj"],
        origin=ModelOrigin.CUSTOM,
    )
    assert meta.model_path == "models/custom/model.pt"


# -----------------------------------------------------------------------------
# TEST 11: CPU device selection
# -----------------------------------------------------------------------------
def test_11_cpu_device_selection(repo_root: Path, sample_metadata: ModelMetadata):
    """TEST 11: CPU is the default execution device; unsupported devices are rejected."""
    # Base adapter default
    adapter = BaseModelAdapter(metadata=sample_metadata, repo_root=repo_root)
    assert adapter.device == "cpu"

    # Explicit cuda request (syntactically valid without requiring GPU hardware)
    adapter_cuda = BaseModelAdapter(metadata=sample_metadata, device="cuda", repo_root=repo_root)
    assert adapter_cuda.device == "cuda"

    # Invalid device rejected
    with pytest.raises(ValueError, match="Invalid device"):
        BaseModelAdapter(metadata=sample_metadata, device="tpu_v4", repo_root=repo_root)

    # Predictor device default
    registry = ModelRegistry(registry_file=repo_root / "models" / "registry.json", repo_root=repo_root)
    predictor = Predictor(registry=registry, repo_root=repo_root)
    assert predictor.default_device == "cpu"


# -----------------------------------------------------------------------------
# TEST 12: Training interface exists
# -----------------------------------------------------------------------------
def test_12_training_interface_exists(repo_root: Path):
    """TEST 12: Trainer abstraction exists, validates parameters, and preserves future contract."""
    config = TrainingConfig(epochs=25, batch_size=8, imgsz=640, learning_rate=0.005)
    assert config.epochs == 25
    assert config.device == "cpu"

    class ConcreteTrainer(ModelTrainer):
        def train(self) -> str:
            return super().train()

    trainer = ConcreteTrainer(
        base_model="yolo11n",
        dataset="data/participant_dataset.yaml",
        training_configuration=config,
        output_directory="models/custom/run_01",
        repo_root=repo_root,
    )

    assert trainer.base_model == "yolo11n"
    assert trainer.dataset == "data/participant_dataset.yaml"
    assert trainer.run_id.startswith("train_")

    # Calling train in Phase 1 must raise NotImplementedError
    with pytest.raises(NotImplementedError, match="deferred to Phase 2"):
        trainer.train()


# -----------------------------------------------------------------------------
# TEST 13: Evaluation interface exists
# -----------------------------------------------------------------------------
def test_13_evaluation_interface_exists(repo_root: Path):
    """TEST 13: Evaluator abstraction exists and defines standardized safety metrics."""
    metrics = EvaluationMetrics(
        precision=0.88,
        recall=0.84,
        map50=0.91,
        map50_95=0.72,
        fitness=0.85,
    )
    assert metrics.precision == 0.88
    assert metrics.map50 == 0.91

    registry = ModelRegistry(registry_file=repo_root / "models" / "registry.json", repo_root=repo_root)

    class ConcreteEvaluator(ModelEvaluator):
        def evaluate(self, model_id: str, dataset_path: str, **kwargs) -> EvaluationMetrics:
            return super().evaluate(model_id, dataset_path, **kwargs)

    evaluator = ConcreteEvaluator(registry=registry, repo_root=repo_root)
    with pytest.raises(NotImplementedError, match="deferred to Phase 2"):
        evaluator.evaluate(model_id="test", dataset_path="test_data.yaml")


# -----------------------------------------------------------------------------
# TEST 14: Export interface exists
# -----------------------------------------------------------------------------
def test_14_export_interface_exists(repo_root: Path):
    """TEST 14: Exporter abstraction exists and defines supported deployment formats."""
    assert ExportFormat.ONNX == "onnx"
    assert ExportFormat.TORCHSCRIPT == "torchscript"
    assert ExportFormat.TFLITE == "tflite"
    assert ExportFormat.ENGINE == "engine"

    export_cfg = ExportConfig(format=ExportFormat.ONNX, half=False, opset=12)
    assert export_cfg.format == ExportFormat.ONNX

    registry = ModelRegistry(registry_file=repo_root / "models" / "registry.json", repo_root=repo_root)

    class ConcreteExporter(ModelExporter):
        def export(self, model_id: str, format: ExportFormat, output_dir=None, **kwargs) -> str:
            return super().export(model_id, format, output_dir, **kwargs)

    exporter = ConcreteExporter(registry=registry, repo_root=repo_root)
    with pytest.raises(NotImplementedError, match="deferred to Phase 2"):
        exporter.export(model_id="test", format=ExportFormat.ONNX)


# -----------------------------------------------------------------------------
# Mock dummy fixture testing (TEST ONLY)
# -----------------------------------------------------------------------------
def test_mock_dummy_fixture_loading(repo_root: Path):
    """Ensure mock dummy fixture can be loaded by YOLOAdapter for architecture checks without fake predictions."""
    dummy_meta = ModelMetadata(
        model_id="dummy-fixture-model",
        name="TEST ONLY Dummy Fixture",
        version="0.0.1",
        task_type=TaskType.OBJECT_DETECTION,
        framework="ultralytics_yolo",
        architecture="yolo11n",
        model_path="tests/fixtures/dummy_model.pt",
        classes=["dummy"],
        origin=ModelOrigin.CUSTOM,
    )
    adapter = YOLOAdapter(metadata=dummy_meta, repo_root=repo_root)
    assert adapter.validate()

    # Loading dummy fixture succeeds in recognizing the TEST ONLY header
    adapter.load_model()
    assert adapter.is_loaded
    assert adapter.metadata.status == ModelStatus.READY

    # But calling predict strictly fails without emitting fake predictions
    with pytest.raises(PredictionError, match="Dummy test fixture cannot execute real inference"):
        adapter.predict(input_data="dummy_input.jpg")

    # Calling predict with empty input fails with InvalidInputError
    with pytest.raises(InvalidInputError):
        adapter.predict(input_data="")
