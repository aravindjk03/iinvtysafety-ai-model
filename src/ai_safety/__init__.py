"""AI Safety Model Lab - Foundation ML Architecture & Model Lifecycle Engine."""

__version__ = "0.2.0"

from ai_safety.adapters.base_adapter import BaseModelAdapter
from ai_safety.adapters.yolo_adapter import YOLOAdapter
from ai_safety.core.exceptions import (
    AISafetyError,
    InvalidInputError,
    ModelLoadError,
    ModelMetadataError,
    ModelNotFoundError,
    PredictionError,
    RegistryError,
    UnsupportedModelError,
)
from ai_safety.core.lineage import ModelLineage
from ai_safety.core.model_adapter import ModelAdapter
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
from ai_safety.dataset.analyzer import DatasetAnalyzer, DatasetSummary
from ai_safety.dataset.validator import (
    DatasetStatistics,
    DatasetValidator,
    ValidationErrorItem,
    ValidationResult,
    ValidationWarningItem,
)
from ai_safety.evaluation.evaluator import (
    EvaluationMetrics,
    ModelEvaluator,
    YOLOEvaluator,
)
from ai_safety.export.exporter import ExportConfig, ExportFormat, ModelExporter
from ai_safety.inference.predictor import Predictor
from ai_safety.training.config import (
    TRAINING_PROFILES,
    TrainingConfig,
    get_training_profile,
    resolve_device,
)
from ai_safety.training.run_manager import TrainingRunHandle, TrainingRunManager
from ai_safety.training.service import TrainingResult, TrainingService
from ai_safety.training.trainer import ModelTrainer

__all__ = [
    "__version__",
    "AISafetyError",
    "ModelNotFoundError",
    "ModelMetadataError",
    "UnsupportedModelError",
    "ModelLoadError",
    "InvalidInputError",
    "PredictionError",
    "RegistryError",
    "TaskType",
    "ModelOrigin",
    "ModelStatus",
    "ExecutionDevice",
    "ModelMetadata",
    "BoundingBox",
    "Keypoint",
    "Detection",
    "PredictionResult",
    "ModelAdapter",
    "BaseModelAdapter",
    "YOLOAdapter",
    "ModelRegistry",
    "Predictor",
    "ModelLineage",
    "DatasetValidator",
    "ValidationResult",
    "ValidationErrorItem",
    "ValidationWarningItem",
    "DatasetStatistics",
    "DatasetAnalyzer",
    "DatasetSummary",
    "ModelTrainer",
    "TrainingConfig",
    "resolve_device",
    "get_training_profile",
    "TRAINING_PROFILES",
    "TrainingRunHandle",
    "TrainingRunManager",
    "TrainingService",
    "TrainingResult",
    "ModelEvaluator",
    "YOLOEvaluator",
    "EvaluationMetrics",
    "ModelExporter",
    "ExportConfig",
    "ExportFormat",
]
