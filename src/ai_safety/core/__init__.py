"""Core abstractions, schemas, registry, and exception definitions."""

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

__all__ = [
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
    "ModelRegistry",
]
