"""Canonical model metadata schema and enumerations for AI Safety Model Lab."""

from datetime import datetime, timezone
from enum import Enum
from pathlib import PurePath, PureWindowsPath, PurePosixPath
from typing import List, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TaskType(str, Enum):
    """Controlled enumeration of supported computer vision task types.

    Industrial safety use cases map directly to these primitive tasks:
      - PPE Detection -> OBJECT_DETECTION
      - Fire & Smoke Detection -> OBJECT_DETECTION
      - Danger Zone Monitoring -> OBJECT_DETECTION (with spatial logic)
      - Fall Detection -> POSE_ESTIMATION
    """
    OBJECT_DETECTION = "object_detection"
    POSE_ESTIMATION = "pose_estimation"
    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"


class ModelOrigin(str, Enum):
    """Distinguishes baseline pretrained safety models from participant-trained models."""
    PRETRAINED = "pretrained"
    CUSTOM = "custom"


class ModelStatus(str, Enum):
    """Controlled lifecycle states of a model."""
    REGISTERED = "registered"
    LOADING = "loading"
    READY = "ready"
    TRAINING = "training"
    TRAINED = "trained"
    EVALUATED = "evaluated"
    FAILED = "failed"


class ExecutionDevice(str, Enum):
    """Execution device abstraction. Defaults to CPU."""
    CPU = "cpu"
    CUDA = "cuda"


class ModelMetadata(BaseModel):
    """Canonical metadata schema for every model in the lab."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    # Required identifiers & descriptors
    model_id: str = Field(..., min_length=1, description="Unique model identifier")
    name: str = Field(..., min_length=1, description="Human-readable model name")
    version: str = Field(..., min_length=1, description="Model semantic version (e.g., '1.0.0')")
    task_type: TaskType = Field(..., description="Primitive CV task type")
    framework: str = Field(..., min_length=1, description="ML framework (e.g. 'ultralytics_yolo')")
    architecture: str = Field(..., min_length=1, description="Model architecture (e.g. 'yolo11n')")
    model_path: str = Field(..., min_length=1, description="Relative path to model weights file")
    classes: List[str] = Field(..., min_length=1, description="Class labels predicted by the model")
    input_size: Tuple[int, int] = Field((640, 640), description="Model input dimensions (width, height)")
    confidence_threshold: float = Field(0.25, ge=0.0, le=1.0, description="Default confidence threshold")
    origin: ModelOrigin = Field(..., description="Origin of the model (pretrained vs custom)")
    base_model: Optional[str] = Field(None, description="Identifier of the base model used for fine-tuning")
    training_run_id: Optional[str] = Field(None, description="Identifier of the training run for lineage")
    status: ModelStatus = Field(ModelStatus.REGISTERED, description="Current operational status")
    training_supported: bool = Field(False, description="Whether this model architecture supports fine-tuning")

    # Optional metadata
    description: Optional[str] = Field(None, description="Educational or operational model description")
    created_at: Optional[str] = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp of registration",
    )
    updated_at: Optional[str] = Field(None, description="ISO 8601 UTC timestamp of last update")
    framework_version: Optional[str] = Field(None, description="Framework library version")

    @field_validator("model_path")
    @classmethod
    def validate_relative_path(cls, path_str: str) -> str:
        """Enforce that model paths are strictly relative to repo root and prevent path traversal."""
        # Check for empty or whitespace
        cleaned = path_str.strip()
        if not cleaned:
            raise ValueError("model_path cannot be empty.")

        # Check for Windows absolute path or drive letter (e.g. C:\... or C:/...)
        win_path = PureWindowsPath(cleaned)
        posix_path = PurePosixPath(cleaned)
        if win_path.is_absolute() or posix_path.is_absolute() or win_path.drive:
            raise ValueError(f"model_path must be relative to repository root, got absolute path: {path_str}")

        # Check for path traversal components (e.g. '..')
        parts = PurePath(cleaned).parts
        if ".." in parts:
            raise ValueError(f"Path traversal detected in model_path: {path_str}")

        # Normalize to forward slashes for cross-platform portability
        return str(PurePosixPath(*parts))

    @field_validator("model_id")
    @classmethod
    def validate_model_id_safety(cls, v: str) -> str:
        """Ensure model_id is a clean identifier without path traversal characters."""
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("model_id cannot be empty.")
        if any(c in cleaned for c in ["/", "\\", "..", "\0", ":"]):
            raise ValueError(f"model_id contains invalid characters: '{v}'")
        return cleaned
