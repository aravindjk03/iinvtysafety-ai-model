"""Normalized prediction data structures for inference output across all CV tasks."""

from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ai_safety.core.model_metadata import TaskType


class BoundingBox(BaseModel):
    """Axis-aligned bounding box coordinates (normalized or pixel-space)."""

    model_config = ConfigDict(extra="forbid")

    x1: float = Field(..., description="Top-left x coordinate")
    y1: float = Field(..., description="Top-left y coordinate")
    x2: float = Field(..., description="Bottom-right x coordinate")
    y2: float = Field(..., description="Bottom-right y coordinate")

    @model_validator(mode="after")
    def validate_coordinates(self) -> "BoundingBox":
        if self.x1 > self.x2:
            raise ValueError(f"Invalid bounding box: x1 ({self.x1}) cannot be greater than x2 ({self.x2})")
        if self.y1 > self.y2:
            raise ValueError(f"Invalid bounding box: y1 ({self.y1}) cannot be greater than y2 ({self.y2})")
        return self

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height


class Keypoint(BaseModel):
    """Single body joint/keypoint coordinate for pose estimation (e.g. Fall detection)."""

    model_config = ConfigDict(extra="forbid")

    id: Optional[int] = Field(None, description="Keypoint index (e.g. COCO 0-16)")
    name: Optional[str] = Field(None, description="Keypoint anatomical label (e.g. 'left_shoulder', 'nose')")
    x: float = Field(..., description="X coordinate")
    y: float = Field(..., description="Y coordinate")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Detection confidence score")


class Detection(BaseModel):
    """Normalized single object or pose detection item."""

    model_config = ConfigDict(extra="forbid")

    class_id: int = Field(..., ge=0, description="Integer class index")
    class_name: str = Field(..., min_length=1, description="String label of detected class")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")
    bounding_box: BoundingBox = Field(..., description="Spatial bounding box")
    keypoints: Optional[List[Keypoint]] = Field(
        default=None,
        description="Optional skeletal keypoints for pose estimation without altering object detection",
    )


class PredictionResult(BaseModel):
    """Normalized response container returned from all model inference calls to Repository 1."""

    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(..., min_length=1, description="ID of the model producing this prediction")
    task_type: TaskType = Field(..., description="Computer vision task type performed")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp of inference completion",
    )
    processing_time_ms: float = Field(..., ge=0.0, description="Total inference latency in milliseconds")
    detections: List[Detection] = Field(
        default_factory=list,
        description="List of detected objects or poses",
    )
