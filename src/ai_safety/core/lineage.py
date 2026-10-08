"""Model Lineage tracking structure answering full provenance for trained models."""

from datetime import datetime, timezone
import platform
import sys
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class ModelLineage(BaseModel):
    """Complete provenance record for a training run and resulting custom model.

    Answers:
      - What model did this custom model come from? (base_model_id, base_model_version)
      - What dataset was used? (dataset_path, dataset_summary)
      - What training configuration was used? (training_configuration)
      - Was CPU or GPU used? (device)
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(..., description="Unique training run identifier")
    base_model_id: str = Field(..., description="Base model identifier or weights path")
    base_model_version: Optional[str] = Field(None, description="Semantic version of base model")
    dataset_path: str = Field(..., description="Dataset directory path")
    dataset_summary: Dict[str, Any] = Field(..., description="Dataset quantitative summary snapshot")
    training_configuration: Dict[str, Any] = Field(..., description="Serialized TrainingConfig snapshot")
    device: str = Field(..., description="Hardware device used for training ('cpu' or 'cuda')")
    framework: str = Field("ultralytics_yolo", description="ML framework used")
    framework_version: Optional[str] = Field(None, description="Framework version")
    python_version: str = Field(
        default_factory=lambda: f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} ({platform.system()})",
        description="Python runtime and OS platform",
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="UTC timestamp of run initialization",
    )
    output_model_id: Optional[str] = Field(None, description="Registered custom model ID if successful")
    status: str = Field("created", description="Lifecycle status of the training run")
