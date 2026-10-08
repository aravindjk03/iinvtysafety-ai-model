"""Training Configuration, Named Profiles, and Hardware Device Resolution."""

from pathlib import Path
from typing import Any, Dict, Optional
import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ai_safety.core.model_metadata import TaskType


SUPPORTED_DEVICES = {"cpu", "cuda", "auto"}


def resolve_device(device_request: str) -> str:
    """Resolve and validate the execution hardware device.

    Rules:
      - If 'cpu' is selected: use 'cpu'.
      - If 'cuda' is selected but unavailable: FAIL CLEARLY. Do NOT silently fallback to CPU.
      - If 'auto' is selected: use 'cuda' if safely available, otherwise 'cpu'.
      - Default is 'cpu'.
    """
    cleaned = device_request.lower().strip()
    if cleaned not in SUPPORTED_DEVICES:
        raise ValueError(
            f"Unsupported device '{device_request}'. Supported options: {sorted(list(SUPPORTED_DEVICES))}"
        )

    if cleaned == "cpu":
        return "cpu"

    if cleaned == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was explicitly requested, but CUDA is not available on this system. "
                "Failing execution clearly. (Silent fallback to CPU is disallowed by safety specification)."
            )
        return "cuda"

    # auto
    return "cuda" if torch.cuda.is_available() else "cpu"


# Named training profiles
TRAINING_PROFILES: Dict[str, Dict[str, Any]] = {
    "workshop_cpu": {
        "epochs": 10,
        "image_size": 320,
        "batch_size": 4,
        "device": "cpu",
        "workers": 0,
        "patience": 5,
        "seed": 42,
    }
}


def get_training_profile(name: str) -> Dict[str, Any]:
    """Retrieve hyperparameters for a named profile."""
    profile_key = name.lower().strip()
    if profile_key not in TRAINING_PROFILES:
        raise ValueError(
            f"Unknown training profile '{name}'. Available profiles: {list(TRAINING_PROFILES.keys())}"
        )
    return dict(TRAINING_PROFILES[profile_key])


class TrainingConfig(BaseModel):
    """Strongly validated configuration for model fine-tuning runs."""

    model_config = ConfigDict(extra="forbid")

    # Primary fields
    base_model: Optional[str] = Field(None, description="Base model identifier or weights path")
    dataset_path: Optional[str] = Field(None, description="Relative path to dataset directory containing data.yaml")
    epochs: int = Field(50, gt=0, le=1000, description="Total number of training epochs (must be > 0)")
    image_size: int = Field(640, gt=0, le=2048, description="Input image size in pixels (must be > 0)")
    imgsz: Optional[int] = Field(None, description="Alias for image_size for backward compatibility")
    batch_size: int = Field(16, gt=0, le=256, description="Batch size for training (must be > 0)")
    device: str = Field("cpu", description="Execution device: 'cpu', 'cuda', or 'auto'")
    output_directory: str = Field("training_runs", min_length=1, description="Root directory where training runs are saved")
    project_name: str = Field("safety_model_run", min_length=1, description="Project name for organizational tagging")
    task_type: Optional[TaskType] = Field(None, description="Task type of the model (e.g. object_detection, pose_estimation)")

    # Hyperparameters & runtime settings
    learning_rate: float = Field(0.01, gt=0.0, le=1.0, description="Initial learning rate")
    optimizer: str = Field("auto", description="Optimizer choice")
    workers: int = Field(0, ge=0, description="Dataloader worker processes (0 is safest on Windows CPU)")
    patience: int = Field(10, ge=1, description="Early stopping patience epochs")
    seed: int = Field(42, description="Random seed for reproducibility")

    def model_post_init(self, __context: Any) -> None:
        if self.imgsz is not None and self.image_size == 640:
            self.image_size = self.imgsz

    @field_validator("device")
    @classmethod
    def validate_device_field(cls, v: str) -> str:
        cleaned = v.lower().strip()
        if cleaned not in SUPPORTED_DEVICES:
            raise ValueError(f"Invalid device '{v}'. Must be one of: {sorted(list(SUPPORTED_DEVICES))}")
        return cleaned

    @classmethod
    def from_profile(
        cls,
        profile_name: str,
        base_model: str,
        dataset_path: str,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> "TrainingConfig":
        """Instantiate TrainingConfig starting from a named profile."""
        profile_kwargs = get_training_profile(profile_name)
        profile_kwargs["base_model"] = base_model
        profile_kwargs["dataset_path"] = dataset_path
        if overrides:
            profile_kwargs.update(overrides)
        return cls(**profile_kwargs)

