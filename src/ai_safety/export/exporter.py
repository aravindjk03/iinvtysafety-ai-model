"""Model export abstraction defining supported target formats and serialization contract."""

from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from ai_safety.core.model_registry import ModelRegistry


class ExportFormat(str, Enum):
    """Controlled enumeration of supported export formats for downstream deployment."""
    ONNX = "onnx"
    TORCHSCRIPT = "torchscript"
    TFLITE = "tflite"
    ENGINE = "engine"


class ExportConfig(BaseModel):
    """Configuration options for model format export."""

    model_config = ConfigDict(extra="forbid")

    format: ExportFormat = Field(..., description="Target export runtime format")
    half: bool = Field(False, description="FP16 half-precision quantization")
    opset: int = Field(12, ge=7, le=20, description="ONNX opset version")
    dynamic: bool = Field(False, description="Enable dynamic input shape batching")


class ModelExporter(ABC):
    """Abstract model exporter contract for Phase 2 deployment packaging."""

    def __init__(self, registry: ModelRegistry, repo_root: Optional[Path] = None):
        self.registry = registry
        self.repo_root = repo_root or registry.repo_root

    @abstractmethod
    def export(
        self,
        model_id: str,
        format: ExportFormat,
        output_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Export model weights to target runtime format and return exported path.

        Raises:
            NotImplementedError: Always raised in Phase 1; export execution is deferred to Phase 2.
        """
        raise NotImplementedError(
            "Model export execution is deferred to Phase 2. "
            "Phase 1 establishes the export interfaces and format enums only."
        )
