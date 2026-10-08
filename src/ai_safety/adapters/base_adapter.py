"""Generic base adapter implementing common lifecycle and validation mechanisms."""

from pathlib import Path
from typing import Any, Dict, Optional

from ai_safety.core.exceptions import (
    InvalidInputError,
    ModelLoadError,
)
from ai_safety.core.model_adapter import ModelAdapter
from ai_safety.core.model_metadata import ModelMetadata, ModelStatus


class BaseModelAdapter(ModelAdapter):
    """Generic adapter base providing lifecycle state tracking, device checks, and validation.

    Free of any domain-specific logic (no PPE, Fire, or Fall specific rules).
    """

    def __init__(self, metadata: ModelMetadata, device: str = "cpu", repo_root: Optional[Path] = None):
        super().__init__(metadata=metadata, device=device, repo_root=repo_root)
        self._model_instance: Any = None

    def validate(self) -> bool:
        """Validate that the model file exists at the declared relative path and is non-empty."""
        try:
            full_path = self.resolve_model_path()
            return full_path.exists() and full_path.is_file() and full_path.stat().st_size > 0
        except Exception:
            return False

    def _ensure_loaded(self) -> None:
        """Ensure model is loaded before executing inference or evaluation."""
        if not self._is_loaded or self._model_instance is None:
            raise ModelLoadError(
                f"Model '{self.metadata.model_id}' is not loaded. Call load_model() first."
            )

    def _validate_input_presence(self, input_data: Any) -> None:
        """Verify input data is present and non-empty."""
        if input_data is None:
            raise InvalidInputError("Inference input data cannot be None.")
        if isinstance(input_data, (str, bytes)) and len(input_data) == 0:
            raise InvalidInputError("Inference input data cannot be empty.")

    def load_model(self, device: Optional[str] = None) -> None:
        """Base load_model implementation. Subclasses override with framework specifics."""
        if device is not None:
            self.device = self._validate_device(device)
        self._ensure_loaded()

    def predict(
        self,
        input_data: Any,
        confidence_threshold: Optional[float] = None,
    ) -> Any:
        """Base predict implementation. Subclasses override with framework specifics."""
        self._validate_input_presence(input_data)
        self._ensure_loaded()
        raise NotImplementedError("Subclasses must implement predict().")

    def evaluate(self, dataset_path: str, **kwargs: Any) -> Dict[str, Any]:
        """Base evaluate implementation."""
        self._ensure_loaded()
        raise NotImplementedError("Subclasses must implement evaluate().")

    def export(self, target_format: str, output_path: str, **kwargs: Any) -> str:
        """Base export implementation."""
        self._ensure_loaded()
        raise NotImplementedError("Subclasses must implement export().")

