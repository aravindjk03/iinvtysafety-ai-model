"""Inference service orchestrating model loading, adapter selection, and normalized prediction execution."""

from pathlib import Path
from typing import Any, Dict, Optional

from ai_safety.adapters.yolo_adapter import YOLOAdapter
from ai_safety.core.exceptions import (
    ModelLoadError,
    ModelNotFoundError,
    UnsupportedModelError,
)
from ai_safety.core.model_adapter import ModelAdapter
from ai_safety.core.model_metadata import ExecutionDevice, ModelMetadata
from ai_safety.core.model_registry import ModelRegistry
from ai_safety.core.prediction import PredictionResult


class Predictor:
    """Central inference service consumed by downstream clients (e.g. Repository 1).

    Orchestrates metadata lookup, adapter instantiation, device configuration,
    and prediction execution. Decoupled from all domain-specific safety logic.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        default_device: str = "cpu",
        repo_root: Optional[Path] = None,
    ):
        """Initialize Predictor with a model registry reference.

        Args:
            registry: Configured ModelRegistry instance.
            default_device: Default execution device ('cpu' or 'cuda').
            repo_root: Repository root path for resolving artifacts.
        """
        self.registry = registry
        self.default_device = default_device.lower().strip()
        if self.default_device not in [d.value for d in ExecutionDevice]:
            raise ValueError(f"Invalid default device '{default_device}'. Must be 'cpu' or 'cuda'.")
        self.repo_root = repo_root or registry.repo_root
        self._adapter_cache: Dict[str, ModelAdapter] = {}

    def _create_adapter(self, metadata: ModelMetadata, device: str) -> ModelAdapter:
        """Instantiate the appropriate adapter implementation based on model metadata."""
        framework = metadata.framework.lower()
        if framework in YOLOAdapter.SUPPORTED_FRAMEWORKS:
            return YOLOAdapter(metadata=metadata, device=device, repo_root=self.repo_root)

        raise UnsupportedModelError(
            f"No adapter available for framework '{metadata.framework}' in model '{metadata.model_id}'."
        )

    def get_adapter(self, model_id: str, device: Optional[str] = None) -> ModelAdapter:
        """Retrieve or instantiate an adapter for the specified model ID."""
        target_device = device.lower().strip() if device else self.default_device
        cache_key = f"{model_id}::{target_device}"

        if cache_key in self._adapter_cache:
            return self._adapter_cache[cache_key]

        metadata = self.registry.get_model(model_id)
        adapter = self._create_adapter(metadata, device=target_device)
        self._adapter_cache[cache_key] = adapter
        return adapter

    def predict(
        self,
        model_id: str,
        input_data: Any,
        confidence_threshold: Optional[float] = None,
        device: Optional[str] = None,
    ) -> PredictionResult:
        """Execute normalized inference for a registered model.

        Downstream systems (Repository 1) need only call this method with
        the model_id and input.

        Args:
            model_id: Registered model identifier.
            input_data: Input representation (image file path, buffer, or array).
            confidence_threshold: Optional threshold override.
            device: Optional hardware device override.

        Returns:
            Normalized PredictionResult schema.

        Raises:
            ModelNotFoundError: If model_id is not registered.
            ModelLoadError: If model cannot be loaded.
            PredictionError: If inference fails.
        """
        target_device = device.lower().strip() if device else self.default_device
        adapter = self.get_adapter(model_id, device=target_device)

        if not adapter.is_loaded:
            adapter.load_model(device=target_device)

        return adapter.predict(input_data=input_data, confidence_threshold=confidence_threshold)
