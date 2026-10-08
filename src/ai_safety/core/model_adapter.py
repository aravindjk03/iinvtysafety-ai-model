"""Abstract ModelAdapter base contract for all model implementations in AI Safety Model Lab."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

from ai_safety.core.model_metadata import ExecutionDevice, ModelMetadata
from ai_safety.core.prediction import PredictionResult


class ModelAdapter(ABC):
    """Abstract interface for wrapping model frameworks (e.g. YOLO, PyTorch).

    Decouples core inference, evaluation, export, and training pipelines from
    specific underlying library implementations. Contains zero application-specific
    logic (no PPE, Fire, Fall, or Danger Zone hardcoded rules).
    """

    def __init__(self, metadata: ModelMetadata, device: str = "cpu", repo_root: Optional[Path] = None):
        """Initialize adapter with metadata and target hardware device.

        Args:
            metadata: Canonical model metadata.
            device: Target execution device ('cpu' or 'cuda'). Defaults to 'cpu'.
            repo_root: Root path of the repository for resolving relative paths safely.
        """
        self.metadata = metadata
        self.device = self._validate_device(device)
        self.repo_root = repo_root or Path.cwd()
        self._is_loaded = False

    @staticmethod
    def _validate_device(device: str) -> str:
        """Validate execution device selection, defaulting to CPU."""
        device_lower = device.lower().strip()
        allowed = [d.value for d in ExecutionDevice]
        if device_lower not in allowed:
            raise ValueError(f"Invalid device '{device}'. Supported devices are: {allowed}")
        return device_lower

    @property
    def is_loaded(self) -> bool:
        """Check if model weights are loaded and ready in memory."""
        return self._is_loaded

    def resolve_model_path(self) -> Path:
        """Safely resolve relative model_path against repo_root, ensuring no traversal."""
        full_path = (self.repo_root / self.metadata.model_path).resolve()
        # Security check: resolved path must stay within repo_root
        try:
            full_path.relative_to(self.repo_root.resolve())
        except ValueError:
            raise ValueError(f"Model path resolved outside repository root: {full_path}")
        return full_path

    @abstractmethod
    def load_model(self, device: Optional[str] = None) -> None:
        """Load model weights into memory on the target device.

        Args:
            device: Optional device override ('cpu' or 'cuda').

        Raises:
            ModelLoadError: If model weights are missing, corrupted, or unsupported.
        """
        pass

    @abstractmethod
    def predict(
        self,
        input_data: Any,
        confidence_threshold: Optional[float] = None,
    ) -> PredictionResult:
        """Perform normalized inference on given input.

        Args:
            input_data: Input representation (e.g. image path, ndarray, or bytes).
            confidence_threshold: Optional override for default confidence threshold.

        Returns:
            Normalized PredictionResult conforming to the common contract.

        Raises:
            ModelLoadError: If model has not been loaded before predict.
            InvalidInputError: If input data is malformed.
            PredictionError: If inference execution fails.
        """
        pass

    @abstractmethod
    def validate(self) -> bool:
        """Verify model weights and metadata integrity without running full benchmark.

        Returns:
            True if model file exists and is valid.
        """
        pass

    @abstractmethod
    def evaluate(self, dataset_path: str, **kwargs: Any) -> Dict[str, Any]:
        """Evaluate model against a test dataset and produce metrics.

        Args:
            dataset_path: Path to dataset descriptor (e.g. data.yaml).

        Returns:
            Dictionary of metrics (e.g. mAP50, precision, recall).
        """
        pass

    @abstractmethod
    def export(self, target_format: str, output_path: str, **kwargs: Any) -> str:
        """Export model to target runtime format (e.g. ONNX, TorchScript).

        Args:
            target_format: Target format identifier.
            output_path: Destination path for exported artifact.

        Returns:
            Path to exported artifact.
        """
        pass

    def train(self, dataset_path: str, config: Dict[str, Any], output_dir: str) -> None:
        """Future training contract hook.

        Allows training pipeline integration in Phase 2 without breaking the inference interface.

        Args:
            dataset_path: Path to dataset configuration.
            config: Training hyperparameters.
            output_dir: Directory where fine-tuned model artifacts are saved.
        """
        raise NotImplementedError("Training execution is deferred to Phase 2.")
