"""Evaluation engine implementing concrete YOLO benchmarking and metrics extraction."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional
import ultralytics
from ultralytics import YOLO

from pydantic import BaseModel, ConfigDict, Field

from ai_safety.core.model_registry import ModelRegistry


class EvaluationMetrics(BaseModel):
    """Normalized evaluation metrics produced during model benchmarking."""

    model_config = ConfigDict(extra="forbid")

    precision: float = Field(..., ge=0.0, le=1.0, description="Mean Precision score")
    recall: float = Field(..., ge=0.0, le=1.0, description="Mean Recall score")
    map50: float = Field(..., ge=0.0, le=1.0, description="Mean Average Precision at IoU=0.50 (mAP50)")
    map50_95: float = Field(..., ge=0.0, le=1.0, description="Mean Average Precision at IoU=0.50:0.95 (mAP50-95)")
    fitness: Optional[float] = Field(None, ge=0.0, description="Overall fitness weighted score")
    per_class_metrics: Optional[Dict[str, Dict[str, float]]] = Field(
        default=None,
        description="Granular metric breakdowns per safety class",
    )


class ModelEvaluator(ABC):
    """Abstract model evaluator interface defining the contract for model validation."""

    def __init__(self, registry: Optional[ModelRegistry] = None, repo_root: Optional[Path] = None):
        self.registry = registry
        self.repo_root = repo_root or (registry.repo_root if registry else Path.cwd())

    @abstractmethod
    def evaluate(
        self,
        model_id: str,
        dataset_path: str,
        **kwargs: Any,
    ) -> EvaluationMetrics:
        """Benchmark a model against a dataset and return normalized metrics."""
        raise NotImplementedError(
            "Model evaluation execution is deferred to Phase 2 (or use YOLOEvaluator). "
            "Subclasses must implement evaluate()."
        )


class YOLOEvaluator(ModelEvaluator):
    """Concrete evaluator wrapping Ultralytics YOLO validation mode for object detection."""

    def evaluate(
        self,
        model_id: str,
        dataset_path: str,
        device: str = "cpu",
        **kwargs: Any,
    ) -> EvaluationMetrics:
        """Evaluate a registered model against a dataset."""
        if not self.registry:
            raise ValueError("Registry is required to evaluate a model by model_id.")

        metadata = self.registry.get_model(model_id)
        weights_path = (self.repo_root / metadata.model_path).resolve()
        dataset_yaml = Path(dataset_path)
        if not dataset_yaml.is_absolute():
            dataset_yaml = (self.repo_root / dataset_yaml).resolve()

        if dataset_yaml.is_dir():
            dataset_yaml = dataset_yaml / "data.yaml"

        return self.evaluate_weights(
            weights_path=weights_path,
            data_yaml_path=dataset_yaml,
            device=device,
        )

    def evaluate_weights(
        self,
        weights_path: Path,
        data_yaml_path: Path,
        device: str = "cpu",
    ) -> EvaluationMetrics:
        """Run validation on model weights file against a YOLO data.yaml configuration."""
        if not weights_path.exists():
            raise FileNotFoundError(f"Model weights not found at {weights_path}")
        if not data_yaml_path.exists():
            raise FileNotFoundError(f"data.yaml not found at {data_yaml_path}")

        try:
            model = YOLO(str(weights_path))
            results = model.val(
                data=str(data_yaml_path),
                device=device,
                verbose=False,
                plots=False,
            )

            # Extract actual metrics calculated by Ultralytics
            mp = float(getattr(results.box, "mp", 0.0))
            mr = float(getattr(results.box, "mr", 0.0))
            map50 = float(getattr(results.box, "map50", 0.0))
            map95 = float(getattr(results.box, "map", 0.0))
            fitness = float(getattr(results, "fitness", 0.0))

            # Clamp float precision within [0.0, 1.0]
            precision = max(0.0, min(1.0, mp))
            recall = max(0.0, min(1.0, mr))
            map50 = max(0.0, min(1.0, map50))
            map50_95 = max(0.0, min(1.0, map95))

            # Extract optional per-class metrics
            per_class_metrics = None
            if hasattr(results, "names") and isinstance(results.names, dict):
                per_class = {}
                p_array = getattr(results.box, "p", None)
                r_array = getattr(results.box, "r", None)
                maps_array = getattr(results.box, "maps", None)
                for idx, cname in results.names.items():
                    class_data = {}
                    if p_array is not None and len(p_array) > idx:
                        class_data["precision"] = round(float(p_array[idx]), 4)
                    if r_array is not None and len(r_array) > idx:
                        class_data["recall"] = round(float(r_array[idx]), 4)
                    if maps_array is not None and len(maps_array) > idx:
                        class_data["map50_95"] = round(float(maps_array[idx]), 4)
                    if class_data:
                        per_class[str(cname)] = class_data
                if per_class:
                    per_class_metrics = per_class

            return EvaluationMetrics(
                precision=precision,
                recall=recall,
                map50=map50,
                map50_95=map50_95,
                fitness=fitness,
                per_class_metrics=per_class_metrics,
            )
        except Exception as e:
            raise RuntimeError(f"YOLO evaluation failed: {str(e)}") from e
