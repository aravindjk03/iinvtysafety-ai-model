"""Training Service orchestrating the complete lifecycle from dataset validation to model registration."""

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Tuple, Union
import yaml

from ultralytics import YOLO

from ai_safety.core.exceptions import (
    ModelLoadError,
    ModelNotFoundError,
    RegistryError,
)
from ai_safety.core.model_metadata import (
    ModelMetadata,
    ModelOrigin,
    ModelStatus,
    TaskType,
)
from ai_safety.core.model_registry import ModelRegistry
from ai_safety.dataset.analyzer import DatasetAnalyzer, DatasetSummary
from ai_safety.dataset.validator import DatasetValidator, ValidationResult
from ai_safety.evaluation.evaluator import EvaluationMetrics, YOLOEvaluator
from ai_safety.training.config import TrainingConfig, resolve_device
from ai_safety.training.run_manager import TrainingRunHandle, TrainingRunManager


logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    """Summary of a completed training workflow."""
    run_id: str
    custom_model_id: str
    version: str
    best_weights_path: Path
    custom_model_dir: Path
    metrics: EvaluationMetrics
    dataset_summary: DatasetSummary
    metadata: ModelMetadata


class TrainingService:
    """Orchestrates validation, training execution, evaluation, and custom model registration."""

    def __init__(
        self,
        registry: ModelRegistry,
        runs_dir: Union[str, Path] = "training_runs",
        custom_models_dir: Union[str, Path] = "models/custom",
        repo_root: Optional[Path] = None,
    ):
        self.registry = registry
        self.repo_root = Path(repo_root or registry.repo_root).resolve()

        runs_path = Path(runs_dir)
        self.runs_dir = runs_path if runs_path.is_absolute() else (self.repo_root / runs_path).resolve()

        custom_path = Path(custom_models_dir)
        self.custom_models_dir = custom_path if custom_path.is_absolute() else (self.repo_root / custom_path).resolve()

        self.run_manager = TrainingRunManager(runs_root=self.runs_dir, repo_root=self.repo_root)
        self.evaluator = YOLOEvaluator(registry=self.registry, repo_root=self.repo_root)

    def train(
        self,
        base_model: str,
        dataset_path: Union[str, Path],
        configuration: Union[TrainingConfig, Dict[str, Any], str],
        custom_name: Optional[str] = None,
        version: Optional[str] = None,
        custom_model_id: Optional[str] = None,
    ) -> TrainingResult:
        """Execute end-to-end training lifecycle."""
        # 1. Parse and validate training configuration
        config = self._prepare_config(base_model, str(dataset_path), configuration)
        target_device = resolve_device(config.device)
        config.device = target_device

        # 2. Resolve and validate base model weights
        base_weights_path, base_model_id, base_version = self._resolve_base_model(config.base_model)

        # 3. Validate dataset
        dataset_dir = Path(dataset_path)
        if not dataset_dir.is_absolute():
            dataset_dir = (self.repo_root / dataset_dir).resolve()

        validator = DatasetValidator(dataset_dir)
        val_result = validator.validate()
        if not val_result.valid:
            error_msgs = "\n".join(f"[{e.code}] {e.file or ''}: {e.message}" for e in val_result.errors)
            raise ValueError(f"Dataset validation failed with {len(val_result.errors)} errors:\n{error_msgs}")

        # 4. Analyze dataset
        analyzer = DatasetAnalyzer(dataset_dir)
        ds_summary = analyzer.analyze()

        # 5. Create unique training run directory and persist initial state
        run_handle = self.run_manager.create_run(
            base_model=base_model_id,
            dataset_path=str(dataset_path),
            config=config,
            dataset_summary=ds_summary,
            base_model_version=base_version,
        )

        data_yaml_file = dataset_dir / "data.yaml"

        # 6. Execute model training
        run_handle.update_status("training")
        try:
            self._execute_training(
                base_weights=base_weights_path,
                data_yaml=data_yaml_file,
                config=config,
                run_handle=run_handle,
            )
        except Exception as e:
            run_handle.update_status("failed")
            raise RuntimeError(f"Training failed: {str(e)}") from e

        # 7. Locate and verify best model artifact
        best_model_path = run_handle.find_best_model()
        if not best_model_path or not best_model_path.exists() or best_model_path.stat().st_size == 0:
            run_handle.update_status("failed")
            raise RuntimeError(
                f"Training failed to produce a valid best model artifact in {run_handle.run_dir}"
            )

        run_handle.update_status("trained")

        # 8. Evaluate best model against dataset validation split
        run_handle.update_status("evaluating")
        try:
            metrics = self.evaluator.evaluate_weights(
                weights_path=best_model_path,
                data_yaml_path=data_yaml_file,
                device=config.device,
            )
            run_handle.save_metrics(metrics.model_dump())
            run_handle.update_status("evaluated")
        except Exception as e:
            run_handle.update_status("failed")
            raise RuntimeError(f"Model evaluation failed: {str(e)}") from e

        # 9. Register Custom Model in models/custom/ and ModelRegistry
        custom_model_id, assigned_version, custom_dir, metadata = self._register_custom_model(
            custom_name=custom_name or config.project_name,
            version=version,
            best_weights=best_model_path,
            metrics=metrics,
            ds_summary=ds_summary,
            config=config,
            run_handle=run_handle,
            base_model_id=base_model_id,
            target_model_id=custom_model_id,
        )

        run_handle.update_status("ready", output_model_id=custom_model_id)

        return TrainingResult(
            run_id=run_handle.run_id,
            custom_model_id=custom_model_id,
            version=assigned_version,
            best_weights_path=best_model_path,
            custom_model_dir=custom_dir,
            metrics=metrics,
            dataset_summary=ds_summary,
            metadata=metadata,
        )

    def _prepare_config(
        self,
        base_model: str,
        dataset_path: str,
        configuration: Union[TrainingConfig, Dict[str, Any], str],
    ) -> TrainingConfig:
        """Create a validated TrainingConfig from profile, dict, or instance."""
        if isinstance(configuration, TrainingConfig):
            cfg = configuration.model_copy(update={"base_model": base_model, "dataset_path": dataset_path})
            return cfg

        if isinstance(configuration, str):
            return TrainingConfig.from_profile(
                profile_name=configuration,
                base_model=base_model,
                dataset_path=dataset_path,
            )

        if isinstance(configuration, dict):
            cfg_dict = dict(configuration)
            cfg_dict["base_model"] = base_model
            cfg_dict["dataset_path"] = dataset_path
            return TrainingConfig(**cfg_dict)

        raise ValueError(f"Invalid configuration type: {type(configuration)}")

    def _resolve_base_model(self, base_model: str) -> Tuple[str, str, Optional[str]]:
        """Resolve base model into weights path, model identifier, and version.

        Never overwrites or points training output into models/pretrained/.
        """
        # 1. Check if base_model is registered in ModelRegistry
        if self.registry.model_exists(base_model):
            meta = self.registry.get_model(base_model)
            resolved_path = (self.repo_root / meta.model_path).resolve()
            if resolved_path.exists():
                return str(resolved_path), meta.model_id, meta.version
            # If registered model file does not exist on disk, fall back to architecture name
            return meta.architecture, meta.model_id, meta.version

        # 2. Check if it's a direct local file
        local_path = Path(base_model)
        if not local_path.is_absolute():
            local_path = (self.repo_root / local_path).resolve()

        if local_path.exists() and local_path.is_file():
            return str(local_path), local_path.stem, "1.0.0"

        # 3. Known YOLO base model name (e.g. 'yolo11n.pt', 'yolo11n', 'yolov8n.pt')
        return base_model, base_model, None

    def _execute_training(
        self,
        base_weights: str,
        data_yaml: Path,
        config: TrainingConfig,
        run_handle: TrainingRunHandle,
    ) -> None:
        """Execute Ultralytics training within isolated run directory."""
        model = YOLO(base_weights)
        train_results = model.train(
            data=str(data_yaml),
            epochs=config.epochs,
            imgsz=config.image_size,
            batch=config.batch_size,
            device=config.device,
            project=str(run_handle.run_dir),
            name="results",
            exist_ok=True,
            workers=config.workers,
            patience=config.patience,
            seed=config.seed,
            plots=False,
            verbose=False,
        )

        # Ensure best.pt is copied to weights/best.pt if saved under results/weights/
        nested_best = run_handle.run_dir / "results" / "weights" / "best.pt"
        target_best = run_handle.weights_dir / "best.pt"
        if nested_best.exists() and not target_best.exists():
            shutil.copy2(nested_best, target_best)

        nested_last = run_handle.run_dir / "results" / "weights" / "last.pt"
        target_last = run_handle.weights_dir / "last.pt"
        if nested_last.exists() and not target_last.exists():
            shutil.copy2(nested_last, target_last)

    def _register_custom_model(
        self,
        custom_name: str,
        version: Optional[str],
        best_weights: Path,
        metrics: EvaluationMetrics,
        ds_summary: DatasetSummary,
        config: TrainingConfig,
        run_handle: TrainingRunHandle,
        base_model_id: str,
        target_model_id: Optional[str] = None,
    ) -> Tuple[str, str, Path, ModelMetadata]:
        """Register custom model in models/custom/ and Phase 1 ModelRegistry."""
        base_slug = custom_name.lower().replace(" ", "-").replace("_", "-")

        # Determine version and model ID
        if target_model_id:
            custom_model_id = target_model_id
            assigned_version = version.lstrip("v") if version else "1.0.0"
        else:
            if version:
                assigned_version = version.lstrip("v")
            else:
                existing_custom = [
                    m for m in self.registry.list_models(origin=ModelOrigin.CUSTOM)
                    if m.name == custom_name or m.model_id.startswith(base_slug)
                ]
                assigned_version = f"1.0.{len(existing_custom)}" if existing_custom else "1.0.0"
            custom_model_id = f"{base_slug}-v{assigned_version.replace('.', '-')}"

        # Prevent duplicate version overwrites
        if self.registry.model_exists(custom_model_id):
            raise RegistryError(
                f"Custom model '{custom_model_id}' (version {assigned_version}) already exists in registry. "
                f"Duplicate version overwrite is prevented."
            )

        # Destination: models/custom/<custom_model_id>/
        custom_dir = self.custom_models_dir / custom_model_id
        custom_dir.mkdir(parents=True, exist_ok=True)

        # 1. Copy model weights as model.pt
        target_model_file = custom_dir / "model.pt"
        shutil.copy2(best_weights, target_model_file)

        # Relative path from repo root
        rel_model_path = str(target_model_file.relative_to(self.repo_root)).replace("\\", "/")

        # Determine task type
        model_task_type = TaskType.OBJECT_DETECTION
        if getattr(config, "task_type", None) is not None:
            model_task_type = config.task_type
        elif self.registry.model_exists(base_model_id):
            model_task_type = self.registry.get_model(base_model_id).task_type

        # 2. Construct canonical metadata
        class_list = [cname for cname in ds_summary.class_names.values()]
        metadata = ModelMetadata(
            model_id=custom_model_id,
            name=custom_name,
            version=assigned_version,
            task_type=model_task_type,
            framework="ultralytics_yolo",
            architecture=config.base_model,
            model_path=rel_model_path,
            classes=class_list,
            input_size=(config.image_size, config.image_size),
            confidence_threshold=0.25,
            origin=ModelOrigin.CUSTOM,
            base_model=base_model_id,
            training_run_id=run_handle.run_id,
            status=ModelStatus.READY,
            training_supported=True,
            description=f"Participant model fine-tuned from {base_model_id}",
        )

        # 3. Write metadata.json
        meta_file = custom_dir / "metadata.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            f.write(metadata.model_dump_json(indent=2))

        # 4. Write metrics.json
        metrics_file = custom_dir / "metrics.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            f.write(metrics.model_dump_json(indent=2))

        # 5. Write lineage.json
        lineage_file = custom_dir / "lineage.json"
        with open(lineage_file, "w", encoding="utf-8") as f:
            f.write(run_handle.lineage.model_dump_json(indent=2))

        # 6. Register model in Phase 1 ModelRegistry
        self.registry.register_model(metadata, overwrite=False)

        return custom_model_id, assigned_version, custom_dir, metadata

    def export_native(self, model_id: str, export_output_dir: Union[str, Path]) -> Path:
        """Export native model bundle: model.pt + metadata.json + metrics.json + lineage.json."""
        meta = self.registry.get_model(model_id)
        source_model_file = (self.repo_root / meta.model_path).resolve()
        source_dir = source_model_file.parent

        dest_dir = Path(export_output_dir)
        if not dest_dir.is_absolute():
            dest_dir = (self.repo_root / dest_dir).resolve()
        dest_dir.mkdir(parents=True, exist_ok=True)

        target_bundle = dest_dir / f"{model_id}-export"
        target_bundle.mkdir(parents=True, exist_ok=True)

        shutil.copy2(source_model_file, target_bundle / "model.pt")

        for json_name in ["metadata.json", "metrics.json", "lineage.json"]:
            src_json = source_dir / json_name
            if src_json.exists():
                shutil.copy2(src_json, target_bundle / json_name)
            elif json_name == "metadata.json":
                with open(target_bundle / "metadata.json", "w", encoding="utf-8") as f:
                    f.write(meta.model_dump_json(indent=2))

        return target_bundle
