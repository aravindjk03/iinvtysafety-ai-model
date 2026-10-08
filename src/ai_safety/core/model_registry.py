"""Local JSON-backed Model Registry managing model metadata and enforcing immutability of pretrained models."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

from ai_safety.core.exceptions import ModelNotFoundError, RegistryError
from ai_safety.core.model_metadata import ModelMetadata, ModelOrigin, TaskType


class ModelRegistry:
    """Manages metadata for pretrained and participant-trained models.

    Ensures that baseline pretrained models cannot be overwritten or altered,
    handles local-first persistence via registry.json, and normalizes paths.
    """

    def __init__(
        self,
        registry_file: Union[str, Path] = "models/registry.json",
        repo_root: Optional[Path] = None,
    ):
        """Initialize registry from local JSON file.

        Args:
            registry_file: Relative or absolute path to registry JSON file.
            repo_root: Repository root directory. If None, uses current working directory.
        """
        self.repo_root = Path(repo_root or Path.cwd()).resolve()
        reg_path = Path(registry_file)
        if not reg_path.is_absolute():
            reg_path = (self.repo_root / reg_path).resolve()
        self.registry_file = reg_path
        self._models: Dict[str, ModelMetadata] = {}
        self._load()

    def _load(self) -> None:
        """Load registry from JSON file if present."""
        if not self.registry_file.exists():
            self.registry_file.parent.mkdir(parents=True, exist_ok=True)
            self._save()
            return

        try:
            with open(self.registry_file, "r", encoding="utf-8") as f:
                content = json.load(f)
        except json.JSONDecodeError as e:
            raise RegistryError(f"Corrupted registry file '{self.registry_file}': {str(e)}") from e

        models_list = content.get("models", [])
        self._models = {}
        for item in models_list:
            metadata = ModelMetadata.model_validate(item)
            self._models[metadata.model_id] = metadata

    def _save(self) -> None:
        """Persist in-memory models to JSON file atomically."""
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "models": [model.model_dump() for model in self._models.values()]
        }
        temp_file = self.registry_file.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        temp_file.replace(self.registry_file)

    def register_model(self, metadata: ModelMetadata, overwrite: bool = False) -> None:
        """Register a new model in the registry.

        Args:
            metadata: Canonical model metadata.
            overwrite: Whether to overwrite existing custom models. Pretrained models can never be overwritten.

        Raises:
            RegistryError: If model already exists and overwrite is False, or if target is pretrained.
        """
        if self.model_exists(metadata.model_id):
            existing = self._models[metadata.model_id]
            if existing.origin == ModelOrigin.PRETRAINED:
                raise RegistryError(
                    f"Cannot overwrite protected pretrained model '{metadata.model_id}'."
                )
            if not overwrite:
                raise RegistryError(
                    f"Model '{metadata.model_id}' is already registered. Set overwrite=True for custom models."
                )

        self._models[metadata.model_id] = metadata
        self._save()

    def get_model(self, model_id: str) -> ModelMetadata:
        """Retrieve model metadata by ID.

        Args:
            model_id: Unique model identifier.

        Returns:
            ModelMetadata object.

        Raises:
            ModelNotFoundError: If model_id is not in registry.
        """
        if model_id not in self._models:
            raise ModelNotFoundError(f"Model '{model_id}' not found in registry.")
        return self._models[model_id]

    def model_exists(self, model_id: str) -> bool:
        """Check whether a model exists in the registry."""
        return model_id in self._models

    def list_models(
        self,
        origin: Optional[Union[str, ModelOrigin]] = None,
        task_type: Optional[Union[str, TaskType]] = None,
    ) -> List[ModelMetadata]:
        """List registered models with optional filtering.

        Args:
            origin: Filter by 'pretrained' or 'custom'.
            task_type: Filter by CV task type.

        Returns:
            List of matching ModelMetadata objects.
        """
        results = list(self._models.values())
        if origin is not None:
            origin_val = origin.value if isinstance(origin, ModelOrigin) else str(origin)
            results = [m for m in results if m.origin.value == origin_val]

        if task_type is not None:
            task_val = task_type.value if isinstance(task_type, TaskType) else str(task_type)
            results = [m for m in results if m.task_type.value == task_val]

        return results

    def update_model(self, metadata: ModelMetadata) -> None:
        """Update an existing custom model's metadata.

        Args:
            metadata: Updated model metadata.

        Raises:
            ModelNotFoundError: If model does not exist.
            RegistryError: If attempting to update a protected pretrained model.
        """
        if metadata.model_id not in self._models:
            raise ModelNotFoundError(f"Model '{metadata.model_id}' not found in registry.")

        existing = self._models[metadata.model_id]
        if existing.origin == ModelOrigin.PRETRAINED:
            raise RegistryError(
                f"Cannot update protected pretrained model '{metadata.model_id}'."
            )

        self._models[metadata.model_id] = metadata
        self._save()
