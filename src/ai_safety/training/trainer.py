"""Training abstraction defining required configuration and lifecycle for future model fine-tuning."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union
import uuid

from ai_safety.training.config import TrainingConfig


class ModelTrainer(ABC):
    """Abstract trainer defining the contract for future model fine-tuning (Phase 2).

    Guarantees that all training operations specify:
      1. Base model identifier or path
      2. Dataset descriptor path
      3. Validated training configuration
      4. Target output directory
    """

    def __init__(
        self,
        base_model: str,
        dataset: str,
        training_configuration: Optional[Union[TrainingConfig, Dict[str, Any]]] = None,
        output_directory: str = "models/custom",
        repo_root: Optional[Path] = None,
    ):
        """Initialize trainer abstraction.

        Args:
            base_model: Registered base model ID or path.
            dataset: Relative path to dataset descriptor (e.g. data.yaml).
            training_configuration: Hyperparameters configuration.
            output_directory: Relative path where trained artifacts will be saved.
            repo_root: Repository root path.
        """
        if not base_model or not base_model.strip():
            raise ValueError("base_model cannot be empty.")
        if not dataset or not dataset.strip():
            raise ValueError("dataset path cannot be empty.")
        if not output_directory or not output_directory.strip():
            raise ValueError("output_directory cannot be empty.")

        self.base_model = base_model.strip()
        self.dataset = dataset.strip()
        self.output_directory = output_directory.strip()
        self.repo_root = repo_root or Path.cwd()

        if isinstance(training_configuration, dict):
            self.config = TrainingConfig(**training_configuration)
        elif isinstance(training_configuration, TrainingConfig):
            self.config = training_configuration
        else:
            self.config = TrainingConfig()

        self.run_id = f"train_{uuid.uuid4().hex[:8]}"

    @abstractmethod
    def train(self) -> str:
        """Execute model training and return the newly generated model ID or artifact path.

        Raises:
            NotImplementedError: Always raised in Phase 1; training will be implemented in Phase 2.
        """
        raise NotImplementedError(
            "Model training execution is deferred to Phase 2. "
            "Phase 1 establishes the architectural contract only."
        )
