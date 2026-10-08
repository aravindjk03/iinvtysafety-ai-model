"""Training Run and Artifact Manager managing run directories, logs, and artifacts."""

from datetime import datetime, timezone
import json
from pathlib import Path
import secrets
from typing import Any, Dict, List, Optional
import ultralytics

from ai_safety.core.lineage import ModelLineage
from ai_safety.dataset.analyzer import DatasetSummary
from ai_safety.training.config import TrainingConfig


class TrainingRunHandle:
    """Active handle for an initialized training run directory."""

    def __init__(self, run_dir: Path, lineage: ModelLineage):
        self.run_dir = run_dir
        self.run_id = lineage.run_id
        self.lineage = lineage
        self.weights_dir = run_dir / "weights"
        self.logs_dir = run_dir / "logs"
        self.results_dir = run_dir / "results"

    def update_status(self, status: str, output_model_id: Optional[str] = None) -> None:
        """Update and persist the run's operational status and lineage."""
        self.lineage.status = status
        if output_model_id:
            self.lineage.output_model_id = output_model_id
        lineage_file = self.run_dir / "lineage.json"
        with open(lineage_file, "w", encoding="utf-8") as f:
            f.write(self.lineage.model_dump_json(indent=2))

    def save_metrics(self, metrics: Dict[str, Any]) -> None:
        """Persist evaluation metrics dictionary to metrics.json."""
        metrics_file = self.run_dir / "metrics.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)

    def find_best_model(self) -> Optional[Path]:
        """Locate the framework's best model artifact without guessing."""
        expected_best = self.weights_dir / "best.pt"
        if expected_best.exists() and expected_best.is_file() and expected_best.stat().st_size > 0:
            return expected_best

        # Check results/ or subdirectories if framework nested them
        for candidate in self.run_dir.rglob("best.pt"):
            if candidate.is_file() and candidate.stat().st_size > 0:
                return candidate

        return None


class TrainingRunManager:
    """Manages unique training runs and their directory lifecycles."""

    def __init__(self, runs_root: Path, repo_root: Optional[Path] = None):
        self.repo_root = Path(repo_root or Path.cwd()).resolve()
        self.runs_root = Path(runs_root)
        if not self.runs_root.is_absolute():
            self.runs_root = (self.repo_root / self.runs_root).resolve()
        self.runs_root.mkdir(parents=True, exist_ok=True)

    def create_run(
        self,
        base_model: str,
        dataset_path: str,
        config: TrainingConfig,
        dataset_summary: DatasetSummary,
        base_model_version: Optional[str] = None,
    ) -> TrainingRunHandle:
        """Initialize a new unique training run directory and persist initial state."""
        # Generate deterministic safe run ID: run-YYYYMMDD-HHMMSS-<4-char-hex>
        now_str = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        suffix = secrets.token_hex(2)
        run_id = f"run-{now_str}-{suffix}"

        run_dir = self.runs_root / run_id
        weights_dir = run_dir / "weights"
        logs_dir = run_dir / "logs"
        results_dir = run_dir / "results"

        weights_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save configuration.json
        cfg_file = run_dir / "configuration.json"
        with open(cfg_file, "w", encoding="utf-8") as f:
            f.write(config.model_dump_json(indent=2))

        # 2. Save dataset_summary.json
        ds_file = run_dir / "dataset_summary.json"
        with open(ds_file, "w", encoding="utf-8") as f:
            f.write(dataset_summary.model_dump_json(indent=2))

        # 3. Create and save lineage.json
        lineage = ModelLineage(
            run_id=run_id,
            base_model_id=base_model,
            base_model_version=base_model_version,
            dataset_path=dataset_path,
            dataset_summary=dataset_summary.model_dump(),
            training_configuration=config.model_dump(),
            device=config.device,
            framework="ultralytics_yolo",
            framework_version=getattr(ultralytics, "__version__", None),
            status="created",
        )
        lineage_file = run_dir / "lineage.json"
        with open(lineage_file, "w", encoding="utf-8") as f:
            f.write(lineage.model_dump_json(indent=2))

        return TrainingRunHandle(run_dir=run_dir, lineage=lineage)

    def list_runs(self) -> List[Dict[str, Any]]:
        """List all training runs recorded on disk."""
        runs: List[Dict[str, Any]] = []
        if not self.runs_root.exists():
            return runs

        for entry in sorted(self.runs_root.iterdir(), reverse=True):
            if entry.is_dir() and entry.name.startswith("run-"):
                lineage_file = entry / "lineage.json"
                if lineage_file.exists():
                    try:
                        with open(lineage_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            runs.append(data)
                            continue
                    except Exception:
                        pass
                runs.append({
                    "run_id": entry.name,
                    "status": "unknown",
                    "path": str(entry),
                })
        return runs
