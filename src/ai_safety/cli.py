"""Command-Line Interface (CLI) for AI Safety Model Lab using standard library argparse."""

import argparse
from pathlib import Path
import sys
from typing import List, Optional

from ai_safety.core.model_metadata import ModelOrigin, TaskType
from ai_safety.core.model_registry import ModelRegistry
from ai_safety.dataset.analyzer import DatasetAnalyzer
from ai_safety.dataset.validator import DatasetValidator
from ai_safety.training.config import TrainingConfig
from ai_safety.training.run_manager import TrainingRunManager
from ai_safety.training.service import TrainingService


def build_parser() -> argparse.ArgumentParser:
    """Build command line argument parser."""
    parser = argparse.ArgumentParser(
        prog="ai_safety",
        description="AI Safety Model Lab - Model Management, Dataset Validation, and Training Engine",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # 1. list-models
    list_p = subparsers.add_parser("list-models", help="List models registered in the lab")
    list_p.add_argument(
        "--origin",
        choices=["pretrained", "custom"],
        default=None,
        help="Filter models by origin ('pretrained' or 'custom')",
    )
    list_p.add_argument(
        "--task-type",
        choices=["object_detection", "pose_estimation", "classification", "segmentation"],
        default=None,
        help="Filter models by computer vision task type",
    )

    # 2. validate-dataset
    val_p = subparsers.add_parser("validate-dataset", help="Validate a YOLO dataset structure and labels")
    val_p.add_argument("dataset_path", help="Path to dataset directory containing data.yaml")

    # 3. dataset-summary
    sum_p = subparsers.add_parser("dataset-summary", help="Analyze and summarize class distributions in a dataset")
    sum_p.add_argument("dataset_path", help="Path to dataset directory containing data.yaml")

    # 4. train
    train_p = subparsers.add_parser("train", help="Train / fine-tune a model on a dataset")
    train_p.add_argument("--model", required=True, help="Base model identifier or weights path")
    train_p.add_argument("--dataset", required=True, help="Path to dataset directory containing data.yaml")
    train_p.add_argument("--profile", default="workshop_cpu", help="Named training profile (default: workshop_cpu)")
    train_p.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    train_p.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    train_p.add_argument("--image-size", type=int, default=None, help="Override input image size (pixels)")
    train_p.add_argument("--device", choices=["cpu", "cuda", "auto"], default=None, help="Target device (default: cpu)")
    train_p.add_argument("--name", default=None, help="Custom model name for registry")
    train_p.add_argument("--version", default=None, help="Custom model semantic version (e.g. 1.0.0)")

    # 5. list-training-runs
    subparsers.add_parser("list-training-runs", help="List historical training runs and their statuses")

    return parser


def handle_list_models(args: argparse.Namespace, repo_root: Path) -> int:
    registry_file = repo_root / "models" / "registry.json"
    registry = ModelRegistry(registry_file=registry_file, repo_root=repo_root)

    origin_enum = ModelOrigin(args.origin) if args.origin else None
    task_enum = TaskType(args.task_type) if args.task_type else None

    models = registry.list_models(origin=origin_enum, task_type=task_enum)
    print(f"\nRegistered Models ({len(models)} found):")
    print("-" * 85)
    print(f"{'Model ID':<28} {'Origin':<12} {'Task':<18} {'Status':<10} {'Version':<8}")
    print("-" * 85)
    for m in models:
        print(f"{m.model_id:<28} {m.origin.value:<12} {m.task_type.value:<18} {m.status.value:<10} {m.version:<8}")
    print("-" * 85 + "\n")
    return 0


def handle_validate_dataset(args: argparse.Namespace, repo_root: Path) -> int:
    ds_path = Path(args.dataset_path)
    if not ds_path.is_absolute():
        ds_path = (repo_root / ds_path).resolve()

    print(f"\nValidating dataset at: {ds_path}...")
    validator = DatasetValidator(ds_path)
    result = validator.validate()

    if result.valid:
        print("\n[SUCCESS] Dataset is VALID.")
        print(f"  Train Images:       {result.statistics.train_images}")
        print(f"  Validation Images:  {result.statistics.validation_images}")
        print(f"  Train Labels:       {result.statistics.train_labels}")
        print(f"  Validation Labels:  {result.statistics.validation_labels}")
        print(f"  Classes:            {result.statistics.classes}")
        if result.warnings:
            print(f"\n  Warnings ({len(result.warnings)}):")
            for w in result.warnings:
                print(f"    - [{w.code}] {w.file or ''}: {w.message}")
        print()
        return 0
    else:
        print(f"\n[FAILED] Dataset validation failed with {len(result.errors)} errors:")
        for e in result.errors:
            loc = f"{e.file or ''}" + (f":{e.line}" if e.line else "")
            print(f"  - [{e.code}] {loc}: {e.message}")
        if result.warnings:
            print(f"\nWarnings ({len(result.warnings)}):")
            for w in result.warnings:
                print(f"  - [{w.code}] {w.file or ''}: {w.message}")
        print()
        return 1


def handle_dataset_summary(args: argparse.Namespace, repo_root: Path) -> int:
    ds_path = Path(args.dataset_path)
    if not ds_path.is_absolute():
        ds_path = (repo_root / ds_path).resolve()

    try:
        analyzer = DatasetAnalyzer(ds_path)
        summary = analyzer.analyze()
        print("\n" + DatasetAnalyzer.format_text_summary(summary) + "\n")
        return 0
    except Exception as e:
        print(f"\n[ERROR] Failed to analyze dataset: {str(e)}\n", file=sys.stderr)
        return 1


def handle_train(args: argparse.Namespace, repo_root: Path) -> int:
    registry_file = repo_root / "models" / "registry.json"
    registry = ModelRegistry(registry_file=registry_file, repo_root=repo_root)
    service = TrainingService(registry=registry, repo_root=repo_root)

    overrides = {}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.image_size is not None:
        overrides["image_size"] = args.image_size
    if args.device is not None:
        overrides["device"] = args.device

    print(f"\n--- Initiating Training Run ---")
    print(f"Base Model:  {args.model}")
    print(f"Dataset:     {args.dataset}")
    print(f"Profile:     {args.profile}")

    try:
        config = TrainingConfig.from_profile(
            profile_name=args.profile,
            base_model=args.model,
            dataset_path=args.dataset,
            overrides=overrides,
        )

        print("Validating dataset...")
        result = service.train(
            base_model=args.model,
            dataset_path=args.dataset,
            configuration=config,
            custom_name=args.name,
            version=args.version,
        )

        print("\n[SUCCESS] Training Completed Successfully!")
        print(f"Run ID:             {result.run_id}")
        print(f"Custom Model ID:    {result.custom_model_id} (version {result.version})")
        print(f"Best Weights:       {result.best_weights_path}")
        print(f"Model Directory:    {result.custom_model_dir}")
        print("\nEvaluation Metrics:")
        print(f"  Precision:        {result.metrics.precision:.4f}")
        print(f"  Recall:           {result.metrics.recall:.4f}")
        print(f"  mAP50:            {result.metrics.map50:.4f}")
        print(f"  mAP50-95:         {result.metrics.map50_95:.4f}")
        print()
        return 0
    except Exception as e:
        print(f"\n[ERROR] Training execution failed: {str(e)}\n", file=sys.stderr)
        return 1


def handle_list_training_runs(args: argparse.Namespace, repo_root: Path) -> int:
    runs_dir = repo_root / "training_runs"
    manager = TrainingRunManager(runs_root=runs_dir, repo_root=repo_root)
    runs = manager.list_runs()

    print(f"\nTraining Runs ({len(runs)} found):")
    print("-" * 80)
    print(f"{'Run ID':<28} {'Base Model':<18} {'Device':<8} {'Status':<12} {'Output Model'}")
    print("-" * 80)
    for r in runs:
        run_id = r.get("run_id", "unknown")
        base_m = str(r.get("base_model_id", "unknown"))
        dev = str(r.get("device", "cpu"))
        stat = str(r.get("status", "unknown"))
        out_m = str(r.get("output_model_id") or "-")
        print(f"{run_id:<28} {base_m:<18} {dev:<8} {stat:<12} {out_m}")
    print("-" * 80 + "\n")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    repo_root = Path.cwd().resolve()

    if args.command == "list-models":
        return handle_list_models(args, repo_root)
    elif args.command == "validate-dataset":
        return handle_validate_dataset(args, repo_root)
    elif args.command == "dataset-summary":
        return handle_dataset_summary(args, repo_root)
    elif args.command == "train":
        return handle_train(args, repo_root)
    elif args.command == "list-training-runs":
        return handle_list_training_runs(args, repo_root)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
