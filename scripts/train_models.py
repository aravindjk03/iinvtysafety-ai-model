"""Training and Registration Pipeline for AI Safety Workshop.

Executes fine-tuning for:
1. PPE Detection Model (ppe-workshop-v1)
2. Fire & Smoke Detection Model (fire-smoke-workshop-v1)

Uses existing TrainingService, DatasetValidator, YOLOEvaluator,
ModelLineage, and ModelRegistry components.
"""

import logging
import shutil
import time
from pathlib import Path

from ai_safety.core.model_metadata import ModelMetadata, ModelOrigin, ModelStatus, TaskType
from ai_safety.core.model_registry import ModelRegistry
from ai_safety.training.config import TrainingConfig
from ai_safety.training.service import TrainingService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_models")

REPO_ROOT = Path(__file__).resolve().parent.parent


def setup_base_models(registry: ModelRegistry):
    """Place base model weights and register factory pretrained baselines."""
    source_pt = REPO_ROOT / "yolo11n.pt"
    if not source_pt.exists():
        raise FileNotFoundError(f"Source yolo11n.pt not found at {source_pt}")

    # PPE base
    ppe_base_dir = REPO_ROOT / "models" / "pretrained" / "ppe"
    ppe_base_dir.mkdir(parents=True, exist_ok=True)
    ppe_base_pt = ppe_base_dir / "model.pt"
    if not ppe_base_pt.exists():
        shutil.copy2(source_pt, ppe_base_pt)

    if not registry.model_exists("ppe-pretrained-base"):
        ppe_meta = ModelMetadata(
            model_id="ppe-pretrained-base",
            name="YOLO11n PPE Base Pretrained",
            version="1.0.0",
            task_type=TaskType.OBJECT_DETECTION,
            framework="ultralytics_yolo",
            architecture="yolo11n",
            model_path="models/pretrained/ppe/model.pt",
            classes=["person", "helmet", "vest"],
            input_size=(320, 320),
            confidence_threshold=0.25,
            origin=ModelOrigin.PRETRAINED,
            status=ModelStatus.READY,
            training_supported=True,
            description="Ultralytics YOLO11n pretrained base weights for PPE detection fine-tuning.",
        )
        registry.register_model(ppe_meta)
        logger.info("Registered ppe-pretrained-base in ModelRegistry.")

    # Fire/Smoke base
    fire_base_dir = REPO_ROOT / "models" / "pretrained" / "fire_smoke"
    fire_base_dir.mkdir(parents=True, exist_ok=True)
    fire_base_pt = fire_base_dir / "model.pt"
    if not fire_base_pt.exists():
        shutil.copy2(source_pt, fire_base_pt)

    if not registry.model_exists("fire-smoke-pretrained-base"):
        fire_meta = ModelMetadata(
            model_id="fire-smoke-pretrained-base",
            name="YOLO11n Fire Smoke Base Pretrained",
            version="1.0.0",
            task_type=TaskType.OBJECT_DETECTION,
            framework="ultralytics_yolo",
            architecture="yolo11n",
            model_path="models/pretrained/fire_smoke/model.pt",
            classes=["fire", "smoke"],
            input_size=(320, 320),
            confidence_threshold=0.25,
            origin=ModelOrigin.PRETRAINED,
            status=ModelStatus.READY,
            training_supported=True,
            description="Ultralytics YOLO11n pretrained base weights for Fire & Smoke fine-tuning.",
        )
        registry.register_model(fire_meta)
        logger.info("Registered fire-smoke-pretrained-base in ModelRegistry.")


def train_ppe(service: TrainingService):
    """Train and register ppe-workshop-v1."""
    logger.info("========================================")
    logger.info("Starting PPE Detection Fine-Tuning")
    logger.info("========================================")
    print("MODEL: YOLO11n (nano)")
    print("DEVICE: CPU")
    print("DATASET: PPE Construction Site Safety (MIT/CC BY 4.0)")
    print("EPOCHS: 5")
    print("IMAGE SIZE: 320")
    print("BATCH SIZE: 4")

    cfg = TrainingConfig(
        base_model="ppe-pretrained-base",
        dataset_path="datasets/ppe/processed",
        epochs=5,
        image_size=320,
        batch_size=4,
        device="cpu",
        workers=0,
        seed=42,
        project_name="ppe-workshop",
    )

    t0 = time.time()
    result = service.train(
        base_model="ppe-pretrained-base",
        dataset_path="datasets/ppe/processed",
        configuration=cfg,
        custom_name="ppe-workshop",
        version="1.0.0",
        custom_model_id="ppe-workshop-v1",
    )
    duration = time.time() - t0
    logger.info(f"PPE Training completed in {duration:.2f} seconds ({duration / 60:.2f} minutes).")
    logger.info(f"Run ID: {result.run_id}")
    logger.info(f"Custom Model ID: {result.custom_model_id}")
    logger.info(f"Metrics: mAP50={result.metrics.map50:.4f}, mAP50-95={result.metrics.map50_95:.4f}, P={result.metrics.precision:.4f}, R={result.metrics.recall:.4f}")
    return result, duration


def train_fire_smoke(service: TrainingService):
    """Train and register fire-smoke-workshop-v1."""
    logger.info("========================================")
    logger.info("Starting Fire & Smoke Detection Fine-Tuning")
    logger.info("========================================")
    print("MODEL: YOLO11n (nano)")
    print("DEVICE: CPU")
    print("DATASET: Fire-Smoke Dataset (CC BY 4.0)")
    print("EPOCHS: 5")
    print("IMAGE SIZE: 320")
    print("BATCH SIZE: 4")

    cfg = TrainingConfig(
        base_model="fire-smoke-pretrained-base",
        dataset_path="datasets/fire_smoke/processed",
        epochs=5,
        image_size=320,
        batch_size=4,
        device="cpu",
        workers=0,
        seed=42,
        project_name="fire-smoke-workshop",
    )

    t0 = time.time()
    result = service.train(
        base_model="fire-smoke-pretrained-base",
        dataset_path="datasets/fire_smoke/processed",
        configuration=cfg,
        custom_name="fire-smoke-workshop",
        version="1.0.0",
        custom_model_id="fire-smoke-workshop-v1",
    )
    duration = time.time() - t0
    logger.info(f"Fire & Smoke Training completed in {duration:.2f} seconds ({duration / 60:.2f} minutes).")
    logger.info(f"Run ID: {result.run_id}")
    logger.info(f"Custom Model ID: {result.custom_model_id}")
    logger.info(f"Metrics: mAP50={result.metrics.map50:.4f}, mAP50-95={result.metrics.map50_95:.4f}, P={result.metrics.precision:.4f}, R={result.metrics.recall:.4f}")
    return result, duration


if __name__ == "__main__":
    registry = ModelRegistry(registry_file="models/registry.json", repo_root=REPO_ROOT)
    setup_base_models(registry)
    service = TrainingService(registry=registry, repo_root=REPO_ROOT)
    
    ppe_res, ppe_dur = train_ppe(service)
    fs_res, fs_dur = train_fire_smoke(service)

    print("\n================== TRAINING COMPLETE ==================")
    print(f"PPE Model: {ppe_res.custom_model_id} (Duration: {ppe_dur:.1f}s)")
    print(f"Fire/Smoke Model: {fs_res.custom_model_id} (Duration: {fs_dur:.1f}s)")
    print(f"Registered Models: {[m.model_id for m in registry.list_models()]}")
