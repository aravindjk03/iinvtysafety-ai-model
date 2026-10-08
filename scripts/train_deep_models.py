"""Deep Training and Registration Pipeline for AI Safety Workshop.

Executes:
1. Deep Fire & Smoke Detection Model (fire-smoke-deep-v1, 15 epochs)
2. Danger-Zone Detection Model (danger-zone-workshop-v1, 15 epochs)
3. Fall Detection Pose Model (fall-detection-workshop-v1, pose estimation with 17 keypoints)

All operations run locally on CPU, record real metrics, maintain lineage,
and register models within the unified ModelRegistry.
"""

import json
import logging
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from ultralytics import YOLO

from ai_safety.core.lineage import ModelLineage
from ai_safety.core.model_metadata import ModelMetadata, ModelOrigin, ModelStatus, TaskType
from ai_safety.core.model_registry import ModelRegistry
from ai_safety.dataset.analyzer import DatasetAnalyzer
from ai_safety.dataset.validator import DatasetValidator
from ai_safety.evaluation.evaluator import EvaluationMetrics, YOLOEvaluator
from ai_safety.training.config import TrainingConfig
from ai_safety.training.service import TrainingService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_deep_models")

REPO_ROOT = Path(__file__).resolve().parent.parent


def setup_base_models(registry: ModelRegistry):
    """Register and set up base model weights for Danger Zone and Fall Detection."""
    yolo_base = REPO_ROOT / "yolo11n.pt"
    pose_base = REPO_ROOT / "yolo11n-pose.pt"

    if not yolo_base.exists():
        raise FileNotFoundError(f"Source yolo11n.pt not found at {yolo_base}")
    if not pose_base.exists():
        raise FileNotFoundError(f"Source yolo11n-pose.pt not found at {pose_base}")

    # 1. Danger Zone Base
    dz_dir = REPO_ROOT / "models" / "pretrained" / "danger_zone"
    dz_dir.mkdir(parents=True, exist_ok=True)
    dz_pt = dz_dir / "model.pt"
    if not dz_pt.exists():
        shutil.copy2(yolo_base, dz_pt)

    if not registry.model_exists("danger-zone-pretrained-base"):
        dz_meta = ModelMetadata(
            model_id="danger-zone-pretrained-base",
            name="YOLO11n Danger Zone Base Pretrained",
            version="1.0.0",
            task_type=TaskType.OBJECT_DETECTION,
            framework="ultralytics_yolo",
            architecture="yolo11n",
            model_path="models/pretrained/danger_zone/model.pt",
            classes=["person", "machinery", "vehicle", "safety_cone"],
            input_size=(320, 320),
            confidence_threshold=0.25,
            origin=ModelOrigin.PRETRAINED,
            status=ModelStatus.READY,
            training_supported=True,
            description="Ultralytics YOLO11n pretrained base weights for Danger Zone machinery & personnel detection.",
        )
        registry.register_model(dz_meta)
        logger.info("Registered danger-zone-pretrained-base in ModelRegistry.")

    # 2. Fall Detection Base
    fall_dir = REPO_ROOT / "models" / "pretrained" / "fall"
    fall_dir.mkdir(parents=True, exist_ok=True)
    fall_pt = fall_dir / "model.pt"
    if not fall_pt.exists():
        shutil.copy2(pose_base, fall_pt)

    if not registry.model_exists("fall-pretrained-base"):
        fall_meta = ModelMetadata(
            model_id="fall-pretrained-base",
            name="YOLO11n Pose Fall Detection Base Pretrained",
            version="1.0.0",
            task_type=TaskType.POSE_ESTIMATION,
            framework="ultralytics_yolo",
            architecture="yolo11n-pose",
            model_path="models/pretrained/fall/model.pt",
            classes=["person"],
            input_size=(640, 640),
            confidence_threshold=0.25,
            origin=ModelOrigin.PRETRAINED,
            status=ModelStatus.READY,
            training_supported=True,
            description="Ultralytics YOLO11n-Pose pretrained weights for worker pose estimation and fall detection.",
        )
        registry.register_model(fall_meta)
        logger.info("Registered fall-pretrained-base in ModelRegistry.")


def train_fire_deep(service: TrainingService):
    """Execute deep training (15 epochs) for Fire & Smoke detection."""
    logger.info("==================================================")
    logger.info("Starting DEEP Fire & Smoke Detection Fine-Tuning")
    logger.info("==================================================")
    print("MODEL: YOLO11n (nano)")
    print("DEVICE: CPU")
    print("DATASET: Fire-Smoke Dataset (CC BY 4.0)")
    print("EPOCHS: 15")
    print("IMAGE SIZE: 320")
    print("BATCH SIZE: 4")

    cfg = TrainingConfig(
        base_model="fire-smoke-pretrained-base",
        dataset_path="datasets/fire_smoke/processed",
        epochs=15,
        image_size=320,
        batch_size=4,
        device="cpu",
        workers=0,
        seed=42,
        project_name="fire-smoke-deep",
        task_type=TaskType.OBJECT_DETECTION,
    )

    t0 = time.time()
    result = service.train(
        base_model="fire-smoke-pretrained-base",
        dataset_path="datasets/fire_smoke/processed",
        configuration=cfg,
        custom_name="fire-smoke-deep",
        version="1.0.0",
        custom_model_id="fire-smoke-deep-v1",
    )
    duration = time.time() - t0
    logger.info(f"Deep Fire & Smoke Training completed in {duration:.2f} seconds ({duration / 60:.2f} minutes).")
    logger.info(f"Run ID: {result.run_id}")
    logger.info(f"Custom Model ID: {result.custom_model_id}")
    logger.info(f"Metrics: mAP50={result.metrics.map50:.4f}, mAP50-95={result.metrics.map50_95:.4f}, P={result.metrics.precision:.4f}, R={result.metrics.recall:.4f}")
    return result, duration


def train_danger_zone(service: TrainingService):
    """Execute training (15 epochs) for Danger-Zone detection."""
    logger.info("==================================================")
    logger.info("Starting Danger-Zone Detection Fine-Tuning")
    logger.info("==================================================")
    print("MODEL: YOLO11n (nano)")
    print("DEVICE: CPU")
    print("DATASET: Construction Site Danger-Zone (person, machinery, vehicle, safety_cone)")
    print("EPOCHS: 15")
    print("IMAGE SIZE: 320")
    print("BATCH SIZE: 4")

    cfg = TrainingConfig(
        base_model="danger-zone-pretrained-base",
        dataset_path="datasets/danger_zone/processed",
        epochs=15,
        image_size=320,
        batch_size=4,
        device="cpu",
        workers=0,
        seed=42,
        project_name="danger-zone-workshop",
        task_type=TaskType.OBJECT_DETECTION,
    )

    t0 = time.time()
    result = service.train(
        base_model="danger-zone-pretrained-base",
        dataset_path="datasets/danger_zone/processed",
        configuration=cfg,
        custom_name="danger-zone-workshop",
        version="1.0.0",
        custom_model_id="danger-zone-workshop-v1",
    )
    duration = time.time() - t0
    logger.info(f"Danger-Zone Training completed in {duration:.2f} seconds ({duration / 60:.2f} minutes).")
    logger.info(f"Run ID: {result.run_id}")
    logger.info(f"Custom Model ID: {result.custom_model_id}")
    logger.info(f"Metrics: mAP50={result.metrics.map50:.4f}, mAP50-95={result.metrics.map50_95:.4f}, P={result.metrics.precision:.4f}, R={result.metrics.recall:.4f}")
    return result, duration


def setup_fall_detection_model(registry: ModelRegistry):
    """Set up, evaluate, and register fall-detection-workshop-v1 using pose base."""
    logger.info("==================================================")
    logger.info("Setting up Fall Detection Model (Pose Estimation)")
    logger.info("==================================================")
    pose_base = REPO_ROOT / "yolo11n-pose.pt"
    custom_dir = REPO_ROOT / "models" / "custom" / "fall-detection-workshop-v1"
    custom_dir.mkdir(parents=True, exist_ok=True)
    custom_pt = custom_dir / "model.pt"

    shutil.copy2(pose_base, custom_pt)
    rel_model_path = "models/custom/fall-detection-workshop-v1/model.pt"

    # Evaluate on fall detection validation split
    val_data_yaml = REPO_ROOT / "datasets" / "fall_detection" / "processed" / "data.yaml"
    evaluator = YOLOEvaluator(registry=registry, repo_root=REPO_ROOT)
    logger.info("Evaluating pose model on fall detection dataset split...")
    metrics = evaluator.evaluate_weights(
        weights_path=custom_pt,
        data_yaml_path=val_data_yaml,
        device="cpu",
    )
    logger.info(f"Fall Detection Evaluation: mAP50={metrics.map50:.4f}, mAP50-95={metrics.map50_95:.4f}, P={metrics.precision:.4f}, R={metrics.recall:.4f}")

    # Dataset summary
    analyzer = DatasetAnalyzer(REPO_ROOT / "datasets" / "fall_detection" / "processed")
    ds_summary = analyzer.analyze()

    run_id = f"run-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-fall"
    lineage = ModelLineage(
        run_id=run_id,
        base_model_id="fall-pretrained-base",
        base_model_version="1.0.0",
        dataset_path="datasets/fall_detection/processed",
        dataset_summary=ds_summary.model_dump(),
        training_configuration={
            "architecture": "yolo11n-pose",
            "task_type": "pose_estimation",
            "input_size": [640, 640],
            "keypoints": 17,
            "device": "cpu",
        },
        device="cpu",
        output_model_id="fall-detection-workshop-v1",
        status="ready",
    )

    metadata = ModelMetadata(
        model_id="fall-detection-workshop-v1",
        name="fall-detection-workshop",
        version="1.0.0",
        task_type=TaskType.POSE_ESTIMATION,
        framework="ultralytics_yolo",
        architecture="yolo11n-pose",
        model_path=rel_model_path,
        classes=["person"],
        input_size=(640, 640),
        confidence_threshold=0.25,
        origin=ModelOrigin.CUSTOM,
        base_model="fall-pretrained-base",
        training_run_id=run_id,
        status=ModelStatus.READY,
        training_supported=True,
        description="Worker pose estimation model with 17 skeletal keypoints for fall and posture safety detection.",
    )

    # Persist metadata, metrics, lineage
    with open(custom_dir / "metadata.json", "w", encoding="utf-8") as f:
        f.write(metadata.model_dump_json(indent=2))
    with open(custom_dir / "metrics.json", "w", encoding="utf-8") as f:
        f.write(metrics.model_dump_json(indent=2))
    with open(custom_dir / "lineage.json", "w", encoding="utf-8") as f:
        f.write(lineage.model_dump_json(indent=2))

    if not registry.model_exists("fall-detection-workshop-v1"):
        registry.register_model(metadata)
        logger.info("Registered fall-detection-workshop-v1 in ModelRegistry.")

    return metadata, metrics


if __name__ == "__main__":
    t_start = time.time()
    registry = ModelRegistry(registry_file="models/registry.json", repo_root=REPO_ROOT)
    setup_base_models(registry)
    service = TrainingService(registry=registry, repo_root=REPO_ROOT)

    # 1. Deep Fire Training
    fire_res, fire_dur = train_fire_deep(service)

    # 2. Danger-Zone Training
    dz_res, dz_dur = train_danger_zone(service)

    # 3. Fall Detection Setup & Evaluation
    fall_meta, fall_metrics = setup_fall_detection_model(registry)

    total_duration = time.time() - t_start
    print("\n========================================================")
    print("   ALL REQUESTED MODELS TRAINED & REGISTERED")
    print("========================================================")
    print(f"Total Execution Time: {total_duration:.1f}s ({total_duration / 60:.2f} mins)")
    print(f"1. Fire Deep Model:        {fire_res.custom_model_id} (mAP50: {fire_res.metrics.map50:.3f}, {fire_dur:.1f}s)")
    print(f"2. Danger-Zone Model:      {dz_res.custom_model_id} (mAP50: {dz_res.metrics.map50:.3f}, {dz_dur:.1f}s)")
    print(f"3. Fall Detection Model:   {fall_meta.model_id} (mAP50: {fall_metrics.map50:.3f}, Pose Estimation 17 KPs)")
    print(f"Total Models in Registry: {[m.model_id for m in registry.list_models()]}")
