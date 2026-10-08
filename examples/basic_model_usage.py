"""Educational example demonstrating the AI Safety Model Lab API contract.

Shows how downstream applications (Repository 1) interact with Repository 2
using the unified Model Registry and Predictor without needing to understand
internal model implementations or weight file paths.
"""

from pathlib import Path

from ai_safety.core.model_metadata import (
    ModelMetadata,
    ModelOrigin,
    ModelStatus,
    TaskType,
)
from ai_safety.core.model_registry import ModelRegistry
from ai_safety.core.exceptions import ModelLoadError, ModelNotFoundError
from ai_safety.inference.predictor import Predictor


def main():
    repo_root = Path(__file__).resolve().parent.parent
    registry_file = repo_root / "models" / "registry.json"

    print("=" * 60)
    print("AI Safety Model Lab - Foundation API Demonstration")
    print("=" * 60)

    # 1. Initialize Registry
    print("\n1. Initializing Model Registry...")
    registry = ModelRegistry(registry_file=registry_file, repo_root=repo_root)
    print(f"   Registry loaded from: {registry.registry_file.name}")
    print(f"   Currently registered models count: {len(registry.list_models())}")

    # 2. Inspect Registered Models
    print("\n2. Querying registered models by category/origin...")
    pretrained_models = registry.list_models(origin=ModelOrigin.PRETRAINED)
    print(f"   Pretrained baseline models: {len(pretrained_models)}")
    for m in pretrained_models:
        print(f"     - [{m.task_type.value}] {m.model_id}: {m.name} (v{m.version})")

    # 3. Register a Custom Model (Simulation of participant fine-tuned model)
    print("\n3. Demonstrating Custom Model Registration...")
    custom_model_meta = ModelMetadata(
        model_id="participant-ppe-detector-v1",
        name="Participant Workshop PPE Detector",
        version="1.0.0",
        task_type=TaskType.OBJECT_DETECTION,
        framework="ultralytics_yolo",
        architecture="yolo11n",
        model_path="models/custom/participant_ppe.pt",
        classes=["helmet", "hi-viz-vest", "gloves", "person"],
        input_size=(640, 640),
        confidence_threshold=0.35,
        origin=ModelOrigin.CUSTOM,
        base_model="yolo11n",
        training_run_id="run_workshop_team_04",
        status=ModelStatus.REGISTERED,
        training_supported=True,
        description="Fine-tuned PPE model created during workshop session",
    )

    if not registry.model_exists(custom_model_meta.model_id):
        registry.register_model(custom_model_meta)
        print(f"   Successfully registered custom model: {custom_model_meta.model_id}")
    else:
        print(f"   Custom model '{custom_model_meta.model_id}' is already registered.")

    # 4. Initialize Predictor Service
    print("\n4. Initializing Central Predictor Service...")
    predictor = Predictor(registry=registry, default_device="cpu", repo_root=repo_root)
    print(f"   Default execution device: {predictor.default_device.upper()}")

    # 5. Show Downstream Contract (Repository 1 Consumer Pattern)
    print("\n5. Demonstrating Downstream Consumption Contract for Repository 1:")
    print("   Repository 1 needs only:")
    print("     predictor = Predictor(registry)")
    print("     result = predictor.predict(model_id='participant-ppe-detector-v1', input_data=image)")
    print("\n   In Phase 1, weights are not downloaded. Testing load behavior:")

    try:
        predictor.predict(
            model_id="participant-ppe-detector-v1",
            input_data="sample_camera_frame.jpg",
        )
    except ModelLoadError as e:
        print(f"   Controlled Exception as Expected in Phase 1: {e}")
        print("   -> The system safely caught missing weights without crashing or producing fake outputs.")

    print("\n" + "=" * 60)
    print("AI Safety Model Lab Phase 1 foundation is operating as specified.")
    print("=" * 60)


if __name__ == "__main__":
    main()
