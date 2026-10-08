# AI Safety Model Lab (Repository 2)

> **Phase 1 through Phase 5: Model Lab Foundation, Dataset & Training Engine, PPE Detection, Deep Fire/Smoke Detection, Danger-Zone Detection, and Fall Detection (Pose Estimation)**

---

> [!CAUTION]
> **Industrial Safety & Emergency Compliance Disclaimer**:
> AI detection output is an aid to safety decision-making and must not replace required safety procedures, supervision, PPE policies, code-compliant smoke/fire alarms, certified safety systems, or mandatory fall-arrest equipment. These computer vision models act strictly as optical perception sensors. They do not evaluate legal compliance, certify hazard abatement, or trigger automated building evacuation.

---

## 1. Purpose

The **AI Safety Model Lab** is the core model intelligence hub for an educational Industrial AI Safety workshop ecosystem. The architecture is split across two independent repositories:

- **Repository 1 (`SafetyVisionAI` / `ai-safety-builder`)**: Front-facing participant interface handling web UI, camera input, real-time video streaming, safety result visualization, danger zone polygon overlays, and hazard alarms.
- **Repository 2 (`ai-safety-model-lab`)**: Model laboratory managing baseline pretrained models, custom fine-tuned safety models, dataset validation, training/fine-tuning pipelines, evaluation benchmarking, model lineage, and artifact registration.

Repository 1 consumes models from Repository 2 through a **stable, normalized interface**. Downstream applications never need to understand internal YOLO weight structures, training hyperparameters, or low-level framework specifics.

---

## 2. Available Models in ModelRegistry

All real safety models and their base pretrained models are registered in `models/registry.json` and accessible via the unified `Predictor`:

| Model ID | Task | Origin | Base Model | Classes / Keypoints | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`ppe-workshop-v2`** | `object_detection` | `custom` | `ppe-pretrained-base` (15 ep, 75 img) | `person`, `helmet`, `vest` | `ready` |
| **`fire-smoke-v2`** | `object_detection` | `custom` | `fire-smoke-pretrained-base` (15 ep, 75 img) | `fire`, `smoke` | `ready` |
| **`danger-zone-v2`** | `object_detection` | `custom` | `danger-zone-pretrained-base` (15 ep, 75 img) | `person`, `machinery`, `vehicle`, `safety_cone` | `ready` |
| **`fall-detection-v2`** | `pose_estimation` | `custom` | `fall-pretrained-base` (YOLO11n-Pose) | `person` (17 skeletal keypoints) | `ready` |
| **`ppe-workshop-v1`** | `object_detection` | `custom` | `ppe-pretrained-base` (5 ep) | `person`, `helmet`, `vest` | `ready` |
| **`fire-smoke-workshop-v1`** | `object_detection` | `custom` | `fire-smoke-pretrained-base` (5 ep) | `fire`, `smoke` | `ready` |
| **`fire-smoke-deep-v1`** | `object_detection` | `custom` | `fire-smoke-pretrained-base` (15 ep) | `fire`, `smoke` | `ready` |
| **`danger-zone-workshop-v1`** | `object_detection` | `custom` | `danger-zone-pretrained-base` (15 ep) | `person`, `machinery`, `vehicle`, `safety_cone` | `ready` |
| **`fall-detection-workshop-v1`**| `pose_estimation` | `custom` | `fall-pretrained-base` (YOLO11n-Pose) | `person` (17 skeletal keypoints) | `ready` |
| **`ppe-pretrained-base`** | `object_detection` | `pretrained` | `yolo11n` | `person`, `helmet`, `vest` | `ready` |
| **`fire-smoke-pretrained-base`** | `object_detection` | `pretrained` | `yolo11n` | `fire`, `smoke` | `ready` |
| **`danger-zone-pretrained-base`**| `object_detection` | `pretrained` | `yolo11n` | `person`, `machinery`, `vehicle`, `safety_cone` | `ready` |
| **`fall-pretrained-base`** | `pose_estimation` | `pretrained` | `yolo11n-pose` | `person` (17 COCO keypoints) | `ready` |

---

## 3. Architecture & Lifecycle Pipeline

```
              PRETRAINED BASELINE / ARCHITECTURE
                               │
                               ▼
                         SELECT MODEL
                               │
                               ▼
                      PARTICIPANT DATASET
         (datasets/ppe/, fire_smoke/, danger_zone/, fall_detection/)
                               │
                               ▼
                       DATASET VALIDATOR
             (Syntax, BBoxes, Coordinates, Keypoints)
                               │
                               ▼
                       DATASET ANALYZER
                 (Class Distribution, Counts)
                               │
                               ▼
                     TRAINING CONFIGURATION
                  (Profile: workshop_cpu / Device)
                               │
                               ▼
                       TRAINING SERVICE
                  (Ultralytics YOLO Engine)
                               │
                               ▼
                      TRAINING ARTIFACTS
              (training_runs/<run_id>/weights/best.pt)
                               │
                               ▼
                        YOLO EVALUATOR
             (Real mAP50, Precision, Recall metrics)
                               │
                               ▼
                     CUSTOM MODEL EXPORT
                  (models/custom/<model_id>/)
                               │
                               ▼
                         MODEL REGISTRY
                     (models/registry.json)
                               │
                               ▼
                          PREDICTOR
            (Normalized PredictionResult + Keypoints)
                               │
                               ▼
                          REPOSITORY 1
```

---

## 4. Dataset Standards & Datasets

### Curated Workshop Datasets
1. **PPE Detection Dataset** (`datasets/ppe/processed`):
   - Classes: `person` (0), `helmet` (1), `vest` (2).
   - Documented in [`datasets/ppe/README.md`](file:///C:/Users/Manikandan/.gemini/antigravity/scratch/ai-safety-model-lab/datasets/ppe/README.md).
2. **Fire & Smoke Detection Dataset** (`datasets/fire_smoke/processed`):
   - Classes: `fire` (0), `smoke` (1).
   - Documented in [`datasets/fire_smoke/README.md`](file:///C:/Users/Manikandan/.gemini/antigravity/scratch/ai-safety-model-lab/datasets/fire_smoke/README.md).
3. **Danger-Zone Dataset** (`datasets/danger_zone/processed`):
   - Classes: `person` (0), `machinery` (1), `vehicle` (2), `safety_cone` (3).
   - Validated: 20 train, 5 val, 2 test images with full bounding boxes.
4. **Fall Detection Dataset** (`datasets/fall_detection/processed`):
   - Classes: `person` (0) with `kpt_shape: [17, 3]` (17 skeletal body keypoints).
   - Validated with both bounding boxes and full normalized keypoint annotations.

### Re-preparing Datasets
To refresh or download canonical subsets:
```powershell
python scripts/prepare_datasets.py
python scripts/prepare_additional_datasets.py
```

---

## 5. Python API Usage

### Multi-Model Inference with Predictor
The unified `Predictor` loads any registered model by `model_id` and emits normalized `PredictionResult` objects:

```python
from ai_safety.core.model_registry import ModelRegistry
from ai_safety.inference.predictor import Predictor

registry = ModelRegistry("models/registry.json")
predictor = Predictor(registry=registry)

# 1. PPE Inference
ppe_result = predictor.predict(
    model_id="ppe-workshop-v1",
    input_data="path/to/worker_frame.jpg",
    confidence_threshold=0.25,
)
for det in ppe_result.detections:
    print(f"PPE: {det.class_name} ({det.confidence:.2f}) at {det.bounding_box}")

# 2. Fire & Smoke Inference (identical Predictor interface)
fire_result = predictor.predict(
    model_id="fire-smoke-deep-v1",
    input_data="path/to/cctv_frame.jpg",
    confidence_threshold=0.25,
)
for det in fire_result.detections:
    print(f"Hazard: {det.class_name} ({det.confidence:.2f}) at {det.bounding_box}")

# 3. Danger-Zone Inference
dz_result = predictor.predict(
    model_id="danger-zone-workshop-v1",
    input_data="path/to/forklift_floor.jpg",
    confidence_threshold=0.25,
)
for det in dz_result.detections:
    print(f"Zone Entity: {det.class_name} ({det.confidence:.2f})")

# 4. Fall Detection (Pose Estimation with 17 Keypoints)
fall_result = predictor.predict(
    model_id="fall-detection-workshop-v1",
    input_data="path/to/fallen_worker.jpg",
    confidence_threshold=0.25,
)
for det in fall_result.detections:
    print(f"Worker Pose: {len(det.keypoints)} skeletal keypoints detected")
    for kp in det.keypoints:
        print(f"  - {kp.name}: ({kp.x}, {kp.y}) conf={kp.confidence}")
```

### Normalized Prediction Schema (Pose Estimation Example)
```json
{
  "model_id": "fall-detection-workshop-v1",
  "task_type": "pose_estimation",
  "processing_time_ms": 142.5,
  "detections": [
    {
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.885,
      "bounding_box": {
        "x1": 150.0,
        "y1": 420.0,
        "x2": 480.0,
        "y2": 590.0
      },
      "keypoints": [
        {"id": 0, "name": "nose", "x": 160.2, "y": 430.1, "confidence": 0.92},
        {"id": 5, "name": "left_shoulder", "x": 195.4, "y": 450.0, "confidence": 0.89},
        {"id": 6, "name": "right_shoulder", "x": 192.1, "y": 480.2, "confidence": 0.91},
        {"id": 11, "name": "left_hip", "x": 310.0, "y": 490.5, "confidence": 0.86},
        {"id": 12, "name": "right_hip", "x": 315.2, "y": 510.3, "confidence": 0.88}
      ]
    }
  ]
}
```

---

## 6. Command-Line Interface (CLI)

The CLI works cross-platform using Python standard library `argparse`:

```powershell
# 1. List all models
python -m ai_safety list-models

# 2. Filter models by origin
python -m ai_safety list-models --origin custom
python -m ai_safety list-models --origin pretrained

# 3. Validate datasets
python -m ai_safety validate-dataset datasets/ppe/processed
python -m ai_safety validate-dataset datasets/fire_smoke/processed
python -m ai_safety validate-dataset datasets/danger_zone/processed
python -m ai_safety validate-dataset datasets/fall_detection/processed

# 4. Generate quantitative dataset summaries
python -m ai_safety dataset-summary datasets/danger_zone/processed
python -m ai_safety dataset-summary datasets/fall_detection/processed

# 5. List past training runs and lineage
python -m ai_safety list-training-runs
```

---

## 7. Testing & Quality Assurance

### Full Unit Test Suite (< 12 seconds, 82 tests)
Executes all unit tests offline on CPU:
```powershell
pytest
```

### Full Integration Test Suite (Real Training & Multi-Model Switching)
```powershell
pytest -m integration
```
**Total Passing Tests: 88 / 88 (100% green: 86 unit, 2 integration).**

---

## 8. Framework & Boundary Constraints
- **Sole Vision Stack**: Ultralytics YOLO (`ultralytics`).
- **Sole DL Backend**: PyTorch (`torch`).
- **No Alternative Detection Frameworks**: No TensorFlow, Detectron2, MMDetection, KerasCV, OpenCV DNN.
- **No Premature Complexity**: No Docker, databases, cloud orchestrators, or UI code in Repository 2.
