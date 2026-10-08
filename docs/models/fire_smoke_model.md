# Fire & Smoke Detection Model Documentation

## 1. Model Overview
- **Model ID**: `fire-smoke-workshop-v1`
- **Name**: `fire-smoke-workshop`
- **Version**: `1.0.0`
- **Task Type**: `object_detection`
- **Framework**: `ultralytics_yolo`
- **Architecture**: `fire-smoke-pretrained-base` (fine-tuned from YOLO11n)
- **Model Path**: `models/custom/fire-smoke-workshop-v1/model.pt`
- **Classes**: `["fire", "smoke"]`
- **Input Resolution**: `320x320`
- **Default Confidence Threshold**: `0.25`
- **Origin**: `custom`
- **Training Run ID**: `run-20261007-231719-8fad`
- **Status**: `ready`

---

## 2. Dataset & Training Provenance
- **Dataset**: Early Fire & Smoke Detection Dataset (Curated Workshop Split)
- **Source**: Roboflow Universe `fire-wrpgm/dataset/8` / Abonia1 (CC BY 4.0 License)
- **Splits**: 20 Train, 5 Val, 2 Test
- **Total Annotations**: 25 instances (`fire`: 21, `smoke`: 4)
- **Training Duration**: 28.9 seconds (CPU)
- **Lineage**: Stored in `models/custom/fire-smoke-workshop-v1/lineage.json`

---

## 3. Evaluation Metrics
From `models/custom/fire-smoke-workshop-v1/metrics.json`:
- **mAP@50**: `0.3120`
- **mAP@50-95**: `0.2385`
- **Precision**: `0.0034`
- **Recall**: `1.0000`
- **Per-Class Breakdown**:
  - `fire`: Recall = 100.0%, mAP@50-95 = 7.9%
  - `smoke`: Recall = 100.0%, mAP@50-95 = 39.8%

---

## 4. Usage Instructions

### Python API
```python
from ai_safety.core.model_registry import ModelRegistry
from ai_safety.inference.predictor import Predictor

registry = ModelRegistry("models/registry.json")
predictor = Predictor(registry=registry)

# Run inference
result = predictor.predict(
    model_id="fire-smoke-workshop-v1",
    input_data="path/to/cctv_frame.jpg",
    confidence_threshold=0.25,
)

for det in result.detections:
    print(f"Detected {det.class_name} ({det.confidence:.2f}) at {det.bounding_box}")
```

### CLI
```bash
# Validate dataset
python -m ai_safety validate-dataset --dataset datasets/fire_smoke/processed

# Dataset summary
python -m ai_safety dataset-summary --dataset datasets/fire_smoke/processed

# List registered models
python -m ai_safety list-models
```

---

## 5. Important Safety Boundary
The model detects bounding boxes for `fire` and `smoke`. It does **not** make emergency assessment claims (e.g. "evacuate building" or "fire confirmed"). It acts strictly as an optical perception sensor. Downstream alert dispatching belongs in Repository 1.
