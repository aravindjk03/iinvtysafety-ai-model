# PPE Detection Model Documentation

## 1. Model Overview
- **Model ID**: `ppe-workshop-v1`
- **Name**: `ppe-workshop`
- **Version**: `1.0.0`
- **Task Type**: `object_detection`
- **Framework**: `ultralytics_yolo`
- **Architecture**: `ppe-pretrained-base` (fine-tuned from YOLO11n)
- **Model Path**: `models/custom/ppe-workshop-v1/model.pt`
- **Classes**: `["person", "helmet", "vest"]`
- **Input Resolution**: `320x320`
- **Default Confidence Threshold**: `0.25`
- **Origin**: `custom`
- **Training Run ID**: `run-20261007-231647-c173`
- **Status**: `ready`

---

## 2. Dataset & Training Provenance
- **Dataset**: Construction Site Safety PPE Dataset (Curated Workshop Split)
- **Source**: Roboflow Universe / VoxDroid (MIT / CC BY 4.0 License)
- **Splits**: 20 Train, 5 Val, 2 Test
- **Total Annotations**: 193 instances (`person`: 132, `helmet`: 48, `vest`: 13)
- **Training Duration**: 31.9 seconds (CPU)
- **Lineage**: Stored in `models/custom/ppe-workshop-v1/lineage.json`

---

## 3. Evaluation Metrics
From `models/custom/ppe-workshop-v1/metrics.json`:
- **mAP@50**: `0.1362`
- **mAP@50-95**: `0.0762`
- **Precision**: `0.0073`
- **Recall**: `0.3942`
- **Per-Class Breakdown**:
  - `person`: Recall = 77.8%, mAP@50-95 = 22.8%
  - `helmet`: Recall = 7.1%, mAP@50-95 = 0.06%
  - `vest`: Recall = 33.3%, mAP@50-95 = 0.02%

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
    model_id="ppe-workshop-v1",
    input_data="path/to/worker_image.jpg",
    confidence_threshold=0.25,
)

for det in result.detections:
    print(f"Detected {det.class_name} ({det.confidence:.2f}) at {det.bounding_box}")
```

### CLI
```bash
# Validate dataset
python -m ai_safety validate-dataset --dataset datasets/ppe/processed

# Dataset summary
python -m ai_safety dataset-summary --dataset datasets/ppe/processed

# List registered models
python -m ai_safety list-models
```

---

## 5. Important Safety Boundary
The model detects spatial bounding boxes for `person`, `helmet`, and `vest`. It does **not** evaluate whether a person is compliant with site safety rules (e.g. wearing the helmet). That decision logic belongs to the downstream business application (Repository 1).
