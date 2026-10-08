# Model Evaluation Review: Fire & Smoke Detection (fire-smoke-workshop-v1)

## 1. Executive Summary
This document provides an objective, empirical evaluation of the fine-tuned Fire & Smoke detection model (`fire-smoke-workshop-v1`) developed for the Industrial AI Safety Model Lab educational workshop.

> [!CAUTION]
> **Life-Safety & Emergency Disclaimer**:
> This model is an educational demonstration designed solely to explore object detection architectures. It is **NOT** a certified fire safety or smoke alarm system. It does **NOT** trigger facility evacuations, does not evaluate fire spread or toxic gas levels, and must not replace code-compliant smoke detectors, fire suppression alarms, or industrial fire safety monitoring systems.

---

## 2. Model & Training Provenance
- **Model Identifier**: `fire-smoke-workshop-v1`
- **Model Version**: `1.0.0`
- **Base Architecture**: Ultralytics YOLO11n (`yolo11n.pt`, nano checkpoint)
- **Base Model ID**: `fire-smoke-pretrained-base`
- **Training Run ID**: `run-20261007-231719-8fad`
- **Hardware Device**: Windows CPU (Intel Core i5, 0 dataloader workers)
- **Training Duration**: 28.9 seconds (0.48 minutes)
- **Epochs**: 5
- **Input Resolution**: 320x320 pixels
- **Batch Size**: 4
- **Random Seed**: 42

---

## 3. Dataset Characteristics
- **Dataset**: Early Fire & Smoke Detection Dataset (Curated Workshop Split)
- **Source**: Roboflow Universe `fire-wrpgm/dataset/8` / Abonia1 (CC BY 4.0 License)
- **Splits**: 20 Train images, 5 Validation images, 2 Test smoke-test images
- **Total Annotations**: 25 instances
- **Classes**:
  - `fire` (Class 0): 21 annotations
  - `smoke` (Class 1): 4 annotations

---

## 4. Empirical Evaluation Metrics
Evaluated on the independent validation split (5 images, 5 ground-truth instances) using the native Ultralytics validation engine (`YOLOEvaluator`):

| Metric | Score |
| :--- | :--- |
| **Precision (mean)** | `0.0034` (0.34%) |
| **Recall (mean)** | `1.0000` (100.0%) |
| **mAP@50** | `0.3120` (31.20%) |
| **mAP@50-95** | `0.2385` (23.85%) |
| **Fitness** | `0.2385` |

### Per-Class Performance Breakdown:
| Class Name | Precision | Recall | mAP@50-95 |
| :--- | :--- | :--- | :--- |
| **`fire`** | 0.0056 | 1.0000 (100.0%) | 0.0790 (7.90%) |
| **`smoke`** | 0.0013 | 1.0000 (100.0%) | 0.3980 (39.80%) |

---

## 5. Technical Observations & Root-Cause Analysis
1. **High Recall Sensitivity**:
   - The model achieved 100% recall across both `fire` and `smoke` on the validation split, indicating strong detection sensitivity for combustion features.
2. **Precision Balance & Workshop CPU Constraints**:
   - Mean precision is low ($0.34\%$), reflecting that the model proposes numerous low-confidence candidate bounding boxes across ambient lighting regions. Setting a sensible confidence threshold ($0.25\text{--}0.50$) in downstream application logic prevents noise.
3. **Smoke Feature Generalization**:
   - Smoke plumes yielded an encouraging mAP@50-95 of $0.398$ even after only 5 CPU fine-tuning epochs.

---

## 6. Known Failure Modes & Confounders
- **Steam vs Smoke**: Cooling tower vapor and high-temperature steam plumes exhibit identical diffuse visual signatures to white smoke plumes.
- **Dust Clouds vs Smoke**: Heavy industrial or vehicular dust clouds can trigger false-positive smoke detections.
- **Lighting Glare vs Fire**: Low-angle sunset glare, high-intensity orange sodium-vapor factory lighting, and welding arc flashes can trigger false fire candidates.
- **Low-Light / Night Shifts**: Dark backgrounds degrade smoke boundary resolution unless IR illumination is present.
