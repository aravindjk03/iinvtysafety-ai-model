# Model Evaluation Review: PPE Detection (ppe-workshop-v1)

## 1. Executive Summary
This document provides an objective, empirical evaluation of the fine-tuned Personal Protective Equipment (PPE) detection model (`ppe-workshop-v1`) developed for the Industrial AI Safety Model Lab educational workshop.

> [!CAUTION]
> **Industrial Safety Compliance Disclaimer**:
> This computer vision model is an educational demonstration designed solely to explore object detection pipelines. It is **NOT** a certified life-safety system, does not guarantee workplace safety, and must not replace required safety personnel, industrial supervision, PPE compliance monitoring protocols, or certified safety devices. Visual detection of safety gear does not constitute proof that the gear is correctly worn, rated, or functional.

---

## 2. Model & Training Provenance
- **Model Identifier**: `ppe-workshop-v1`
- **Model Version**: `1.0.0`
- **Base Architecture**: Ultralytics YOLO11n (`yolo11n.pt`, nano checkpoint)
- **Base Model ID**: `ppe-pretrained-base`
- **Training Run ID**: `run-20261007-231647-c173`
- **Hardware Device**: Windows CPU (Intel Core i5, 0 dataloader workers)
- **Training Duration**: 31.9 seconds (0.53 minutes)
- **Epochs**: 5
- **Input Resolution**: 320x320 pixels
- **Batch Size**: 4
- **Random Seed**: 42

---

## 3. Dataset Characteristics
- **Dataset**: Construction Site Safety PPE Dataset (Curated Workshop Split)
- **Source**: Roboflow Universe / VoxDroid (MIT / CC BY 4.0 License)
- **Splits**: 20 Train images, 5 Validation images, 2 Test smoke-test images
- **Total Annotations**: 193 instances
- **Classes**:
  - `person` (Class 0): 132 annotations
  - `helmet` (Class 1): 48 annotations
  - `vest` (Class 2): 13 annotations

---

## 4. Empirical Evaluation Metrics
Evaluated on the independent validation split (5 images, 35 ground-truth instances) using the native Ultralytics validation engine (`YOLOEvaluator`):

| Metric | Score |
| :--- | :--- |
| **Precision (mean)** | `0.0073` (0.73%) |
| **Recall (mean)** | `0.3942` (39.42%) |
| **mAP@50** | `0.1362` (13.62%) |
| **mAP@50-95** | `0.0762` (7.62%) |
| **Fitness** | `0.0762` |

### Per-Class Performance Breakdown:
| Class Name | Precision | Recall | mAP@50-95 |
| :--- | :--- | :--- | :--- |
| **`person`** | 0.0143 | 0.7778 (77.8%) | 0.2278 (22.8%) |
| **`helmet`** | 0.0038 | 0.0714 (7.1%) | 0.0006 (0.06%) |
| **`vest`** | 0.0039 | 0.3333 (33.3%) | 0.0002 (0.02%) |

---

## 5. Technical Observations & Root-Cause Analysis
1. **Person Detection Sensitivity**:
   - The model achieved 77.8% recall on `person`, benefiting heavily from COCO pretraining priors where humans are prominent.
2. **Class Imbalance Impact**:
   - The dataset exhibited a natural $10.1:1$ imbalance between `person` (132) and `vest` (13). As expected, low-sample classes (`vest` and `helmet`) suffered significantly lower precision and mean average precision.
3. **Low-Epoch Fine-Tuning Realities**:
   - Training for 5 epochs on CPU at $320\times320$ resolution allows rapid workshop demonstration (~30 seconds on participant laptops) and demonstrates loss convergence. However, higher confidence detection requires additional fine-tuning epochs ($25\text{--}50$) and higher resolution ($640\times640$).
4. **Perception vs Compliance**:
   - The model strictly emits bounding boxes for `person`, `helmet`, and `vest`. It does not perform geometric association or semantic deduction (e.g., determining whether a detected helmet is located on the head of a detected person). That logic belongs exclusively in the future Repository 1 decision layer.

---

## 6. Known Failure Modes
- **Occluded Gear**: Workers partially behind scaffolding or machinery frequently fail to produce `vest` detections.
- **Small Distant Objects**: Helmets occupying $< 20\times20$ pixels at $320\times320$ image size are difficult to resolve.
- **Lighting & Color Confounding**: Yellow or white hard hats in high-glare environments risk being confused with industrial light fixtures or white headgear.
