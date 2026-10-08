# Personal Protective Equipment (PPE) Detection Dataset

## 1. Overview & Provenance
- **Dataset Name**: Construction Site Safety PPE Detection Dataset (Curated Workshop Split)
- **Primary Source**: Construction Site Safety Image Dataset via Roboflow Universe / VoxDroid Project
  - Repository: `https://github.com/VoxDroid/Construction-Site-Safety-PPE-Detection`
  - Origin: Roboflow Universe Construction Site Safety Dataset
- **License**: MIT License (VoxDroid) / CC BY 4.0 (Roboflow Universe)
- **Dataset Version**: v1.0 (Exported from CSS Revision 30)
- **Date Obtained**: October 2026

## 2. Class Definitions & Preprocessing
The upstream Construction Site Safety dataset annotates 10 raw classes (`Hardhat`, `Mask`, `NO-Hardhat`, `NO-Mask`, `NO-Safety Vest`, `Person`, `Safety Cone`, `Safety Vest`, `machinery`, `vehicle`).

For the Industrial AI Safety Model Lab workshop, the canonical classes are strictly mapped to the primary PPE safety objects:
| Canonical Class ID | Canonical Class Name | Raw Source Label | Description |
| :--- | :--- | :--- | :--- |
| `0` | `person` | `Person` (ID 5) | Human worker on industrial site |
| `1` | `helmet` | `Hardhat` (ID 0) | Protective safety helmet/hard hat |
| `2` | `vest` | `Safety Vest` (ID 7) | High-visibility reflective safety vest |

*Non-target auxiliary classes (machinery, vehicles, cones, negative labels) are filtered out during canonical preprocessing.*

## 3. Dataset Splits & Statistics
- **Train Images**: 20 images
- **Validation Images**: 5 images
- **Test (Smoke Test) Images**: 2 images
- **Total Images**: 25 (Train + Val)
- **Total Annotations**: 193
  - `person`: 132 annotations
  - `helmet`: 48 annotations
  - `vest`: 13 annotations

## 4. Class Distribution & Known Warnings
- **Class Imbalance**: Ratio `person:vest` is 132:13 (~10.1:1). In industrial imagery, multiple workers appear at varying distances, with vests frequently occluded by equipment or tools.
- **Small Objects**: Helmets and vests on distant workers comprise small normalized bounding box areas ($< 0.05$ relative width/height).

## 5. Directory Structure
```text
datasets/ppe/
├── raw/               # Original raw downloads (git-ignored)
├── processed/         # Canonical YOLO structure
│   ├── images/
│   │   ├── train/     # 20 images
│   │   ├── val/       # 5 images
│   │   └── test/      # 2 images
│   ├── labels/
│   │   ├── train/     # 20 label files
│   │   ├── val/       # 5 label files
│   │   └── test/      # 2 label files
│   └── data.yaml      # Canonical YOLO configuration
└── README.md
```

## 6. Preparation Instructions
To recreate or re-download this curated subset:
```bash
python scripts/prepare_datasets.py
```
This script downloads, normalizes coordinates, verifies image integrity via Pillow, and outputs the canonical `data.yaml`.
