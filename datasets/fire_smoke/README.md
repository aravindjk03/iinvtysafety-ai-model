# Fire & Smoke Detection Dataset

## 1. Overview & Provenance
- **Dataset Name**: Early Fire & Smoke Detection Dataset (Curated Workshop Split)
- **Primary Source**: Roboflow Universe `custom-thxhn/fire-wrpgm` (Dataset Version 8)
  - Reference Repository: `https://github.com/Abonia1/YOLOv8-Fire-and-Smoke-Detection`
  - Upstream URL: `https://universe.roboflow.com/custom-thxhn/fire-wrpgm/dataset/8`
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Dataset Version**: v8 (Exported 2023-02-02)
- **Date Obtained**: October 2026

## 2. Class Definitions & Preprocessing
The upstream Roboflow dataset exports 3 raw labels (`Fire`, `default`, `smoke`).
The dummy background label `default` (an unannotated export artifact) is explicitly removed, and class IDs are normalized:
| Canonical Class ID | Canonical Class Name | Raw Source Label | Description |
| :--- | :--- | :--- | :--- |
| `0` | `fire` | `Fire` (ID 0) | Active open flames or combusting fire plumes |
| `1` | `smoke` | `smoke` (ID 2) | Airborne smoke columns and smoke plumes |

## 3. Dataset Splits & Statistics
- **Train Images**: 20 images
- **Validation Images**: 5 images
- **Test (Smoke Test) Images**: 2 images
- **Total Images**: 25 (Train + Val)
- **Total Annotations**: 25
  - `fire`: 21 annotations
  - `smoke`: 4 annotations

## 4. Known Data Limitations
- **Environmental Diversity**: Sourced primarily from CCTV video captures of early-stage fire experiments.
- **Ambiguity & False Positives**:
  - Steam from industrial cooling towers vs. thin white smoke.
  - Dust kicked up by construction vehicles vs. dense smoke plumes.
  - Orange halogen safety lights or sunset glare vs. distant fire flames.
- **Perception Boundary**: Visual detection indicates the pixel presence of combustion artifacts; it does **not** evaluate fire hazard severity or trigger automated building alarms.

## 5. Directory Structure
```text
datasets/fire_smoke/
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
