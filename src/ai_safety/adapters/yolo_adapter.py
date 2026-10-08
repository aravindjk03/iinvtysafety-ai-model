"""YOLO Model Adapter boundary abstraction for Ultralytics/PyTorch compatible models."""

import time
from pathlib import Path
from typing import Any, Dict, Optional

from ai_safety.adapters.base_adapter import BaseModelAdapter
from ai_safety.core.exceptions import (
    ModelLoadError,
    PredictionError,
    UnsupportedModelError,
)
from ai_safety.core.model_metadata import ModelMetadata, ModelStatus, TaskType
from ai_safety.core.prediction import (
    BoundingBox,
    Detection,
    Keypoint,
    PredictionResult,
)

COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


class YOLOAdapter(BaseModelAdapter):
    """Adapter boundary for Ultralytics YOLO models (Object Detection & Pose Estimation).

    Establishes the runtime boundary through which future YOLO weights will execute.
    Phase 1 does NOT download or package real weights. If weights are missing,
    loading fails with a controlled ModelLoadError. No fake predictions are emitted.
    """

    SUPPORTED_FRAMEWORKS = {"ultralytics_yolo", "yolo", "pytorch_yolo"}
    SUPPORTED_TASKS = {
        TaskType.OBJECT_DETECTION,
        TaskType.POSE_ESTIMATION,
        TaskType.SEGMENTATION,
        TaskType.CLASSIFICATION,
    }

    def __init__(
        self,
        metadata: ModelMetadata,
        device: str = "cpu",
        repo_root: Optional[Path] = None,
    ):
        super().__init__(metadata=metadata, device=device, repo_root=repo_root)

        # Validate that the framework matches YOLO expectations
        if metadata.framework.lower() not in self.SUPPORTED_FRAMEWORKS:
            raise UnsupportedModelError(
                f"YOLOAdapter does not support framework '{metadata.framework}'. "
                f"Expected one of: {sorted(list(self.SUPPORTED_FRAMEWORKS))}"
            )

        if metadata.task_type not in self.SUPPORTED_TASKS:
            raise UnsupportedModelError(
                f"YOLOAdapter does not support task type '{metadata.task_type}'. "
                f"Expected one of: {sorted([t.value for t in self.SUPPORTED_TASKS])}"
            )

    def load_model(self, device: Optional[str] = None) -> None:
        """Attempt to load model weights.

        In Phase 1, weights are not bundled. If the weight file does not exist
        or is invalid, raises ModelLoadError.

        Args:
            device: Optional target device override ('cpu' or 'cuda').
        """
        if device is not None:
            self.device = self._validate_device(device)

        full_path = self.resolve_model_path()
        if not full_path.exists():
            self.metadata.status = ModelStatus.FAILED
            raise ModelLoadError(
                f"Model weights file not found at '{full_path}'. "
                f"Phase 1 establishes the architecture boundary without downloading weights."
            )

        if not full_path.is_file() or full_path.stat().st_size == 0:
            self.metadata.status = ModelStatus.FAILED
            raise ModelLoadError(
                f"Model file at '{full_path}' is empty or invalid."
            )

        # For dummy test fixture (TEST ONLY)
        with open(full_path, "rb") as f:
            header = f.read(32)
            if b"TEST ONLY" in header:
                # Mock handle for testing architecture wiring only
                self._model_instance = {"type": "mock_fixture", "path": str(full_path)}
                self._is_loaded = True
                self.metadata.status = ModelStatus.READY
                return

        # For real weights in future phases:
        # ultralytics YOLO(full_path) will be invoked here in Phase 2.
        # Since Phase 1 does not install Ultralytics, fail with controlled ModelLoadError.
        try:
            from ultralytics import YOLO  # type: ignore
            self._model_instance = YOLO(str(full_path))
            self._is_loaded = True
            self.metadata.status = ModelStatus.READY
        except ImportError as e:
            self.metadata.status = ModelStatus.FAILED
            raise ModelLoadError(
                "Ultralytics runtime is not installed in Phase 1 environment. "
                "Real model execution will be enabled in Phase 2."
            ) from e
        except Exception as e:
            self.metadata.status = ModelStatus.FAILED
            raise ModelLoadError(f"Failed to load YOLO model: {str(e)}") from e

    def predict(
        self,
        input_data: Any,
        confidence_threshold: Optional[float] = None,
    ) -> PredictionResult:
        """Execute inference through YOLO boundary.

        Does not create fake predictions. Requires loaded model and raises PredictionError
        if execution is attempted on dummy fixtures without real inference engine.
        """
        t0 = time.time()
        self._validate_input_presence(input_data)
        self._ensure_loaded()

        # Check if running under dummy mock fixture
        if isinstance(self._model_instance, dict) and self._model_instance.get("type") == "mock_fixture":
            raise PredictionError(
                "Dummy test fixture cannot execute real inference. "
                "Real predictions require trained weights and ML runtime in Phase 2."
            )

        conf = confidence_threshold if confidence_threshold is not None else self.metadata.confidence_threshold

        try:
            results = self._model_instance.predict(
                source=input_data,
                conf=conf,
                device=self.device,
                verbose=False,
            )
            elapsed_ms = (time.time() - t0) * 1000.0

            detections = []
            if results and len(results) > 0:
                first_res = results[0]
                boxes = getattr(first_res, "boxes", None)
                raw_kps = getattr(first_res, "keypoints", None)

                if boxes is not None:
                    for i, box in enumerate(boxes):
                        coords = box.xyxy[0].tolist()
                        cls_idx = int(box.cls[0].item())
                        confidence = float(box.conf[0].item())

                        # Class name lookup
                        if cls_idx < len(self.metadata.classes):
                            cname = self.metadata.classes[cls_idx]
                        elif hasattr(first_res, "names") and cls_idx in first_res.names:
                            cname = str(first_res.names[cls_idx])
                        else:
                            cname = f"class_{cls_idx}"

                        bbox = BoundingBox(
                            x1=float(coords[0]),
                            y1=float(coords[1]),
                            x2=float(coords[2]),
                            y2=float(coords[3]),
                        )

                        # Extract skeletal keypoints if available (Pose Estimation)
                        extracted_kps = None
                        if raw_kps is not None and i < len(raw_kps.data):
                            extracted_kps = []
                            kp_row = raw_kps.data[i]  # shape: (17, 2) or (17, 3)
                            for kp_idx, pt in enumerate(kp_row):
                                pt_list = pt.tolist()
                                x_pt = float(pt_list[0])
                                y_pt = float(pt_list[1])
                                conf_pt = float(pt_list[2]) if len(pt_list) > 2 else None
                                kp_name = (
                                    COCO_KEYPOINT_NAMES[kp_idx]
                                    if kp_idx < len(COCO_KEYPOINT_NAMES)
                                    else f"kp_{kp_idx}"
                                )
                                extracted_kps.append(
                                    Keypoint(
                                        id=kp_idx,
                                        name=kp_name,
                                        x=round(x_pt, 2),
                                        y=round(y_pt, 2),
                                        confidence=round(conf_pt, 4) if conf_pt is not None else None,
                                    )
                                )

                        detections.append(
                            Detection(
                                class_id=cls_idx,
                                class_name=cname,
                                confidence=round(confidence, 4),
                                bounding_box=bbox,
                                keypoints=extracted_kps,
                            )
                        )

            return PredictionResult(
                model_id=self.metadata.model_id,
                task_type=self.metadata.task_type,
                processing_time_ms=round(elapsed_ms, 2),
                detections=detections,
            )
        except Exception as e:
            if isinstance(e, PredictionError):
                raise
            raise PredictionError(f"YOLO inference failed: {str(e)}") from e

    def evaluate(self, dataset_path: str, **kwargs: Any) -> Dict[str, Any]:
        """Evaluate model against dataset using YOLO validation engine."""
        self._ensure_loaded()
        raise NotImplementedError("Model evaluation pipeline is deferred to Phase 2.")

    def export(self, target_format: str, output_path: str, **kwargs: Any) -> str:
        """Export YOLO weights to target format (ONNX, TensorRT, TorchScript)."""
        self._ensure_loaded()
        raise NotImplementedError("Model export pipeline is deferred to Phase 2.")
