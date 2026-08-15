"""YOLO Player detector using ONNX Runtime + DirectML (AMD GPU)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import onnxruntime as ort


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int = 0
    label: str = "Player"


def _build_session(model_path: str, use_directml: bool) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1

    providers = ["CPUExecutionProvider"]
    if use_directml and "DmlExecutionProvider" in ort.get_available_providers():
        providers = ["DmlExecutionProvider", "CPUExecutionProvider"]

    return ort.InferenceSession(model_path, sess_options=opts, providers=providers)


class PlayerOnnxDetector:
    def __init__(
        self,
        model_path: Union[str, Path],
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        use_directml: bool = True,
    ) -> None:
        self.model_path = str(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz

        self.session = _build_session(self.model_path, use_directml)
        self.input_name = self.session.get_inputs()[0].name
        self.active_providers = self.session.get_providers()
        self.class_names = {0: "Player"}

        # Pre-allocated input buffer for fixed-size inference
        self._blob = np.empty((1, 3, imgsz, imgsz), dtype=np.float32)
        self._infer_frame: Optional[np.ndarray] = None

    @property
    def using_gpu(self) -> bool:
        return "DmlExecutionProvider" in self.active_providers

    def _prepare_square(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, float]:
        """Resize to imgsz x imgsz. Returns (square_bgr, scale_to_original)."""
        h, w = image_bgr.shape[:2]
        if h == self.imgsz and w == self.imgsz:
            return image_bgr, 1.0
        resized = cv2.resize(image_bgr, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        return resized, w / self.imgsz  # uniform scale for square crop

    @staticmethod
    def _letterbox(
        image: np.ndarray, new_shape: int = 640, color: Tuple[int, int, int] = (114, 114, 114)
    ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        h, w = image.shape[:2]
        scale = min(new_shape / h, new_shape / w)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_w = new_shape - new_w
        pad_h = new_shape - new_h
        left = pad_w // 2
        top = pad_h // 2
        padded = cv2.copyMakeBorder(
            resized, top, pad_h - top, left, pad_w - left, cv2.BORDER_CONSTANT, value=color
        )
        return padded, scale, (left, top)

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
        if len(boxes) == 0:
            return []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep: List[int] = []

        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return keep

    def _fill_blob(self, square_bgr: np.ndarray) -> None:
        rgb = square_bgr[:, :, ::-1].astype(np.float32)
        rgb *= (1.0 / 255.0)
        self._blob[0] = np.transpose(rgb, (2, 0, 1))

    def _decode(self, outputs: np.ndarray, scale: float, pad_x: float, pad_y: float, orig_w: int, orig_h: int) -> List[Detection]:
        preds = np.squeeze(outputs).T
        boxes = preds[:, :4]
        scores = preds[:, 4]

        mask = scores >= self.conf_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        if len(boxes) == 0:
            return []

        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = (cx - w / 2 - pad_x) / scale
        y1 = (cy - h / 2 - pad_y) / scale
        x2 = (cx + w / 2 - pad_x) / scale
        y2 = (cy + h / 2 - pad_y) / scale

        x1 = np.clip(x1, 0, orig_w - 1)
        y1 = np.clip(y1, 0, orig_h - 1)
        x2 = np.clip(x2, 0, orig_w - 1)
        y2 = np.clip(y2, 0, orig_h - 1)

        xyxy = np.stack([x1, y1, x2, y2], axis=1)
        keep = self._nms(xyxy, scores, self.iou_threshold)

        return [
            Detection(
                x1=int(xyxy[i, 0]),
                y1=int(xyxy[i, 1]),
                x2=int(xyxy[i, 2]),
                y2=int(xyxy[i, 3]),
                confidence=float(scores[i]),
            )
            for i in keep
        ]

    def detect(self, image_bgr: np.ndarray, square_input: bool = False) -> List[Detection]:
        orig_h, orig_w = image_bgr.shape[:2]

        if square_input or (orig_h == orig_w):
            square, scale = self._prepare_square(image_bgr)
            self._fill_blob(square)
            outputs = self.session.run(None, {self.input_name: self._blob})[0]
            return self._decode(outputs, scale, 0.0, 0.0, orig_w, orig_h)

        letterboxed, scale, (pad_x, pad_y) = self._letterbox(image_bgr, self.imgsz)
        self._fill_blob(letterboxed)
        outputs = self.session.run(None, {self.input_name: self._blob})[0]
        return self._decode(outputs, scale, float(pad_x), float(pad_y), orig_w, orig_h)

    @staticmethod
    def draw_detections(image_bgr: np.ndarray, detections: List[Detection]) -> np.ndarray:
        output = image_bgr.copy()
        for det in detections:
            cv2.rectangle(output, (det.x1, det.y1), (det.x2, det.y2), (0, 255, 0), 2)
            label = f"{det.label} {det.confidence:.2f}"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            y_top = max(det.y1 - th - baseline - 4, 0)
            cv2.rectangle(output, (det.x1, y_top), (det.x1 + tw + 4, y_top + th + baseline + 4), (0, 255, 0), -1)
            cv2.putText(
                output,
                label,
                (det.x1 + 2, y_top + th + 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
        return output
