"""
Common interface for object detectors.

The object-detection slot is backend-agnostic: YOLO today, RT-DETR tomorrow. A
detector takes one RGB frame and returns a list of class-labelled bounding
boxes. The *use* of those detections (online 3D-localisation in
`scripts/vings_utils/object_tracker.py`) lives elsewhere -- detectors only know
how to detect.

This is the contract a new detector (e.g. RT-DETR) must satisfy:

    dets = detector.detect(rgb)        # rgb -> list[Detection]

Register it with @register_detector("name") in its own module (mirroring
`detector_factory.py` / the selector + segmentation factories), and it becomes
selectable via `object_detector.kind`. No change to the run loop is needed for
a swap.

Contrast with the segmentation backends (`segmentation_base.py`): those return
class-agnostic instance *masks* for dynamic-masking; detectors here return
*class labels + boxes + confidence* so each object can be named and localised.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch


# 80 COCO class names, index = class id (ultralytics YOLO/RT-DETR default order).
COCO_CLASSES: tuple[str, ...] = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)


def class_color(cls_id: int) -> tuple[int, int, int]:
    """Deterministic, well-spread RGB colour for a class id (for markers/overlays)."""
    # Golden-angle hashing in hue -> distinct colours without a palette table.
    h = (cls_id * 0.61803398875) % 1.0
    # simple HSV(h, 0.65, 1.0) -> RGB
    i = int(h * 6.0)
    f = h * 6.0 - i
    s, v = 0.65, 1.0
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    r, g, b = [(v, t, p), (q, v, p), (p, v, t),
               (p, q, v), (t, p, v), (v, p, q)][i % 6]
    return int(r * 255), int(g * 255), int(b * 255)


def to_uint8_rgb(rgb) -> np.ndarray:
    """
    Normalise an arbitrary RGB input to a contiguous (H, W, 3) uint8 array.

    Mirrors `segmentation_base.to_uint8_rgb`: accepts torch/numpy, float [0,1]
    or uint8, (H,W,3) or (3,H,W). `viz_out` images are float [0,1] (H,W,3).
    """
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.detach().cpu().numpy()
    rgb = np.asarray(rgb)
    if rgb.ndim == 3 and rgb.shape[0] == 3 and rgb.shape[2] != 3:
        rgb = np.transpose(rgb, (1, 2, 0))  # (3,H,W) -> (H,W,3)
    if rgb.dtype != np.uint8:
        if rgb.max() <= 1.0 + 1e-6:
            rgb = rgb * 255.0
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(rgb[..., :3])


@dataclass
class Detection:
    """One detected object in one frame.

    `bbox_xyxy` is in pixel coordinates of the input frame, OpenCV convention
    (x = column, y = row) -- the same order ultralytics returns and the same
    order `object_tracker` feeds into back-projection.
    """
    bbox_xyxy: tuple[float, float, float, float]  # (x1, y1, x2, y2)
    cls_id: int
    cls_name: str
    conf: float

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return 0.5 * (x1 + x2), 0.5 * (y1 + y2)  # (cx_col, cy_row)


def boxes_to_detections(boxes, names=None) -> list[Detection]:
    """Convert an ultralytics `Results.boxes` object to a list[Detection].

    Shared by every ultralytics-backed detector (YOLO, RT-DETR): both expose
    the same `.xyxy / .cls / .conf` tensors. `names` is the model's own id->name
    mapping (`results[0].names`) -- using it (not a hardcoded COCO table) keeps
    labels correct for non-COCO models like VisDrone (car=3, not 2). Falls back
    to COCO_CLASSES when `names` is None. Returns [] when there are no boxes.
    """
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.detach().cpu().numpy()
    cls = boxes.cls.detach().cpu().numpy().astype(int)
    conf = boxes.conf.detach().cpu().numpy()

    def _name(c: int) -> str:
        if names is not None:
            n = names.get(c) if isinstance(names, dict) else (
                names[c] if 0 <= c < len(names) else None)
            if n is not None:
                return n
        return COCO_CLASSES[c] if 0 <= c < len(COCO_CLASSES) else str(c)

    out: list[Detection] = []
    for (x1, y1, x2, y2), c, p in zip(xyxy, cls, conf):
        out.append(Detection((float(x1), float(y1), float(x2), float(y2)),
                             int(c), _name(int(c)), float(p)))
    return out


class ObjectDetectorBase(ABC):
    """Abstract base every object detector implements."""

    @abstractmethod
    def detect(self, rgb) -> list[Detection]:
        """
        Detect objects in one frame.

        Args:
            rgb: (H, W, 3) or (3, H, W); torch/numpy; float [0,1] or uint8.

        Returns:
            List of `Detection` (possibly empty).
        """
        raise NotImplementedError
