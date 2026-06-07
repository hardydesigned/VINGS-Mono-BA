"""
YOLO object detector (via the `ultralytics` package).

Returns class-labelled bounding boxes (COCO by default) so each detection can
be named and localised in 3D by `object_tracker`. Weights resolve from a normal
repo-relative checkpoint (`ckpts/yolov8n.pt`, same convention as
`ckpts/FastSAM-x.pt`); if the file is missing, ultralytics auto-downloads the
named asset.

Standalone smoketest (mirrors the segmentation-backend smoketests):

    python scripts/vings_utils/yolo_detector.py [image.jpg]
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

try:  # repo runs scripts/ on sys.path; support both layouts.
    from vings_utils.detector_base import (
        ObjectDetectorBase, Detection, boxes_to_detections, to_uint8_rgb)
    from vings_utils.detector_factory import register_detector
except ImportError:  # pragma: no cover - standalone execution
    from detector_base import (
        ObjectDetectorBase, Detection, boxes_to_detections, to_uint8_rgb)
    from detector_factory import register_detector


# =============================================================================
# Config
# =============================================================================

@dataclass
class YoloConfig:
    model: str = "yolov8n"              # yolov8n (fast) ... yolov8x (accurate)
    ckpt_path: str = "ckpts/yolov8n.pt"
    conf: float = 0.35                  # confidence threshold
    iou: float = 0.7                    # NMS IoU
    imgsz: int = 640                    # inference resolution
    device: str = "cuda"
    classes: Optional[list] = None      # COCO ids to keep; None = all 80
    max_det: int = 100


# =============================================================================
# Detector
# =============================================================================

class YoloDetector(ObjectDetectorBase):
    """YOLO detection behind the ObjectDetectorBase contract."""

    def __init__(self, cfg: YoloConfig):
        self.cfg = cfg
        self.device = cfg.device
        self._model = None  # lazy: only load weights on first detect() call.

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg_dict: dict, device: str = "cuda") -> "YoloDetector":
        fields = set(YoloConfig.__dataclass_fields__.keys())
        kwargs = {k: v for k, v in cfg_dict.items() if k in fields}
        kwargs.setdefault("device", device)
        return cls(YoloConfig(**kwargs))

    # ------------------------------------------------------------------
    def _resolve_weights(self) -> str:
        """Prefer the repo-relative checkpoint; otherwise hand the bare model
        name to ultralytics so it can auto-download a known asset."""
        if self.cfg.ckpt_path and os.path.isfile(self.cfg.ckpt_path):
            return self.cfg.ckpt_path
        name = self.cfg.model
        return name if name.endswith(".pt") else f"{name}.pt"

    def _ensure_model(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self._resolve_weights())

    # ------------------------------------------------------------------
    @torch.no_grad()
    def detect(self, rgb) -> list[Detection]:
        self._ensure_model()
        img = to_uint8_rgb(rgb)                     # (H, W, 3) RGB uint8
        bgr = img[..., ::-1]                        # ultralytics numpy convention is BGR

        results = self._model(
            bgr,
            device=self.device,
            imgsz=self.cfg.imgsz,
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            classes=self.cfg.classes,
            max_det=self.cfg.max_det,
            verbose=False,
        )
        if not results:
            return []
        return boxes_to_detections(results[0].boxes, results[0].names)


# =============================================================================
# Registration + smoketest
# =============================================================================

@register_detector("yolo")
def _build(cfg_dict, device):
    return YoloDetector.from_config(cfg_dict, device)


if __name__ == "__main__":
    import sys

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    detector = YoloDetector(YoloConfig(device=dev))

    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        import cv2
        bgr = cv2.imread(sys.argv[1])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        src = sys.argv[1]
    else:
        # Synthetic frame: a few bright blobs (YOLO likely finds nothing -- the
        # point is to exercise load + detect + parse without crashing).
        rng = np.random.default_rng(0)
        rgb = (rng.random((480, 640, 3)) * 40).astype(np.uint8)
        for _ in range(5):
            cy, cx = rng.integers(60, 420), rng.integers(60, 580)
            rgb[cy - 30:cy + 30, cx - 30:cx + 30] = rng.integers(150, 255, 3)
        src = "<synthetic>"

    dets = detector.detect(rgb)
    print(f"[yolo smoketest] src={src} device={dev}")
    print(f"  {len(dets)} detections")
    for d in dets[:20]:
        x1, y1, x2, y2 = d.bbox_xyxy
        print(f"    {d.cls_name:<14} conf={d.conf:.2f} "
              f"box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")
    try:
        import cv2
        from detector_base import class_color
        overlay = to_uint8_rgb(rgb).copy()[..., ::-1].copy()
        for d in dets:
            x1, y1, x2, y2 = [int(v) for v in d.bbox_xyxy]
            col = class_color(d.cls_id)[::-1]  # RGB -> BGR
            cv2.rectangle(overlay, (x1, y1), (x2, y2), col, 2)
            cv2.putText(overlay, f"{d.cls_name} {d.conf:.2f}", (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        out = "/tmp/yolo_smoketest_overlay.png"
        cv2.imwrite(out, overlay)
        print(f"  overlay -> {out}")
    except Exception as e:
        print(f"  (overlay skipped: {e})")
