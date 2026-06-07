"""
Registry-based factory for object detectors.

Configs choose a detector via `object_detector.kind`:

    object_detector:
        kind: yolo               # or rtdetr | none
        model: yolov8n
        ckpt_path: ckpts/yolov8n.pt
        classes: [2, 5, 7]       # COCO car/bus/truck; null = all
        # ... detector-specific fields ...

To add a new detector, register a builder in its own module (mirroring
`segmentation_factory.py`):

    from vings_utils.detector_factory import register_detector

    @register_detector("rtdetr")
    def _build_rtdetr(cfg_dict, device):
        from vings_utils.rtdetr_detector import RtdetrDetector
        return RtdetrDetector.from_config(cfg_dict, device)

All detectors expose `detect(rgb) -> list[Detection]`
(see `detector_base.ObjectDetectorBase`). Swapping YOLO for RT-DETR is a
one-line config change (`kind: rtdetr`); no run-loop edits needed.

The master gate is `cfg['detect_objects']` (checked by the caller in run.py);
`kind: none` / a missing block also yields no detector.
"""

from __future__ import annotations

from typing import Callable


# detector_kind -> builder(cfg_dict, device) -> detector instance
_REGISTRY: dict[str, Callable] = {}


def register_detector(kind: str):
    """Decorator: register a builder under a detector kind."""
    def deco(fn):
        _REGISTRY[kind] = fn
        return fn
    return deco


# -----------------------------------------------------------------------------
# Built-in registrations
# -----------------------------------------------------------------------------

@register_detector("yolo")
def _build_yolo(cfg_dict, device):
    from vings_utils.yolo_detector import YoloDetector
    return YoloDetector.from_config(cfg_dict, device)


@register_detector("rtdetr")
def _build_rtdetr(cfg_dict, device):
    from vings_utils.rtdetr_detector import RtdetrDetector
    return RtdetrDetector.from_config(cfg_dict, device)


# -----------------------------------------------------------------------------
# Public entry
# -----------------------------------------------------------------------------

def make_object_detector(cfg: dict, device: str = "cuda"):
    """
    Read `cfg['object_detector']`, dispatch to the right builder, return a
    detector instance or None.

    Decision order:
      1. kind in registry        -> build that one
      2. kind in (None, 'none')  -> None
      3. else                    -> ValueError
    """
    det_cfg = (cfg.get("object_detector") or {})
    kind = det_cfg.get("kind")

    if kind in (None, "none"):
        return None

    if kind not in _REGISTRY:
        raise ValueError(
            f"Unknown object_detector.kind={kind!r}. "
            f"Known: {sorted(_REGISTRY.keys())}"
        )

    # device override from config wins over the caller default.
    device = det_cfg.get("device", device)
    return _REGISTRY[kind](det_cfg, device)


def known_kinds() -> list[str]:
    return sorted(_REGISTRY.keys())
