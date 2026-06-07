"""
SAM3 segmentation backend (via the `ultralytics` package, text-grounded).

Unlike FastSAM / SAM2 -- which run prompt-free "segment everything" and let the
downstream dynamic-object logic flag high-error segments -- SAM3 ("Segment
Anything with Concepts", Meta, Nov 2025) is *concept-driven*: you hand it a list
of text class names and it returns instance masks for exactly those concepts.
For dynamic-object masking that is a natural fit: prompt SAM3 directly with the
movable-object classes (car, person, truck, ...) and the returned masks are the
candidate dynamics. The existing high-error filter (`compute_dynamic_mask`) then
still composes on top (a parked car won't be high-error; a moving one will).

Because the high-level `ultralytics.SAM("sam3*.pt")` wrapper only exposes SAM3's
*interactive* (point/box) predictor, the text path is driven through
`SAM3SemanticPredictor` directly (it is exported from `ultralytics.models.sam`).

----------------------------------------------------------------------------
WEIGHTS ARE GATED -- this backend cannot self-download (verified 2026-06-05)
----------------------------------------------------------------------------
SAM3 weights are NOT in the ultralytics asset index (unlike sam2.1_*.pt), so
`ultralytics` will not auto-fetch them. Meta's HF repo `facebook/sam3` is
`gated: manual` (HTTP 401 GatedRepo) and ships the *transformers* format, not the
ultralytics `.pt` layout. To use this backend you must obtain a checkpoint:

  Route A (ultralytics .pt, matches this file):
    Meta's raw SAM3 checkpoint (detector./tracker. keys) from the official
    facebookresearch/sam3 release -> save as ckpts/sam3.pt.

  Route B (HF transformers):
    1. Accept the license at https://huggingface.co/facebook/sam3 (manual
       approval) with your HF account.
    2. `pip install huggingface_hub` (not installed in the vings env) and
       `huggingface-cli login` with a token that has access.
    3. Download, then convert to the ultralytics .pt layout -- OR re-target this
       backend at the transformers `Sam3` API. (Route B needs a different
       inference path than the ultralytics one wired below.)

Until a checkpoint exists at `ckpt_path`, `_ensure_model()` raises a clear,
actionable error instead of silently doing nothing.

Standalone smoketest (mirrors fastsam/sam2; needs real weights):

    python scripts/vings_utils/sam3_backend.py [image.jpg]
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn.functional as F

try:  # repo runs scripts/ on sys.path; support both layouts.
    from vings_utils.segmentation_base import SegmentationBackend, to_uint8_rgb
    from vings_utils.segmentation_factory import register_segmentation
except ImportError:  # pragma: no cover - standalone execution
    from segmentation_base import SegmentationBackend, to_uint8_rgb
    from segmentation_factory import register_segmentation


# Default movable-object concepts for dynamic-object masking on street scenes.
_DEFAULT_CLASSES = ["car", "truck", "bus", "person", "bicycle", "motorcycle"]


@dataclass
class Sam3Config:
    model: str = "sam3"
    ckpt_path: str = "ckpts/sam3.pt"
    classes: List[str] = field(default_factory=lambda: list(_DEFAULT_CLASSES))
    conf: float = 0.4
    iou: float = 0.9
    imgsz: int = 1024
    device: str = "cuda"
    min_area_px: int = 0


class Sam3Backend(SegmentationBackend):
    """SAM3 concept/text-grounded segmentation behind the SegmentationBackend contract."""

    def __init__(self, cfg: Sam3Config):
        self.cfg = cfg
        self.device = cfg.device
        self._predictor = None  # lazy: only load weights on first segment() call.

    @classmethod
    def from_config(cls, cfg_dict: dict, device: str = "cuda") -> "Sam3Backend":
        fields = set(Sam3Config.__dataclass_fields__.keys())
        kwargs = {k: v for k, v in cfg_dict.items() if k in fields}
        kwargs.setdefault("device", device)
        return cls(Sam3Config(**kwargs))

    def _resolve_weights(self) -> str:
        if self.cfg.ckpt_path and os.path.isfile(self.cfg.ckpt_path):
            return self.cfg.ckpt_path
        # SAM3 is NOT an auto-downloadable ultralytics asset -- fail loudly.
        raise FileNotFoundError(
            f"SAM3 checkpoint not found at {self.cfg.ckpt_path!r}. SAM3 weights are "
            "gated and cannot be auto-downloaded (Meta `facebook/sam3` is "
            "gated:manual). See the module docstring in sam3_backend.py for how to "
            "obtain a checkpoint, or switch to `segmentation.kind: sam2`."
        )

    def _ensure_model(self):
        if self._predictor is None:
            # Text-grounded SAM3 -> SAM3SemanticPredictor (the high-level SAM(...)
            # wrapper only exposes the interactive point/box predictor).
            from ultralytics.models.sam import SAM3SemanticPredictor
            weights = self._resolve_weights()
            self._predictor = SAM3SemanticPredictor(
                overrides=dict(
                    model=weights,
                    task="segment",
                    mode="predict",
                    imgsz=self.cfg.imgsz,
                    conf=self.cfg.conf,
                    iou=self.cfg.iou,
                    device=self.device,
                    verbose=False,
                )
            )
            # Concepts to segment (the candidate dynamics).
            self._predictor.set_prompts({"text": list(self.cfg.classes)})

    @torch.no_grad()
    def segment(self, rgb) -> torch.Tensor:
        self._ensure_model()
        img = to_uint8_rgb(rgb)                     # (H, W, 3) RGB uint8
        H, W = img.shape[:2]
        bgr = img[..., ::-1]                        # ultralytics numpy convention is BGR

        results = self._predictor(source=bgr)

        empty = torch.zeros((0, H, W), dtype=torch.bool)
        if not results:
            return empty
        masks = results[0].masks
        if masks is None or masks.data is None or masks.data.shape[0] == 0:
            return empty

        m = masks.data.float()                      # (K, h, w) on device
        if m.shape[1:] != (H, W):                   # align to the input frame
            m = F.interpolate(m.unsqueeze(1), size=(H, W),
                              mode="nearest").squeeze(1)
        m = (m > 0.5)

        if self.cfg.min_area_px > 0:
            keep = m.flatten(1).sum(dim=1) >= self.cfg.min_area_px
            m = m[keep]

        return m.to(torch.bool).cpu()               # CPU: keep mapper VRAM free


@register_segmentation("sam3")
def _build(cfg_dict, device):
    return Sam3Backend.from_config(cfg_dict, device)


if __name__ == "__main__":
    import sys

    import numpy as np

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    backend = Sam3Backend(Sam3Config(device=dev))

    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        import cv2
        bgr = cv2.imread(sys.argv[1])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        src = sys.argv[1]
    else:
        rng = np.random.default_rng(0)
        rgb = (rng.random((344, 616, 3)) * 40).astype(np.uint8)
        src = "<synthetic>"

    try:
        masks = backend.segment(rgb)
        print(f"[sam3 smoketest] src={src} device={dev} classes={backend.cfg.classes}")
        print(f"  masks: {tuple(masks.shape)}  (K, H, W)  dtype={masks.dtype}")
    except FileNotFoundError as e:
        print(f"[sam3 smoketest] BLOCKED: {e}")
