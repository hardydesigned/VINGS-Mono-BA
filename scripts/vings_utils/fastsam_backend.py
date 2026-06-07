"""
FastSAM segmentation backend (via the `ultralytics` package).

Replaces the dead, hardcoded CASIA-clone path that used to live in
`scripts/dynamic/dynamic_utils.py`. Model weights are resolved from a normal
repo-relative checkpoint (`ckpts/FastSAM-x.pt`, same convention as
`droid.pth` / `metric_depth_vit_small_800k.pth`); if the file is missing,
ultralytics auto-downloads the named asset.

Standalone smoketest (mirrors the selector smoketests):

    python scripts/vings_utils/fastsam_backend.py [image.jpg]
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

try:  # repo runs scripts/ on sys.path; support both layouts.
    from vings_utils.segmentation_base import SegmentationBackend, to_uint8_rgb
    from vings_utils.segmentation_factory import register_segmentation
except ImportError:  # pragma: no cover - standalone execution
    from segmentation_base import SegmentationBackend, to_uint8_rgb
    from segmentation_factory import register_segmentation


# =============================================================================
# Config
# =============================================================================

@dataclass
class FastSamConfig:
    model: str = "FastSAM-x"            # FastSAM-x (accurate) | FastSAM-s (fast)
    ckpt_path: str = "ckpts/FastSAM-x.pt"
    conf: float = 0.4                   # confidence threshold
    iou: float = 0.9                    # NMS IoU
    imgsz: int = 512                    # inference resolution
    device: str = "cuda"
    min_area_px: int = 0                # drop masks smaller than this (0 = keep all)


# =============================================================================
# Backend
# =============================================================================

class FastSamBackend(SegmentationBackend):
    """FastSAM 'everything' segmentation behind the SegmentationBackend contract."""

    def __init__(self, cfg: FastSamConfig):
        self.cfg = cfg
        self.device = cfg.device
        self._model = None  # lazy: only load weights on first segment() call.

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg_dict: dict, device: str = "cuda") -> "FastSamBackend":
        fields = set(FastSamConfig.__dataclass_fields__.keys())
        kwargs = {k: v for k, v in cfg_dict.items() if k in fields}
        kwargs.setdefault("device", device)
        return cls(FastSamConfig(**kwargs))

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
            from ultralytics import FastSAM
            self._model = FastSAM(self._resolve_weights())

    # ------------------------------------------------------------------
    @torch.no_grad()
    def segment(self, rgb) -> torch.Tensor:
        self._ensure_model()
        img = to_uint8_rgb(rgb)                     # (H, W, 3) RGB uint8
        H, W = img.shape[:2]
        bgr = img[..., ::-1]                        # ultralytics numpy convention is BGR

        results = self._model(
            bgr,
            device=self.device,
            retina_masks=True,
            imgsz=self.cfg.imgsz,
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            verbose=False,
        )

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


# =============================================================================
# Registration + smoketest
# =============================================================================

@register_segmentation("fastsam")
def _build(cfg_dict, device):
    return FastSamBackend.from_config(cfg_dict, device)


if __name__ == "__main__":
    import sys

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    backend = FastSamBackend(FastSamConfig(device=dev))

    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        import cv2
        bgr = cv2.imread(sys.argv[1])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        src = sys.argv[1]
    else:
        # Synthetic frame: a few bright blobs on a dark background.
        rng = np.random.default_rng(0)
        rgb = (rng.random((344, 616, 3)) * 40).astype(np.uint8)
        for _ in range(5):
            cy, cx = rng.integers(40, 300), rng.integers(40, 570)
            rgb[cy - 30:cy + 30, cx - 30:cx + 30] = rng.integers(150, 255, 3)
        src = "<synthetic>"

    masks = backend.segment(rgb)
    print(f"[fastsam smoketest] src={src} device={dev}")
    print(f"  masks: {tuple(masks.shape)}  (K, H, W)  dtype={masks.dtype}")
    if masks.shape[0] > 0:
        cover = masks.any(dim=0).float().mean().item()
        print(f"  {masks.shape[0]} instances, union covers {cover*100:.1f}% of frame")
        try:
            import cv2
            overlay = to_uint8_rgb(rgb).copy()[..., ::-1].copy()
            union = masks.any(dim=0).numpy()
            overlay[union] = (0.5 * overlay[union] + 0.5 *
                              np.array([0, 0, 255])).astype(np.uint8)
            out = "/tmp/fastsam_smoketest_overlay.png"
            cv2.imwrite(out, overlay)
            print(f"  overlay -> {out}")
        except Exception as e:
            print(f"  (overlay skipped: {e})")
