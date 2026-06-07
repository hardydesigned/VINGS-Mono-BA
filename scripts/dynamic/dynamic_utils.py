"""
Dynamic-object masking for the Gaussian mapper.

Two pieces, decoupled:

  * Segmentation -- a swappable backend (FastSAM today, SAM3 tomorrow) selected
    via `cfg['segmentation'].kind`. See `vings_utils/segmentation_factory.py`.
    It only turns one RGB frame into K instance masks.

  * Dynamic detection -- `compute_dynamic_mask()` flags the instance masks whose
    pixels carry disproportionately high photometric (L1 x (1-SSIM)) error
    between the rendered and the ground-truth frame. Those pixels are the
    dynamic-object mask and get excluded from the mapping loss
    (`gaussian/loss_utils.get_loss`).

This replaces the old dead version that hardcoded `/data/wuke/workspace/FastSAM`
and was never called in the run loop.
"""

import os
import sys

import torch

# Ensure scripts/ (parent of dynamic/) is importable for both standalone runs
# and in-pipeline use. No-op when the run loop already put scripts/ on the path.
_SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from gaussian.loss_utils import ssim_img

try:
    from vings_utils.segmentation_factory import make_segmentation_backend
except ImportError:  # pragma: no cover - alternate sys.path layout
    from scripts.vings_utils.segmentation_factory import make_segmentation_backend


# Default dynamic-detection thresholds (overridable via cfg['segmentation']).
DEFAULT_LOSS_QUANTILE = 0.9     # pixels above this error quantile count as "high loss"
DEFAULT_HIGH_RATE = 0.2         # segment is dynamic if >20% of its pixels are high-loss ...
DEFAULT_MEAN_LOSS = 0.002       # ... AND its mean error exceeds this floor


def compute_dynamic_mask(raw_ann, gt_rgb, pred_rgb,
                         loss_quantile: float = DEFAULT_LOSS_QUANTILE,
                         high_rate: float = DEFAULT_HIGH_RATE,
                         mean_loss: float = DEFAULT_MEAN_LOSS) -> torch.Tensor:
    """
    Args:
        raw_ann:  (K, H, W) bool instance masks (any device), K may be 0/None.
        gt_rgb:   (3, H, W) ground-truth frame.
        pred_rgb: (3, H, W) rendered frame.

    Returns:
        (H, W) bool mask, True where a pixel belongs to a dynamic instance.
        All-False when nothing qualifies (no masking) -- the safe default.
    """
    device = gt_rgb.device
    H, W = gt_rgb.shape[-2:]
    empty = torch.zeros((H, W), dtype=torch.bool, device=device)

    if raw_ann is None or raw_ann.shape[0] == 0:
        return empty
    raw_ann = raw_ann.to(device=device, dtype=torch.bool)

    # Per-pixel photometric error (L1 x (1 - SSIM)), (H, W).
    rgb_l1 = torch.abs(pred_rgb - gt_rgb).mean(dim=0)
    rgb_ssim = 1.0 - ssim_img(pred_rgb, gt_rgb).mean(dim=0)
    multi_loss = rgb_l1 * rgb_ssim

    thr = torch.quantile(multi_loss, loss_quantile)
    high_loss_mask = multi_loss > thr                       # (H, W) bool

    # Flag segments that are mostly high-loss AND error-heavy on average.
    dynamic_idx = []
    for k in range(raw_ann.shape[0]):
        seg = raw_ann[k]
        n = int(seg.sum())
        if n == 0:
            continue
        rate = float(high_loss_mask[seg].sum()) / n
        if rate > high_rate and float(multi_loss[seg].mean()) > mean_loss:
            dynamic_idx.append(k)

    if not dynamic_idx:
        return empty

    return raw_ann[dynamic_idx].any(dim=0)                  # (H, W) bool


class DynamicModel:
    """Wraps a segmentation backend + the dynamic-detection logic."""

    def __init__(self, cfg, device: str = "cuda"):
        self.cfg = cfg
        seg_cfg = cfg.get("segmentation", {}) if isinstance(cfg, dict) else {}
        self.loss_quantile = seg_cfg.get("dyn_loss_quantile", DEFAULT_LOSS_QUANTILE)
        self.high_rate = seg_cfg.get("dyn_high_rate", DEFAULT_HIGH_RATE)
        self.mean_loss = seg_cfg.get("dyn_mean_loss", DEFAULT_MEAN_LOSS)

        self.backend = make_segmentation_backend(cfg, device)
        if self.backend is None:
            raise ValueError(
                "DynamicModel needs a segmentation backend, but "
                "cfg['segmentation'].kind is None/'none'."
            )

    # ------------------------------------------------------------------
    def get_anns_raw(self, gt_rgb):
        """gt_rgb: (H,W,3) or (3,H,W); returns (K, H, W) bool masks (CPU)."""
        return self.backend.segment(gt_rgb)

    def get_dynamic_mask(self, raw_ann, gt_rgb, pred_rgb) -> torch.Tensor:
        """Thin wrapper applying this model's configured thresholds."""
        return compute_dynamic_mask(
            raw_ann, gt_rgb, pred_rgb,
            loss_quantile=self.loss_quantile,
            high_rate=self.high_rate,
            mean_loss=self.mean_loss,
        )

    # ------------------------------------------------------------------
    # Optional offline pre-compute path (run once, load at train time).
    def generate_anns(self, idx_list, rgb_loader):
        """Pre-segment frames and cache masks to `{dataset.folder}/sam_anns/`.
        `rgb_loader(idx)` returns an (H,W,3) frame."""
        folder = self.cfg["dataset"]["folder"]
        out_dir = os.path.join(folder, "sam_anns")
        os.makedirs(out_dir, exist_ok=True)
        for idx in idx_list:
            raw_ann = self.backend.segment(rgb_loader(idx))
            torch.save(raw_ann, os.path.join(out_dir, f"{int(idx):06d}.pt"))

    def get_anns_load(self, idx):
        folder = self.cfg["dataset"]["folder"]
        return torch.load(os.path.join(folder, "sam_anns", f"{int(idx):06d}.pt"))


if __name__ == "__main__":
    import numpy as np

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = {"segmentation": {"kind": "fastsam", "model": "FastSAM-x", "device": dev}}
    model = DynamicModel(cfg, dev)

    # Synthetic frame with a few blobs.
    rng = np.random.default_rng(0)
    rgb = (rng.random((344, 616, 3)) * 40).astype(np.uint8)
    for _ in range(5):
        cy, cx = rng.integers(40, 300), rng.integers(40, 570)
        rgb[cy - 30:cy + 30, cx - 30:cx + 30] = rng.integers(150, 255, 3)

    raw_ann = model.get_anns_raw(rgb)
    print(f"[dynamic_utils smoketest] device={dev}  raw_ann={tuple(raw_ann.shape)}")

    gt = torch.from_numpy(rgb).float().permute(2, 0, 1).to(dev) / 255.0   # (3,H,W)
    pred = gt.clone()
    pred[:, 100:160, 100:160] += 0.6   # inject a high-error region (fake "dynamic")
    pred = pred.clamp(0, 1)

    dyn = model.get_dynamic_mask(raw_ann, gt, pred)
    print(f"  dynamic mask: {tuple(dyn.shape)} dtype={dyn.dtype} "
          f"dynamic_px={int(dyn.sum())} ({100*dyn.float().mean():.2f}%)")

    # K=0 hardening: empty masks must not crash and must mask nothing.
    empty = torch.zeros((0, 344, 616), dtype=torch.bool)
    dyn0 = model.get_dynamic_mask(empty, gt, pred)
    print(f"  K=0 -> dynamic_px={int(dyn0.sum())} (expected 0), no crash OK")
