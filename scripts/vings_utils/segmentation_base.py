"""
Common interface for image segmentation backends.

The segmentation slot is backend-agnostic: FastSAM today, SAM3 tomorrow. A
backend takes one RGB frame and returns a stack of binary instance masks. The
*use* of those masks (dynamic-object detection in `scripts/dynamic/dynamic_utils.py`)
lives elsewhere -- backends only know how to segment.

This is the contract a new backend (e.g. SAM3) must satisfy:

    masks = backend.segment(rgb)        # rgb -> (K, H, W) bool tensor

Register it with @register_segmentation("name") in its own module (mirroring
`selector_factory.py`), and it becomes selectable via `segmentation.kind`.
No change to the mapper, loss, or run loop is needed for a swap.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch


def to_uint8_rgb(rgb) -> np.ndarray:
    """
    Normalise an arbitrary RGB input to a contiguous (H, W, 3) uint8 array.

    Accepts torch tensors or numpy arrays, float in [0, 1] or uint8 in [0, 255],
    shaped (H, W, 3) or (3, H, W). Backends call this so callers don't have to
    care about the exact tensor convention used in the pipeline (`viz_out`
    images are float [0, 1] (H, W, 3)).
    """
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.detach().cpu().numpy()
    rgb = np.asarray(rgb)
    if rgb.ndim == 3 and rgb.shape[0] == 3 and rgb.shape[2] != 3:
        rgb = np.transpose(rgb, (1, 2, 0))  # (3,H,W) -> (H,W,3)
    if rgb.dtype != np.uint8:
        # Heuristic: float in [0,1] -> scale; anything else assume already 0..255.
        if rgb.max() <= 1.0 + 1e-6:
            rgb = rgb * 255.0
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(rgb[..., :3])


class SegmentationBackend(ABC):
    """Abstract base every segmentation backend implements."""

    @abstractmethod
    def segment(self, rgb) -> torch.Tensor:
        """
        Segment one frame.

        Args:
            rgb: (H, W, 3) or (3, H, W); torch/numpy; float [0,1] or uint8.

        Returns:
            Bool tensor (K, H, W) of K instance masks. K may be 0
            (shape (0, H, W)) when nothing is segmented.
        """
        raise NotImplementedError
