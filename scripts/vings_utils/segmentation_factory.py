"""
Registry-based factory for segmentation backends.

Configs choose a backend via `segmentation.kind`:

    segmentation:
        kind: fastsam            # or sam3 | none
        model: FastSAM-x
        ckpt_path: ckpts/FastSAM-x.pt
        # ... backend-specific fields ...

To add a new backend (e.g. SAM3), register a builder in its own module:

    from vings_utils.segmentation_factory import register_segmentation

    @register_segmentation("sam3")
    def _build_sam3(cfg_dict, device):
        from vings_utils.sam3_backend import Sam3Backend
        return Sam3Backend.from_config(cfg_dict, device)

All backends expose `segment(rgb) -> (K, H, W) bool tensor`
(see `segmentation_base.SegmentationBackend`). Swapping FastSAM for SAM3 is a
one-line config change (`kind: sam3`); no mapper/loss/run-loop edits needed.
"""

from __future__ import annotations

from typing import Callable


# segmentation_kind -> builder(cfg_dict, device) -> backend instance
_REGISTRY: dict[str, Callable] = {}


def register_segmentation(kind: str):
    """Decorator: register a builder under a segmentation kind."""
    def deco(fn):
        _REGISTRY[kind] = fn
        return fn
    return deco


# -----------------------------------------------------------------------------
# Built-in registrations
# -----------------------------------------------------------------------------

@register_segmentation("fastsam")
def _build_fastsam(cfg_dict, device):
    from vings_utils.fastsam_backend import FastSamBackend
    return FastSamBackend.from_config(cfg_dict, device)


@register_segmentation("sam2")
def _build_sam2(cfg_dict, device):
    from vings_utils.sam2_backend import Sam2Backend
    return Sam2Backend.from_config(cfg_dict, device)


@register_segmentation("sam3")
def _build_sam3(cfg_dict, device):
    from vings_utils.sam3_backend import Sam3Backend
    return Sam3Backend.from_config(cfg_dict, device)


# -----------------------------------------------------------------------------
# Public entry
# -----------------------------------------------------------------------------

def make_segmentation_backend(cfg: dict, device: str = "cuda"):
    """
    Read `cfg['segmentation']`, dispatch to the right builder, return a backend
    instance or None.

    Decision order:
      1. kind in registry  -> build that one
      2. kind in (None, 'none') -> None
      3. else              -> ValueError
    """
    seg_cfg = (cfg.get("segmentation") or {})
    kind = seg_cfg.get("kind")

    if kind in (None, "none"):
        return None

    if kind not in _REGISTRY:
        raise ValueError(
            f"Unknown segmentation.kind={kind!r}. "
            f"Known: {sorted(_REGISTRY.keys())}"
        )

    # device override from config wins over the caller default.
    device = seg_cfg.get("device", device)
    return _REGISTRY[kind](seg_cfg, device)


def known_kinds() -> list[str]:
    return sorted(_REGISTRY.keys())
