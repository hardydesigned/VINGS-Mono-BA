"""
Registry-based factory for the single-image depth model.

Configs choose a backend via `depth_model.kind`:

    depth_model:
        kind: metric3d          # currently the only backend
        variant: v2-S           # v2-S | v2-L | v2-g  (Metric3D backbone size)
        checkpoint: ckpts/metric_depth_vit_small_800k.pth

To add a new backend:

    from metric.depth_factory import register_depth_model

    @register_depth_model("my_backend")
    def _build_my_backend(cfg, u_scale, v_scale):
        from metric.my_backend import MyModel
        return MyModel(cfg, u_scale, v_scale)

All backends must expose

    predict(img) -> torch.Tensor  # (H, W), float32, on img.device

That is the only contract the depth cache (run.py) and the mapper
injection path rely on.

Backward compatibility: the `depth_model` block is optional. If absent,
`kind` defaults to "metric3d" and Metric_Model falls back to its historic
v2-S / small-checkpoint defaults — i.e. exactly the old behaviour.

Mirrors `vings_utils/selector_factory.py`.
"""

from __future__ import annotations

from typing import Callable, Optional


# depth_model_kind -> builder(cfg, u_scale, v_scale) -> model instance
_REGISTRY: dict[str, Callable] = {}


def register_depth_model(kind: str):
    """Decorator: register a builder under a depth-model kind."""
    def deco(fn):
        _REGISTRY[kind] = fn
        return fn
    return deco


# -----------------------------------------------------------------------------
# Built-in registrations
# -----------------------------------------------------------------------------

@register_depth_model("metric3d")
def _build_metric3d(cfg, u_scale, v_scale):
    # Lazy import: metric_model pulls heavy CUDA deps on construction.
    from metric.metric_model import Metric_Model
    return Metric_Model(cfg, u_scale, v_scale)


# -----------------------------------------------------------------------------
# Public entry points
# -----------------------------------------------------------------------------

def make_depth_model(cfg: dict, u_scale: Optional[float] = None, v_scale: Optional[float] = None):
    """Build the depth model selected by cfg['depth_model']['kind'].

    Defaults to 'metric3d' when the block is missing, preserving the old
    hardcoded behaviour.
    """
    dm = cfg.get("depth_model") or {}
    kind = dm.get("kind", "metric3d")
    if kind not in _REGISTRY:
        raise ValueError(
            f"Unknown depth_model.kind={kind!r}. Known: {known_depth_kinds()}"
        )
    return _REGISTRY[kind](cfg, u_scale, v_scale)


def known_depth_kinds() -> list[str]:
    return sorted(_REGISTRY.keys())
