"""
Registry-based factory for keyframe selectors.

Configs choose a selector via `frame_selector.kind`:

    frame_selector:
        kind: nurbs_lvi          # or vista | mm3dgs | game_kfs | adaptive_kf | orbslam3 | coko_slam | aim_slam | two_gate | two_gate_v2 | none
        # ... selector-specific fields ...

To add a new selector:

    from vings_utils.selector_factory import register_selector

    @register_selector("my_selector")
    def _build_my_selector(cfg_dict, K, image_hw):
        from vings_utils.my_selector_module import MySelector
        return MySelector.from_config(cfg_dict, K, image_hw)

All selectors must expose

    should_accept(depth, t, R, rgb=None) -> tuple[bool, score_or_None]

`rgb` is passed through unconditionally; selectors that don't need it
(VISTA) just ignore it.

Legacy compatibility: `kind` is optional. If absent but `enabled: true`,
this falls back to the VISTA `FrameSelector` (old config behaviour).
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np


# selector_kind -> builder(cfg_dict, K, image_hw) -> selector instance
_REGISTRY: dict[str, Callable] = {}


def register_selector(kind: str):
    """Decorator: register a builder under a selector kind."""
    def deco(fn):
        _REGISTRY[kind] = fn
        return fn
    return deco


# -----------------------------------------------------------------------------
# Built-in registrations
# -----------------------------------------------------------------------------

@register_selector("vista")
def _build_vista(cfg_dict, K, image_hw):
    from vings_utils.frame_selector import FrameSelector
    return FrameSelector.from_config(cfg_dict, K, image_hw)


@register_selector("nurbs_lvi")
def _build_nurbs_lvi(cfg_dict, K, image_hw):
    from vings_utils.nurbs_lvi_selector import NurbsLviSelector
    return NurbsLviSelector.from_config(cfg_dict, K, image_hw)


@register_selector("mm3dgs")
def _build_mm3dgs(cfg_dict, K, image_hw):
    from vings_utils.mm3dgs_selector import Mm3dgsSelector
    return Mm3dgsSelector.from_config(cfg_dict, K, image_hw)


@register_selector("game_kfs")
def _build_game_kfs(cfg_dict, K, image_hw):
    from vings_utils.game_kfs_selector import GameKfsSelector
    return GameKfsSelector.from_config(cfg_dict, K, image_hw)


@register_selector("adaptive_kf")
def _build_adaptive_kf(cfg_dict, K, image_hw):
    from vings_utils.adaptive_kf_selector import AdaptiveKfSelector
    return AdaptiveKfSelector.from_config(cfg_dict, K, image_hw)


@register_selector("orbslam3")
def _build_orbslam3(cfg_dict, K, image_hw):
    from vings_utils.orbslam3_selector import OrbSlam3Selector
    return OrbSlam3Selector.from_config(cfg_dict, K, image_hw)


@register_selector("coko_slam")
def _build_coko_slam(cfg_dict, K, image_hw):
    from vings_utils.coko_slam_selector import CokoSlamSelector
    return CokoSlamSelector.from_config(cfg_dict, K, image_hw)


@register_selector("aim_slam")
def _build_aim_slam(cfg_dict, K, image_hw):
    from vings_utils.aim_slam_selector import AimSlamSelector
    return AimSlamSelector.from_config(cfg_dict, K, image_hw)


@register_selector("two_gate")
def _build_two_gate(cfg_dict, K, image_hw):
    from vings_utils.two_gate_selector import TwoGateSelector
    return TwoGateSelector.from_config(cfg_dict, K, image_hw)


@register_selector("two_gate_v2")
def _build_two_gate_v2(cfg_dict, K, image_hw):
    from vings_utils.two_gate_v2_selector import TwoGateV2Selector
    return TwoGateV2Selector.from_config(cfg_dict, K, image_hw)


# -----------------------------------------------------------------------------
# Public entry
# -----------------------------------------------------------------------------

def make_frame_selector(cfg: dict, K: np.ndarray, image_hw: tuple[int, int]):
    """
    Read `cfg['frame_selector']`, dispatch to the right builder, return a
    selector instance or None.

    Decision order:
      1. kind in registry  -> build that one
      2. kind == 'none'    -> None
      3. kind missing, enabled==True -> legacy: VISTA
      4. else              -> None
    """
    fs_cfg = (cfg.get("frame_selector") or {})
    kind = fs_cfg.get("kind")

    if kind is None:
        if fs_cfg.get("enabled", False):
            kind = "vista"
        else:
            return None

    if kind == "none":
        return None

    if kind not in _REGISTRY:
        raise ValueError(
            f"Unknown frame_selector.kind={kind!r}. "
            f"Known: {sorted(_REGISTRY.keys())}"
        )

    return _REGISTRY[kind](fs_cfg, K, image_hw)


def known_kinds() -> list[str]:
    return sorted(_REGISTRY.keys())
