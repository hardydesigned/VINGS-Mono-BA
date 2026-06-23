"""Serialize VINGS 2D-Gaussians into the antimatter15 ``.splat`` web format.

One Gaussian == 32 bytes, packed exactly like the antimatter15/splat viewer
(and gsplat.js) expects::

    offset  bytes  field
    0       12     position   3 x float32  (little-endian, world units)
    12      12     scale      3 x float32  (linear world-space stddev, exp(_scaling))
    24       4     rgba       4 x uint8    (R,G,B from _rgb in [0,1]; A from sigmoid(_opacity))
    28       4     rot        4 x uint8    (quaternion w,x,y,z normalized -> round(q*128+128))

Notes on VINGS specifics
------------------------
* ``_rgb`` is stored *directly* as RGB in roughly [0,1] -- there is no SH stage,
  so no ``0.5 + C0*f_dc`` conversion (unlike a raw 3DGS .ply -> .splat path).
* 2DGS keeps only **two** scale axes (``_scaling`` is (N,2) in log-space). The
  third splat axis is synthesized small but non-zero (``flat_scale_eps``); a
  literal zero degenerates the covariance and makes web viewers emit NaNs.
* The quaternion is written in VINGS/3DGS native order ``(w, x, y, z)`` -- the
  same order the antimatter15 ``convert.py`` writes ``rot_0..rot_3``. If your
  frontend decodes as ``(x, y, z, w)`` it must reorder; the bundled
  ``static/test_viewer.html`` decodes ``(w, x, y, z)`` to stay self-consistent.

The two ``encode_splat_from_*`` helpers exist because the mapper (GPU, float32,
*activated* via ``get_property``) and the StorageManager (CPU, float16, **raw**
parameters) hold the Gaussians in different states.
"""

from __future__ import annotations

import numpy as np

BYTES_PER_SPLAT = 32


# ---------------------------------------------------------------------------
# core packer
# ---------------------------------------------------------------------------
def _pad_scale(scale2: np.ndarray, flat_scale_eps: float = 1e-3) -> np.ndarray:
    """(N,2) linear world-space scales -> (N,3) with a small flat third axis.

    The third axis is ``flat_scale_eps`` times the per-splat min of the two
    real axes (so it scales with the disk), floored at ``flat_scale_eps`` itself
    so degenerate/zero disks still get a positive thickness.
    """
    scale2 = np.asarray(scale2, dtype=np.float32)
    if scale2.ndim != 2 or scale2.shape[1] != 2:
        raise ValueError(f"_pad_scale expects (N,2), got {scale2.shape}")
    third = np.maximum(scale2.min(axis=1) * flat_scale_eps, flat_scale_eps)
    return np.concatenate([scale2, third[:, None]], axis=1).astype(np.float32)


def _select_indices(scale3: np.ndarray, max_n: int) -> np.ndarray:
    """Keep the ``max_n`` largest splats (by volume proxy) to limit bandwidth.

    Prioritising big splats over a plain stride reduces pop-in on the viewer.
    """
    vol = scale3.prod(axis=1)
    # argpartition is O(N); take the top-max_n then sort those for stable order.
    idx = np.argpartition(vol, -max_n)[-max_n:]
    return np.sort(idx)


def _to_splat_bytes(xyz, scale3, rgb01, alpha01, quat_wxyz,
                    max_n: int | None = None) -> bytes:
    """Pack activated Gaussian attributes into ``N*32`` ``.splat`` bytes.

    Args:
        xyz:        (N,3) world positions.
        scale3:     (N,3) linear world-space scales.
        rgb01:      (N,3) colour in [0,1].
        alpha01:    (N,) opacity in [0,1].
        quat_wxyz:  (N,4) normalized quaternion, (w,x,y,z) order.
        max_n:      optional cap; keeps the largest splats.
    """
    xyz = np.ascontiguousarray(xyz, dtype='<f4')
    scale3 = np.ascontiguousarray(scale3, dtype='<f4')
    rgb01 = np.asarray(rgb01, dtype=np.float32)
    alpha01 = np.asarray(alpha01, dtype=np.float32).reshape(-1)
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float32)

    n = xyz.shape[0]
    if max_n is not None and n > max_n:
        idx = _select_indices(scale3, max_n)
        xyz, scale3 = xyz[idx], scale3[idx]
        rgb01, alpha01, quat_wxyz = rgb01[idx], alpha01[idx], quat_wxyz[idx]
        n = max_n

    buf = np.zeros((n, BYTES_PER_SPLAT), dtype=np.uint8)
    buf[:, 0:12] = np.ascontiguousarray(xyz).view(np.uint8).reshape(n, 12)
    buf[:, 12:24] = np.ascontiguousarray(scale3).view(np.uint8).reshape(n, 12)

    buf[:, 24:27] = np.clip(rgb01 * 255.0, 0, 255).astype(np.uint8)
    buf[:, 27] = np.clip(alpha01 * 255.0, 0, 255).astype(np.uint8)

    # normalize quat defensively, then pack (w,x,y,z) native order.
    norm = np.linalg.norm(quat_wxyz, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    q = quat_wxyz / norm
    buf[:, 28:32] = np.clip(np.round(q * 128.0 + 128.0), 0, 255).astype(np.uint8)
    return buf.tobytes()


# ---------------------------------------------------------------------------
# source adapters
# ---------------------------------------------------------------------------
def encode_splat_from_mapper_kf(mapper, kf_id: int, max_n: int | None = None,
                                flat_scale_eps: float = 1e-3) -> bytes:
    """Encode Gaussians for one KF group (kf_id) from the live GPU mapper.

    Returns b"" when no Gaussians belong to that kf_id.
    """
    import torch
    mask = mapper._globalkf_id == kf_id
    if not mask.any():
        return b""
    xyz  = mapper.get_property('_xyz')[mask].detach().cpu().numpy()
    sc2  = mapper.get_property('_scaling')[mask].detach().cpu().numpy()
    rgb  = mapper.get_property('_rgb')[mask].detach().cpu().numpy()
    op   = mapper.get_property('_opacity')[mask].detach().cpu().numpy()
    quat = mapper.get_property('_rotation')[mask].detach().cpu().numpy()
    return _to_splat_bytes(xyz, _pad_scale(sc2, flat_scale_eps), rgb, op, quat, max_n=max_n)


def encode_splat_from_mapper(mapper, max_n: int | None = None,
                             flat_scale_eps: float = 1e-3) -> bytes:
    """Encode the live GPU mapper set. Uses activated ``get_property`` values."""
    xyz = mapper.get_property('_xyz').detach().cpu().numpy()
    sc2 = mapper.get_property('_scaling').detach().cpu().numpy()        # exp() applied
    rgb = mapper.get_property('_rgb').detach().cpu().numpy()
    op = mapper.get_property('_opacity').detach().cpu().numpy()         # sigmoid() applied
    quat = mapper.get_property('_rotation').detach().cpu().numpy()      # normalized
    if xyz.shape[0] == 0:
        return b""
    return _to_splat_bytes(xyz, _pad_scale(sc2, flat_scale_eps), rgb, op, quat,
                           max_n=max_n)


def encode_splat_from_storage(sm, mask, flat_scale_eps: float = 1e-3) -> bytes:
    """Encode a subset of the CPU StorageManager set selected by ``mask``.

    StorageManager tensors are **raw** float16 (no activation applied), so we
    replicate ``get_property``'s activations here: exp(scaling), sigmoid(opacity),
    normalize(rotation).
    """
    import torch

    if sm._xyz.shape[0] == 0 or int(mask.sum()) == 0:
        return b""
    xyz = sm._xyz[mask].detach().float().numpy()
    sc2 = torch.exp(sm._scaling[mask].detach().float()).numpy()
    rgb = sm._rgb[mask].detach().float().numpy()
    op = torch.sigmoid(sm._opacity[mask].detach().float()).numpy()
    quat = torch.nn.functional.normalize(sm._rotation[mask].detach().float(), dim=-1).numpy()
    return _to_splat_bytes(xyz, _pad_scale(sc2, flat_scale_eps), rgb, op, quat)


# ---------------------------------------------------------------------------
# point-cloud adapters (for the viser viewer: raw xyz + rgb, no .splat packing)
# ---------------------------------------------------------------------------
def _filter_points(xyz, rgb01, op01, scale2, max_n, min_opacity):
    """Drop near-transparent floaters, cap to max_n (keep biggest), pack rgb u8.

    Returns (xyz float32 (M,3), rgb uint8 (M,3)) or None if nothing survives.
    """
    op = op01.reshape(-1)
    finite = (np.isfinite(xyz).all(1) & np.isfinite(scale2).all(1)
              & np.isfinite(rgb01).all(1) & np.isfinite(op))
    keep = finite & (op >= float(min_opacity))
    xyz, rgb01, scale2 = xyz[keep], rgb01[keep], scale2[keep]
    n = xyz.shape[0]
    if n == 0:
        return None
    if max_n is not None and n > max_n:
        # keep the largest disks (area proxy) -> less pop-in, like the .splat path
        area = scale2.prod(axis=1)
        idx = np.sort(np.argpartition(area, -max_n)[-max_n:])
        xyz, rgb01 = xyz[idx], rgb01[idx]
    rgb_u8 = np.clip(np.asarray(rgb01) * 255.0, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(xyz, dtype=np.float32), np.ascontiguousarray(rgb_u8)


def _filter_attrs(xyz, rgb01, op01, scale2, quat, max_n, min_opacity, flat_scale_eps):
    """Floater-filter + max_n cap, returns full per-Gaussian attrs for viser.

    Returns (xyz f32 (M,3), rgb u8 (M,3), scale3 f32 (M,3), quat f32 (M,4) wxyz,
    opacity f32 (M,1)) or None. ``scale3`` has the 2DGS flat third axis padded in.
    """
    op = np.asarray(op01, dtype=np.float32).reshape(-1)
    # CRITICAL: drop non-finite Gaussians. 2DGS positions/scales can go NaN/Inf
    # during optimisation; a single NaN in the gaussian buffer crashes the WASM
    # splat sorter in the browser ("memory access out of bounds") and the WHOLE
    # scene blanks. Filter here so the viewer never sees a NaN.
    finite = (np.isfinite(xyz).all(1) & np.isfinite(scale2).all(1)
              & np.isfinite(quat).all(1) & np.isfinite(rgb01).all(1)
              & np.isfinite(op))
    keep = finite & (op >= float(min_opacity))
    xyz, rgb01, scale2, quat, op = (xyz[keep], rgb01[keep], scale2[keep],
                                    quat[keep], op[keep])
    n = xyz.shape[0]
    if n == 0:
        return None
    if max_n is not None and n > max_n:
        area = scale2.prod(axis=1)
        idx = np.sort(np.argpartition(area, -max_n)[-max_n:])
        xyz, rgb01, scale2, quat, op = (xyz[idx], rgb01[idx], scale2[idx],
                                        quat[idx], op[idx])
    scale3 = _pad_scale(scale2, flat_scale_eps)
    return (np.ascontiguousarray(xyz, np.float32),
            np.clip(np.asarray(rgb01) * 255.0, 0, 255).astype(np.uint8),
            np.ascontiguousarray(scale3, np.float32),
            np.ascontiguousarray(quat, np.float32),
            np.ascontiguousarray(op.reshape(-1, 1), np.float32))


def attrs_from_mapper_kf(mapper, kf_id: int, max_n=None, min_opacity: float = 0.0,
                         flat_scale_eps: float = 1e-3):
    """Full attrs (xyz, rgb, scale3, quat, opacity) for one live mapper KF group."""
    mask = mapper._globalkf_id == kf_id
    if not mask.any():
        return None
    xyz = mapper.get_property('_xyz')[mask].detach().cpu().numpy()
    sc2 = mapper.get_property('_scaling')[mask].detach().cpu().numpy()
    rgb = mapper.get_property('_rgb')[mask].detach().cpu().numpy()
    op = mapper.get_property('_opacity')[mask].detach().cpu().numpy()
    quat = mapper.get_property('_rotation')[mask].detach().cpu().numpy()
    return _filter_attrs(xyz, rgb, op, sc2, quat, max_n, min_opacity, flat_scale_eps)


def attrs_from_storage(sm, mask, max_n=None, min_opacity: float = 0.0,
                       flat_scale_eps: float = 1e-3):
    """Full attrs for a CPU StorageManager subset. Raw -> activate, detached."""
    import torch
    if sm._xyz.shape[0] == 0 or int(mask.sum()) == 0:
        return None
    xyz = sm._xyz[mask].detach().float().numpy()
    sc2 = torch.exp(sm._scaling[mask].detach().float()).numpy()
    rgb = sm._rgb[mask].detach().float().numpy()
    op = torch.sigmoid(sm._opacity[mask].detach().float()).numpy()
    quat = torch.nn.functional.normalize(sm._rotation[mask].detach().float(), dim=-1).numpy()
    return _filter_attrs(xyz, rgb, op, sc2, quat, max_n, min_opacity, flat_scale_eps)


def points_from_mapper_kf(mapper, kf_id: int, max_n=None, min_opacity: float = 0.0):
    """(xyz, rgb_u8) for one live GPU mapper KF group, or None."""
    mask = mapper._globalkf_id == kf_id
    if not mask.any():
        return None
    xyz = mapper.get_property('_xyz')[mask].detach().cpu().numpy()
    sc2 = mapper.get_property('_scaling')[mask].detach().cpu().numpy()
    rgb = mapper.get_property('_rgb')[mask].detach().cpu().numpy()
    op = mapper.get_property('_opacity')[mask].detach().cpu().numpy()
    return _filter_points(xyz, rgb, op, sc2, max_n, min_opacity)


def points_from_storage(sm, mask, max_n=None, min_opacity: float = 0.0):
    """(xyz, rgb_u8) for a CPU StorageManager subset, or None. Raw -> activate."""
    import torch
    if sm._xyz.shape[0] == 0 or int(mask.sum()) == 0:
        return None
    # StorageManager tensors inherit requires_grad from the mapper (gpu2cpu does
    # `concat(.., mapper._xyz[..].cpu().half())` and .cpu()/.half() keep the grad),
    # so detach before numpy() -- otherwise every frozen push raises and the map
    # never grows past the active GPU window.
    xyz = sm._xyz[mask].detach().float().numpy()
    sc2 = torch.exp(sm._scaling[mask].detach().float()).numpy()
    rgb = sm._rgb[mask].detach().float().numpy()
    op = torch.sigmoid(sm._opacity[mask].detach().float()).numpy()
    return _filter_points(xyz, rgb, op, sc2, max_n, min_opacity)
