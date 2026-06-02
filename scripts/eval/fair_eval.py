"""
fair_eval.py -- selection-independent evaluation for VINGS-Mono runs.

Why this exists
---------------
The default `metrics.json` PSNR is computed on the `rgbdnua/` images, which are
saved *only for the frames a selector chose to map*, rendered in the very last
training iteration on that exact frame (train-view). That makes PSNR
**selection-dependent**: a selector that maps 3 frames and one that maps 20
frames are scored on different frame sets, with different sample counts, on
memorised views. It is not a fair cross-selector comparison.

This module provides two selection-INDEPENDENT metrics, computed post-run from
the final Gaussian map (still GPU-resident) and a fixed evaluation protocol:

1. ATE (tracking quality)
   Sim(3)-aligned Absolute Trajectory Error of the estimated keyframe poses
   against the GT trajectory (Umeyama alignment, scale included so the arbitrary
   mono-SLAM scale is absorbed). Independent of how many frames were mapped.

2. Held-out novel-view PSNR/SSIM/LPIPS (mapping quality)
   A FIXED set of dataset frame indices (same for every config, e.g. every
   `stride`-th frame of the slice). Each eval frame is rendered from the final
   map at the system's OWN estimated camera pose, interpolated (SLERP + lerp)
   in the native SLAM frame to that frame index, and compared to the GT input
   image. We deliberately do NOT render from GT poses: the map is only
   self-consistent with the estimated camera orientations, and the DJI/GT pose
   convention differs (positions align under Sim(3), orientations do not), so
   GT-pose rendering produces black images. Estimated-pose rendering on a fixed
   frame set is the standard SLAM-GS rendering metric (cf. MonoGS / SplaTAM):
   same frames + count for every config, novel-view (interpolated, not the
   frame's training step), and pose drift is correctly penalised. GT is still
   used for ATE and as the comparison image -> fair across selectors.

Coordinate conventions (verified against middleware_utils.judge_and_package_v3)
-------------------------------------------------------------------------------
* `video.poses[i]` is a **w2c** tq = [tx,ty,tz,qx,qy,qz,qw] (lietorch SE3 order).
* `tq_to_matrix(tq) = SE3(tq).matrix()` is the **w2c** 4x4; `c2w = inv(w2c)`.
* The mapper renders with `w2c = SE3(w2c_tq).matrix()` directly, so the map's
  world frame is exactly `c2w = inv(SE3(video.poses[i]).matrix())`.
* `dji_poses_all_w2c.txt` is the same w2c TUM format; GT pose for slice index
  `s` is row `start_frame + s` (camstamp_all line N == image 00000N.jpg).

Public entry point
------------------
    metrics = run_fair_eval(mapper, video, cfg, intrinsic_dict, save_dir)

Returns a dict (also written to `<save_dir>/fair_metrics.json`):
    ate_rmse_m, ate_mean_m, n_ate_pairs, n_tracked,
    psnr_ho, ssim_ho, lpips_ho, n_eval_ho, eval_stride
"""
from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import torch

try:
    from lietorch import SE3
except Exception as e:  # pragma: no cover
    raise ImportError("fair_eval requires lietorch (same as the tracker).") from e

try:
    import cv2
except Exception as e:  # pragma: no cover
    raise ImportError("fair_eval requires opencv-python.") from e

from skimage.metrics import peak_signal_noise_ratio, structural_similarity


# =============================================================================
# Pose helpers
# =============================================================================

def _w2c_tq_to_c2w_np(tq: np.ndarray) -> np.ndarray:
    """w2c tq [tx,ty,tz,qx,qy,qz,qw] -> 4x4 c2w numpy matrix."""
    t = torch.as_tensor(tq, dtype=torch.float64).reshape(1, 7)
    w2c = SE3(t).matrix()[0].cpu().numpy()          # 4x4 w2c
    return np.linalg.inv(w2c)                        # c2w


def load_gt_w2c_tq(metadata_dir: str,
                   fname: str = "dji_poses_all_w2c.txt") -> np.ndarray:
    """Load GT w2c poses, indexed by ABSOLUTE image index.

    Returns array (N, 7) of [tx,ty,tz,qx,qy,qz,qw]; row N == image 00000N.jpg.
    """
    path = os.path.join(metadata_dir, fname)
    arr = np.loadtxt(path, comments="#")             # (N, 8): ts + tq
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr[:, 1:].astype(np.float64)             # drop timestamp


# =============================================================================
# Sim(3) Umeyama alignment
# =============================================================================

def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """Least-squares similarity transform mapping src -> dst.

    src, dst: (N, 3). Returns (s, R, t) with  dst ~= s * R @ src + t.
    Standard Umeyama (1991).
    """
    assert src.shape == dst.shape and src.shape[1] == 3
    n = src.shape[0]
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    sc = src - mu_s
    dc = dst - mu_d
    cov = (dc.T @ sc) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    if with_scale:
        var_s = (sc ** 2).sum() / n
        s = float((D * np.diag(S)).sum() / var_s) if var_s > 0 else 1.0
    else:
        s = 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


# =============================================================================
# Estimated-pose collection
# =============================================================================

def collect_est_w2c_tq(video) -> dict[int, np.ndarray]:
    """Map slice-relative frame index -> estimated w2c tq (7,).

    Combines marginalised KFs (poses_save/tstamp_save) and the active window
    (poses/tstamp); on duplicate tstamps the later (active) one wins.
    """
    out: dict[int, np.ndarray] = {}

    n_save = int(getattr(video, "count_save", 0) or 0)
    if n_save > 0:
        ps = video.poses_save[:n_save].detach().cpu().numpy()
        ts = video.tstamp_save[:n_save].detach().cpu().numpy()
        for i in range(n_save):
            out[float(ts[i])] = ps[i].astype(np.float64)

    counter = getattr(video, "counter", None)
    n_act = int(counter.value) if counter is not None else 0
    n_act = max(0, min(n_act, video.poses.shape[0]))
    if n_act > 0:
        pa = video.poses[:n_act].detach().cpu().numpy()
        ta = video.tstamp[:n_act].detach().cpu().numpy()
        for i in range(n_act):
            out[float(ta[i])] = pa[i].astype(np.float64)

    return out


def _est_keys_to_slice(est: dict, ds: dict, start_frame: int) -> dict[int, np.ndarray]:
    """Map est-dict keys (frame tstamps) to slice-relative integer indices.

    VO-Modus: tstamps sind bereits slice-relative Frame-Indizes (0,1,2,...).
    VIO-Modus: generic_vo setzt tstamps = echte Unix-Zeit (muss zur IMU passen);
    diese ueber camstamp_all auf den absoluten Frame-Index und dann auf den
    slice-Index (abs_idx - start_frame) zuruecksetzen. Ohne das laeuft abs_idx
    out-of-bounds und ATE/Render werden uebersprungen.
    """
    if not est:
        return {}
    if max(est.keys()) < 1e8:
        return {int(round(k)): v for k, v in est.items()}
    cam_file = ds.get("camstamp_file")
    if not cam_file or not os.path.exists(cam_file):
        print("[fair_eval] WARN: Unix-tstamps aber kein camstamp_file -- "
              "kann nicht auf Frame-Index mappen; ATE/Render uebersprungen.")
        return {}
    cam_t = np.loadtxt(cam_file, comments="#", usecols=(0,)).astype(np.float64)
    out: dict[int, np.ndarray] = {}
    for k, v in est.items():
        abs_idx = int(np.argmin(np.abs(cam_t - float(k))))
        out[abs_idx - start_frame] = v
    return out


# =============================================================================
# Metrics
# =============================================================================

def _to_uint8_chw(rgb: torch.Tensor) -> np.ndarray:
    """(3,H,W) float [0,1] tensor -> (H,W,3) uint8 RGB numpy."""
    img = rgb.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return (img * 255.0 + 0.5).astype(np.uint8)


def run_fair_eval(
    mapper,
    video,
    cfg: dict,
    intrinsic_dict: dict,
    save_dir: str,
    *,
    gt_metadata_dir: Optional[str] = None,
    eval_stride: Optional[int] = None,
    save_renders: bool = True,
    use_lpips: bool = True,
) -> dict:
    """Compute Sim(3)-ATE + held-out novel-view metrics for a finished run.

    Parameters are read from `cfg['fair_eval']` when not passed explicitly:
        gt_metadata_dir : dir holding dji_poses_all_w2c.txt (+ images via dataset)
        eval_stride     : evaluate every Nth slice frame (default 10)
    """
    fe = dict(cfg.get("fair_eval", {}) or {})
    ds = cfg.get("dataset", {}) or {}

    start_frame = int(ds.get("start_frame", 0))
    n_frames = int(cfg.get("max_frames", ds.get("max_frames", 0)) or 0)
    if n_frames <= 0:
        n_frames = len(_safe_listdir_images(ds))

    if gt_metadata_dir is None:
        gt_metadata_dir = fe.get("gt_metadata_dir")
        if gt_metadata_dir is None:
            root = ds.get("root", "")
            gt_metadata_dir = os.path.join(root, "metadata")
    gt_fname = fe.get("gt_poses_file", "dji_poses_all_w2c.txt")
    if eval_stride is None:
        eval_stride = int(fe.get("eval_stride", 10))

    images_dir = os.path.join(ds.get("root", ""), ds.get("image_dir", "images_all"))
    img_ext = ds.get("image_ext", "*.jpg").lstrip("*").lstrip(".") or "jpg"

    result: dict = {
        "ate_rmse_m": None, "ate_mean_m": None, "n_ate_pairs": 0,
        "n_tracked": 0, "psnr_ho": None, "ssim_ho": None, "lpips_ho": None,
        "n_eval_ho": 0, "eval_stride": eval_stride,
        "start_frame": start_frame, "n_frames": n_frames,
    }

    # ---- GT poses ----------------------------------------------------------
    try:
        gt_w2c = load_gt_w2c_tq(gt_metadata_dir, gt_fname)   # (M, 7) abs-indexed
    except Exception as e:
        print(f"[fair_eval] GT poses unavailable ({e}); skipping fair eval.")
        _write(result, save_dir)
        return result

    # ---- Estimated poses + Sim(3) alignment -------------------------------
    est = collect_est_w2c_tq(video)
    # VIO-Modus: tstamps sind Unix-Zeit -> auf slice-relative Frame-Indizes mappen
    # (VO bleibt unveraendert, da dort schon Indizes vorliegen).
    est = _est_keys_to_slice(est, ds, start_frame)
    result["n_tracked"] = len(est)

    est_centers, gt_centers = [], []
    for s_idx, w2c_tq in est.items():
        abs_idx = start_frame + s_idx
        if abs_idx < 0 or abs_idx >= gt_w2c.shape[0]:
            continue
        est_centers.append(_w2c_tq_to_c2w_np(w2c_tq)[:3, 3])
        gt_centers.append(_w2c_tq_to_c2w_np(gt_w2c[abs_idx])[:3, 3])

    sim = None
    if len(est_centers) >= 3:
        est_centers = np.asarray(est_centers)
        gt_centers = np.asarray(gt_centers)
        # Align estimate -> GT so ATE is reported in GT (metric-ish) units.
        s, R, t = umeyama(est_centers, gt_centers, with_scale=True)
        aligned = (s * (R @ est_centers.T)).T + t
        err = np.linalg.norm(aligned - gt_centers, axis=1)
        result["ate_rmse_m"] = float(np.sqrt((err ** 2).mean()))
        result["ate_mean_m"] = float(err.mean())
        result["n_ate_pairs"] = int(len(err))
        sim = (s, R, t)
        print(f"[fair_eval] ATE rmse={result['ate_rmse_m']:.3f} m "
              f"over {result['n_ate_pairs']} KFs (scale={s:.4f}).")
    else:
        print(f"[fair_eval] only {len(est_centers)} matched KFs; ATE skipped.")

    # ---- Held-out novel-view rendering ------------------------------------
    # We render the FIXED eval frames from the system's OWN estimated trajectory
    # (interpolated in the native SLAM frame to the fixed frame indices), NOT
    # from the GT poses. Rationale: the map lives in the SLAM frame and is only
    # self-consistent with the estimated camera orientations; the DJI/GT pose
    # convention differs (positions align under Sim(3), orientations do not),
    # so GT-pose rendering yields black images. Estimated-pose rendering at a
    # fixed frame set is the standard SLAM-GS rendering metric (cf. MonoGS /
    # SplaTAM): identical frames for every config, novel-view (interpolated, not
    # the training step), and pose drift is correctly penalised because a config
    # with worse tracking renders the same frame from a worse pose. GT is still
    # used for ATE (above) and as the comparison image (below).
    if len(est) >= 2 and n_frames > 0:
        from scipy.spatial.transform import Rotation, Slerp

        items = sorted(est.items())                       # (slice_idx, w2c_tq)
        key_times = np.array([k for k, _ in items], dtype=float)
        c2ws = [_w2c_tq_to_c2w_np(v) for _, v in items]
        key_rots = Rotation.from_matrix(np.array([c[:3, :3] for c in c2ws]))
        key_trans = np.array([c[:3, 3] for c in c2ws])
        slerp = Slerp(key_times, key_rots)

        def est_c2w_at(s_idx: float) -> np.ndarray:
            s_idx = float(np.clip(s_idx, key_times[0], key_times[-1]))
            c = np.eye(4)
            c[:3, :3] = slerp(s_idx).as_matrix()
            c[:3, 3] = [np.interp(s_idx, key_times, key_trans[:, j]) for j in range(3)]
            return c

        net = None
        if use_lpips:
            try:
                import lpips as lpips_lib
                # CPU: avoids extra VRAM at end-of-run when the map is largest
                # (s1000 runs sit near the 8 GB watchdog). 20 frames is cheap.
                net = lpips_lib.LPIPS(net="alex", verbose=False)
            except Exception as e:
                print(f"[fair_eval] LPIPS unavailable ({e}); skipping LPIPS.")
                net = None

        H = int(intrinsic_dict["H"]); W = int(intrinsic_dict["W"])
        out_dir = os.path.join(save_dir, "fair_eval")
        if save_renders:
            os.makedirs(out_dir, exist_ok=True)

        psnrs, ssims, lpipss = [], [], []
        eval_idxs = list(range(0, n_frames, max(1, eval_stride)))
        for s_idx in eval_idxs:
            abs_idx = start_frame + s_idx
            if abs_idx < 0 or abs_idx >= gt_w2c.shape[0]:
                continue
            img_path = os.path.join(images_dir, f"{abs_idx:06d}.{img_ext}")
            gt_img = cv2.imread(img_path)
            if gt_img is None:
                continue
            gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
            gt_img = cv2.resize(gt_img, (W, H), interpolation=cv2.INTER_AREA)

            # Render from the estimated trajectory (interpolated) in SLAM frame.
            c2w_slam = est_c2w_at(s_idx)
            w2c = np.linalg.inv(c2w_slam)
            w2c_t = torch.as_tensor(w2c, dtype=torch.float32, device="cuda")

            with torch.no_grad():
                pred = mapper.render(w2c_t, intrinsic_dict)
            pred_rgb = _to_uint8_chw(pred["rgb"])

            psnrs.append(peak_signal_noise_ratio(gt_img, pred_rgb, data_range=255))
            ssims.append(structural_similarity(gt_img, pred_rgb,
                                                channel_axis=2, data_range=255))
            if net is not None:
                def _t(im):
                    return (torch.from_numpy(im).float().permute(2, 0, 1)
                            .unsqueeze(0) / 127.5 - 1.0)   # CPU
                with torch.no_grad():
                    lpipss.append(float(net(_t(pred_rgb), _t(gt_img)).item()))

            if save_renders:
                side = np.concatenate([gt_img, pred_rgb], axis=1)
                cv2.imwrite(os.path.join(out_dir, f"ho_{abs_idx:06d}.png"),
                            cv2.cvtColor(side, cv2.COLOR_RGB2BGR))

        if psnrs:
            result["psnr_ho"] = round(float(np.mean(psnrs)), 4)
            result["ssim_ho"] = round(float(np.mean(ssims)), 4)
            result["lpips_ho"] = (round(float(np.mean(lpipss)), 4)
                                  if lpipss else None)
            result["n_eval_ho"] = len(psnrs)
            print(f"[fair_eval] held-out PSNR={result['psnr_ho']} "
                  f"SSIM={result['ssim_ho']} LPIPS={result['lpips_ho']} "
                  f"over {result['n_eval_ho']} fixed frames (stride={eval_stride}).")

    _write(result, save_dir)
    return result


def _safe_listdir_images(ds: dict) -> int:
    try:
        d = os.path.join(ds.get("root", ""), ds.get("image_dir", "images_all"))
        return len([f for f in os.listdir(d) if f.lower().endswith(".jpg")])
    except Exception:
        return 0


def _write(result: dict, save_dir: str) -> None:
    try:
        with open(os.path.join(save_dir, "fair_metrics.json"), "w") as f:
            json.dump(result, f, indent=2)
    except Exception as e:
        print(f"[fair_eval] could not write fair_metrics.json: {e}")
