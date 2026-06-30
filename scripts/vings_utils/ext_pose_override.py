"""Externe-Pose-Override (RTK/DJI) fuer Mapper + BA-Seed.

Aus scripts/run.py ausgelagert. `ExtPoseOverrider` kapselt die drei Stellen, an
denen externe Posen (`dataset.ext_poses`, TUM-w2c aus `ext_poses_file`) den
maßstabsverzerrten DROID-DBA-Output korrigieren:

- ``apply_to_vizout``   : nach judge_and_package, vor dem Mapper -- ersetzt
  ``viz_out['poses']`` durch RTK-c2w und skaliert depths/depths_cov um einen
  kumulativen ``sum(d_rtk)/sum(d_droid)``-Faktor (robust gegen DROID-Drift, der
  per-Pair zu Skalen-Spikes -> Rasterizer-Crash fuehrt; Clamp [0.1, 1000]).
- ``seed_video_poses``  : schreibt RTK-Posen + reskalierte Disparity in
  ``video.poses``/``video.disps`` als Anti-Scale-Collapse-Prior fuer die
  naechste BA-Iteration (optional via ``cfg['seed_video_with_ext_pose']``).
- ``override_poses_save``: Legacy-One-Shot-Override pro marginalisiertem Save-Slot.

Alle drei sind no-ops, wenn der Loader keine ``ext_poses`` anbietet. Der
Scale-State (kumulative Distanzsummen + gecachte Skala) lebt als Instanz-State,
sodass ``apply_to_vizout`` und ``seed_video_poses`` denselben Schaetzwert teilen.
"""

import numpy as np
import torch


class ExtPoseOverrider:
    def __init__(self, cfg, dataset):
        self.cfg = cfg
        self.dataset = dataset
        self._ext_pose_dist_sums = [0.0, 0.0]   # [sum_d_rtk, sum_d_droid]
        self._ext_pose_cached_scale = 1.0
        self._ext_pose_logged_apply_first = False
        self._ext_pose_call_count = 0
        self._seed_logged_first = False
        self._last_pose_override_idx = 0

    def apply_to_vizout(self, viz_out):
        """Ersetze viz_out['poses'] durch RTK c2w aus dataset.ext_poses, und
        skaliere viz_out['depths'] um den per-window Skalenfaktor.

        Rationale: judge_and_package_v3 liefert Posen aus DROID-DBA's lokalem
        Koordinatensystem -- auf Nadir-Aerial bricht der Maßstab (35× Shrink
        auf amtown03 dokumentiert). Override an dieser Stelle ist sauber:
        - keine Beruehrung von video.poses oder video.poses_save (kein BA-Risiko)
        - Mapper sieht konsistente RTK-Posen + RTK-skalierte Depths
        - alle KFs im selben aktiven Window haben einheitlichen Frame -> kein Mix

        Skala kommt aus dem Verhaeltnis konsekutiver Distanzen RTK / DROID-DBA
        im aktiven Window (Median ueber Paare). EMA-glaettung ueber Aufrufe
        gegen Burst-Rauschen. Bei n=1 oder degeneriertem droid-Window (alle
        Distanzen ~0) fallback auf letzte cached Skala.
        """
        if not (hasattr(self.dataset, 'ext_poses') and self.dataset.ext_poses is not None):
            return viz_out
        if viz_out is None or 'poses' not in viz_out:
            return viz_out

        tstamps = viz_out['viz_out_idx_to_f_idx']
        n = tstamps.shape[0]
        if n == 0:
            return viz_out

        from scipy.spatial.transform import Rotation as _R

        # 1) Gather RTK w2c [tx,ty,tz,qx,qy,qz,qw] for each KF in window.
        ext_arr = self.dataset.ext_poses
        rtk_tq = np.zeros((n, 7), dtype=np.float32)
        any_missing = False
        for i in range(n):
            fi = int(tstamps[i].item())
            if not (0 <= fi < len(ext_arr)):
                any_missing = True
                break
            rtk_tq[i] = ext_arr[fi]
        if any_missing:
            return viz_out  # bail; don't half-override

        # 2) Convert RTK w2c -> c2w 4x4 numpy.
        Rw2c = _R.from_quat(rtk_tq[:, 3:7]).as_matrix()        # (n,3,3)
        tw2c = rtk_tq[:, 0:3]                                  # (n,3)
        Rc2w = Rw2c.transpose(0, 2, 1)                         # (n,3,3)
        tc2w = -np.einsum('nij,nj->ni', Rc2w, tw2c)            # (n,3)
        rtk_c2w = np.zeros((n, 4, 4), dtype=np.float32)
        rtk_c2w[:, :3, :3] = Rc2w
        rtk_c2w[:, :3, 3]  = tc2w
        rtk_c2w[:, 3,  3]  = 1.0

        # 3) Scale from RTK/DROID consecutive-distance ratio over the ENTIRE
        # known trajectory so far (cumulative path lengths), not just per-pair.
        # Per-pair is unstable when DROID drifts and produces near-zero motion
        # while RTK reports normal motion -> ratio explodes (saw scale=460
        # crash gaussian rasterizer). Cumulative ratio is robust to local
        # zero-motion segments.
        #
        # Strategy: collect (d_rtk, d_droid) pairs over all calls into running
        # sums. Discard pairs where either distance is near zero. The scale =
        # sum(d_rtk) / sum(d_droid) is the global Procrustes-like scale.
        droid_xyz = viz_out['poses'][:, :3, 3].detach().cpu().numpy()  # (n,3)
        if n >= 2:
            d_rtk   = np.linalg.norm(np.diff(tc2w,     axis=0), axis=1)
            d_droid = np.linalg.norm(np.diff(droid_xyz, axis=0), axis=1)
            # Require meaningful motion in BOTH frames (filter hovering / drift).
            mask = (d_droid > 5e-3) & (d_rtk > 0.05)
            if mask.sum() > 0:
                # Use the LAST pair only -- avoids double-counting across calls
                # since adjacent calls share most KFs. The last pair is the
                # newest one not yet integrated.
                if mask[-1]:
                    self._ext_pose_dist_sums[0] += float(d_rtk[-1])
                    self._ext_pose_dist_sums[1] += float(d_droid[-1])

        sum_rtk, sum_droid = self._ext_pose_dist_sums
        if sum_droid > 0.01:  # need enough cumulative motion
            scale = sum_rtk / sum_droid
        else:
            scale = self._ext_pose_cached_scale
        # Hard clamp -- amtown03 measured ratio is ~327x (cf. drift_diagnostic
        # plot v5/v9), Bell412/smaller scenes are deutlich darunter. Wir
        # erlauben grosszuegig bis 1000x, damit Aerial-Nadir nicht aufs
        # alte 100x-Limit clipt (das hat depths um Faktor 3 unter-skaliert).
        scale = float(np.clip(scale, 0.1, 1000.0))
        self._ext_pose_cached_scale = scale
        scale_est = scale  # for logging compatibility
        hist = self._ext_pose_dist_sums  # for log formatting

        device = viz_out['poses'].device
        dtype  = viz_out['poses'].dtype
        viz_out['poses'] = torch.from_numpy(rtk_c2w).to(device=device, dtype=dtype)
        viz_out['depths'] = viz_out['depths'] * scale
        if 'depths_cov' in viz_out and viz_out['depths_cov'] is not None:
            viz_out['depths_cov'] = viz_out['depths_cov'] * (scale * scale)

        if not self._ext_pose_logged_apply_first:
            print(f"[ext_pose] first viz_out apply: n_kfs={n} scale={scale:.4f} "
                  f"rtk_xyz_first={tc2w[0]} droid_xyz_first={droid_xyz[0]}")
            self._ext_pose_logged_apply_first = True
        # Log scale every K calls so we can spot spikes.
        c = self._ext_pose_call_count + 1
        self._ext_pose_call_count = c
        if c % 100 == 1 or scale > 200.0:
            est_str = f"{scale_est:.3f}" if scale_est is not None else "None"
            print(f"[ext_pose] call={c} n={n} scale_est={est_str} "
                  f"rolling={scale:.4f} hist_len={len(hist)}")

        return viz_out

    def seed_video_poses(self, tracker):
        """Schreibe RTK-Posen + rescale disparity in video.poses / video.disps,
        damit die naechste BA-Iteration von einer RTK-verankerten Skala
        startet. Greift nur wenn dataset.ext_poses_file gesetzt UND
        cfg['seed_video_with_ext_pose'] == True.

        Mechanik:
          1) Bestimme aktuelle Scale-Schaetzung (gleicher Algorithmus wie
             apply_to_vizout: kumulatives sum_rtk / sum_droid).
          2) Fuer jede aktive KF k im Active-Window (0..counter):
             - Setze video.poses[k] auf RTK c2w-pose (in TUM-tq w2c-Format)
             - Skaliere video.disps[k] um 1/scale (so dass Tiefe der RTK-Skala
               entspricht; disparity = 1/depth).

        Caveat: BA wird in den naechsten Aufrufen die Posen wieder optimieren
        und ggf. zurueck-driften. Aber jedes track() startet von der
        RTK-Seed -- der Scale kann nicht mehr unbeschraenkt collapsen.
        """
        if not self.cfg.get('seed_video_with_ext_pose', False):
            return
        if not (hasattr(self.dataset, 'ext_poses')
                and self.dataset.ext_poses is not None):
            return

        video = (tracker.video if hasattr(tracker, 'video')
                 else tracker.frontend.video)
        counter = video.counter.value if hasattr(video, 'counter') else 0
        if counter < 2:
            return  # zu frueh fuer scale-Schaetzung

        scale = float(self._ext_pose_cached_scale)
        if scale <= 0.0 or scale == 1.0:
            # Wir haben noch keine valide Scale-Messung -- skippe.
            return

        # Per-KF tstamp im DROID-Buffer ist der Dataset-Frame-Idx.
        tstamps = video.tstamp[:counter].detach().cpu().numpy().astype(np.int64)
        n_ext = len(self.dataset.ext_poses)
        # Build TUM-tq w2c poses (the format video.poses expects).
        seed = np.zeros((counter, 7), dtype=np.float32)
        seed[:, -1] = 1.0  # qw
        valid_mask = np.zeros(counter, dtype=bool)
        for k in range(counter):
            f = int(tstamps[k])
            if 0 <= f < n_ext:
                seed[k] = self.dataset.ext_poses[f]
                valid_mask[k] = True
        if not valid_mask.any():
            return

        device = video.poses.device
        dtype  = video.poses.dtype
        # Write seed where valid.
        seed_t = torch.from_numpy(seed[valid_mask]).to(device=device, dtype=dtype)
        idx_t  = torch.from_numpy(np.nonzero(valid_mask)[0]).to(device=device)
        video.poses[idx_t] = seed_t

        # Rescale disparity in proportion. disparity = 1/depth. RTK-scaled
        # depth = DROID_depth * scale -> RTK_disp = DROID_disp / scale.
        if hasattr(video, 'disps'):
            video.disps[:counter] = video.disps[:counter] / float(scale)

        if not self._seed_logged_first:
            print(f"[seed] erste video.poses-Seed: counter={counter} "
                  f"scale={scale:.2f} valid={int(valid_mask.sum())}/{counter}")
            self._seed_logged_first = True

    def override_poses_save(self, tracker):
        """Legacy poses_save override (one-shot per marginalized slot).

        Wenn der Tracker count_save inkrementiert hat (1+ neue KFs gefreezed),
        iteriere ueber alle neuen Save-Slots [_last_pose_override_idx, count_save)
        und schreibe pro Slot k die ext_pose fuer Frame tstamp_save[k]. tstamp_save[k]
        gibt den Original-Frame-Index fuer Slot k zurueck (die frueher verwendete
        data_packet['pose']-Variante hat fuer alle Slots die *aktuelle* Frame-Pose
        geschrieben -- falsch wenn der gefreezte KF N Frames zurueck liegt).

        Der eigentliche Mapper-Override passiert in apply_to_vizout; dieser Pfad
        haelt nur den History-Buffer konsistent.
        """
        if not (hasattr(self.dataset, 'ext_poses')
                and self.dataset.ext_poses is not None
                and self.cfg['mode'] != 'vo_nerfslam'):
            return
        video = tracker.video if hasattr(tracker, 'video') else tracker.frontend.video
        if hasattr(video, 'count_save'):
            cs = int(video.count_save)
            last_k = self._last_pose_override_idx
            if cs > last_k:
                for k in range(last_k, cs):
                    frame_idx = int(video.tstamp_save[k].item())
                    if 0 <= frame_idx < len(self.dataset.ext_poses):
                        video.poses_save[k] = torch.as_tensor(
                            self.dataset.ext_poses[frame_idx], dtype=torch.float32)
                self._last_pose_override_idx = cs
