"""Post-Run-Finalisierung des VINGS-Laufs.

Aus scripts/run.py ausgelagert. ``finalize_run`` laeuft einmal nach der
Frame-Schleife und erledigt: chunk-weisen PLY-Save, Fusion+Dump der online
detektierten Objekte, Stoppen des Stream-Servers, faire selektionsunabhaengige
Eval (Sim(3)-ATE + Held-out-PSNR), Dump der finalen Tracker-Posen (w2c) und das
Profiling-Summary inkl. finalem profiling.json.

Bekommt den ``Runner`` als ersten Parameter (greift auf dessen mapper/tracker/
storage_manager/stream/timer/cfg zu) plus die Loop-Counter als Keyword-Args.
"""

import os
import time

from gaussian.vis_utils import save_ply_streaming
from eval.fair_eval import run_fair_eval
from vings_utils.phase_timer import write_profiling_json


def finalize_run(runner, *, n_keyframes, n_mapped, n_processed, last_idx,
                 frame_skip, mapper_kf_skip, wall_t0):
    cfg = runner.cfg
    timer = runner.timer

    # PLY-Save: chunk-weise schreiben, um Peak-RAM zu minimieren.
    # save_ply_streaming iteriert über StorageManager (CPU) und Mapper (GPU)
    # getrennt in Chunks von 500k Gaussians (~120 MB/Chunk statt ~5 GB auf einmal).
    n_cpu = runner.storage_manager._xyz.shape[0] if runner.use_storage_manager else 0
    n_gpu = runner.mapper._xyz.shape[0]
    if n_cpu + n_gpu > 0:
        sm = runner.storage_manager if runner.use_storage_manager else None
        with timer.time('save_ply'):
            save_ply_streaming(runner.mapper, sm, len(runner.dataset) - 1, save_mode='2dgs')

    # Fuse + write the online object detections (objects_droid.csv,
    # object_markers_droid.ply, object_overlay.mp4). Markers live in the
    # same DROID frame as the map PLY just written above.
    if runner.object_tracker is not None:
        try:
            runner.object_tracker.finalize(cfg['output']['save_dir'])
        except Exception as _e:
            print(f"[object_tracker] finalize failed: {_e}")

    # WebSocket-Stream-Server stoppen (daemon-Thread; harmlos wenn aus).
    runner.stream.stop()

    # Faire, selektionsunabhaengige Eval (Sim(3)-ATE + Held-out-Novel-View-
    # PSNR an FIXEN Frame-Positionen aus der finalen Map). Gated ueber
    # cfg['fair_eval']['enabled']; Mapper ist hier noch GPU-resident.
    if (cfg.get('fair_eval', {}) or {}).get('enabled', False):
        try:
            video = (runner.tracker.video if hasattr(runner.tracker, 'video')
                     else runner.tracker.frontend.video)
            intr = getattr(runner, '_last_map_intrinsic', None)
            if intr is None:
                print('[fair_eval] no map intrinsic captured (no KF mapped?); skipping.')
            else:
                with timer.time('fair_eval'):
                    run_fair_eval(runner.mapper, video, cfg, intr,
                                  cfg['output']['save_dir'])
        except Exception as _e:
            import traceback
            print(f"[fair_eval] failed: {_e}")
            traceback.print_exc()

    # Dump finale Tracker-Posen (w2c in TUM-tq) fuer Drift-Diagnose.
    # Kombiniert MARGINALISIERTE KFs (poses_save[:count_save]) + ACTIVE-Window
    # (poses[:counter.value]) -- die zweite Quelle ist wichtig bei kurzen
    # Sequenzen wo viele KFs noch nicht marginalisiert wurden.
    try:
        video = (runner.tracker.video if hasattr(runner.tracker, 'video')
                 else runner.tracker.frontend.video)
        chunks = []
        # 1) Marginalisierte (append-only history)
        poses_save = video.poses_save.detach().cpu().numpy()
        n_save = int(getattr(video, 'count_save', 0))
        n_save = max(0, min(n_save, poses_save.shape[0]))
        if n_save > 0:
            chunks.append(('marg', poses_save[:n_save]))
        # 2) Active-Window-Posen (aktuelle BA-Schaetzungen)
        poses_act = video.poses.detach().cpu().numpy()
        counter_val = getattr(video, 'counter', None)
        n_act = int(counter_val.value) if counter_val is not None else 0
        n_act = max(0, min(n_act, poses_act.shape[0]))
        if n_act > 0:
            chunks.append(('act', poses_act[:n_act]))
        if chunks:
            rows = []
            idx = 0
            for src, arr in chunks:
                for tq in arr:
                    rows.append([idx, src] + [float(x) for x in tq])
                    idx += 1
            # Write with mixed int/str/float: do it manually since np.savetxt struggles.
            out_path = os.path.join(cfg['output']['save_dir'],
                                    'tracker_poses_w2c.txt')
            with open(out_path, 'w') as f:
                f.write('# idx src tx ty tz qx qy qz qw  (w2c, VINGS)\n')
                for r in rows:
                    f.write(f"{r[0]} {r[1]} " + " ".join(f"{v:.6f}" for v in r[2:]) + "\n")
            print(f"tracker_poses_w2c.txt geschrieben "
                  f"({n_save} marginalisiert + {n_act} aktiv).")
    except Exception as _e:
        print(f"[WARN] tracker_poses_w2c dump failed: {_e}")

    wall_total = time.time() - wall_t0
    print(f"\n=== Profiling Summary ({n_keyframes} KFs, {n_mapped} mapped "
          f"/ {n_processed} processed / {len(runner.dataset)} dataset, "
          f"frame_skip={frame_skip}, mapper_kf_skip={mapper_kf_skip}, "
          f"wall={wall_total:.1f}s) ===")
    timer.summary(total_wall=wall_total)

    write_profiling_json(timer, cfg,
                         n_keyframes=n_keyframes, n_mapped=n_mapped,
                         n_processed=n_processed, n_frames=len(runner.dataset),
                         last_idx=last_idx, frame_skip=frame_skip,
                         mapper_kf_skip=mapper_kf_skip, wall_t0=wall_t0,
                         partial=False)
    print(f"profiling.json -> {os.path.join(cfg['output']['save_dir'], 'profiling.json')}")
