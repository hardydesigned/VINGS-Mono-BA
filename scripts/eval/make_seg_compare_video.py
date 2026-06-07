"""
Baue Vergleichsvideos der Dynamic-Mask-Overlays zweier Segmentation-Backends.

Liest die `dynamic/FrameId=*.png`-Overlays (GT-Frame mit rot getönten maskierten
Pixeln) aus zwei Run-Ordnern und schreibt:
  - <out>/fastsam_dynamic.mp4
  - <out>/sam2_dynamic.mp4
  - <out>/compare_sidebyside.mp4   (links FastSAM, rechts SAM2, nach FrameId gepaart)

Aufruf:
  python scripts/eval/make_seg_compare_video.py <fastsam_run_dir> <sam2_run_dir> <out_dir> [fps]
"""
import glob
import os
import re
import sys

import cv2
import numpy as np

FPS = float(sys.argv[4]) if len(sys.argv) > 4 else 4.0


def frames(run_dir):
    """{frame_id:int -> path}, sortiert nach FrameId."""
    out = {}
    for p in glob.glob(os.path.join(run_dir, "dynamic", "FrameId=*.png")):
        m = re.search(r"FrameId=([0-9.]+)\.png", os.path.basename(p))
        if m:
            out[int(float(m.group(1)))] = p
    return dict(sorted(out.items()))


def label(img, text):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 18), (0, 0, 0), -1)
    cv2.putText(img, text, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (255, 255, 255), 1, cv2.LINE_AA)
    return img


def write_video(path, paths, tag):
    if not paths:
        print(f"  [skip] {tag}: keine Frames")
        return
    h, w = cv2.imread(paths[0]).shape[:2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    for p in paths:
        vw.write(label(cv2.imread(p), tag))
    vw.release()
    print(f"  {tag}: {len(paths)} Frames -> {path}")


def main():
    fs_dir, s2_dir, out = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out, exist_ok=True)
    fs, s2 = frames(fs_dir), frames(s2_dir)
    print(f"FastSAM: {len(fs)} frames | SAM2: {len(s2)} frames")

    write_video(os.path.join(out, "fastsam_dynamic.mp4"), list(fs.values()), "FastSAM")
    write_video(os.path.join(out, "sam2_dynamic.mp4"), list(s2.values()), "SAM2.1")

    # Side-by-side über gemeinsame FrameIds.
    common = sorted(set(fs) & set(s2))
    print(f"gemeinsame FrameIds: {len(common)}")
    if common:
        h, w = cv2.imread(fs[common[0]]).shape[:2]
        gap = 6
        vw = cv2.VideoWriter(os.path.join(out, "compare_sidebyside.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w * 2 + gap, h))
        sep = np.full((h, gap, 3), 255, np.uint8)
        for fid in common:
            l = label(cv2.imread(fs[fid]), f"FastSAM  f{fid}")
            r = label(cv2.imread(s2[fid]), f"SAM2.1  f{fid}")
            vw.write(np.hstack([l, sep, r]))
        vw.release()
        print(f"  side-by-side: {len(common)} Frames -> {os.path.join(out, 'compare_sidebyside.mp4')}")


if __name__ == "__main__":
    main()
