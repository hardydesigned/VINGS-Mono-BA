#!/usr/bin/env python3
"""Refine cloud->satellite alignment by registering the cloud's top-down
orthophoto against the satellite image (same modality: nadir RGB of the ground).

Reads the GLOBAL-transformed cloud assets + scene.json, rasterises a top-down
orthophoto on the satellite's geo grid, runs ECC (euclidean+scale) to find the
residual 2D similarity, and prints/saves the correction. Debug images in shots/.
"""
import os, json, numpy as np, cv2
from PIL import Image

A = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app", "assets"))
S = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shots"))

def main():
    scn = json.load(open(os.path.join(A, "scene.json")))
    sat = scn["satellite"]
    pos = np.fromfile(os.path.join(A, "cloud_pos.f32"), np.float32).reshape(-1, 3)
    col = np.fromfile(os.path.join(A, "cloud_col.u8"), np.uint8).reshape(-1, 3)
    e, n, u = pos[:, 0], pos[:, 1], pos[:, 2]

    # downscale grid for speed
    sc = 0.5
    W = int(sat["px_w"] * sc); H = int(sat["px_h"] * sc)
    e_w, e_e, n_n, n_s = sat["e_w"], sat["e_e"], sat["n_n"], sat["n_s"]
    cx = ((e - e_w) / (e_e - e_w) * W).astype(np.int32)
    cy = ((n_n - n) / (n_n - n_s) * H).astype(np.int32)
    m = (cx >= 0) & (cx < W) & (cy >= 0) & (cy < H)
    cx, cy = cx[m], cy[m]
    cu, cc = u[m], col[m]
    # paint highest-u point per pixel (rooftop/canopy wins) -> orthophoto,
    # splat a 1px disk to fill speckle
    ortho = np.zeros((H, W, 3), np.uint8)
    zbuf = np.full((H, W), -1e9, np.float32)
    order = np.argsort(cu)            # low to high; later overwrite -> highest wins
    for i in order:
        y, x = cy[i], cx[i]
        if cu[i] > zbuf[y, x]:
            zbuf[y, x] = cu[i]; ortho[y, x] = cc[i]
    cover = (zbuf > -1e8)
    # densify: morphological close to fill small gaps
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    ortho = cv2.morphologyEx(ortho, cv2.MORPH_CLOSE, k)
    cover_d = cv2.dilate(cover.astype(np.uint8), k, iterations=2).astype(bool)
    print(f"[ortho] {W}x{H}  coverage {cover.mean()*100:.1f}%")

    # satellite on same grid
    satimg = np.array(Image.open(os.path.join(A, sat["png"])).convert("RGB").resize((W, H)))
    Image.fromarray(ortho).save(os.path.join(S, "dbg_ortho.png"))

    mask = (cover_d.astype(np.uint8) * 255)
    def feat(img):
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
        g = cv2.GaussianBlur(g, (7, 7), 0)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0); gy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
        m = cv2.magnitude(gx, gy)
        return cv2.normalize(m, None, 0, 1, cv2.NORM_MINMAX)
    F_o = feat(ortho) * cover_d.astype(np.float32)
    F_s = feat(satimg)
    cen = (W / 2.0, H / 2.0)
    han = cv2.createHanningWindow((W, H), cv2.CV_32F)
    best = (-1, 0.0, 0.0, 0.0)   # response, theta, dx, dy
    for theta in np.arange(-12, 12.01, 1.0):
        Rm = cv2.getRotationMatrix2D(cen, theta, 1.0)
        Fr = cv2.warpAffine(F_o, Rm, (W, H))
        (dx, dy), resp = cv2.phaseCorrelate(F_s * han, Fr * han)
        if resp > best[0]:
            best = (resp, theta, dx, dy)
    resp, theta, dx, dy = best
    print(f"[pc] best response={resp:.4f} theta={theta:.1f} shift=({dx:.1f},{dy:.1f})px")
    # build warp ortho->sat: rotate about center by -theta? We rotated ortho by
    # +theta to match sat, then shifted by (dx,dy). Compose: T(dx,dy) * R(theta).
    Rm = cv2.getRotationMatrix2D(cen, theta, 1.0)
    warp = Rm.copy(); warp[0, 2] += dx; warp[1, 2] += dy
    # guardrails
    shift_m = np.hypot(dx * (e_e - e_w) / W, dy * (n_n - n_s) / H)
    if resp < 0.06 or shift_m > 60:
        print(f"[pc] rejected (resp {resp:.3f}, shift {shift_m:.1f}m) -> keep global")
        return
    cc_val = resp
    print("[pc] warp=\n", np.round(warp, 4))

    # warp maps ortho->sat in pixel space. decompose to rot+trans (px), scale.
    a, b = warp[0, 0], warp[0, 1]
    theta = np.degrees(np.arctan2(b, a))
    tx, ty = warp[0, 2], warp[1, 2]
    mpx_e = (e_e - e_w) / W; mpx_n = (n_n - n_s) / H
    print(f"[ecc] rot={theta:.2f} deg  trans=({tx*mpx_e:.1f}E, {-ty*mpx_n:.1f}N) m")

    # apply warp to ortho for a debug overlay
    aligned = cv2.warpAffine(ortho, warp, (W, H))
    blend = (0.5 * satimg + 0.5 * aligned).astype(np.uint8)
    Image.fromarray(blend).save(os.path.join(S, "dbg_aligned_blend.png"))
    before = (0.5 * satimg + 0.5 * ortho).astype(np.uint8)
    Image.fromarray(before).save(os.path.join(S, "dbg_before_blend.png"))
    json.dump(dict(warp=warp.tolist(), theta_deg=float(theta),
                   tx_m=float(tx*mpx_e), ty_m=float(-ty*mpx_n)),
              open(os.path.join(A, "align.json"), "w"), indent=1)
    print("[done] wrote align.json + debug blends")

if __name__ == "__main__":
    main()
