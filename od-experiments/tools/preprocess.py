#!/usr/bin/env python3
"""Preprocess a VINGS object-detection run into web assets for the map viewer.

Outputs (into app/assets/):
  cloud_pos.f32   - Float32 [N,3] gaussian centers in ENU metres (east,north,up)
  cloud_col.u8    - Uint8   [N,3] RGB (SH dc -> colour)
  objects.json    - detected objects in ENU metres (pos, yaw, class, dims)
  scene.json      - metadata (counts, origin latlon, satellite geo-extent, cam path)
  satellite.png   - stitched Esri World Imagery for the trajectory bbox

Coordinate handling
-------------------
GPS flight is a near-straight line (degenerate for Umeyama rotation), so we do
NOT fit a rigid Sim3 from camera centres. Instead we build the world frame from:
  * UP    = -mean(camera forward)   (nadir cam looks down -> up is opposite)
  * HEADING= DROID flight tangent  ->  rotated to match GPS heading in EN plane
  * SCALE = GPS line length / DROID horizontal path length
Then a per-keyframe local-Sim3 unwarp (rotation fixed global, scale+translation
local, blended per point by k-nearest cameras) straightens the residual DROID
drift onto the straight GPS line.
"""
from __future__ import annotations
import sys, os, json, math, ssl, urllib.request, io
import numpy as np
from plyfile import PlyData

RUN = "/home/philipp/Dokumente/Github/VINGS-Mono-BA/output/exp_amtown03_s5000_200f_visdrone_stream/06-19-00-21-generic_vo-amtown03_s5000_400f_visdrone_stream-"
GPS_CSV = "/home/philipp/Dokumente/datasets/amtown03/metadata/rtk.csv"
OUT = os.path.join(os.path.dirname(__file__), "..", "app", "assets")
OUT = os.path.abspath(OUT)
SH_C0 = 0.28209479177387814
EARTH_R = 6378137.0

# ---------------------------------------------------------------- poses / gps
def load_poses():
    r = np.loadtxt(os.path.join(RUN, "tracker_raw_c2w.txt"), comments="#")
    t = r[:, 1]
    C = r[:, [5, 9, 13]]                       # c2w translation = cam centre
    fwd = r[:, [4, 8, 12]]                     # c2w 3rd col = cam +z (look dir)
    # keep only keyframes where the pose actually changed
    keep = np.concatenate([[True], np.any(np.abs(np.diff(C, axis=0)) > 1e-9, 1)])
    return t[keep], C[keep], fwd[keep]

def gps_enu(t_query):
    g = np.loadtxt(GPS_CSV)
    tg, lat, lon, alt = g[:, 0], g[:, 1], g[:, 2], g[:, 3]
    lat0 = float(np.interp(t_query[0], tg, lat))
    lon0 = float(np.interp(t_query[0], tg, lon))
    alt0 = float(np.interp(t_query[0], tg, alt))
    def to_enu(la, lo, al):
        e = np.radians(lo - lon0) * EARTH_R * math.cos(math.radians(lat0))
        n = np.radians(la - lat0) * EARTH_R
        u = al - alt0
        return np.stack([e, n, u], -1)
    Cg = to_enu(np.interp(t_query, tg, lat),
                np.interp(t_query, tg, lon),
                np.interp(t_query, tg, alt))
    return Cg, (lat0, lon0, alt0)

def enu_to_latlon(e, n, lat0, lon0):
    lat = lat0 + np.degrees(n / EARTH_R)
    lon = lon0 + np.degrees(e / (EARTH_R * math.cos(math.radians(lat0))))
    return lat, lon

# ---------------------------------------------------------------- world frame
def build_global_frame(C, fwd, Cg):
    """Return s, R, t mapping DROID coords -> ENU metres (global, pre-unwarp)."""
    up_d = -np.mean(fwd / np.linalg.norm(fwd, axis=1, keepdims=True), axis=0)
    up_d /= np.linalg.norm(up_d)
    # DROID flight heading = CHORD start->end (matches the GPS chord; robust to the
    # curved-path drift, unlike the PCA axis which bisects the curve). Use mean of
    # first/last few keyframes for stability, then project ⟂ up.
    k = max(2, len(C) // 8)
    chord_d = C[-k:].mean(0) - C[:k].mean(0)
    fwd_d = chord_d - up_d * (chord_d @ up_d)
    fwd_d /= np.linalg.norm(fwd_d)
    right_d = np.cross(fwd_d, up_d)             # right-handed (right,fwd,up), det +1
    right_d /= np.linalg.norm(right_d)
    Rd = np.stack([right_d, fwd_d, up_d], axis=1)   # cols = basis in DROID coords

    # GPS heading = matching chord in EN plane (z up)
    g_head = Cg[-k:].mean(0) - Cg[:k].mean(0); g_head[2] = 0
    g_head /= np.linalg.norm(g_head)
    g_right = np.array([g_head[1], -g_head[0], 0.0])  # right-handed with +Z up
    Rw = np.stack([g_right, g_head, np.array([0, 0, 1.0])], axis=1)

    R = Rw @ Rd.T                               # DROID basis -> ENU basis
    # scale: GPS straight length / DROID horizontal (along-tangent) span
    droid_span = ((C - C.mean(0)) @ fwd_d).ptp()
    gps_span = np.linalg.norm((Cg[-1] - Cg[0])[:2])
    s = gps_span / droid_span
    # translation: align path centroids
    t = Cg.mean(0) - s * (R @ C.mean(0))
    return s, R, t, Rd, up_d, fwd_d

def umeyama_st(src, dst, R):
    """scale+translation only, rotation fixed = R. (1D-safe)"""
    s = np.linalg.norm(dst - dst.mean(0), axis=1).sum() / \
        (np.linalg.norm(src - src.mean(0), axis=1).sum() + 1e-12)
    Rsrc = (R @ src.T).T
    t = dst.mean(0) - s * Rsrc.mean(0)
    return s, t

def local_unwarp(P, C, Cg, s_g, R, window=14, knn=4):
    """Per-keyframe local Sim3 (R fixed), blended per point by knn cameras."""
    N = len(C)
    half = max(window // 2, 4)
    loc_s = np.empty(N); loc_t = np.empty((N, 3))
    for i in range(N):
        a, b = max(0, i - half), min(N, i + half + 1)
        si, ti = umeyama_st(C[a:b], Cg[a:b], R)
        loc_s[i] = si; loc_t[i] = ti
    # blend: for each point, knn nearest cameras (in DROID space), inverse-dist weights
    from scipy.spatial import cKDTree
    tree = cKDTree(C)
    k = min(knn, N)
    d, idx = tree.query(P, k=k)
    if k == 1:
        d = d[:, None]; idx = idx[:, None]
    w = 1.0 / (d + 1e-6); w /= w.sum(1, keepdims=True)
    out = np.zeros_like(P)
    RP = (R @ P.T).T
    for j in range(k):
        sj = loc_s[idx[:, j]][:, None]
        tj = loc_t[idx[:, j]]
        out += w[:, j][:, None] * (sj * RP + tj)
    return out

# ---------------------------------------------------------------- ply
def detrend_ground(Penu, target_z):
    """Remove the low-frequency vertical DRIFT (smooth bow from DROID drift) so the
    ground is level, without flattening real relief (buildings/trees kept). Fit a
    smooth ground trend from the low-u percentile per along-track bin."""
    en = Penu[:, :2] - Penu[:, :2].mean(0)
    _, _, Vt = np.linalg.svd(en, full_matrices=False)
    a = en @ Vt[0]                      # along-track coordinate
    nb = 24
    edges = np.linspace(a.min(), a.max(), nb + 1)
    ac = 0.5 * (edges[:-1] + edges[1:]); gz = np.full(nb, np.nan)
    for i in range(nb):
        sel = (a >= edges[i]) & (a < edges[i + 1])
        if sel.sum() > 50:
            gz[i] = np.percentile(Penu[sel, 2], 18)   # ground level in bin
    ok = ~np.isnan(gz)
    if ok.sum() < 4:
        return Penu
    coef = np.polyfit(ac[ok], gz[ok], 3)               # smooth drift trend
    trend = np.polyval(coef, a)
    out = Penu.copy()
    out[:, 2] = Penu[:, 2] - trend + target_z          # flatten to target ground
    return out

def load_cloud():
    print("[ply] reading", flush=True)
    ply = PlyData.read(os.path.join(RUN, "ply", "idx=351_2dgs.ply"))
    v = ply["vertex"].data
    P = np.stack([v["x"], v["y"], v["z"]], -1).astype(np.float64)
    fdc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], -1).astype(np.float32)
    rgb = np.clip(0.5 + SH_C0 * fdc, 0, 1)
    op = 1.0 / (1.0 + np.exp(-v["opacity"].astype(np.float32)))
    return P, rgb, op

# ---------------------------------------------------------------- satellite
def lonlat_to_tile(lon, lat, z):
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    latr = math.radians(lat)
    y = (1.0 - math.log(math.tan(latr) + 1.0 / math.cos(latr)) / math.pi) / 2.0 * n
    return x, y

def tile_to_lonlat(x, y, z):
    n = 2 ** z
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lon, lat

def fetch_satellite(lat_min, lat_max, lon_min, lon_max, z=18):
    from PIL import Image
    ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
    x0f, y0f = lonlat_to_tile(lon_min, lat_max, z)   # top-left
    x1f, y1f = lonlat_to_tile(lon_max, lat_min, z)   # bottom-right
    x0, y0 = int(math.floor(x0f)), int(math.floor(y0f))
    x1, y1 = int(math.floor(x1f)), int(math.floor(y1f))
    nx, ny = x1 - x0 + 1, y1 - y0 + 1
    print(f"[sat] z={z} tiles {nx}x{ny}", flush=True)
    canvas = Image.new("RGB", (nx * 256, ny * 256))
    for ix in range(nx):
        for iy in range(ny):
            tx, ty = x0 + ix, y0 + iy
            url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{ty}/{tx}"
            for attempt in range(4):
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                    data = urllib.request.urlopen(req, timeout=30, context=ctx).read()
                    canvas.paste(Image.open(io.BytesIO(data)), (ix * 256, iy * 256))
                    break
                except Exception as e:
                    if attempt == 3:
                        print(f"[sat] tile {tx},{ty} FAIL {e}", flush=True)
    # geo-extent of the stitched canvas (tile-aligned)
    lon_w, lat_n = tile_to_lonlat(x0, y0, z)
    lon_e, lat_s = tile_to_lonlat(x1 + 1, y1 + 1, z)
    canvas.save(os.path.join(OUT, "satellite.png"))
    return dict(lon_w=lon_w, lon_e=lon_e, lat_n=lat_n, lat_s=lat_s,
                px_w=nx * 256, px_h=ny * 256)

# ---------------------------------------------------------------- main
def main():
    os.makedirs(OUT, exist_ok=True)
    t, C, fwd = load_poses()
    Cg, (lat0, lon0, alt0) = gps_enu(t)
    s_g, R, t_g, Rd, up_d, fwd_d = build_global_frame(C, fwd, Cg)
    print(f"[frame] scale={s_g:.3f}", flush=True)

    P, rgb, op = load_cloud()
    # opacity filter (drop near-transparent floaters)
    m = op > 0.05
    P, rgb = P[m], rgb[m]
    print(f"[ply] {len(P)} pts after opacity filter", flush=True)

    # Global GPS-anchored similarity is the default: it keeps the reconstruction
    # a clean continuous strip. The per-keyframe local unwarp (UNWARP=1) tries to
    # straighten residual drift but bow-ties the cloud on this near-straight flight.
    GLOBAL_ONLY = os.environ.get("UNWARP", "0") != "1"
    def apply_global(X):
        return (s_g * (R @ X.T).T + t_g)
    if GLOBAL_ONLY:
        Penu = apply_global(P); cam_enu = apply_global(C)
        print("[frame] GLOBAL-ONLY transform", flush=True)
    else:
        Penu = local_unwarp(P, C, Cg, s_g, R)
        cam_enu = local_unwarp(C, C, Cg, s_g, R)

    # crop wild outliers (gauge-free floaters far from the path)
    centre = cam_enu.mean(0)
    rad = np.linalg.norm(Penu[:, :2] - centre[:2], axis=1)
    span = np.linalg.norm(cam_enu[:, :2].ptp(0)) + 50
    keep = rad < span
    Penu, rgb = Penu[keep], rgb[keep]
    # flatten low-frequency vertical drift (keep relief) unless disabled
    if os.environ.get("NO_DETREND", "0") != "1":
        tz = float(np.median(Penu[:, 2]))
        Penu = detrend_ground(Penu, tz)
        print("[frame] vertical de-trend applied", flush=True)
    # vertical floater crop: keep a band around the ground (robust percentiles)
    glo, ghi = np.percentile(Penu[:, 2], [2, 98])
    band = max(40.0, (ghi - glo))
    gmed = np.median(Penu[:, 2])
    vkeep = (Penu[:, 2] > gmed - band) & (Penu[:, 2] < gmed + band)
    Penu, rgb = Penu[vkeep], rgb[vkeep]
    print(f"[ply] {len(Penu)} pts after radius+vert crop (span {span:.0f}m, vband ±{band:.0f}m)", flush=True)

    # write cloud
    Penu.astype(np.float32).tofile(os.path.join(OUT, "cloud_pos.f32"))
    (rgb * 255).astype(np.uint8).tofile(os.path.join(OUT, "cloud_col.u8"))

    # objects
    obj = np.genfromtxt(os.path.join(RUN, "objects_droid.csv"), delimiter=",", names=True,
                        dtype=None, encoding="utf-8")
    if obj.ndim == 0:
        obj = obj.reshape(1)
    Po = np.stack([obj["x"], obj["y"], obj["z"]], -1).astype(np.float64)
    Po_enu = apply_global(Po) if GLOBAL_ONLY else local_unwarp(Po, C, Cg, s_g, R)
    objects = []
    for i in range(len(obj)):
        # object forward (csv quat is yaw about DROID up). rotate a DROID-plane
        # vector and read its ENU heading after the global rotation.
        qw, qx, qy, qz = obj["qw"][i], obj["qx"][i], obj["qy"][i], obj["qz"][i]
        # quaternion rotates about DROID up -> derive in-plane forward in DROID,
        # transform by R, project to EN, take heading.
        # build the object's local x-axis in DROID using up_d as rotation axis:
        ang = 2 * math.atan2(math.sqrt(qx*qx + qy*qy + qz*qz), qw)
        fdir = fwd_d.copy()           # DROID in-plane reference (flight forward)
        # rotate fdir around up_d by ang (Rodrigues)
        k = up_d
        fr = (fdir * math.cos(ang) + np.cross(k, fdir) * math.sin(ang)
              + k * (k @ fdir) * (1 - math.cos(ang)))
        fr_enu = R @ fr
        yaw = math.atan2(fr_enu[0], fr_enu[1])   # heading from +N toward +E
        objects.append(dict(
            id=int(obj["object_id"][i]), cls=str(obj["class"][i]),
            conf=float(obj["conf"][i]), n=int(obj["n_detections"][i]),
            x=float(Po_enu[i, 0]), y=float(Po_enu[i, 1]), z=float(Po_enu[i, 2]),
            yaw=float(yaw)))
    json.dump(objects, open(os.path.join(OUT, "objects.json"), "w"), indent=1)
    print(f"[obj] {len(objects)} objects", flush=True)

    # satellite bbox from robust cloud ENU extent (+pad), ignoring lateral floaters
    pad = 20.0
    e_lo, e_hi = np.percentile(Penu[:, 0], [1, 99])
    n_lo, n_hi = np.percentile(Penu[:, 1], [1, 99])
    e_min, e_max = e_lo - pad, e_hi + pad
    n_min, n_max = n_lo - pad, n_hi + pad
    la_min, lo_min = enu_to_latlon(e_min, n_min, lat0, lon0)
    la_max, lo_max = enu_to_latlon(e_max, n_max, lat0, lon0)
    sat = fetch_satellite(la_min, la_max, lo_min, lo_max, z=18)
    # convert satellite geo-corners to ENU for the textured plane
    e_w = math.radians(sat["lon_w"] - lon0) * EARTH_R * math.cos(math.radians(lat0))
    e_e = math.radians(sat["lon_e"] - lon0) * EARTH_R * math.cos(math.radians(lat0))
    n_n = math.radians(sat["lat_n"] - lat0) * EARTH_R
    n_s = math.radians(sat["lat_s"] - lat0) * EARTH_R

    scene = dict(
        n_points=int(len(Penu)),
        origin=dict(lat=lat0, lon=lon0, alt=alt0),
        scale=float(s_g),
        cam_path=cam_enu.astype(float).round(3).tolist(),
        ground_z=float(np.median(Penu[:, 2])),
        satellite=dict(png="satellite.png", e_w=e_w, e_e=e_e, n_n=n_n, n_s=n_s,
                       px_w=sat["px_w"], px_h=sat["px_h"]),
        cloud_bbox=dict(e=[float(e_lo), float(e_hi)], n=[float(n_lo), float(n_hi)],
                        u=[float(Penu[:,2].min()), float(Penu[:,2].max())]),
    )
    json.dump(scene, open(os.path.join(OUT, "scene.json"), "w"), indent=1)
    print("[done] wrote assets to", OUT, flush=True)

if __name__ == "__main__":
    main()
