"""Live geo-referencing for the streaming viewer.

Turns the gauge-free DROID world (in which the streamed Gaussians/objects live)
into a metric ENU world anchored on the flight GPS, then into three.js world
coords, so the live map sits on real satellite imagery — the same fix validated
offline in ``od-experiments``.

Why not Umeyama: amtown-style flights are a near-straight GPS line → rotation is
degenerate. The frame is instead built from
  * up      = -mean(camera forward)          (nadir cam looks down)
  * heading = DROID start→end chord  ->  GPS start→end chord
  * scale   = GPS chord length / DROID along-track span
  * position= path-centroid match
with a right-handed basis (``right = fwd × up`` — using ``up × fwd`` mirrors the
whole scene, a bug worth never reintroducing).

The matrix sent to the frontend maps DROID coords straight to three.js world
(East→X, Up→Y, North→−Z):  ``M = A · Sim3``  with ``A = ENU→three`` axis swap.

A background thread fetches Esri World Imagery tiles for the trajectory bbox and
saves the mosaic into ``static/`` (served by stream_server) so the page can load
it same-origin.
"""
from __future__ import annotations

import io
import math
import os
import ssl
import threading
import urllib.request

import numpy as np

EARTH_R = 6378137.0
# ENU(e,n,u) -> three(x=e, y=u, z=-n)
_A = np.array([[1.0, 0.0, 0.0],
               [0.0, 0.0, 1.0],
               [0.0, -1.0, 0.0]], dtype=np.float64)


# --------------------------------------------------------------------------- frame
def build_geo_frame(C, fwd, Cenu):
    """C, fwd, Cenu: (N,3) DROID centres, DROID cam-forwards, ENU centres.
    Returns (s, R, t) mapping DROID -> ENU metres, or None if degenerate."""
    C = np.asarray(C, float); fwd = np.asarray(fwd, float); Cenu = np.asarray(Cenu, float)
    if len(C) < 4:
        return None
    up_d = -np.mean(fwd / (np.linalg.norm(fwd, axis=1, keepdims=True) + 1e-12), axis=0)
    n = np.linalg.norm(up_d)
    if n < 1e-6:
        return None
    up_d /= n
    k = max(2, len(C) // 8)
    chord_d = C[-k:].mean(0) - C[:k].mean(0)
    fwd_d = chord_d - up_d * (chord_d @ up_d)
    if np.linalg.norm(fwd_d) < 1e-6:
        return None
    fwd_d /= np.linalg.norm(fwd_d)
    right_d = np.cross(fwd_d, up_d)               # right-handed (det +1) — no mirror
    right_d /= np.linalg.norm(right_d)
    Rd = np.stack([right_d, fwd_d, up_d], axis=1)

    g_head = Cenu[-k:].mean(0) - Cenu[:k].mean(0); g_head[2] = 0.0
    gh = np.linalg.norm(g_head)
    if gh < 1e-6:
        return None
    g_head /= gh
    g_right = np.array([g_head[1], -g_head[0], 0.0])
    Rw = np.stack([g_right, g_head, np.array([0.0, 0.0, 1.0])], axis=1)

    R = Rw @ Rd.T
    droid_span = ((C - C.mean(0)) @ fwd_d).ptp()
    gps_span = np.linalg.norm((Cenu[-k:].mean(0) - Cenu[:k].mean(0))[:2])
    if droid_span < 1e-6:
        return None
    s = gps_span / droid_span
    t = Cenu.mean(0) - s * (R @ C.mean(0))
    return s, R, t


def droid_to_three_matrix(s, R, t):
    """4x4 (column-major list of 16) mapping DROID coords -> three.js world."""
    M = np.eye(4)
    M[:3, :3] = s * (_A @ R)
    M[:3, 3] = _A @ t
    return M.T.flatten().tolist()       # column-major for THREE.Matrix4.fromArray


# --------------------------------------------------------------------------- tiles
def _lonlat_to_tile(lon, lat, z):
    nn = 2 ** z
    x = (lon + 180.0) / 360.0 * nn
    lr = math.radians(lat)
    y = (1.0 - math.log(math.tan(lr) + 1.0 / math.cos(lr)) / math.pi) / 2.0 * nn
    return x, y


def _tile_to_lonlat(x, y, z):
    nn = 2 ** z
    lon = x / nn * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / nn))))
    return lon, lat


def fetch_satellite(lat_min, lat_max, lon_min, lon_max, out_png, z=18):
    """Stitch Esri World Imagery tiles -> out_png. Returns geo extent dict."""
    from PIL import Image
    ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
    x0f, y0f = _lonlat_to_tile(lon_min, lat_max, z)
    x1f, y1f = _lonlat_to_tile(lon_max, lat_min, z)
    x0, y0 = int(math.floor(x0f)), int(math.floor(y0f))
    x1, y1 = int(math.floor(x1f)), int(math.floor(y1f))
    nx, ny = x1 - x0 + 1, y1 - y0 + 1
    canvas = Image.new("RGB", (nx * 256, ny * 256))
    for ix in range(nx):
        for iy in range(ny):
            tx, ty = x0 + ix, y0 + iy
            url = (f"https://server.arcgisonline.com/ArcGIS/rest/services/"
                   f"World_Imagery/MapServer/tile/{z}/{ty}/{tx}")
            for attempt in range(4):
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                    data = urllib.request.urlopen(req, timeout=30, context=ctx).read()
                    canvas.paste(Image.open(io.BytesIO(data)), (ix * 256, iy * 256))
                    break
                except Exception:
                    if attempt == 3:
                        pass
    lon_w, lat_n = _tile_to_lonlat(x0, y0, z)
    lon_e, lat_s = _tile_to_lonlat(x1 + 1, y1 + 1, z)
    canvas.save(out_png)
    return dict(lon_w=lon_w, lon_e=lon_e, lat_n=lat_n, lat_s=lat_s,
                px_w=nx * 256, px_h=ny * 256)


# --------------------------------------------------------------------------- live
class LiveGeoReferencer:
    """Accumulate per-keyframe (DROID centre+forward, ENU centre); once the
    trajectory is long enough, expose a DROID->three matrix and kick off a
    one-shot background satellite fetch. All methods are cheap + non-blocking."""

    def __init__(self, gps_lat0, gps_lon0, static_dir, *,
                 min_kfs=10, min_span_m=40.0, pad_m=100.0, zoom=18,
                 sat_name="satellite.png"):
        self.lat0, self.lon0 = float(gps_lat0), float(gps_lon0)
        self.static_dir = static_dir
        self.min_kfs, self.min_span_m = int(min_kfs), float(min_span_m)
        self.pad_m, self.zoom, self.sat_name = float(pad_m), int(zoom), sat_name
        self._C, self._F, self._E = [], [], []
        self._M = None
        self._sat = None                  # dict once tiles are ready
        self._sat_covered = None          # (e0,e1,n0,n1) ENU bbox the PNG covers
        self._sat_fetching = False
        self._sat_gen = 0
        self._lock = threading.Lock()

    def add_keyframe(self, C_droid, fwd_droid, C_enu):
        if C_enu is None:
            return
        self._C.append(np.asarray(C_droid, float).reshape(3))
        self._F.append(np.asarray(fwd_droid, float).reshape(3))
        self._E.append(np.asarray(C_enu, float).reshape(3))
        self._recompute()

    def tick(self):
        """Re-evaluate whether the satellite needs (re)fetching. Safe to call
        repeatedly (e.g. from a replay loop after all keyframes are added)."""
        self._maybe_fetch_satellite()

    def _recompute(self):
        if len(self._C) < self.min_kfs:
            return
        E = np.array(self._E)
        if np.linalg.norm(E[:, :2].ptp(0)) < self.min_span_m:
            return
        fr = build_geo_frame(np.array(self._C), np.array(self._F), E)
        if fr is None:
            return
        s, R, t = fr
        self._M = droid_to_three_matrix(s, R, t)
        self._maybe_fetch_satellite()

    def _maybe_fetch_satellite(self):
        """(Re)fetch tiles when the trajectory grows beyond the covered area.
        Guarded so only one fetch runs at a time."""
        if self._M is None or self._sat_fetching or len(self._E) < self.min_kfs:
            return
        E = np.array(self._E)
        e0, e1 = E[:, 0].min() - self.pad_m, E[:, 0].max() + self.pad_m
        n0, n1 = E[:, 1].min() - self.pad_m, E[:, 1].max() + self.pad_m
        cov = self._sat_covered
        # refetch if uncovered, or the trajectory bbox spilled outside the PNG
        need = (cov is None or e0 < cov[0] - 1 or e1 > cov[1] + 1
                or n0 < cov[2] - 1 or n1 > cov[3] + 1)
        if not need:
            return
        self._sat_fetching = True
        self._sat_gen += 1
        threading.Thread(target=self._fetch_sat_bg, args=((e0, e1, n0, n1),),
                         daemon=True, name="geo-sat").start()

    def _enu_to_latlon(self, e, n):
        lat = self.lat0 + math.degrees(n / EARTH_R)
        lon = self.lon0 + math.degrees(e / (EARTH_R * math.cos(math.radians(self.lat0))))
        return lat, lon

    def _fetch_sat_bg(self, bbox):
        try:
            e0, e1, n0, n1 = bbox
            la0, lo0 = self._enu_to_latlon(e0, n0)
            la1, lo1 = self._enu_to_latlon(e1, n1)
            out = os.path.join(self.static_dir, self.sat_name)
            geo = fetch_satellite(min(la0, la1), max(la0, la1), min(lo0, lo1), max(lo0, lo1),
                                  out, z=self.zoom)
            # the PNG covers the TILE-ALIGNED region (>= requested bbox); place the
            # plane on that true extent or the imagery is shifted vs the cloud.
            ce = math.cos(math.radians(self.lat0))
            e_w = math.radians(geo["lon_w"] - self.lon0) * EARTH_R * ce
            e_e = math.radians(geo["lon_e"] - self.lon0) * EARTH_R * ce
            n_n = math.radians(geo["lat_n"] - self.lat0) * EARTH_R
            n_s = math.radians(geo["lat_s"] - self.lat0) * EARTH_R
            with self._lock:
                self._sat = dict(url=self.sat_name, e_w=e_w, e_e=e_e, n_s=n_s, n_n=n_n,
                                 gen=self._sat_gen)
                self._sat_covered = (e0, e1, n0, n1)
            print(f"[geo] satellite mosaic ready ({geo['px_w']}x{geo['px_h']}) -> {out}")
        except Exception as e:
            print(f"[geo] satellite fetch failed: {e}")
        finally:
            self._sat_fetching = False

    @property
    def ready(self):
        return self._M is not None

    def geo_message(self, epoch):
        if self._M is None:
            return None
        with self._lock:
            sat = dict(self._sat) if self._sat else None
        return {"type": "geo", "epoch": int(epoch), "M": self._M, "sat": sat}


# --------------------------------------------------------------------------- smoke
if __name__ == "__main__":
    # quick numeric self-check on the real run dir
    run = ("/home/philipp/Dokumente/Github/VINGS-Mono-BA/output/"
           "exp_amtown03_s5000_200f_visdrone_stream/"
           "06-19-00-21-generic_vo-amtown03_s5000_400f_visdrone_stream-")
    r = np.loadtxt(run + "/tracker_raw_c2w.txt", comments="#")
    keep = np.concatenate([[True], np.any(np.abs(np.diff(r[:, [5, 9, 13]], axis=0)) > 1e-9, 1)])
    r = r[keep]
    C = r[:, [5, 9, 13]]; fwd = r[:, [4, 8, 12]]; td = r[:, 1]
    g = np.loadtxt("/home/philipp/Dokumente/datasets/amtown03/metadata/rtk.csv")
    E = np.stack([np.interp(td, g[:, 0], g[:, 2]) * 0, np.interp(td, g[:, 0], g[:, 1]) * 0,
                  np.interp(td, g[:, 0], g[:, 3])], 1)  # placeholder; real ENU in run path
    fr = build_geo_frame(C, fwd, np.stack([
        np.deg2rad(np.interp(td, g[:, 0], g[:, 2]) - g[0, 2]) * EARTH_R * math.cos(math.radians(g[0, 1])),
        np.deg2rad(np.interp(td, g[:, 0], g[:, 1]) - g[0, 1]) * EARTH_R,
        np.interp(td, g[:, 0], g[:, 3]) - g[0, 3]], 1))
    print("frame:", None if fr is None else f"scale={fr[0]:.2f} detR={np.linalg.det(fr[1]):.3f}")
