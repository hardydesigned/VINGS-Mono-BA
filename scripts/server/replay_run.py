#!/usr/bin/env python3
"""Replay a finished VINGS run through the live streaming path to validate the
GPS-anchored map projection + object placement end-to-end (no SLAM needed).

Feeds tracker_raw_c2w poses + rtk GPS into LiveGeoReferencer exactly like the
run loop does, pushes the run's PLY as frozen splat chunks and objects_droid.csv
as object markers, then serves the viewer. Open http://localhost:8765/ (or point
Playwright at it). The map should match what od-experiments produced.
"""
import os, sys, math, time, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))   # scripts/

from stream_server import SplatStreamServer
from splat_encode import _to_splat_bytes, _pad_scale
from geo_frame import LiveGeoReferencer, EARTH_R

DEF_RUN = ("/home/philipp/Dokumente/Github/VINGS-Mono-BA/output/"
           "exp_amtown03_s5000_200f_visdrone_stream/"
           "06-19-00-21-generic_vo-amtown03_s5000_400f_visdrone_stream-")
DEF_GPS = "/home/philipp/Dokumente/datasets/amtown03/metadata/rtk.csv"
SH_C0 = 0.28209479177387814


def load_ply_splat(path, max_n=260000):
    from plyfile import PlyData
    v = PlyData.read(path)["vertex"].data
    xyz = np.stack([v["x"], v["y"], v["z"]], -1).astype(np.float32)
    op = 1.0 / (1.0 + np.exp(-v["opacity"].astype(np.float32)))
    m = op > 0.05
    xyz, op = xyz[m], op[m]
    fdc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], -1).astype(np.float32)[m]
    rgb = np.clip(0.5 + SH_C0 * fdc, 0, 1)
    sc2 = np.exp(np.stack([v["scale_0"], v["scale_1"]], -1).astype(np.float32)[m])
    quat = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], -1).astype(np.float32)[m]
    if len(xyz) > max_n:
        idx = np.random.default_rng(0).choice(len(xyz), max_n, replace=False)
        xyz, rgb, op, sc2, quat = xyz[idx], rgb[idx], op[idx], sc2[idx], quat[idx]
    return xyz, rgb, op, sc2, quat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=DEF_RUN)
    ap.add_argument("--gps", default=DEF_GPS)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--hold", type=float, default=0.0, help="seconds to keep serving (0=forever)")
    ap.add_argument("--shot", default="", help="if set: screenshot the viewer to this path then exit")
    ap.add_argument("--shot-wait", type=float, default=16.0)
    ap.add_argument("--max-splats", type=int, default=260000,
                    help="cap splats (lower for the slow headless client; real browsers handle 260k+)")
    a = ap.parse_args()

    # poses (dedup repeated) + ENU at KF times (origin = first rtk row, like generic_vo)
    r = np.loadtxt(os.path.join(a.run, "tracker_raw_c2w.txt"), comments="#")
    keep = np.concatenate([[True], np.any(np.abs(np.diff(r[:, [5, 9, 13]], axis=0)) > 1e-9, 1)])
    r = r[keep]
    td = r[:, 1]
    C = r[:, [5, 9, 13]]
    fwd = r[:, [4, 8, 12]]
    g = np.loadtxt(a.gps, comments="#")
    lat0, lon0, alt0 = g[0, 1], g[0, 2], g[0, 3]
    e = np.deg2rad(np.interp(td, g[:, 0], g[:, 2]) - lon0) * EARTH_R * math.cos(math.radians(lat0))
    n = np.deg2rad(np.interp(td, g[:, 0], g[:, 1]) - lat0) * EARTH_R
    u = np.interp(td, g[:, 0], g[:, 3]) - alt0
    Cenu = np.stack([e, n, u], 1)

    srv = SplatStreamServer(port=a.port)
    srv.start()
    geo = LiveGeoReferencer(lat0, lon0, os.path.join(HERE, "static"), min_kfs=10, min_span_m=40.0)

    # feed all keyframes -> geo ready + satellite fetch kicks off
    for i in range(len(C)):
        geo.add_keyframe(C[i], fwd[i], Cenu[i])
    msg = geo.geo_message(0)
    print("[replay] geo ready:", geo.ready, "| matrix:", "yes" if msg and msg["M"] else "no")
    if msg:
        srv.push(msg)

    # PLY -> frozen splat chunks (DROID frame; georeferenced by geoM on frontend)
    xyz, rgb, op, sc2, quat = load_ply_splat(os.path.join(a.run, "ply", "idx=351_2dgs.ply"),
                                             max_n=a.max_splats)
    print(f"[replay] PLY {len(xyz)} splats")
    nchunks = 12
    for ci, sl in enumerate(np.array_split(np.arange(len(xyz)), nchunks)):
        blob = _to_splat_bytes(xyz[sl], _pad_scale(sc2[sl]), rgb[sl], op[sl], quat[sl])
        srv.push({"type": "append_frozen", "epoch": 0, "kf_id": ci, "data": blob})

    # objects_droid.csv -> object markers (DROID xyz/quat/size)
    obj = np.genfromtxt(os.path.join(a.run, "objects_droid.csv"), delimiter=",",
                        names=True, dtype=None, encoding="utf-8")
    if obj.ndim == 0:
        obj = obj.reshape(1)
    objs = [dict(object_id=int(obj["object_id"][i]), cls_id=int(obj["cls_id"][i]),
                 **{"class": str(obj["class"][i])}, conf=float(obj["conf"][i]),
                 xyz=[float(obj["x"][i]), float(obj["y"][i]), float(obj["z"][i])],
                 quat=[float(obj["qw"][i]), float(obj["qx"][i]), float(obj["qy"][i]), float(obj["qz"][i])],
                 size=[float(obj["sx"][i]), float(obj["sy"][i]), float(obj["sz"][i])])
            for i in range(len(obj))]
    srv.push({"type": "objects", "epoch": 0, "objects": objs})
    print(f"[replay] {len(objs)} objects")

    # wait for the satellite to cover the FULL trajectory (it may re-fetch as the
    # bbox grows), then push the final geo with the satellite info.
    full = np.array(Cenu)
    fe0, fe1 = full[:, 0].min() - geo.pad_m, full[:, 0].max() + geo.pad_m
    fn0, fn1 = full[:, 1].min() - geo.pad_m, full[:, 1].max() + geo.pad_m
    covered = lambda c: c and c[0] <= fe0 + 1 and c[1] >= fe1 - 1 and c[2] <= fn0 + 1 and c[3] >= fn1 - 1
    for _ in range(90):
        geo.tick()
        if not geo._sat_fetching and covered(geo._sat_covered):
            srv.push(geo.geo_message(0)); print("[replay] full satellite pushed"); break
        time.sleep(1.0)
    else:
        m = geo.geo_message(0)
        if m and m.get("sat"):
            srv.push(m); print("[replay] partial satellite pushed")
        else:
            print("[replay] WARN satellite not ready (network?)")

    if a.shot:
        _screenshot(f"http://localhost:{a.port}/", a.shot, a.shot_wait)
        srv.stop()
        return

    print(f"[replay] serving http://localhost:{a.port}/  (Ctrl-C to stop)")
    t0 = time.time()
    try:
        while a.hold == 0 or time.time() - t0 < a.hold:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    srv.stop()


def _screenshot(url, out, wait_s):
    # NOTE: run the whole process under `ulimit -s 1024` so headless Chrome can
    # spawn under this host's strict memory overcommit (see od-experiments/tools/shot.py).
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True, args=[
            "--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu",
            "--use-gl=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist",
            "--single-process", "--no-zygote",
            "--js-flags=--max-old-space-size=512 --jitless", "--disable-extensions"])
        pg = b.new_page(viewport={"width": 1100, "height": 900})
        errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.goto(url, wait_until="load", timeout=60000)
        pg.wait_for_timeout(int(wait_s * 1000))
        view = os.environ.get("VIEW", "")
        if view == "top":
            try: pg.evaluate("window.__topView && window.__topView()")
            except Exception: pass
            pg.wait_for_timeout(800)
        elif view in ("closeup", "closeup_obl"):
            try: pg.evaluate(f"window.__closeup && window.__closeup({'true' if view=='closeup_obl' else 'false'})")
            except Exception: pass
            pg.wait_for_timeout(800)
        elif view.startswith("obj"):
            i = int(view[3:] or "0")
            try: pg.evaluate(f"window.__zoomObj && window.__zoomObj({i})")
            except Exception: pass
            pg.wait_for_timeout(800)
        click = os.environ.get("CLICK", "")
        if click:
            try:
                pg.click("#" + click); pg.wait_for_timeout(600)
                print(f"[replay] clicked #{click}")
            except Exception as ex:
                print("[replay] click failed:", ex)
        try:
            print("[replay] viewer state:", pg.evaluate("window.__dbg ? window.__dbg() : 'no hook'"))
            print("[replay] satMesh visible:", pg.evaluate("window.__satVisible ? window.__satVisible() : '?'"))
        except Exception as ex:
            print("[replay] dbg eval failed:", ex)
        pg.screenshot(path=out)
        b.close()
    print(f"[replay] screenshot -> {out}")
    for e in errs[:10]:
        print("   PAGEERR", e[:160])


if __name__ == "__main__":
    main()
