#!/usr/bin/env python3
"""GPS-verankertes Boden-Leveling für Survey-Segmente (Naht-Fix in 3D, v2).

Idee (User, 2026-06-04): sim3_unwarp verankert die KAMERAS perfekt an GPS
(cam_alt über alle Segmente konstant ~80 m im PLY-Frame = Cruise-Höhe), aber die
monokulare TIEFEN-Skala je Segment löst falsch auf -> der Boden landet pro Segment
auf einer ganz anderen Höhe (gemessen: −13 .. +72 m statt konstant). Da die Kamera-
höhe konstant und bekannt ist und der Boden einer flachen Stadt auf ~konstanter
Absoluthöhe liegen MUSS, levle ich jedes Segment so, dass sein Boden auf einer
gemeinsamen Referenzebene sitzt -> alle Chunks koplanar, keine komische Verdrehung.

Boden-Schätzung (robust gegen die vertikale Schmiere + schließt Häuser aus): pro
grober (x,y)-Zelle der DICHTESTE z-Layer (Histogramm-Mode) = Boden (Häuser/hohe
Strukturen sind dünn besetzt und fallen raus). Aggregat = Median der Zell-Böden;
Streuung der Zell-Böden = Reliabilität (kleiner = sauberer flacher Boden).

Korrektur: pro Segment z += (ref - g_i). Optionaler horizontaler De-Tilt für
Segmente mit zuverlässigem (flachem) Bodenfeld; verrauschte werden nur geshiftet
(kein Tilt aus Müll-Boden). Referenz ref = reliabilitäts-gewichteter Median der g_i
(GPS-Frame -> Absoluthöhe bleibt erhalten; Offset ist nur global kosmetisch).

Usage:
  python scripts/eval/detilt_gps.py s*_gps.ply --out survey.ply \
     [--cell 25] [--zbins 25] [--min-cell-pts 30] [--detilt] [--max-tilt 120]
"""
import argparse, numpy as np, os, glob
from plyfile import PlyData, PlyElement

# numpy-dtype -> PLY-Property-Typname (für den manuell gestreamten Header)
_PLY_T = {"f4": "float", "f8": "double", "i4": "int", "u4": "uint",
          "i2": "short", "u2": "ushort", "i1": "char", "u1": "uchar"}


def _ply_prop_lines(dt):
    """structured numpy-dtype -> Liste 'property <typ> <name>' in Feldreihenfolge."""
    lines = []
    for name in dt.names:
        key = dt[name].str[1:]  # '<f4' -> 'f4'
        t = _PLY_T.get(key)
        if t is None:
            raise ValueError(f"PLY-Property-Typ für dtype {dt[name].str} ({name}) nicht unterstützt")
        lines.append(f"property {t} {name}")
    return lines


def cell_mode_grounds(xyz, cell, zbins, min_pts):
    """Pro (x,y)-Zelle den dichtesten z-Layer (Mode) = lokaler Boden. -> dict + array."""
    k = np.floor(xyz[:, :2] / cell).astype(np.int64)
    d = {}
    for key, zz in zip(map(tuple, k), xyz[:, 2]):
        d.setdefault(key, []).append(zz)
    out = {}
    for c, vv in d.items():
        if len(vv) < min_pts:
            continue
        vv = np.asarray(vv)
        h, e = np.histogram(vv, bins=zbins)
        out[c] = 0.5 * (e[np.argmax(h)] + e[np.argmax(h) + 1])
    return out


def fit_level(grounds, cell, origin):
    """Median-Boden + robuste Ebene (für optionalen De-Tilt) + Reliabilität (MAD).

    Sparse/degenerierte Segmente (z.B. ein Selektor der nur 1-2 KFs pro Chunk
    mappt -> wenige hundert Gaussians, keine Zelle erreicht min_cell_pts) liefern
    ein leeres `grounds`-Dict. Dann gibt es keine Bodenschätzung: g_med=nan,
    mad=inf -> das Segment bekommt im Reliab.-gewichteten ref Gewicht 0 und wird
    in Pass 2 (dz=nan -> zband-keep komplett False) verworfen. Das fängt den
    frueheren IndexError in `np.stack([cs[:,0], ...])` (cs war 1-D) ab.
    """
    if len(grounds) == 0:
        return float("nan"), float("inf"), np.array([0., 0., float("nan")])
    cs = np.array([[c[0] * cell - origin[0], c[1] * cell - origin[1]] for c in grounds])
    z = np.array(list(grounds.values()))
    g_med = np.median(z)
    mad = np.median(np.abs(z - g_med))
    if len(z) < 3:
        # < 3 Zellen: keine robuste Ebene moeglich -> nur Shift, kein Tilt.
        return g_med, mad, np.array([0., 0., g_med])
    A = np.stack([cs[:, 0], cs[:, 1], np.ones(len(z))], 1)
    keep = np.ones(len(z), bool); co = np.array([0., 0., g_med])
    for _ in range(3):
        co, *_ = np.linalg.lstsq(A[keep], z[keep], rcond=None)
        r = z - A @ co; keep = np.abs(r) < 2.5 * (r[keep].std() + 1e-9)
    return g_med, mad, co


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plys", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cell", type=float, default=25.0)
    ap.add_argument("--zbins", type=int, default=25)
    ap.add_argument("--min-cell-pts", type=int, default=30)
    ap.add_argument("--detilt", action="store_true", help="zusätzlich horizontalen De-Tilt (nur reliable Segmente)")
    ap.add_argument("--max-tilt", type=float, default=120.0, help="max. erlaubter Boden-Tilt (mm/m) für De-Tilt")
    ap.add_argument("--reliab-mad", type=float, default=8.0, help="MAD-Schwelle (m): darüber = nur Shift, kein Tilt")
    ap.add_argument("--clip-scale", type=float, default=2.0, help="Floater-Vorfilter: max Gaussian-Scale (m); 0=aus")
    ap.add_argument("--zband", type=float, default=120.0, help="behalte nur |z-ref|<zband (m) nach Leveling")
    a = ap.parse_args()

    # --- Pass 1: Bodenfelder + globalen Origin bestimmen ---
    info = []  # (path, g_med, mad, plane_coef, origin_local-unused)
    grids = []
    all_centers = []
    for p in a.plys:
        v = PlyData.read(p)["vertex"].data
        xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
        m = np.isfinite(xyz).all(1)
        if a.clip_scale > 0 and "scale_0" in v.dtype.names and "scale_1" in v.dtype.names:
            sc = np.exp(np.maximum(v["scale_0"], v["scale_1"]))
            m &= sc < a.clip_scale
        g = cell_mode_grounds(xyz[m], a.cell, a.zbins, a.min_cell_pts)
        grids.append(g)
        all_centers += [[c[0] * a.cell, c[1] * a.cell] for c in g]
    origin = np.mean(all_centers, 0)

    levels = []
    for p, g in zip(a.plys, grids):
        g_med, mad, co = fit_level(g, a.cell, origin)
        levels.append((g_med, mad, co))
    # reliabilitäts-gewichtete Referenzhöhe
    gmeds = np.array([l[0] for l in levels]); mads = np.array([l[1] for l in levels])
    w = 1.0 / (mads + 1.0)
    order = np.argsort(gmeds)
    cw = np.cumsum(w[order]); ref = gmeds[order][np.searchsorted(cw, 0.5 * cw[-1])]  # gewichteter Median
    print(f"[gps-level] Referenz-Bodenhöhe ref={ref:.1f} m (reliab.-gewichteter Median)")
    print(f"{'seg':16s} {'ground':>7s} {'MAD':>5s} {'tilt mm/m':>9s} {'shift':>7s} {'mode'}")

    # --- Pass 2: anwenden + segmentweise als Binär-Body streamen ---
    # Statt alle Segmente zu np.concatenate'n (Peak ~2x Gesamtgröße im RAM -> OOM bei
    # ~26M Gaussians), schreiben wir jedes Segment einzeln als rohe little-endian Bytes
    # in eine Temp-Body-Datei (nur 1 Segment gleichzeitig im RAM). Header (mit Gesamt-
    # Count) wird danach davor gehängt.
    tmp = a.out + ".body.tmp"
    total = 0; dtype0 = None
    with open(tmp, "wb") as body:
        for p, (g_med, mad, co) in zip(a.plys, levels):
            v = PlyData.read(p)["vertex"].data
            xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
            X = xyz[:, 0] - origin[0]; Y = xyz[:, 1] - origin[1]
            tilt = np.hypot(co[0], co[1]) * 1000
            do_tilt = a.detilt and mad < a.reliab_mad and tilt < a.max_tilt
            if do_tilt:
                # Boden-Ebene -> horizontal auf ref ziehen: z -= (plane(x,y) - ref)
                dz = ref - (co[0] * X + co[1] * Y + co[2]); mode = "shift+detilt"
            else:
                dz = ref - g_med; mode = "shift-only"
            znew = xyz[:, 2] + dz
            keep = np.isfinite(xyz).all(1)
            if a.clip_scale > 0 and "scale_0" in v.dtype.names and "scale_1" in v.dtype.names:
                keep &= np.exp(np.maximum(v["scale_0"], v["scale_1"])) < a.clip_scale
            keep &= np.abs(znew - ref) < a.zband        # grobes z-Band um die Referenz (Floater raus)
            out = v[keep].copy(); out["z"] = znew[keep].astype(out["z"].dtype)
            if dtype0 is None:
                dtype0 = out.dtype
            elif out.dtype != dtype0:
                raise ValueError(f"dtype-Mismatch zwischen Segmenten: {out.dtype} != {dtype0} ({p})")
            body.write(np.ascontiguousarray(out).tobytes())
            total += int(keep.sum())
            del v, xyz, out  # RAM früh freigeben
            sh = float(np.median(dz)) if np.ndim(dz) else float(dz)
            print(f"{os.path.basename(p):16s} {g_med:7.1f} {mad:5.1f} {tilt:9.1f} {sh:7.1f}  {mode}  ({keep.sum()/1e3:.0f}k)")

    # Finale PLY: ASCII-Header (binary_little_endian, dtype0 ist '<f4' -> portabel) + Body.
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {total}\n" + "\n".join(_ply_prop_lines(dtype0)) + "\nend_header\n")
    with open(a.out, "wb") as f:
        f.write(header.encode("ascii"))
        with open(tmp, "rb") as b:
            while True:
                chunk = b.read(1 << 24)  # 16 MiB
                if not chunk:
                    break
                f.write(chunk)
    os.remove(tmp)
    print(f"[gps-level] {total} Gaussians -> {a.out}")


if __name__ == "__main__":
    main()
