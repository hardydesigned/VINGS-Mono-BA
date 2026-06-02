#!/usr/bin/env python3
"""Floater-Filter fuer 2DGS/3DGS-PLYs (binary little-endian).

Entfernt Ausreisser-Gaussians, die auf Nadir-Aerial durch unbestimmte Mono-Tiefe
entstehen und die Bounding-Box (und damit die Viewer-Ansicht) dominieren. Drei
Kriterien (UND-verknuepft, alle optional):

  * Distanz : Gaussians weiter als `dist_mult` x (1-99-Perzentil-Halbweite je Achse)
              vom Median entfernt -> raus (haelt die echte Szene + Rand).
  * Opacity : sigmoid(opacity) < `min_alpha` -> raus (transparente Floater).
  * Scale   : exp(max(scale_i)) > `max_scale_m` -> raus (riesige Schmier-Gaussians).

Schreibt eine neue PLY mit identischem Header (nur kleinere vertex-Zahl).

Usage: python scripts/filter_ply_floaters.py IN.ply OUT.ply
       [--dist-mult 2.0] [--min-alpha 0.05] [--max-scale 8.0]
"""
import sys, argparse, numpy as np


def read_ply(path):
    with open(path, 'rb') as f:
        hdr = b''
        while b'end_header' not in hdr:
            hdr += f.readline()
        lines = hdr.decode('ascii', 'ignore').splitlines()
        n = 0; props = []
        for l in lines:
            if l.startswith('element vertex'): n = int(l.split()[-1])
            elif l.startswith('property'):     props.append(l.split()[-1])
        data = np.frombuffer(f.read(n * len(props) * 4), dtype='<f4').reshape(n, len(props))
    return hdr, props, data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('inp'); ap.add_argument('out')
    ap.add_argument('--dist-mult', type=float, default=2.0)
    ap.add_argument('--min-alpha', type=float, default=0.05)
    ap.add_argument('--max-scale', type=float, default=8.0)
    a = ap.parse_args()

    hdr, props, data = read_ply(a.inp)
    n0 = len(data)
    xyz = data[:, :3]
    keep = np.ones(n0, dtype=bool)

    # 1) Distanz: per-Achse robuste Box (1-99-Perzentil) x dist_mult um den Median
    med = np.median(xyz, 0)
    lo, hi = np.percentile(xyz, [1, 99], axis=0)
    half = np.maximum((hi - lo) / 2.0, 1e-3)
    keep &= np.all(np.abs(xyz - med) <= a.dist_mult * half, axis=1)

    # 2) Opacity (logit gespeichert -> sigmoid)
    if 'opacity' in props:
        op = data[:, props.index('opacity')]
        alpha = 1.0 / (1.0 + np.exp(-op))
        keep &= alpha >= a.min_alpha

    # 3) Scale (log gespeichert -> exp), groesste der 2DGS-Achsen
    sc_idx = [i for i, p in enumerate(props) if p.startswith('scale_')]
    if sc_idx:
        sc = np.exp(data[:, sc_idx]).max(axis=1)
        keep &= sc <= a.max_scale

    out = data[keep]
    new_hdr = hdr.replace(f'element vertex {n0}'.encode(),
                          f'element vertex {len(out)}'.encode())
    with open(a.out, 'wb') as f:
        f.write(new_hdr)
        f.write(out.astype('<f4').tobytes())

    ext0 = xyz.max(0) - xyz.min(0)
    ext1 = out[:, :3].max(0) - out[:, :3].min(0)
    print(f"in : {n0:,} Gaussians, bbox {np.round(ext0)} m")
    print(f"out: {len(out):,} Gaussians ({100*len(out)/n0:.1f}% behalten), bbox {np.round(ext1)} m")
    print(f"-> {a.out}")


if __name__ == '__main__':
    main()
