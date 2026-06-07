#!/usr/bin/env python3
"""Sequentielles Chunk-Chaining gegen die Rest-Naht (Position + Rotation).

Motivation (User 2026-06-03): jeder Chunk wird unabhaengig per Sim3 an GT
ausgerichtet -> Chunk N+1 erbt weder Lage noch Drehung von Chunk N, daher
relativer Versatz/Kippung an der Naht. Fix: Chunk 0 = Anker, jeder folgende
Chunk wird ueber seine UEBERLAPP-Frames an die schon platzierten Vorgaenger
"angeklebt" (Kontinuitaet), auf den restlichen Frames aber weiter an GT gehalten
(kein akkumulierender Drift). Die GT-Posen liefern Position, der Overlap die
relative Drehung.

Mathematik:
  - Pro Chunk i: unabhaengiger GT-Sim3  T_i: C_droid_i -> G_gt_i  (Umeyama).
  - Chunk 0 unveraendert (Anker an GT).
  - Chunk i>=1: Ziel pro Frame f =
       W_prev(f)  wenn f mit einem schon platzierten Chunk ueberlappt (Gewicht w_ov)
       G_gt(f)    sonst                                              (Gewicht 1)
    Gewichteter Umeyama  C_droid_i -> Ziele  =>  T_i'.
  - Auf die schon-metrische fix.ply wird die KORREKTUR  Δ_i = T_i' ∘ T_i^{-1}
    angewandt (xyz, scale += log(Δs), rot = ΔR ∘ q). Keine Roh-PLY noetig.

Usage:
  python scripts/eval/chain_chunks.py \
    --chunks c3000:FIX0.ply:RUNDIR0  c3075:FIX1.ply:RUNDIR1 ... \
    --gt-poses .../poses_w2c.txt --out-dir OUTDIR [--overlap-weight 5] [--match-dt 0.06]
Schreibt OUTDIR/<label>_chained.ply je Chunk und gibt die Liste auf stdout aus.
"""
import argparse, os, numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as Rot


def umeyama_w(src, dst, w=None):
    """Gewichteter Sim3 src->dst (Nx3)."""
    if w is None:
        w = np.ones(len(src))
    w = w / w.sum()
    mu_s = (w[:, None] * src).sum(0)
    mu_d = (w[:, None] * dst).sum(0)
    s0, d0 = src - mu_s, dst - mu_d
    cov = (w[:, None] * d0).T @ s0
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var = (w * (s0 ** 2).sum(1)).sum()
    s = np.trace(np.diag(D) @ S) / var
    t = mu_d - s * R @ mu_s
    return s, R, t


def load_droid(path):
    r = np.loadtxt(path, comments="#")
    if r.ndim == 1:
        r = r[None]
    t = r[:, 1]
    C = r[:, [5, 9, 13]]                          # centers
    Rc2w = r[:, [2, 3, 4, 6, 7, 8, 10, 11, 12]].reshape(-1, 3, 3)  # cam->droidworld
    return t, C, Rc2w


def rot_mean(Rs):
    U, _, Vt = np.linalg.svd(Rs.sum(0))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    return U @ S @ Vt


def load_gt(path):
    g = np.loadtxt(path, comments="#")
    Rc2w = Rot.from_quat(g[:, 4:8]).as_matrix().transpose(0, 2, 1)
    C = -np.einsum('nij,nj->ni', Rc2w, g[:, 1:4])
    return g[:, 0], C


def apply_sim3_ply(ply_in, ply_out, s, R, t):
    v = PlyData.read(ply_in)["vertex"].data
    out = v.copy()
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    xyz2 = s * (R @ xyz.T).T + t
    out["x"], out["y"], out["z"] = xyz2[:, 0], xyz2[:, 1], xyz2[:, 2]
    for sc in [p for p in v.dtype.names if p.startswith("scale_")]:
        out[sc] = (v[sc].astype(np.float64) + np.log(s)).astype(v[sc].dtype)
    rn = sorted([p for p in v.dtype.names if p.startswith("rot")])
    if len(rn) == 4:
        q = np.stack([v[rn[0]], v[rn[1]], v[rn[2]], v[rn[3]]], 1).astype(np.float64)
        nrm = np.linalg.norm(q, axis=1, keepdims=True); nrm[nrm == 0] = 1
        qx = (q / nrm)[:, [1, 2, 3, 0]]
        qn = (Rot.from_matrix(R) * Rot.from_quat(qx)).as_quat()
        out[rn[0]], out[rn[1]], out[rn[2]], out[rn[3]] = qn[:, 3], qn[:, 0], qn[:, 1], qn[:, 2]
    PlyData([PlyElement.describe(out, "vertex")]).write(ply_out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", nargs="+", required=True, help="label:fix.ply:rundir je Chunk")
    ap.add_argument("--gt-poses", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--overlap-weight", type=float, default=5.0)
    ap.add_argument("--match-dt", type=float, default=0.06, help="Zeit-Toleranz (s) fuer Frame-Match")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    tg, Cg = load_gt(a.gt_poses)

    chunks = []
    for spec in a.chunks:
        label, ply, rd = spec.split(":", 2)
        td, Cd, Rd = load_droid(os.path.join(rd, "tracker_raw_c2w.txt"))
        idx = np.clip(np.searchsorted(tg, td), 0, len(tg) - 1)
        s, R, t = umeyama_w(Cd, Cg[idx])            # unabhaengiger GT-Sim3 T_i (nur als Δ-Referenz)
        chunks.append(dict(label=label, ply=ply, td=td, Cd=Cd, Rd=Rd, Gd=Cg[idx],
                           s=s, R=R, t=t))
    chunks.sort(key=lambda c: c["td"].min())

    # schon platzierte Welt-Kameras: Zeit, Welt-Zentrum, Welt-Orientierung (cam->welt)
    pt, pW, pO = [], [], []
    outs = []
    for i, c in enumerate(chunks):
        Cd, Rd, Gd, td = c["Cd"], c["Rd"], c["Gd"], c["td"]
        if i == 0:
            sp, Rp, tp = c["s"], c["R"], c["t"]      # Anker = GT-Sim3
            print(f"[chain] {c['label']}: ANCHOR (GT)  scale={sp:.1f}")
        else:
            PT = np.concatenate(pt); PW = np.concatenate(pW); PO = np.concatenate(pO)
            ov = []                                   # (k_in_chunk, j_in_placed)
            for k, tk in enumerate(td):
                j = int(np.argmin(np.abs(PT - tk)))
                if abs(PT[j] - tk) < a.match_dt:
                    ov.append((k, j))
            if len(ov) >= 3:
                ks = np.array([o[0] for o in ov]); js = np.array([o[1] for o in ov])
                # (1) Rotation aus ORIENTIERUNGEN (gut konditioniert, unabh. von Bahn-Geometrie):
                #     welt-orient(f) = Rp @ Rd_i(f)  soll = platzierte welt-orient(f)
                #     -> Rp = mean_f[ PO(f) @ Rd_i(f)^T ]
                Rp = rot_mean(np.einsum('nij,nkj->nik', PO[js], Rd[ks]))
                # (2) Scale+Translation bei FIXEM Rp aus Positionen (Overlap an Vorgaenger, Rest an GT):
                target = Gd.copy(); w = np.ones(len(td))
                target[ks] = PW[js]; w[ks] = a.overlap_weight
                p = (Rp @ Cd.T).T                     # rotierte droid-zentren
                wn = w / w.sum()
                mp = (wn[:, None] * p).sum(0); mt = (wn[:, None] * target).sum(0)
                dp = p - mp; dt_ = target - mt
                sp = (wn * (dp * dt_).sum(1)).sum() / (wn * (dp * dp).sum(1)).sum()
                tp = mt - sp * mp
                n_ov = len(ov)
            else:
                sp, Rp, tp = c["s"], c["R"], c["t"]   # kein Overlap -> GT-Fallback
                n_ov = 0
            si, Ri, ti = c["s"], c["R"], c["t"]
            ds = sp / si; dR = Rp @ Ri.T; dt = tp - ds * dR @ ti
            jump = np.linalg.norm((sp * (Rp @ Cd.T).T + tp) - (si * (Ri @ Cd.T).T + ti), axis=1).mean()
            print(f"[chain] {c['label']}: overlap={n_ov}f  scale={sp:.1f}  mean-shift={jump:.2f}m  "
                  f"Δscale={ds:.3f}  Δrot={np.degrees(np.linalg.norm(Rot.from_matrix(dR).as_rotvec())):.2f}deg")
        out = os.path.join(a.out_dir, c["label"] + "_chained.ply")
        if i == 0:
            apply_sim3_ply(c["ply"], out, 1.0, np.eye(3), np.zeros(3))   # fix.ply schon T_0
        else:
            apply_sim3_ply(c["ply"], out, ds, dR, dt)
        outs.append(out)
        pt.append(td); pW.append(sp * (Rp @ Cd.T).T + tp)
        pO.append(np.einsum('ij,njk->nik', Rp, Rd))    # platzierte welt-orient
    print("CHAINED_PLYS=" + " ".join(outs))


if __name__ == "__main__":
    main()
