#!/usr/bin/env python3
"""Echter 2DGS-Render einer fertigen PLY aus (GT-)Drohnenposen.

Anders als render_ply_check.py (matplotlib-Scatter der Gaussian-Zentren ->
grauer Nebel) rastert dieses Skript die Gaussians mit dem *echten*
diff_surfel-Rasterizer aus den nadir-Flugposen. Ergebnis: photo- aehnliche
Bodenbilder, an denen man die tatsaechliche Map-Schaerfe ablesen kann.

Usage:
  python scripts/eval/render_ply_views.py PLY \
      --poses .../vings/poses_w2c.txt \
      --frames 600,1500,2700,3900,5100 \
      --res 480 576 --out views.png
"""
import os, sys, argparse, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lietorch import SE3
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from gaussian.cameras import get_camera
from gaussian.vis_utils import load_ply

NATIVE = dict(fu=1469.4899, fv=1469.4899, W=2448, H=2048)


def render_view(g, w2c, intr, crop_r=250.0):
    # nadir sieht nur ~crop_r m Umkreis -> spatialer Crop spart VRAM massiv
    cam_c = torch.inverse(w2c)[:3, 3]
    d2 = ((g['xyz'][:, :2] - cam_c[:2]) ** 2).sum(1)
    keep = d2 < crop_r * crop_r
    gv = {k: v[keep] for k, v in g.items()}
    cam = get_camera(w2c, intr)
    pixel_mask = torch.ones(int(cam.height) * int(cam.width), dtype=torch.bool).cuda()
    rs = GaussianRasterizationSettings(
        image_height=int(cam.height), image_width=int(cam.width),
        tanfovx=cam.tanfovx, tanfovy=cam.tanfovy,
        bg=torch.zeros(3, device="cuda"), scale_modifier=1.0,
        viewmatrix=cam.world_view_transform, projmatrix=cam.full_proj_transform,
        sh_degree=0, campos=cam.camera_center, prefiltered=False, debug=False,
        pixel_mask=pixel_mask)
    rasterizer = GaussianRasterizer(raster_settings=rs)
    zeros = torch.zeros_like(gv['xyz'][:, :2]).contiguous()
    img, radii, allmap = rasterizer(
        means3D=gv['xyz'], means2D=torch.zeros_like(gv['xyz']),
        shs=None, colors_precomp=gv['rgb'], opacities=gv['opacity'],
        scales=gv['scaling'], rotations=gv['rotation'],
        scores=zeros, cov3D_precomp=None)
    return torch.clamp(img, 0, 1).permute(1, 2, 0).detach().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ply")
    ap.add_argument("--poses", default="/home/philipp/Dokumente/datasets/interval1_AMtown03/vings/poses_w2c.txt")
    ap.add_argument("--frames", default="600,1500,2700,3900,5100")
    ap.add_argument("--res", nargs=2, type=int, default=[480, 576], help="H W")
    ap.add_argument("--crop", type=float, default=250.0, help="nadir crop radius m")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rh, rw = args.res
    fu = NATIVE['fu'] * rh / NATIVE['H']
    fv = NATIVE['fv'] * rw / NATIVE['W']
    intr = {'W': rw, 'H': rh, 'fu': fu, 'fv': fv}

    print(f"[render] loading {args.ply}")
    p = load_ply(args.ply)
    g = {
        'xyz':      torch.tensor(p['_xyz'], dtype=torch.float32, device="cuda"),
        'rgb':      torch.tensor(p['_rgb'], dtype=torch.float32, device="cuda").clamp(0, 1),
        'opacity':  torch.sigmoid(torch.tensor(p['_opacity'], dtype=torch.float32, device="cuda")),
        'scaling':  torch.exp(torch.tensor(p['_scaling'], dtype=torch.float32, device="cuda")),
        'rotation': torch.nn.functional.normalize(torch.tensor(p['_rotation'], dtype=torch.float32, device="cuda")),
    }
    print(f"[render] {g['xyz'].shape[0]} gaussians, render {rh}x{rw}")

    poses = np.loadtxt(args.poses)
    frames = [int(x) for x in args.frames.split(",")]
    n = len(frames)
    cols = min(n, 3)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4.2))
    axes = np.atleast_1d(axes).ravel()
    for k, fr in enumerate(frames):
        fr = min(fr, poses.shape[0] - 1)
        tq = torch.tensor(poses[fr, 1:8], dtype=torch.float32)
        w2c = SE3(tq).matrix().cuda()
        img = render_view(g, w2c, intr, crop_r=args.crop)
        axes[k].imshow(img)
        axes[k].set_title(f"frame {fr}", fontsize=10)
        axes[k].axis("off")
    for k in range(n, len(axes)):
        axes[k].axis("off")
    plt.tight_layout()
    out = args.out or os.path.splitext(args.ply)[0] + "_views.png"
    plt.savefig(out, dpi=110)
    print(f"[render] wrote {out}")


if __name__ == "__main__":
    main()
