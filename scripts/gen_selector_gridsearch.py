#!/usr/bin/env python3
"""Grid-search config generator for ONE selector on amtown03 s3100_200f.

Purpose: find the best parameter values for a single selector by brute force.
You give each config field a LIST of values to try; this emits one YAML per
element of the Cartesian PRODUCT of those lists (all permutations).

Everything except the swept fields is fixed:
  - dataset / slice : amtown03, frames 3100..3299   (override via --start/--frames/--dataset)
  - fair_eval       : on (so every config yields psnr_ho + ate_rmse_m)
  - frontend patches: plugin-selector mode (filter_thresh=-1, mapper_kf_skip=1)

Pick which selector to sweep with --kind (vista | mm3dgs; default vista).

Configuration surface = the PRESETS registry below, one entry per selector:
  base  the full frame_selector block; any field NOT in `grid` keeps this.
  grid  field -> [values...]. Product of all lists = number of configs.

Output (shared save_dir so a runner can collect one CSV):
  configs/local/gridsearch/<kind>/amtown03_s3100_200f/<name>.yaml
  where <name> encodes the swept values, e.g.
  amtown03_s3100_200f_vista__cov0p10_ang0p12_vox0p5.yaml
  amtown03_s3100_200f_mm3dgs__covis0p85_gap5.yaml

Generate, then run each serially to get metrics:
  python scripts/gen_selector_gridsearch.py --kind mm3dgs
  for f in configs/local/gridsearch/mm3dgs/amtown03_s3100_200f/*.yaml; do
      python scripts/run_experiment.py "$f"
  done
"""

from __future__ import annotations

import argparse
import copy
import itertools
from pathlib import Path

import gen_s3100_200f_configs as g
from gen_slice_configs import DATASET_BASE

REPO = Path(__file__).resolve().parents[1]
TARGET_ROOT = REPO / "configs/local/gridsearch"


# ── What to grid-search ───────────────────────────────────────────────────────
# Per-selector presets. Pick one with --kind. Each entry:
#   base : full frame_selector block; fields NOT in `grid` stay fixed at these.
#   grid : field -> [values...]; product of all lists = number of configs.
# To add a new selector, drop a "<kind>" entry here — no other change needed.
PRESETS: dict[str, dict] = {
    # "vista": {
    #     "base": {
    #         "kind": "vista",
    #         "voxel_size": 1.0,
    #         "coverage_thresh": 0.12,
    #         "angular_thresh": 0.06,
    #         "trans_thresh_m": 4.0,
    #         "rot_thresh_deg": 10.0,
    #         "n_rays_score": 256,
    #         "n_rays_integrate": 2048,
    #         "min_depth": 20.0,
    #         "max_depth": 150.0,
    #     },
    #     "grid": {
    #         "coverage_thresh": [0.05, 0.12, 0.20],
    #         "voxel_size":      [0.5, 1.0, 2.0],
    #     },
    # },
    "mm3dgs": {
        # Mirrors configs/local/amtown03/s3100_200f/mm3dgs/*.yaml (depth 0.2/35).
        "base": {
            "kind": "mm3dgs",
            "covis_thresh": 0.95,
            "niqe_window": 5,
            "n_samples": 2048,
            "min_gap_after_kf": 5,
            "force_accept_after": 0,
            "min_depth": 20.0,
            "max_depth": 150.0,
        },
        # covis_thresh is the primary knob; min_gap_after_kf controls spacing.
        "grid": {
            "covis_thresh": [0.8, 0.85, 0.90, 0.95],
            "niqe_window": [5, 7],
            "min_gap_after_kf": [0, 5, 10],
        },
    },
}

# Default selector (override with --kind) = first active PRESETS entry, so
# commenting presets in/out never breaks this binding. main() rebinds these
# three names from PRESETS based on the chosen --kind; the helpers read them.
SELECTOR_KIND = next(iter(PRESETS))
BASE_PARAMS: dict = PRESETS[SELECTOR_KIND]["base"]
GRID: dict[str, list] = PRESETS[SELECTOR_KIND]["grid"]


# Short filename tokens for the swept fields (fallback: the field name).
ABBREV = {
    "voxel_size": "vox", "coverage_thresh": "cov", "angular_thresh": "ang",
    "trans_thresh_m": "trans", "rot_thresh_deg": "rot",
    "n_rays_score": "nrs", "n_rays_integrate": "nri",
    "min_depth": "mind", "max_depth": "maxd", "max_views_per_voxel": "mvv",
    # mm3dgs
    "covis_thresh": "covis", "niqe_window": "niqe", "n_samples": "ns",
    "min_gap_after_kf": "gap", "force_accept_after": "fac",
}


def fmt(v) -> str:
    """Filesystem-safe compact value token: 0.05 -> 0p05, 1.0 -> 1, -3 -> m3."""
    s = f"{v:g}" if isinstance(v, float) else str(v)
    return s.replace("-", "m").replace(".", "p")


def build_base(dataset: str, start: int, frames: int) -> dict:
    """Load dataset template; override slice window / save_dir / fair_eval."""
    spec = DATASET_BASE[dataset]
    template: Path = spec["template"]
    if not template.exists():
        raise SystemExit(f"[FEHLER] Base template fehlt: {template}")
    with open(template) as f:
        base = g.yaml.full_load(f)
    slug = f"s{start}_{frames}f"
    base.setdefault("dataset", {})
    base["dataset"]["start_frame"] = start
    base["dataset"]["max_frames"] = frames
    base["output"]["save_dir"] = str(
        REPO / "output" / "exp_gridsearch" / f"{SELECTOR_KIND}_{dataset}_{slug}"
    ) + "/"
    base["fair_eval"] = copy.deepcopy(spec["fair_eval"])
    return base


def make_selector_block(combo: dict) -> dict:
    fs = copy.deepcopy(BASE_PARAMS)
    fs.update(combo)
    return fs


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", default=next(iter(PRESETS)), choices=sorted(PRESETS),
                    help="which selector's preset to grid-search")
    ap.add_argument("--dataset", default="amtown03", choices=sorted(DATASET_BASE))
    ap.add_argument("--start", type=int, default=3100)
    ap.add_argument("--frames", type=int, default=200)
    args = ap.parse_args()

    # Rebind the module-level names the helpers read to the chosen preset.
    global SELECTOR_KIND, BASE_PARAMS, GRID
    SELECTOR_KIND = args.kind
    BASE_PARAMS = PRESETS[args.kind]["base"]
    GRID = PRESETS[args.kind]["grid"]

    # Validate grid keys against the selector block (catch typos early).
    for k in GRID:
        if k not in BASE_PARAMS:
            raise SystemExit(f"[FEHLER] GRID-Feld '{k}' ist kein {SELECTOR_KIND}-Parameter "
                             f"(bekannt: {sorted(BASE_PARAMS)})")

    slug = f"s{args.start}_{args.frames}f"
    end = args.start + args.frames - 1
    out_dir = TARGET_ROOT / SELECTOR_KIND / f"{args.dataset}_{slug}"
    name_prefix = f"{args.dataset}_{slug}_{SELECTOR_KIND}"

    keys = list(GRID)
    combos = list(itertools.product(*(GRID[k] for k in keys)))
    print(f"\n=== {SELECTOR_KIND} grid on {args.dataset} {slug} "
          f"(frames {args.start}..{end}) — {len(combos)} configs ===")
    for key in keys:
        print(f"    {key}: {GRID[key]}")
    print()

    base = build_base(args.dataset, args.start, args.frames)
    for values in combos:
        combo = dict(zip(keys, values))
        token = "_".join(f"{ABBREV.get(k, k)}{fmt(v)}" for k, v in combo.items())
        name = f"{name_prefix}__{token}"

        cfg = copy.deepcopy(base)
        cfg["frame_selector"] = make_selector_block(combo)
        cfg["mapper_kf_skip"] = 1
        cfg["frontend"]["filter_thresh"] = -1.0
        cfg["frontend"]["keyframe_thresh"] = 0.0
        cfg.pop("frame_skip", None)

        path = out_dir / f"{name}.yaml"
        header = (
            f"# Auto-generated by scripts/gen_selector_gridsearch.py — DO NOT EDIT.\n"
            f"# selector = {SELECTOR_KIND}   grid point = {combo}\n"
            f"# dataset  = {DATASET_BASE[args.dataset]['frame_comment']} "
            f"(frames {args.start}..{end})\n"
        )
        g.write_yaml(path, cfg, header)
        print(f"  {path.relative_to(REPO)}")

    print(f"\n=== {len(combos)} configs -> {out_dir.relative_to(REPO)}/ ===")


if __name__ == "__main__":
    main()
