#!/usr/bin/env python3
"""Generate the *validated* single-config-per-sequence tree, ordered by selector.

Unlike gen_slice_configs.py (which fans a full baseline/mapskip/selector SWEEP
out per slice), this emits ONE config per (selector, sequence): the validated
default parameters for that selector, ready to run and score with fair_eval.

Output layout — grouped by selector first, then one YAML per sequence:

    configs/local/validated_selectors/<selector>/<dataset>_s<start>_<frames>f_<selector>.yaml

e.g. configs/local/validated_selectors/vista/amtown03_s3100_200f_vista.yaml
     configs/local/validated_selectors/vista/HKisland03_s2600_200f_vista.yaml

The sequence list mirrors docs/results/<dataset>_s<start>_<frames>f_results.csv
exactly (the slices this BA actually evaluated). Per-dataset base template,
intrinsics and fair_eval block are reused verbatim from gen_slice_configs.py's
DATASET_BASE registry — only the slice window, output.save_dir and the
frame_selector block are overridden.

Currently only `vista` is registered (with the CORRECTED coverage/angular gate
fields — NOT the dead `gain_thresh` key that SELECTOR_DEFAULTS still carries).
Add a selector by dropping one entry into VALIDATED_SELECTORS below.

Run every generated config standalone to get both metric families:
    python scripts/run_experiment.py <config>     # -> metrics.json + fair_metrics.json

Examples:
    python scripts/gen_validated_selectors.py                       # all selectors, all sequences
    python scripts/gen_validated_selectors.py --selectors vista
    python scripts/gen_validated_selectors.py --datasets amtown03 agz
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

# Reuse the canonical write helper + the per-dataset base-template/fair_eval
# registry. Both imports are side-effect-free (main() is __main__-guarded).
import gen_s3100_200f_configs as g
from gen_slice_configs import DATASET_BASE

REPO = Path(__file__).resolve().parents[1]
TARGET_ROOT = REPO / "configs/local/validated_selectors"


# ── Sequences to cover (mirrors docs/results/<dataset>_s<start>_<frames>f_*) ──
# Each entry: (start_frame, n_frames). Kept in sync with the result CSVs so the
# validated set lines up 1:1 with the slices already swept.
SEQUENCES: dict[str, list[tuple[int, int]]] = {
    "amtown03":    [(400, 200), (3100, 200), (5000, 200), (1000, 400), (5400, 400)],
    "agz":         [(2000, 200), (5600, 200), (7950, 200), (2675, 400), (9000, 400)],
    "AMvalley03":  [(2900, 200), (800, 400), (4300, 500)],
    "HKairport03": [(1600, 200), (670, 400), (2800, 500)],
    "HKisland03":  [(2600, 200), (770, 400), (2800, 500)],
}


# ── Validated selector parameters (frame_selector block) ─────────────────────
# The dict KEY is a free LABEL (also the filename suffix); the "kind" field
# inside is the real selector the factory dispatches on AND the folder it lands
# in. So multiple labels can share one `kind` (parameter variants) and all sit
# together under configs/local/validated_selectors/<kind>/.
#
# vista: the CORRECT gate fields that FrameSelectorConfig actually reads.
# `gain_thresh` was split into coverage_thresh + angular_thresh; the old sweep
# configs still emit the dead `gain_thresh` key (silently dropped by
# from_config). These are the code defaults, i.e. the honest baseline.
VALIDATED_SELECTORS: dict[str, dict] = {
    # VISTA Varianten
    # varieren über voxel_size und coverage_thresh
    # voxel_size: {0.5, 1.0, 2.0, 3.0}
    # coverage_thresh: 0.05, 0.10, 0.15, 0.20, 0.30}
    "vista_cov0p1_ang0p12_vox1": {
        "kind": "vista",
        "voxel_size": 1.0,
        "coverage_thresh": 0.12,
        "angular_thresh": 0.12,
        "trans_thresh_m": 4.0,
        "rot_thresh_deg": 10.0,
        "n_rays_score": 256,
        "n_rays_integrate": 2048,
        "min_depth": 20.0,
        "max_depth": 150.0,
    },
    "vista_cov0p2_vox2": {
        "kind": "vista",
        "voxel_size": 2.0,
        "coverage_thresh": 0.2,
        "angular_thresh": 0.06,
        "trans_thresh_m": 4.0,
        "rot_thresh_deg": 10.0,
        "n_rays_score": 256,
        "n_rays_integrate": 2048,
        "min_depth": 20.0,
        "max_depth": 150.0,
    },
        "vista_cov0p05_ang0p12_vox1": {
        "kind": "vista",
        "voxel_size": 1.0,
        "coverage_thresh": 0.05,
        "angular_thresh": 0.12,
        "trans_thresh_m": 4.0,
        "rot_thresh_deg": 10.0,
        "n_rays_score": 256,
        "n_rays_integrate": 2048,
        "min_depth": 20.0,
        "max_depth": 150.0,
    },

}


def patch_validated(base: dict, sel: str) -> dict:
    """Apply the validated selector block + the standard plugin-selector
    frontend patches (identical semantics to gen_s3100_200f_configs.patch_selector,
    minus the sweep-only override machinery)."""
    c = copy.deepcopy(base)
    c["frame_selector"] = copy.deepcopy(VALIDATED_SELECTORS[sel])
    c["mapper_kf_skip"] = 1
    c["frontend"]["filter_thresh"] = -1.0       # plugin selector replaces Stage 1
    c["frontend"]["keyframe_thresh"] = 0.0
    c.pop("frame_skip", None)
    return c


def build_base(dataset: str, start: int, frames: int) -> dict:
    """Load the dataset template and override slice window / save_dir / fair_eval."""
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
    # Separate output root so validated runs don't clobber the sweep dirs.
    base["output"]["save_dir"] = str(
        REPO / "output" / "exp_validated" / f"{dataset}_{slug}"
    ) + "/"
    base["fair_eval"] = copy.deepcopy(spec["fair_eval"])
    return base


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selectors", nargs="+", default=sorted(VALIDATED_SELECTORS),
                    choices=sorted(VALIDATED_SELECTORS),
                    help="which selectors to emit (default: all registered)")
    ap.add_argument("--datasets", nargs="+", default=sorted(SEQUENCES),
                    choices=sorted(SEQUENCES),
                    help="which datasets to emit (default: all)")
    args = ap.parse_args()

    total = 0
    for sel in args.selectors:
        kind = VALIDATED_SELECTORS[sel]["kind"]   # folder = base selector kind
        for dataset in args.datasets:
            for start, frames in SEQUENCES[dataset]:
                slug = f"s{start}_{frames}f"
                end = start + frames - 1
                name = f"{dataset}_{slug}_{sel}"    # filename suffix = label
                base = build_base(dataset, start, frames)
                cfg = patch_validated(base, sel)
                path = TARGET_ROOT / kind / f"{name}.yaml"
                header = (
                    f"# Auto-generated by scripts/gen_validated_selectors.py — DO NOT EDIT.\n"
                    f"# selector = {kind} (label: {sel})\n"
                    f"# dataset  = {DATASET_BASE[dataset]['frame_comment']} "
                    f"(frames {start}..{end})\n"
                )
                g.write_yaml(path, cfg, header)
                print(f"  {path.relative_to(REPO)}")
                total += 1

    print(f"\n=== {total} validated configs written under "
          f"{TARGET_ROOT.relative_to(REPO)}/ ===")


if __name__ == "__main__":
    main()
