"""
Coko-SLAM-style keyframe selector.

Implements the feature-vector-based keyframe selection from Li et al.,
"Compact Keyframe-Optimized Multi-Agent Gaussian Splatting SLAM"
(Coko-SLAM), arXiv:2604.00804, April 2026, Section 3.1.

This implementation is aligned with the original reference code at
https://github.com/lemonci/coko-slam (`src/entities/agent.py`,
`src/entities/loop_detection/feature_extractors.py`,
`configs/ReplicaMultiagent/replica_multiagent.yaml`). Where the paper
text and the repo disagree on detail, we follow the repo and document
the divergence below.

Algorithm per Tracker-KF candidate
----------------------------------

1. Extract a DINOv2-Small feature ϕ(E) ∈ R^384 from the RGB frame, mean
   over CLS + patch tokens, then L2-normalize (so cos(a,b) = a·b).
2. Submap stage — decide whether the *current* submap is done:
        c_anchor = cos(ϕ(E), ϕ(submap_anchor))
        if (1 - c_anchor) > submap_threshold AND
           n_kfs_in_submap   >= min_kfs_per_submap:
            close submap → reset memory → seed new submap with this frame
            return ACCEPT
3. Keyframe stage — inside the current submap:
        c_max = max_{K in submap_memory} cos(ϕ(E), ϕ(K))
        if (1 - c_max) > alpha:
            ACCEPT → add to memory
        else:
            skip

In repo code: `should_start_new_submap` + `should_start_mapping` in
`src/entities/agent.py`. The two are checked in order; if a frame opens a
new submap, the keyframe check is skipped (the seed is the new KF).

Repo defaults (Replica config)
------------------------------
  alpha (keyframing_threshold)   = 0.02   (cosine distance)
  submap_threshold               = 0.05   (cosine distance)
  min_kfs_per_submap (keyframe_num) = 10
  feature_extractor              = DINOv2-Small via HuggingFace
  feature                        = last_hidden_state.mean(dim=1)
  L2-normalize                   = yes
  search                         = FAISS IndexFlatIP (inner product on
                                   unit vectors = cosine similarity)

Adaptations to VINGS (single-agent)
-----------------------------------
- The paper / repo store submaps as actual data structures and ship them
  to a central server. VINGS is single-agent, so the "submap" here only
  exists as the selector's local comparison memory; closing a submap
  means clearing this memory. No on-disk artifact is produced.
- L2 distance is offered as a legacy `distance_metric` mode for runs
  predating the repo lookup (some configs were tuned against L2-distance
  thresholds). The paper notation `||phi(E)-phi(K)||` and the repo's
  cosine-distance are monotonically related on unit vectors via
  `||a-b||^2 = 2(1 - cos(a,b))`, but the threshold value is not the
  same. Default is cosine, matching the repo.
- `fifo` memory mode is offered for ablation: drop-oldest sliding window
  instead of submap-reset. Not paper, useful for comparison.
- `force_accept_all` diagnostic mode (not in paper) accepts every frame
  while still logging the score, for threshold calibration.

Calling convention (shared with the other selectors)
----------------------------------------------------
    sel = CokoSlamSelector.from_config(cfg_dict, K, (H, W))
    accept, score = sel.should_accept(depth, t, R, rgb=rgb_uint8_bgr)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError as e:
    raise ImportError("CokoSlamSelector requires torch.") from e


# =============================================================================
# DINOv2 cache compatibility patch
# =============================================================================
# DINOv2's main branch (May 2024+) uses Python 3.10+ union syntax (`float | None`)
# in annotation positions outside of `from __future__ import annotations`. On
# Python 3.9 these are evaluated at class-construction time and raise TypeError.
# We patch the cached source files idempotently by prepending the future-import.

_DINOV2_PATCH_FILES = ("layers/attention.py", "layers/block.py")
_FUTURE_IMPORT = "from __future__ import annotations\n"


def _patch_dinov2_cache() -> None:
    """Add `from __future__ import annotations` to DINOv2 cache files if missing.

    Idempotent. No-op if the hub cache directory does not yet exist; the first
    torch.hub.load() will create it, and we patch on the second call. To handle
    the very first load, we call this *after* a failed load attempt and retry.
    """
    hub_root = os.path.expanduser("~/.cache/torch/hub")
    dinov2_root = os.path.join(hub_root, "facebookresearch_dinov2_main", "dinov2")
    if not os.path.isdir(dinov2_root):
        return
    for rel in _DINOV2_PATCH_FILES:
        path = os.path.join(dinov2_root, rel)
        if not os.path.isfile(path):
            continue
        with open(path, "r") as f:
            content = f.read()
        if content.startswith(_FUTURE_IMPORT) or "from __future__ import annotations" in content[:200]:
            continue
        with open(path, "w") as f:
            f.write(_FUTURE_IMPORT + content)


# =============================================================================
# Config / data classes
# =============================================================================

@dataclass
class CokoSlamConfig:
    # Keyframing threshold. On L2-normalized features:
    #   distance_metric="cosine" -> threshold on (1 - cos), i.e. cosine distance.
    #                               Repo default: 0.02.
    #   distance_metric="l2"     -> threshold on Euclidean distance.
    #                               Equivalent cos-dist: alpha_l2^2 / 2.
    # Accept iff the per-frame distance exceeds alpha.
    alpha: float = 0.02

    # Submap-reset trigger threshold (same metric as `alpha`). When the
    # distance from the current frame to the *submap anchor* (= first
    # frame of the current submap) exceeds this, AND the submap already
    # holds at least `min_kfs_per_submap` accepted frames, the submap is
    # closed and the current frame seeds a new submap. Repo default 0.05.
    # Set to inf to disable data-driven reset (then only `max_kfs` cap
    # applies, in either `submap_reset` or `fifo` mode).
    submap_threshold: float = 0.05

    # Minimum number of accepted KFs in a submap before a data-driven
    # submap reset is allowed (`keyframe_num` in the repo config).
    min_kfs_per_submap: int = 10

    # Hard cap on memory size. Acts as a safety net so a very static scene
    # cannot grow the comparison set without bound. When reached, the
    # selected `memory_mode` decides what happens. 0 = no cap (rely solely
    # on data-driven reset).
    max_kfs: int = 0

    # Memory-eviction policy when the cap `max_kfs > 0` is hit:
    #   "submap_reset"  clear memory, start fresh submap (analogous to the
    #                   data-driven reset). Default.
    #   "fifo"          drop-oldest sliding window. Non-paper; ablation.
    memory_mode: str = "submap_reset"

    # Distance metric for both `alpha` and `submap_threshold`.
    #   "cosine"  (default, repo-faithful): `1 - cos(a, b)` on L2-norm
    #              vectors. Range [0, 2].
    #   "l2"      Euclidean on L2-norm vectors. Range [0, 2]. Kept for
    #              legacy configs predating the repo lookup.
    distance_metric: str = "cosine"

    # DINOv2 model id for torch.hub. Options: dinov2_vits14 (384-dim, ~22M),
    # dinov2_vitb14 (768-dim, ~86M), dinov2_vitl14 (1024-dim, ~300M).
    model_name: str = "dinov2_vits14"

    # Resize target for DINOv2 input. MUST be a multiple of 14 (patch size).
    image_size: int = 224

    # Device for the backbone. "cuda" or "cpu".
    device: str = "cuda"

    # Feature aggregation over the ViT token sequence:
    #   "patch_mean_with_cls"  (repo): mean over [CLS, patch_1, ...].
    #                                  Matches HuggingFace
    #                                  `last_hidden_state.mean(dim=1)`.
    #   "cls"                  legacy: only the CLS token. Different feature.
    feature_aggregation: str = "patch_mean_with_cls"

    # Diagnostic mode: accept everything but still record the score
    # (used to calibrate alpha / submap_threshold).
    force_accept_all: bool = False


@dataclass
class CokoSlamScore:
    # Closest-keyframe distance under the configured metric. 0 means
    # bootstrap (no comparison was possible).
    min_dist: float = float("inf")
    # Distance to the submap anchor (frame that opened the current
    # submap). Inf before the first accept of a submap.
    submap_anchor_dist: float = float("inf")
    # Number of features currently in the submap memory at decision time.
    n_refs: int = 0
    # Threshold used for the keyframing decision (`cfg.alpha`).
    alpha: float = 0.0
    # Was the accept forced (bootstrap or `force_accept_all`)?
    forced: bool = False
    # Final decision.
    accepted: bool = False
    # Did this call open a new submap (anchor moved, memory cleared)?
    submap_reset: bool = False
    # Running submap counter (0 = first submap, increments on reset).
    submap_idx: int = 0


# =============================================================================
# Selector
# =============================================================================

class CokoSlamSelector:
    """Coko-SLAM-style keyframe selector.

    Implements the *two-stage* selection algorithm from the reference
    repo: a submap-reset check followed by an in-submap keyframe check.
    """

    def __init__(self, cfg: CokoSlamConfig, K: np.ndarray, image_hw: tuple[int, int]):
        self.cfg = cfg
        self.K = np.asarray(K, dtype=np.float32)
        self.H, self.W = image_hw

        if cfg.image_size % 14 != 0:
            raise ValueError(
                f"CokoSlamConfig.image_size={cfg.image_size} must be a multiple of 14 "
                f"(DINOv2 patch size)."
            )
        if cfg.memory_mode not in ("submap_reset", "fifo"):
            raise ValueError(
                f"CokoSlamConfig.memory_mode={cfg.memory_mode!r} must be "
                f"'submap_reset' or 'fifo'."
            )
        if cfg.distance_metric not in ("cosine", "l2"):
            raise ValueError(
                f"CokoSlamConfig.distance_metric={cfg.distance_metric!r} must be "
                f"'cosine' (repo) or 'l2' (legacy)."
            )
        if cfg.feature_aggregation not in ("patch_mean_with_cls", "cls"):
            raise ValueError(
                f"CokoSlamConfig.feature_aggregation={cfg.feature_aggregation!r} "
                f"must be 'patch_mean_with_cls' (repo) or 'cls' (legacy)."
            )

        self.device = torch.device(cfg.device if torch.cuda.is_available()
                                   or cfg.device == "cpu"
                                   else "cpu")

        # Load DINOv2 lazily; fail loud if unavailable. On Python 3.9 the
        # DINOv2 main-branch source raises TypeError because of `float | None`
        # union syntax — patch the cache and retry once.
        #
        # Prefer source="local" when the cache directory exists: torch.hub.load
        # with the standard ("user/repo") syntax always pings GitHub to check
        # the branch revision, which fails offline (RemoteDisconnected) even
        # though the model is already cached.
        def _try_load():
            hub_root = os.path.expanduser("~/.cache/torch/hub")
            local_dir = os.path.join(hub_root, "facebookresearch_dinov2_main")
            if os.path.isdir(local_dir):
                return torch.hub.load(
                    local_dir, cfg.model_name, source="local", verbose=False
                )
            return torch.hub.load(
                "facebookresearch/dinov2", cfg.model_name, verbose=False
            )

        try:
            self.model = _try_load()
        except TypeError as e:
            _patch_dinov2_cache()
            try:
                self.model = _try_load()
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to load DINOv2 model '{cfg.model_name}' even after "
                    f"patching the cache for Python-3.9 compatibility. "
                    f"Original error: {e!r}; retry error: {e2!r}"
                ) from e2
        except Exception as e:
            raise RuntimeError(
                f"Failed to load DINOv2 model '{cfg.model_name}' via torch.hub. "
                f"First-time load requires internet; subsequent runs use the cache "
                f"in ~/.cache/torch/hub/. Original error: {e!r}"
            ) from e

        self.model = self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # ImageNet normalization constants on the right device.
        self._mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        # Memory of accepted keyframe features in the *current* submap
        # (each: (D,) on device). Cleared on submap reset.
        self.kf_features: list[torch.Tensor] = []
        # Anchor feature for the current submap (first KF of the submap).
        # Used for the data-driven submap-reset trigger.
        self._submap_anchor: Optional[torch.Tensor] = None
        # Running counter of submaps that have been opened.
        self._submap_idx: int = 0

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg_dict: dict, K: np.ndarray,
                    image_hw: tuple[int, int]) -> "CokoSlamSelector":
        fields = set(CokoSlamConfig.__dataclass_fields__.keys())
        kwargs = {k: v for k, v in cfg_dict.items() if k in fields}
        return cls(CokoSlamConfig(**kwargs), K, image_hw)

    # ------------------------------------------------------------------
    def should_accept(
        self,
        depth: np.ndarray,
        t: np.ndarray,
        R: np.ndarray,
        rgb: Optional[np.ndarray] = None,
        **_: object,
    ) -> tuple[bool, CokoSlamScore]:
        # RGB is mandatory — the paper algorithm is purely image-based.
        # We refuse to silently force-accept so a missing-rgb call site
        # surfaces as an error rather than as "selector accepts everything".
        if rgb is None:
            raise RuntimeError(
                "CokoSlamSelector requires an RGB image; got rgb=None. "
                "Check the call site or use a depth-only selector."
            )

        feat = self._extract(rgb)  # (D,) L2-normalized
        score = CokoSlamScore(
            alpha=float(self.cfg.alpha),
            submap_idx=self._submap_idx,
            n_refs=len(self.kf_features),
        )

        # --- Bootstrap: first ever frame seeds submap 0. ---------------
        if self._submap_anchor is None:
            self._open_submap(feat)
            score.min_dist = 0.0
            score.submap_anchor_dist = 0.0
            score.forced = True
            score.accepted = True
            score.submap_reset = True
            return True, score

        anchor_dist = self._distance(feat, self._submap_anchor.unsqueeze(0))[0]
        score.submap_anchor_dist = float(anchor_dist)

        # --- Stage 1: submap reset (data-driven, repo Sec. 3.1). -------
        # Mirrors `should_start_new_submap()` in src/entities/agent.py: both
        # the feature divergence AND the minimum-submap-size condition
        # must hold. When triggered, the current frame becomes the seed
        # (force-accept) of a new submap.
        if (
            float(anchor_dist) > self.cfg.submap_threshold
            and len(self.kf_features) >= self.cfg.min_kfs_per_submap
        ):
            self._open_submap(feat)
            score.min_dist = 0.0  # bootstrap of the new submap
            score.forced = True
            score.accepted = True
            score.submap_reset = True
            score.submap_idx = self._submap_idx
            return True, score

        # --- Stage 2: in-submap keyframe decision. ---------------------
        # Mirrors `should_start_mapping()`: search the submap memory for
        # the closest existing KF; accept if its distance exceeds alpha.
        refs = torch.stack(self.kf_features, dim=0)        # (n, D)
        dists = self._distance(feat, refs)                 # (n,)
        d_min = float(dists.min().item())
        score.min_dist = d_min

        accept = (d_min > self.cfg.alpha) or self.cfg.force_accept_all
        if self.cfg.force_accept_all and d_min <= self.cfg.alpha:
            score.forced = True

        if accept:
            self._commit(feat)
            score.accepted = True

        return accept, score

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _open_submap(self, feat: torch.Tensor) -> None:
        """Start a new submap seeded with `feat` (and bump the counter)."""
        if self._submap_anchor is not None:
            self._submap_idx += 1
        self.kf_features.clear()
        self.kf_features.append(feat.detach())
        self._submap_anchor = feat.detach()

    def _commit(self, feat: torch.Tensor) -> None:
        """Append `feat` to the current submap's memory.

        Applies the `max_kfs` cap policy only — submap-level resets are
        handled in `should_accept`.
        """
        if self.cfg.max_kfs > 0 and len(self.kf_features) >= self.cfg.max_kfs:
            if self.cfg.memory_mode == "submap_reset":
                # Hard cap reached → treat like a submap boundary.
                self._open_submap(feat)
                return
            # fifo: drop the oldest, then append.
            self.kf_features.pop(0)
        self.kf_features.append(feat.detach())

    def _distance(self, feat: torch.Tensor, refs: torch.Tensor) -> torch.Tensor:
        """Pairwise distance between one (D,) feature and a stack of (n, D).

        Both inputs assumed L2-normalized. Returns a (n,) tensor under
        the configured `distance_metric` (cosine-dist or Euclidean).
        """
        if self.cfg.distance_metric == "cosine":
            sims = refs @ feat                       # (n,) inner products
            return 1.0 - sims
        # l2
        return torch.norm(feat[None, :] - refs, dim=1)

    def _extract(self, rgb: np.ndarray) -> torch.Tensor:
        """
        Convert (H, W, 3) uint8 BGR image to a (D,) L2-normalized feature on `self.device`.

        Feature aggregation follows `cfg.feature_aggregation`:
          - "patch_mean_with_cls": mean over CLS + all patch tokens. Matches
            the reference repo's HuggingFace `last_hidden_state.mean(dim=1)`.
          - "cls": only the LayerNorm-ed CLS token. Legacy.
        """
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"Expected (H, W, 3) image, got shape {rgb.shape}.")

        # BGR -> RGB, contiguous copy required because negative strides are not
        # supported by torch.from_numpy.
        img = np.ascontiguousarray(rgb[..., ::-1])
        tensor = torch.from_numpy(img).to(self.device).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0)                                       # (1, 3, H, W)
        tensor = (tensor - self._mean) / self._std
        tensor = F.interpolate(
            tensor,
            size=(self.cfg.image_size, self.cfg.image_size),
            mode="bilinear",
            align_corners=False,
        )

        with torch.no_grad():
            out = self.model.forward_features(tensor)
            if isinstance(out, dict):
                if self.cfg.feature_aggregation == "patch_mean_with_cls":
                    cls = out["x_norm_clstoken"].unsqueeze(1)              # (1, 1, D)
                    patches = out["x_norm_patchtokens"]                    # (1, N, D)
                    feat = torch.cat([cls, patches], dim=1).mean(dim=1)    # (1, D)
                else:  # "cls"
                    feat = out["x_norm_clstoken"]                          # (1, D)
            else:
                # Unexpected: some non-standard checkpoints return raw tensors.
                feat = out if out.ndim == 2 else out.mean(dim=1)
        feat = F.normalize(feat, dim=1).squeeze(0)
        return feat


# =============================================================================
# Smoke test
# =============================================================================

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    H, W = 256, 256
    fx = fy = 128.0
    K = np.array([[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1]], np.float32)

    # Lower the threshold for the smoketest — random synthetic patterns
    # produce smaller DINOv2 feature distances than natural images.
    try:
        cfg = CokoSlamConfig(
            alpha=0.01, submap_threshold=0.05, min_kfs_per_submap=3,
            image_size=224, max_kfs=0,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        sel = CokoSlamSelector(cfg, K, (H, W))
    except Exception as e:
        print(f"skipped: dinov2 not loadable ({e})")
        raise SystemExit(0)

    def stripe_rgb(angle_deg: float, freq: float = 0.05) -> np.ndarray:
        """Sinusoidal stripes at a given angle. Different angles -> distinct features."""
        v, u = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        theta = np.deg2rad(angle_deg)
        proj = u * np.cos(theta) + v * np.sin(theta)
        intensity = ((np.sin(proj * freq) + 1.0) * 127.5).astype(np.uint8)
        return np.stack([intensity, intensity, intensity], axis=-1)

    def checker_rgb(cell: int) -> np.ndarray:
        """Checkerboard with given cell size."""
        v, u = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        pat = ((u // cell + v // cell) & 1).astype(np.uint8) * 255
        return np.stack([pat, pat, pat], axis=-1)

    def disc_rgb(cx: float, cy: float, r: float) -> np.ndarray:
        """Filled disc on a black background."""
        v, u = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        mask = ((u - cx) ** 2 + (v - cy) ** 2 < r ** 2).astype(np.uint8) * 255
        return np.stack([mask, mask, mask], axis=-1)

    depth = np.full((H, W), 3.0, dtype=np.float32)
    t = np.zeros(3, dtype=np.float32)
    R = np.eye(3, dtype=np.float32)

    def report(i: int, accept: bool, sc: CokoSlamScore, label: str) -> None:
        flag = "ACCEPT" if accept else "skip  "
        extra = " (forced)" if sc.forced else ""
        reset = " [RESET]" if sc.submap_reset else ""
        print(
            f"frame {i:2d} {label:>20s}  d_min={sc.min_dist:6.3f}  "
            f"d_anchor={sc.submap_anchor_dist:6.3f}  "
            f"n_refs={sc.n_refs:2d}  sub{sc.submap_idx}  {flag}{extra}{reset}"
        )

    test_frames = [
        ("bootstrap",     stripe_rgb(0.0)),         # 0
        ("duplicate-of-0", stripe_rgb(0.0)),        # 1 -> skip
        ("stripes-90",    stripe_rgb(90.0)),        # 2 -> accept (orthogonal)
        ("stripes-45",    stripe_rgb(45.0)),        # 3 -> accept (diagonal)
        ("checker-16",    checker_rgb(16)),         # 4 -> accept; may RESET (>=3 KFs, big jump)
        ("checker-32",    checker_rgb(32)),         # 5
        ("disc-center",   disc_rgb(128, 128, 50)),  # 6
        ("disc-corner",   disc_rgb(50, 50, 30)),    # 7
        ("duplicate-of-0", stripe_rgb(0.0)),        # 8
        ("checker-16-dup", checker_rgb(16)),        # 9
    ]

    n_accept = 0
    for i, (label, rgb) in enumerate(test_frames):
        ok, sc = sel.should_accept(depth, t, R, rgb=rgb)
        n_accept += int(ok)
        report(i, ok, sc, label)

    print(f"\nTotal accepted: {n_accept}/10")

    # --- Distance-metric sanity: cosine vs L2 on the same stream -------
    print("\n--- distance_metric comparison (alpha tuned per metric) ---")
    for metric, alpha in (("cosine", 0.01), ("l2", 0.15)):
        cfg_m = CokoSlamConfig(
            alpha=alpha, submap_threshold=0.05, min_kfs_per_submap=3,
            max_kfs=0, distance_metric=metric, image_size=224,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        sel_m = CokoSlamSelector(cfg_m, K, (H, W))
        path = []
        for _, rgb in test_frames:
            ok, sc = sel_m.should_accept(depth, t, R, rgb=rgb)
            path.append(int(ok))
        print(f"  metric={metric:6s} alpha={alpha:.3f} -> "
              f"accepts {sum(path)}/{len(path)} ({''.join(str(x) for x in path)})")
