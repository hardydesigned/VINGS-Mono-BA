01.07:
Änderungen im Code: Umstellen auf angle und coverage
Configs hatten max depth von 35m, jetzt 150m. Voxel size 0.1m -> 1.0m. Coverage threshold 0.15 -> 0.12. Angle threshold 0.06 -> 0.12.

Nach Gridsearch: Top 3 Selektoren:
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
