# Storage-Manager Bug-Fix

`scripts/storage/storage_manage.py` hatte einen Bug der bei langen VINGS-
Sequenzen (~Frame 489+) zu Tensor-Shape-Mismatch crashes führte. Gefixt Mai 2026.

## Symptom

Bei aktivem `use_storage_manager: true` + Sequenzen mit > ~500 Frames + häufigen
KF-Rejections crashed der Run mit:
```
RuntimeError: Trying to create tensor with negative dimension -4: [-4]
  at storage_manage.py:196 (torch.ones(new_added_size, ...))
```
oder später mit:
```
RuntimeError: The size of tensor a (93) must match the size of tensor b (89) at non-singleton dimension 0
  at storage_manage.py:166 (oncpu_kf_id_mask & near_kf_id_mask)
```

Beide treten genau dann auf wenn:
1. Der Tracker einen KF zurücknimmt (= `global_kf_id[-1]` schrumpft)
2. Storage-Manager-`run()` versucht `c2ws_storage_place` zu erweitern

## Root-Cause

Original-Code in `storage_manager.run()`:
```python
new_added_size = viz_out['global_kf_id'][-1] - self.c2ws_storage_place.shape[0]
self.c2ws_storage_place = torch.concat(
    (self.c2ws_storage_place,
     torch.ones(new_added_size, dtype=torch.float32)), dim=0)
```

Bug: `new_added_size` kann negativ werden wenn `global_kf_id[-1]` schrumpft (z.B.
nach einer Pose-refinement-Rollback-Operation). Dann ruft `torch.ones(-3, ...)`
crashed.

Zusätzlich in `cpu2gpu()`:
```python
near_kf_id_mask  = distance_to_cur_c2w < threshold   # shape N (=global_kf count)
oncpu_kf_id_mask = (self.c2ws_storage_place == 0)    # shape M
convey_kf_id = torch.arange(M)[oncpu_kf_id_mask & near_kf_id_mask]
```
Wenn M != N → bitwise-AND crashed mit shape-mismatch.

## Fix

In `scripts/storage/storage_manage.py:198-208`:
```python
# Grow c2ws_storage_place wenn nötig — niemals shrinken.
target_size = int(viz_out['global_kf_id'][-1])
if target_size > self.c2ws_storage_place.shape[0]:
    new_added_size = target_size - self.c2ws_storage_place.shape[0]
    self.c2ws_storage_place = torch.concat(
        (self.c2ws_storage_place,
         torch.ones(new_added_size, dtype=torch.float32)), dim=0)
```

Und in beiden Stellen wo `c2ws_storage_place` mit `distance_to_cur_c2w`
kombiniert wird (`cpu2gpu()` line 165-168 und `gpu2cpu()` line 124-129):
```python
n_kfs = near_kf_id_mask.shape[0]  # bzw distance_to_cur_c2w.shape[0]
oncpu_kf_id_mask = (self.c2ws_storage_place[:n_kfs] == 0)
```

Trim auf passende Größe — wenn `storage_place` aufgrund KF-Rejection größer
geworden ist als die aktuelle KF-Liste, werden die überzähligen Slots ignoriert.

## Validierung

Vor dem Fix: MARS 1500-Frame-Run crashed reproduzierbar bei Frame 489.
Nach dem Fix: läuft bis VRAM-Watchdog bei ~1300 Frames durch (n_frames=147
mapped). Crash kommt jetzt vom VRAM, nicht mehr vom Bug.

## Verbleibendes Limit

Storage-Manager ist FUNKTIONAL gefixt, aber **VRAM-Wand** bei ~150 mapped
Frames mit native intrinsic 2448×2048 bleibt — Wand kommt vom kumulierten
Gaussian-Count, nicht vom Storage-Bug. Aggressivere `distance_threshold` ≤ 2.0
führt zu vielen Convey-Spitzen-Allokationen die wiederum VRAM-Spikes erzeugen.
Sweet spot: `distance_threshold: 3.0..5.0`, `mapper_kf_skip: 3..5`.
