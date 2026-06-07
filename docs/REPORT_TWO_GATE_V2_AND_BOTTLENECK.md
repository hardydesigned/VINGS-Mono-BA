# Report: `two_gate_v2`-Selektor & Bottleneck-Analyse (Tracking vs. Mapping)

*Stand: 2026-06-05. Quellen: `scripts/vings_utils/two_gate_v2_selector.py`,
`scripts/vings_utils/gate_a_v2.py`, `scripts/run.py` (PhaseTimer),
`KEYFRAME.md`, `MAPPING_TRACKING.md`, `scripts/analyze_profiling.py`,
`output/{s1000_400f,s3100_200f,full_6199f}_results.csv`, `docs/TWO_GATE_RUNLOG.md`.*

---

## Teil 1 — Wie `two_gate_v2` funktioniert

### 1.1 Einordnung: zwei Gates an zwei verschiedenen Stellen

VINGS hat zwei Engpässe, an denen ein Frame aussortiert werden kann. `two_gate`
besetzt **beide**:

```
RGB-Frame
  → Gate A   (PRE-Tracker, gate_a_v2.py)   →  überspringt den ~450ms-Tracker
  → motion_filter + DepthVideo + dbaf-Frontend  (Tracker, erzeugt Tracker-KF)
  → Gate B   (POST-Tracker, two_gate_v2_selector.py)  →  do_map ja/nein
  → mapper.run(...)                          →  ~1150ms Mapping nur bei accept
```

- **Gate A** (`GateAV2`, config-Block `gate_a:`) läuft *vor* dem Tracker. Wenn A
  ablehnt, wird der teure Tracker-BA gar nicht erst gestartet. A hat drei
  Sub-Gates: A1 Altitude (AGL-Mindesthöhe), A2 Visual-Quality (Blur/Über-/
  Unterbelichtung/Gradient-Dichte), **A3 GPS-Motion** (min. ENU-Distanz, opt-in).
- **Gate B** ist der eigentliche `two_gate_v2`-Selektor (`frame_selector.kind:
  two_gate_v2`), läuft *nach* dem Tracker und entscheidet, ob der Mapper auf
  diesen Keyframe angesetzt wird. B hat drei Sub-Gates: B1 Motion, B2
  Covisibility, B3 DINO-Novelty.

Registrierung: `selector_factory.py` → `@register_selector("two_gate_v2")` →
`TwoGateV2Selector.from_config(cfg, K, image_hw)`. Schnittstelle wie alle
Selektoren: `should_accept(depth, t, R, rgb=None, ...) -> (accept, score)`.

### 1.2 Der einzige inhaltliche Unterschied v1 → v2

**v2 verlagert die GPS-Distanzprüfung aus Gate B1 nach Gate A3 (vor den
Tracker).** Motivation: Ein stationärer Frame (Drohne im Hover) soll *nicht erst
durch den 450ms-Tracker* laufen, nur um danach in B1 verworfen zu werden. In v2
filtert A3 die GPS-bewegungsarmen Frames bereits pre-Tracker weg; B1 ist dadurch
reine **Pose-Translation + SSIM-Veto**.


### 1.3 Ablauf von `should_accept` (Gate B), Schritt für Schritt

`should_accept` in `two_gate_v2_selector.py:214-358`. Reihenfolge:

**0. Bootstrap.** Erster Frame (`prev_kf_t is None`) → immer accept, committen,
`triggered_by="first"`.

**1. B1 — Motion + SSIM-Veto.**
- `pose_d_m = ‖t − prev_kf_t‖` (Translation aus der Tracker-BA, SLAM-Scale).
- Bei `b1_motion_source=="pose"` (Default): `motion_ok = pose_d_m ≥ pose_d_min_m`
  (0.15 m).
- Bei `"gps"`: `gps_d_m = ‖xyz_enu − prev_kf_xyz‖`, `motion_ok = gps_d_m ≥
  gps_d_min_m`; ohne GPS-Meta Fallback auf den pose-Pfad.
- **SSIM-Veto:** wenn motion_ok *und* `enable_ssim_veto`: berechne billiges
  Graustufen-SSIM (kurze Seite auf `ssim_resize=80` herunterskaliert) gegen das
  letzte KF-Bild. Ist `visual_ssim > visual_change_max_ssim` (0.98), wird
  `motion_ok=False` (`B1_ssim_veto`). → fängt den Fall ab, dass die gemeldete
  Bewegung nur Pose-Rauschen/Scale-Collapse war, das Bild aber faktisch
  identisch ist.
- Ergebnis: `score.b1_pass`.

**2. B2 — Covisibility.** Über einen *neutralisierten* `Mm3dgsSelector`-Helper
(`covis_thresh=1.01`, damit der Helper selbst nie entscheidet): backprojiziere
`n_samples_covis=2048` Pixel der prev-KF-Tiefe in den aktuellen Frame, miss den
Overlap-Anteil. `score.covis ∈ [0,1]`. `b2_pass = covis < covis_thresh` (0.85).
Daraus die Neuheit `b2_novelty = clip(1 − covis, 0, 1)`.

**3. B3 — DINO-Content-Novelty.** Über einen neutralisierten `CokoSlamSelector`-
Helper (`force_accept_all=True`): DINOv2-Small-Feature des Frames extrahieren,
min. L2-Distanz zu den letzten `dino_max_kfs=10` KF-Features bilden
(`d_min`). `b3_pass = d_min ≥ alpha` (0.35). `b3_novelty = min(d_min, 2)/2`.
Wenn B3 deaktiviert (`enable_b3=false`), fällt B3 aus und `b3_pass=True`.

**4. Composite-Score.** 
`composite = 0.5·b2_novelty + 0.5·b3_novelty` (mit B3), sonst `= b2_novelty`.

**5. Adaptive Threshold θ** (vor der Entscheidung berechnet). Score wird in einen
Ringpuffer der Länge `window_size=30` gelegt. Sobald voll:
`θ = max(theta0, mean + sensitivity·std)` über das Fenster. In der Warm-up-Phase
linear interpoliert zwischen `theta_init=0.35` und `theta0=0.30`. → der
Schwellwert „atmet" mit der jüngsten Szenendynamik mit.

**6. Entscheidung.** `accept = b1_pass AND (composite ≥ θ)`.
- `triggered_by`: `"motion+novelty"` / der B1-Ablehngrund / `"B2B3_novelty_below_theta"`.

**7. Failsafe.** Wenn nicht akzeptiert und `frames_since_kf ≥ force_after` (50):
force-accept (`triggered_by="force_after"`) — verhindert, dass der Mapper über
lange Strecken verhungert.

**8. Budget-Cap.**
- `frames_since_kf < min_spacing` → blockieren (`spacing_blocked`).
- akzeptierte Frames im `rate_window=30` ≥ `max_per_window=6` → blockieren
  (`rate_capped`).

**9. Commit (nur bei accept).** `prev_kf_t/R` (und bei gps-Modus `prev_kf_xyz`)
setzen, Mm3dgs-Helper-State syncen, DINO-Feature in die FIFO pushen,
`prev_kf_rgb_small` cachen, `frames_since_kf=0`, und **θ *= decay** (0.85) — nach
jeder Annahme wird die Schwelle gesenkt, sodass kurz danach leichter ein
weiterer KF akzeptiert wird (Bursts in echt neuen Regionen).

### 1.4 Kernformel kompakt

```
accept_B = b1_pass ∧ (composite ≥ θ)          mit θ = max(θ₀, μ_W + k·σ_W)
b1_pass  = (motion ≥ d_min) ∧ ¬(SSIM > 0.98)
composite= ½(1−covis) + ½·min(d_dino,2)/2
… danach: force_after-Override, dann min_spacing/max_per_window-Cap.
```

### 1.5 Implementierungstrick: Komposition statt Neu-Implementierung

B2 und B3 werden nicht neu geschrieben, sondern durch **Wiederverwendung
bestehender Selektoren** realisiert: ein intern gehaltener `Mm3dgsSelector`
liefert `_covisibility`, ein `CokoSlamSelector` liefert `_extract`/`_commit` für
DINO. Beide sind per Config so neutralisiert, dass sie selbst nie ablehnen —
`two_gate_v2` ruft nur ihre Helper auf und trifft die Entscheidung selbst.

### 1.6 Alle Config-Parameter (`TwoGateV2Config`)

| Gruppe | Param | Default | Bedeutung |
|---|---|---|---|
| B1 | `b1_motion_source` | `pose` | `pose`=Tracker-Translation; `gps`=ENU-Distanz |
| B1 | `pose_d_min_m` | 0.15 | Min-Translation (pose-Modus) |
| B1 | `gps_d_min_m` | 0.5 | Min-ENU-Distanz (gps-Modus) |
| B1 | `visual_change_max_ssim` | 0.98 | SSIM darüber → Veto |
| B1 | `ssim_resize` | 80 | Downsample für SSIM |
| B1 | `enable_ssim_veto` | True | SSIM-Veto an/aus |
| B2 | `covis_thresh` | 0.85 | accept wenn covis darunter |
| B2 | `n_samples_covis` | 2048 | Pixel für Backprojection |
| B2 | `min_depth`/`max_depth` | 0.2/60.0 | Tiefenfenster |
| B3 | `enable_b3` | True | DINO-Term an/aus |
| B3 | `alpha` | 0.35 | min L2-Distanz (b3_pass) |
| B3 | `dino_model`/`dino_image_size`/`dino_device` | vits14/224/cuda | DINOv2-Backend |
| B3 | `dino_max_kfs` | 10 | FIFO-Länge KF-Features |
| θ | `theta0` | 0.30 | Floor |
| θ | `theta_init` | 0.35 | Warm-up-Start |
| θ | `window_size` | 30 | Fenster W |
| θ | `sensitivity` | 0.5 | k in μ+k·σ |
| θ | `decay` | 0.85 | γ (θ *= γ bei accept) |
| Budget | `min_spacing` | 1 | harte Mindestframes |
| Budget | `max_per_window` | 6 | max accepts/Fenster (0=aus) |
| Budget | `rate_window` | 30 | Fensterlänge |
| Budget | `force_after` | 50 | Failsafe-Frames |
| Log | `verbose`/`log_skips_only` | False/False | per-call-Logzeile |

Gate A v2 (`gate_a:`-Block, `version: v2`): `enable_a3`(False)/`gps_d_min_m`(0.5)
für das Pre-Tracker-GPS-Gate, plus A1 (`min_altitude_m` 8.0) und A2
(`blur_thresh` 80, `overexp` 240, `underexp` 15, `grad_density_thresh` 0.03).

### 1.7 Configs, Doku, offene Punkte

- Configs: `configs/local/amtown03/{s1000_400f,s3100_200f}/two_gate_v2/` — je 7
  Varianten (`_loose`, `_strict`, `_b2only`, `_a3_loose`, `_a3_strict`,
  `_a3off`, Basis). Auto-generiert via `scripts/gen_{s1000_400f,s3100_200f}_configs.py`.
- Doku: `docs/TWO_GATE_RUNLOG.md` (Runlog + GPS-Gate-Sweep). **Toter Verweis:**
  Docstrings zeigen auf `docs/TWO_GATE.md`, die nicht existiert.
- Sweep-Erkenntnisse (`docs/TWO_GATE_RUNLOG.md`, GPS-Gate-Sweep):
  - **Winner `gps_d_min_m = 0.5`** (bestes psnr_ho bei niedriger ATE). 0.8/1.0
    driften (ATE ~26 m), 3.0/5.0 hungern die Map aus.
  - **`gps_d_min_m = 1.5` crasht reproduzierbar** im 2DGS-Rasterizer-Backward
    (`gaussian_base.py:382`, CUDA „invalid configuration argument").
  - fair-eval-ATE ist **run-to-run verrauscht** → Schlüssel-Configs 2–3× wdh.
  - Nur die `_a3*`-Varianten ändern das Tracking (A3 filtert pre-Tracker), alle
    anderen tracken identisch → ATE konstant (`docs/FAIR_EVAL.md`).
  - Run: `conda run -n vings python scripts/run_experiment.py <config>` (Config
    **positional**, nicht `--config`).

---

## Teil 2 — Bottleneck-Analyse: Tracking vs. Mapping

### 2.1 Kernergebnis

**Der Bottleneck ist das Mapping (`map.train_loop`), nicht das Tracking.**

Auf dem kanonischen Referenz-Run `smallcity_200` (skip=1):

| Phase | Anteil Wandzeit | Kosten | gemessen auf |
|---|---|---|---|
| `map.train_loop` | **~46 %** | ~1145 ms / **KF** | nur Keyframes |
| `track.frontend_ba` | ~36 % | ~444 ms / **Frame** | allen Frames |

Mapping ist der teuerste Einzelposten, läuft aber nur auf Keyframes — genau das
macht es zum richtigen Optimierungsziel: weniger/schlankere Mapper-Aufrufe sparen
am 46-%-Block, ohne dass am Tracking (Posenqualität) gerührt wird.

### 2.2 Wie gemessen wird (`PhaseTimer` in `scripts/run.py`)

Jede Phase ist mit `with self.timer.time(name):` umschlossen; vor und nach jeder
Messung `torch.cuda.synchronize()`, damit GPU-Async-Kernels korrekt erfasst
werden. Erfasste Phasen u.a.: `track.total`, `track.frontend_ba`,
`track.motion_filter`, `map.total`, `map.train_loop`, `map.add_new_frame`,
`frame_select`, `metric`, `judge_pkg`. `summary(total_wall)` sortiert nach
Gesamtzeit und druckt den %-Anteil an der Wandzeit. Dump nach `profiling.json`
ist atomar und überlebt OOM/SIGKILL (`snapshot_every_kf`).

Auswertung: `python scripts/analyze_profiling.py --output-root output` aggregiert
alle `profiling.json` und gibt pro Run/Phase n, min/med/mean/p95/max/total aus —
Tracking-Werte pro Frame, Mapping-Werte pro KF.

### 2.3 Volle Messung `smallcity_200`, skip=1 (`KEYFRAME.md`)

| Phase | n | total[s] | mean[ms] | p95[ms] | % Wandzeit |
|---|---|---|---|---|---|
| map.total | 104 | 119.30 | 1147.1 | 1329.5 | **48.1 %** |
| └ map.train_loop | 100 | 114.48 | 1144.8 | 1281.3 | **46.2 %** |
| track.total | 200 | 90.29 | 451.5 | 530.4 | 36.4 % |
| └ track.frontend_ba | 200 | 88.74 | 443.7 | 523.7 | 35.8 % |
| metric (depth) | 200 | 22.20 | 111.0 | 115.9 | 9.0 % |
| save_ply | 1 | 10.50 | 10495.6 | — | 4.2 % |
| map.add_new_frame | 99 | 4.75 | 48.0 | 56.6 | 1.9 % |
| track.motion_filter | 200 | 1.54 | 7.7 | 7.2 | 0.6 % |
| judge_pkg (alter Selektor) | 200 | 1.22 | 6.1 | 13.7 | 0.5 % |

Mit aktivem Storage-Manager praktisch unverändert (map 48.6 %, track 37.1 %);
der Storage-Manager kostet ~0.02 s gesamt.

### 2.4 Warum `frame_skip` der falsche Hebel ist (`MAPPING_TRACKING.md`)

frame_skip-Sweep (mean ms/Call):

| skip | KFs | track.mean | track.med | map.mean | map.med |
|---|---|---|---|---|---|
| 1 | 192 | 502.0 | 514.6 | 1194.2 | 1234.1 |
| 2 | 92 | 490.6 | 513.9 | 1095.5 | 1178.2 |
| 5 | 32 | 440.6 | 507.9 | 932.3 | 1046.0 |
| 10 | 12 | 369.6 | 435.2 | 692.1 | 990.3 |

Der **Median** der Tracking-Kosten bleibt über alle skip-Werte bei ~500–515 ms:
die Per-Frame-Kosten sinken kaum, der Speedup kommt nur aus „weniger Frames".
Gleichzeitig zerstört frame_skip die Posenqualität (Drift). Mapping bleibt pro KF
teuer (~1.2 s), weil `iters=50` fix ist — skip senkt nur die KF-Anzahl, nicht die
Kosten pro KF.

### 2.5 Die zwei wirksamen Hebel

1. **Weniger Keyframes an den Mapper geben** — `mapper_kf_skip: N` oder ein
   smarter FrameSelector (`two_gate_v2` & Co). Tracking bleibt dicht
   (Posenqualität erhalten), nur die Mapper-Aufrufe sinken → man spart am
   46-%-Block. In den Live-JSONs sichtbar: z.B. 592 Tracker-KFs aber nur 74
   Mapping-Calls bei `kfskip=8`, Wandzeit 226 s statt deutlich mehr.
2. **`training_args.iters` senken** — direkt die Kosten *pro KF*. Alles andere
   skaliert nur linear über die KF-Anzahl.

### 2.6 Budget für den Selektor (`KEYFRAME.md`)

Damit ein Selektor sich lohnt, muss seine Per-Frame-Kosten `S` kleiner sein als
die anteilig gesparte Mapping-Zeit: `S_ms < 1194 ms × (gesparte_KFs /
total_Frames)`.

| KF-Reduktion | gesparte Zeit | Selektor-Budget/Frame |
|---|---|---|
| 192→150 (−22 %) | 50 s | 250 ms |
| 192→100 (−48 %) | 110 s | 550 ms |
| 192→60 (−69 %) | 158 s | 790 ms |
| 192→30 (−84 %) | 193 s | 965 ms |

Zielvorgabe: ~50 ms, weich bis 250 ms, hart unter 1 s.

### 2.7 Gemessene Selektor-Kosten (`frame_select_mean_ms`, Sweep-CSVs)

| Selektor | ms/Frame | | Selektor | ms/Frame |
|---|---|---|---|---|
| mm3dgs | 1.17 | | orbslam3 | 6.05 |
| vista | 1.51 | | aim_slam | 8.00 |
| **two_gate_v2** | **5.32** | | adaptive_kf | 8.77 |
| coko_slam | 5.41 | | game_kfs | 11.61 |
| | | | nurbs_lvi | 24.28 |

**Alle Selektoren liegen weit unter dem 50-ms-Ziel** — der Selektor-Overhead
(`two_gate_v2`: ~5 ms) ist gegenüber ~1 s gespartem Mapping pro vermiedenem KF
vernachlässigbar. Der Ansatz „smart auswählen statt blind skippen" ist damit
quasi gratis und greift genau am bestätigten Bottleneck an.

### 2.8 Dataset-Abhängigkeit

Auf amtown03 (niedrigere Auflösung) ist Tracking billiger
(`track_frontend_ba` ~185–196 ms), Mapping aber noch dominanter (`map.total` bis
>2000 ms/KF dicht). Auf den aerial/interval1-Configs: Tracking flach ~200 ms,
`map.train_loop` 1.5–3.9 s/KF. Über alle Datasets gilt: **Tracking ist flach,
Mapping skaliert die Wandzeit.**

---

## Fazit

1. **`two_gate_v2`** ist ein zweistufiges Gate (Pre-Tracker A + Post-Tracker B).
   Gate B akzeptiert einen KF iff Bewegung vorhanden (B1, Pose + SSIM-Veto) UND
   der Composite aus Covisibility-Drop (B2) und DINO-Content-Novelty (B3) eine
   adaptiv mitatmende Schwelle θ überschreitet — abgesichert durch
   force_after-Failsafe und Budget-Cap. v2-Neuerung gegenüber v1: GPS-Motion
   wandert von B1 nach Gate A3, um stationäre Frames vor dem teuren Tracker zu
   verwerfen.
2. **Bottleneck = Mapping** (`map.train_loop`, ~46 % Wandzeit, ~1.15 s/KF).
   Tracking ist mit ~36 % der zweite Posten, läuft aber pro Frame und ist pro
   Frame kaum komprimierbar. Richtig optimiert wird durch (a) weniger KFs an den
   Mapper (smarter Selektor / `mapper_kf_skip`) und (b) weniger Mapper-Iterationen
   pro KF. `frame_skip` ist kontraproduktiv (zerstört Posen, senkt Per-Frame-
   Kosten nicht). Der Selektor-Overhead von `two_gate_v2` (~5 ms) ist gegenüber
   dem gesparten Mapping vernachlässigbar.
