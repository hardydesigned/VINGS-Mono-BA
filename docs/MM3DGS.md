# MM3DGS-Selector

Keyframe-Auswahl nach **Sun L.C., Bhatt N.P. et al., „MM3DGS SLAM: Multi-modal
3D Gaussian Splatting for SLAM Using Vision, Depth, and Inertial Measurements",
IEEE/RSJ IROS 2024**, Sec. III.E. Nur der Keyframe-Selection-Teil ist
übernommen; IMU-Fusion (Sec. III.C), Gaussian-Initialization (Sec. III.F) und
das Mapping-Loss-Design gehören nicht zu VINGS-Mono.

## Algorithmus

Pro Tracker-KF-Kandidaten zwei Gates:

```
# Gate 1 — Covisibility
uv          = sample_grid(image)           # ~n_samples Pixel
P_world     = backproject(uv, depth, T_c)  # mit aktueller Pose
P_cam_prev  = R_prev.T @ (P_world - t_prev)
visible     = (z > 0) AND (uv_prev in [0,W]x[0,H])
covis       = #visible / #valid_world_points

# Gate 2 — NIQE-Min im Sliding-Window  (Proxy: max(Laplacian-Var))
lap_var     = var(Laplacian(gray(rgb)))
window.append(lap_var)
is_best     = lap_var == max(window)

accept iff  (covis < covis_thresh)  AND  is_best
```

Bei Accept wird `prev_kf` auf den aktuellen Frame gesetzt und das Window
geleert. Erster Frame: immer Accept.

**Intuition.** Gate 1 schiebt KFs nur dann an den Mapper, wenn das aktuelle
Bild „neues Terrain" zeigt (Overlap mit dem letzten KF < 95 %). Gate 2 wartet
in einem N-Frame-Fenster auf das schärfste Bild, anstatt das erstbeste mit
ausreichend Overlap zu nehmen — das filtert Motion-Blur-Frames raus.

## Variablen

| Symbol | Bedeutung |
|---|---|
| `prev_kf` | zuletzt akzeptierter Keyframe (Pose) |
| `window` | deque der letzten `niqe_window` Lap-Var-Werte |
| `covis` | Anteil der Punkte aus dem aktuellen Frame, die in `prev_kf` projizieren |
| `lap_var` | Variance-of-Laplacian = Schärfe-Proxy (hoch = scharf) |
| `covis_thresh` | Akzeptanz-Schwelle (Paper-Default 0.95) |
| `niqe_window` | Fenstergröße (Paper-Default 5) |
| `n_samples` | Anzahl gleichmäßig gewählter Bildpixel für die Backprojection |

## Adaptionen vs. Original (UT-MM RGB-D + IMU)

| Original | VINGS-Adaption | Grund |
|---|---|---|
| Depth-Rendering aus der Gaussian-Map an der aktuellen Pose | VINGS' Tracker-Depth (`viz_out['depths']`, DBA-Fusion / motion3d) | Konsistent zu VISTA/NURBS-LVI; die underoptimierte Map liefert früh keinen verlässlichen Depth-Render; spart einen zusätzlichen Splat-Render-Pass pro Frame |
| NIQE-Metrik (lernbasiertes No-Reference-Quality-Modell) | `cv2.Laplacian(gray, CV_64F).var()` | Deckt das eigentliche Ziel (Motion-Blur-Suppression) ab; keine neue Dependency; ms-Aufwand statt ~50-100 ms NIQE-Inferenz |
| Pose-Optimierung in 3DGS-Splat-Pipeline integriert | Selector sieht nur fertige Tracker-Pose | Plugin-Architektur in VINGS gibt dem Selector keinen Map-Zugriff |

Der Kern-Entscheidungsmechanismus (`covis < 0.95` AND `is_window_best`) ist
verbatim aus dem Paper, Sec. III.E.

## Sensitivität

### `covis_thresh` (wichtigster Knopf)

| Wert | Effekt |
|---|---|
| 0.95 (Default) | Paper. Akzeptiert KF sobald 5 % „neues Bild" sichtbar ist. Hohe Recall, hohe Mapping-Last. |
| 0.90 | Wartet bis 10 % Neuheit. Weniger KFs, weniger Mapping-Last. Trade-off mit Drift. |
| 0.85 | Aggressiv. KFs sehr selten, deutlich entlastetes Mapping. Bei schmalen Trajektorien (`smallcity_200`) noch ok; bei langen Pfaden Drift-Risiko. |

### `niqe_window`

| Wert | Effekt |
|---|---|
| 3 | Schnellere Reaktion auf neue Geometrie, schwächere Blur-Filterung. |
| 5 (Default) | Paper. Im VINGS-Frame-Rate-Regime (30 fps) ~167 ms Latenz, vertretbar. |
| 7–10 | Stärkere Blur-Filterung, längere Latenz bis zum ersten Mapping nach Re-Entry in unbekannte Region. |

### `n_samples`

Beeinflusst nur die Genauigkeit der Covisibility-Schätzung, nicht die
Entscheidung. 1024–4096 reichen; 2048 ist Default. Cost ist linear, bleibt im
sub-ms-Bereich (reine NumPy-Operation).

## Tuning-Workflow

1. **Default laufen lassen** (`covis_thresh: 0.95`, `niqe_window: 5`) auf der
   gewünschten Sequenz.
2. **PhaseTimer-Summary** prüfen: `frame_select` sollte < 5 ms/Frame liegen
   (Laplacian-Var + Backprojection sind beide CPU-Operationen auf
   Pixel-Subsamples).
3. **KF-Rate** ablesen: Anteil akzeptierter KFs vs. Tracker-KFs. Erwartung
   bei smallcity_200: ~15–25 %.
4. **Vergleich zum Mapping-Budget** aus `KEYFRAME.md`: liegt
   `accepted_KFs × map.train_loop ≈ 1150 ms` unter dem gewünschten
   Wandzeit-Budget, ist der Tuning-Punkt erreicht.
5. **Wenn zu wenig KFs**: `covis_thresh` Richtung 0.97 schieben oder
   `niqe_window` reduzieren.
6. **Wenn zu viele Blur-KFs durchrutschen**: `niqe_window` erhöhen.

## Code-Pointer

| Datei | Inhalt |
|---|---|
| `scripts/vings_utils/mm3dgs_selector.py` | Selector + Config + Score + Smoketest |
| `scripts/vings_utils/selector_factory.py` | Registrierung als `kind: mm3dgs` |
| `scripts/vings_utils/nurbs_lvi_selector.py:142` | wiederverwendete `backproject()` |
| `scripts/run.py:258–271` | unveränderter Call-Site |
| `configs/local/smallcity/mm3dgs/` | smallcity-Beispiel-YAMLs |
| `configs/local/ntu_eee_03_500/mm3dgs/` | NTU-VIRAL-Beispiel-YAMLs |

Standalone-Smoketest:

```bash
PYTHONPATH=scripts python scripts/vings_utils/mm3dgs_selector.py
```

Erwartete Ausgabe: erster Frame Accept, stationäre Frames skippen (covis≈1.0),
nach lateraler Translation drückt covis < 0.95 und der schärfste Frame im
Window wird akzeptiert.

## Was im BA-Methodenkapitel stehen sollte

Die Implementierung weicht in zwei dokumentierten Punkten vom Paper ab:

1. **Depth-Quelle**: VINGS-Tracker-Depth statt Map-Rendering. Vorteil:
   konsistent zu VISTA/NURBS-LVI, robust bei underoptimierter Map. Nachteil:
   die Covisibility-Schätzung folgt der Tracker-Tiefe; bei systematischen
   Tracker-Depth-Fehlern (z.B. Skywall) verschiebt sich der effektive
   Akzeptanz-Threshold.
2. **NIQE → Variance-of-Laplacian**: pragmatischer Standard-Proxy aus der
   OpenCV-Praxis. Misst hochfrequenten Image-Gradient, fällt steil bei
   Motion-Blur. Empirisch in Robotik-Pipelines verbreitet. Nachteil:
   reagiert ebenfalls auf Sensor-Noise und Textur — kann textur-arme aber
   scharfe Frames (homogene Wände) künstlich schlecht bewerten.

Beide Abweichungen sind in der `Adaptionen vs. Original`-Tabelle oben
festgehalten und im Code dokumentiert.
