# MM3DGS-Selector

Keyframe-Auswahl nach **Sun L.C., Bhatt N.P. et al., „MM3DGS SLAM: Multi-modal
3D Gaussian Splatting for SLAM Using Vision, Depth, and Inertial Measurements",
IEEE/RSJ IROS 2024**, Sec. III.E + Referenz-Repo
[`VITA-Group/MM3DGS-SLAM`](https://github.com/VITA-Group/MM3DGS-SLAM)
(`slam/mapper.py:need_new_keyframe`, `add_keyframe`, `is_covisible`). Nur der
Keyframe-Selection-Teil ist übernommen; IMU-Fusion (Sec. III.C),
Gaussian-Initialization (Sec. III.F) und das Mapping-Loss-Design gehören nicht
zu VINGS-Mono.

## Algorithmus (Paper-Richtung)

Pro Tracker-KF-Kandidaten zwei Gates plus zwei Failsafes:

```
# Gate 1 — Covisibility (gegen den letzten KF)
# Reference: backprojected wird PREV_KF's Tiefe an PREV_KF's Pose,
# nicht die Tiefe des aktuellen Frames. Dann werden diese Welt-Punkte
# in den AKTUELLEN Frame projeziert.
uv          = sample_grid(image)                                   # ~n_samples Pixel
P_world     = backproject(uv, prev_kf_depth, T_prev_kf)            # aus PREV_KF
P_cam_curr  = R_curr.T @ (P_world - t_curr)                        # in CURRENT
visible     = (z > 0) AND (uv_curr in [0,W]x[0,H])
covis       = #visible / #valid_prev_kf_points
below       = covis < covis_thresh

# Gate 2 — NIQE-Min im Sliding-Window  (Proxy: max(Laplacian-Var))
# Reference pusht JEDEN Frame ins Fenster (kein Candidate-Filter).
lap_var     = var(Laplacian(gray(rgb)))
window.append(lap_var)
is_best     = lap_var >= max(window)         # current frame is the recent best

accept iff  below AND is_best
```

Plus Reference-faithful Min-Gap + ein Quality-Streak-Failsafe:

```
# Min-Gap (Paper: `kf_every`, TUM-Default 5)
# Reference (`mapper.py:170`): nach is_covisible()==False wird *zusätzlich*
# geprüft, ob seit dem letzten KF mindestens `kf_every` Frames vergangen
# sind — sonst KEIN Spawn. Kein unconditional force-accept.
gap_eligible    = frames_since_last_kf >= min_gap_after_kf
spawn_eligible  = below AND gap_eligible
accept          = spawn_eligible AND is_best

# Failsafe — Quality-Streak (VINGS-spezifisch, kein Reference-Pendant)
# Bei monoton blurrer werdender below-Phase würde is_best nie feuern;
# nach N spawn-eligible Frames in Folge → force-accept.
if force_accept_after > 0 AND stalled_count >= force_accept_after:
    accept = True
```

**Wichtige Eigenschaft (Reference-faithful):** solange `covis > covis_thresh`
bleibt — z.B. komplett statische Kamera — wird **niemals** ein KF gespawnt,
unabhängig von der Anzahl vergangener Frames. Das war im vorherigen Patch
(unconditional `max_gap_frames`) noch falsch.

Bei Accept wird `prev_kf_R`, `prev_kf_t` und `prev_kf_depth` auf den aktuellen
Frame gesetzt. Das Window wird **nicht** geleert — die deque rutscht natürlich
weiter (Paper-Verhalten: idx-basierte Eviction in monotonem Deque). Erster
Frame: immer Accept.

**Intuition.** Gate 1 schiebt KFs nur dann an den Mapper, wenn das aktuelle
Bild den letzten KF *nicht mehr ausreichend abdeckt* — gemessen daran, wie
viel von der letzten KF-Geometrie heute noch sichtbar ist. Gate 2 verlangt,
dass der aktuelle Frame mindestens so scharf ist wie alle Frames im jüngsten
N-Frame-Fenster — keine Motion-Blur-KFs. Failsafe B (zeit-basiert) verhindert,
dass das Quality-Gate den Mapper komplett aushungert.

## Variablen

| Symbol | Bedeutung |
|---|---|
| `prev_kf` | zuletzt akzeptierter Keyframe (Pose + Tiefenkarte zum Commit-Zeitpunkt) |
| `window` | deque der letzten `niqe_window` Lap-Var-Werte (jeder Frame, kein Filter) |
| `covis` | Anteil der Punkte aus `prev_kf`-Tiefe, die in den aktuellen Frame projizieren |
| `lap_var` | Variance-of-Laplacian = Schärfe-Proxy (hoch = scharf) |
| `covis_thresh` | Akzeptanz-Schwelle (Paper-Default 0.95) |
| `niqe_window` | Fenstergröße (Paper-Default 5) |
| `n_samples` | Anzahl gleichmäßig gewählter Bildpixel für die Backprojection |
| `min_gap_after_kf` | Min. Gap seit letztem KF, Paper `kf_every` (TUM-Default 5; 0 = aus). Wirkt **zusammen** mit below_thresh, **nicht** als unconditional Force-Accept. |
| `force_accept_after` | Quality-Streak-Failsafe (VINGS-spezifisch, kein Reference-Pendant; 0 = aus) |

## Adaptionen vs. Reference-Code (`VITA-Group/MM3DGS-SLAM`)

| Reference (`slam/mapper.py`) | VINGS-Adaption | Grund |
|---|---|---|
| **Depth-Quelle**: gerenderter Map-Depth an `keyframes[-1].pose`, silhouette-maskiert (`silhouette > 0.99`) | **Tracker-Depth** von VINGS' DBA-Fusion / motion3d, die zur Commit-Zeit von `prev_kf` gespeichert wurde | Konsistent zu VISTA/NURBS-LVI; die underoptimierte Map liefert früh keinen verlässlichen Depth-Render; spart einen zusätzlichen Splat-Render-Pass pro Frame |
| **Covisibility-Richtung**: backprojected `prev_kf`-Depth → projeziert in *current frame* → „what fraction of prev_kf's map points still visible now?" | **Selbe Richtung** (Paper-treu seit diesem Patch). Vorher fälschlich umgekehrt (current→prev_kf). | Direkter Reference-Match (`is_covisible` Z. 205-240) |
| **NIQE**-Metrik (lernbasiertes No-Reference-Quality-Modell) | `cv2.Laplacian(gray, CV_64F).var()` | Deckt das eigentliche Ziel (Motion-Blur-Suppression) ab; keine neue Dependency; ms-Aufwand statt ~50-100 ms NIQE-Inferenz |
| **Window-Population**: jeder Frame wird gepusht; intern monotone-min-Deque keyed by NIQE | **Window pusht jeden Frame** (matched); aber: nur Lap-Var-Float, keine monotone-Deque-Struktur — die Window-Best-Prüfung läuft per `lap_var >= max(window)` | Monotone-Deque ist ein O(1)-Trick für die `niqe_window[0]`-Lookup; ohne Retroactive Pick (s.u.) brauchen wir den Effizienz-Gewinn nicht |
| **Retroactive KF**: `add_keyframe` committed `niqe_window[0]` (den schärfsten Frame im Fenster), möglicherweise einen **älteren** Frame als den Trigger-Frame | **Nicht implementierbar** im VINGS-Interface: `should_accept` returnt pro Frame, `viz_out` ist ephemer. Wir akzeptieren nur den aktuellen Frame, falls er Window-Best ist | Pipeline-Constraint; `viz_out` müsste für N Frames gebuffert werden, das wäre eine Call-Site-Änderung in `run.py`. `force_accept_after` ist die Sicherheitsleine. |
| **`kf_every`** (TUM-Default 5): Min-Gap-Requirement *nach* below_thresh — `if not covisible AND gap >= kf_every: spawn` | Implementiert als `min_gap_after_kf` (Default 0 = off; VINGS-Sequenzen tunen Spacing meist via `frame_skip`) | Direkter Reference-Match (`need_new_keyframe` Z. 170). Wichtig: nicht unconditional — bei `covis > 0.95` wird trotz beliebig großer Gap **nicht** gespawnt. |
| **Schwellen-Test**: spawn iff `percent_inside > min_covisibility` ist *False*, d.h. `<= 0.95` (strict `>` für skip) | `covis < covis_thresh` (strict `<`) | Off-by-Epsilon am Rand; in floats praktisch identisch. |
| **Set-of-KFs** (Sec. III.G): Mapping läuft über covisibility-graph | Wir vergleichen nur gegen `keyframes[-1]` | Reference macht für die *Selection* (Sec. III.E) auch nur den Last-KF-Vergleich; der Graph ist ein Mapping-Detail außerhalb des Selector-Slots |
| **Pose-Optimierung** in 3DGS-Splat-Pipeline integriert | Selector sieht nur fertige Tracker-Pose | Plugin-Architektur in VINGS gibt dem Selector keinen Map-Zugriff |
| **Implizit**: silhouette-maskierter Map-Render ist nie leer | Bei Tracker-Depth komplett außerhalb `[min_depth, max_depth]` → `covis = 1.0` (skip) | Sky-only / Aerial-Frames mit kaputter Tracker-Tiefe sind kein Beweis für „neues Terrain". „No data" → „kann nicht entscheiden" → konservativ skippen. Defensive Guard ohne Reference-Pendant. |

Der Kern-Entscheidungsmechanismus (`covis < 0.95` AND `is_window_best`,
Push-Every-Frame, Time-Failsafe `kf_every`) ist nach diesem Patch
**reference-treu** modulo:

1. **Depth-Quelle** (Tracker statt Map-Render),
2. **NIQE → Lap-Var** (Proxy),
3. **Retroactive Pick fehlt** — wir akzeptieren nur, wenn der Trigger-Frame
   selbst window-best ist; der Reference-Code würde an dieser Stelle einen
   bis zu `niqe_window-1` Frames älteren Frame committen.

Punkt 3 ist die schwerwiegendste verbleibende Abweichung. Mitigation:
`force_accept_after` als Quality-Streak-Failsafe, falls das Window-Best-Gate
durch monoton blurrer werdende Sequenzen nie feuern würde.

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

1. **Default laufen lassen** (`covis_thresh: 0.95`, `niqe_window: 5`,
   `min_gap_after_kf: 5` wie Paper-TUM, `force_accept_after: 0`) auf der
   gewünschten Sequenz.
2. **PhaseTimer-Summary** prüfen: `frame_select` sollte < 5 ms/Frame liegen
   (Laplacian-Var + Backprojection sind beide CPU-Operationen auf
   Pixel-Subsamples).
3. **KF-Rate** ablesen: Anteil akzeptierter KFs vs. Tracker-KFs. Erwartung
   bei smallcity_200: ~15–25 %. `forced_reason` im Score-Log beobachten:
   wenn viele KFs via `force_accept_after` getriggert werden, ist das
   Quality-Gate dauerhaft blockiert — eventuell `niqe_window` reduzieren.
4. **Vergleich zum Mapping-Budget** aus `KEYFRAME.md`: liegt
   `accepted_KFs × map.train_loop ≈ 1150 ms` unter dem gewünschten
   Wandzeit-Budget, ist der Tuning-Punkt erreicht.
5. **Wenn zu wenig KFs**: `covis_thresh` Richtung 0.97 schieben,
   `niqe_window` reduzieren, oder `min_gap_after_kf` runter.
6. **Wenn zu viele Blur-KFs durchrutschen**: `niqe_window` erhöhen.
7. **Wenn Mapping-Hunger** (lange below-thresh-Phasen ohne KF):
   `force_accept_after` aktivieren (z.B. 10–20). Greift, wenn das
   Quality-Gate wegen monoton sinkender Schärfe nie window-best emittiert.

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

Die Implementierung wurde nach Cross-Check mit dem offiziellen Reference-Repo
(`VITA-Group/MM3DGS-SLAM`, `slam/mapper.py`) zweimal überarbeitet. Stand jetzt:

**Reference-treu:**

- Covisibility-Richtung: `prev_kf`-Depth → projeziert in aktuellen Frame
  (gleiche Direction wie `is_covisible` im Reference-Code).
- Window-Population: jeder Frame pusht, nicht nur Kandidaten.
- Window wird nicht geleert bei Accept (sliding deque).
- Schwellen-Logik (`covis < 0.95` AND `is_window_best`).
- `kf_every`-Min-Gap (als `min_gap_after_kf` exposed) — wirkt nur in
  Kombination mit below_thresh, kein unconditional Force-Accept.
- Single-KF-Vergleich gegen `keyframes[-1]` (auch im Reference so).

**Dokumentierte Abweichungen:**

1. **Depth-Quelle**: VINGS-Tracker-Depth statt gerendertem Map-Depth mit
   Silhouette-Maske. Vorteil: kein zusätzlicher Splat-Render-Pass, konsistent
   zu VISTA/NURBS-LVI, robust bei underoptimierter Map. Nachteil: die
   Covisibility-Schätzung folgt der Tracker-Tiefe; bei systematischen
   Depth-Fehlern (Skywall, kaputte Aerial-Tiefe) verschiebt sich die
   Schwelle. Defensive Guard: invalid-depth → `covis=1.0` → skip.
2. **NIQE → Variance-of-Laplacian**: pragmatischer OpenCV-Standard-Proxy.
   Misst hochfrequenten Image-Gradient, fällt steil bei Motion-Blur. ~50x
   schneller als NIQE-Inferenz. Nachteil: reagiert auf Sensor-Noise und
   Textur — kann textur-arme aber scharfe Frames künstlich abwerten.
3. **Keine retroaktive KF-Emission**: Reference committed `niqe_window[0]`
   (den schärfsten Frame im Fenster, möglicherweise einen älteren Frame).
   VINGS' per-frame `should_accept`-Interface erlaubt das nicht — `viz_out`
   ist ephemer, eine retroaktive Auswahl würde Call-Site-Buffer in `run.py`
   benötigen. Wir akzeptieren nur den aktuellen Frame, falls er gerade
   window-best ist. Konsequenz: bei monoton blurrer werdenden Sequenzen
   verlieren wir den scharfen Trigger-Frame. Mitigation: `force_accept_after`
   als Quality-Streak-Failsafe (kein Reference-Pendant).

Alle Abweichungen sind in der `Adaptionen vs. Reference-Code`-Tabelle oben
festgehalten und im Modul-Docstring von `mm3dgs_selector.py` aufgeführt.
