● Tracking vs Mapping in VINGS-Mono

  Beides sind die zwei Hälften eines SLAM-Systems (Simultaneous Localization And Mapping). Sie lösen unterschiedliche Probleme und laufen mit unterschiedlicher Frequenz.

  ---
  Tracking — "Wo bin ich?"

  Frage: Wo war die Kamera, als dieses Bild aufgenommen wurde?

  Was passiert:
  - Läuft auf jedem Frame (200 Frames → 200 Tracking-Calls)
  - Nimmt das aktuelle Bild + die Bilder davor
  - Schätzt mit Optical Flow (DROID-Net) die Korrespondenz zwischen Pixeln aus aufeinanderfolgenden Bildern
  - Löst ein Bundle Adjustment (BA): "welche Kameraposen + welche Tiefe machen alle Pixel-Korrespondenzen konsistent?"
  - Output: 6-DoF-Pose (Position + Rotation) für jeden Frame

  Im Code: track.frontend_ba (dbaf_frontend.py), läuft in self.tracker.track(...) (run.py:247)

  Was Tracking braucht, um gut zu sein:
  - Kleine Baselines zwischen aufeinanderfolgenden Frames (kleine Bewegung → Optical Flow findet Korrespondenzen leichter)
  - → genau deshalb killt frame_skip deine Pose-Qualität

  ---
  Mapping — "Wie sieht die Welt aus?"
  
  Frage: Wie sieht die 3D-Szene aus, durch die ich mich bewege?

  Was passiert:
  - Läuft nur auf Keyframes (ausgewählte Frames, ~100 von 200)
  - Bekommt Bild + die schon vom Tracking geschätzte Pose
  - Trainiert die 3D Gaussian Splats: optimiert Position, Farbe, Opacity, Skalierung jedes Gaussians, sodass das gerenderte Bild aus dieser Pose dem echten Bild entspricht
  - 50 Trainings-Iterationen pro KF (training_args.iters: 50)
  - Output: 3D-Punktwolke aus Gaussians, die die ganze Szene darstellt

  Im Code: map.train_loop (Gaussian-Splatting Training), läuft in self.mapper.run(viz_out, True) (run.py:270)

  Was Mapping braucht, um gut zu sein:
  - Verschiedene Blickwinkel auf dieselbe Szene (Multi-View-Constraints)
  - Korrekte Posen vom Tracking (sonst landen die Gaussians an der falschen 3D-Stelle)
  - Genug Trainings-Iterationen pro KF

  ---
  Das Zusammenspiel

  Frame N kommt rein
      │
      ▼
  [Tracking]  ── schätzt Pose für Frame N ──┐
      │                                     │
      ▼                                     │
  "Ist Frame N ein Keyframe?"               │
      │                                     │
      ├── Nein → nächster Frame             │
      │                                     │
      └── Ja  ─────────────────────────────►│
                                            ▼
                                      [Mapping]
                                      rendert Frame N aus Pose,
                                      vergleicht mit echtem Bild,
                                      aktualisiert Gaussians

  Wichtige Hierarchie: Mapping ist abhängig vom Tracking. Wenn Posen schlecht sind, kann das Mapping nichts retten — die Gaussians werden an die falschen 3D-Koordinaten gerendert.

  ---
  Zurück zu deinem Problem

  ┌──────────┬──────────────────────────────────────────────────────────────┬───────────────────────────────────────────────────┐
  │          │                  Was frame_skip damit macht                  │                    Wie schlimm                    │
  ├──────────┼──────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────┤
  │ Tracking │ Vergrößerte Baselines → Optical Flow versagt → Posen driften │ ⚠️  Sehr schlimm (Posen sind das Fundament)        │
  ├──────────┼──────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────┤
  │ Mapping  │ Weniger KFs → weniger Trainingsviews → Lücken in der Map     │ Schlimm, aber reparierbar (Posen sind ja noch da) │
  ├──────────┼──────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────┤
  │ Speed    │ track.frontend_ba: 36% — wird linear weniger                 │ map.train_loop: 46% — wird linear weniger         │
  └──────────┴──────────────────────────────────────────────────────────────┴───────────────────────────────────────────────────┘

  Deswegen mein Vorschlag: Tracking dicht lassen (akkurate Posen), Mapping seltener triggern oder mit weniger Iterationen pro KF — du sparst beim teureren Teil (46%) und behältst
  die Pose-Qualität.


  -> Man sieht auch deutlich in den Rekonstruktionen das wenn man Tracking verkürzt das Modell immmer kleiner wird -> Nicht mehr akkurat mit der Welt
  -> Zweites Problem: Meine 3D Weltrekonstruktion ist komplett abhängig von der FPS zahl


  ==============================================================================================================
Zusammenfassung (Mean pro Call, ms):
==============================================================================================================
Run                                                      skip   KFs  track.mean  track.med   map.mean    map.med
----------------------------------------------------------------------------------------------------------------
05-11-22-02-generic_vo-smallcity_200_nofilter-              1   192       502.0      514.6     1194.2     1234.1
05-11-22-08-generic_vo-smallcity_200_nofilter_skip2-        2    92       490.6      513.9     1095.5     1178.2
05-11-22-12-generic_vo-smallcity_200_nofilter_skip3-        3    59       466.1      507.3     1020.4     1100.8
05-11-22-14-generic_vo-smallcity_200_nofilter_skip4-        4    42       453.3      507.3      985.5     1075.0
05-11-22-15-generic_vo-smallcity_200_nofilter_skip5-        5    32       440.6      507.9      932.3     1046.0
05-11-22-16-generic_vo-smallcity_200_nofilter_skip6-        6    26       427.8      505.7      890.6     1034.5
05-11-22-17-generic_vo-smallcity_200_nofilter_skip7-        7    21       428.1      505.9      827.1      998.1
05-11-22-18-generic_vo-smallcity_200_nofilter_skip8-        8    17       396.0      500.2      786.4     1014.9
05-11-22-19-generic_vo-smallcity_200_nofilter_skip10-      10    12       369.6      435.2      692.1      990.3
05-11-22-19-generic_vo-smallcity_200_nofilter_skip9-        9    15       388.6      482.8      765.1     1015.6

  Tracking (pro getracktem Frame, n_processed)
  - Median bleibt stabil bei ~500-515 ms über fast alle skip-Werte → die Tracking-Kosten pro Frame sinken kaum, weil die Frontend-BA-Arbeit pro Frame gleich teuer ist.
  - Mean fällt leicht (502 → 370 ms), weil mit mehr Skip der Anteil der schnellen Init-Frames (motion_filter ohne BA, ~6 ms) relativ wächst.
  - Max stabil bei ~1.2-1.4 s (selten Window-BA-Spikes).
  - → Tracking-Speedup kommt fast komplett aus "weniger Frames", nicht aus "pro Frame schneller".

  Mapping (pro Keyframe, n_keyframes)
  - Median fällt leicht: 1234 → 990 ms (skip=1 → skip=10).
  - Train-Loop ist stabil bei ~960-1200 ms/KF — das sind die 50 Trainingsiterationen.
  - Map-Mean fällt stärker (1194 → 692 ms), aber das ist nur ein Artefakt: Mapping-Median enthält Null-Einträge wenn viz_out is None ist. Beim Train-Loop (n nur auf echten
  Trainingsiterationen) bleibt der Median bei ~960-1200 ms.
  
  Was das für dich bedeutet:
  1. Frame-Skip macht Tracking nicht schneller pro Frame — es macht es nur insgesamt schneller, weil weniger Frames laufen. Wenn du Genauigkeit verlierst, gewinnst du nur Wandzeit,
   nicht Effizienz.
  2. Mapping ist pro KF die teuerste Phase (~1.2 s) und bleibt teuer, weil iters=50 fix ist — Skip reduziert nur die KF-Anzahl, nicht die Kosten pro KF.
  3. Wenn du Mapping effizienter machen willst (Kosten pro KF senken), musst du an training_args.iters ran — alles andere skaliert nur linear über die Anzahl der KFs.