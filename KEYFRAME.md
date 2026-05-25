Wo am besten Keyframes setzen?

-> Erstmal schauen was am längsten dauert

Smoke Test ohne Score Manager;

  Profiling-Summary

  phase                       n   total[s]  mean[ms]   med   p95     %
  map.total                 104     119.30   1147.1  1194.3 1329.5  48.1%
  map.train_loop            100     114.48   1144.8  1152.9 1281.3  46.2%
  track.total               200      90.29    451.5   487.4  530.4  36.4%
  track.frontend_ba         200      88.74    443.7   480.9  523.7  35.8%
  metric                    200      22.20    111.0   106.2  115.9   9.0%
  save_ply                    1      10.50  10495.6 10495.6 10495.6  4.2%
  map.add_new_frame          99       4.75     48.0    48.3   56.6   1.9%
  track.motion_filter       200       1.54      7.7     6.2    7.2   0.6%
  judge_pkg                 200       1.22      6.1     6.0   13.7   0.5%

  Ergebnis

  map.train_loop ist der größte Brocken: 46% der Gesamtzeit, ~1.15s pro Keyframe.
  track.frontend_ba ist der zweite: 36%, ~0.45s pro Frame (auf ALLEN 200 Frames, nicht 
  nur KFs).

  Profiling Summary — Run mit Storage Manager an

  === Profiling Summary (104 KFs / 200 frames, wall=243.6s) ===
  phase                             n   total[s]   mean[ms]    med[ms]    p95[ms]
  %
  --------------------------------------------------------------------------------------
  map.total                       104     118.28     1137.3     1180.2     1327.4
  48.6%
  map.train_loop                  100     113.54     1135.4     1138.7     1304.1
  46.6%
  track.total                     200      90.28      451.4      487.2      531.6
  37.1%
  track.frontend_ba               200      88.72      443.6      481.1      524.7
  36.4%
  metric                          200      22.29      111.5      106.6      116.0
  9.2%
  save_ply                          1       7.86     7859.1     7859.1     7859.1
  3.2%
  map.add_new_frame                99       4.66       47.0       46.9       55.6
  1.9%
  track.motion_filter             200       1.54        7.7        6.2        7.2
  0.6%
  judge_pkg                       200       1.23        6.1        5.9       14.0
  0.5%
  storage                           6       0.02        3.2        0.9        3.6
  0.0%

  Storage Manager kostet praktisch nichts (0.02s gesamt, 6 Aufrufe). Der Rest ist quasi
  identisch zur Baseline ohne Storage (248s vs 244s — Rauschen).


  Idee: Ich nehme das Filtern von VINGS komplett raus da mapping und tracking bei denen trotzdem sehr lang ist. Nur Mapping weiter zu optimieren reicht da nicht, ich erstelle also beim dataset loading schon meinen eigenen filter.

  Jetzt die Tests:

Run auf 200 Frames mit Store Manager (Smoke config duplizieren)
  - Komplett ohne VINGS Filter
  - Komplett ohne VINGS Filter jedes n-te Frame (2,3,4,5,6,7,8,9,10)


● Budget für einen neuen Keyframe-Selektor

  Ausgangslage aus den Daten

  Pro Frame im Baseline-Run (skip=1, 200 Frames):

  ┌────────────────────────────────┬─────────────┬─────────┬──────────────────────┐
  │             Phase              │    Wann     │  Zeit   │        Status        │
  ├────────────────────────────────┼─────────────┼─────────┼──────────────────────┤
  │ metric (depth)                 │ jeder Frame │ 106 ms  │ fix                  │
  ├────────────────────────────────┼─────────────┼─────────┼──────────────────────┤
  │ track.total                    │ jeder Frame │ 502 ms  │ fix                  │
  ├────────────────────────────────┼─────────────┼─────────┼──────────────────────┤
  │ judge_pkg (aktueller Selektor) │ jeder Frame │ 6 ms    │ austauschbar ← deins │
  ├────────────────────────────────┼─────────────┼─────────┼──────────────────────┤
  │ map.total                      │ nur auf KFs │ 1194 ms │ nur falls KF         │
  └────────────────────────────────┴─────────────┴─────────┴──────────────────────┘

  Wandzeit pro Frame: ~614 ms ohne KF, ~1.8 s mit KF. Aktuell sind 192/200 = 96 % Keyframes.

  Zwei verschiedene Budgets — je nachdem was du willst

  A) "Genauso schnell wie vorher" (Selektor ersetzt nur judge_pkg)

  Wenn dein Algorithmus die KF-Rate nicht reduziert: er muss vergleichbar billig sein wie das aktuelle judge_pkg.
  → Budget ~6-10 ms pro Frame.

  B) "Smart genug, KFs zu reduzieren" (Break-even-Analyse)

  Der Selektor läuft auf jedem Frame, spart aber bei jeder vermiedenen KF ~1194 ms Mapping. Break-even:

  S_ms (pro Frame) < 1194 ms × (gesparte_KFs / total_Frames)

  Konkrete Beispiele aus deinen 200 Frames:

  ┌───────────────────────┬───────────────────────┬───────────────────────────────┐
  │     Reduktion KFs     │     Gesparte Zeit     │ Max-Budget Selektor pro Frame │
  ├───────────────────────┼───────────────────────┼───────────────────────────────┤
  │ 192 → 150 KFs (−22 %) │ 42 × 1194 ms = 50 s   │ 250 ms                        │
  ├───────────────────────┼───────────────────────┼───────────────────────────────┤
  │ 192 → 100 KFs (−48 %) │ 92 × 1194 ms = 110 s  │ 550 ms                        │
  ├───────────────────────┼───────────────────────┼───────────────────────────────┤
  │ 192 → 60 KFs (−69 %)  │ 132 × 1194 ms = 158 s │ 790 ms                        │
  ├───────────────────────┼───────────────────────┼───────────────────────────────┤
  │ 192 → 30 KFs (−84 %)  │ 162 × 1194 ms = 193 s │ 965 ms                        │
  └───────────────────────┴───────────────────────┴───────────────────────────────┘

  Meine konkrete Empfehlung als Zielzahl
  Konkrete Beispiele aus deinen 200 Frames:

  ┌───────────────────────┬───────────────────────┬───────────────────────────────┐
  │     Reduktion KFs     │     Gesparte Zeit     │ Max-Budget Selektor pro Frame │
  ├───────────────────────┼───────────────────────┼───────────────────────────────┤
  │ 192 → 150 KFs (−22 %) │ 42 × 1194 ms = 50 s   │ 250 ms                        │
  ├───────────────────────┼───────────────────────┼───────────────────────────────┤
  │ 192 → 100 KFs (−48 %) │ 92 × 1194 ms = 110 s  │ 550 ms                        │
  ├───────────────────────┼───────────────────────┼───────────────────────────────┤
  │ 192 → 60 KFs (−69 %)  │ 132 × 1194 ms = 158 s │ 790 ms                        │
  ├───────────────────────┼───────────────────────┼───────────────────────────────┤
  │ 192 → 30 KFs (−84 %)  │ 162 × 1194 ms = 193 s │ 965 ms                        │
  └───────────────────────┴───────────────────────┴───────────────────────────────┘

  Meine konkrete Empfehlung als Zielzahl

  ~50 ms pro Frame ist eine vernünftige Obergrenze, und zwar weil:

  1. Tracking ist eh ~500 ms → 50 ms sind +10 % Overhead pro Frame, kaum spürbar.
  2. Break-even erreicht ab ~9 gesparten KFs auf 200 Frames (4,5 %) → realistisches Minimum für jeden vernünftigen Selektor.
  3. Headroom für komplexere Heuristiken (z.B. View-Overlap, Feature-Coverage, kleine GPU-Kernels) — diese liegen typischerweise bei 10-50 ms wenn sie auf Tracking-Outputs
  aufsetzen.
  4. Real-time-kompatibel bleibt nichts ohnehin: 614 ms/Frame ist schon weit weg von 33 ms@30FPS. Du bist also nicht durch Echtzeit gebunden, sondern nur durch "schneller als das
  Mapping spart" → siehe Tabelle.

  Hartes Maximum

  ~1000 ms pro Frame. Darüber kostet der Selektor mehr Zeit als das gesamte Mapping einsparen könnte (selbst bei 0 KFs auf 200 Frames sparst du nur 1194 × 192/200 ≈ 1147 ms/Frame),
   und du verlierst Wandzeit netto.

  TL;DR: Ziel 50 ms, weich bis 250 ms, hart unter 1 s. Wenn dein Algorithmus GPU-Operationen nutzt und auf Tracking-Outputs (Korrelationsvolumen, Posen) aufsetzt, bist du
  typischerweise sowieso weit unter 50 ms.