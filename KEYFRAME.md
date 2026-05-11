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
