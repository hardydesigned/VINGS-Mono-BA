# Faire Selektor-Evaluation (ATE + Held-out-PSNR)

## In einfachen Worten

Der „normale" PSNR in `metrics.json` ist **unfair**, wenn man Frame-Selektoren
vergleicht. Grund: Er wird nur auf den Frames berechnet, die der Selektor
*zufällig selbst zum Mappen ausgewählt* hat — und zwar genau in dem Moment, in
dem die Gaussians auf diesem Frame trainiert wurden (train-view). Ein Selektor,
der 3 Frames mappt, und einer, der 20 mappt, werden also auf **verschiedenen
Bildern, in verschiedener Anzahl, auf auswendig-gelernten Ansichten** bewertet.
Mehr/weniger gemappte Frames → andere Zahl, nicht vergleichbar.

Damit der Vergleich fair wird, braucht man Metriken, die **nicht davon abhängen,
welche Frames der Selektor gewählt hat**:

1. **Tracking-Qualität → ATE.** Wie nah ist die geschätzte Flugbahn an der
   GT-Bahn (DJI)? Hat nichts mit der Zahl gemappter Frames zu tun.
2. **Mapping-Qualität → Held-out-PSNR.** An *festen* Frame-Positionen (jede 10.,
   für *alle* Configs gleich) wird aus der fertigen Karte ein Bild gerendert und
   mit dem echten Bild verglichen. Gleiches Eval-Set, gleicher Maßstab.

So sieht man ehrlich: spart ein Selektor Rechenzeit, *ohne* dass Posen oder
Rekonstruktion schlechter werden?

## Technisch

Modul: `scripts/eval/fair_eval.py`, aufgerufen am Ende von `run.py` (nach dem
PLY-Save, Mapper noch GPU-resident) über das Config-Gate `fair_eval.enabled`.
Schreibt `fair_metrics.json` in den Run-Ordner; `log_sweep_row.py` zieht die
Werte in die Sweep-CSV (Spalten `ate_rmse_m, ate_mean_m, n_ate_pairs,
n_tracked, psnr_ho, ssim_ho, lpips_ho, n_eval_ho`).

### Koordinaten-Konventionen (verifiziert gegen `middleware_utils`)

* `video.poses[i]` ist eine **w2c**-tq `[tx,ty,tz,qx,qy,qz,qw]` (lietorch-SE3).
* `tq_to_matrix(tq) = SE3(tq).matrix()` = w2c-4×4; `c2w = inv(w2c)`.
* Der Mapper rendert mit `w2c = SE3(w2c_tq).matrix()`; das Karten-Weltsystem ist
  also `c2w = inv(SE3(video.poses[i]).matrix())`.
* `dji_poses_all_w2c.txt` ist dasselbe w2c-TUM-Format; GT für Slice-Frame `s` ist
  Zeile `start_frame + s` (`camstamp_all`-Zeile N == Bild `00000N.jpg`).

### ATE (Tracking)

Umeyama-Sim(3)-Alignment der geschätzten KF-Kamerazentren auf die GT-Zentren
(Skala inklusive — absorbiert die arbiträre Mono-SLAM-Skala). `ate_rmse_m` ist
die RMSE der ausgerichteten Translationen. Reine Mapping-Selektoren (vista,
mm3dgs, two_gate ohne A3, …) tracken identisch → **ATE konstant**; nur Configs,
die das *Tracking* ändern (z. B. `two_gate_v2_a3_*`, skip_no_filter), bewegen
die ATE. Hinweis: DJI-Posen haben ~10 % Skalenfehler ggü. RTK, der durch das
Sim(3)-Alignment (Skala) wegfällt — die Bahn**form** bleibt vergleichbar.

### Held-out-Novel-View-PSNR (Mapping)

Festes Eval-Set: jede `eval_stride`-te (Default 10) Slice-Position, für alle
Configs identisch. Pro Eval-Frame:

* Pose = **eigene geschätzte Trajektorie**, per SLERP+lerp im SLAM-Frame auf den
  Frame-Index interpoliert (nicht jeder Eval-Frame ist KF).
* Render aus der finalen Map an dieser Pose → Vergleich gegen das GT-Bild.

**Warum nicht direkt aus GT-Posen rendern?** Probiert — gibt schwarze Bilder.
Die Karte ist nur mit den *geschätzten* Kameraorientierungen konsistent; die
DJI/GT-Pose-Konvention unterscheidet sich (Positionen alignen unter Sim(3),
Orientierungen nicht). Render-aus-eigener-Pose auf festem Frame-Set ist die
Standard-SLAM-GS-Render-Metrik (vgl. MonoGS / SplaTAM): gleiches Frame-Set für
alle, Novel-View (interpoliert, nicht der Trainingsschritt), und Posendrift wird
korrekt bestraft (schlechteres Tracking → schlechtere Render-Pose → niedrigerer
PSNR). GT dient für ATE und als Vergleichsbild.

Die Side-by-side-Renders (`<run>/fair_eval/ho_<absidx>.png`, GT | Pred) liegen
zur Sichtkontrolle im Run-Ordner.

### Bekannte Grenzen

* Held-out-PSNR ist niedriger als der alte train-view-PSNR (z. B. ~14 statt ~23
  dB) — das ist der *ehrliche* Novel-View-Wert inkl. Drift, kein Bug.
* Interpolation bestraft sehr spärliches Tracking leicht zusätzlich (gröbere
  Pose an Nicht-KF-Frames) — gewollt, da spärliches Tracking real schlechtere
  Posen an beliebigen Frames bedeutet.
* GT = DJI (`local_position`-basiert, ~10 % Skala). Für metrisch exakte ATE
  müsste man RTK (`metadata/rtk.csv`, lat/lon→ENU) statt DJI nehmen.
