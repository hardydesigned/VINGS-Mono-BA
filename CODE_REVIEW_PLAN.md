# Review-Plan: Thesis-Code VINGS-Mono-BA

## Kontext

3 Wochen vor Abgabe der Bachelorarbeit soll der gesamte Thesis-Code reviewt werden.
Ziel ist **vierfach** (alles davon): den Code (a) vollständig **verstehen & verteidigen**
können, (b) auf **Korrektheit/Bugs** prüfen, (c) **Paper-Treue** verifizieren und
(d) **vor Abgabe aufräumen** (toter Code, Inkonsistenzen).

**Was ist Thesis-Beitrag vs. Upstream?** Dieser Fork startet mit ~157 Dateien aus dem
originalen VINGS-Mono (Commit `5bf4d37`, „First Commit"). Die Thesis-Arbeit sind die
**75 geänderten/neuen `.py`-Dateien** danach. Wichtig: Die Git-Commits sind grobkörnig
(„full run", „Divererses") — ein Review pro Commit ist unsauber. Deshalb ist dieser Plan
**nach Subsystem/Beitrag** organisiert, in Lese-Reihenfolge von einfach → komplex.

**Thesis-Kern** (laut CLAUDE.md: *„Mapping-Last reduzieren, ohne Posenqualität aufzugeben"*)
ist die **Frame-Selection-Pipeline** mit 11 austauschbaren Selektoren (~7.300 LOC). Das ist
der Hauptbeitrag und bekommt das meiste Gewicht. Streaming/Object-Detection/Segmentation/
Depth-Models sind Nebenfeatures der letzten Commits und kommen zum Schluss.

## Wie reviewen (Methode pro Datei)

Für jede Datei in dieser Reihenfolge:
1. **Doc zuerst** (`docs/<NAME>.md` + CLAUDE.md-Tabellenzeile) → *woher kommt es* (welches
   Paper, welche Gleichung) und *wie soll es funktionieren* (Entscheidungsregel).
2. **`from_config` + `__init__`** → welche Hyperparameter, welche Defaults.
3. **`should_accept(...)`** → der Entscheidungspfad. Hier liegen die Bugs.
4. **Standalone-Smoketest** (`python scripts/vings_utils/<selector>.py`) → läuft es?
5. **Abgleich Code ↔ Paper-Formel** → stimmen Schwelle, Vorzeichen, Decision-Regel
   (AND vs OR), Force-Accept/Warm-up-Sonderfälle?

---

## Phase 0 — Orientierung (½ Tag)

Bevor irgendein Selektor: das Gerüst verstehen.

- **`CLAUDE.md`** (schon gelesen) — die Selektor-Tabelle + Pipeline-Diagramm sind die Landkarte.
- **`MAPPING_TRACKING.md`** — Tracker (Posen, jeder Frame) vs. Mapper (Gaussians, nur KFs).
  Begründet *warum* der Selektor zwischen Tracker-KF und Mapper sitzt.
- **`KEYFRAME.md`** — Profiling-Zahlen + Budget-Tabelle: das quantitative Argument der Arbeit.
- **`REPO.md`** — Repo-Struktur/Submodule.

## Phase 1 — Selektor-Gerüst / Integration (1 Tag) ★ kritisch

Das Skelett, durch das *alle* Selektoren laufen. Hier verstehst du die gemeinsame Schnittstelle.

- **`scripts/vings_utils/selector_factory.py`** (150 LOC) — Registry + `make_frame_selector`.
  *Wie es funktioniert:* `@register_selector("name")` füllt `_REGISTRY`; `kind` aus der Config
  wählt den Builder. Legacy-Fallback: `kind` fehlt + `enabled:true` → VISTA.
  **Befund/Check:** Docstring nennt die dokumentierte Schnittstelle
  `should_accept(depth, t, R, rgb=None)` — die **echte** Aufrufstelle in `run.py:794` übergibt
  aber zusätzlich `depth_cov=` und `meta=`. Prüfen, dass *jeder* Selektor diese kwargs
  akzeptiert (sonst `TypeError` zur Laufzeit) und Docstring angleichen.
- **`scripts/run.py`** (+1067 Zeilen — größter Churn) — die Orchestrierung. Gezielt lesen:
  - `make_frame_selector(cfg, ...)` Init (~Z.203) + K/image_hw-Auswahl.
  - Aufrufpunkt `~Z.772-799`: `frame_select`-Timer, Aufbau von `depth_np/t_np/R_np/rgb_np/cov_np/meta_b`,
    `should_accept(...)` → `do_map`. **Check:** Fallback-Pfad `do_map = mapper_kf_skip<=1 or (n_keyframes-1)%N==0`
    nur aktiv wenn Selektor `None`. Init-KF wird in beiden Pfaden gemappt — verifizieren.
  - **Gate A** (`~Z.264`): optionaler *Pre-Tracker*-Filter (≠ Selektor = Gate B). Verstehen,
    dass es zwei verschiedene Filter-Ebenen gibt.
- **`scripts/frontend/dbaf_frontend.py`** (+24), **`depth_video.py`** (+21), **`dbaf.py`** (+17)
  — die Tracker-seitigen Eingriffe (Stage 1+2 der Pipeline). Kleiner Churn, schnell zu prüfen:
  *was genau* wurde am Upstream-Tracker geändert und warum (Profiling-Hooks? viz_out-Batch?).

## Phase 2 — Die Selektoren einzeln (Kern, ~5–7 Tage) ★ Hauptbeitrag

Reihenfolge bewusst einfach → komplex. Jeder Selektor hat ein eigenes Doc unter `docs/`.
Pro Selektor: Doc → `from_config` → `should_accept` → Smoketest → Paper-Abgleich.

1. **`frame_selector.py` — VISTA** (345 LOC, `docs/` via CLAUDE.md). View-Angle-Diversity pro
   Voxel + Pose-Filter + Reservoir-Sampling. Braucht kein RGB → einfachster Einstieg.
2. **`orbslam3_selector.py`** (398 LOC, `docs/ORB_SLAM3.md`). Regel `(c1a ∨ c1b) ∧ c2`.
   **Check:** `ratio = matches/baseline_matches`, Baseline = erster Frame nach KF-Commit
   (Warm-up = immer skip). Historischer 3-OR-Bug ist laut Doku gefixt — verifizieren, dass
   wirklich `(c1a∨c1b)∧c2` im Code steht, nicht drei OR.
3. **`mm3dgs_selector.py`** (366 LOC, `docs/MM3DGS.md`). `covis<0.95 AND argmax(lap_var) im Window`.
   **Check:** NIQE→Variance-of-Laplacian-Proxy; `min_gap_after_kf` wirkt nur *mit* below_thresh
   (kein Force-Accept). Depth-Quelle = Tracker-Depth.
4. **`adaptive_kf_selector.py`** (395 LOC, `docs/ADAPTIVE_KF.md`). `θ=max(θ₀, μ+k·σ)` + Decay γ
   über Hybrid-Error (Photo+SSIM via Depth-Warping). **Check:** WarpFrame = Forward-Splat mit
   Z-Buffer aus `D_kf`; Defaults α=0.7 β=0.3 W=5 k=1.5 γ=0.95 stimmen mit Paper.
5. **`nurbs_lvi_selector.py`** (514 LOC, `docs/NURBS_LVI.md`). Regel `Or+Oc > Q` (Q ist Schwelle,
   nicht Score!). **Check:** `λ=1/|P₂|` (kein Hyperparam); Sektor-Migrations-Schwelle
   bei `sector_angle_deg/2` (Paper-Diskrepanz 15°/30° dokumentiert).
6. **`coko_slam_selector.py`** (559 LOC, `docs/COKO_SLAM.md`). DINOv2 + Cosine, zweistufig
   (Submap-Reset + In-Submap-KF). **Check:** `image_size` Multiple von 14; L2-normalisierte
   Embeddings; `distance_metric` cosine vs l2.
7. **`game_kfs_selector.py`** (721 LOC, `docs/GAME_KFS.md`). Composite `L=λ·A+(1−λ)·B`, EMA-λ.
   **Check:** v2-Formeln (Δflow 3-Frame Eq.11, PSNR-Warp Eq.7, Jaccard-IoU Eq.8, tanh Eq.12,
   Sigmoid literal Eq.13). Komplexester Score — sorgfältig.
8. **`aim_slam_selector.py`** (878 LOC, `docs/AIM_SLAM.md`). 3-Stage AND: Voxel-Overlap +
   EKF-Info-Gain + Reduced-Chi-Square auf Hybrid-Residual Eq.5. **Check:** ORB-Korrespondenz
   paper-treu vs Reprojection-Fallback; größte Datei → höchstes Bug-Risiko.
9. **`two_gate_selector.py`** (594) + **`two_gate_v2_selector.py`** (546) + Helfer
   **`gate_a.py`** (286) / **`gate_a_v2.py`** (394). Kombinierter Gate-A(Pre-Tracker)+Gate-B.
   **Befund:** in `selector_factory.py:101/107` registriert, aber **nicht** in der
   CLAUDE.md-Selektor-Tabelle → entscheiden: dokumentieren oder als experimentell entfernen.

> Priorisierung bei Zeitdruck: Selektoren, deren Ergebnisse tatsächlich in die Arbeit
> geschrieben werden, zuerst und gründlich. Nicht verwendete Selektoren mind. auf
> „läuft + Paper-treu" prüfen, da sie im Vergleichs-Benchmark der Arbeit auftauchen.

## Phase 3 — Eval-Pipeline (2 Tage) ★ Ergebnis-Validität

Diese Skripte erzeugen die **Zahlen in der Arbeit** — Bugs hier = falsche Ergebnisse.

- **`FAIR_EVAL.md`** lesen → *warum* train-view-PSNR unfair ist (das methodische Argument).
- **`scripts/eval/fair_eval.py`** — Sim(3)-ATE + Held-out-Novel-View-PSNR. Kernmetrik.
- **`scripts/eval/sim3_transform_ply.py` / `sim3_unwarp.py`** — Sim(3)-Alignment GT↔DROID.
  **Check:** korrekte Scale/Rotation/Translation, kein doppeltes Anwenden.
- **`measure_survey.py`, `merge_plys.py`, `clean_ply.py`, `render_ply_*.py`, `detilt_gps.py`,
  `gen_opt_cfg.py`** — überfliegen; gründlich nur, was Thesis-Zahlen liefert.
- **`scripts/analyze_profiling.py`** + `PhaseTimer` (`time.time` Phasen in `run.py`) — die
  Mapping-Last-Messung. **Check:** misst `frame_select` + `map.train_loop` korrekt; keine
  Doppelzählung von Sub-Phasen (Snapshot-Logik `~Z.802`).

## Phase 4 — Nebenfeatures (2–3 Tage)

Niedrigere Priorität; reviewen v.a. falls in der Arbeit erwähnt.

- **Object Detection** — `docs/OBJECT_DETECTION.md`; `detector_factory.py`, `detector_base.py`,
  `yolo_detector.py`, `rtdetr_detector.py`, `object_tracker.py`.
- **Segmentation** — `docs/SEGMENTATION_BACKEND.md`; `segmentation_factory.py`, `*_backend.py`
  (fastsam/sam2/sam3), `dynamic/dynamic_utils.py` (+155). Dynamic-Mask raus aus Loss.
- **Depth-Models** — `docs/DEPTH_MODELS.md`; `metric/depth_factory.py`, `metric_model.py`;
  `scale_align`-Pfad. Berührt `gaussian/loss_utils.py` (+6, `metric_cov`).
- **Streaming** — `docs/STREAMING.md`; `scripts/server/{stream_server,splat_encode}.py`,
  frozen/active-Delta. **Check:** non-blocking daemon-Thread blockiert den Mapper nicht.
- **`storage/storage_manage.py`** (+154) — Off-by-one/Shape-Fix (`docs/STORAGE_MANAGER_FIX.md`).
  **Check:** der Frame-~489-Crash ist wirklich behoben.

## Phase 5 — Aufräum- & Konsistenz-Checkliste (laufend, ½ Tag final)

Bereits identifizierte Punkte (während des Reviews ergänzen):

- [ ] **`two_gate`/`two_gate_v2`** registriert, aber nicht in CLAUDE.md-Tabelle → dokumentieren
      **oder** entfernen (Entscheidung treffen, nichts Halbgares abgeben).
- [ ] **`should_accept`-Signatur-Drift**: Factory-Docstring (`rgb=None`) vs. echter Call
      (`depth_cov`, `meta`). Schnittstelle vereinheitlichen + dokumentieren.
- [ ] **`scripts/deprecated/`** — gehört das in die Abgabe? README prüfen, ggf. raus.
- [ ] Konsistenz: akzeptieren **alle** Selektoren `depth_cov`/`meta` als kwargs ohne Crash?
- [ ] Auskommentierter Code / Debug-Prints / `force_accept_all`-Diagnose-Flags in
      Produktions-Defaults auf `false`.
- [ ] `docs/`-Stand ↔ Code-Stand pro Selektor (Hyperparameter-Defaults synchron?).

---

## Verifikation (begleitend, nicht erst am Ende)

1. **Smoketests pro Selektor** (CLAUDE.md listet sie):
   `python scripts/vings_utils/<selector>.py` für jeden der 11 — muss ohne Crash durchlaufen.
2. **`/code-review`-Skill** auf den Diff `5bf4d37..HEAD` laufen lassen (effort `high`) — findet
   Korrektheits-Bugs automatisiert; deckt sich gut mit Phase 2/3. Kann pro Subsystem statt
   auf einmal laufen, um fokussierte Findings zu bekommen.
3. **End-to-End-Lauf** pro verwendeter Config (`configs/local/<algo>/`, smallcity_200):
   ein Selektor-Lauf + ein `mapper_kf_skip`-Baseline-Lauf, dann `fair_eval` — bestätigt,
   dass die in der Arbeit berichtete Mapping-Last-Reduktion bei gleicher Posen-/PSNR-Qualität
   reproduzierbar ist.
4. **Paper-Abgleich-Notiz** je Selektor: 2–3 Sätze „Code-Zeile X = Paper-Eq. Y" festhalten —
   genau das brauchst du in der Verteidigung und es deckt Abweichungen auf.

## Zeitbudget (~3 Wochen)

| Woche | Inhalt |
|---|---|
| 1 | Phase 0+1 (Gerüst) + Phase 2 Selektoren 1–5 |
| 2 | Phase 2 Selektoren 6–9 + Phase 3 (Eval) |
| 3 | Phase 4 (Nebenfeatures) + Phase 5 (Cleanup) + Verifikation + Puffer |
