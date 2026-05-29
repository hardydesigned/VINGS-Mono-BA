# VISTA-Selector (View-Angle-Diversity)

VINGS-spezifische Adaption der geometrischen Information-Gain-Metrik aus
**Nagami et al., „VISTA: Open-Vocabulary, Task-Relevant Robot Exploration",
IEEE RA-L Vol. 11 No. 3, March 2026.**

Wichtig vorweg: das Paper beschreibt ein komplettes Explorations-System
(semantische 3DGS-Map + GMM-Trajektoriensampling + MPC-Planner). **Hier wird
nur die geometrische Information-Gain-Komponente (Paper Eq. 1) verwendet —
als binäres Mapper-Skip-Gate pro Tracker-Keyframe.** Wer das System mit dem
Paper vergleicht, sollte das im Kopf behalten.

---

## Idee in einfach

Pro Voxel speichert der Selector eine Liste von Sichtrichtungen (von welchen
Kameraposen aus wurde dieser Voxel schon gesehen). Ein neuer Frame ist
„informativ", wenn seine Strahlen Voxel treffen, die bisher nur aus stark
abweichenden Richtungen gesehen wurden. Je ähnlicher die neue Richtung zu
einer bereits gespeicherten ist, desto niedriger der Gain — desto eher wird
der Frame als redundant verworfen.

---

## Pipeline (was tatsächlich passiert)

```
viz_out['depths'][-1], viz_out['poses'][-1]  (c2w)
   │
   ▼
Stage 1: Pose-Filter   (NICHT im Paper, VINGS-Addition)
   │  reject if ∃ KF mit ‖t − tᵢ‖ < trans_thresh AND ∠(R, Rᵢ) < rot_thresh
   ▼
Stage 2: Geometric Gain   (Paper Eq. 1)
   │  - n_rays_score Pixel mit valider Tiefe random samplen
   │  - jeden Pixel zurückprojizieren → Welt-Punkt p_w, Welt-Richtung d_x
   │  - voxel = floor(p_w / voxel_size)
   │  - für jeden Treffer:
   │      g_n = (1 − max_{d_v ∈ voxel} (d_v · d_x)) / 2     ∈ [0,1]
   │      neuer Voxel:  g_n = 1.0   (UNOBSERVED)
   │  - G_I = mean(g_n)
   │  reject if G_I < gain_thresh
   ▼
Accept → _integrate(): n_rays_integrate Strahlen via Reservoir-Sampling
                       in die getroffenen Voxel einfügen, KF-Pose merken.
```

---

## Formel (Paper Eq. 1, verbatim)

$$
G_I(x) \;=\; \frac{1}{N_r} \sum_{n=1}^{N_r} \frac{\min(-d_v^{\top} d_x^{n}) + 1}{2}
$$

Im Code (`frame_selector.py:_geometric_gain`):

```python
cos_max = float((stored_arr @ d_new).max())   # = max(d_v · d_x)
gain   = (-cos_max + 1.0) * 0.5               # = (1 − max)/2  ∈ [0,1]
```

Identitäten: `min(−a) = −max(a)`, also `(min(−d_v·d_x) + 1)/2 = (1 − max(d_v·d_x))/2`.

**Interpretation der Werte:**
- max-Dot ≈ +1 → Voxel bereits aus sehr ähnlicher Richtung gesehen → gain ≈ 0
- max-Dot ≈ 0 → orthogonal zur nächsten existierenden Richtung → gain ≈ 0.5
- max-Dot ≈ −1 → exakt entgegengesetzte Richtung gespeichert → gain ≈ 1

---

## Bewusste Abweichungen vom Paper

| Paper | Hier | Grund |
|---|---|---|
| Voxel-Traversal (Amanatides/Woo) entlang jedes Strahls bis zum ersten occupied-Voxel | Depth-Backprojection direkt in die Treffer-Zelle | Tracker liefert Per-Pixel-Tiefe; kein Strahl-Marching nötig |
| 3-State-Grid (free / occupied / unobserved) | Sparse Hash: nur „je beobachtet" vs. „leer" | Free-State wird hier nicht gebraucht (kein Frontier-Sampling) |
| Fester Voxel-Grid um Roboter | Sparse Hash, unbegrenztes Wachstum | SLAM-Roboter bewegt sich frei; Speicher ist OK (zehntausende Zellen) |
| `G_S` (Semantik) + `G(x̄) = Σ γ^{K−k}(c·G_I + G_S)` (Eq. 2) | Nur `G_I` pro Frame | Mapper-Skip ist *single-frame*; keine Trajektorie zu scoren, kein CLIP-Stack |
| Algorithmus 1 (GMM + Dijkstra + Trajektorien-Score) | Schwellwert `gain_thresh` auf `G_I` | Wir wählen nicht den besten Pfad, wir gaten Frames |
| Per-Voxel Liste unbegrenzt | Reservoir-Sampling (Vitter 1985), `max_views_per_voxel=16` | Bounded ⇒ Scoring O(N·cap). Reservoir statt FIFO, weil FIFO bei Loop-Revisits die *frühen* Richtungen verliert und so künstlich hohen Gain erzeugt |
| Quadrotor-Voxel `0.2×0.2×0.16` (anisotrop) | `0.10³` (isotrop) | Default für SLAM; bei Nadir-Aerial könnte anisotrop (großes XY, kleines Z) besser sein — nicht getestet |

## Zusätzliche Stages (NICHT im Paper)

**Stage 1: Pose-Filter** (`_pose_is_redundant`).
Reject, wenn *irgendein* existierender KF *beide* Bedingungen erfüllt:
`‖t − tᵢ‖ < trans_thresh_m` AND `∠(R, Rᵢ) < rot_thresh_deg`.

Begründung: bevor wir die teurere Gain-Berechnung machen, filtern wir trivial
nahe KFs raus. Hat nichts mit VISTA zu tun — reine Engineering-Optimierung.
Effekt: bei Hover-Sequenzen filtert Stage 1 viel weg, *bevor* Stage 2 läuft;
das ist gewollt, sollte aber bei der Auswertung mitgenannt werden („VISTA +
Pose-Gate", nicht „VISTA pur").

Iteration ist linear über alle KFs — bei 1000+ KFs wird das langsam. Bei
Bedarf k-d-Tree über `keyframes`.

---

## Limitierungen / Risiken

1. **Keine Okklusionsprüfung.** Der Voxel, gegen den gescort wird, ist der,
   in dem die neue Depth den Punkt platziert. Wenn die Tracker-Tiefe drift-
   oder skalierungsverzerrt ist (siehe `project_ext_poses_broken`), wird der
   *falsche* Voxel konsultiert. Bei Nadir-Aerial-Szenen mit grossen Reichweiten
   ist das ein realer Confound.

2. **Threshold ist nicht aus dem Paper.** `gain_thresh=0.30` ist
   self-tuned. Im Paper wird die *beste* Trajektorie gewählt, nicht eine über
   einer absoluten Schwelle. Sweeps über `gain_thresh ∈ {0.20, 0.30, 0.40}`
   liegen unter `configs/local/vista/`.

3. **Scoring-Varianz durch Subsampling.** `n_rays_score=256` ist eine kleine
   Stichprobe. RNG-Seed ist fix (`default_rng(0)`), aber Reihenfolge der
   Frames ändert den State und damit indirekt das Scoring.

4. **Reservoir-Cap = 16** quantisiert die Gain-Schätzung. Bei dichten Loops
   und stark variierender Sichtgeometrie könnte man die Cap erhöhen
   (Speicher: 16·3·4 B = 192 B pro Zelle, bei 100k Voxeln ≈ 19 MB — die Cap
   ist also nicht aus Speichergründen niedrig).

---

## Wo im Code

- `scripts/vings_utils/frame_selector.py` — `FrameSelector`, `FrameSelectorConfig`,
  Reservoir-Sampling, Stage-1-Pose-Gate.
- `scripts/vings_utils/selector_factory.py` — `kind: vista` → Instantiierung.
- `scripts/run.py:601` — Aufruf in der Mapper-Schleife (Phase `frame_select`).
- `configs/local/vista/` — Basis-YAML + Sweep-Varianten (g020, g040).
- `scripts/run_vista_experiments.sh` — Sweep-Skript analog
  `run_mapskip_experiments.sh`.

Smoke-Test: `python scripts/vings_utils/frame_selector.py` (synthetischer
Boxraum, 60 Yaw-Frames; nach Reservoir-Switch unverändert 26/60 + Replay 0/60).
