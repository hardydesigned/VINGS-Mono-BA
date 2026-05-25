# ActiveSplat — warum es *kein* Keyframe-Selektor ist

Hinweis zu Li et al., **"ActiveSplat: High-Fidelity Scene Reconstruction
through Active Gaussian Splatting"**, IEEE RA-L 2025. Das Paper wird im Umfeld
von Gaussian-Splatting-SLAM oft als „hat einen KF-Selektor" zitiert — das stimmt
nicht. Diese Notiz dokumentiert, was das Paper wirklich macht, und welche
*Adaption* in VINGS-Mono überhaupt denkbar wäre.

## Was im Paper steht

Drei Gleichungen werden gerne missgedeutet. Was sie tun:

| Gl. | Symbol | Aufgabe im Paper | Trigger / Konsumiert von |
|---|---|---|---|
| Eq. 8 | `Mk` | „newly-observed areas" identifizieren | **Spawn neuer Gaussians** in unbeobachteten Regionen (Densifikation) |
| Eq. 9 | `Mh` | „high-loss samples" clustern | **DBSCAN-Yaw-Auswahl** wenn der Roboter am Knoten steht |
| Eq. 10 | `Si` | Voronoi-Knoten-Score | **Pfadplanung** (Dijkstra über das Voronoi-Graph) |

Formeln:

```
Mk = (Ô_k < 0.98) ∨ ((D_k < D̂_k) ∧ (|D_k − D̂_k| > 50 · ε_MDE))     (Eq. 8)
Mh = (Ô_k > 0.80) ∧ (D_k < D̂_k) ∧ (|D_k − D̂_k| > 0.3)               (Eq. 9)
Si = 20·s_o(i) + 10·s_c(i) + 10·s_u(i) + 10·s_h(i)                    (Eq. 10)
```

mit `Ô_k`, `D̂_k` = aus der Gaussian-Map gerendete Akkumulations-Opazität /
Tiefe am Frame *k*, `ε_MDE` = 50× Median der Tiefenabweichung.

## Variablen (Eq. 8/9)

| Symbol | Bedeutung |
|---|---|
| `D_k`   | beobachtete RGB-D-Tiefe am Frame *k* |
| `D̂_k`  | aus der Gaussian-Map gerendete Tiefe an derselben Pose |
| `Ô_k`  | akkumulierte Opazität (α-Blending) an derselben Pose |
| `τ_o1` | 0.98 — Schwelle für „nicht gemappt" |
| `τ_o2` | 0.80 — Schwelle für „gemappt aber prüfen" |
| `ε_1`  | 0.3 m — feste Tiefen-Toleranz für Mh |

## Warum das *kein* Keyframe-Selektor ist

ActiveSplat **erzeugt seinen Frame-Stream selbst** durch Roboter-Aktionen
(`MOVE_FORWARD 6.5 cm`, `TURN_LEFT 10°`, …, Sec. IV-A des Papers). Jeder
beobachtete Frame ist per Konstruktion an einer informativen Pose, weil das
System ihn dort *bestellt* hat. Die Frage „lohnt sich Mapping auf diesem Frame?"
stellt sich gar nicht.

Konsequenzen pro Gleichung:

- **Eq. 8 (Mk)**: Wird *jeden* Frame berechnet, um neue Gaussians zu
  initialisieren. Bei einem KF-Selektor würdest du Mk als Accept-Gate auswerten;
  im Paper steuert Mk hingegen `densify_and_add`, völlig unabhängig davon, ob
  der Frame als „KF" abgelegt wird.
- **Eq. 9 (Mh)**: Wird nur ausgewertet, *nachdem der Roboter an einem
  Voronoi-Knoten angekommen ist*. Mh-Cluster bestimmen die Roll-Reihenfolge
  der Yaw-Winkel beim Rundumblick (Sec. III-B3). Hat keinen Bezug zu „nehmen
  oder verwerfen".
- **Eq. 10 (Si)**: Score eines **Voronoi-Knotens**, nicht eines Frames. Steuert,
  zu welcher Position der Roboter fährt. Die `s_u`-/`s_h`-Boolflags
  (unvisited / in-horizon) sind topologische Knoteneigenschaften, keine
  Frame-Eigenschaften.

Zudem: das Wort *keyframe* erscheint im Paper genau zweimal, beide Male im
**Post-Processing-Abschnitt** (Sec. III-D4, Sec. IV-D3): „stored keyframe
data" für offline-Refinement und „50 frames … selected uniformly as the
train split" für den Test-Split der Ablation. Beides sind Speicher-/Eval-
Begriffe, keine Online-Selektionsregeln.

## VINGS-Mono ist das gegenteilige Problem

| Aspekt | ActiveSplat | VINGS-Mono |
|---|---|---|
| Frame-Quelle | Roboter wählt aktiv | passiver Videostream |
| Frage des Selektors | „wohin soll ich fahren/drehen?" | „lohnt sich Mapping dieses Frames?" |
| Output | Goal-Pose, Yaw-Liste | accept / reject |
| Voraussetzung | RGB-D-Sensor + steuerbarer Mover | nur RGB-Stream |
| Voronoi-Graph nötig | ja | nein (kein Pfad-Planungs-Bedarf) |

Ein 1:1-Port ist deshalb nicht möglich.

## Was *könnte* man ehrlich adaptieren

Wenn man die Coverage-Idee aus Eq. 8 in den passiven Slot umfunktioniert
(analog zu wie NURBS-LVI von LiDAR-VIO auf Mono adaptiert wurde):

1. Am Tracker-KF die aktuelle Gaussian-Map an der Kandidaten-Pose rendern
   (`mapper.render(w2c, intrinsic_dict)` liefert `accum` und `depth`).
2. Mk per Eq. 8 berechnen (ε_MDE = 50× Median von `|D_k − D̂_k|` auf
   `Ô_k > 0.98`-Pixeln).
3. Score = `mean(Mk)` über tiefen-valide Pixel.
4. Accept iff `score > coverage_thresh`.
5. Bootstrap: erste N Frames force-accept (analog NURBS `prev_kf`-Boot).

**Das wäre eine Adaption, keine Reproduktion.** Im Methodenkapitel müsste
explizit stehen:

- ActiveSplat hat keinen KF-Selektor; Eq. 8 wird im Original für Densifikation
  benutzt.
- Diese Adaption funktioniert Mk als Accept-Gate um, weil die zugrunde-
  liegende Größe („Bruchteil noch nicht gemappter Pixel") auch passiv
  informativ ist.
- Robotersteuerung, Voronoi-Graph, Eq. 9 und Eq. 10 sind nicht übernommen.

Pipeline-Voraussetzung in VINGS: der Selektor muss eine Referenz auf den
Mapper (`GaussianModel`) bekommen, um zu rendern. Aktuelle Factory-Signatur
`make_frame_selector(cfg, K, image_hw)` müsste um eine optionale
Mapper-Referenz erweitert werden (z.B. Setter `selector.set_mapper(mapper)`
nach Konstruktion in `scripts/run.py`).

## Budget-Einschätzung (falls implementiert)

`mapper.render(...)` liegt laut `gaussian_base.py:render_raw()` bei
~10–20 ms pro Pose. Das ist innerhalb des in `KEYFRAME.md` definierten
50-ms-Soft-Targets für den Selektor.

## Empfehlung

- **Nicht implementieren als „ActiveSplat-Selektor"**, weil im Paper keiner
  existiert und der Name irreführend wäre.
- Falls eine Render-basierte Coverage-Heuristik gewünscht ist: als eigener
  Selektor `coverage_render` registrieren und in der BA als „inspiriert von
  ActiveSplat Eq. 8" zitieren — nicht als Reproduktion.
- Diese Doku bleibt im Repo, damit die Recherche-Arbeit erhalten ist und das
  Methodenkapitel sauber darauf verweisen kann.

## Code-Pointer (für eine spätere Adaption, falls man sie baut)

| Datei | Wofür gebraucht |
|---|---|
| `scripts/gaussian/gaussian_base.py:render_raw()` | rendert `accum` (Ô_k) + `depth` (D̂_k) aus aktueller Karte |
| `scripts/gaussian/gaussian_model.py:add_new_frame()` | Vorlage, wie `render(w2c, intrinsic_dict)` aufgerufen wird |
| `scripts/vings_utils/selector_factory.py` | Registrierungspunkt für neuen Selektor |
| `scripts/run.py:~173` | Stelle, an der `make_frame_selector` aufgerufen wird; hier ggf. `set_mapper(self.mapper)` ergänzen |

## Was im Methodenkapitel der BA stehen sollte

Falls ActiveSplat im Lit-Review auftaucht:

1. ActiveSplat löst **active exploration**, nicht passive KF-Selektion.
2. Die im Paper publizierten Eq. 8/9/10 sind keine KF-Tests, sondern
   Densifikations-, Rotations- bzw. Knoten-Scoring-Funktionen.
3. Eine Übernahme von Eq. 8 als Coverage-Accept-Gate wäre eine *Re-Purpose-
   Adaption*, vergleichbar mit der NURBS-LVI-Adaption (siehe `NURBS_LVI.md`).
4. In dieser BA *nicht* übernommen, weil [Begründung — z.B. Voronoi-Graph
   nicht verfügbar, oder Mapper-Render-Coupling unerwünscht, oder Mk-Signal
   in Mono-Stereo zu rauschanfällig].
