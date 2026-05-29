# NURBS-LVI-Selector

Adaptive Keyframe-Auswahl nach **Wu et al., "NURBS-Based Continuous LiDAR-
Visual-Inertial SLAM with Adaptive Keyframe Selection", IEEE/ASME TMECH 2026**,
Sec. III.A. Nur der Keyframe-Selection-Teil (Front-End) ist übernommen; das
NURBS-Backend gehört nicht zu VINGS-Mono.

## Algorithmus (Eq. 4 + 5 + 6)

```
Q = phi * (Mc*Or/Mr + Dc*Or/Dr)            # adaptiver Schwellwert
phi   = s + gamma - beta - alpha
gamma = (2 - N/2) * (1 - Mc/Mr)
beta  = Oc/Mc - Oc/Mr
alpha = Oc/Mc - 0.5

accept iff  Or + Oc > Q
```

**Wichtig**: Q ist die **Schwelle**, nicht der Score. Die Akzeptanzregel
vergleicht die Migration `Or + Oc` mit Q. Die publizierte C++-Test-Datei macht
`Q >= threshold_Q` — das ist nur ein Test-Harness, nicht der Algorithmus.

## Variablen

| Symbol | Bedeutung |
|---|---|
| `prev_kf` | zuletzt akzeptierter Keyframe |
| `reference` | direkt darauf folgender Frame (Anker für Hauptachse) |
| `current` | Kandidat, N Frames nach dem reference |
| `Mc` | getrackte Features im current (Matches mit prev_kf) |
| `Mr` | getrackte Features im reference (Matches mit prev_kf) |
| `Dc` | **gesamte** extrahierte Features im current (ORB-Keypoints) |
| `Dr` | gesamte extrahierte Features im reference |
| `Or` | Sektor-Migrationen zwischen prev_kf und reference |
| `Oc` | Sektor-Migrationen zwischen prev_kf und current |
| `N`  | Frames zwischen prev_kf und current |
| `s`  | `exp(-λ · symmetric_chamfer(P1, P2))` |

**Sektor**: drei Bins um eine Hauptachse pro Landmark. Hauptachse =
Richtung vom **reference-Kamerazentrum** zum 3D-Punkt. Center-Sektor = 1
(Winkel zur Hauptachse < sector_angle_deg / 2).

> **Achtung — Sektor-Design weicht vom Paper ab.** Das Paper hat *drei*
> gleich-breite 15°-Bins, also **zwei** Migrations-Schwellen (bei 15° und 30°).
> Die Implementierung hat **einen** Center-Bin der Breite `sector_angle_deg`
> und zwei „outside"-Bins die per `np.cross(a,b)[2]`-Vorzeichen split werden —
> also **eine** echte Migrations-Schwelle bei `sector_angle_deg/2`. Der
> z-Achsen-Cross-Product ist im 3D-Welt-Frame geometrisch willkürlich,
> kollabiert bei kleinen Baselines aber in der Praxis auf „beide views im
> selben outer bin", sodass das Drei-Sektor-Design effektiv ein binäres
> „inner-vs-outer"-Signal ist. Funktional ähnlich aber nicht paper-äquivalent.
>
> **Implikation fürs Tuning**: `sector_angle_deg = 15°` im Code entspricht
> *nicht* dem Paper-Default (Paper-Default hätte erste Migration bei 15°, Code
> bei 7.5°). Wer dem Paper möglichst nahekommen will, müsste `sector_angle_deg
> = 30°` setzen (erste Schwelle dann bei 15°). Für VINGS irrelevant weil die
> Inter-Frame-Parallaxe so klein ist, dass das Paper-Setup sowieso nie feuern
> würde — der gewählte `sector_angle_deg = 2°` ist eine **deliberate
> Re-Skalierung**, kein Paper-Mapping.

**Reference-Frame**: laut Paper Sec. III.A.3 ist der reference-Frame der
*direkte Nachfolger* von `prev_kf` und dient **ausschließlich** als Anker für
die Hauptachse — er wird ausdrücklich „**not selected as keyframes**"
beschrieben. Der Selector gibt ihn deshalb mit `accept=False` zurück (Mapper
sieht ihn nicht), behält ihn aber intern für die Sektor-Geometrie.

**Chamfer-λ**: Paper Eq. 3 definiert `λ = 1/|P₂|` (Inverse der Punktzahl im
current-Frame). Das ist **kein freier Hyperparameter** — die Implementierung
berechnet λ pro Frame aus der Zahl der gemeinsam-validen Matches.

## Adaptionen vs. Original (NTU-VIRAL LiDAR-VIO)

| Original | VINGS-Adaption |
|---|---|
| IMU-Propagation für initialen Pose-Guess | VINGS-Pose aus `viz_out['poses']` direkt |
| LiDAR-registrierte Tiefe via LOAM-Sphere-Fit | VINGS' dense Tiefe als Lookup |
| Feature-Tracking via LK über IMU-Prediction | ORB-Detektion + BFMatcher (cross-check) |
| sector_angle_deg = 15° (Paper-Default) | 1°-5° (Inter-Frame-Parallaxe in VINGS ist 0.5-2°) |

## Sensitivität: sector_angle_deg

Der **einzige Tuning-Knopf** den du wirklich brauchst. Kleiner = sensibler =
mehr Sektor-Migrationen = häufiger akzeptiert.

Im Paper-Default (15°) müsste die Inter-KF-Parallaxe > 7.5° sein für eine
Migration. Bei VINGS-Frames mit baseline ~5-10 cm und Tiefe ~5 m liegt sie
bei 0.5-1°. ⇒ Or strukturell 0 ⇒ Schwelle Q kollabiert auf 0 ⇒ jeder Frame
mit Oc > 0 wird akzeptiert. Praktisch unsinnig.

Bei 1-2° gibt es eine echte Variation in Or und Oc, und die Schwelle Q wird
modulierend wirksam.

## Tuning-Workflow

1. Run `nurbs_diag` (mit `force_accept_all: true`) — akzeptiert jeden Frame,
   loggt aber `mig=` und `Q=` pro Frame.
2. Schau auf die Verteilung. Du siehst typischerweise:
   - mig wächst monoton mit N (mehr Bewegung = mehr Migration)
   - Q ist mal hoch, mal niedrig (durch β-Suppression bewusst gedrückt wenn nötig)
3. Wähle `sector_angle_deg` so dass `mig > Q` für ~25-30% der Frames natürlich
   feuert.

## Code-Pointer

| Datei | Inhalt |
|---|---|
| `scripts/vings_utils/nurbs_lvi_selector.py` | Selector + Score-Funktion + Smoketest |
| `scripts/vings_utils/selector_factory.py` | Registrierung als `kind: nurbs_lvi` |
| `scripts/config_profiles.py` (`nurbs_lvi`) | Sweep-Definition für `gen_configs.py` |

## Was im Methodenkapitel der BA stehen sollte

Die Implementierung weicht in drei Punkten transparent von der Originalvorlage
ab und das gehört dokumentiert:

1. **IMU → VINGS-Pose**: keine IMU-Integration für initialen Pose-Guess
2. **LiDAR-Tiefe → VINGS-Tiefe**: dense Tiefenkarte statt sphere-fit auf
   LiDAR-Punktwolke
3. **sector_angle_deg = 2° statt 15°**: Inter-Frame-Baseline in VINGS ist
   eine Größenordnung kleiner als im NTU-VIRAL-LiDAR-VIO-Setup

Der Score selbst (Q-Berechnung, Entscheidungsregel `Or+Oc > Q`) ist verbatim
aus dem Paper.
