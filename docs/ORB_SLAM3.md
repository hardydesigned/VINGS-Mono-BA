# ORB-SLAM3-Selector

Keyframe-Auswahl nach **Campos et al., „ORB-SLAM3: An Accurate Open-Source
Library for Visual, Visual-Inertial and Multi-Map SLAM", IEEE Transactions on
Robotics 2021**. Übernommen wird der **New-Keyframe-Decision-Teil** des
Tracking-Threads (Fig. 1, Box „New KeyFrame Decision"). ORB-SLAM3 erbt die
Heuristik im Wesentlichen aus ORB-SLAM/ORB-SLAM2; das Paper selbst nennt sie
in Tabelle VI als Operation `New KF dec` mit ~0.04–0.20 ms pro Frame.

## Worum es geht — die einfache Version

Stell dir vor, du läufst durch eine Stadt und machst ab und zu ein Foto, damit
du dich später noch erinnerst, wo du warst. Wann genau machst du ein neues
Foto?

* **Wenn du auf dem letzten Foto fast nichts mehr wiedererkennst** — du bist
  weit gelaufen, neues Terrain → neues Foto.
* **Wenn du gerade die Orientierung verlierst** (Tunnel, dunkel, schneller
  Schwenk) → *kein* neues Foto, weil das Foto selbst zu schlecht wäre. Erst
  warten bis das Tracking sich erholt.
* **Bei reiner Standkamera (gleicher Inhalt) wird ebenfalls kein Foto
  gemacht**, auch wenn schon eine Weile keins mehr aufgenommen wurde — das
  wäre Redundanz.

Genau das macht ORB-SLAM3. „Wiedererkennen" heißt: das System extrahiert pro
Bild kleine markante Punkte (sogenannte ORB-Features, das sind Ecken/Kanten)
und matcht sie gegen die Punkte des letzten Keyframes. Solange genug Punkte
wiedergefunden werden, läuft das Tracking weiter — kein neuer Keyframe nötig.
Sobald die Trefferquote unter eine Schwelle (Paper: 90 %) fällt **und** seit
dem letzten KF mindestens `min_frames` vergangen sind, gibt's einen neuen KF.

Das war's. Es ist eine **„Drop-below-Ratio"-Heuristik** mit einem
Spacing-Sicherheitsnetz drumherum. Wahrscheinlich die am häufigsten zitierte
Keyframe-Auswahl in der SLAM-Welt — fast jeder feature-basierte VO/SLAM
(VINS-Mono, OKVIS, Kimera) nutzt eine Variante davon. Der Charme: **ein
einziger Knopf** (`tracked_ratio_thresh`) deckt den Großteil der Tuning-
Wünsche ab.

## Vergleich zum Rest des VINGS-Selector-Zoos

* **VISTA** belohnt neue Blickwinkel auf 3D-Voxel (View-Diversity).
* **NURBS-LVI** schaut auf Sektor-Migrationen einzelner Landmarks (Parallaxe).
* **MM3DGS** misst Bildüberlapp zwischen Kamerakegeln (Covisibility) und
  filtert verwackelte Frames raus.
* **Game-KFS** balanciert mehrere Sub-Scores via Spieltheorie-Lambda.
* **Adaptive-KF** adaptiert eine Schwelle auf den photometrischen Hybrid-Error.
* **ORB-SLAM3** zählt einfach die wiedergefundenen Features. Direkt, robust,
  zwei Hyperparameter zählen wirklich (`tracked_ratio_thresh`, `max_frames`).

## Algorithmus (ORB-SLAM/2/3, „New Keyframe Decision")

Original-Pseudocode aus
[`src/Tracking.cc::NeedNewKeyFrame`](https://github.com/UZ-SLAMLab/ORB_SLAM3/blob/master/src/Tracking.cc)
(Z. 3064-3214, HEAD):

```cpp
nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs=3);
thRefRatio  = 0.9;                                  // mono
c1a = N >= MaxFrames;
c1b = (N >= MinFrames) && bLocalMappingIdle;
c1c = (mSensor != MONOCULAR) && (...);              // stereo-only
c2  = (mnMatchesInliers < nRefMatches * thRefRatio)
      && (mnMatchesInliers > 15);
c3  = (inertial timestamp gate);                    // IMU-only
c4  = (IMU_MONOCULAR tracking-weak gate);

accept iff ((c1a || c1b || c1c) && c2) || c3 || c4
```

Für VINGS-Mono (kein Stereo, kein IMU) reduziert sich das auf
`(c1a OR c1b) AND c2`. Pseudocode der VINGS-Implementierung:

```
# Voraussetzung: ORB(current) berechnet, Matches zu prev_kf gezählt
matches          = #ORB_matches(current, prev_kf)
baseline_matches = matches gemessen am ERSTEN Frame nach KF-Commit
                   (ein KF-Lebensdauer-konstanter Wert; nRefMatches-Analog)
N                = frames_since_kf

# Erster Frame nach jedem Commit: Baseline-Warm-Up.
#   Setze baseline_matches := matches, ratio := 1.0, c2 := False (zwingend).
#   Damit kommt direkt nach einem KF nie sofort der nächste KF.
if first_frame_after_commit:
    baseline_matches = matches
    ratio = 1.0
    c2 = False
else:
    ratio = matches / baseline_matches

# Spacing-Seite (Paper c1a OR c1b; c1c stereo-only, entfällt)
c1a  = N >= max_frames                  # paper "MaxFrames"
c1b  = N >= min_frames                  # paper "MinFrames"
                                        #   (mapper-idle-Pfad fällt weg,
                                        #    siehe Adaptionen unten)
spacing_ok = c1a OR c1b

# Novelty-Seite (Paper c2)
c2 = (ratio < tracked_ratio_thresh)     # default 0.9
     AND (matches >= min_tracked)       # paper-precondition "> 15"

accept iff spacing_ok AND c2
```

Bei Accept wird `prev_kf` auf den aktuellen Frame gesetzt, sein ORB-Keypoint-
Set + Deskriptoren gecached, `baseline_matches` zurückgesetzt (wird beim
nächsten Frame neu gemessen) und `frames_since_kf = 0`. Der erste Frame
wird immer akzeptiert (Bootstrap).

### Intuition pro Bedingung

* **c2 (Novelty)** ist der eigentliche Trigger: solange genug ORB-Features
  des letzten KF noch im Bild sichtbar sind, ist die Map-Information „frisch
  genug" — kein neuer KF nötig. Sobald >10 % der Features rausgewandert sind
  *und* noch genug Inliers da sind, gibt's neuen Content → KF.
* **c1a (Force-Rate)** ist das Spacing-Failsafe für reine Bewegung: nach
  `max_frames` Frames darf ein KF auch ohne `mapper_idle` ausgelöst werden.
  Im Originalpaper ist während der IMU-Init explizit von 4 Hz Force-Rate die
  Rede (Sec. V.B). **Wichtig: c1a allein reicht nicht — Novelty muss ebenfalls
  zustimmen.** Bei reiner Standkamera erzeugt das System keinen KF.
* **c1b (Min-Spacing)** verhindert KF-Bursts direkt hintereinander. Bei
  `min_frames=1` heißt das nur: der Frame direkt nach einem KF kann nicht
  selbst sofort KF werden.
* **`matches >= min_tracked`** ist im Paper Precondition für c2: ein
  Ratio-Drop bei nur 5 Matches wäre kein meaningful Novelty-Signal, sondern
  Tracking-Verlust. Solche Frames werden bewusst *nicht* zum KF erklärt
  (würden die Map kontaminieren); stattdessen wartet das System auf
  Tracking-Erholung.

## Variablen

| Symbol | Bedeutung |
|---|---|
| `prev_kf` | zuletzt akzeptierter Keyframe (ORB-Keypoints + Deskriptoren + baseline) |
| `matches` | Anzahl ORB-Cross-Check-Matches zwischen `current` und `prev_kf` |
| `n_kp_curr` | Gesamtzahl ORB-Keypoints im aktuellen Frame (Diagnose) |
| `n_kp_prev` | Gesamtzahl ORB-Keypoints im `prev_kf` (Diagnose) |
| `baseline_matches` | nRefMatches-Analog: BFMatch-Count am ersten Frame nach Commit, danach für die KF-Lebensdauer fix |
| `ratio` | `matches / baseline_matches`, **fällt** mit Neuheit. Direkt am Start ~1.0 |
| `N` | Frames seit dem letzten akzeptierten KF (Spacing-Counter) |
| `min_frames` | Untere Spacing-Grenze (c1b) |
| `max_frames` | Obere Spacing-Grenze (c1a) |
| `min_tracked` | Mindest-Inliers für c2 (paper-precondition, paper-default 15, hier 50) |
| `tracked_ratio_thresh` | Akzeptanz-Schwelle für Ratio (Paper-Default 0.9) |

## So implementiert man das (Schritt für Schritt)

1. **ORB-Detector + Cross-Check-BFMatcher initialisieren**
   ```python
   self.orb     = cv2.ORB_create(nfeatures=cfg.orb_n_features)
   self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
   ```
   `crossCheck=True` filtert asymmetrische Matches raus, gleicher Schritt wie
   in `nurbs_lvi_selector.py`.

2. **Erster Frame** (`prev_kf is None`): ORB extrahieren, Deskriptoren cachen,
   Accept zurückgeben (Bootstrap).

3. **Pathological Frame** (`desc is None or len(kps) == 0`): keine ORB-Features
   extrahierbar (extreme Blur, dunkel, featureless). Reject; `prev_kf` bleibt
   unverändert. Im Diagnose-Modus (`force_accept_all=true`) wird trotzdem
   akzeptiert, damit der Mapper nicht hängt.

4. **Folge-Frame**: ORB extrahieren, `matcher.match(desc_curr, desc_prev)`
   aufrufen, `n_matches = len(matches)` zählen.

5. **Baseline-Warm-Up oder Ratio berechnen**:
   ```python
   if prev_kf.baseline_matches < 0:        # erster Frame nach Commit
       prev_kf.baseline_matches = n_matches
       ratio = 1.0                          # c2 zwingend False in diesem Frame
   else:
       baseline = prev_kf.baseline_matches
       if baseline <= 0:
           baseline = max(min(n_kp_curr, n_kp_prev), 1)   # defensiver Fallback
       ratio = n_matches / baseline
   ```
   Damit ist `baseline_matches` ein **per-KF stabiler Referenzwert** —
   strukturell das Analog zu `mpReferenceKF->TrackedMapPoints(3)` im
   Original (siehe Adaptionstabelle).

6. **Spacing-Test (OR)**:
   ```python
   c1a = N >= cfg.max_frames
   c1b = N >= cfg.min_frames
   spacing_ok = c1a or c1b
   ```

7. **Novelty-Test (AND)**:
   ```python
   c2 = (ratio < cfg.tracked_ratio_thresh) and (n_matches >= cfg.min_tracked)
   ```

8. **Finale Entscheidung (AND von 6 und 7)**:
   ```python
   accept = spacing_ok and c2
   ```

9. **State-Update**: bei Accept → `prev_kf` aktualisieren, `N = 0`. Bei Reject
   → `N += 1`.

10. **Score zurückgeben** mit `(accept, OrbSlam3Score(...))` damit das
    PhaseTimer-Summary diagnostisch bleibt. Der Score exponiert `c1a`, `c1b`,
    `c2` einzeln, damit man bei der Auswertung sieht *warum* etwas abgelehnt
    wurde.

## Adaptionen vs. Original (Feature-SLAM mit Covisibility-Graph)

| Original ORB-SLAM3 | VINGS-Adaption | Grund |
|---|---|---|
| Reference-KF K_ref aus dem Covisibility-Graph (KF mit max. gemeinsamen Map-Punkten) | Sequentieller `prev_kf` (zuletzt akzeptierter Mapper-KF) | VINGS hat keinen ORB-Covisibility-Graph; sequentiell ist der konservativere Default |
| Numerator: `mnMatchesInliers` aus `SearchByProjection` (Map-Point-Reprojektion + Inlier-Filter via RANSAC/PnP) | ORB-zu-ORB-Match zwischen `current.desc` und `prev_kf.desc` (BFMatcher, NORM_HAMMING, crossCheck=True) | VINGS-Map sind Gaussians, kein Feature-Set; ohne Map-Point-Reprojektion ist BFMatch der direkteste Ersatz-Operator. Misst dieselbe Größe: „wie viele Features des Ref-Frames sind noch wiederfindbar" |
| Ratio-Nenner = `nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs)` (Map-Punkte im Ref-KF mit ≥`nMinObs` Beobachtungen; per-KF stabil) | Ratio-Nenner = `baseline_matches` (BFMatch-Count am 1. Frame nach KF-Commit; per-KF stabil) | Ohne Map-Punkt-Tracking ist die natürliche Ersatzgröße der Match-Count direkt nach KF — beides „was war zur KF-Zeit da" — und macht 0.9 semantisch identisch zum Original („10 % Drop") |
| `nMinObs = 3`, bzw. `nMinObs = 2` bei `nKFs ≤ 2` (Original Z. 3097-3099): Map-Point-Beobachtungsschwelle | entfällt | VINGS hat keine Map-Points mit Beobachtungszählern. `baseline_matches` ist *eine* per-KF stabile Größe, nicht *die* gleiche |
| c1b prüft „N≥MinFrames AND Local Mapping idle" | Nur `N≥min_frames` | Selector wird synchron vor `mapper.run()` aufgerufen; Mapper ist per Konstruktion idle, das Signal entfällt |
| c1c stereo-only (`mSensor != MONOCULAR && mnMatchesInliers < nRefMatches*0.25`) | entfällt | VINGS ist Mono — keine close/far-Map-Punkt-Unterscheidung |
| c2 enthält stereo-only `bNeedToInsertClose` | entfällt | dito |
| `mnMatchesInliers > 15` (c2 Precondition) | `n_matches >= min_tracked` (Default 50, konservativer) | Defensiver für Gaussian-Mapper (low-quality KF kontaminiert weniger als bei Map-Points, aber unnötiger Mapping-Aufwand) |
| Bootstrap-Threshold `thRefRatio = 0.4` bei `nKFs < 2` (Original Z. 3131-3132) | entfällt; immer 0.9 | VINGS-Bootstrap erfolgt über „first frame always accept"-Pfad. Konsequenz: in den ersten 1-2 KFs einer Sequenz minimal aggressiver als Original (Trigger bei 10 % statt 60 % Drop). Praktisch vernachlässigbar; falls relevant ist es ein 4-Zeilen-Patch |
| Reloc-Gate (Original Z. 3091-3094: `if mnId < mnLastRelocFrameId + MaxFrames && nKFs > MaxFrames → return false`) | entfällt | VINGS hat keine Relokalisation |
| Mapper-busy-Return-Logik (Original Z. 3191-3209): bei busy Mapper im Mono-Pfad `return false`, auch wenn `(c1a||c1b)∧c2` true ist | entfällt | VINGS-Selector ist synchron; Mapper ist bei Aufruf per Konstruktion idle |
| c3 (inertial timestamp ≥ 0.5 s, Original Z. 3166-3179) | entfällt | VINGS ist Pure-Mono ohne IMU |
| c4 (IMU_MONOCULAR tracking-weak gate, Original Z. 3181-3185) | entfällt | dito |

Der Kern-Entscheidungsmechanismus — **AND zwischen `(c1a ∨ c1b)` und `c2`**,
mit `ratio < 0.9` als primärem Novelty-Trigger — ist **verbatim** aus
ORB-SLAM/ORB-SLAM2/ORB-SLAM3 (siehe `Tracking.cc::NeedNewKeyFrame` im
offiziellen Repo).

## Historie / Korrektur

Eine frühere Implementierung (vor 2026-05-26) verwendete eine **drei-OR-Logik**
mit `(force-rate ∨ tracking-weak ∨ ratio-drop)`, was strukturell vom Paper
abweicht. Konkrete Befunde des Reviews:

1. **AND/OR-Logik invertiert**: Original ist `(c1a ∨ c1b) ∧ c2`. Die OR-Form
   führte dazu, dass `c1a` allein (Force-Rate ohne Novelty) KFs auslöste —
   auch bei reiner Standkamera. Paper-Verhalten: kein KF auf Standstill.
2. **„Tracking-weak → KF" war sinngemäß invertiert**: Original verlangt
   `matches > 15` als **Precondition** (zu wenige Matches → kein KF, Map
   nicht kontaminieren). Die alte Version triggerte bei `matches < 50` einen
   **Force-Insert** — das Gegenteil.
3. **Ratio-Nenner falsch skaliert**: alt war `matches / n_kp_prev`. Bei
   `orb_n_features=800` und typischen 150–400 BFMatch-Treffern lag die Ratio
   bei 0.2–0.5 und das 0.9-Threshold feuerte effektiv jeden Frame.

Eine erste Korrektur (commit am gleichen Tag) stellte den Nenner auf
`min(n_kp_curr, n_kp_prev)` um. Das ist eine harte Obergrenze möglicher
Matches — aber im realen Betrieb erreicht BFMatch+crossCheck zwischen
zwei aufeinanderfolgenden Frames nur 30–60 % davon. Damit lag `ratio`
wieder strukturell viel zu tief, und 0.9 hatte nicht die Paper-Semantik
„10 % Drop vom Tracking-Niveau am KF-Anfang".

Zweite Korrektur (jetzt aktiv): `baseline_matches` als per-KF stabiler
Referenzwert, gesetzt auf den BFMatch-Count des ersten Frames nach
Commit. Das ist das strukturelle Analog zu `mpReferenceKF->TrackedMapPoints(3)`
im Original — beide messen „Tracking-Niveau am KF-Anfang" und sind
über die KF-Lebensdauer konstant. Damit ist das 0.9-Threshold direkt
paper-äquivalent.

Triggernamen im Score: `bootstrap`, `novelty`, `force_rate+novelty`,
`baseline_warmup`, `forced_diag`, `pathological`.

## Sensitivität

### `tracked_ratio_thresh` (wichtigster Knopf)

| Wert | Effekt |
|---|---|
| 0.95 | Sehr früh KF — fast jedes Bewegungs-Inkrement erzeugt einen KF. Hohe Mapping-Last. |
| 0.90 (Paper-Default) | ORB-SLAM3-Standard. 10 % Feature-Migration triggert KF. Bewährt in EuRoC/TUM-VI. |
| 0.80 | Aggressiver Sparsifier — wartet bis 20 % der Features rausgewandert sind. Weniger KFs, leicht erhöhtes Drift-Risiko bei langen Sequenzen. |
| 0.60 | Sehr aggressiv. Kann auf VINGS funktionieren weil Mapper Lücken durch Gaussian-Densification füllt; nur wenn Mapping-Budget hart constraint ist. |

### `max_frames`

Spacing-Failsafe (c1a). Bei 30 fps und `max_frames=15` ist nach 0.5 s
spacing_ok auch ohne c1b. **Aber:** ohne Novelty (c2) wird trotzdem kein KF
erzeugt. Das ist nicht mehr „force jeder N Frames" wie in der alten
OR-Version; wer einen harten Mindesttakt will, sollte `mapper_kf_skip`
verwenden (kompletter Selector-Bypass).

### `min_frames`

Untere Spacing-Grenze. Default 1 = darf in jedem Tracker-KF nach dem KF
zuschlagen, falls Ratio-Condition feuert. Auf 3–5 setzen wenn Tracker-KF-
Rate hoch ist und Mapping nicht hinterherkommt.

### `min_tracked`

Im Original 15. Bei VINGS mit `orb_n_features=800` setzen wir defensiv 50 —
fungiert als Precondition für c2: bei sehr wenigen Inliers wird das Ratio-
Signal unzuverlässig. Wenn `c2`-Spalte im Diagnose-Modus oft an `min_tracked`
scheitert (statt am Ratio-Threshold), `min_tracked` senken oder ORB-Budget
erhöhen.

## Tuning-Workflow

1. **Default laufen lassen** (`tracked_ratio_thresh: 0.9`, `max_frames: 30`,
   `min_tracked: 50`) auf der Zielsequenz.
2. **PhaseTimer-Summary** prüfen: `frame_select` sollte 10–40 ms/Frame liegen
   (ORB-Detektion + BFMatch). Linear in `orb_n_features`.
3. **KF-Rate** ablesen (`n_mapped / n_keyframes`). Erwartung bei smallcity_200:
   ~25–35 % bei aktivem Movement. Bei statischen Sequenzen bewusst niedriger.
4. **Vergleich zum Mapping-Budget** aus `KEYFRAME.md`: Pass-Time
   `n_mapped × 1150 ms` muss unter dem Wandzeit-Budget liegen.
5. **Wenn zu viele KFs**: `tracked_ratio_thresh` Richtung 0.85/0.80.
6. **Wenn zu wenig KFs** (Tracking-Drift sichtbar): `tracked_ratio_thresh` auf
   0.93/0.95. **Hinweis**: `max_frames` senken hilft *nicht* mehr (AND-Logik;
   c1a allein reicht nicht).

## Code-Pointer

| Datei | Inhalt |
|---|---|
| `scripts/vings_utils/orbslam3_selector.py` | Selector + Config + Score + Smoketest |
| `scripts/vings_utils/selector_factory.py` | Registrierung als `kind: orbslam3` |
| `scripts/run.py:258–275` | unveränderter Call-Site |
| `configs/local/smallcity/orbslam3/` | smallcity-Beispiel-YAMLs |

Standalone-Smoketest:

```bash
PYTHONPATH=scripts python scripts/vings_utils/orbslam3_selector.py
```

Erwartete Ausgabe: erster Frame Accept (bootstrap), stationäre Frames skippen
(`c2`=0 wegen hoher Ratio), nach lateraler Translation drückt `ratio < 0.9`
**und** Mindest-Inliers passen → `c2`=1 → Accept. Im zweiten Static-Block
(`static-2`) bleibt `c2`=0 → **kein** KF, auch wenn `c1a` (Force-Rate) feuert
— das ist die paper-konforme AND-Semantik.

## Was im BA-Methodenkapitel stehen sollte

Die Implementierung folgt der originalen ORB-SLAM/2/3-Logik
`(c1a ∨ c1b) ∧ c2`, mit den folgenden dokumentierten Mono-/VINGS-Adaptionen.

**Strukturelle Adaptionen** (unvermeidbar wegen Map-Point-loser
Gaussian-Architektur):

1. **Reference-KF**: sequentieller `prev_kf` statt Covisibility-Graph-K_ref.
   In ORB-SLAM3 wird die Ratio gegen den KF gemessen, der die meisten
   Map-Punkte mit dem aktuellen Frame teilt. Da VINGS keinen
   ORB-Covisibility-Graph führt (Map sind Gaussians, keine ORB-Punkte),
   nehmen wir den letzten akzeptierten Mapper-KF.
2. **Numerator-Operator**: Original verwendet `mnMatchesInliers` aus
   `SearchByProjection` (Map-Point-Reprojektion ins aktuelle Bild +
   Inlier-Filter via RANSAC/PnP). Wir matchen ORB-Deskriptoren des
   aktuellen Frames gegen die gecachten ORB-Deskriptoren von `prev_kf`
   (BFMatcher, NORM_HAMMING, cross-check). Beide Operatoren messen
   semantisch dasselbe; ohne Map-Point-Datenstruktur ist BFMatch der
   direkteste Ersatz.
3. **Ratio-Nenner = `baseline_matches`** (BFMatch-Count am ersten Frame
   nach KF-Commit). Original-Nenner
   `nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs)` ist ohne
   Map-Punkt-Tracking nicht verfügbar. `baseline_matches` ist das
   strukturelle Analog: per-KF stabil, repräsentiert das Tracking-Niveau
   am KF-Anfang, und macht das Paper-Threshold 0.9 direkt äquivalent zu
   „10 % Drop vom Baseline-Tracking".
4. **`nMinObs`-Schwelle entfällt**: Original filtert Map-Points danach, ob
   sie von mindestens `nMinObs = 3` (bzw. `2` bei `nKFs ≤ 2`)
   Keyframes beobachtet wurden. VINGS hat keine Beobachtungszähler;
   `baseline_matches` umgeht das durch eine andere stabile Größe.

**Architektonische Adaptionen** (synchroner Selector statt asynchrones
Tracking-Thread):

5. **c1b ohne `bLocalMappingIdle`**: Im Original ist
   `c1b = (N ≥ MinFrames) ∧ Mapper.AcceptKeyFrames()`. VINGS-Selector
   wird synchron *vor* `mapper.run()` aufgerufen; Mapper ist bei
   Selector-Entscheidung per Konstruktion idle, das Signal entfällt.
6. **Mapper-busy-Return-Logik entfällt**: Original Z. 3191-3209 — bei
   busy Mapper im Mono-Pfad gibt das Original `false` zurück, auch wenn
   `(c1a||c1b) ∧ c2` true ist. Bei synchronem VINGS-Selector ist Mapper
   nie busy zum Entscheidungszeitpunkt.
7. **Reloc-Gate entfällt**: Original Z. 3091-3094 blockiert KF-Insertion
   nach Relokalisation. VINGS hat keine Relokalisation.

**Mono-Reduktionen** (Bedingungen, die in jedem Mono-System wegfallen):

8. **c1c (stereo-only) entfällt**: Bedingt durch
   `mSensor != MONOCULAR && mSensor != IMU_MONOCULAR && …` — gating
   close-/far-Map-Punkt-Logik aus. VINGS ist Mono.
9. **c3, c4 (inertial-only) entfallen**: Timestamp-Gate (c3) und
   IMU_MONOCULAR-Tracking-Weak-Gate (c4) erfordern IMU; VINGS ist
   Pure-Mono.
10. **`bNeedToInsertClose` entfällt**: Stereo/RGB-D close-point-Logik;
    irrelevant für Mono.

**Tuning-Adaptionen** (bewusst gesetzte VINGS-Defaults):

11. **`min_tracked = 50` statt Paper-15**: defensivere Mindest-Inlier-
    Precondition für c2. Bei ORB-Budget 800 ist 15 zu permissiv und
    führt zu unzuverlässigen Ratio-Signalen.
12. **Bootstrap-`thRefRatio = 0.4` entfällt**: Original Z. 3131-3132
    setzt das Threshold in den ersten 1-2 KFs einer Sequenz auf 0.4
    (60 % Drop erforderlich, konservativer). VINGS-Bootstrap nutzt den
    „first frame always accept"-Pfad und behält 0.9 durchgehend. Effekt
    auf die ersten 1-2 KFs minimal aggressiver; praktisch vernachlässigbar.
    Falls relevant: 4-Zeilen-Patch in `should_accept`.

Die Drei-Bedingungs-Struktur (`c1a`, `c1b`, `c2`) und die finale AND-Logik
sind verbatim aus Campos et al. 2021 bzw. Mur-Artal et al. 2017
(ORB-SLAM2). Der Algorithmus ist klein genug, dass es keine versteckten
Implementierungsdetails gibt — alles Wesentliche steht oben.
