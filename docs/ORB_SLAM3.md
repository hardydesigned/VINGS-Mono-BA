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
  Schwenk) → Notfall-Foto, damit du den Faden nicht ganz verlierst.
* **Wenn schon eine Weile kein Foto mehr gemacht wurde** → einfach ein
  Routine-Foto, damit die Lücke nicht zu groß wird.

Genau das macht ORB-SLAM3. „Wiedererkennen" heißt: das System extrahiert pro
Bild kleine markante Punkte (sogenannte ORB-Features, das sind Ecken/Kanten)
und matcht sie gegen die Punkte des letzten Keyframes. Solange genug Punkte
wiedergefunden werden, läuft das Tracking weiter — kein neuer Keyframe nötig.
Sobald die Trefferquote unter eine Schwelle (Paper: 90 %) fällt, gibt's einen
neuen KF.

Das war's. Es ist eine **„Drop-below-Ratio"-Heuristik** mit zwei
Sicherheitsnetzen drumherum (Force-Rate + Tracking-Notfall). Wahrscheinlich die
am häufigsten zitierte Keyframe-Auswahl in der SLAM-Welt — fast jeder
feature-basierte VO/SLAM (VINS-Mono, OKVIS, Kimera) nutzt eine Variante davon.
Der Charme: **ein einziger Knopf** (`tracked_ratio_thresh`) deckt den Großteil
der Tuning-Wünsche ab.

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

Pro Tracker-KF-Kandidaten wird **alles** Folgende geprüft. Akzeptiere falls
**eine** der drei Bedingungen feuert:

```
# Voraussetzung: ORB(current) berechnet, Matches zu prev_kf gezählt
matches      = #ORB_matches(current, prev_kf)
n_kp_prev    = #keypoints(prev_kf)
ratio        = matches / max(n_kp_prev, 1)
N            = frames_since_kf

# Bedingung A — Force-Rate (C1 im Originalpaper)
accept iff   N >= max_frames

# Bedingung B — Tracking droht zu kippen (C3 im Original)
accept iff   matches < min_tracked          # "less than 50 points"

# Bedingung C — substantielle Neuheit (C4 im Original)
accept iff   ratio < tracked_ratio_thresh   # default 0.9 (paper)
        AND  N >= min_frames                # min_frames Spacing-Untergrenze

reject sonst.
```

Bei Accept wird `prev_kf` auf den aktuellen Frame gesetzt, sein ORB-Keypoint-
Set + Deskriptoren gecached und `frames_since_kf = 0`. Der erste Frame wird
immer akzeptiert (Bootstrap).

### Intuition pro Bedingung

* **Bedingung C** ist der eigentliche Trigger: solange genug ORB-Features des
  letzten KF noch im Bild sichtbar sind, ist die Map-Information „frisch genug"
  — kein neuer KF nötig. Sobald >10 % der Features rausgewandert sind, gibt's
  neuen Content → KF.
* **Bedingung A** ist der Failsafe gegen Stillstand: bei statischer Kamera
  würde C nie feuern, aber wir wollen trotzdem regelmäßig KFs. Im Originalpaper
  ist während der IMU-Init explizit von 4 Hz Force-Rate die Rede (Sec. V.B).
* **Bedingung B** ist der Notfall-Pfad: wenn das Tracking gerade kollabiert
  (Motion-Blur, schnelle Drehung), brauchen wir sofort einen neuen KF, damit
  das System nicht komplett wegläuft. Im Originalpaper Sec. V.B: weniger als
  15 Map-Punkte tracked → System gilt als visuell verloren.

## Variablen

| Symbol | Bedeutung |
|---|---|
| `prev_kf` | zuletzt akzeptierter Keyframe (Pose, ORB-Keypoints, Deskriptoren) |
| `matches` | Anzahl ORB-Cross-Check-Matches zwischen `current` und `prev_kf` |
| `n_kp_prev` | Gesamtzahl ORB-Keypoints im `prev_kf` (Nenner für Ratio) |
| `ratio` | `matches / n_kp_prev` ∈ [0, 1], **fällt** mit zunehmender Neuheit |
| `N` | Frames seit dem letzten akzeptierten KF (Spacing-Counter) |
| `min_frames` | Untere Spacing-Grenze (vermeidet KF-Bursts) |
| `max_frames` | Obere Spacing-Grenze (Force-Rate-Failsafe) |
| `min_tracked` | Tracking-Notfall-Schwelle (analog zu Original „less than 50 points") |
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
   Pose merken, Accept zurückgeben. Hier gibt es noch nichts zu vergleichen.

3. **Folge-Frame**: ORB extrahieren, `matcher.match(desc_curr, desc_prev)`
   aufrufen, `n_matches = len(matches)` zählen.

4. **Ratio berechnen**: `ratio = n_matches / max(n_kp_prev, 1)`.

5. **Drei-Conditions-Test**:
   ```python
   if N >= cfg.max_frames:          accept = True; trigger = "force_rate"
   elif n_matches < cfg.min_tracked: accept = True; trigger = "tracking_weak"
   elif ratio < cfg.tracked_ratio_thresh and N >= cfg.min_frames:
                                    accept = True; trigger = "ratio_drop"
   else:                            accept = False
   ```

6. **State-Update**: bei Accept → `prev_kf` aktualisieren, `N = 0`. Bei Reject
   → `N += 1`.

7. **Score zurückgeben** mit `(accept, OrbSlam3Score(...))` damit das
   PhaseTimer-Summary diagnostisch bleibt.

## Adaptionen vs. Original (Feature-SLAM mit Covisibility-Graph)

| Original ORB-SLAM3 | VINGS-Adaption | Grund |
|---|---|---|
| Reference-KF K_ref aus dem Covisibility-Graph (KF mit max. gemeinsamen Map-Punkten) | Sequentieller `prev_kf` (zuletzt akzeptierter Mapper-KF) | VINGS hat keinen ORB-Covisibility-Graph; sequentiell ist der konservativere Default |
| „Map points tracked" via Projektion-und-Match gegen die globale Map | ORB-zu-ORB-Match zwischen `current` und `prev_kf` (BFMatcher, cross-check) | VINGS-Map sind Gaussians, kein Feature-Set; Frame-zu-Frame-ORB ist der direkteste Ersatz |
| C2 prüft „Local Mapping idle ODER N≥max_frames" | Nur `N≥max_frames` | Selector hat keine Mapper-Idle-Info; Mapper läuft sowieso pro accepted KF |
| Stereo/RGB-D-Sonderpfad (close points) | weggelassen | VINGS ist Mono — entfällt |

Der Kern-Entscheidungsmechanismus (drei Bedingungen, OR-verknüpft, mit
`ratio < tracked_ratio_thresh` als primärem Trigger) ist **verbatim** aus
ORB-SLAM/ORB-SLAM2/ORB-SLAM3.

## Sensitivität

### `tracked_ratio_thresh` (wichtigster Knopf)

| Wert | Effekt |
|---|---|
| 0.95 | Sehr früh KF — fast jedes Bewegungs-Inkrement erzeugt einen KF. Hohe Mapping-Last. |
| 0.90 (Paper-Default) | ORB-SLAM3-Standard. 10 % Feature-Migration triggert KF. Bewährt in EuRoC/TUM-VI. |
| 0.80 | Aggressiver Sparsifier — wartet bis 20 % der Features rausgewandert sind. Weniger KFs, leicht erhöhtes Drift-Risiko bei langen Sequenzen. |
| 0.60 | Sehr aggressiv. Kann auf VINGS funktionieren weil Mapper Lücken durch Gaussian-Densification füllt; nur wenn Mapping-Budget hart constraint ist. |

### `max_frames`

Force-Rate-Failsafe. Bei 30 fps und `max_frames=15` werden mindestens 2 KFs/s
erzwungen (Original ORB-SLAM3 nutzt 4 Hz während IMU-Init). Bei VINGS ohne IMU
ist Force-Rate optional — kann auf 30–60 hochgeschoben werden ohne dass
Tracking leidet (DBaF kümmert sich um Posen, der Selector entscheidet nur
Mapper-KFs).

### `min_frames`

Verhindert Burst-Inserts (Mapper-Latenz!). Default 1 = darf in jedem
Tracker-KF zuschlagen, falls Ratio-Condition feuert. Auf 3–5 setzen wenn
Tracker-KF-Rate hoch ist und Mapping nicht hinterherkommt.

### `min_tracked`

Im Original 50. Bei VINGS mit `orb_n_features=800` ist das defensiv —
fungiert als reiner Notfall-Pfad. Selten aktiv außer bei katastrophalem Blur.
Wenn nie feuert, ist alles ok.

## Tuning-Workflow

1. **Default laufen lassen** (`tracked_ratio_thresh: 0.9`, `max_frames: 30`,
   `min_tracked: 50`) auf der Zielsequenz.
2. **PhaseTimer-Summary** prüfen: `frame_select` sollte 10–40 ms/Frame liegen
   (ORB-Detektion + BFMatch). Linear in `orb_n_features`.
3. **KF-Rate** ablesen (`n_mapped / n_keyframes`). Erwartung bei smallcity_200:
   ~25–35 %. Bei dichten Sequenzen wie NTU-VIRAL ähnlich.
4. **Vergleich zum Mapping-Budget** aus `KEYFRAME.md`: Pass-Time
   `n_mapped × 1150 ms` muss unter dem Wandzeit-Budget liegen.
5. **Wenn zu viele KFs**: `tracked_ratio_thresh` Richtung 0.85/0.80.
6. **Wenn zu wenig KFs** (Tracking-Drift sichtbar): `tracked_ratio_thresh` auf
   0.93/0.95 oder `max_frames` auf 15 senken.

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

Erwartete Ausgabe: erster Frame Accept, stationäre Frames skippen
(ratio≈1.0), nach lateraler Translation drückt ratio < 0.9 und der Frame wird
akzeptiert, Force-Rate feuert spätestens bei N=max_frames.

## Was im BA-Methodenkapitel stehen sollte

Die Implementierung weicht in zwei dokumentierten Punkten vom Originalvorgehen
ab:

1. **Reference-KF**: sequentieller `prev_kf` statt Covisibility-Graph-K_ref.
   In ORB-SLAM3 wird die Ratio gegen den KF gemessen, der die meisten
   Map-Punkte mit dem aktuellen Frame teilt. Da VINGS keinen
   ORB-Covisibility-Graph führt (Map sind Gaussians, keine ORB-Punkte),
   nehmen wir den letzten akzeptierten Mapper-KF. Das ist konservativer
   (kann früher KFs fordern als das Original), aber semantisch konsistent zur
   übrigen VINGS-Plugin-Architektur.
2. **„Map points tracked" → „ORB matches Frame-zu-Frame"**: Original projiziert
   Map-Punkte aus K_ref ins aktuelle Bild und prüft Match-Erfolg. Wir matchen
   stattdessen ORB-Deskriptoren des aktuellen Frames gegen die gecachten
   ORB-Deskriptoren von `prev_kf` (BFMatcher, Hamming, cross-check). Beide
   messen das gleiche: wie viele Features des letzten KF sind im aktuellen
   Frame noch wiederfindbar.

Die Drei-Bedingungs-OR-Logik (`force-rate ∨ tracking-weak ∨ ratio-drop`) und
der Default `tracked_ratio_thresh = 0.9` sind verbatim aus Campos et al. 2021
bzw. Mur-Artal et al. 2017 (ORB-SLAM2). Der Algorithmus ist klein genug, dass
es keine versteckten Implementierungsdetails gibt — alles Wesentliche steht
oben.
