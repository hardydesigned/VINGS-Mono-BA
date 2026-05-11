/backup
- Backup von dem alten conda env des Repo, jetzt neu die setup.sh mit eigenem conda

/build_gtsam
- GTSAM = Georgia Tech Smoothing and Mapping – eine C++-Bibliothek für Faktor-Graph-Optimierung, entwickelt für Robotik und SLAM.
- Was GTSAM in VINGS-Mono macht: In dbaf_frontend.py werden diese GTSAM-Faktoren zusammengebaut: CombinedImuFactor "Zwischen Frame i und j hat der IMU diese Bewegung gemessen", PriorFactor "Startpose ist ungefähr hier", BetweenFactor "Diese zwei Posen haben diesen relativen Abstand"
-> GTSAM löst dann das gesamte System und gibt optimierte Posen + IMU-Bias + Geschwindigkeiten zurück.
-> Nur aktiv bei Modus "vio" (Kamera + IMU) und berechnet dann die Posen wenn man IMU Daten hat

/ckpts
- Vortrainierte Modelle heruntergeladen:
  droid.pth – das Herzstück: Das DroidNet – ein neuronales Netz (trainiert auf vielen Videosequenzen), das aus zwei Frames gleichzeitig:
  - optischen Fluss schätzt (welcher Pixel gehört wohin?)
  - Unsicherheiten berechnet (wie verlässlich ist diese Korrespondenz?)
  DroidNet bekommt einen Graphen von Frame-Paaren und läuft in
  einer iterativen Schleife:
  Schritt i:
    1. Aktuelle Posen + Tiefen → projiziere Frame j auf Frame i → erwartete
  Pixelpositionen
    2. Korreliere Features an diesen Positionen → Fluss-Residuum
    3. ConvGRU schätzt:
       - delta  = Flusskorrektur ("Pixel sollte eher hier sein")
       - weight = Unsicherheit dieser Korrektur (pro Pixel, x und y)
    4. Bundle Adjustment nutzt (delta + weight) → updated Posen + Tiefen
    5. Zurück zu 1.

  Unsicherheiten worüber?
  weight ist eine 2D-Karte pro Pixel (x- und y-Richtung) – die Unsicherheit über den optischen Fluss an jedem Bildpunkt.
  Diese Informationen werden im Bundle Adjustment genutzt, um Posen und Tiefen zu verfeinern.

  DroidNet ist das Netz aus DROID-SLAM (Teed & Deng, 2021). Es wurde trainiert auf:
  - TartanAir – synthetischer Datensatz mit Ground-Truth-Tiefen und -Posen (Drohnenflüge
   durch simulierte Umgebungen)
  - Möglicherweise auch FlyingThings3D und anderen Optical-Flow-Datensätzen

  Das Training optimiert gleichzeitig optischen Fluss + Posen + Tiefen über alle Frames gemeinsam (nicht nur paarweise wie klassische Optical-Flow-Netze).

  ---

  metric_depth_vit_small_800k.pth – monokulare Tiefe
  Das Metric3D v2 (Small) Modell. Berechnet aus einem einzelnen RGB-Bild eine metrisch skalierte Tiefenkarte (also in echten Metern, nicht relativ).
  Wird nur genutzt wenn use_metric: True in der Config. Es läuft für jeden Frame, bei dem kein Tiefensensor vorhanden ist.
  Wie es hier eingebunden ist:
  depth = self.predictor(rgb_image=img_numpy, intrinsic=self.intr, d_max=self.d_max)
  - Es bekommt das Bild + deine Kameraparameter (fu, fv, cu, cv) → gibt Tiefe in Metern
  zurück
  - d_max = 300.0 – alles über 300m wird gekappt
  - Das Ergebnis wird auf die Originalgröße reskaliert und direkt als
  data_packet['depth'] übergeben

  Metric3D v2 wurde auf einem sehr großen gemischten Datensatz trainiert,
  der viele verschiedene Kameratypen und Szenen umfasst – indoor, outdoor, driving, aerial.
  ---
  lightglue/superpoint.onnx – Keypoint-Extraktion
  SuperPoint – ein neuronales Netz, das in einem Bild markante Punkte (Keypoints) erkennt und jedem einen Deskriptor-Vektor zuweist. 
  Nimmt ein einzelnes Bild und gibt zurück:
  - Keypoints – markante Punkte im Bild (Ecken, Kanten, charakteristische Strukturen)
  - Deskriptoren – ein Vektor pro Keypoint, der beschreibt wie die lokale Umgebung
  aussieht
  Trainiert auf synthetischen geometrischen Formen + Homographie-Augmentierung
  (Self-Supervised). Lernt selbst, welche Punkte stabil und wiedererkenntbar sind.

  ---
  lightglue/superpoint_lightglue.onnx – Feature-Matching
  
  LightGlue – nimmt die SuperPoint-Keypoints aus zwei Bildern und entscheidet, welche Punkte zusammengehören.   Bekommt die Keypoints + Deskriptoren aus zwei SuperPoint-Aufrufen und entscheidet,
  welche Punkte zusammengehören. Nutzt einen Transformer (Attention zwischen den beiden
  Keypoint-Mengen) und gibt zurück:
  - Matches – welcher Keypoint in Bild A entspricht welchem in Bild B
  - Match-Scores – Konfidenz pro Match 
  Warum zwei getrennte Modelle?  
  SuperPoint läuft einmal pro Bild und das Ergebnis kann gecacht werden. LightGlue läuft dann nur für Paare, die tatsächlich verglichen werden. Das ist effizienter als ein End-to-End-Modell das immer beide Bilder gleichzeitig braucht.
  Trainiert auf: MegaDepth (outdoor Landmark-Szenen mit SfM Ground-Truth) + weitere. LightGlue lernt explizit mit SuperPoint-Features umzugehen – daher der Name superpoint_lightglue.onnx.



/configs
- Config Dateien für den Run, Parameter:
  - vo Modus – nur Kamera, Posen werden rein visuell durch Bundle Adjustment (DroidNet +
   CUDA-Backend) berechnet. Kein GTSAM.
  - vio Modus – Kamera + IMU. GTSAM baut einen Faktor-Graph aus visuellen Posen +
  IMU-Messungen und optimiert beides gemeinsam. Gibt genauere, metrisch skalierte Posen.


    Sehr relevant für dich

  Datei/Ordner: frontend/dbaf_frontend.py
  Warum: Kernlogik Frame-Auswahl – __update(), Keyframe-Entscheidung, BA-Iterationen
  ────────────────────────────────────────
  Datei/Ordner: frontend/motion_filter.py
  Warum: Stufe 1 der Frame-Auswahl
  ────────────────────────────────────────
  Datei/Ordner: frontend/covisible_graph.py
  Warum: Wie Kanten zwischen Frames verwaltet werden
  ────────────────────────────────────────
  Datei/Ordner: frontend/depth_video.py
  Warum: Der zentrale Datenpuffer – alle Frames, Posen, Tiefen
  ────────────────────────────────────────
  Datei/Ordner: run.py
  Warum: Einstiegspunkt, Gesamtablauf

Erklärungen:

  motion_filter.py – Stufe 1: Frame vorfiltern

  Läuft auf jedem eingehenden Frame. Berechnet den optischen Fluss zwischen dem letzten
  akzeptierten Frame und dem neuen – aber nur eine einzige Update-Iteration (schnell):

  _, delta, weight = self.update(self.net[None], self.inp[None], corr)

  if delta.norm(dim=-1).mean().item() > self.thresh:  # thresh=2.4
      # genug Bewegung → Frame in den Buffer
      self.video.append(...)
  else:
      self.count += 1  # Frame still

  Überschreitet der mittlere Fluss den Schwellwert nicht → Frame wird komplett 
  verworfen, kommt nie in den Buffer.

  ---
  depth_video.py – der zentrale Ringpuffer
  
  Hält alle Daten aller Keyframes im GPU-Speicher:

  tstamp    [buffer]         – Timestamps
  images    [buffer, 3, H, W]  – RGB-Bilder
  poses     [buffer, 7]        – Posen als [t, q] (Translation + Quaternion)
  disps     [buffer, H/8, W/8] – Disparität (= 1/Tiefe) von DroidNet
  disps_sens[buffer, H/8, W/8] – Disparität von Metric3D
  disps_up  [buffer, H, W]     – Disparität hochskaliert
  fmaps     [buffer, 128, ...]  – Feature-Maps (fnet)
  nets      [buffer, 128, ...]  – Context-Features (cnet)

  buffer=80 in deiner Config – d.h. maximal 80 Keyframes gleichzeitig im aktiven
  Fenster. Daher der __rollup() alle 65 Frames.

  ---
  covisible_graph.py – der Faktor-Graph
  
  Verwaltet welche Frames miteinander verglichen werden (Kanten ii → jj):

  Aktive Kanten (ii, jj) – werden gerade im BA optimiert
  Inaktive Kanten (ii_inac, jj_inac) – zu alt fürs aktive Fenster, aber ihre
  target/weight fließen noch in den BA ein

  Wichtige Methoden:
  - add_proximity_factors() – fügt neue Kanten basierend auf Pose-Distanz hinzu
  - rm_factors() – entfernt alte Kanten (mit optionalem Speichern als inaktiv)
  - rm_keyframe() – löscht einen Frame komplett aus dem Graphen
  - update() – ein BA-Schritt: Fluss berechnen → DroidNet-Update → Bundle Adjustment

  ---
  dbaf_frontend.py – Stufe 2: Keyframe-Entscheidung
  
  Das ist deine Hauptdatei. Der kritische Abschnitt in __update():

  # Photometrische Distanz zwischen Frame t1-3 und t1-2
  d = self.video.distance([self.t1-3], [self.t1-2], beta=self.beta, bidirectional=True)

  if d.item() < self.keyframe_thresh:   # keyframe_thresh=4.0
      self.graph.rm_keyframe(self.t1-2) # → kein Keyframe, Frame löschen
  else:
      # → Keyframe behalten
      self.graph.update(...)             # noch eine BA-Iteration
      self.new_frame_added = True

  distance() misst wie verschieden zwei Frames sind – kombiniert aus Pose-Distanz und
  Feature-Distanz (gesteuert durch beta).

  ---
  run.py – Gesamtablauf

  Orchestriert alles. Der relevante Teil nach dem Tracking:

  viz_out = judge_and_package(self.tracker, data_packet['intrinsic'])

  if viz_out is not None:             # neues Keyframe?
      self.mapper.run(viz_out, True)  # Gaussians trainieren
      self.looper.run(...)            # Loop Closure prüfen

  judge_and_package() in middleware_utils.py entscheidet, ob new_frame_added=True
  gesetzt ist und packt dann alle relevanten Daten für den Mapper zusammen.

  ---
  Zusammengefasst als Pipeline:

  MotionFilter (motion_filter.py)
    → genug Bewegung? → DepthVideo.append() (depth_video.py)
       → DBAFusionFrontend.__update() (dbaf_frontend.py)
          → CovisibleGraph.add_proximity_factors() (covisible_graph.py)
          → CovisibleGraph.update() – BA
          → Keyframe-Entscheidung via distance()
             → rm_keyframe() oder new_frame_added=True
                → judge_and_package() → mapper (run.py)

  ---
  Nützlich zum Verstehen

  Datei/Ordner: frontend/dbaf.py
  Warum: Initialisierung des gesamten Tracking-Stacks                
  ────────────────────────────────────────
  Datei/Ordner: datasets/generic_vo.py
  Warum: Wie deine Daten geladen werden
  ────────────────────────────────────────
  Datei/Ordner: vings_utils/middleware_utils.py
  Warum: judge_and_package() – was passiert zwischen Tracker und Mapper

  ---
  Kannst du ignorieren (für dein Ziel)

  ┌───────────────────┬───────────────────────────────────────────┐
  │      Ordner       │              Warum unwichtig              │
  ├───────────────────┼───────────────────────────────────────────┤
  │ gaussian/         │ Mapping, läuft nach der Frame-Auswahl     │
  ├───────────────────┼───────────────────────────────────────────┤
  │ loop/             │ Loop Closure, unabhängig                  │
  ├───────────────────┼───────────────────────────────────────────┤
  │ metric/           │ Tiefenschätzung, vorgelagert              │
  ├───────────────────┼───────────────────────────────────────────┤
  │ storage/          │ Speicherverwaltung, nachgelagert          │
  ├───────────────────┼───────────────────────────────────────────┤
  │ frontend_vo/      │ Alternatives Frontend, du nutzt frontend/ │
  ├───────────────────┼───────────────────────────────────────────┤
  │ server/, dynamic/ │ Mobile/Dynamic-Features, nicht relevant   │
  └───────────────────┴───────────────────────────────────────────┘



/submoduls

  submodules/dbaf – das CUDA-Backend des Trackings
  
  Das ist der leistungskritische C++/CUDA-Teil. Enthält:
  - droid_kernels.cu – CUDA-Kernel für Dense Bundle Adjustment (Matrixoperationen auf
  GPU)
  - bacore.h – BA-Kernlogik
  - altcorr_kernel.cu / correlation_kernels.cu – Feature-Korrelation auf GPU

  Wird als Python-Extension kompiliert: droid_backends – das ist das import 
  droid_backends in dbaf.py. Du rufst es nie direkt auf, aber ohne es läuft nichts. Für
  deine Frame-Auswahl-Änderungen musst du hier nicht rein.

  ---
  submodules/dbef – alternatives Tracking-Backend
  
  Steht für "DBA-EF" (eine Variante). Enthält nur thirdparty/ mit eigen und lietorch.
  Wird in diesem Repo kaum genutzt – du kannst es ignorieren.

  ---
  submodules/diff-surfel-rasterization – der Gaussian-Renderer
  
  Der differenzierbare Rasterizer für 2D Gaussian Splatting. C++/CUDA-Code der die
  Splats rendert und Gradienten zurückpropagiert. Wird in gaussian_model.py importiert
  als:
  from diff_surfel_rasterization import GaussianRasterizationSettings,
  GaussianRasterizer
  Für dich nicht relevant.

  ---
  submodules/gtsam – Faktor-Graph-Bibliothek
  
  Wie besprochen: nur im vio-Modus relevant. Große C++-Bibliothek, bereits kompiliert in
   build_gtsam/. Für dich nicht relevant.

  ---
  submodules/metric_modules – Metric3D-Modell
  
  Der Code hinter metric_depth_vit_small_800k.pth. Enthält metric.py, fusion.py,
  droid.py – die Inferenz-Pipeline für Metric3D v2. Wird als Python-Package importiert.
  Für dich nicht relevant.

  ---
  Zusammenfassung für dein Ziel: Alle Submodule sind für die Frame-Auswahl irrelevant.
  Der einzige Berührungspunkt ist droid_backends (aus dbaf), der im Hintergrund die
  BA-Berechnungen ausführt – aber du änderst dessen Aufrufe nicht, sondern die Logik
  darum herum in dbaf_frontend.py.