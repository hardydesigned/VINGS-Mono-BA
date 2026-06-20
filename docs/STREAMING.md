# Live-Streaming zu einem Web-Frontend

## In einfachen Worten

Bisher sah man das Ergebnis eines Runs erst **am Ende** (die `.ply` der Map, die
Objekt-CSV/PLY). Mit dem Stream-Modul schickt VINGS **während** des Laufs zwei
Dinge live an ein Browser-Frontend:

1. **Die Gaussians** als kompakte `.splat`-Pakete — die Punktwolke wächst im
   Browser mit, während der Mapper arbeitet.
2. **Die erkannten Objekte** (YOLO/RT-DETR + 3D-Lokalisierung) als kleine
   JSON-Nachrichten — Marker erscheinen live an ihrer Weltposition.

Verbindung läuft über **WebSocket**: der Run ist der Server (pusht), dein
Frontend verbindet sich als Client.

### Das Problem, das wir lösen mussten

Naiv „immer nur die neuen Gaussians schicken" geht **nicht**, weil der Mapper
bestehende Gaussians laufend **weiter-optimiert und teils löscht** (Prune in
`add_new_frame`, Gradient-Steps, `storage_control`). Das Frontend liefe sonst
out-of-sync.

Lösung = **frozen/active-Split** (nutzt die vorhandene StorageManager-Partition):

| Teil | Quelle | Verhalten | Stream-Strategie |
|---|---|---|---|
| **frozen** | StorageManager (CPU) | ausoptimiert, wächst KF-weise | **inkrementell** (`append_frozen`), Frontend hängt an |
| **active** | Mapper (GPU) | klein, ändert sich ständig | **jedes Mal voll ersetzen** (`replace_active`) |

So wandert nur das kleine active-Set komplett über die Leitung, das große
frozen-Set wächst nur. Der Delta-Schlüssel ist `_globalkf_id` (stabil pro
Gaussian, überlebt Prune — der Array-Index nicht).

## Benutzen

1. **Einmalig** das `websockets`-Paket in die vings-env installieren:
   ```bash
   pip install "websockets>=12,<13"
   ```
2. `stream:`-Block in die Run-Config (Default ist **aus**):
   ```yaml
   stream:
     enabled: true
     host: 0.0.0.0
     port: 8765
     every_kf: 1                # alle N gemappten KFs Gaussians pushen
     max_active_splats: 200000  # Cap nur fuer den active-Snapshot
     max_queue: 16              # drop-oldest-Tiefe
     flat_scale_eps: 0.001      # 3. Splat-Achse fuer 2DGS-Disks (>0!)
   ```
   Fertige Beispiel-Config: `configs/local/stream/interval1_stream.yaml`
   (mit `detect_objects: true`, streamt also auch Objekt-Marker).
3. Run starten wie gewohnt. Im Log erscheint
   `[stream] viewer at http://0.0.0.0:8765/  (WebSocket on same port)`.
4. **Im Browser einfach `http://localhost:8765/` öffnen.** Der Stream-Server
   liefert die `viewer.html` über **denselben Port** aus, auf dem auch der
   WebSocket läuft — die Seite verbindet sich dann automatisch same-origin
   zurück. Nur **ein** Port, kein zweiter HTTP-Server, kein http/ws-Mismatch.

   Das Haupt-Frontend ist eine 3D-Karten-Szene (Grid-Boden, Achsen, Orbit-
   Steuerung), in die die Gaussians live reingerendert werden, plus beschriftete
   Objekt-Marker und ein HUD. **Zwei Render-Modi umschaltbar:**
   - **Disks** (Default): jeder Gaussian als orientiertes, eingefärbtes
     2DGS-Scheibchen, depth-tested → solide, navigierbare 3D-Oberfläche.
   - **Splats**: echte EWA-Gaussian-Splats (Custom-Shader, additive, order-
     independent) → weicher Splat-Look.

   Controls: Modus-Toggle, Splat-Größe, Opacity, Labels/Grid an-aus, Up-Achse
   (Z-up/Y-up), models-Toggle, **camera-frame-Toggle**, **drei obj-Rotations-
   Slider (rotX/rotY/rotZ, −180…+180°)** zum Live-Kalibrieren der Modell-
   Orientierung (Werte per Klasse in `models/registry.json` als `rotX/rotY/rotZ`
   persistierbar), „frame scene".
   WS-Adresse per Eingabefeld oder URL-Hash
   (`…:8765/#ws://anderer-host:8765`) überschreibbar. Auto-Reconnect ist eingebaut.

   Smoketest-Viewer (Punkte): `http://localhost:8765/test_viewer.html`.
   Roher Frame-Strom: `wscat -c ws://localhost:8765`.

> **VS Code / Remote / Port-Forwarding:** Genau **einen** Port (8765) forwarden
> und `http://localhost:8765/` im Browser öffnen. Weil Seite **und** WebSocket
> auf demselben Port und gleicher Origin liegen, geht der WS-Upgrade sauber durch
> VS Codes Forwarding (auch über `https`-Tunnel → die Seite nutzt dann
> automatisch `wss://`). Den ws-Port **nicht** separat „direkt" im Browser
> aufrufen und **nicht** auf einem zweiten Port serven — das löst sonst die
> Meldung *„invalid Connection header: keep-alive / you cannot access a WebSocket
> server directly with a browser"* aus.

> Hinweis zu den Modi: **Disks** ist die robuste Baseline mit korrekter
> Verdeckung (opak, Depth-Buffer). **Splats** nutzt **premultiplied Alpha-Over-
> Blendung** (`C = src + dst·(1−srcα)`) — dasselbe Transmittance-Compositing-
> Schema wie der VINGS-2DGS-Rasterizer, sodass die Farbe in [0,1] beschränkt
> bleibt. Früher lag hier reine additive Blendung; die summierte überlappende
> Splats über 1.0 auf und ließ dichte, opake Flächen zu **Weiß** ausbrennen.
> Alpha-Over ist ohne Tiefensortierung zwar zeichenreihenfolge-abhängig
> (approximativ), gibt aber korrekte, gesättigte Farben statt Blowout. Beide
> Modi teilen sich dieselben gestreamten Rohdaten; der Umschalter baut nur die
> GPU-Objekte neu.

Die HTML-Dateien laufen auch direkt per `file://` (ES-Module + three.js vom CDN);
dann muss die ws-Adresse aber manuell auf den Server zeigen (Default
`ws://localhost:8765`). Der Ein-Port-Weg über `http://…:8765/` ist robuster.

Standalone-Smoketest ohne GPU-Run (Dummy-Gaussians):
```bash
python scripts/server/stream_server.py
```

## Wire-Protokoll (für dein eigenes Frontend)

Das Frontend unterscheidet zuerst nach **Frame-Typ**, dann nach dem `type`-Feld.

**Binary-Frame = Splat-Daten:**
```
[uint32-LE header_len][utf8 JSON header][.splat bytes]
```
Header: `{"type": ..., "epoch": int, "kf_id"?: int, "n": int}` mit `type` in
`append_frozen` | `replace_active` | `replace_all`.

`.splat`-Layout (32 bytes/Gaussian, antimatter15/gsplat-kompatibel):

| Offset | Bytes | Feld |
|---|---|---|
| 0 | 12 | position `3×f32` (DROID-Welt) |
| 12 | 12 | scale `3×f32` (linear, `exp(_scaling)`; 3. Achse synthetisch klein) |
| 24 | 4 | rgba `4×u8` (RGB aus `_rgb`, A aus `sigmoid(_opacity)`) |
| 28 | 4 | rot `4×u8` (Quaternion **`(w,x,y,z)`** normalisiert, `round(q*128+128)`) |

> Quaternion-Reihenfolge ist `(w,x,y,z)` (VINGS/3DGS-nativ). Manche Web-Viewer
> dekodieren `(x,y,z,w)` — dann im Frontend umsortieren. `test_viewer.html`
> nutzt nur die Position, ist also unkritisch.

**Text-Frame = JSON:**

| `type` | Aktion im Frontend |
|---|---|
| `append_frozen` (binary) | Splats des KF zur frozen-Wolke **hinzufügen** |
| `replace_active` (binary) | active-Wolke **komplett ersetzen** |
| `replace_all` (binary) | (Fallback ohne StorageManager) ganze Szene ersetzen |
| `objects` (text) | Marker-Liste ersetzen: `[{object_id, class, cls_id, conf, n_hits, xyz:[x,y,z], quat:[w,x,y,z], size:[sx,sy,sz]}]` |
| `frame` (text) | Live-Kamera-Frame: `{idx, w, h, jpeg:<base64-JPEG>}` → Frontend zeigt es in der **Kamera-Karte** unten rechts (Toggle „camera frame"). Eigener Takt `stream.frame_stride`, downscaled auf `stream.frame_max_px`. |
| `resync` (text) | frozen+active **leeren**, `currentEpoch = epoch` setzen |

Der Server sendet **komplette** Chunks (neue frozen-KF-Gruppen via `append_frozen` + den vollen live-Mapper-Satz via `replace_active`; ein `replace_all` vor dem ersten Convey). Das „nach und nach in die Karte laden" passiert rein **frontend-seitig** über den progressiven Reveal (siehe `viewer.html`-Zeile in der Architektur-Tabelle) — die Leitung bleibt simpel und vollständig.

**Objekt-Orientierung + Größe (für 3D-Modelle):** Jedes Objekt trägt zusätzlich zu
`xyz` jetzt `quat:[w,x,y,z]` (Rotation um die Welt-Hoch-Achse, DROID ist Z-up;
`(w,x,y,z)`-Konvention wie die `.splat`-Quaternionen) und `size:[long, lateral,
vertical]` (metrische Extents im gauge-freien DROID-Frame). Beide kommen aus einer
PCA über die entprojizierte Tiefen-Punktwolke der Detektions-Box
(`object_tracker.estimate_pose_size`). **180°-Yaw-Ambiguität:** eine PCA-Hauptachse
ist vorzeichenlos → die Heading ist auf `[0, π)` kanonisiert; vorne/hinten ist aus
Geometrie allein nicht bestimmbar. Bei zu wenigen Tiefenpixeln (`min_pca_px`) oder
inkohärenter Orientierung über die Sichtungen → Identitäts-Quaternion + isotrope
Fallback-Größe (keine falsche Konfidenz).

**3D-Modell-Modus im Frontend:** `viewer.html` rendert jedes Objekt als **echtes
glTF-Modell pro Klasse** (`MODEL_REGISTRY`, via `GLTFLoader`), an `xyz` positioniert,
per `quat` orientiert und per `size` skaliert. Fehlt für eine Klasse ein Modell
(oder lädt es nicht), fällt der Viewer auf eine **orientierte, per-Achse-skalierte
Wireframe-Box** zurück — gleiche Pose/Größe, nur eben eine Box. Toggle „models"
schaltet Modelle ↔ Box-Marker. Die `.glb`-Assets liegen in
`scripts/server/static/models/` und werden vom Stream-Server über `…/models/<name>.glb`
ausgeliefert (der Static-Server erlaubt genau dieses eine `models/`-Unterverzeichnis,
mit realpath-Traversal-Guard). Klassen-Mapping ohne Code-Change überschreibbar via
`static/models/registry.json`. Details + CC0-Quellen: `scripts/server/static/models/README.md`.

**`epoch`-Gating:** Bei Loop-Closure (`use_loop`) werden frozen Gaussians global
verschoben → der Server schickt `resync` mit erhöhter `epoch` und re-streamt
danach alles. Das Frontend hält `currentEpoch` und **verwirft** Frames mit
`epoch < currentEpoch` (verspätete pre-resync-Pakete). Neu verbindende Clients
bekommen automatisch einen Backlog (alle frozen-Chunks + letzter active-Snapshot
+ letzte Objekte) der aktuellen Epoch.

## Architektur / wo im Code

| Datei | Inhalt |
|---|---|
| `scripts/server/stream_server.py` | `SplatStreamServer`: WebSocket-Server in einem **daemon-Thread** mit eigener asyncio-Loop. Run-Loop → Server über `queue.Queue` mit **drop-oldest** (nur `replace_active`/`objects` droppbar; `append_frozen`/`resync` nicht). Broadcast an alle Clients + Late-Join-Backlog. Liefert per `process_request` auch die `static/`-HTML **über denselben Port** aus (Browser-GET → `viewer.html`; WS-Upgrade → Handshake) → Ein-Port-Setup für Port-Forwarding. |
| `scripts/server/splat_encode.py` | `.splat`-Serializer: `encode_splat_from_mapper` (GPU, aktiviert via `get_property`), `encode_splat_from_storage` (CPU, raw float16 → manuell aktiviert), `_pad_scale`, `_select_indices`. |
| `scripts/server/static/viewer.html` | **Haupt-Frontend**: three.js 3D-Karten-Szene mit zwei umschaltbaren Render-Modi (orientierte 2DGS-Disks via `InstancedMesh` / echte EWA-Gaussian-Splats via Custom-`ShaderMaterial`, premultiplied Alpha-Over), Objekt-Markern mit Labels, HUD, UI-Controls, Auto-Reconnect. Gemeinsames Rohdaten-Modell, Modus-Toggle baut nur die GPU-Objekte neu. **Progressive Reveal (rate-basiert):** ankommende Chunks ploppen nicht als Block rein, sondern die gerenderte Instanz-Anzahl wächst mit fester **Rate** `REVEAL_RATE` (Punkte/Sekunde, ein globales Budget) hoch (`InstancedMesh.count` / `geometry.instanceCount`). Folge: **große Chunks bauen proportional länger auf** (ein 50k-Chunk baut sich sichtbar über Sekunden auf statt in einem Frame), und frozen-Chunks werden **sequenziell in Ankunftsreihenfolge** gefüllt → die Karte wächst KF für KF, Punkt nach Punkt. Der active-Satz (per-Tick full-replace) reveal-t ebenfalls, trägt seinen shown-count aber über die Replaces (kein Flackern; baut einmalig von leer auf). `REVEAL_RATE` ist der einzige Knopf (niedriger = gradueller). HUD zeigt „streaming N" solange Reveals laufen; `window.__reveal()` ist der Debug-Hook. |
| `scripts/server/static/test_viewer.html` | Minimaler Smoketest-Viewer (Gaussian-Zentren als Punkte). Schneller Protokoll-Check, kein Render-Schnickschnack. |
| `scripts/run.py` | Init-Gate (`stream.enabled`), `_stream_push_gaussians()` (Delta-Logik), OD-Push (nach `object_tracker.update`), Gaussian-Push (nach Storage-Run), Loop-Resync (nach `looper.run`), Cleanup. PhaseTimer-Phase `stream`. |
| `scripts/vings_utils/object_tracker.py` | `snapshot()` — disk-freie Live-Variante von `finalize()`. |

**Non-blocking-Garantie:** Alle Run-Loop-Hooks sind best-effort (`try/except`)
und rufen nur `queue.put_nowait`-basierte Methoden. Der `.splat`-Encode läuft im
Run-Thread (wenige ms für ≤200k active, nur alle `every_kf` KFs). Der Server ist
ein daemon-Thread → ein SIGKILL durch den VRAM-Watchdog lässt den Prozess nicht
hängen. Ein toter/langsamer Client kann den Mapper nie stallen (drop-oldest).

## GPS-verankerte Karten-Projektion (Map-Mode, 2026-06-18)

Wenn ein `dataset.gps_csv` konfiguriert ist, projiziert der Viewer die gauge-freie
DROID-Map **live auf echte Satellitenbilder** (Esri World Imagery) — derselbe Fix
wie in `od-experiments`. Standard an (`stream.geo: false` schaltet ab).

**Producer** (`scripts/server/geo_frame.py`, `LiveGeoReferencer`): sammelt pro
Tracker-KF DROID-Kamerazentrum+Blickrichtung und GPS-ENU (`data_packet['xyz_enu']`),
baut daraus eine Sim3 DROID→ENU (kein Umeyama — Flug ist eine fast gerade GPS-Linie
und damit rotations-degeneriert; stattdessen `up = -mean(cam-forward)`,
`heading = DROID-Sehne → GPS-Sehne`, `scale = GPS/DROID-Längs-Span`,
Zentroid-Match; **rechtshändige Basis `right = fwd × up`** — `up × fwd` spiegelt die
ganze Szene). Daraus wird die 4×4 DROID→three-Matrix (`M = A·Sim3`, `A = ENU→three`)
und per `geo`-Message ans Frontend geschickt. Ein Hintergrund-Thread lädt die
Satelliten-Kacheln für die Trajektorien-Bbox und **re-fetcht, sobald der Flug über
die abgedeckte Fläche hinauswächst** (sonst deckt die Karte nur den Flugbeginn ab).

**Frontend** (`viewer.html`, Map-Mode): wendet `M` als Matrix auf die Splat-Gruppe
an (Cloud georeferenziert „for free"), legt die Satelliten-Ebene auf die tile-
alignte ENU-Extent, cullt SLAM-Floater per Clipping-Box (Satelliten-Footprint +
vertikales Band um den Boden-Modus `groundY`) und platziert Objekte als **aufrechte
per-Klasse-glTF-Modelle in realen Fahrzeug-Maßen** (Position = `M·xyz` auf den Boden
gepinnt, Heading aus der geo-rotierten Objekt-Orientierung) plus Ring+Beam-Marker.

`geo`-Message + Objekte liegen im Backlog (late-joining Clients), und werden VOR
dem schweren Frozen-Blob gesendet, damit Karte+Marker auch bei Verbindungsabbruch da
sind. Three.js ist nach `static/vendor/` vendored (offline-fähig, kein unpkg-CDN).

**End-to-End-Test ohne SLAM:** `scripts/server/replay_run.py` spielt einen fertigen
Run (PLY + tracker_raw_c2w + rtk.csv + objects_droid.csv) durch den Streaming-Pfad.
Mit `--shot OUT.png` rendert es headless (Playwright, swiftshader; `ulimit -s 1024`,
siehe `od-experiments`) und screenshotet; `VIEW=top|closeup_obl|objN` steuert die
Kamera, `--max-splats N` drosselt für den langsamen Headless-Client (echte Browser
schaffen 260k+).

Config: `stream.geo` (bool, default true), `stream.geo_min_kfs`, `stream.geo_min_span_m`,
`stream.geo_zoom`.
