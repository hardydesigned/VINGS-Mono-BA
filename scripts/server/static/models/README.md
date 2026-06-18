# Per-class 3D models for the live viewer

`viewer.html` renders each streamed object as a **real glTF model** looked up in
`MODEL_REGISTRY` (in `viewer.html`) by its class name, placed at the object's
DROID-world position, **oriented** by the streamed `quat` (yaw about world up) and
**scaled** by the streamed `size = [long, lateral, vertical]`.

Drop `.glb` files here and they are served by `stream_server.py` at
`http://<host>:<port>/models/<file>.glb` (the static file server allows exactly
this one `models/` subdirectory; nothing nested or traversing out).

## Expected files (default registry)

| class | file        |
|-------|-------------|
| car   | `car.glb`   |
| truck | `truck.glb` |
| bus   | `bus.glb`   |
| van   | `car.glb`   |

Override the mapping without touching `viewer.html` by adding `registry.json`
here, e.g.:

An entry is either a URL string or an object with `url` and optional fixed
rotation offsets in **degrees**: `rotX`, `rotY`, `rotZ` (and `yaw`, an alias that
adds to `rotZ`). Use the viewer's three **obj rotX/rotY/rotZ sliders** to dial in
the right orientation live, then bake those values here per class (the sliders are
a *global* offset added on top, so reset them to 0 after baking):

```json
{ "car":   { "url": "models/car.glb",   "rotX": 0,   "rotY": 0, "rotZ": 0 },
  "truck": { "url": "models/truck.glb", "rotX": 180, "rotY": 0, "rotZ": 0 },
  "bus":   { "url": "models/bus.glb",   "rotX": 180, "rotY": 0, "rotZ": 0 } }
```

`rotX`/`rotY` fix upside-down / on-its-side; `rotZ` (or `yaw`) is the front/back
heading flip (the streamed yaw is **180°-ambiguous** by construction). It is
fetched at startup and merged over the hardcoded defaults.

**Placement (automatic):** the viewer auto-aligns the asset to the marker box —
assumes the model is **Z-up** (matches the DROID scene) and that the vehicle
**length is the longer of the model's X/Y extents**, rotates that length axis onto
the box heading (+X), recentres the asset on its bbox centre, and scales it
**uniformly** (largest factor that still fits the whole model inside the detected
`size` box — no per-axis squash). Y-up or odd-facing assets: bake the correction
into the export or add a `yaw`.

## No asset shipped? You still see objects.

If a `.glb` is missing (404) or fails to load, the viewer falls back to an
**oriented, per-axis-scaled wireframe box** — same position/orientation/size, just
a box instead of a model. So the feature works end-to-end without any asset; the
`.glb` is purely a visual upgrade.

## Getting a CC0 car model

Use a **CC0 / public-domain** low-poly car (keep it small, < ~200 KB, plain `.glb`,
no Draco so the frontend stays build-free), e.g. from Kenney
(kenney.nl/assets — "Car Kit"), Quaternius, or Poly Pizza (filter: CC0). Export/
convert to `.glb` and save as `car.glb` here. Note the model's forward axis: the
box convention is local **+X = vehicle length** (the yaw heading). If a model
faces another axis, bake a fixed rotation into the export (or extend the registry
with a per-class yaw offset).

Document the source + license of whatever you add below:

- `car.glb`: "Car" by Quaternius via Poly Pizza (https://poly.pizza/m/HQ0hvRM2XR),
  direct: https://static.poly.pizza/098ec278-73b6-4ef8-a825-338cb06b675f.glb —
  CC-BY 3.0 (Quaternius releases CC0 originals; the same low-poly cars are CC0 at
  https://quaternius.itch.io/lowpoly-cars). Z-up, length along model +Y, ~133 KB,
  no Draco.
- `truck.glb`: "Dump truck" by jeremy via Poly Pizza (https://poly.pizza/m/1BpGYg14QGD),
  direct: https://static.poly.pizza/1511f362-d635-4dc6-b9a2-a70fdb6370b2.glb —
  CC-BY 3.0. Y-up, length along model +Z, ~126 KB, no Draco.
- `bus.glb`: "Bus" by Poly by Google via Poly Pizza (https://poly.pizza/m/4CPpvEmrMoF),
  direct: https://static.poly.pizza/635a02b7-4260-42a5-bd52-987511d6e3e0.glb —
  CC-BY 3.0. Y-up, length along model +Z, ~45 KB, no Draco.

(The viewer auto-handles the differing up/forward axes — see "Placement" above —
so no manual rotation is needed per asset; only the front/back `yaw` flip is manual.)
