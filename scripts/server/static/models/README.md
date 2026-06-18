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
| truck | `car.glb`   |
| bus   | `car.glb`   |

Override the mapping without touching `viewer.html` by adding `registry.json`
here, e.g.:

```json
{ "truck": "models/truck.glb", "person": "models/person.glb" }
```

It is fetched at startup and merged over the hardcoded defaults.

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

- `car.glb`: _(add source + license here)_
