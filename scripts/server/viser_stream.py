"""viser-based live viewer for VINGS Gaussians + object markers.

Replaces the hand-rolled ``stream_server.py`` (raw .splat-over-WebSocket +
Three.js ``viewer.html``). viser ships its own WebSocket server + WebGL
frontend, so we only feed it point clouds and markers and it handles transport,
camera, GUI and incremental in-place updates.

Why this looks better than the old viewer (same client-side-render model):
* **rounded, perspective-sized points** (``point_shape="rounded"``) instead of
  flat fixed-thickness sprites,
* **adaptive world-space point size** derived from the actual point spacing, so
  neighbouring points just touch and form a continuous surface (no gaps/blobs);
  a GUI slider scales it live,
* **floater filtering** by opacity in the point adapters (``min_opacity``).

Interface mirrors the old ``SplatStreamServer`` (``start`` / ``stop`` /
``push``) so ``run.py`` barely changes. Payload types accepted by ``push``:

* ``{'type': 'kf', 'kf_id': int, 'xyz': (N,3) f32, 'rgb': (N,3) u8, 'frozen': bool}``
* ``{'type': 'objects', 'objects': [ {object_id, class, cls_id, xyz, ...}, ... ]}``
* ``{'type': 'detections', 'frame_b64': <jpeg b64>, 'boxes': [ {cls_name, cls_id,
  conf, bbox_xyxy}, ... ]}``  -> live camera frame + boxes in the GUI 'Camera'
  panel (2D PiP, not placed in the 3D scene)
* ``{'type': 'resync'}``  -> drop the whole scene (loop closure / new epoch)
"""

from __future__ import annotations

import threading

import numpy as np

try:
    import viser
except Exception as _e:  # pragma: no cover - import guard
    viser = None
    _VISER_IMPORT_ERR = _e


def _estimate_spacing(xyz: np.ndarray) -> float | None:
    """Median nearest-neighbour distance on a subsample (world-space spacing)."""
    n = len(xyz)
    if n < 8:
        return None
    m = min(n, 3000)
    sel = xyz if n <= m else xyz[np.random.choice(n, m, replace=False)]
    try:
        from scipy.spatial import cKDTree
        d, _ = cKDTree(sel).query(sel, k=2)
        d1 = d[:, 1]
        d1 = d1[np.isfinite(d1) & (d1 > 0)]
        if d1.size == 0:
            return None
        return float(np.median(d1))
    except Exception:
        # scipy missing: coarse fallback from bbox volume / count
        ext = xyz.max(0) - xyz.min(0)
        vol = float(np.prod(np.clip(ext, 1e-6, None)))
        return (vol / max(n, 1)) ** (1.0 / 3.0)


def _covariances(scale3: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    """(N,3) world scales + (N,4) wxyz quats -> (N,3,3) covariances = R diag(s^2) Rᵀ."""
    q = np.asarray(quat_wxyz, np.float32)
    q = q / np.clip(np.linalg.norm(q, axis=1, keepdims=True), 1e-8, None)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((len(q), 3, 3), np.float32)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - w * z); R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y); R[:, 2, 1] = 2 * (y * z + w * x); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    RS = R * np.asarray(scale3, np.float32)[:, None, :]      # scale the columns
    return np.einsum('nij,nkj->nik', RS, RS).astype(np.float32)


class ViserStreamServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 8765,
                 point_shape: str = "rounded", min_opacity: float = 0.0,
                 render_mode: str = "gaussians", max_total_splats: int = 500_000,
                 splat_bucket: int = 50_000, **_ignored):
        if viser is None:
            raise RuntimeError(
                f"`viser` package required for streaming: {_VISER_IMPORT_ERR}")
        self._host, self._port = host, port
        self._point_shape = point_shape
        self.min_opacity = float(min_opacity)
        self._render_mode = render_mode if render_mode in ("gaussians", "points") \
            else "gaussians"
        self._max_total_splats = max(0, int(max_total_splats))
        # Splat-buffer N must stay constant between rebuilds (a changing N makes
        # the viser splat client zero its sort buffer → blank). We round the live
        # count up to this bucket and pad with invisible filler, so N only steps
        # when a whole bucket fills (rare) instead of on every per-KF push.
        self._splat_bucket = max(1, int(splat_bucket))
        self._splat_capacity = 0         # current allocated (non-shrinking) N
        self._capped_warned = False      # one-time log when the memory ceiling bites

        self._lock = threading.Lock()
        self._frozen: dict = {}          # kf_id -> bool (is this group frozen?)
        # kf_id -> (xyz, rgb_u8, scale3, quat_wxyz, opacity) full-res attr cache,
        # so we can re-render in either mode on a GUI toggle without a re-run.
        self._raw: dict = {}
        # kf_id -> (xyz, rgb, cov_or_None, opacity) rendered for current settings.
        self._render: dict = {}
        # The whole scene is ONE merged splat object + ONE merged point cloud
        # (web gaussian-splat renderers keep a single sorted buffer; adding a 2nd
        # splat object blanks the first). New KFs => in-place buffer update, never
        # a new object -> no flicker, nothing disappears.
        self._gh = None                  # merged gaussian-splats handle
        self._ph = None                  # merged point-cloud handle
        # Debounced rebuild: the producer pushes ~16 groups PER CYCLE, each as a
        # separate push(). Rebuilding (= full re-send of the merged buffer) on every
        # push floods the websocket (16x/cycle) -> the client connection drops and
        # the whole scene blanks. Instead push() only marks dirty; a timer thread
        # coalesces bursts into ONE rebuild at a bounded rate.
        self._dirty = threading.Event()
        self._rebuild_hz = 4.0
        self._rebuild_thread = None
        self._stop_rebuild = threading.Event()
        self._obj_handles: list = []     # marker handles to clear each 'objects'
        self._base_size: float | None = None   # adaptive world-space point size
        self._size_scale = 1.0           # GUI multiplier (log slider, 10**v)
        self._downsample = 1             # GUI: keep every Nth point
        self._show_frozen = True
        self._show_active = True
        self._gui = {}                   # name -> gui handle
        self._cam_handle = None          # GUI image handle (live camera PiP)
        self._cam_folder = None          # folder the PiP lives in
        self._draw_boxes = True          # GUI: draw detection boxes on the frame
        self._show_camera = True         # GUI: show the camera panel at all
        self._last_frame = None          # (H,W,3) u8 RGB of the last camera frame
        self._last_boxes = []            # boxes of the last 'detections' payload
        self.server = None

    # ------------------------------------------------------------------ lifecycle
    def start(self):
        self.server = viser.ViserServer(host=self._host, port=self._port)
        self.server.scene.set_up_direction("-y")   # VINGS/DROID world is y-down
        self._build_gui()
        self._rebuild_thread = threading.Thread(
            target=self._rebuild_loop, daemon=True, name="viser-rebuild")
        self._rebuild_thread.start()
        print(f"[viser] viewer at http://{self._host}:{self._port}/")

    def _rebuild_loop(self):
        # Coalesce the ~16 per-cycle group pushes into ONE merged-buffer re-send
        # at a bounded rate. ALWAYS sleep the full period (don't wake on the dirty
        # event), so a rapid push burst collapses into a single rebuild per tick.
        period = 1.0 / max(0.5, self._rebuild_hz)
        while not self._stop_rebuild.is_set():
            self._stop_rebuild.wait(period)
            if self._stop_rebuild.is_set():
                break
            if self._dirty.is_set():
                self._dirty.clear()
                try:
                    self._rebuild_scene()
                except Exception as e:
                    print(f"[viser] rebuild failed: {e}")

    def _build_gui(self):
        g = self.server.gui
        gui = self._gui

        with g.add_folder("Render"):
            gui['mode'] = g.add_dropdown(
                "mode", ("gaussians", "points"), initial_value=self._render_mode,
                hint="gaussians = true 2DGS ellipsoids; points = round dots")
            gui['size'] = g.add_slider(
                "size", min=-1.5, max=1.5, step=0.02, initial_value=0.0,
                hint="log multiplier on point size / splat size")
            gui['shape'] = g.add_dropdown(
                "point shape", ("rounded", "circle", "square", "diamond"),
                initial_value=self._point_shape)
            gui['downsample'] = g.add_slider(
                "downsample", min=1, max=20, step=1, initial_value=1,
                hint="render every Nth Gaussian (visual only; full data kept)")

        with g.add_folder("Layers"):
            gui['show_active'] = g.add_checkbox("active (GPU)", initial_value=True)
            gui['show_frozen'] = g.add_checkbox("frozen (CPU)", initial_value=True)
            gui['bg'] = g.add_rgb("background", initial_value=(12, 12, 16))

        with g.add_folder("Stats"):
            gui['points'] = g.add_number("points", initial_value=0, disabled=True)
            gui['groups'] = g.add_number("groups", initial_value=0, disabled=True)
            gui['reset'] = g.add_button("reset view")

        # Live camera panel: the newest keyframe RGB with detection boxes drawn on
        # it. Fed by the 'detections' push (base64 JPEG + scaled boxes from run.py).
        self._cam_folder = g.add_folder("Camera")
        with self._cam_folder:
            gui['show_camera'] = g.add_checkbox("show camera", initial_value=True)
            gui['draw_boxes'] = g.add_checkbox("draw boxes", initial_value=True)
            # Placeholder until the first frame arrives (16:9 dark grey).
            ph = np.full((180, 320, 3), 32, np.uint8)
            self._cam_handle = g.add_image(
                ph, label="newest KF", format="jpeg", jpeg_quality=80)

        @gui['mode'].on_update
        def _(_e):
            self._render_mode = gui['mode'].value
            self._rebuild_all()

        @gui['size'].on_update
        def _(_e):
            self._size_scale = 10.0 ** float(gui['size'].value)
            if self._render_mode == "points":
                self._apply_point_size()       # cheap in-place
            else:
                self._rebuild_all()            # covariance depends on size

        @gui['shape'].on_update
        def _(_e):
            self._point_shape = gui['shape'].value
            if self._render_mode == "points":
                self._rebuild_all()

        @gui['downsample'].on_update
        def _(_e):
            self._downsample = max(1, int(gui['downsample'].value))
            self._rebuild_all()

        @gui['show_active'].on_update
        def _(_e):
            self._show_active = bool(gui['show_active'].value); self._apply_visibility()

        @gui['show_frozen'].on_update
        def _(_e):
            self._show_frozen = bool(gui['show_frozen'].value); self._apply_visibility()

        @gui['bg'].on_update
        def _(_e):
            self._set_background(gui['bg'].value)

        @gui['show_camera'].on_update
        def _(_e):
            self._show_camera = bool(gui['show_camera'].value)
            try:
                if self._cam_handle is not None:
                    self._cam_handle.visible = self._show_camera
            except Exception:
                pass

        @gui['draw_boxes'].on_update
        def _(_e):
            self._draw_boxes = bool(gui['draw_boxes'].value)
            self._refresh_camera()       # redraw last frame with/without boxes

        @gui['reset'].on_click
        def _(_e):
            for cid, ch in self.server.get_clients().items():
                try:
                    ch.camera.look_at = (0.0, 0.0, 0.0)
                except Exception:
                    pass
        self._set_background((12, 12, 16))

    def stop(self, timeout: float = 2.0):
        self._stop_rebuild.set()
        self._dirty.set()
        try:
            if self._rebuild_thread is not None:
                self._rebuild_thread.join(timeout=timeout)
        except Exception:
            pass
        try:
            if self.server is not None:
                self.server.stop()
        except Exception:
            pass

    # ------------------------------------------------------------------ helpers
    def _point_size(self) -> float:
        base = self._base_size if self._base_size else 0.01
        return max(1e-5, base * self._size_scale)

    def _apply_point_size(self):
        if self._render_mode == "points" and self._ph is not None:
            try:
                self._ph.point_size = self._point_size()
                return
            except Exception:
                pass
        self._rebuild_all()             # gaussians: covariance depends on size

    def _apply_visibility(self):
        self._dirty.set()               # include/exclude groups in the merge

    def _render_group_locked(self, kf_id):
        """Compute the per-group render arrays for current settings. Lock held."""
        attrs = self._raw.get(kf_id)
        if attrs is None:
            self._render.pop(kf_id, None)
            return
        xyz, rgb, scale3, quat, opacity = attrs
        ds = self._downsample
        if ds > 1:
            xyz, rgb, scale3, quat, opacity = (xyz[::ds], rgb[::ds], scale3[::ds],
                                               quat[::ds], opacity[::ds])
        cov = (_covariances(scale3 * float(self._size_scale), quat)
               if self._render_mode == "gaussians" else None)
        self._render[kf_id] = (np.ascontiguousarray(xyz, np.float32),
                               np.ascontiguousarray(rgb, np.uint8), cov,
                               np.ascontiguousarray(opacity, np.float32))

    def _gaussian_capacity(self, m: int) -> int:
        """Quantized, non-shrinking splat-buffer size for `m` live Gaussians.

        The viser splat client zeros its sort buffer whenever N changes
        (sizeChanged → every instance renders as Gaussian #0 → blank). N grows by
        one KF-worth of splats on every push, so a buffer sized exactly to the
        live count would resize — and blank — on *every* rebuild. Rounding up to a
        coarse bucket (and never shrinking) makes N step only when a whole bucket
        fills, so it stays constant across the per-cycle push bursts in between.
        """
        cap_max = self._max_total_splats
        if cap_max and m >= cap_max:
            cap = cap_max
        else:
            bucket = self._splat_bucket
            cap = ((m + bucket - 1) // bucket) * bucket
            cap = max(cap, self._splat_capacity)        # never shrink → no resize on dips
            if cap_max:
                cap = min(cap, cap_max)
        self._splat_capacity = cap
        return cap

    @staticmethod
    def _pad_gaussians(xyz, rgb, cov, opac, cap):
        """Pad the merged buffer up to `cap` with invisible (opacity-0) filler so
        N == cap regardless of the live count. Filler splats are alpha-blended with
        weight 0, so they never touch the image."""
        pad = cap - xyz.shape[0]
        if pad <= 0:
            return xyz, rgb, cov, opac
        xyz_p  = np.zeros((pad, 3), np.float32)
        rgb_p  = np.zeros((pad, 3), np.uint8)
        cov_p  = np.broadcast_to(np.eye(3, dtype=np.float32) * 1e-6, (pad, 3, 3)).copy()
        opac_p = np.zeros((pad, 1), np.float32)
        return (np.concatenate([xyz, xyz_p]), np.concatenate([rgb, rgb_p]),
                np.concatenate([cov, cov_p]), np.concatenate([opac, opac_p]))

    def _rebuild_all(self):
        """Re-render every group from cached attrs (mode/size/shape/downsample)."""
        with self._lock:
            for kid in list(self._raw):
                self._render_group_locked(kid)
        self._dirty.set()

    def _rebuild_scene(self):
        """Merge all visible groups into the single splat / point-cloud object."""
        with self._lock:
            # `self._raw` is the per-kf_id store. push() only ever ADDS a new id
            # or OVERWRITES an existing one (by id); ids are never dropped except
            # on an explicit resync. So a committed Gaussian group can never
            # disappear — every rebuild re-emits all live ids. Sorting gives a
            # stable merge order, i.e. a given id always occupies the same region
            # of the single buffer across rebuilds.
            ids = sorted(k for k in self._raw
                         if (self._show_frozen if self._frozen.get(k) else self._show_active)
                         and k in self._render)
            mode = self._render_mode
            if not ids:
                for h in (self._gh, self._ph):
                    try:
                        if h is not None:
                            h.visible = False
                    except Exception:
                        pass
            else:
                xyz = np.concatenate([self._render[k][0] for k in ids])
                rgb = np.concatenate([self._render[k][1] for k in ids])
                if mode == "gaussians":
                    cov  = np.concatenate([self._render[k][2] for k in ids])
                    opac = np.concatenate([self._render[k][3] for k in ids])
                    # Guard: drop any non-finite Gaussian AND any covariance value
                    # that would overflow float16 (viser packs cov as float16 in
                    # _scene_api.py). Inf in the covariance buffer can crash the
                    # WASM splat sorter and blank the whole scene.
                    F16MAX = float(np.finfo(np.float16).max)
                    cov_flat = cov.reshape(cov.shape[0], -1)
                    good = (np.isfinite(xyz).all(1)
                            & np.isfinite(cov_flat).all(1)
                            & (np.abs(cov_flat) < F16MAX).all(1)
                            & np.isfinite(opac).all(1))
                    if not good.all():
                        xyz, rgb, cov, opac = xyz[good], rgb[good], cov[good], opac[good]
                    if xyz.shape[0] == 0:
                        self._update_count(); return

                    # Hard MEMORY ceiling — NOT the disappear-fix. Only once the
                    # whole scene exceeds max_total_splats do we subsample, so the
                    # browser doesn't OOM. Below the ceiling nothing is ever dropped;
                    # N-stability comes purely from the bucketed padding below. Raise
                    # stream.max_total_splats to keep every Gaussian.
                    max_n = self._max_total_splats
                    if max_n > 0 and xyz.shape[0] > max_n:
                        if not self._capped_warned:
                            print(f"[viser] scene exceeds max_total_splats={max_n}; "
                                  f"subsampling to cap (raise stream.max_total_splats "
                                  f"to keep all)")
                            self._capped_warned = True
                        stride = max(1, xyz.shape[0] // max_n)
                        sl = slice(None, None, stride)
                        xyz  = xyz[sl][:max_n]
                        rgb  = rgb[sl][:max_n]
                        cov  = cov[sl][:max_n]
                        opac = opac[sl][:max_n]

                    # Pad up to a quantized, non-shrinking capacity so N stays
                    # constant while ids are added/updated (a changing N makes the
                    # viser splat client zero its sort buffer → blank). N only steps
                    # when a whole bucket fills, never on a normal per-id update.
                    cap = self._gaussian_capacity(xyz.shape[0])
                    xyz, rgb, cov, opac = self._pad_gaussians(xyz, rgb, cov, opac, cap)

                    # Always hide the point-cloud handle when switching to Gaussians.
                    try:
                        if self._ph is not None:
                            self._ph.visible = False
                    except Exception:
                        pass

                    if self._gh is None:
                        # First creation: establishes a stable UUID in the
                        # viser client's groupBufferFromId store.
                        self._gh = self.server.scene.add_gaussian_splats(
                            "/gaussians", centers=xyz, covariances=cov,
                            rgbs=rgb, opacities=opac)
                    else:
                        # In-place update via setters: SAME UUID every tick.
                        # Calling add_gaussian_splats() every rebuild emits a
                        # new UUID each time; the client accumulates all old
                        # UUIDs in groupBufferFromId → merged-N grows without
                        # bound → sizeChanged every 250ms → scene blanks every
                        # 250ms. Setters reuse the existing UUID so the client
                        # store stays at one entry. The bucketed padding above
                        # keeps N constant between rebuilds → sizeChanged fires
                        # only when a bucket boundary is crossed, not per push.
                        with self.server.atomic():
                            self._gh.centers = xyz
                            self._gh.covariances = cov
                            self._gh.rgbs = rgb
                            self._gh.opacities = opac
                            self._gh.visible = True
                else:
                    # Points mode: same hard memory ceiling (point clouds tolerate a
                    # changing N — no splat sort buffer — so no padding needed here).
                    max_n = self._max_total_splats
                    if max_n > 0 and xyz.shape[0] > max_n:
                        if not self._capped_warned:
                            print(f"[viser] scene exceeds max_total_splats={max_n}; "
                                  f"subsampling to cap (raise stream.max_total_splats "
                                  f"to keep all)")
                            self._capped_warned = True
                        stride = max(1, xyz.shape[0] // max_n)
                        sl = slice(None, None, stride)
                        xyz = xyz[sl][:max_n]
                        rgb = rgb[sl][:max_n]

                    try:
                        if self._gh is not None:
                            self._gh.visible = False
                    except Exception:
                        pass
                    if self._ph is None:
                        self._ph = self.server.scene.add_point_cloud(
                            "/points", points=xyz, colors=rgb,
                            point_size=self._point_size(),
                            point_shape=self._point_shape)
                    else:
                        with self.server.atomic():
                            self._ph.points = xyz
                            self._ph.colors = rgb
                            self._ph.point_size = self._point_size()
                            self._ph.visible = True
        self._update_count()

    def _set_background(self, rgb):
        try:
            img = np.zeros((2, 2, 3), np.uint8)
            img[:] = np.array([int(rgb[0]), int(rgb[1]), int(rgb[2])], np.uint8)
            self.server.scene.set_background_image(img)
        except Exception:
            pass

    def _update_count(self):
        try:
            self._gui['points'].value = int(sum(
                a[0].shape[0] for a in self._raw.values()))
            self._gui['groups'].value = len(self._raw)
        except Exception:
            pass

    # ------------------------------------------------------------------ producer
    def push(self, payload: dict):
        """NON-BLOCKING-ish. Called from the run loop; never raises."""
        try:
            t = payload.get("type")
            if t == "kf":
                self._push_kf(payload)
            elif t == "objects":
                self._push_objects(payload.get("objects") or [])
            elif t == "resync":
                self._clear_scene()
            elif t == "detections":
                # Live camera PiP: the newest KF RGB + detection boxes. Shown in
                # the GUI 'Camera' panel, NOT placed into the 3D scene.
                self._push_detections(payload)
        except Exception as e:
            print(f"[viser] push failed: {e}")

    def _push_kf(self, payload):
        kf_id = int(payload["kf_id"])
        xyz = np.ascontiguousarray(payload["xyz"], dtype=np.float32)
        rgb = np.ascontiguousarray(payload["rgb"], dtype=np.uint8)
        if xyz.shape[0] == 0:
            return
        # scale/quat/opacity are optional (older 'kf' payloads only had xyz/rgb);
        # synthesize benign defaults so gaussian mode still works.
        scale = payload.get("scale")
        quat = payload.get("quat")
        opacity = payload.get("opacity")
        n = xyz.shape[0]
        if scale is None:
            scale = np.full((n, 3), max(self._base_size or 0.02, 1e-3), np.float32)
        if quat is None:
            quat = np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1))
        if opacity is None:
            opacity = np.ones((n, 1), np.float32)
        attrs = (xyz, rgb, np.ascontiguousarray(scale, np.float32),
                 np.ascontiguousarray(quat, np.float32),
                 np.ascontiguousarray(opacity, np.float32).reshape(-1, 1))
        if self._base_size is None:
            sp = _estimate_spacing(xyz)
            if sp is not None:
                self._base_size = sp
                try:
                    self._gui['size'].hint = f"adaptive base = {sp:.4f} m (log mult)"
                except Exception:
                    pass
        self._add_or_update(kf_id, attrs, bool(payload.get("frozen")))

    def _add_or_update(self, kf_id, attrs, frozen):
        # Cache + render this one group, then mark dirty. The debounce thread does
        # the actual merged re-send (coalescing the per-cycle group burst).
        with self._lock:
            self._raw[kf_id] = attrs
            self._frozen[kf_id] = frozen
            self._render_group_locked(kf_id)
        self._dirty.set()

    def _push_objects(self, objects):
        with self._lock:
            for h in self._obj_handles:
                try:
                    h.remove()
                except Exception:
                    pass
            self._obj_handles = []
            for o in objects:
                xyz = o.get("xyz")
                if xyz is None:
                    continue
                oid = int(o.get("object_id", 0))
                cls_id = int(o.get("cls_id", 0))
                col = _class_color(cls_id)
                r = float(o.get("marker_radius", 0.6))
                pos = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
                try:
                    sh = self.server.scene.add_icosphere(
                        f"/obj/{oid}/s", radius=r, color=col, position=pos)
                    self._obj_handles.append(sh)
                    label = f"{o.get('class', cls_id)}#{oid}"
                    lh = self.server.scene.add_label(
                        f"/obj/{oid}/l", text=label, position=pos)
                    self._obj_handles.append(lh)
                except Exception:
                    pass

    # ------------------------------------------------------------------ camera PiP
    def _push_detections(self, payload):
        """Decode the streamed camera JPEG + boxes and update the GUI image panel.

        Payload (from run.py): ``frame_b64`` = base64 JPEG of the newest KF, and
        ``boxes`` = [{cls_name, cls_id, conf, bbox_xyxy}] already scaled to that
        JPEG's pixel size. We cache frame+boxes so a GUI 'draw boxes' toggle can
        re-render without a new push.
        """
        frame_b64 = payload.get("frame_b64")
        if not frame_b64:
            return
        try:
            import base64 as _b64, cv2 as _cv2
            buf = np.frombuffer(_b64.b64decode(frame_b64), np.uint8)
            bgr = _cv2.imdecode(buf, _cv2.IMREAD_COLOR)
            if bgr is None:
                return
            rgb = np.ascontiguousarray(bgr[..., ::-1])
        except Exception:
            return
        with self._lock:
            self._last_frame = rgb
            self._last_boxes = list(payload.get("boxes") or [])
        self._refresh_camera()

    def _refresh_camera(self):
        """Re-render the cached camera frame (boxes on/off) into the GUI panel."""
        if self._cam_handle is None:
            return
        with self._lock:
            frame = self._last_frame
            boxes = list(self._last_boxes)
        if frame is None:
            return
        img = frame
        if self._draw_boxes and boxes:
            img = self._draw_detection_boxes(frame, boxes)
        try:
            self._cam_handle.image = img
        except Exception:
            pass

    @staticmethod
    def _draw_detection_boxes(frame, boxes):
        """Draw labelled detection rectangles on a copy of `frame` (RGB u8)."""
        try:
            import cv2 as _cv2
        except Exception:
            return frame
        img = frame.copy()
        h, w = img.shape[:2]
        for b in boxes:
            xyxy = b.get("bbox_xyxy")
            if not xyxy or len(xyxy) < 4:
                continue
            x0, y0, x1, y1 = (int(round(v)) for v in xyxy[:4])
            x0 = max(0, min(w - 1, x0)); x1 = max(0, min(w - 1, x1))
            y0 = max(0, min(h - 1, y0)); y1 = max(0, min(h - 1, y1))
            col = _class_color(int(b.get("cls_id", 0)))   # RGB; img is RGB too
            _cv2.rectangle(img, (x0, y0), (x1, y1), col, 2)
            name = str(b.get("cls_name", b.get("cls_id", "")))
            conf = b.get("conf")
            label = f"{name} {conf:.2f}" if isinstance(conf, (int, float)) else name
            (tw, th), _bl = _cv2.getTextSize(label, _cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            ly = max(0, y0 - th - 4)
            _cv2.rectangle(img, (x0, ly), (x0 + tw + 2, ly + th + 4), col, -1)
            _cv2.putText(img, label, (x0 + 1, ly + th + 1),
                         _cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, _cv2.LINE_AA)
        return img

    def _clear_scene(self):
        with self._lock:
            for h in [self._gh, self._ph] + self._obj_handles:
                try:
                    if h is not None:
                        h.remove()
                except Exception:
                    pass
            self._gh = None
            self._ph = None
            self._raw.clear()
            self._frozen.clear()
            self._render.clear()
            self._obj_handles = []
        self._update_count()

    # convenience for run.py diagnostics
    @property
    def _clients(self):
        try:
            return self.server.get_clients()
        except Exception:
            return {}


def _class_color(cls_id: int):
    h = (cls_id * 47) % 360
    # simple HSL->RGB at S=0.7 L=0.6
    import colorsys
    r, g, b = colorsys.hls_to_rgb(h / 360.0, 0.6, 0.7)
    return (int(r * 255), int(g * 255), int(b * 255))


# ---------------------------------------------------------------------------
# standalone smoke test:  python scripts/server/viser_stream.py
# ---------------------------------------------------------------------------
def _smoketest():
    """Reproduces the production push pattern so you can eyeball that nothing
    disappears: each tick commits ONE new frozen KF (the map grows forever, ids
    are never deleted) AND re-pushes a single 'active' KF whose point count
    *changes* every tick (in-place update by id). If the add-or-update-by-id /
    never-delete model holds, the accumulated frozen KFs must stay rock-stable
    while the active blob pulses — no blanking when new KFs come in.
    """
    import time
    srv = ViserStreamServer(port=8765)
    srv.start()
    print("[smoketest] open http://localhost:8765/")
    rng = np.random.default_rng(0)
    ACTIVE_ID = 10_000           # one stable id, re-pushed (updated) every tick
    kf = 0
    try:
        while True:
            # 1. New frozen KF — accumulates, never removed.
            c = np.array([kf * 0.5, 0.0, 2.0], np.float32)
            xyz = (rng.normal(c, 0.25, size=(8000, 3))).astype(np.float32)
            rgb = (rng.random((8000, 3)) * 255).astype(np.uint8)
            srv.push({"type": "kf", "kf_id": kf, "xyz": xyz, "rgb": rgb,
                      "frozen": True})

            # 2. Active KF — SAME id, varying count -> tests in-place update.
            n_act = int(4000 + 4000 * abs(np.sin(kf * 0.5)))
            ca = np.array([kf * 0.5, 1.5, 2.0], np.float32)
            xa = (rng.normal(ca, 0.20, size=(n_act, 3))).astype(np.float32)
            ra = np.tile(np.array([255, 80, 80], np.uint8), (n_act, 1))
            srv.push({"type": "kf", "kf_id": ACTIVE_ID, "xyz": xa, "rgb": ra,
                      "frozen": False})

            srv.push({"type": "objects", "objects": [
                {"object_id": 0, "class": "car", "cls_id": 2,
                 "xyz": [kf * 0.5, 0.0, 2.0], "marker_radius": 0.4}]})

            # Camera PiP: a synthetic frame with a moving detection box, so the
            # GUI 'Camera' panel can be eyeballed alongside the 3D scene.
            try:
                import cv2 as _cv2, base64 as _b64
                fr = np.full((180, 320, 3), 40, np.uint8)
                fr[:, :, 1] = (40 + 40 * abs(np.sin(kf * 0.3))).astype(np.uint8)
                bx = int(20 + (kf * 13) % 220)
                _ok, _buf = _cv2.imencode('.jpg', fr[..., ::-1])
                if _ok:
                    srv.push({"type": "detections",
                              "frame_b64": _b64.b64encode(_buf.tobytes()).decode('ascii'),
                              "boxes": [{"cls_name": "car", "cls_id": 2, "conf": 0.9,
                                         "bbox_xyxy": [bx, 60, bx + 70, 130]}]})
            except Exception:
                pass
            print(f"[smoketest] frozen kf={kf} (+1)  active id={ACTIVE_ID} n={n_act}")
            kf += 1
            time.sleep(1.0)
    except KeyboardInterrupt:
        srv.stop()


if __name__ == "__main__":
    _smoketest()
