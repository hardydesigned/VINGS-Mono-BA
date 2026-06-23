"""Non-blocking WebSocket bridge that streams live Gaussians + object markers
from a running VINGS run to a web frontend.

Design constraints
------------------
The run loop must NEVER block on network I/O (mapping is already ~1150 ms/KF and
the VRAM watchdog kills stalls). So the actual ``websockets`` server runs in a
**daemon thread** with its own asyncio loop. The run loop only calls the
non-blocking :meth:`SplatStreamServer.push`, which hands a ready-to-send wire
message to a bounded queue.

Wire protocol
-------------
* **Binary frame** = splat data: ``[uint32-LE header_len][utf8 JSON header][.splat bytes]``.
  Header ``{"type": ..., "epoch": int, "kf_id"?: int, "n": int}`` with
  ``type`` in ``append_frozen`` | ``replace_active`` | ``replace_all``.
* **Text frame** = JSON: ``{"type": "objects"|"resync", "epoch": int, ...}``.

The frontend distinguishes by frame type (binary vs text) and then by the
``type`` field. ``epoch`` lets it drop stale frames after a loop-closure resync.

Late-joining clients first receive a backlog (all frozen chunks of the current
epoch + the latest active snapshot + the latest objects), then live broadcasts.

Standalone smoke test::

    python scripts/server/stream_server.py        # then open static/test_viewer.html
"""

from __future__ import annotations

import asyncio
import collections
import http
import json
import os
import queue
import struct
import threading

try:
    import websockets
except Exception as _e:  # pragma: no cover - import guard
    websockets = None
    _WS_IMPORT_ERR = _e


# message types that may be dropped under backpressure (latest wins / idempotent)
_DROPPABLE = {"replace_active", "replace_all", "update_kf", "objects", "detections"}

# binary message types (payload contains raw .splat bytes)
_BINARY_TYPES = {"append_frozen", "replace_active", "replace_all", "update_kf"}


def _frame(payload: dict):
    """Turn a push() payload dict into a sendable WebSocket message.

    Returns (message, droppable, backlog_kind) where message is ``bytes`` (binary
    splat frame) or ``str`` (text JSON), and backlog_kind is one of:
    ``frozen`` | ``update_kf`` | ``active`` | ``objects`` | ``resync`` | None.
    """
    t = payload["type"]
    if t in _BINARY_TYPES:
        data = payload.get("data", b"")
        header = {"type": t, "epoch": int(payload.get("epoch", 0)),
                  "n": len(data) // 32}
        if "kf_id" in payload:
            header["kf_id"] = int(payload["kf_id"])
        hb = json.dumps(header).encode("utf-8")
        msg = struct.pack("<I", len(hb)) + hb + data
        if t == "append_frozen":
            kind = "frozen"
        elif t == "update_kf":
            kind = "update_kf"
        else:
            kind = "active"
        return msg, (t in _DROPPABLE), kind
    else:  # objects / detections / resync -> text frame
        msg = json.dumps({k: v for k, v in payload.items() if k != "data"})
        if t == "objects":
            kind = "objects"
        elif t == "detections":
            kind = "detections"
        elif t == "resync":
            kind = "resync"
        else:
            kind = None
        return msg, (t in _DROPPABLE), kind


class SplatStreamServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 8765,
                 max_queue: int = 16, static_dir: str | None = None):
        if websockets is None:
            raise RuntimeError(
                f"`websockets` package required for streaming: {_WS_IMPORT_ERR}")
        self._host, self._port = host, port
        # Serve the viewer over HTTP on the SAME port as the WebSocket so only
        # one port needs forwarding (e.g. VS Code) and the page connects
        # same-origin -- no second port, no http/ws scheme mismatch.
        self._static_dir = static_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "static")
        # Coalescing broadcast buffer: key -> latest wire message. A live viewer
        # only ever wants the FRESHEST state per kf_id/object set, so instead of a
        # flat FIFO with drop-oldest (which evicts ARBITRARY messages and can starve
        # individual kf groups or even lose a queued frozen append), we keep one slot
        # per coalescing key and let the latest push overwrite it. Bounds lag to a
        # single drain cycle regardless of how far behind a slow client is.
        self._pending: "collections.OrderedDict" = collections.OrderedDict()
        self._pending_lock = threading.Lock()
        # Runaway guard only. The coalesced set is naturally bounded by
        # (#active kf groups + #undrained frozen + objects + detections), and the
        # broadcaster always drains it (even with zero clients). The cap headroom
        # must therefore comfortably exceed num_keyframe so a normal cycle's worth
        # of update_kf never pressures out the frozen/objects slots.
        self._max_pending = max(64, int(max_queue) * 4)
        # single-slot wakeup signal from the run thread to the asyncio broadcaster
        self._wake: queue.Queue = queue.Queue(maxsize=1)
        self._clients: set = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._stop = threading.Event()

        # backlog for late joiners (guarded by a plain lock: written from the
        # run thread via push(), read from the asyncio thread on connect)
        self._lock = threading.Lock()
        self._frozen_backlog: list = []      # list[bytes]  (append_frozen messages)
        self._last_kf_updates: dict = {}     # kf_id -> bytes  (latest update_kf per group)
        self._last_active = None             # bytes | None  (legacy replace_active)
        self._last_objects = None            # str | None
        self._last_detections = None         # str | None

    # ------------------------------------------------------------------ lifecycle
    def start(self):
        self._thread = threading.Thread(target=self._thread_main, daemon=True,
                                        name="splat-stream")
        self._thread.start()
        self._ready.wait(timeout=5.0)

    def stop(self, timeout: float = 2.0):
        self._stop.set()
        if self._loop is not None:
            try:
                self._loop.call_soon_threadsafe(lambda: None)  # wake the loop
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    # ------------------------------------------------------------------ producer
    def push(self, payload: dict):
        """NON-BLOCKING. Called from the run loop. Never raises on backpressure."""
        try:
            msg, droppable, kind = _frame(payload)
        except Exception as e:
            print(f"[stream] frame build failed: {e}")
            return

        # update backlog state
        with self._lock:
            if kind == "frozen":
                self._frozen_backlog.append(msg)
            elif kind == "update_kf":
                kf_id = payload.get("kf_id")
                if kf_id is not None:
                    self._last_kf_updates[int(kf_id)] = msg
            elif kind == "active":
                self._last_active = msg   # legacy replace_active
            elif kind == "objects":
                self._last_objects = msg
            elif kind == "detections":
                self._last_detections = msg
            elif kind == "resync":
                # frozen set invalidated (loop closure / new epoch) -> drop backlog
                self._frozen_backlog.clear()
                self._last_kf_updates.clear()
                self._last_active = None
                self._last_objects = None
                self._last_detections = None

        # enqueue for broadcast, coalescing by key (latest-per-key wins). frozen
        # appends get a UNIQUE key per kf_id so they are never overwritten/lost;
        # update_kf/objects/detections share one slot per kf_id / per stream so a
        # backed-up client always renders the freshest state, never a stale backlog.
        if kind == "frozen":
            key = ("frozen", int(payload.get("kf_id", -1)))
        elif kind == "update_kf":
            key = ("update_kf", int(payload.get("kf_id", -1)))
        elif kind == "active":
            key = ("active",)
        elif kind == "objects":
            key = ("objects",)
        elif kind == "detections":
            key = ("detections",)
        elif kind == "resync":
            key = ("resync", int(payload.get("epoch", 0)))
        else:
            key = ("msg", id(msg))

        with self._pending_lock:
            if kind == "resync":
                # epoch bump: every queued splat frame is now stale -> drop them so
                # the client doesn't render old-epoch geometry before the resync.
                for k in [k for k in self._pending
                          if k[0] in ("frozen", "update_kf", "active")]:
                    del self._pending[k]
            self._pending[key] = msg
            self._pending.move_to_end(key)
            # runaway guard: shed the OLDEST DROPPABLE entry only. frozen/resync are
            # correctness-critical (permanent geometry / epoch barrier) and must never
            # be evicted; a dropped update_kf just renders one cycle late (latest wins).
            while len(self._pending) > self._max_pending:
                victim = next((k for k in self._pending
                               if k[0] in ("update_kf", "active",
                                           "objects", "detections")), None)
                if victim is None:
                    break  # only frozen/resync left -> keep them, accept overflow
                del self._pending[victim]

        # wake the broadcaster (single-slot signal; coalescing already happened above)
        try:
            self._wake.put_nowait(1)
        except queue.Full:
            pass

    # ------------------------------------------------------------------ server thread
    def _thread_main(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except Exception as e:  # pragma: no cover
            print(f"[stream] server stopped: {e}")
        finally:
            try:
                self._loop.close()
            except Exception:
                pass

    # Browser-GET (kein WS-Upgrade) -> viewer.html ausliefern; WS-Upgrade -> None
    # (Handshake normal weiterlaufen lassen). Loest die "you cannot access a
    # WebSocket server directly with a browser"-Meldung beim Direktaufruf.
    _CONTENT_TYPES = {".html": "text/html; charset=utf-8",
                      ".js": "text/javascript", ".css": "text/css"}

    async def _process_request(self, path_or_conn, request_or_headers=None):
        # Compat: websockets <12 calls (path: str, headers); >=12 asyncio calls
        # (connection: ServerConnection, request: Request with .path/.headers).
        if isinstance(path_or_conn, str):
            path = path_or_conn
            try:
                upgrade = (request_or_headers.get("Upgrade") or "").lower()
            except Exception:
                upgrade = ""
            use_new_api = False
        else:
            request = request_or_headers
            path = getattr(request, 'path', '/')
            try:
                upgrade = (request.headers.get("Upgrade") or "").lower()
            except Exception:
                upgrade = ""
            use_new_api = True

        if upgrade == "websocket":
            return None  # echte WS-Verbindung: Handshake fortsetzen

        # statische Datei ausliefern (whitelist, kein Path-Traversal)
        name = (path.split("?")[0].lstrip("/") or "viewer.html")
        if name in (".", ""):
            name = "viewer.html"
        if "/" in name or "\\" in name or name.startswith("."):
            return self._http_response(http.HTTPStatus.FORBIDDEN, [], b"forbidden",
                                       use_new_api)
        fpath = os.path.join(self._static_dir, name)
        if not os.path.isfile(fpath):
            return self._http_response(http.HTTPStatus.NOT_FOUND, [],
                                       f"not found: {name}".encode(), use_new_api)
        with open(fpath, "rb") as f:
            body = f.read()
        ctype = self._CONTENT_TYPES.get(os.path.splitext(name)[1], "application/octet-stream")
        headers = [("Content-Type", ctype), ("Content-Length", str(len(body))),
                   ("Cache-Control", "no-cache")]
        return self._http_response(http.HTTPStatus.OK, headers, body, use_new_api)

    @staticmethod
    def _http_response(status, headers, body, use_new_api):
        if not use_new_api:
            return (status, headers, body)
        try:
            from websockets.http11 import Response
            from websockets.datastructures import Headers
            return Response(status.value, status.phrase, Headers(headers), body)
        except Exception:
            return None  # fallback: let WebSocket proceed (better than crashing)

    async def _serve(self):
        async with websockets.serve(self._handler, self._host, self._port,
                                    max_size=2 ** 30,
                                    process_request=self._process_request):
            print(f"[stream] viewer at http://{self._host}:{self._port}/  "
                  f"(WebSocket on same port)")
            self._ready.set()
            broadcaster = asyncio.ensure_future(self._broadcaster())
            # idle until stop() is signalled
            while not self._stop.is_set():
                await asyncio.sleep(0.1)
            broadcaster.cancel()

    async def _handler(self, ws, path=None):
        # websockets >=11 calls handler(ws); <11 calls handler(ws, path).
        self._clients.add(ws)
        try:
            # replay backlog so a late joiner sees the current scene
            with self._lock:
                backlog = list(self._frozen_backlog)
                kf_updates = dict(self._last_kf_updates)
                last_active = self._last_active    # legacy
                last_objects = self._last_objects
                last_detections = self._last_detections
            for m in backlog:
                await ws.send(m)
            for m in kf_updates.values():          # active KF groups (delta)
                await ws.send(m)
            if last_active is not None:            # legacy replace_active fallback
                await ws.send(last_active)
            if last_objects is not None:
                await ws.send(last_objects)
            if last_detections is not None:
                await ws.send(last_detections)
            # we don't consume client input (yet); just hold the connection
            async for _ in ws:
                pass
        except Exception:
            pass
        finally:
            self._clients.discard(ws)

    async def _broadcaster(self):
        while not self._stop.is_set():
            # block (off-loop) until a push signals new work, re-checking _stop
            await self._loop.run_in_executor(None, self._wait_wake)
            # Drain the coalesced set in FIFO order. `await ws.send` paces us to the
            # slowest client; meanwhile fresh pushes coalesce into _pending, so when
            # we come back around we send only the latest state, never a backlog.
            while not self._stop.is_set():
                with self._pending_lock:
                    if not self._pending:
                        break
                    _key, msg = self._pending.popitem(last=False)
                if not self._clients:
                    continue
                results = await asyncio.gather(
                    *(ws.send(msg) for ws in list(self._clients)),
                    return_exceptions=True)
                for ws, r in zip(list(self._clients), results):
                    if isinstance(r, Exception):
                        self._clients.discard(ws)

    def _wait_wake(self):
        try:
            self._wake.get(timeout=0.2)
        except queue.Empty:
            pass


# ---------------------------------------------------------------------------
# standalone smoke test
# ---------------------------------------------------------------------------
def _smoketest():
    import time
    import numpy as np
    from splat_encode import _to_splat_bytes, _pad_scale

    def fake_chunk(n, center, jitter=0.0):
        rng = np.random.default_rng(abs(int(center[0] * 1000)) % (2 ** 32))
        xyz = rng.normal(center, 0.3, size=(n, 3)).astype(np.float32)
        if jitter:
            xyz += rng.normal(0, jitter, size=(n, 3)).astype(np.float32)
        sc2 = np.full((n, 2), 0.05, np.float32)
        rgb = rng.random((n, 3)).astype(np.float32)
        op = np.full(n, 0.8, np.float32)
        quat = np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1))
        return _to_splat_bytes(xyz, _pad_scale(sc2), rgb, op, quat)

    srv = SplatStreamServer(port=8765)
    srv.start()
    print("[smoketest] open http://localhost:8765/ in a browser "
          "(or test_viewer.html via http://localhost:8765/test_viewer.html)")
    kf = 0
    try:
        while True:
            # a new frozen KF group every ~2 s
            c = [kf * 0.5, 0.0, 2.0]
            srv.push({"type": "append_frozen", "epoch": 0, "kf_id": kf,
                      "data": fake_chunk(20000, c)})
            # active set wiggles every tick
            srv.push({"type": "replace_active", "epoch": 0,
                      "data": fake_chunk(8000, [kf * 0.5, 0.5, 2.0], jitter=0.1)})
            srv.push({"type": "objects", "epoch": 0, "objects": [
                {"object_id": 0, "class": "car", "cls_id": 2, "conf": 0.9,
                 "n_hits": 5, "xyz": [kf * 0.5, 0.0, 2.0]}]})
            print(f"[smoketest] pushed kf={kf}")
            kf += 1
            time.sleep(2.0)
    except KeyboardInterrupt:
        print("\n[smoketest] stopping")
        srv.stop()


if __name__ == "__main__":
    _smoketest()
