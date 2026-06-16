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
_DROPPABLE = {"replace_active", "replace_all", "objects"}


def _frame(payload: dict):
    """Turn a push() payload dict into a sendable WebSocket message.

    Returns (message, droppable, backlog_kind) where message is ``bytes`` (binary
    splat frame) or ``str`` (text JSON), and backlog_kind is one of
    ``frozen`` | ``active`` | ``objects`` | ``resync`` | None.
    """
    t = payload["type"]
    if t in ("append_frozen", "replace_active", "replace_all"):
        data = payload.get("data", b"")
        header = {"type": t, "epoch": int(payload.get("epoch", 0)),
                  "n": len(data) // 32}
        if "kf_id" in payload:
            header["kf_id"] = int(payload["kf_id"])
        hb = json.dumps(header).encode("utf-8")
        msg = struct.pack("<I", len(hb)) + hb + data
        kind = "frozen" if t == "append_frozen" else "active"
        return msg, (t in _DROPPABLE), kind
    else:  # objects / resync -> text frame
        msg = json.dumps({k: v for k, v in payload.items() if k != "data"})
        kind = "objects" if t == "objects" else ("resync" if t == "resync" else None)
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
        self._q: queue.Queue = queue.Queue(maxsize=max(2, int(max_queue)))
        self._clients: set = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._stop = threading.Event()

        # backlog for late joiners (guarded by a plain lock: written from the
        # run thread via push(), read from the asyncio thread on connect)
        self._lock = threading.Lock()
        self._frozen_backlog: list = []   # list[bytes]   (append_frozen messages)
        self._last_active = None          # bytes | None
        self._last_objects = None         # str | None

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
            elif kind == "active":
                self._last_active = msg
            elif kind == "objects":
                self._last_objects = msg
            elif kind == "resync":
                # frozen set invalidated (loop closure / new epoch) -> drop backlog
                self._frozen_backlog.clear()
                self._last_active = None
                self._last_objects = None

        # enqueue for broadcast
        try:
            self._q.put_nowait(msg)
        except queue.Full:
            if droppable:
                # evict one item (drop-oldest) and retry; latest active/objects wins
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._q.put_nowait(msg)
                except queue.Full:
                    pass
            else:
                # frozen/resync must not be lost: brief bounded wait, then give up
                try:
                    self._q.put(msg, timeout=0.5)
                except queue.Full:
                    print("[stream] WARN dropped non-droppable frame (slow client)")

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

    async def _process_request(self, path, request_headers):
        upgrade = ""
        try:
            upgrade = (request_headers.get("Upgrade") or "").lower()
        except Exception:
            pass
        if upgrade == "websocket":
            return None  # echte WS-Verbindung: Handshake fortsetzen

        # statische Datei ausliefern (whitelist, kein Path-Traversal)
        name = (path.split("?")[0].lstrip("/") or "viewer.html")
        if name in (".", ""):
            name = "viewer.html"
        if "/" in name or "\\" in name or name.startswith("."):
            return (http.HTTPStatus.FORBIDDEN, [], b"forbidden")
        fpath = os.path.join(self._static_dir, name)
        if not os.path.isfile(fpath):
            return (http.HTTPStatus.NOT_FOUND, [],
                    f"not found: {name}".encode())
        with open(fpath, "rb") as f:
            body = f.read()
        ctype = self._CONTENT_TYPES.get(os.path.splitext(name)[1], "application/octet-stream")
        headers = [("Content-Type", ctype), ("Content-Length", str(len(body))),
                   ("Cache-Control", "no-cache")]
        return (http.HTTPStatus.OK, headers, body)

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
                last_active, last_objects = self._last_active, self._last_objects
            for m in backlog:
                await ws.send(m)
            if last_active is not None:
                await ws.send(last_active)
            if last_objects is not None:
                await ws.send(last_objects)
            # we don't consume client input (yet); just hold the connection
            async for _ in ws:
                pass
        except Exception:
            pass
        finally:
            self._clients.discard(ws)

    async def _broadcaster(self):
        while not self._stop.is_set():
            msg = await self._loop.run_in_executor(None, self._q_get)
            if msg is None:
                continue
            if not self._clients:
                continue
            results = await asyncio.gather(
                *(ws.send(msg) for ws in list(self._clients)),
                return_exceptions=True)
            for ws, r in zip(list(self._clients), results):
                if isinstance(r, Exception):
                    self._clients.discard(ws)

    def _q_get(self):
        try:
            return self._q.get(timeout=0.2)
        except queue.Empty:
            return None


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
