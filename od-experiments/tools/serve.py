#!/usr/bin/env python3
"""Serve the viewer for interactive use:  python tools/serve.py [port]
Then open http://127.0.0.1:8000/index.html  (fetch() needs http, not file://)."""
import sys, os, functools, http.server, socketserver
APP = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app"))
port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=APP)
print(f"serving {APP} at http://127.0.0.1:{port}/index.html")
socketserver.TCPServer(("127.0.0.1", port), handler).serve_forever()
