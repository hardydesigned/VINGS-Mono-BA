#!/usr/bin/env python3
"""Render the viewer headless and screenshot it. Validated swiftshader recipe.

Usage: python tools/shot.py <view> [out.png]   view = top|persp|obl
Run with `ulimit -s 1024` so Chrome can reserve thread stacks under the strict
overcommit on this host.
"""
import sys, os, threading, functools, http.server, socketserver, time
from playwright.sync_api import sync_playwright

APP = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app"))
SHOTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shots"))
os.makedirs(SHOTS, exist_ok=True)

def serve(directory):
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=directory)
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
    httpd.allow_reuse_address = True
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, port

def main():
    view = sys.argv[1] if len(sys.argv) > 1 else "top"
    out = sys.argv[2] if len(sys.argv) > 2 else f"{view}.png"
    out = out if os.path.isabs(out) else os.path.join(SHOTS, out)
    extra = sys.argv[3] if len(sys.argv) > 3 else ""   # e.g. "sat=0&psize=2"
    W, H = 1100, 900

    httpd, port = serve(APP)
    url = f"http://127.0.0.1:{port}/index.html?view={view}"
    if extra:
        url += "&" + extra
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True, args=[
            "--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu",
            "--use-gl=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist",
            "--single-process", "--no-zygote",
            "--js-flags=--max-old-space-size=512 --jitless",
            "--disable-extensions", "--disable-background-networking"])
        pg = b.new_page(viewport={"width": W, "height": H}, device_scale_factor=1)
        errs = []
        pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
        pg.on("pageerror", lambda e: errs.append("PAGEERR " + str(e)))
        pg.goto(url, wait_until="load", timeout=60000)
        try:
            pg.wait_for_function("window.__ready === true", timeout=60000)
        except Exception:
            print("WARN: __ready timeout")
        pg.wait_for_timeout(1500)   # let textures/models settle + a few frames
        pg.screenshot(path=out)
        b.close()
    httpd.shutdown()
    print("saved", out)
    if errs:
        print("CONSOLE ERRORS:")
        for e in errs[:15]:
            print("  ", e[:200])

if __name__ == "__main__":
    main()
