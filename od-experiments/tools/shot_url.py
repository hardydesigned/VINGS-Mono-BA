#!/usr/bin/env python3
"""Screenshot a live URL (e.g. the streaming viewer). Validated swiftshader recipe.
Usage: python tools/shot_url.py <url> <out.png> [wait_s]   (run with ulimit -s 1024)
"""
import sys, os
from playwright.sync_api import sync_playwright

SHOTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shots"))
os.makedirs(SHOTS, exist_ok=True)

def main():
    url = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "url.png"
    out = out if os.path.isabs(out) else os.path.join(SHOTS, out)
    wait_s = float(sys.argv[3]) if len(sys.argv) > 3 else 8.0
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True, args=[
            "--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu",
            "--use-gl=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist",
            "--single-process", "--no-zygote",
            "--js-flags=--max-old-space-size=512 --jitless",
            "--disable-extensions"])
        pg = b.new_page(viewport={"width": 1100, "height": 900}, device_scale_factor=1)
        errs = []
        pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
        pg.on("pageerror", lambda e: errs.append("PAGEERR " + str(e)))
        pg.goto(url, wait_until="load", timeout=60000)
        pg.wait_for_timeout(int(wait_s * 1000))   # let WS data arrive + reveal
        pg.screenshot(path=out)
        b.close()
    print("saved", out)
    for e in errs[:12]:
        print("  ERR", e[:200])

if __name__ == "__main__":
    main()
