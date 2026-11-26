"""Auto-screenshot a URL.
Setup once:  pip install playwright  &&  playwright install chromium

Runs Playwright in a separate process (safe inside Streamlit on Windows).
Use take_screenshot(url) from the app, or:  python screenshot.py <url> <out.png>
"""
import subprocess
import sys
import tempfile
from pathlib import Path


def take_screenshot(url: str, out_path: str | None = None, timeout_s: int = 40) -> str:
    out = out_path or str(Path(tempfile.gettempdir()) / "p360_shot.png")
    Path(out).unlink(missing_ok=True)
    r = subprocess.run([sys.executable, __file__, url, out],
                       capture_output=True, text=True, timeout=timeout_s)
    if r.returncode != 0 or not Path(out).exists():
        raise RuntimeError((r.stderr or r.stdout or "screenshot failed").strip()[-300:])
    return out


if __name__ == "__main__":
    from playwright.sync_api import sync_playwright
    url, out = sys.argv[1], sys.argv[2]
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    with sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page(viewport={"width": 1280, "height": 800})
        page.goto(url, timeout=15000, wait_until="load")
        page.screenshot(path=out)
        b.close()
