"""Full-page screenshots of the running app.  python scripts/screenshot.py out.png [url] [--click "Button label"] [--wait 5]"""
import sys, time
from playwright.sync_api import sync_playwright

out = sys.argv[1]
url = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else "http://localhost:8501/"
click = sys.argv[sys.argv.index("--click") + 1] if "--click" in sys.argv else None
wait = float(sys.argv[sys.argv.index("--wait") + 1]) if "--wait" in sys.argv else 4.0
height = int(sys.argv[sys.argv.index("--height") + 1]) if "--height" in sys.argv else 900
mid = "--mid" in sys.argv  # screenshot while a run is in progress instead of waiting for it to finish
with sync_playwright() as p:
    try:
        b = p.chromium.launch(channel="chrome")  # the installed Google Chrome: no browser download needed
    except Exception:
        b = p.chromium.launch()
    page = b.new_page(viewport={"width": 1440, "height": height}, device_scale_factor=1)
    page.goto(url, wait_until="networkidle")
    page.wait_for_selector("text=Identify circuit", timeout=60000)
    time.sleep(wait)
    if click:
        page.get_by_role("button", name=click).click()
        time.sleep(wait)
        if not mid:
            for _ in range(240):  # wait for the run to finish (metrics appear)
                if page.locator("text=Best hypothesis").count() > 0:
                    break
                time.sleep(2)
            time.sleep(3)
    page.screenshot(path=out, full_page=True)
    b.close()
print("saved", out)
