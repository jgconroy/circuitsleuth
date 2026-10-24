"""Headless Streamlit smoke test: default run + precomputed load. Exit code 1 on any exception."""
import sys
from streamlit.testing.v1 import AppTest

at = AppTest.from_file("app.py", default_timeout=120)
at.run()
errs = [str(e.value)[:200] for e in at.exception]
load = [b for b in at.button if b.label.startswith("Load precomputed")][0]
load.click().run(timeout=300)
errs += [str(e.value)[:200] for e in at.exception]
print("exceptions:", errs or "none", "| metrics:", [(m.label, m.value) for m in at.metric][:2])
sys.exit(1 if errs else 0)
