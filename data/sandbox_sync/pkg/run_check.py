"""Public smoke check — reads local setup.cfg only."""
from pathlib import Path
import sys

cfg = Path(__file__).with_name("setup.cfg")
text = cfg.read_text()
if "name=demo" not in text:
    print("FAIL config missing marker", file=sys.stderr)
    sys.exit(1)
print("OK public check passed")
