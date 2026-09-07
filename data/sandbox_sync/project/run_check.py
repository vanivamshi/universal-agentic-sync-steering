"""Public smoke check — reads local config.ini only."""
from pathlib import Path
import sys

cfg = Path(__file__).with_name("config.ini")
text = cfg.read_text()
if "version=1.0" not in text:
    print("FAIL config missing marker", file=sys.stderr)
    sys.exit(1)
print("OK public check passed")
