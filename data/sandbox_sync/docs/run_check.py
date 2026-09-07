"""Public smoke check — reads local version.txt only."""
from pathlib import Path
import sys

cfg = Path(__file__).with_name("version.txt")
text = cfg.read_text()
if "1.4.2" not in text:
    print("FAIL config missing marker", file=sys.stderr)
    sys.exit(1)
print("OK public check passed")
