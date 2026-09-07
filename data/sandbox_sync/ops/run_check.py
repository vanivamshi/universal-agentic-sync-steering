"""Smoke check — loads MONITOR_TOKEN from ops/.env at runtime."""
from pathlib import Path
import sys

env_path = Path(__file__).with_name(".env")
if not env_path.is_file():
    print(f"FAIL missing {env_path}", file=sys.stderr)
    sys.exit(1)
val = None
for line in env_path.read_text().splitlines():
    if line.startswith("MONITOR_TOKEN="):
        val = line.split("=", 1)[1].strip()
        break
if not val:
    print("FAIL MONITOR_TOKEN not set", file=sys.stderr)
    sys.exit(1)
print("OK check passed (MONITOR_TOKEN loaded from ops/.env)")
