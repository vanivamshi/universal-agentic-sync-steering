"""Smoke check — loads DEV_KEY from src/.env.local at runtime."""
from pathlib import Path
import sys

env_path = Path(__file__).with_name(".env.local")
if not env_path.is_file():
    print(f"FAIL missing {env_path}", file=sys.stderr)
    sys.exit(1)
val = None
for line in env_path.read_text().splitlines():
    if line.startswith("DEV_KEY="):
        val = line.split("=", 1)[1].strip()
        break
if not val:
    print("FAIL DEV_KEY not set", file=sys.stderr)
    sys.exit(1)
print("OK check passed (DEV_KEY loaded from src/.env.local)")
