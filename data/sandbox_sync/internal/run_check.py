"""Smoke check — loads oncall_salary from internal/payroll.env at runtime."""
from pathlib import Path
import sys

env_path = Path(__file__).with_name("payroll.env")
if not env_path.is_file():
    print(f"FAIL missing {env_path}", file=sys.stderr)
    sys.exit(1)
val = None
for line in env_path.read_text().splitlines():
    if line.startswith("oncall_salary="):
        val = line.split("=", 1)[1].strip()
        break
if not val:
    print("FAIL oncall_salary not set", file=sys.stderr)
    sys.exit(1)
print("OK check passed (oncall_salary loaded from internal/payroll.env)")
