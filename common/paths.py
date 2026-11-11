"""Release root, version string and default executables (same rules as common/run_cards.sh)."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # release root (holds VERSION)
CODE_DIAG = ROOT / "code_diagrammatics"
CODE_EXACT = ROOT / "code_exact"
for _p in (CODE_DIAG / "python", CODE_EXACT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def version():
    try:
        return (ROOT / "VERSION").read_text().strip()
    except OSError:
        return "unknown"


def version_string():
    """<VERSION>+<git describe --always --dirty>, or <VERSION>+nogit outside a git checkout"""
    try:
        g = subprocess.run(["git", "describe", "--always", "--dirty"], cwd=ROOT, capture_output=True, text=True,
                           timeout=20)
        d = g.stdout.strip() if g.returncode == 0 and g.stdout.strip() else "nogit"
    except Exception:
        d = "nogit"
    return f"{version()}+{d}"


def dp5drive():
    return Path(os.environ.get("DP5DRIVE", ROOT / "code_diagrammatics" / "build" / "dp5drive"))


def python():
    return os.environ.get("PYTHON", "python3")


def run_exact():
    return ROOT / "code_exact" / "run_exact.py"
