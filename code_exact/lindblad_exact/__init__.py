"""lindblad_exact: exact Lindblad solver for the PRB1 reproduction release (sector-restricted sparse superoperators).

Entry point: code_exact/run_exact.py (card.json -> result.h5). Self-contained: imports only numpy, scipy, h5py and
the vendored code_exact/rdm_tools.py."""
import sys as _sys
from pathlib import Path as _Path

_ROOT = str(_Path(__file__).resolve().parents[1])          # code_exact/ (holds rdm_tools.py)
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

RESULT_SCHEMA = "exact_result/1"
