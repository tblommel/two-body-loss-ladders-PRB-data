"""fig_plaq: output/ -> data/curves.csv (N, D = sum_r D_r, <S^2> versus t for the four panels, all methods + exact).
usage: python postprocess.py [--output DIR] [--data DIR] [--allow-missing]"""
import argparse
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import readers as R, io as IO                 # noqa: E402


def _check(rows, out, allow):
    bad = [r["run_id"] for r in rows
           if R.exit_code(out, r["run_id"]) not in (R.FINISHED_DRIVER if r["kind"] == "driver" else {0})]
    if bad and not allow:
        sys.exit("missing/unfinished runs:\n  " + "\n  ".join(bad))
    return set(bad)


def _build(output_dir, allow_missing=False):
    out = Path(output_dir); rows = R.read_index(HERE); skip = _check(rows, out, allow_missing)
    prod = {"dp5drive": [], "code_exact": []}
    cols = {k: [] for k in ("method", "ic", "gamma", "status", "t", "N", "D", "S2")}
    tex = {}
    for r in sorted(rows, key=lambda r: r["kind"] == "driver"):          # exact first (reference grid)
        if r["run_id"] in skip:
            continue
        key = (r["ic"], r["gamma"])
        if r["kind"] == "exact":
            e = R.load_exact(out / r["run_id"] / "result.h5"); prod["code_exact"].append(e["version"])
            tr = e["transient"]; t = tr["t"]; tex[key] = t
            v = dict(N=tr["N"], D=tr["D"].sum(axis=1), S2=tr["S2"]); st = "Ok"
        else:
            d = R.load(out / r["run_id"] / "out.h5"); prod["dp5drive"].append(d["version"])
            t = d["t"]; nt = len(t)
            assert np.allclose(t, tex[key][:nt], rtol=0, atol=1e-9)
            v = dict(N=d["N"], D=d["D"], S2=d["S2"]); st = d["status"]
        n = len(t)
        cols["method"] += [r["method"]] * n; cols["ic"] += [r["ic"]] * n; cols["gamma"] += [r["gamma"]] * n
        cols["status"] += [st] * n; cols["t"] += list(t)
        for q in ("N", "D", "S2"):
            cols[q] += list(v[q])
    return {"curves.csv": {k: np.array(v) for k, v in cols.items()}}, prod


def build(output_dir, allow_missing=False):
    """output/ -> {data file name: content} (pure, no writing)"""
    return _build(output_dir, allow_missing)[0]


DESC = {"curves.csv": {"description": "plaquette trajectories of the four panels (frames every 0.1/J, t <= 200/J); "
                                      "mGW <S^2> uses the driver's mean-field two-body density matrix (not plotted)",
                       "columns": {"method": "exact or data label", "ic": "initial state", "gamma": "gamma [J]",
                                   "status": "driver status (exact: Ok)", "t": "time [1/J]",
                                   "N": "total particle number [1]", "D": "total double occupancy sum_r D_r [1]",
                                   "S2": "total spin <S^2> [1]"}}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(HERE / "output")); ap.add_argument("--data", default=str(HERE / "data"))
    ap.add_argument("--allow-missing", action="store_true")
    a = ap.parse_args()
    D, prod = _build(a.output, a.allow_missing)
    IO.write_data(a.data, D, "fig_plaq/postprocess.py", DESC, prod)
    print("wrote", ", ".join(D), "->", a.data)


if __name__ == "__main__":
    main()
