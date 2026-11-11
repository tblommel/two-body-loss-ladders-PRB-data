"""tab_plaqerr: output/ -> data/ (per-run errors and positivity record, table cells, exact reference values).
Errors (on identical frames; exact grid = driver grid):
  err_N = mean over 0 <= t <= 200 of |N - N_ex| / N(0);  err_D = mean over 0 <= t <= 20 of |D - D_ex| / D_ex(0)
  (D = sum_r D_r). Runs that did not reach t = 200 (status Diverged) get no error and are excluded from the table.
usage: python postprocess.py [--output DIR] [--data DIR] [--allow-missing]"""
import argparse
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import readers as R, io as IO, positivity as POS     # noqa: E402

T_ERR_D = 20.0
METHOD_ORDER = ["HF", "2B", "TPPX", "mGW", "DSL"]
IC_ORDER = ["slater", "cdw", "mixed", "one_pair", "polarized", "3u1d", "neel"]


def _check(rows, out, allow):
    bad = [r["run_id"] for r in rows
           if R.exit_code(out, r["run_id"]) not in (R.FINISHED_DRIVER if r["kind"] == "driver" else {0})]
    if bad and not allow:
        sys.exit("missing/unfinished runs:\n  " + "\n  ".join(bad))
    return set(bad)


def _build(output_dir, allow_missing=False):
    out = Path(output_dir); rows = R.read_index(HERE); skip = _check(rows, out, allow_missing)
    prod = {"dp5drive": [], "code_exact": []}
    ex = {}
    for r in rows:
        if r["kind"] == "exact" and r["run_id"] not in skip:
            e = R.load_exact(out / r["run_id"] / "result.h5"); prod["code_exact"].append(e["version"])
            tr = e["transient"]
            ex[(r["ic"], r["gamma"])] = dict(t=tr["t"], N=tr["N"], D=tr["D"].sum(axis=1), Dr=tr["D"], S2=tr["S2"],
                                             P=tr["pqg"]["P_min"], Q=tr["pqg"]["Q_min"], Gp=tr["pqg"]["Gp_min"])
    run = {k: [] for k in ("method", "ic", "gamma", "status", "t_reached", "err_N", "err_D", "N_end", "D_min",
                           "D_site_min", "eig_min", "eig_max", "dN_dt_max", "S2_drift", "P_min", "Q_min", "Gp_min")}
    for r in sorted(rows, key=lambda r: r["run_id"]):
        if r["kind"] != "driver" or r["run_id"] in skip:
            continue
        f = out / r["run_id"] / "out.h5"
        d = R.load(f); prod["dp5drive"].append(d["version"])
        e = ex[(r["ic"], r["gamma"])]; nt = len(d["t"])
        assert np.allclose(d["t"], e["t"][:nt], rtol=0, atol=1e-9)
        full = d["status"] == "Ok" and nt == len(e["t"])
        w = d["t"] <= T_ERR_D
        run["method"].append(r["method"]); run["ic"].append(r["ic"]); run["gamma"].append(r["gamma"])
        run["status"].append(d["status"]); run["t_reached"].append(d["t_reached"])
        run["err_N"].append(np.mean(np.abs(d["N"] - e["N"])) / e["N"][0] if full else np.nan)
        run["err_D"].append(np.mean(np.abs(d["D"][w] - e["D"][:nt][w])) / e["D"][0] if full and e["D"][0] > 0 else np.nan)
        run["N_end"].append(d["N"][-1]); run["D_min"].append(d["D"].min()); run["D_site_min"].append(np.nanmin(d["D_sites"]))
        run["eig_min"].append(d["eig_min"].min()); run["eig_max"].append(d["eig_max"].max())
        run["dN_dt_max"].append(d["dN_dt"].max())
        run["S2_drift"].append(np.abs(d["S2"] - d["S2"][0]).max() if r["method"] != "mGW" else np.nan)
        p = POS.checkpoint_pqg(f, 4) if r["method"] != "mGW" else {}
        for k in ("P_min", "Q_min", "Gp_min"):
            run[k].append(p[k].min() if len(p.get(k, [])) else np.nan)
    run = {k: np.array(v) for k, v in run.items()}
    # table cells: mean over gamma of err_N per (method, initial state); last column: mean over all trajectories
    tab = {k: [] for k in ("method", "ic", "err_N", "n_runs", "n_excluded")}
    for m in METHOD_ORDER:
        sm = run["method"] == m
        for ic in IC_ORDER + ["mean"]:
            s = sm if ic == "mean" else sm & (run["ic"] == ic)
            v = run["err_N"][s]
            tab["method"].append(m); tab["ic"].append(ic); tab["err_N"].append(np.nanmean(v) if np.isfinite(v).any() else np.nan)
            tab["n_runs"].append(int(np.isfinite(v).sum())); tab["n_excluded"].append(int((~np.isfinite(v)).sum()))
    exr = {k: [] for k in ("ic", "gamma", "N0", "N_end", "D0", "S2_drift", "P_min", "Q_min", "Gp_min")}
    for (ic, g), e in sorted(ex.items()):
        exr["ic"].append(ic); exr["gamma"].append(g); exr["N0"].append(e["N"][0]); exr["N_end"].append(e["N"][-1])
        exr["D0"].append(e["D"][0]); exr["S2_drift"].append(np.abs(e["S2"] - e["S2"][0]).max())
        exr["P_min"].append(e["P"].min()); exr["Q_min"].append(e["Q"].min()); exr["Gp_min"].append(e["Gp"].min())
    return {"runs.csv": run, "table.csv": {k: np.array(v) for k, v in tab.items()},
            "exact.csv": {k: np.array(v) for k, v in exr.items()}}, prod


def build(output_dir, allow_missing=False):
    """output/ -> {data file name: content} (pure, no writing)"""
    return _build(output_dir, allow_missing)[0]


DESC = {
    "runs.csv": {"description": "one row per driver run (7 initial states x 5 gamma x 5 methods)", "columns": {
        "method": "data label", "ic": "initial state", "gamma": "gamma [J]", "status": "driver status",
        "t_reached": "last time reached [1/J]",
        "err_N": "mean_{0<=t<=200} |N - N_exact| / N(0) [1] (nan if the run stopped early)",
        "err_D": "mean_{0<=t<=20} |D - D_exact| / D_exact(0), D = sum_r D_r [1] (nan if D(0) = 0 or stopped early)",
        "N_end": "N at the last frame [1]", "D_min": "min_t sum_r D_r [1]", "D_site_min": "min_{t,r} D_r [1]",
        "eig_min": "min_t smallest eigenvalue of rho [1]", "eig_max": "max_t largest eigenvalue of rho [1]",
        "dN_dt_max": "max_t dN/dt [J]", "S2_drift": "max_t |<S^2>(t) - <S^2>(0)| [1] (nan for mGW)",
        "P_min": "min over checkpoints (every 1/J) of the smallest eigenvalue of P [1] (nan for mGW)",
        "Q_min": "same for Q [1]", "Gp_min": "same for G' [1]"}},
    "table.csv": {"description": "cells of Table tab:plaqerr: err_N averaged over the five gamma (ic = initial state) "
                                 "or over all trajectories of the method (ic = mean); early-stopped runs excluded",
                  "columns": {"method": "data label", "ic": "initial state or mean", "err_N": "mean err_N [1]",
                              "n_runs": "trajectories averaged", "n_excluded": "trajectories excluded (stopped early)"}},
    "exact.csv": {"description": "exact reference values per trajectory", "columns": {
        "ic": "initial state", "gamma": "gamma [J]", "N0": "N(0) [1]", "N_end": "N(t = 200/J) [1]",
        "D0": "sum_r D_r(0) [1]", "S2_drift": "max_t |<S^2>(t) - <S^2>(0)| [1]",
        "P_min": "min_t smallest eigenvalue of P [1]", "Q_min": "same for Q [1]", "Gp_min": "same for G' [1]"}},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(HERE / "output")); ap.add_argument("--data", default=str(HERE / "data"))
    ap.add_argument("--allow-missing", action="store_true")
    a = ap.parse_args()
    D, prod = _build(a.output, a.allow_missing)
    IO.write_data(a.data, D, "tab_plaqerr/postprocess.py", DESC, prod)
    print("wrote", ", ".join(D), "->", a.data)


if __name__ == "__main__":
    main()
