"""fig_site: output/ -> data/ (curves of panels a, b; endpoints of panel c; per-run summary for Sec. VII A numbers).
usage: python postprocess.py [--output DIR] [--data DIR] [--allow-missing]"""
import argparse
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import readers as R, io as IO                 # noqa: E402

CURVE_N0 = [0.5, 0.9]                  # n0 of panels (a) and (b)
GT_CHECK = [100.0, 500.0, 2000.0]      # gamma t of the tail values


def keep_frames(nf, dense=200, nlog=700):
    """all frames up to index `dense`, then log-spaced frame indices (curves of panels a, b)"""
    k = np.unique(np.round(np.logspace(np.log10(dense), np.log10(nf - 1), nlog)).astype(int))
    return np.unique(np.concatenate([np.arange(min(dense, nf)), k]))


def _missing(rows, out, allow):
    bad = []
    for r in rows:
        c = R.exit_code(out, r["run_id"])
        ok = c in (R.FINISHED_DRIVER if r["kind"] == "driver" else {0})
        if not ok:
            bad.append(f"{r['run_id']} (exit {c})")
    if bad and not allow:
        sys.exit("missing/unfinished runs:\n  " + "\n  ".join(bad))
    return {b.split()[0] for b in bad}


def _build(output_dir, allow_missing=False):
    out = Path(output_dir)
    rows = R.read_index(HERE)
    skip = _missing(rows, out, allow_missing)
    ex, drv, prod = {}, {}, {"dp5drive": [], "code_exact": []}
    for r in rows:
        if r["run_id"] in skip:
            continue
        n0 = float(r["ic"][1:])
        if r["kind"] == "exact":
            e = R.load_exact(out / r["run_id"] / "result.h5"); prod["code_exact"].append(e["version"])
            tr = e["transient"]
            ex[n0] = dict(gt=tr["t"] * r["gamma"], n=tr["n"][:, 0, 0], D=tr["D"][:, 0], n_inf=float(e["n_inf"]["n"][0, 0]) if "n_inf" in e else np.nan)
        else:
            d = R.load(out / r["run_id"] / "out.h5"); prod["dp5drive"].append(d["version"])
            drv[(r["method"], n0)] = dict(gt=d["t"] * r["gamma"], n=d["N_sigma"][:, 0], D=d["D"], status=d["status"],
                                          t_reached=d["t_reached"], eig_min=d["eig_min"], eig_max=d["eig_max"],
                                          dN_dt=d["dN_dt"])
    methods = sorted({m for m, _ in drv}, key=["HF", "2B", "TPPX", "mGW", "DSL"].index)
    # panels (a), (b)
    cur = {k: [] for k in ("method", "n0", "gt", "n_up", "D")}
    for n0 in CURVE_N0:
        e = ex[n0]; k = keep_frames(len(e["gt"]))
        for m in ["exact"] + methods:
            s = e if m == "exact" else drv[(m, n0)]
            assert np.allclose(s["gt"][k], e["gt"][k], rtol=0, atol=1e-9)
            cur["method"] += [m] * len(k); cur["n0"] += [n0] * len(k); cur["gt"] += list(e["gt"][k])
            cur["n_up"] += list(s["n"][k]); cur["D"] += list(s["D"][k])
    # panel (c) and summary
    end = {k: [] for k in ("method", "n0", "n_up")}
    summ = {k: [] for k in ("method", "n0", "status", "gt_reached", "err_n", "err_D", "n_min", "D_min", "eig_min",
                            "eig_max", "dN_dt_max") + tuple(f"n_gt{g:g}" for g in GT_CHECK)}
    for n0 in sorted(ex):
        end["method"].append("exact"); end["n0"].append(n0); end["n_up"].append(ex[n0]["n"][-1])
    for m in methods:
        for n0 in sorted(n for mm, n in drv if mm == m):
            s = drv[(m, n0)]; e = ex[n0]; nt = len(s["gt"])
            assert np.allclose(s["gt"], e["gt"][:nt], rtol=0, atol=1e-9)
            end["method"].append(m); end["n0"].append(n0); end["n_up"].append(s["n"][-1])
            summ["method"].append(m); summ["n0"].append(n0); summ["status"].append(s["status"])
            summ["gt_reached"].append(s["gt"][-1])
            summ["err_n"].append(np.abs(s["n"] - e["n"][:nt]).max()); summ["err_D"].append(np.abs(s["D"] - e["D"][:nt]).max())
            summ["n_min"].append(s["n"].min()); summ["D_min"].append(s["D"].min())
            summ["eig_min"].append(s["eig_min"].min()); summ["eig_max"].append(s["eig_max"].max())
            summ["dN_dt_max"].append(s["dN_dt"].max())
            for g in GT_CHECK:
                summ[f"n_gt{g:g}"].append(s["n"][np.argmin(np.abs(s["gt"] - g))])
    exs = {"n0": [n for n in sorted(ex) if not np.isnan(ex[n]["n_inf"])]}
    exs["n_inf"] = [ex[n]["n_inf"] for n in exs["n0"]]
    D = {"curves.csv": {k: np.array(v) for k, v in cur.items()},
         "endpoints.csv": {k: np.array(v) for k, v in end.items()},
         "summary.csv": {k: np.array(v) for k, v in summ.items()},
         "exact_ninf.csv": {k: np.array(v) for k, v in exs.items()}}
    return D, prod


def build(output_dir, allow_missing=False):
    """output/ -> {data file name: content} (pure, no writing)"""
    return _build(output_dir, allow_missing)[0]


DESC = {
    "curves.csv": {"description": "n_sigma(gamma t) and D(gamma t) from n0 = 0.5 and 0.9 (panels a, b); frames: all up to "
                                  "gamma t = 10, then 700 log-spaced output frames", "columns": {
        "method": "exact or data label", "n0": "initial occupation per spin", "gt": "gamma t [1]",
        "n_up": "occupation per spin n_sigma (spin up) [1]", "D": "double occupancy <n_up n_dn> [1]"}},
    "endpoints.csv": {"description": "n_sigma at gamma t = 2000 versus n0 (panel c)", "columns": {
        "method": "exact or data label", "n0": "initial occupation per spin", "n_up": "n_sigma(gamma t = 2000) [1]"}},
    "summary.csv": {"description": "per driver run: status and deviations from the exact run on identical frames "
                                   "(Sec. VII A numbers)", "columns": {
        "method": "data label", "n0": "initial occupation per spin", "status": "driver status",
        "gt_reached": "last frame [gamma t]", "err_n": "max_t |n - n_exact| [1]", "err_D": "max_t |D - D_exact| [1]",
        "n_min": "min_t n_sigma [1]", "D_min": "min_t D [1]", "eig_min": "min_t smallest eigenvalue of rho [1]",
        "eig_max": "max_t largest eigenvalue of rho [1]", "dN_dt_max": "max_t dN/dt [gamma]",
        "n_gt100": "n_sigma at gamma t = 100 [1]", "n_gt500": "n_sigma at gamma t = 500 [1]",
        "n_gt2000": "n_sigma at gamma t = 2000 [1]"}},
    "exact_ninf.csv": {"description": "exact t -> infinity limit (zero-mode projection)", "columns": {
        "n0": "initial occupation per spin", "n_inf": "n_sigma(t -> infinity) [1]"}},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(HERE / "output")); ap.add_argument("--data", default=str(HERE / "data"))
    ap.add_argument("--allow-missing", action="store_true")
    a = ap.parse_args()
    D, prod = _build(a.output, a.allow_missing)
    IO.write_data(a.data, D, "fig_site/postprocess.py", DESC, prod)
    print("wrote", ", ".join(D), "->", a.data)


if __name__ == "__main__":
    main()
