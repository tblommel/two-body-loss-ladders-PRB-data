"""quoted_numbers: output/ -> data/ for the runs of this directory, plus quantities derived from the shipped data of
fig_sigma (shape of Sigma^R). Definitions of kappa and the kappa window are those of fig_zeno/postprocess.py.
  stability.csv        G1-G2 runs of the strongly damped impurity to t = 600 (MF, TPPX): impurity M_z at start/end
  kappa_biggamma.csv   exact plaquette kappa at gamma/J = 50, 100 (mixed, Neel)
  sigma_features.csv   per method and gamma (fig_sigma data): position/height of the -Im Sigma peak at omega < 0, the
                       local minimum of -Im Sigma at omega > 0, the extrema of Re Sigma, int |Sigma - Sigma_exact| d omega
  sigma_extrema.csv    per method, gamma and local extremum of the EXACT -Im Sigma / Re Sigma on [-6, 6]: its position and
                       that of the nearest local extremum of the same kind (max/min) of the method (sec7C quotes the
                       DSL mean/max over the extrema with |omega_exact| <= 4.5)
usage: python postprocess.py [--output DIR] [--data DIR] [--allow-missing]"""
import argparse
import importlib.util
import sys
from pathlib import Path
import numpy as np
from scipy.signal import argrelextrema
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import readers as R, io as IO                 # noqa: E402

_spec = importlib.util.spec_from_file_location("zeno_pp", HERE.parent / "fig_zeno" / "postprocess.py")
ZP = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(ZP)


def _check(rows, out, allow):
    bad = [r["run_id"] for r in rows
           if R.exit_code(out, r["run_id"]) not in (R.FINISHED_DRIVER if r["kind"] == "driver" else {0})]
    if bad and not allow:
        sys.exit("missing/unfinished runs:\n  " + "\n  ".join(bad))
    return set(bad)


ORDER = 20               # local extrema: larger than 20 neighbours on each side (0.2 in omega)


def sigma_features(sig_h5):
    out = {k: [] for k in ("method", "gamma", "peak_w", "peak", "posmin_w", "posmin", "re_max_w", "re_max",
                           "re_min_w", "re_min", "int_abs_err")}
    for m in sorted(k for k in sig_h5 if not k.startswith("@")):
        for grp, v in sorted(sig_h5[m].items()):
            if grp.startswith("@"):
                continue
            w = v["omega"]; s = v["Sigma"][:, 0]; y = -s.imag; x = s.real
            se = sig_h5["exact"][grp]["Sigma"][:, 0]
            neg, pos = w < 0, w > 0
            lmax = argrelextrema(y, np.greater, order=ORDER)[0]; lmin = argrelextrema(y, np.less, order=ORDER)[0]
            lmax = lmax[neg[lmax]]; lmin = lmin[pos[lmin]]
            i = lmax[np.argmax(y[lmax])] if len(lmax) else np.argmax(np.where(neg, y, -np.inf))
            j = lmin[np.argmin(y[lmin])] if len(lmin) else np.argmin(np.where(pos, y, np.inf))
            rmax = argrelextrema(x, np.greater, order=ORDER)[0]; rmin = argrelextrema(x, np.less, order=ORDER)[0]
            ia = rmax[np.argmax(x[rmax])] if len(rmax) else np.argmax(x)
            ib = rmin[np.argmin(x[rmin])] if len(rmin) else np.argmin(x)
            out["method"].append(m); out["gamma"].append(float(v["@attrs"]["gamma"]))
            out["peak_w"].append(w[i]); out["peak"].append(y[i]); out["posmin_w"].append(w[j]); out["posmin"].append(y[j])
            out["re_max_w"].append(w[ia]); out["re_max"].append(x[ia])
            out["re_min_w"].append(w[ib]); out["re_min"].append(x[ib])
            out["int_abs_err"].append(float(np.trapezoid(np.abs(s - se), w)))
    return {k: np.array(v) for k, v in out.items()}


def sigma_extrema(sig_h5):
    """every local extremum (order ORDER) of the exact -Im Sigma and Re Sigma, paired with the nearest local extremum of the
    same kind of each method; d_w = |omega_method - omega_exact| (nan if the method has none of that kind)"""
    out = {k: [] for k in ("method", "gamma", "part", "kind", "w_exact", "w", "d_w")}
    for m in sorted(k for k in sig_h5 if not k.startswith("@") and k != "exact"):
        for grp, v in sorted(sig_h5[m].items()):
            if grp.startswith("@"):
                continue
            w = v["omega"]; s = v["Sigma"][:, 0]; se = sig_h5["exact"][grp]["Sigma"][:, 0]
            for part, fe, fm in (("-Im", -se.imag, -s.imag), ("Re", se.real, s.real)):
                for kind, cmp in (("max", np.greater), ("min", np.less)):
                    ie = argrelextrema(fe, cmp, order=ORDER)[0]; im = argrelextrema(fm, cmp, order=ORDER)[0]
                    for i in ie:
                        j = im[np.argmin(np.abs(w[im] - w[i]))] if len(im) else None
                        out["method"].append(m); out["gamma"].append(float(v["@attrs"]["gamma"]))
                        out["part"].append(part); out["kind"].append(kind); out["w_exact"].append(w[i])
                        out["w"].append(w[j] if j is not None else np.nan)
                        out["d_w"].append(abs(w[j] - w[i]) if j is not None else np.nan)
    return {k: np.array(v) for k, v in out.items()}


def _build(output_dir, allow_missing=False):
    out = Path(output_dir); rows = R.read_index(HERE); skip = _check(rows, out, allow_missing)
    prod = {"dp5drive": [], "code_exact": []}
    st = {k: [] for k in ("method", "ic", "gamma", "status", "t_end", "Mz0", "Mz_end", "Mz_absmax", "nd_end")}
    kb = {k: [] for k in ("ic", "gamma", "N_inf", "kappa", "gamma_kappa", "D_over_O")}
    for r in sorted(rows, key=lambda r: r["run_id"]):
        if r["run_id"] in skip:
            continue
        if r["kind"] == "driver":
            d = R.load(out / r["run_id"] / "out.h5", with_rtde=False); prod["dp5drive"].append(d["version"])
            nu, nd = d["n_site"][:, 0, 0], d["n_site"][:, 1, 0]; Mz = 0.5 * (nu - nd)
            for k, v in (("method", r["method"]), ("ic", r["ic"]), ("gamma", r["gamma"]), ("status", d["status"]),
                         ("t_end", d["t"][-1]), ("Mz0", Mz[0]), ("Mz_end", Mz[-1]), ("Mz_absmax", np.abs(Mz).max()),
                         ("nd_end", nu[-1] + nd[-1])):
                st[k].append(v)
        else:
            e = R.load_exact(out / r["run_id"] / "result.h5"); prod["code_exact"].append(e["version"])
            tr = e["transient"]; Ni = float(e["n_inf"]["N"]); g = r["gamma"]
            k = ZP.kappa(tr["t"], tr["N"], Ni, g); w = ZP.window(tr["t"], tr["n"], tr["D"].sum(1), Ni, g)
            for kk, v in (("ic", r["ic"]), ("gamma", g), ("N_inf", Ni), ("kappa", k), ("gamma_kappa", g * k),
                          ("D_over_O", w["D_over_O"] if w else np.nan)):
                kb[kk].append(v)
    D = {"stability.csv": {k: np.array(v) for k, v in st.items()},
         "kappa_biggamma.csv": {k: np.array(v) for k, v in kb.items()}}
    sig = IO.read_data(HERE.parent / "fig_sigma" / "data")["sigma.h5"]
    D["sigma_features.csv"] = sigma_features(sig)
    D["sigma_extrema.csv"] = sigma_extrema(sig)
    return D, prod


def build(output_dir, allow_missing=False):
    """output/ -> {data file name: content} (pure, no writing)"""
    return _build(output_dir, allow_missing)[0]


DESC = {
    "stability.csv": {"description": "strongly damped impurity, G1-G2 to t = 600 from a slightly polarized impurity "
                                     "(pert: n_up 0.101, n_dn 0.100) and from the polarized start (ic10)", "columns": {
        "method": "data label", "ic": "pert | ic10", "gamma": "pair-loss rate [h1 units]", "status": "driver status",
        "t_end": "last frame [1/h1]", "Mz0": "impurity M_z(0) [1]", "Mz_end": "M_z at the last frame [1]",
        "Mz_absmax": "max_t |M_z| [1]", "nd_end": "impurity occupation at the last frame [1]"}},
    "kappa_biggamma.csv": {"description": "exact plaquette loss coefficient at strong loss (definitions of "
                                          "fig_zeno/postprocess.py)", "columns": {
        "ic": "initial state", "gamma": "gamma [J]", "N_inf": "exact t -> infinity N [1]", "kappa": "kappa [J]",
        "gamma_kappa": "gamma kappa [J^2]", "D_over_O": "window mean of D/O [1]"}},
    "sigma_extrema.csv": {"description": "local extrema (order 20, i.e. 0.2 in omega) of the exact -Im/Re Sigma^R_up on "
                          "[-6, 6] (fig_sigma data), each paired with the nearest extremum of the same kind of the method",
                          "columns": {"method": "scheme", "gamma": "gamma [h1]", "part": "-Im or Re", "kind": "max or min",
                                      "w_exact": "exact position [h1]", "w": "method position [h1]",
                                      "d_w": "|w - w_exact| [h1]"}},
    "sigma_features.csv": {"description": "shape of Sigma^R_up(omega) on omega in [-6, 6] (fig_sigma data)", "columns": {
        "method": "exact or data label", "gamma": "pair-loss rate [h1 units]",
        "peak_w": "omega of the largest local maximum of -Im Sigma at omega < 0 [h1]", "peak": "its value [h1]",
        "posmin_w": "omega of the deepest local minimum of -Im Sigma at omega > 0 [h1]", "posmin": "its value [h1]",
        "re_max_w": "omega of the largest local maximum of Re Sigma [h1]", "re_max": "its value [h1]",
        "re_min_w": "omega of the deepest local minimum of Re Sigma [h1]", "re_min": "its value [h1]",
        "int_abs_err": "int |Sigma - Sigma_exact| d omega over [-6, 6] (trapezoid) [h1^2]"}},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(HERE / "output")); ap.add_argument("--data", default=str(HERE / "data"))
    ap.add_argument("--allow-missing", action="store_true")
    a = ap.parse_args()
    D, prod = _build(a.output, a.allow_missing)
    IO.write_data(a.data, D, "quoted_numbers/postprocess.py", DESC, prod)
    print("wrote", ", ".join(D), "->", a.data)


if __name__ == "__main__":
    main()
