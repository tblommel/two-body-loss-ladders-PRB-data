"""fig_zeno: output/ -> data/ (loss coefficient kappa and kappa-window means of O, D, D/O per run).
Definitions (main Sec. VII B):
  N_inf   exact: zero-mode projection (n_inf task); schemes: Aitken extrapolation of N at t = 300, sqrt(300*3000), 3000
          (N converged: N(3000)); undefined if the run stopped early or N is not monotonic there.
  kappa   X = N - N_inf, t1 = 3/(2 gamma) (three doublon lifetimes); tau = time in which X falls by 25% after t1
          (linear interpolation between frames); kappa = 1/(3 tau)  [two-body law dX/dt = -kappa X^2/X(t1)].
  window  frames from the first frame at t >= t1 to the first frame with X <= 0.75 X(t1); time means over these frames
          of O = sum_r n_r,up n_r,dn, D = sum_r D_r, D/O, S = sum_r |n_r,up - n_r,dn| / N, A = O/(N^2/16).
  Exact mixed start at gamma/J in {10, 20, 30}: window means from the fine task (dt = 0.05/gamma).
usage: python postprocess.py [--output DIR] [--data DIR] [--allow-missing]"""
import argparse
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import readers as R, io as IO                 # noqa: E402

FRAC = 0.25
C_T1 = 3.0


def ninf_scheme(t, N, ok):
    if not ok:
        return np.nan, "fail"
    at = lambda T: np.interp(T, t, N)
    a, b, c = at(300.0), at(np.sqrt(300 * 3000.0)), at(3000.0)
    den = a + c - 2 * b
    if abs(c - b) < 1e-6:
        return c, "conv"
    if den <= 0 or not (a >= b >= c or a <= b <= c):
        return np.nan, "nonmono"
    return (a * c - b * b) / den, "aitken"


def kappa(t, N, Ni, g):
    if not np.isfinite(Ni):
        return np.nan
    X = N - Ni; t1 = C_T1 / (2 * g); i1 = np.searchsorted(t, t1)
    if i1 >= len(t):
        return np.nan
    X1 = X[i1]; i2 = np.where((t > t1) & (X <= (1 - FRAC) * X1))[0]
    if X1 <= 0 or not len(i2):
        return np.nan
    j = i2[0]; tt = np.interp((1 - FRAC) * X1, [X[j], X[j - 1]], [t[j], t[j - 1]])
    return FRAC / (1 - FRAC) / (tt - t1)


def window(t, n, D, Ni, g):
    N = n.sum((1, 2)); t1 = C_T1 / (2 * g); i1 = np.searchsorted(t, t1); X = N - Ni; X1 = X[i1]
    i2 = np.where((t > t1) & (X <= (1 - FRAC) * X1))[0]
    if not len(i2):
        return None
    w = slice(i1, i2[0] + 1)
    O = (n[:, 0, :] * n[:, 1, :]).sum(1)
    H = D / np.where(O > 0, O, np.nan)
    return dict(t1=t[i1], t2=t[i2[0]], O=O[w].mean(), D=D[w].mean(), D_over_O=np.nanmean(H[w]),
                S=(np.abs(n[:, 0, :] - n[:, 1, :]).sum(1) / N)[w].mean(), A=(O / (N ** 2 / 16))[w].mean())


def _check(rows, out, allow):
    bad = [r["run_id"] for r in rows
           if R.exit_code(out, r["run_id"]) not in (R.FINISHED_DRIVER if r["kind"] == "driver" else {0})]
    if bad and not allow:
        sys.exit("missing/unfinished runs:\n  " + "\n  ".join(bad))
    return set(bad)


def _build(output_dir, allow_missing=False):
    out = Path(output_dir); rows = R.read_index(HERE); skip = _check(rows, out, allow_missing)
    prod = {"dp5drive": [], "code_exact": []}
    runs = {}
    for r in rows:
        if r["run_id"] in skip:
            continue
        key = (r["method"], r["ic"], r["gamma"])
        if r["kind"] == "exact":
            e = R.load_exact(out / r["run_id"] / "result.h5"); prod["code_exact"].append(e["version"])
            tr = e["transient"]
            runs[key] = dict(t=tr["t"], N=tr["N"], n=tr["n"], D=tr["D"].sum(1), Ni=float(e["n_inf"]["N"]), how="exact",
                             status="Ok", t_reached=tr["t"][-1])
            if "fine" in e:
                f = e["fine"]; runs[key]["fine"] = dict(t=f["t"], n=f["n"], D=f["D"].sum(1))
        else:
            d = R.load(out / r["run_id"] / "out.h5", with_rtde=False); prod["dp5drive"].append(d["version"])
            runs.setdefault(key, {})[r["segment"]] = d
    kap = {k: [] for k in ("method", "ic", "gamma", "status", "N_inf", "N_inf_how", "kappa")}
    win = {k: [] for k in ("method", "ic", "gamma", "source", "t1", "t2", "O", "D", "D_over_O", "S", "A")}
    for key in sorted(runs, key=lambda k: (k[1], ["exact", "HF", "TPPX", "DSL"].index(k[0]), k[2])):
        m, ic, g = key; v = runs[key]
        if m != "exact":
            f, l = v["fine"], v.get("long")
            k = l["t"] > f["t"][-1] + 1e-9
            t = np.concatenate([f["t"], l["t"][k]]); N = np.concatenate([f["N"], l["N"][k]])
            n = np.concatenate([f["n_site"], l["n_site"][k]]); D = np.concatenate([f["D"], l["D"][k]])
            Ni, how = ninf_scheme(t, N, l["status"] == "Ok" and l["t_reached"] >= 2999)
            st = l["status"]
        else:
            t, N, n, D, Ni, how, st = v["t"], v["N"], v["n"], v["D"], v["Ni"], "exact", "Ok"
        kap["method"].append(m); kap["ic"].append(ic); kap["gamma"].append(g); kap["status"].append(st)
        kap["N_inf"].append(Ni); kap["N_inf_how"].append(how); kap["kappa"].append(kappa(t, N, Ni, g))
        if not np.isfinite(Ni):
            continue
        src = "main"
        if m == "exact" and "fine" in v:
            t, n, D, src = v["fine"]["t"], v["fine"]["n"], v["fine"]["D"], "fine"
        w = window(t, n, D, Ni, g)
        if w is None:
            continue
        win["method"].append(m); win["ic"].append(ic); win["gamma"].append(g); win["source"].append(src)
        for q in ("t1", "t2", "O", "D", "D_over_O", "S", "A"):
            win[q].append(w[q])
    return {"kappa.csv": {k: np.array(v) for k, v in kap.items()},
            "window.csv": {k: np.array(v) for k, v in win.items()}}, prod


def build(output_dir, allow_missing=False):
    """output/ -> {data file name: content} (pure, no writing)"""
    return _build(output_dir, allow_missing)[0]


DESC = {
    "kappa.csv": {"description": "loss coefficient kappa (25% definition, see postprocess.py) per method, initial "
                                 "state and gamma", "columns": {
        "method": "exact or data label", "ic": "initial state", "gamma": "gamma [J]",
        "status": "driver status of the long segment (exact: Ok)", "N_inf": "t -> infinity particle number [1]",
        "N_inf_how": "exact | conv | aitken | nonmono | fail", "kappa": "loss coefficient [J]"}},
    "window.csv": {"description": "time means over the kappa window", "columns": {
        "method": "exact or data label", "ic": "initial state", "gamma": "gamma [J]",
        "source": "main transient or fine task (exact mixed, gamma >= 10)", "t1": "window start [1/J]",
        "t2": "window end [1/J]", "O": "sum_r n_r,up n_r,dn [1]", "D": "sum_r D_r [1]",
        "D_over_O": "correlation-hole factor D/O [1]", "S": "staggering sum_r |n_r,up - n_r,dn| / N [1]",
        "A": "O / (N^2/16) [1]"}},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(HERE / "output")); ap.add_argument("--data", default=str(HERE / "data"))
    ap.add_argument("--allow-missing", action="store_true")
    a = ap.parse_args()
    D, prod = _build(a.output, a.allow_missing)
    IO.write_data(a.data, D, "fig_zeno/postprocess.py", DESC, prod)
    print("wrote", ", ".join(D), "->", a.data)


if __name__ == "__main__":
    main()
