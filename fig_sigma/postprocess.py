"""fig_sigma: output/ -> data/sigma.h5 (G^R(tau) at the steady state, G^R(omega), Sigma^R(omega) for both spins).
G^R(omega) = int_0^{tau_end} e^{i omega tau} G^R(tau) dtau, eta = 0, piecewise-cubic Filon quadrature
(common.sigma_tools.ft); tau_end = the run's own end (driver: automatic RTDE end; exact: tau_max of the card).
Sigma^R(omega) = 1/G0(omega) - 1/G^R(omega), G0 = [(omega - h_o)^{-1}]_00 (gamma = 0 one-body problem, exact).
usage: python postprocess.py [--output DIR] [--data DIR] [--allow-missing]"""
import argparse
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import readers as R, io as IO, models as M, sigma_tools as T     # noqa: E402

OMEGA = np.linspace(-6, 6, 1201)
ETA = 0.0
TAU_STRIDE = 10          # G^R(tau) stored every 10th sample (data size); the transform uses every sample


def _check(rows, out, allow):
    bad = [r["run_id"] for r in rows
           if R.exit_code(out, r["run_id"]) not in (R.FINISHED_DRIVER if r["kind"] == "driver" else {0})]
    if bad and not allow:
        sys.exit("missing/unfinished runs:\n  " + "\n  ".join(bad))
    return set(bad)


def _build(output_dir, allow_missing=False):
    out = Path(output_dir); rows = R.read_index(HERE); skip = _check(rows, out, allow_missing)
    prod = {"dp5drive": [], "code_exact": []}
    h5 = {"@attrs": {"units": "omega, Sigma in units of h1; tau in 1/h1"}}
    summ = {k: [] for k in ("method", "system", "ic", "gamma", "status", "t_prime", "tau_end", "n_up", "n_dn",
                            "gamma_n_dn", "Sigma_inf_up_re", "Sigma_inf_up_im")}
    for r in sorted(rows, key=lambda r: r["run_id"]):
        if r["run_id"] in skip:
            continue
        s, g, m = r["system"], r["gamma"], r["method"]; ho = M.get(s).h_o()
        if r["kind"] == "exact":
            e = R.load_exact(out / r["run_id"] / "result.h5"); prod["code_exact"].append(e["version"])
            tau, GR = e["GR"]["tau"], e["GR"]["GR"]
            nup, ndn = e["ness"]["n"][0, 0], e["ness"]["n"][1, 0]; tp, st = np.nan, "Ok"
            extra = {"omega_resolvent": e["GR"]["resolvent"]["omega"], "GR_resolvent": e["GR"]["resolvent"]["GR"]}
        else:
            d = R.load(out / r["run_id"] / "out.h5"); prod["dp5drive"].append(d["version"])
            tau = d["rtde"]["tau"]; GR = d["rtde"]["GR"][:, 0, :, 0]         # [T_R, spin] at x = j = 0
            tp, st = d["rtde"]["t_prime"], d["status"]
            k = int(np.argmin(np.abs(d["t"] - tp))); nup, ndn = d["n_site"][k, 0, 0], d["n_site"][k, 1, 0]
            extra = {}
        G0 = T.g0_resolvent(ho, OMEGA, ETA)
        GRw = np.stack([T.ft(tau, GR[:, j], OMEGA, ETA) for j in range(2)], axis=1)
        Sig = np.stack([T.sigma(GRw[:, j], G0) for j in range(2)], axis=1)
        h = tau[1] - tau[0]; G = GR[:, 0]
        d0 = (-25 * G[0] + 48 * G[1] - 36 * G[2] + 16 * G[3] - 3 * G[4]) / (12 * h)
        sinf = -d0 - ho[0, 0]                    # Sigma(omega -> inf) = m1 - h_o,00 (G ~ -i e^{-i m1 tau})
        grp = f"{s}_{r['ic']}_g{g:g}"
        h5.setdefault(m, {})[grp] = {"tau": tau[::TAU_STRIDE], "GR": GR[::TAU_STRIDE], "omega": OMEGA, "GR_w": GRw, "Sigma": Sig, **extra,
                                     "@attrs": {"gamma": g, "ic": r["ic"], "t_prime": tp, "tau_end": float(tau[-1]),
                                                "dt": float(h), "tau_stride": TAU_STRIDE, "eta": ETA, "orbital": 0, "status": st}}
        for kk, v in (("method", m), ("system", s), ("ic", r["ic"]), ("gamma", g), ("status", st), ("t_prime", tp),
                      ("tau_end", float(tau[-1])), ("n_up", nup), ("n_dn", ndn), ("gamma_n_dn", g * ndn),
                      ("Sigma_inf_up_re", sinf.real), ("Sigma_inf_up_im", sinf.imag)):
            summ[kk].append(v)
    return {"sigma.h5": h5, "summary.csv": {k: np.array(v) for k, v in summ.items()}}, prod


def build(output_dir, allow_missing=False):
    """output/ -> {data file name: content} (pure, no writing)"""
    return _build(output_dir, allow_missing)[0]


DESC = {
    "sigma.h5": {"description": "groups /<method>/<system>_<ic>_g<gamma>/: tau [n], GR complex [n, 2] (spin up, dn; "
                              "G^R_dd(t'+tau, t'), every tau_stride-th sample of the run's grid dt), omega [m], GR_w complex [m, 2], Sigma complex [m, 2]; exact also "
                              "omega_resolvent, GR_resolvent (independent resolvent G^R(omega), spin up then dn); attrs "
                              "gamma, ic, t_prime, tau_end, dt, eta, orbital, status"},
    "summary.csv": {"description": "one row per run", "columns": {
        "method": "exact or data label", "system": "c3", "ic": "ic00", "gamma": "pair-loss rate [h1 units]",
        "status": "driver status", "t_prime": "RTDE start [1/h1]", "tau_end": "RTDE duration [1/h1]",
        "n_up": "n_d,up at the steady state [1]", "n_dn": "n_d,dn [1]",
        "gamma_n_dn": "gamma n_d,dn: |Sigma_up(omega -> inf)| expected [h1 units]",
        "Sigma_inf_up_re": "Re Sigma_up(omega -> inf) from G'(tau = 0) [h1 units]",
        "Sigma_inf_up_im": "Im Sigma_up(omega -> inf) [h1 units]"}},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(HERE / "output")); ap.add_argument("--data", default=str(HERE / "data"))
    ap.add_argument("--allow-missing", action="store_true")
    a = ap.parse_args()
    D, prod = _build(a.output, a.allow_missing)
    IO.write_data(a.data, D, "fig_sigma/postprocess.py", DESC, prod)
    print("wrote", ", ".join(D), "->", a.data)


if __name__ == "__main__":
    main()
