"""fig_impss: output/ -> data/ (steady states and per-run record; magnetization curves from the polarized start).
Driver steady state = the frame at t' (detected; ss_on_timeout = start opens the RTDE at the timeout). Exact steady
state = the NESS. Exact transients from IC 00, 11, 10 to t = 150.
usage: python postprocess.py [--output DIR] [--data DIR] [--allow-missing]"""
import argparse
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import readers as R, io as IO, positivity as POS, models as M     # noqa: E402

T_CURVE = 20.0          # magnetization curves kept for t <= T_CURVE
STRIDE = 5              # every 5th output frame (dt 0.05)


def t_1e(t, Mz):
    """first time at which Mz falls to Mz(0)/e (linear interpolation); nan if it never does"""
    x = Mz / Mz[0] - np.exp(-1)
    j = np.where(x <= 0)[0]
    if Mz[0] == 0 or not len(j) or j[0] == 0:
        return np.nan
    j = j[0]
    return float(np.interp(0.0, [x[j], x[j - 1]], [t[j], t[j - 1]]))


def t_last_above(t, Mz, level=0.05):
    """time of the last downward crossing of Mz(t) = level: linear interpolation between the last frame with
    Mz > level and the next frame; the last frame time if the series ends above the level; nan if never above"""
    j = np.where(Mz > level)[0]
    if not len(j):
        return np.nan
    j = j[-1]
    if j == len(t) - 1:
        return float(t[j])
    return float(np.interp(level, [Mz[j + 1], Mz[j]], [t[j + 1], t[j]]))


def _check(rows, out, allow):
    bad = [r["run_id"] for r in rows
           if R.exit_code(out, r["run_id"]) not in (R.FINISHED_DRIVER if r["kind"] == "driver" else {0})]
    if bad and not allow:
        sys.exit("missing/unfinished runs:\n  " + "\n  ".join(bad))
    return set(bad)


COLS = ("method", "system", "ic", "gamma", "status", "t_prime", "ss_detected", "tau_end", "n_up", "n_dn", "D", "Mz",
        "D_min", "n_min", "eig_min", "eig_max", "P_min", "Q_min", "Gp_min", "Mz_t1e", "Mz_last005")


def _build(output_dir, allow_missing=False):
    out = Path(output_dir); rows = R.read_index(HERE); skip = _check(rows, out, allow_missing)
    prod = {"dp5drive": [], "code_exact": []}
    st = {k: [] for k in COLS}
    cur = {k: [] for k in ("method", "system", "gamma", "t", "Mz", "n_up", "n_dn")}

    def put(**kw):
        for k in COLS:
            st[k].append(kw.get(k, np.nan))
    for r in sorted(rows, key=lambda r: r["run_id"]):
        if r["run_id"] in skip:
            continue
        s, ic, g, m = r["system"], r["ic"], r["gamma"], r["method"]
        if r["kind"] == "exact":
            e = R.load_exact(out / r["run_id"] / "result.h5"); prod["code_exact"].append(e["version"])
            tr = e["transient"]; t = tr["t"]; nup, ndn = tr["n"][:, 0, 0], tr["n"][:, 1, 0]; Mz = 0.5 * (nup - ndn)
            pq = tr["pqg"]
            common = dict(method="exact", system=s, ic=ic, gamma=g, D_min=tr["D"][:, 0].min(),
                          n_min=min(nup.min(), ndn.min()), eig_min=tr["rho1_eig"].min(), eig_max=tr["rho1_eig"].max(),
                          P_min=pq["P_min"].min(), Q_min=pq["Q_min"].min(), Gp_min=pq["Gp_min"].min(),
                          Mz_t1e=t_1e(t, Mz) if ic == "ic10" else np.nan,
                          Mz_last005=t_last_above(t, Mz) if ic == "ic10" else np.nan)
            if "ness" in e:          # IC 00 card: the NESS is the steady state
                ns = e["ness"]
                put(status="Ok", n_up=ns["n"][0, 0], n_dn=ns["n"][1, 0], D=ns["D"][0], Mz=0.5 * (ns["n"][0, 0] - ns["n"][1, 0]),
                    **common)
            else:                    # transient-only card: the state at t = 150
                put(status="Ok", n_up=nup[-1], n_dn=ndn[-1], D=tr["D"][-1, 0], Mz=Mz[-1], **common)
            if ic == "ic10":
                k = (t <= T_CURVE + 1e-9) & (np.arange(len(t)) % STRIDE == 0)
                n = int(k.sum())
                cur["method"] += ["exact"] * n; cur["system"] += [s] * n; cur["gamma"] += [g] * n
                cur["t"] += list(t[k]); cur["Mz"] += list(Mz[k]); cur["n_up"] += list(nup[k]); cur["n_dn"] += list(ndn[k])
            continue
        f = out / r["run_id"] / "out.h5"
        d = R.load(f); prod["dp5drive"].append(d["version"])
        t = d["t"]; nup, ndn = d["n_site"][:, 0, 0], d["n_site"][:, 1, 0]; Mz = 0.5 * (nup - ndn)
        tp = d["rtde"]["t_prime"] if "rtde" in d else np.nan
        k = int(np.argmin(np.abs(t - tp))) if np.isfinite(tp) else len(t) - 1
        p = POS.checkpoint_pqg(f, M.get(s).N) if m != "mGW" else {}
        mn = lambda q: (p[q].min() if len(p.get(q, [])) else np.nan)
        put(method=m, system=s, ic=ic, gamma=g, status=d["status"], t_prime=tp,
            ss_detected=int("lambda_used" in d.get("ss", {})),
            tau_end=float(d["rtde_end"]["tau_end"]) if "rtde_end" in d else np.nan,
            n_up=nup[k], n_dn=ndn[k], D=d["D_sites"][k, 0], Mz=Mz[k], D_min=d["D_sites"][:, 0].min(),
            n_min=min(nup.min(), ndn.min()), eig_min=d["eig_min"].min(), eig_max=d["eig_max"].max(),
            P_min=mn("P_min"), Q_min=mn("Q_min"), Gp_min=mn("Gp_min"), Mz_t1e=t_1e(t, Mz) if ic == "ic10" else np.nan,
            Mz_last005=t_last_above(t, Mz) if ic == "ic10" else np.nan)
        if ic == "ic10":
            idx = np.unique(np.concatenate([np.arange(0, len(t), STRIDE), [k]]))
            idx = idx[t[idx] <= T_CURVE + 1e-9]; n = len(idx)
            cur["method"] += [m] * n; cur["system"] += [s] * n; cur["gamma"] += [g] * n
            cur["t"] += list(t[idx]); cur["Mz"] += list(Mz[idx]); cur["n_up"] += list(nup[idx]); cur["n_dn"] += list(ndn[idx])
    return {"steady.csv": {k: np.array(v) for k, v in st.items()},
            "mz_curves.csv": {k: np.array(v) for k, v in cur.items()}}, prod


def build(output_dir, allow_missing=False):
    """output/ -> {data file name: content} (pure, no writing)"""
    return _build(output_dir, allow_missing)[0]


DESC = {
    "steady.csv": {"description": "one row per run: steady state (driver: frame at t'; exact IC 00: NESS; exact IC 11/10: "
                                  "state at t = 150) and trajectory record", "columns": {
        "method": "exact or data label", "system": "c3 (strongly damped) or c3G0.2 (weakly damped)",
        "ic": "impurity start ic00 | ic11 | ic10", "gamma": "pair-loss rate [h1 units]", "status": "driver status",
        "t_prime": "RTDE start t' [1/h1]", "ss_detected": "1 if t' was detected (0: detector timeout)",
        "tau_end": "RTDE duration [1/h1]", "n_up": "impurity occupation n_d,up [1]", "n_dn": "n_d,dn [1]",
        "D": "impurity double occupancy [1]", "Mz": "(n_d,up - n_d,dn)/2 [1]", "D_min": "min_t D [1]",
        "n_min": "min_t of n_d,up and n_d,dn [1]", "eig_min": "min_t smallest eigenvalue of rho [1]",
        "eig_max": "max_t largest eigenvalue of rho [1]",
        "P_min": "min over checkpoints (driver, every 2) or frames (exact, every 1) of the smallest eigenvalue of P [1]",
        "Q_min": "same for Q [1]", "Gp_min": "same for G' [1]",
        "Mz_t1e": "IC 10: first time at which Mz(t) = Mz(0)/e [1/h1]",
        "Mz_last005": "IC 10: last time at which Mz(t) = 0.05 is crossed from above, linear interpolation between frames (end of the series if it ends above 0.05; driver: up to the end of the run; exact: t <= 150) [1/h1]"}},
    "mz_curves.csv": {"description": "impurity occupations and magnetization from the polarized start (IC 10), t <= 20, "
                                     "every 5th output frame (driver: plus the frame at t')", "columns": {
        "method": "exact or data label", "system": "c3 or c3G0.2", "gamma": "pair-loss rate [h1 units]",
        "t": "time [1/h1]", "Mz": "(n_d,up - n_d,dn)/2 [1]", "n_up": "n_d,up [1]", "n_dn": "n_d,dn [1]"}},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(HERE / "output")); ap.add_argument("--data", default=str(HERE / "data"))
    ap.add_argument("--allow-missing", action="store_true")
    a = ap.parse_args()
    D, prod = _build(a.output, a.allow_missing)
    IO.write_data(a.data, D, "fig_impss/postprocess.py", DESC, prod)
    print("wrote", ", ".join(D), "->", a.data)


if __name__ == "__main__":
    main()
