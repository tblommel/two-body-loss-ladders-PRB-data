"""fig_impss: impurity steady state and spin relaxation (main Fig. fig:impss), strongly damped (c3) and weakly damped
(c3G0.2) impurity, and the impurity numbers of Sec. VII C. Writes runcards/ and index.csv.
Driver: RTDE runs with automatic t' (steady-state detection) and automatic RTDE end, tight tolerances.
Exact: per (system, gamma) the NESS, the transient from IC 00 and G^R(tau) at the NESS; transients from IC 11, 10."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import runcards as RC, models as M           # noqa: E402

# ----------------------------------------------------------------------------------------- parameters (only here)
SYSTEMS = ["c3", "c3G0.2"]
GAMMAS = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0]
ICS = ["00", "11", "10"]
METHODS = ["HF", "2B", "TPPX", "mGW", "DSL"]
T_END, DT_OUT = 4000.0, 0.01             # t_end unused with t_prime = auto (run ends at t' + tau_end)
TOL = RC.TOL_TIGHT
CHECKPOINT_INTERVAL = 2.0
RTDE = RC.RTDE_C3                        # detector and automatic RTDE end (tol_ss 1e-8, c_max 500, tol_A 1e-7, cap 1200)
EXACT_OBS = ["N", "N_sigma", "n", "D", "S2", "Mz", "rho1", "rho1_eig"]
EXACT_T, EXACT_DT, PQG_EVERY = 150.0, 0.01, 1.0
NESS = {"method": "propagate", "propagate": {"block": 50.0, "tol": 1e-15, "t_max": 3000.0}}
GR = {"tau_max": 400.0, "dtau": 0.01,
      "resolvent": {"omega_min": -6.0, "omega_max": 6.0, "n": 25, "solver": "gmres", "rtol": 1e-13}}
COST_S = {"driver": 20, "exact00": 3000, "exact": 900}
# ---------------------------------------------------------------------------------------------------------------


def exact_tasks(ic):
    tr = {"type": "transient", "name": "transient", "grid": [{"t_end": EXACT_T, "dt": EXACT_DT}],
          "observables": EXACT_OBS, "pqg_every": PQG_EVERY}
    if ic != "00":
        return [tr]
    return [{"type": "ness", "name": "ness", **NESS, "observables": EXACT_OBS, "pqg": True}, tr,
            {"type": "greens_R", "name": "GR", "state": "ness", "orbital": 0, "spins": ["up", "dn"], **GR}]


def main():
    D = RC.fresh_tree(HERE / "runcards")
    rows = []
    for s in SYSTEMS:
        RC.write_model(D, s)
        for g in GAMMAS:
            for ic in ICS:
                for m in METHODS:
                    rid = RC.run_id(s, f"ic{ic}", g, m)
                    RC.write_driver_card(D, rid, s, g, m, M.imp_rho0(s, ic), T_END, DT_OUT, "rtde", TOL,
                                         CHECKPOINT_INTERVAL, rtde=RTDE)
                    rows.append(dict(run_id=rid, kind="driver", system=s, ic=f"ic{ic}", gamma=g, method=m,
                                     cost_s=COST_S["driver"]))
                rid = RC.run_id(s, f"ic{ic}", g, "exact")
                RC.write_exact_card(D, rid, s, g, M.imp_rho0(s, ic).real, exact_tasks(ic))
                rows.append(dict(run_id=rid, kind="exact", system=s, ic=f"ic{ic}", gamma=g, method="exact",
                                 cost_s=COST_S["exact00" if ic == "00" else "exact"]))
    RC.write_index(D, rows)
    print(len(rows), "cards ->", D)


if __name__ == "__main__":
    main()
