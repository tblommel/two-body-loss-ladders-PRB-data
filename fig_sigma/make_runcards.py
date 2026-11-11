"""fig_sigma: retarded self-energy of the strongly damped impurity (main Fig. fig:sigma). Writes runcards/, index.csv.
Driver: RTDE runs from the empty impurity (IC 00), automatic t' and RTDE end, tight tolerances (as fig_impss).
Exact: NESS and G^R(tau) at the NESS (quantum regression), plus G^R(omega) by the resolvent at 25 frequencies."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import runcards as RC, models as M           # noqa: E402

# ----------------------------------------------------------------------------------------- parameters (only here)
SYSTEM = "c3"
IC = "00"
GAMMAS = [4.0, 8.0, 16.0]
METHODS = ["HF", "2B", "TPPX", "TPPX_0", "mGW", "DSL"]     # figure: HF, TPPX, TPPX_0, DSL; caption also 2B, mGW
T_END, DT_OUT = 4000.0, 0.01
TOL = RC.TOL_TIGHT
CHECKPOINT_INTERVAL = 2.0
RTDE = RC.RTDE_C3
EXACT_OBS = ["N", "N_sigma", "n", "D", "rho1_eig"]
NESS = {"method": "propagate", "propagate": {"block": 50.0, "tol": 1e-15, "t_max": 3000.0}}
GR = {"tau_max": 400.0, "dtau": 0.01,
      "resolvent": {"omega_min": -6.0, "omega_max": 6.0, "n": 25, "solver": "gmres", "rtol": 1e-13}}
COST_S = {"driver": 20, "exact": 2400}
# ---------------------------------------------------------------------------------------------------------------


def main():
    D = RC.fresh_tree(HERE / "runcards")
    RC.write_model(D, SYSTEM)
    rows = []
    for g in GAMMAS:
        for m in METHODS:
            rid = RC.run_id(SYSTEM, f"ic{IC}", g, m)
            RC.write_driver_card(D, rid, SYSTEM, g, m, M.imp_rho0(SYSTEM, IC), T_END, DT_OUT, "rtde", TOL,
                                 CHECKPOINT_INTERVAL, rtde=RTDE)
            rows.append(dict(run_id=rid, kind="driver", system=SYSTEM, ic=f"ic{IC}", gamma=g, method=m,
                             cost_s=COST_S["driver"]))
        rid = RC.run_id(SYSTEM, f"ic{IC}", g, "exact")
        RC.write_exact_card(D, rid, SYSTEM, g, M.imp_rho0(SYSTEM, IC).real,
                            [{"type": "ness", "name": "ness", **NESS, "observables": EXACT_OBS},
                             {"type": "greens_R", "name": "GR", "state": "ness", "orbital": 0, "spins": ["up", "dn"],
                              **GR}])
        rows.append(dict(run_id=rid, kind="exact", system=SYSTEM, ic=f"ic{IC}", gamma=g, method="exact",
                         cost_s=COST_S["exact"]))
    RC.write_index(D, rows)
    print(len(rows), "cards ->", D)


if __name__ == "__main__":
    main()
