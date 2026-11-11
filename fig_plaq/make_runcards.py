"""fig_plaq: lossy 2x2 plaquette dynamics (main Fig. fig:plaq). Writes runcards/ (driver + exact) and index.csv."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import runcards as RC, models as M           # noqa: E402

# ----------------------------------------------------------------------------------------- parameters (only here)
SYSTEM = "plaq"
PANELS = [("mixed", 3.0), ("slater", 10.0), ("one_pair", 1.0), ("cdw", 3.0)]     # (initial state, gamma/J) of (a)-(d)
METHODS = ["HF", "2B", "TPPX", "mGW", "DSL"]
T_END, DT_OUT = 200.0, 0.1
TOL = RC.TOL_DEFAULT
CHECKPOINT_INTERVAL = 1.0                     # as the paper's plaquette runs (P/Q/G' checkpoints; not used here)
EXACT_OBS = ["N", "N_sigma", "n", "D", "S2", "rho1_eig"]
COST_DRIVER_S, COST_EXACT_S = 2, 10
# ---------------------------------------------------------------------------------------------------------------


def main():
    D = RC.fresh_tree(HERE / "runcards")
    RC.write_model(D, SYSTEM)
    rows = []
    for ic, g in PANELS:
        for m in METHODS:
            rid = RC.run_id(SYSTEM, ic, g, m)
            RC.write_driver_card(D, rid, SYSTEM, g, m, M.plaq_rho0(ic), T_END, DT_OUT, "g1g2", TOL, CHECKPOINT_INTERVAL)
            rows.append(dict(run_id=rid, kind="driver", system=SYSTEM, ic=ic, gamma=g, method=m, cost_s=COST_DRIVER_S))
        rid = RC.run_id(SYSTEM, ic, g, "exact")
        RC.write_exact_card(D, rid, SYSTEM, g, M.plaq_rho0(ic).real,
                            [{"type": "transient", "name": "transient", "grid": [{"t_end": T_END, "dt": DT_OUT}],
                              "observables": EXACT_OBS}])
        rows.append(dict(run_id=rid, kind="exact", system=SYSTEM, ic=ic, gamma=g, method="exact", cost_s=COST_EXACT_S))
    RC.write_index(D, rows)
    print(len(rows), "cards ->", D)


if __name__ == "__main__":
    main()
