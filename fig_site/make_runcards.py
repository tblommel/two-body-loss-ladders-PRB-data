"""fig_site: single lossy site (main Fig. fig:site). Writes runcards/ (driver + exact cards) and runcards/index.csv."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import runcards as RC, models as M           # noqa: E402

# ----------------------------------------------------------------------------------------- parameters (only here)
SYSTEM = "site"
GAMMA = 1.0
N0 = [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,
      0.98, 1.0]                                    # initial filling per spin (driver runs)
N0_EXACT_EXTRA = [k / 40 for k in range(41)]        # exact runs also on this grid (exact curve of panel c)
METHODS = ["HF", "2B", "TPPX", "mGW", "DSL"]
GT_END, DGT = 2000.0, 0.05                          # gamma t range and output step (in units of gamma t)
TOL = RC.TOL_DEFAULT
CHECKPOINT_INTERVAL = 0.0
EXACT_OBS = ["N", "N_sigma", "n", "D", "S2", "rho1_eig"]
COST_DRIVER_S, COST_EXACT_S = 8, 15
# ---------------------------------------------------------------------------------------------------------------


def main():
    D = RC.fresh_tree(HERE / "runcards")
    RC.write_model(D, SYSTEM)
    rows = []
    for n0 in N0:
        for m in METHODS:
            rid = RC.run_id(SYSTEM, f"n{n0:g}", GAMMA, m)
            RC.write_driver_card(D, rid, SYSTEM, GAMMA, m, M.site_rho0(n0), GT_END / GAMMA, DGT / GAMMA, "g1g2", TOL,
                                 CHECKPOINT_INTERVAL)
            rows.append(dict(run_id=rid, kind="driver", system=SYSTEM, ic=f"n{n0:g}", gamma=GAMMA, method=m,
                             cost_s=COST_DRIVER_S))
    for n0 in sorted({round(x, 12) for x in N0 + N0_EXACT_EXTRA}):
        rid = RC.run_id(SYSTEM, f"n{n0:g}", GAMMA, "exact")
        tasks = [{"type": "transient", "name": "transient", "grid": [{"t_end": GT_END / GAMMA, "dt": DGT / GAMMA}],
                  "observables": EXACT_OBS},
                 {"type": "n_inf", "name": "n_inf", "observables": ["N", "N_sigma", "n", "D"]}]
        RC.write_exact_card(D, rid, SYSTEM, GAMMA, M.site_rho0(n0).real, tasks)
        rows.append(dict(run_id=rid, kind="exact", system=SYSTEM, ic=f"n{n0:g}", gamma=GAMMA, method="exact",
                         cost_s=COST_EXACT_S))
    RC.write_index(D, rows)
    print(len(rows), "cards ->", D)


if __name__ == "__main__":
    main()
