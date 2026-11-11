"""fig_zeno: quantum Zeno crossover on the plaquette (main Fig. fig:zeno). Writes runcards/ and index.csv.
Driver: two segments per (initial state, gamma, method): fine (t <= 20, dt 0.005) and long (t <= 3000, dt 1).
Exact: one card per (initial state, gamma) with the transient on a grid containing both driver grids, the t -> infinity
limit, and (mixed, gamma >= 10) a short fine transient for the window mean of D/O."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import runcards as RC, models as M           # noqa: E402

# ----------------------------------------------------------------------------------------- parameters (only here)
SYSTEM = "plaq"
ICS = ["mixed", "neel"]
GAMMAS = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0]
METHODS = ["HF", "TPPX", "DSL"]
SEGMENTS = {"fine": (20.0, 0.005), "long": (3000.0, 1.0)}          # driver: (t_end, dt_out) [1/J]
TOL = RC.TOL_DEFAULT
CHECKPOINT_INTERVAL = 0.0
EXACT_GRID = {"mixed": [(20.0, 0.005), (3000.0, 1.0)],              # exact transient segments (t_end, dt)
              "neel": [(60.0, 0.005), (3000.0, 1.0)]}
EXACT_FINE = {"mixed": [10.0, 20.0, 30.0]}                          # extra task 'fine': t <= 60/gamma, dt 0.05/gamma
EXACT_OBS = ["N", "n", "D", "S2"]
COST_S = {"fine": 1, "long": 15, "exact_mixed": 400, "exact_neel": 60}
# ---------------------------------------------------------------------------------------------------------------


def main():
    D = RC.fresh_tree(HERE / "runcards")
    RC.write_model(D, SYSTEM)
    rows = []
    for ic in ICS:
        for g in GAMMAS:
            for m in METHODS:
                for seg, (T, dt) in SEGMENTS.items():
                    rid = RC.run_id(SYSTEM, ic, g, m, seg)
                    RC.write_driver_card(D, rid, SYSTEM, g, m, M.plaq_rho0(ic), T, dt, "g1g2", TOL, CHECKPOINT_INTERVAL)
                    rows.append(dict(run_id=rid, kind="driver", system=SYSTEM, ic=ic, gamma=g, method=m, segment=seg,
                                     cost_s=COST_S[seg]))
            tasks = [{"type": "transient", "name": "transient",
                      "grid": [{"t_end": T, "dt": dt} for T, dt in EXACT_GRID[ic]], "observables": EXACT_OBS},
                     {"type": "n_inf", "name": "n_inf", "observables": ["N", "D", "S2"]}]
            if g in EXACT_FINE.get(ic, []):
                tasks.append({"type": "transient", "name": "fine", "grid": [{"t_end": 60.0 / g, "dt": 0.05 / g}],
                              "observables": EXACT_OBS})
            rid = RC.run_id(SYSTEM, ic, g, "exact")
            RC.write_exact_card(D, rid, SYSTEM, g, M.plaq_rho0(ic).real, tasks)
            rows.append(dict(run_id=rid, kind="exact", system=SYSTEM, ic=ic, gamma=g, method="exact",
                             cost_s=COST_S["exact_" + ic]))
    RC.write_index(D, rows)
    print(len(rows), "cards ->", D)


if __name__ == "__main__":
    main()
