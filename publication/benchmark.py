"""Cost of the two methods as the population grows.

The mesoscopic solver integrates a density on a fixed grid, so the number of
neurons enters only through the amplitude of the finite-size noise and the
cost does not depend on it. The microscopic network integrates one equation
per neuron, so its cost is linear in the population. That is the claim the
abstract opens on, and this measures it.

Honest caveat, which belongs in the caption: the solver is C and the
microscopic reference is vectorised numpy, so the absolute crossing point
flatters the solver. The scaling, flat against linear, is a property of the
methods and does not depend on the language.

Run:  python publication/benchmark.py
"""

import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import configs                                              # noqa: E402
import pipeline                                             # noqa: E402

# Short runs: we are measuring throughput, not statistics.
T_MAX_MS = 1000.0

# Unconnected, so the microscopic side stays O(N) rather than needing an
# N-by-N connectivity matrix, which would run out of memory long before the
# scaling becomes interesting.
N_VALUES = [100, 300, 1000, 3000, 10000, 30000, 100000, 300000]

CACHE_FILE = os.path.join(pipeline.CACHE, "benchmark.npz")


def time_solver(n_neurons, repeats=2):
    params = dict(configs.RUNS["psd_relaxation"])
    params["N_NEURONS"] = int(n_neurons)
    params["T_MAX"] = T_MAX_MS
    best = np.inf
    for _ in range(repeats):
        workdir = pipeline.run_solver(f"bench_meso_{n_neurons}", params,
                                      verbose=False)
        shutil.rmtree(workdir, ignore_errors=True)
        best = min(best, pipeline.LAST_RUN["seconds"])
    return best


def time_micro(n_neurons):
    params = dict(configs.RUNS["psd_relaxation"])
    params["N_NEURONS"] = int(n_neurons)
    t0 = time.time()
    act = pipeline.run_micro(params, T_MAX_MS, seed=1, verbose=False)
    elapsed = time.time() - t0
    del act
    return elapsed


def main():
    sim_seconds = T_MAX_MS / 1000.0
    meso, micro = [], []

    print(f"Timing {T_MAX_MS:.0f} ms of simulated time at each population "
          f"size.\n")
    print(f"{'N':>8s} {'mesoscopic':>12s} {'microscopic':>13s} {'ratio':>8s}")
    print("-" * 46)
    for n in N_VALUES:
        t_meso = time_solver(n) / sim_seconds
        t_micro = time_micro(n) / sim_seconds
        meso.append(t_meso)
        micro.append(t_micro)
        print(f"{n:8d} {t_meso:11.2f}s {t_micro:12.2f}s "
              f"{t_micro / t_meso:8.2f}")

    meso = np.array(meso)
    micro = np.array(micro)
    n_arr = np.array(N_VALUES, dtype=float)

    # Scaling exponents, fitted over the upper half where overheads no longer
    # dominate.
    upper = n_arr >= 1000
    slope_meso = np.polyfit(np.log(n_arr[upper]), np.log(meso[upper]), 1)[0]
    slope_micro = np.polyfit(np.log(n_arr[upper]), np.log(micro[upper]), 1)[0]
    print(f"\nscaling above N = 1000:")
    print(f"  mesoscopic   cost ~ N^{slope_meso:+.2f}   (expected 0)")
    print(f"  microscopic  cost ~ N^{slope_micro:+.2f}   (expected 1)")

    # Where the two cross, from the fitted lines.
    cross = np.nan
    if slope_micro > slope_meso:
        a = np.polyfit(np.log(n_arr[upper]), np.log(micro[upper]), 1)
        b = np.polyfit(np.log(n_arr[upper]), np.log(meso[upper]), 1)
        cross = float(np.exp((b[1] - a[1]) / (a[0] - b[0])))
        print(f"  crossing at N ~ {cross:,.0f}")

    np.savez_compressed(CACHE_FILE, n=n_arr, meso=meso, micro=micro,
                        slope_meso=slope_meso, slope_micro=slope_micro,
                        crossing=cross, t_max_ms=T_MAX_MS,
                        grid_n=configs.BASE["GRID_N"])
    print(f"\ncached -> {CACHE_FILE}")


if __name__ == "__main__":
    main()
