"""Poster figure 5 -- why a density description is worth the trouble.

The microscopic network integrates one equation per neuron, so its cost is
linear in the population. The solver integrates a density on a fixed grid,
so the population size enters only through the amplitude of the finite-size
noise and the cost is flat. This is the claim the abstract opens on.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pipeline                                             # noqa: E402
import pubstyle as ps                                       # noqa: E402

FIG_W, FIG_H = 6.4, 4.6
PLACED_CM = ps.POSTER_COLUMN_CM

CACHE_FILE = os.path.join(pipeline.CACHE, "benchmark.npz")


def main():
    ps.set_style_for(FIG_W, PLACED_CM)

    if not os.path.exists(CACHE_FILE):
        raise FileNotFoundError(
            "no benchmark data. Run: python publication/benchmark.py")
    d = np.load(CACHE_FILE)
    n = d["n"]
    meso, micro = d["meso"], d["micro"]

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ps.log_grid(ax)

    ax.plot(n, micro, color=ps.C["micro"], lw=ps.LW_MICRO, marker="o", ms=7,
            zorder=4, label=ps.LABEL["micro"])
    ax.plot(n, meso, color=ps.C["meso"], lw=ps.LW_DATA, marker="s", ms=7,
            zorder=5, label=ps.LABEL["meso"])

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Population size $N$")
    ax.set_ylabel("Wall-clock seconds\nper second simulated")
    ps.loglog_ticks(ax)
    ax.legend(loc="upper left", bbox_to_anchor=(0.015, 0.99),
              fontsize=plt.rcParams["legend.fontsize"] * 1.30)

    fig.tight_layout()
    ps.save(fig, "fig5_cost_scaling", placed_cm=PLACED_CM)


if __name__ == "__main__":
    main()
