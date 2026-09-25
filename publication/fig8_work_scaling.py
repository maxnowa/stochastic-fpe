"""Poster figure 8 -- the cost claim without a machine in it.

Figure 5 measures wall-clock on one laptop, which invites the fair objection
that a parallel implementation of the network would change the picture. This
figure removes the machine: it counts state updates per unit of simulated
time, which is invariant to language, hardware and degree of parallelism.
Parallelism redistributes work across processors; it does not reduce it.

The solver advances a fixed 200-point grid whatever the population size. The
network advances one state variable per neuron. So the work ratio is exactly
linear in N, reaching 338 at N = 300,000 -- twice the factor the wall-clock
measurement shows, because the vectorised network reaches a higher update
throughput than the solver, which does a tridiagonal solve and a flux limiter
per update against the network's single Euler step.

Honest limit, worth knowing if asked: both methods have an irreducible
sequential chain of time steps, and the solver's CFL condition gives it 4.4
times more of them. Given unlimited processors the network finishes sooner.
The claim this figure supports is about work and memory, not about elapsed
time on arbitrarily large hardware.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import configs                                              # noqa: E402
import pipeline                                             # noqa: E402
import pubstyle as ps                                       # noqa: E402

FIG_W, FIG_H = 6.4, 4.6
PLACED_CM = ps.POSTER_COLUMN_CM

N_RANGE = np.logspace(2, 5.7, 200)      # 100 to 500,000


def main():
    ps.set_style_for(FIG_W, PLACED_CM)

    params = dict(configs.BASE)
    meso, micro = pipeline.state_updates_per_ms(params, N_RANGE)
    meso_line = np.full_like(N_RANGE, meso)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ps.log_grid(ax)

    ax.plot(N_RANGE, micro, color=ps.C["micro"], lw=ps.LW_MICRO, zorder=4,
            label=ps.LABEL["micro"])
    ax.plot(N_RANGE, meso_line, color=ps.C["meso"], lw=ps.LW_DATA, zorder=5,
            label=ps.LABEL["meso"])

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(N_RANGE[0], N_RANGE[-1])
    ax.set_xlabel("Population size $N$")
    ax.set_ylabel("State updates\nper second simulated")
    ps.loglog_ticks(ax)

    ps.annotate(ax, 0.97, 0.10, "independent of $N$", ps.C["meso"],
                transform=ax.transAxes, va="bottom", ha="right")
    ax.legend(loc="upper left", bbox_to_anchor=(0.015, 0.99),
              fontsize=plt.rcParams["legend.fontsize"] * 1.30)

    fig.tight_layout()
    ps.save(fig, "fig8_work_scaling", placed_cm=PLACED_CM)


if __name__ == "__main__":
    main()
