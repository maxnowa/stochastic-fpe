"""Poster figure 7 -- sparse random coupling changes the spectrum, and the
solver changes with it.

Figure 4 makes the same comparison at p = 0.5 and the two panels come out
indistinguishable, which is a correct result rather than a failure: the mean
input is p-independent by the 1/p renormalisation, so the only thing random
connectivity adds is a diffusion term from the variance of the quenched
in-degree, and at p = 0.5 that is a fraction of a percent of the total. The
extra diffusion goes as w^2 r / (2 D k) with k = pN the number of inputs per
neuron, so separating the topologies needs few inputs per neuron while
keeping the feedback noise tolerable needs a large enough population.

N = 1000 with ten inputs per neuron is the compromise, at a connection
probability of 0.01 that is biologically defensible. The scout measured the
topologies separating by 0.194 there in the solver against 0.196 in the
microscopic network.

The fully connected mesoscopic curve is repeated in panel B as a fixed
reference, so the departure is visible inside the panel rather than only
across the gap between them.

Three seeds, and the microscopic reference here is the plain Euler-Maruyama
scheme without the bridge correction used in figure 6 -- its recurrent sparse
network is the most expensive thing in the set, and the uncorrected rate
deficit in this suprathreshold regime is around one percent. Bands are one
standard deviation across seeds.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import configs                                              # noqa: E402
import pipeline                                             # noqa: E402
import pubstyle as ps                                       # noqa: E402
from pipeline import val                                    # noqa: E402

PANELS = [
    ("sparse_fc", "Fully connected"),
    ("sparse_rc", "Sparse random"),
]

F_MIN, F_MAX = 3.0, 1e4

FIG_W, FIG_H = 11.6, 5.1
PLACED_CM = ps.POSTER_COLUMN_CM


def main():
    ps.set_style_for(FIG_W, PLACED_CM)
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H), sharey=True)

    reference = None        # fully connected mesoscopic, repeated in panel B

    for col, (tag, title) in enumerate(PANELS):
        data = pipeline.load_ensemble(tag)
        ax = axes[col]
        ps.log_grid(ax)

        f_mic = np.asarray(data["freq_micro"])
        f_mes = np.asarray(data["freq_meso"])
        m_mic, lo_mic, hi_mic = pipeline.mean_band(data["psd_micro_all"])
        m_mes, lo_mes, hi_mes = pipeline.mean_band(data["psd_meso_all"])

        if col == 1 and reference is not None:
            ax.plot(reference[0], reference[1], color=ps.C["level"],
                    lw=ps.LW_THIN, ls=(0, (5, 3)), zorder=6,
                    label="Fully connected")

        ax.fill_between(f_mic, lo_mic, hi_mic,
                        color=ps.blend(ps.C["micro"], 0.30), lw=0, zorder=2)
        ax.fill_between(f_mes, lo_mes, hi_mes,
                        color=ps.blend(ps.C["meso"], 0.22), lw=0, zorder=3)
        ax.plot(f_mic, m_mic, color=ps.C["micro"], lw=ps.LW_MICRO, zorder=4,
                label=ps.LABEL["micro"])
        ax.plot(f_mes, m_mes, color=ps.C["meso"], lw=ps.LW_DATA, zorder=5,
                label=ps.LABEL["meso"])

        if col == 0:
            reference = (f_mes, m_mes)

        lsd, _ = pipeline.deviation_metrics(f_mic, m_mic, f_mes, m_mes)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(F_MIN, F_MAX)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_title(f"{title}\n"
                     f"$p={val(data, 'param_CONNECTIVITY'):g}$,  "
                     f"$w={val(data, 'param_W'):g}$", pad=10)
        ps.loglog_ticks(ax, label_minor_x=(F_MIN,))
        ps.panel_tag(ax, "AB"[col], dx=-0.17 if col == 0 else -0.06)

        ps.annotate(ax, 0.985, 0.03, f"LSD {lsd:.2f}", ps.C["ink_soft"],
                    transform=ax.transAxes, va="bottom", ha="right")

    axes[0].set_ylabel(r"$S_{AA}(f)$  (Hz)")
    axes[0].legend(loc="upper left", bbox_to_anchor=(0.015, 0.99))
    axes[1].legend(loc="upper left", bbox_to_anchor=(0.015, 0.99))

    fig.tight_layout(w_pad=2.2)
    ps.save(fig, "fig7_sparse_connectivity", placed_cm=PLACED_CM)


if __name__ == "__main__":
    main()
