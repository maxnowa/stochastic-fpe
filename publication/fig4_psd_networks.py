"""Poster figure 4 -- the relaxation term carries over to recurrent networks.

All-to-all coupling feeds the population rate back as a time-dependent
drift. Sparse random coupling adds, on top of that, a rate-dependent
diffusion term from the variance of the quenched weights. In both cases the
mesoscopic spectrum follows the microscopic network across four decades.

Both conditions carry the corrected time step. The solver's step-size
heuristic assumed an anticipated rate that made the step about fifteen times
smaller than the drive needs, which produced a spurious low-frequency
deficit of a third. See configs.CFL_FIX.

Curves are means over independent seeds, bands are plus or minus one
standard deviation across them. Run-to-run scatter at low frequency is about
three percent, so a single realisation cannot support a claim at that level.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pipeline                                             # noqa: E402
import pubstyle as ps                                       # noqa: E402
from pipeline import val                                    # noqa: E402

PANELS = [
    ("net_fc_supra", "Fully connected", r"$p=1$"),
    ("net_rc_supra", "Random connectivity", r"$p=0.5$"),
]

F_MIN, F_MAX = 3.0, 1e4     # start where the spectra actually begin


# Authored size, and how wide the figure is meant to be printed on the
# poster. The font scale is derived from the ratio, so every figure ends
# up with the same apparent text size. Change PLACED_CM if you give this
# panel a different amount of the sheet.
FIG_W, FIG_H = 11.6, 4.9
PLACED_CM = ps.POSTER_COLUMN_CM


def main():
    ps.set_style_for(FIG_W, PLACED_CM)

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H), sharey=True)

    for col, (tag, title, subtitle) in enumerate(PANELS):
        data = pipeline.load_ensemble(tag)
        ax = axes[col]
        ps.log_grid(ax)

        f_mic = np.asarray(data["freq_micro"])
        f_mes = np.asarray(data["freq_meso"])
        m_mic, lo_mic, hi_mic = pipeline.mean_band(data["psd_micro_all"])
        m_mes, lo_mes, hi_mes = pipeline.mean_band(data["psd_meso_all"])

        ax.fill_between(f_mic, lo_mic, hi_mic, color=ps.blend(ps.C["micro"], 0.30),
                        lw=0, zorder=2)
        ax.fill_between(f_mes, lo_mes, hi_mes, color=ps.blend(ps.C["meso"], 0.22),
                        lw=0, zorder=3)
        ax.plot(f_mic, m_mic, color=ps.C["micro"], lw=ps.LW_MICRO, zorder=4,
                label=ps.LABEL["micro"])
        ax.plot(f_mes, m_mes, color=ps.C["meso"], lw=ps.LW_DATA, zorder=5,
                label=ps.LABEL["meso"])

        lsd, _ = pipeline.deviation_metrics(f_mic, m_mic, f_mes, m_mes)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(F_MIN, F_MAX)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_title(f"{title}\n{subtitle},  $w={val(data, 'param_W'):g}$",
                     pad=10)
        ps.loglog_ticks(ax, label_minor_x=(F_MIN,))
        ps.panel_tag(ax, "AB"[col], dx=-0.17 if col == 0 else -0.06)

        ps.annotate(ax, 0.985, 0.03, f"LSD {lsd:.2f}", ps.C["ink_soft"],
                    transform=ax.transAxes, va="bottom", ha="right")

    axes[0].set_ylabel(r"$S_{AA}(f)$  (Hz)")
    axes[0].legend(loc="upper left", bbox_to_anchor=(0.015, 0.99))

    fig.tight_layout(w_pad=2.2)
    ps.save(fig, "fig4_psd_networks", placed_cm=PLACED_CM)


if __name__ == "__main__":
    main()
