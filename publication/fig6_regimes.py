"""Poster figure 6 -- the solver holds across all four operating regimes.

Columns are the drive relative to threshold, rows are the noise intensity.
Unconnected, so the analytical finite-size spectrum is available in every
panel and the solver is being checked against theory rather than against
another simulation.

Subthreshold low noise is the demanding corner. The firing rate there
depends exponentially on the drive, so the relaxation rate responds about
six times more steeply than it does above threshold, and the rate itself is
an order of magnitude lower.

Curves are means over independent seeds; bands are one standard deviation
across them. The quoted deviation is against the analytical spectrum, which
is exact here; the microscopic network is shown as a second reference but is
itself discretisation-limited (see the note in panel()).
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pipeline                                             # noqa: E402
import pubstyle as ps                                       # noqa: E402
from pipeline import val                                    # noqa: E402

FIG_W, FIG_H = 12.0, 9.2
PLACED_CM = ps.POSTER_COLUMN_CM

# (row, col) -> run. Columns: subthreshold, suprathreshold.
#               Rows:    low noise, high noise.
PANELS = [["regime_sub_low", "regime_supra_low"],
          ["regime_sub_high", "regime_supra_high"]]

COL_TITLE = ["Subthreshold", "Suprathreshold"]
ROW_LABEL = ["Low noise", "High noise"]

F_MIN, F_MAX = 3.0, 1e4

# Multiplies the top of each panel's y-range, to keep the mu/D annotation
# off the curve.
HEADROOM = 1.7

# Fraction of the figure height reserved at the bottom for the shared legend.
LEGEND_BAND = 0.055


def panel(ax, tag):
    d = pipeline.load_ensemble(tag)
    ps.log_grid(ax)

    f_mic = np.asarray(d["freq_micro"])
    f_mes = np.asarray(d["freq_meso"])
    m_mic, lo_mic, hi_mic = pipeline.mean_band(d["psd_micro_all"])
    m_mes, lo_mes, hi_mes = pipeline.mean_band(d["psd_meso_all"])

    ax.fill_between(f_mic, lo_mic, hi_mic,
                    color=ps.blend(ps.C["micro"], 0.30), lw=0, zorder=2)
    ax.fill_between(f_mes, lo_mes, hi_mes,
                    color=ps.blend(ps.C["meso"], 0.22), lw=0, zorder=3)
    ax.plot(f_mic, m_mic, color=ps.C["micro"], lw=ps.LW_MICRO, zorder=5,
            label=ps.LABEL["micro"])
    ax.plot(d["freq_exact"], d["psd_exact"], color=ps.C["exact"],
            lw=ps.LW_REF, zorder=7, label=ps.LABEL["exact"])
    ax.plot(f_mes, m_mes, color=ps.C["meso"], lw=ps.LW_DATA, zorder=6,
            label=ps.LABEL["meso"])

    # Against theory, not against the microscopic run. These are unconnected,
    # so the analytical spectrum is exact, whereas the microscopic reference
    # carries its own Euler-Maruyama error: its rate sits 7.5% below the
    # Siegert value in the subthreshold low-noise corner, and refining the
    # network step shrinks that as sqrt(dt) (-7.45%, -3.69%, -2.05% over two
    # factors of four), extrapolating to -0.4% at dt -> 0. That deficit is
    # the visible gap between the grey and black curves in panel A.
    lsd, _ = pipeline.deviation_metrics(
        np.asarray(d["freq_exact"]), np.asarray(d["psd_exact"]), f_mes, m_mes)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(F_MIN, F_MAX)

    # Headroom so the parameter annotation clears the curve. Panel A's
    # spectrum overshoots to within a few percent of the top otherwise.
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * HEADROOM)

    ps.loglog_ticks(ax, label_minor_x=(F_MIN,))

    mu = val(d, "param_MU")
    dd = val(d, "param_D")
    ps.annotate(ax, 0.03, 0.95,
                f"$\\mu={mu:g}$,  $D={dd:g}$", ps.C["ink"],
                transform=ax.transAxes, va="top", ha="left")
    ps.annotate(ax, 0.985, 0.03, f"LSD {lsd:.2f}", ps.C["ink_soft"],
                transform=ax.transAxes, va="bottom", ha="right")

    return lsd


def main():
    ps.set_style_for(FIG_W, PLACED_CM)
    fig, axes = plt.subplots(2, 2, figsize=(FIG_W, FIG_H),
                             sharex=True, sharey=False)

    tags = "ABCD"
    for r in range(2):
        for c in range(2):
            ax = axes[r, c]
            panel(ax, PANELS[r][c])
            ps.panel_tag(ax, tags[r * 2 + c],
                         dx=-0.20 if c == 0 else -0.12)
            if r == 0:
                ax.set_title(COL_TITLE[c], pad=10)
            if r == 1:
                ax.set_xlabel("Frequency (Hz)")
            if c == 0:
                ax.set_ylabel(f"{ROW_LABEL[r]}\n$S_{{AA}}(f)$  (Hz)")

    # All four panels carry the same three series, so one shared legend
    # below the figure. Inside a panel it lands on the data in at least one
    # regime whichever corner it is put in.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               frameon=False, handlelength=1.6, columnspacing=2.4,
               bbox_to_anchor=(0.5, 0.0))

    fig.tight_layout(h_pad=1.8, w_pad=2.4, rect=(0.0, LEGEND_BAND, 1.0, 1.0))
    ps.save(fig, "fig6_regimes", placed_cm=PLACED_CM)


if __name__ == "__main__":
    main()
