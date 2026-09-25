"""Shared panel for the supplementary spectra.

All four supplementary figures are the same object: a 2x2 grid of mesoscopic
against microscopic spectra, differing only in what the four panels vary. No
analytical curve -- it exists only for the unconnected case, so these are
checked against the network.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pipeline                                             # noqa: E402
import pubstyle as ps                                       # noqa: E402

FIG_W, FIG_H = 12.0, 9.2
PLACED_CM = ps.POSTER_COLUMN_CM
F_MIN, F_MAX = 3.0, 1e4
HEADROOM = 1.7


def panel(ax, tag, note):
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
    ax.plot(f_mic, m_mic, color=ps.C["micro"], lw=ps.LW_MICRO, zorder=4,
            label=ps.LABEL["micro"])
    ax.plot(f_mes, m_mes, color=ps.C["meso"], lw=ps.LW_DATA, zorder=5,
            label=ps.LABEL["meso"])

    lsd, _ = pipeline.deviation_metrics(f_mic, m_mic, f_mes, m_mes)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(F_MIN, F_MAX)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * HEADROOM)
    ps.loglog_ticks(ax, label_minor_x=(F_MIN,))

    ps.annotate(ax, 0.03, 0.95, note, ps.C["ink"],
                transform=ax.transAxes, va="top", ha="left")
    ps.annotate(ax, 0.985, 0.03, f"LSD {lsd:.2f}", ps.C["ink_soft"],
                transform=ax.transAxes, va="bottom", ha="right")
    return lsd


def grid(name, panels, col_titles=None, row_labels=None, ylabel=None):
    """panels: 2x2 nested list of (tag, note)."""
    ps.set_style_for(FIG_W, PLACED_CM)
    fig, axes = plt.subplots(2, 2, figsize=(FIG_W, FIG_H),
                             sharex=True, sharey=False)

    tags = "ABCD"
    for r in range(2):
        for c in range(2):
            ax = axes[r, c]
            tag, note = panels[r][c]
            panel(ax, tag, note)
            ps.panel_tag(ax, tags[r * 2 + c], dx=-0.20 if c == 0 else -0.12)
            if r == 0 and col_titles:
                ax.set_title(col_titles[c], pad=10)
            if r == 1:
                ax.set_xlabel("Frequency (Hz)")
            if c == 0:
                lead = f"{row_labels[r]}\n" if row_labels else ""
                ax.set_ylabel(f"{lead}{ylabel or r'$S_{AA}(f)$  (Hz)'}")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               handlelength=1.6, columnspacing=2.4, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(h_pad=1.8, w_pad=2.4, rect=(0.0, 0.055, 1.0, 1.0))
    ps.save(fig, name, placed_cm=PLACED_CM)
