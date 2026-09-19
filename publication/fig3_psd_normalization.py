"""Poster figure 3 -- hard normalisation buys stability at the wrong price.

Rescaling p(v,t) to unit mass every step is the obvious fix, and it does
keep the norm exact. But the rescaling acts on the fluctuations as well as
the mean, and the damage is concentrated at low frequency: the spectrum
settles on the Poisson level instead of dropping to the CV-limited plateau
that the analytical result and the microscopic network both show.

Curves are means over independent seeds, bands are plus or minus one
standard deviation across them.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pipeline                                             # noqa: E402
import pubstyle as ps                                       # noqa: E402
from pipeline import val                                    # noqa: E402

RUN_NORM = "psd_renormalise"     # METHOD = 2
RUN_RELAX = "psd_relaxation"     # METHOD = 1

SHOW_RELAXATION = True
F_MIN, F_MAX = 3.0, 1e4     # start where the spectra actually begin


# Authored size, and how wide the figure is meant to be printed on the
# poster. The font scale is derived from the ratio, so every figure ends
# up with the same apparent text size. Change PLACED_CM if you give this
# panel a different amount of the sheet.
FIG_W, FIG_H = 8.8, 5.8
PLACED_CM = ps.POSTER_COLUMN_CM


def main():
    ps.set_style_for(FIG_W, PLACED_CM)

    norm = pipeline.load_ensemble(RUN_NORM)
    relax = pipeline.load_ensemble(RUN_RELAX)

    f_mic = np.asarray(norm["freq_micro"])
    m_mic, lo_mic, hi_mic = pipeline.mean_band(norm["psd_micro_all"])
    f_nrm = np.asarray(norm["freq_meso"])
    m_nrm, lo_nrm, hi_nrm = pipeline.mean_band(norm["psd_meso_all"])
    f_rel = np.asarray(relax["freq_meso"])
    m_rel, lo_rel, hi_rel = pipeline.mean_band(relax["psd_meso_all"])

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ps.log_grid(ax)

    # --- asymptotic levels ------------------------------------------------
    for level, text in ((val(norm, "psd_poisson_level"), r"$r_0/N$"),
                        (val(norm, "psd_lowfreq_level"), r"$r_0 CV^2/N$")):
        ax.axhline(level, color=ps.C["level"], lw=ps.LW_THIN, ls=(0, (2, 2.5)),
                   zorder=2)
        ps.annotate(ax, F_MAX * 0.97, level * 1.06, text, ps.C["level"],
                    va="bottom", ha="right")

    # --- references -------------------------------------------------------
    ax.fill_between(f_mic, lo_mic, hi_mic, color=ps.blend(ps.C["micro"], 0.30),
                    lw=0, zorder=2)
    ax.plot(f_mic, m_mic, color=ps.C["micro"], lw=ps.LW_MICRO, zorder=5,
            label=ps.LABEL["micro"])
    ax.plot(norm["freq_exact"], norm["psd_exact"], color=ps.C["exact"],
            lw=ps.LW_REF, zorder=8, label=ps.LABEL["exact"])

    # --- solver variants --------------------------------------------------
    if SHOW_RELAXATION:
        ax.fill_between(f_rel, lo_rel, hi_rel, color=ps.blend(ps.C["meso"], 0.22),
                        lw=0, zorder=3)
        ax.plot(f_rel, m_rel, color=ps.C["meso"], lw=ps.LW_DATA, zorder=6,
                label=r"Mesoscopic ($\Lambda$ term)")
    ax.fill_between(f_nrm, lo_nrm, hi_nrm, color=ps.blend(ps.C["meso_alt"], 0.22),
                    lw=0, zorder=4)
    ax.plot(f_nrm, m_nrm, color=ps.C["meso_alt"], lw=ps.LW_DATA,
            ls=(0, (5, 2)), zorder=7,
            label="Mesoscopic (normalization)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(F_MIN, F_MAX)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(r"$S_{AA}(f)$  (Hz)")
    ps.loglog_ticks(ax, label_minor_x=(F_MIN,))

    ax.legend(loc="upper left", bbox_to_anchor=(0.015, 0.99))

    fig.tight_layout()
    ps.save(fig, "fig3_psd_normalization", placed_cm=PLACED_CM)


if __name__ == "__main__":
    main()
