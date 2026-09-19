"""Poster figure 1 -- the naive rate definition destroys probability mass.

Setting r(t) equal to the instantaneous outflux J_out(t) drains the density
within the first second: mass leaves through the threshold faster than the
stochastic re-injection at reset returns it, and nothing pushes the norm
back. The relaxation term supplies the missing flux and holds the norm at
one without damping the finite-size fluctuations.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pipeline                                             # noqa: E402
import pubstyle as ps                                       # noqa: E402

RUN_FAIL = "mass_flux"          # METHOD = 0
RUN_OK = "mass_relaxation"      # METHOD = 1

# Both runs are 10 s long. Nothing changes after the collapse, so the
# panel is cropped; say so in the caption rather than spending four
# fifths of the width on a flat line.
T_SHOW = 3.0


# Authored size, and how wide the figure is meant to be printed on the
# poster. The font scale is derived from the ratio, so every figure ends
# up with the same apparent text size. Change PLACED_CM if you give this
# panel a different amount of the sheet.
FIG_W, FIG_H = 6.4, 4.4
PLACED_CM = ps.POSTER_COLUMN_CM


def main():
    ps.set_style_for(FIG_W, PLACED_CM)

    fail = pipeline.load(RUN_FAIL)
    ok = pipeline.load(RUN_OK)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    ax.axhline(1.0, color=ps.C["level"], lw=ps.LW_THIN, ls=(0, (4, 3)),
               zorder=1)

    for data, colour, label in (
            (ok, ps.C["meso"], r"$r(t)=J_{\rm out}(t)+\Lambda\,[1-\|p\|]$"),
            (fail, ps.C["meso_alt"], r"$r(t)=J_{\rm out}(t)$")):
        t = np.asarray(data["t_ms"]) / 1000.0
        ax.plot(t, data["mass"], color=ps.blend(colour, ps.ALPHA_RAW),
                lw=ps.LW_RAW, zorder=2, solid_capstyle="butt")
        ax.plot(t, data["mass_smooth100"], color=colour, lw=ps.LW_DATA,
                zorder=4, label=label)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"Probability mass  $\|p\|$")
    ax.set_xlim(0, T_SHOW)
    ax.set_ylim(-0.04, 1.28)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_xticks([0, 1, 2, 3])

    ps.annotate(ax, T_SHOW * 0.99, 1.0, "unit mass", ps.C["level"],
                va="bottom", ha="right")

    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.55),
              fontsize=plt.rcParams["legend.fontsize"] * 1.30)

    fig.tight_layout()
    ps.save(fig, "fig1_mass_conservation", placed_cm=PLACED_CM)


if __name__ == "__main__":
    main()
