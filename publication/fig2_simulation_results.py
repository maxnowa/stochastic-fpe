"""Poster figure 2 -- the relaxation term is accurate, not just stable.

Restoring mass could in principle have been bought by distorting the density
or biasing the rate. It is not: the stationary density has the expected
shape in both regimes, and the mean activity reproduces the exact Siegert
rate to well under a percent while keeping the finite-size fluctuations.
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

COLUMNS = [
    ("sim_supra_lownoise", "Suprathreshold, low noise",
     r"$\mu=1.2$,  $D=0.01$"),
    ("sim_sub_highnoise", "Subthreshold, high noise",
     r"$\mu=0.8$,  $D=0.1$"),
]

EXCERPT_MS = 100.0   # width of the activity window shown
BIN_MS = 0.5         # bin width for the population rate


def panel_density(ax, data):
    v = np.asarray(data["v_grid"])
    snaps = np.asarray(data["density_snapshots"])

    # Transient snapshots, fading in towards the stationary solution.
    for k in range(1, len(snaps) - 1):
        ax.plot(v, snaps[k],
                color=ps.blend(ps.C["meso"], 0.12 + 0.07 * k), lw=ps.LW_RAW,
                zorder=2)
    ax.plot(v, data["density_stationary"], color=ps.C["meso"],
            lw=ps.LW_DATA, zorder=4)

    v_th = val(data, "param_V_TH")
    ax.axvline(v_th, color=ps.C["level"], lw=ps.LW_THIN, ls=(0, (4, 3)),
               zorder=1)
    ax.set_xlim(v[0], configs.V_MAX)
    ax.set_ylim(bottom=0)
    ps.annotate(ax, v_th, ax.get_ylim()[1], r"$V_{\rm th}$ ", ps.C["level"],
                va="top", ha="right")
    ax.set_xlabel("Membrane potential $v$")


def panel_activity(ax, data):
    """Population activity, binned.

    A(t) carries a white-noise term, so its value at a single solver step is
    not a rate -- its variance diverges as the step shrinks. Binning at a
    fixed window gives the quantity an experimentalist would actually
    measure, and makes the finite-size fluctuation size meaningful.
    """
    dt = val(data, "excerpt_dt_ms")
    t = np.asarray(data["excerpt_t_ms"])
    a = np.asarray(data["excerpt_rate_hz"])
    sel = t <= EXCERPT_MS
    t, a = t[sel], a[sel]

    per_bin = max(1, int(round(BIN_MS / dt)))
    n_bins = len(a) // per_bin
    binned = a[:n_bins * per_bin].reshape(n_bins, per_bin).mean(axis=1)
    t_bin = (np.arange(n_bins) + 0.5) * per_bin * dt

    ax.plot(t_bin, binned, color=ps.C["meso"], lw=ps.LW_DATA * 0.6,
            zorder=4)

    siegert = val(data, "rate_siegert_hz")
    ax.axhline(siegert, color=ps.C["exact"], lw=ps.LW_THIN, ls=(0, (4, 3)),
               zorder=5)

    ax.set_xlim(0, EXCERPT_MS)
    lo, hi = binned.min(), binned.max()
    pad = 0.35 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Time (ms)")


# Authored size, and how wide the figure is meant to be printed on the
# poster. The font scale is derived from the ratio, so every figure ends
# up with the same apparent text size. Change PLACED_CM if you give this
# panel a different amount of the sheet.
FIG_W, FIG_H = 10.6, 6.6
PLACED_CM = ps.POSTER_COLUMN_CM


def main():
    ps.set_style_for(FIG_W, PLACED_CM)

    fig, axes = plt.subplots(2, 2, figsize=(FIG_W, FIG_H))

    for col, (tag, title, subtitle) in enumerate(COLUMNS):
        data = pipeline.load(tag)

        axes[0, col].set_title(f"{title}\n{subtitle}", pad=10)
        panel_density(axes[0, col], data)
        panel_activity(axes[1, col], data)

        # The dashed line is the exact stationary rate and the axis carries
        # the values, so only the deviation needs stating.
        mean_hz = val(data, "rate_mean_hz")
        siegert = val(data, "rate_siegert_hz")
        err = abs(mean_hz - siegert) / siegert * 100.0
        ps.annotate(axes[1, col], 0.98, 0.96, f"{err:.1f}% from theory",
                    ps.C["ink_soft"], transform=axes[1, col].transAxes,
                    va="top", ha="right")

        ps.panel_tag(axes[0, col], "AB"[col],
                     dx=-0.20 if col == 0 else -0.14)
        ps.panel_tag(axes[1, col], "CD"[col],
                     dx=-0.20 if col == 0 else -0.14)

    axes[0, 0].set_ylabel("$p(v)$")
    axes[1, 0].set_ylabel("$A(t)$  (Hz)")

    fig.tight_layout(h_pad=2.0, w_pad=2.6)
    ps.save(fig, "fig2_simulation_results", placed_cm=PLACED_CM)


if __name__ == "__main__":
    main()
