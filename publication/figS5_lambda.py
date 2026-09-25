"""Supplementary figure 5 -- the relaxation rate during a recurrent run.

Lambda is read from the lookup table every step, so in a recurrent network it
follows the fluctuating drive rather than sitting at the static r0/CV the
unconnected runs use. This is a 20 ms excerpt, well clear of the startup
transient.

Two things the excerpt shows. Lambda is not tracking a slow envelope -- it is
being driven by the per-step feedback noise, and scatters by 37% about its
mean, because mu_eff jitters by 0.12 against a lookup grid whose spacing in mu
is 0.016, so consecutive steps land many cells apart.

And its mean sits 27% above the static value. Almost all of that is the drive
itself: recurrent excitation raises the mean drive from 1.2 to 1.27, worth 25
points. The convexity of lambda in the drive -- averaging a convex function
over a noisy argument -- adds only the remaining 2. At stronger coupling that
second term is what grows.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pipeline                                             # noqa: E402
import pubstyle as ps                                       # noqa: E402

FIG_W, FIG_H = 11.6, 6.6
PLACED_CM = ps.POSTER_COLUMN_CM

CACHE_FILE = os.path.join(pipeline.CACHE, "lambda_trace.npz")
SHOW_MS = 2.0          # at 904 ns a longer window renders as a solid block
SMOOTH_MS = 0.05       # envelope, to show the scatter has no slow structure


def main():
    ps.set_style_for(FIG_W, PLACED_CM)

    if not os.path.exists(CACHE_FILE):
        raise FileNotFoundError(
            "no trace. Run: python3 publication/lambda_trace.py")
    d = np.load(CACHE_FILE, allow_pickle=True)

    t = np.asarray(d["t_ms"])
    keep = t <= SHOW_MS
    t = t[keep]
    mu = np.asarray(d["mu_eff"])[keep]
    lam = np.asarray(d["lambda"])[keep]
    static = float(d["lambda_static"])

    win = max(2, int(SMOOTH_MS / float(d["dt_ms"])))

    fig, axes = plt.subplots(2, 1, figsize=(FIG_W, FIG_H), sharex=True)

    ax = axes[0]
    ax.plot(t, mu, color=ps.blend(ps.C["meso_alt"], 0.35), lw=ps.LW_RAW,
            rasterized=True, zorder=2)
    ax.plot(t, pipeline.moving_average(mu, win), color=ps.C["meso_alt"],
            lw=ps.LW_DATA, zorder=4)
    ax.axhline(mu.mean(), color=ps.C["level"], lw=ps.LW_REF,
               ls=(0, (5, 3)), zorder=5)
    ps.annotate(ax, 0.995, 0.94, f"mean {mu.mean():.2f}", ps.C["ink_soft"],
                transform=ax.transAxes, va="top", ha="right")
    ax.set_ylabel(r"$\mu_{\mathrm{eff}}(t)$")

    ax = axes[1]
    ax.plot(t, lam, color=ps.blend(ps.C["meso"], 0.30), lw=ps.LW_RAW,
            rasterized=True, zorder=2)
    ax.plot(t, pipeline.moving_average(lam, win), color=ps.C["meso"],
            lw=ps.LW_DATA, zorder=4)
    ax.axhline(lam.mean(), color=ps.C["level"], lw=ps.LW_REF,
               ls=(0, (5, 3)), zorder=5)
    ax.axhline(static, color=ps.C["exact"], lw=ps.LW_REF,
               ls=(0, (1.5, 2.5)), zorder=6)
    ps.annotate(ax, 0.995, 0.94,
                f"mean {lam.mean():.2f}   (static $r_0/CV$ = {static:.2f})",
                ps.C["ink_soft"], transform=ax.transAxes,
                va="top", ha="right")
    ax.set_ylabel(r"$\Lambda(t)$")
    ax.set_xlabel("Time (ms)")

    for i, ax in enumerate(axes):
        ax.set_xlim(0, SHOW_MS)
        ps.panel_tag(ax, "AB"[i], dx=-0.075)

    fig.tight_layout(h_pad=1.2)
    ps.save(fig, "figS5_lambda", placed_cm=PLACED_CM)


if __name__ == "__main__":
    main()
