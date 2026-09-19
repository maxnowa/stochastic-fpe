"""Shared plotting style for the Bernstein poster figures.

Journal conventions, sized for a poster. Figures are authored at roughly
half their printed width, so a 15 pt label here becomes about 23 pt when the
PDF is placed at 38 cm on an A0 sheet. That is the readable range at a
metre. Keep the ratio of font size to figure width fixed and the figures
stay consistent with each other however they are scaled.

Everything presentational lives in this module; the fig_*.py scripts hold
only the content of a figure.
"""

import os

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

# --------------------------------------------------------------------------
# Colour roles
# --------------------------------------------------------------------------
# Categorical hues are the first three slots of the Okabe-Ito palette, the
# usual choice for print figures that must survive colour-vision deficiency.
# Validated all-pairs for CVD separation and for contrast against white.
#
# References stay achromatic on purpose: colour means "this is a variant of
# our method", never "this is ground truth".
C = {
    "meso":      "#0072B2",   # blue      -- mesoscopic SFPE solver
    "meso_alt":  "#D55E00",   # vermilion -- second solver variant
    "meso_alt2": "#009E73",   # green     -- third variant, if ever needed

    "micro":     "#9A9A94",   # microscopic Monte-Carlo network
    "exact":     "#000000",   # analytical result
    "level":     "#767670",   # asymptotic levels, target lines

    "ink":       "#000000",
    "ink_soft":  "#4A4A45",
    "grid":      "#E2E1DC",
    "surface":   "#FFFFFF",
}

LABEL = {
    "meso":  "Mesoscopic",
    "micro": "Microscopic",
    "exact": "Analytic",
}

# Background the figures are composited onto. Semi-transparent fills are
# pre-blended against this instead of being written as real transparency,
# because some print RIPs flatten transparency into grey boxes or banding.
# Change this if the poster background stops being white.
BACKGROUND = "#FFFFFF"


def blend(color, alpha, background=None):
    """Composite `color` at `alpha` over the poster background.

    Returns an opaque colour that looks identical to the transparent version
    on that background, so nothing in the exported file relies on the
    renderer handling alpha correctly.
    """
    c = np.array(mcolors.to_rgb(color))
    b = np.array(mcolors.to_rgb(background or BACKGROUND))
    return mcolors.to_hex(alpha * c + (1.0 - alpha) * b)


# Base line weights, at scale 1. set_style rescales them by the same factor
# it applies to the fonts, so a figure authored wide and printed at the same
# size as a narrow one does not end up with thinner lines. Reference them as
# ps.LW_DATA from inside a figure's main(), after set_style_for has run.
_LW_BASE = {
    "LW_DATA": 2.7,    # a model curve
    "LW_MICRO": 3.6,   # the microscopic reference, drawn under the model
    "LW_REF": 2.2,     # analytical reference
    "LW_THIN": 1.3,    # target lines, asymptotic levels
    "LW_RAW": 0.8,     # unsmoothed trace beneath a smoothed one
}

LW_DATA = _LW_BASE["LW_DATA"]
LW_MICRO = _LW_BASE["LW_MICRO"]
LW_REF = _LW_BASE["LW_REF"]
LW_THIN = _LW_BASE["LW_THIN"]
LW_RAW = _LW_BASE["LW_RAW"]
ALPHA_RAW = 0.30


# One column of a two-column A0 portrait sheet, after margins and gutter.
POSTER_COLUMN_CM = 38.0
POSTER_FULL_CM = 78.0

# Smallest text on the figure, as it should appear on the printed poster.
# Tick labels are the binding constraint; everything else is larger.
TARGET_TICK_PT = 28.0


def set_style_for(fig_width_in, placed_cm=POSTER_COLUMN_CM,
                  target_tick_pt=TARGET_TICK_PT):
    """Set the font scale from how wide the figure will be printed.

    A figure authored 11 inches wide and placed at the same width on the
    poster as one authored 6 inches wide ends up with text almost half the
    size. Deriving the scale from the intended placement keeps every figure
    at the same apparent text size, so the set reads as one thing.
    """
    placed_in = placed_cm / 2.54
    base = (target_tick_pt / 0.88) * fig_width_in / placed_in
    set_style(scale=base / 15.0)
    return base


def set_style(scale=1.0):
    """Install the figure rcParams.

    `scale` multiplies font sizes and line weights together, so a figure
    keeps the same apparent weight whatever width it was authored at.
    """
    global LW_DATA, LW_MICRO, LW_REF, LW_THIN, LW_RAW
    LW_DATA = _LW_BASE["LW_DATA"] * scale
    LW_MICRO = _LW_BASE["LW_MICRO"] * scale
    LW_REF = _LW_BASE["LW_REF"] * scale
    LW_THIN = _LW_BASE["LW_THIN"] * scale
    LW_RAW = _LW_BASE["LW_RAW"] * scale

    base = 15.0 * scale

    mpl.rcParams.update({
        # --- text -------------------------------------------------------
        # Times, with STIX for the maths. STIX was designed as a
        # Times-compatible maths face, so symbols and text share a colour and
        # weight rather than clashing. This is the usual pairing for APS-style
        # journals, which is what the report is typeset in.
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "Nimbus Roman",
                       "DejaVu Serif"],
        "font.size": base,
        "axes.titlesize": base,
        "axes.labelsize": base,
        "xtick.labelsize": base * 0.88,
        "ytick.labelsize": base * 0.88,
        "legend.fontsize": base * 0.88,
        "figure.titlesize": base * 1.10,
        "mathtext.fontset": "stix",

        # --- colour -----------------------------------------------------
        "text.color": C["ink"],
        "axes.labelcolor": C["ink"],
        "axes.edgecolor": C["ink"],
        "xtick.color": C["ink"],
        "ytick.color": C["ink"],
        "figure.facecolor": C["surface"],
        "axes.facecolor": C["surface"],
        "savefig.facecolor": C["surface"],

        # --- axes -------------------------------------------------------
        # L-shaped frame, ticks pointing in: the common convention in
        # computational-neuroscience journals.
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.9 * scale,
        "axes.titlepad": 9.0,
        "axes.titleweight": "normal",
        "axes.labelpad": 4.0,
        "axes.axisbelow": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4.5,
        "ytick.major.size": 4.5,
        "xtick.major.width": 0.9 * scale,
        "ytick.major.width": 0.9 * scale,
        "xtick.minor.size": 2.5,
        "ytick.minor.size": 2.5,
        "xtick.minor.width": 0.7,
        "ytick.minor.width": 0.7,
        "xtick.major.pad": 4.0,
        "ytick.major.pad": 4.0,

        # --- grid: off by default, enabled only where it earns its place -
        "axes.grid": False,
        "grid.color": C["grid"],
        "grid.linewidth": 0.6,
        "grid.alpha": 1.0,

        # --- lines ------------------------------------------------------
        "lines.linewidth": LW_DATA,  # already scaled above
        "lines.solid_capstyle": "round",
        "lines.dash_capstyle": "round",
        "lines.markersize": 4.5,

        # --- legend -----------------------------------------------------
        "legend.frameon": False,
        "legend.handlelength": 1.9,
        "legend.handletextpad": 0.6,
        "legend.labelspacing": 0.35,
        "legend.borderaxespad": 0.0,
        "legend.borderpad": 0.0,

        # --- output -----------------------------------------------------
        "figure.dpi": 110,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,        # embed glyphs, keep text selectable
        "ps.fonttype": 42,
        # LibreOffice substitutes fonts it does not have, which shifts text
        # metrics and breaks layout. Writing SVG text as outlines removes the
        # dependency entirely.
        "svg.fonttype": "path",
        "pdf.compression": 6,
        "path.simplify": True,
        "path.simplify_threshold": 1.0,
        "agg.path.chunksize": 20000,
    })


def log_grid(ax):
    """A faint grid, for log-log panels only.

    On a spectrum the reader has to compare values across three decades, so
    a grid earns its place. Linear panels get none.
    """
    ax.set_axisbelow(True)
    ax.grid(True, which="major", color=C["grid"], lw=0.6, zorder=0)


def loglog_ticks(ax, label_minor_x=()):
    """Decade labels with unlabelled minor ticks.

    `label_minor_x` additionally labels those minor ticks on the x axis. Use
    it when an axis limit falls mid-decade, so the reader can see where the
    axis actually starts.
    """
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(LogLocator(base=10.0))
        axis.set_minor_locator(
            LogLocator(base=10.0, subs=tuple(range(2, 10)), numticks=100))
        axis.set_minor_formatter(NullFormatter())

    if label_minor_x:
        wanted = [float(v) for v in label_minor_x]

        def fmt(x, _pos):
            for v in wanted:
                if abs(x - v) < 1e-9 * max(1.0, v):
                    return f"{v:g}"
            return ""

        ax.xaxis.set_minor_formatter(FuncFormatter(fmt))


def panel_tag(ax, letter, dx=-0.16, dy=1.0):
    """Bold panel letter, set outside the axes at the top left."""
    ax.text(dx, dy, letter, transform=ax.transAxes,
            fontsize=mpl.rcParams["axes.labelsize"], fontweight="bold",
            va="baseline", ha="left", color=C["ink"])


def annotate(ax, x, y, text, color, **kwargs):
    """Inline curve label: same colour as the curve, normal weight."""
    kwargs.setdefault("fontsize", mpl.rcParams["legend.fontsize"])
    return ax.text(x, y, text, color=color, **kwargs)


OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def save(fig, name, formats=("pdf", "svg", "png"), placed_cm=None):
    """Write to publication/output/ in every requested format.

    With `placed_cm`, also report the size the tick labels will actually have
    once the figure is printed at that width, measured from the saved file
    rather than from the requested figure size, which tight bounding boxes
    change.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    paths = []
    for ext in formats:
        path = os.path.join(OUTPUT_DIR, f"{name}.{ext}")
        fig.savefig(path)
        paths.append(path)
        print(f"  wrote {path}")

    if placed_cm:
        saved_in = fig.get_tightbbox(
            fig.canvas.get_renderer()).width if fig.canvas else None
        if not saved_in:
            saved_in = fig.get_size_inches()[0]
        tick_pt = mpl.rcParams["xtick.labelsize"] * \
            (placed_cm / 2.54) / saved_in
        print(f"  at {placed_cm:.0f} cm wide: tick labels print at "
              f"{tick_pt:.0f} pt")
    plt.close(fig)
    return paths
