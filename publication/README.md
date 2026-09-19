# Poster figures (Bernstein Conference)

Self-contained pipeline for the figures on the poster. Nothing here writes to
the project's `data/`, `plots/` or `src/params.h` — each run gets its own
throw-away working directory, so you can keep using the `Makefile` workflow
next to it.

## Layout

```
publication/
├── configs.py       parameter set for every simulation run
├── pipeline.py      build + run the C solver, microscopic reference, reduction
├── pubstyle.py      shared poster style (fonts, colours, line weights)
├── make.py          command-line driver
├── fig1_mass_conservation.py
├── fig2_simulation_results.py
├── fig3_psd_normalization.py
├── fig4_psd_networks.py
├── cache/           reduced data, one .npz per run (small, keep these)
└── output/          final figures, PDF + PNG
```

## Usage

```bash
python publication/make.py list                     # runs and cache state
python publication/make.py compute                  # everything not yet cached
python publication/make.py compute --only net_rc_supra --force
python publication/make.py plot                     # redraw all figures
python publication/make.py plot --only fig3_psd_normalization
```

`compute` is the slow half. `plot` reads only the cache and takes a second, so
that is the loop to stay in while adjusting how a figure looks.

## Why the cache exists

One 40 s solver run writes 1–3 GB of `activity.bin`, and the spectra need the
series at full time resolution. `pipeline.reduce_run` collapses that to a few
hundred kilobytes — smoothed traces, a full-resolution excerpt, density
snapshots, the log-binned spectrum, and summary statistics — then deletes the
raw output. Restyling a figure therefore never costs a simulation.

Approximate cost per run on this machine:

| run type                        | solver | microscopic reference |
|---------------------------------|--------|-----------------------|
| unconnected, 10 s               | 25 s   | —                     |
| unconnected, 40 s               | 100 s  | 1 min                 |
| recurrent, 40 s                 | 6 min  | 1–5 min               |

## What each figure shows

Numbering follows the poster, which runs problem first: pose the mass
instability, fix it, show the fix is accurate, show it generalises.

| figure | claim |
|--------|-------|
| `fig1_mass_conservation`  | defining `r(t) = J_out(t)` drains the probability mass within a second; the relaxation term holds it at one |
| `fig2_simulation_results` | the relaxation term is accurate, not just stable: correct stationary density and a mean rate within 0.3% of Siegert, in both regimes |
| `fig3_psd_normalization`  | normalising the density every step inflates low-frequency power (LSD 0.48 against the microscopic network, versus 0.06 for the relaxation term) |
| `fig4_psd_networks`       | the match survives all-to-all and sparse random coupling |

Note on figure 2: the activity panels show the rate in 0.5 ms bins, not per
solver step. The unbinned quantity carries a white-noise term whose variance
diverges as the step shrinks, so a per-step trace is not a rate and plots as a
meaningless noise band.

## Style

`pubstyle.py` holds every presentation choice. Colour roles are fixed:
model variants take categorical hues (blue, then orange), while references
stay achromatic — grey for the microscopic network, black for analytical
results, light grey for asymptotic levels. The categorical hues are checked
for colour-vision-deficiency separation, and line style is used as a second
channel wherever two model curves overlap.

Figure sizes are set in inches at the size the figure is meant to be printed,
so the point sizes in `set_style()` are literal. If a figure gets scaled down
on the poster, pass a larger `scale` to `ps.set_style()` rather than editing
individual font sizes.

## Reproducing on another machine

Requires the same prerequisites as the main project (`gcc`, `numpy`, `scipy`,
`matplotlib`, `mpmath`) plus `src/lambda_table.bin`, which `make lut`
generates. Then `python publication/make.py compute && python
publication/make.py plot`.
