"""Driver for the poster figures.

  python publication/make.py compute              # run every missing simulation
  python publication/make.py compute --only mass_flux
  python publication/make.py compute --force      # recompute even if cached
  python publication/make.py plot                 # redraw every figure from cache
  python publication/make.py plot --only fig4_mass_conservation
  python publication/make.py list                 # show runs and cache state

`compute` is the expensive half (minutes to hours per run); `plot` is
instant and is what you re-run while tuning the look of a figure.
"""

import argparse
import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import configs                                              # noqa: E402
import pipeline                                             # noqa: E402

FIGURE_MODULES = list(configs.FIGURES.keys())


def cmd_list(_args):
    print(f"{'run':22s} {'cached':8s} description")
    print("-" * 78)
    for tag in configs.RUNS:
        path = pipeline.cache_path(tag)
        if os.path.exists(path):
            state = f"{os.path.getsize(path) / 1e6:.1f} MB"
        else:
            state = "-"
        print(f"{tag:22s} {state:8s} {configs.describe(tag)}")
    print()
    print(f"{'figure':28s} runs")
    print("-" * 78)
    for name, spec in configs.FIGURES.items():
        print(f"{name:28s} {', '.join(spec['runs'])}")


def cmd_compute(args):
    tags = args.only or list(configs.RUNS)
    unknown = [t for t in tags if t not in configs.RUNS]
    if unknown:
        sys.exit(f"unknown run(s): {', '.join(unknown)}")
    for tag in tags:
        pipeline.ensure(tag, force=args.force, keep_raw=args.keep_raw)


def cmd_ensemble(args):
    tags = args.only or [t for t in configs.RUNS if configs.needs_micro(t)]
    unknown = [t for t in tags if t not in configs.RUNS]
    if unknown:
        sys.exit(f"unknown run(s): {', '.join(unknown)}")
    for tag in tags:
        pipeline.ensemble(tag, force=args.force)


def cmd_plot(args):
    names = args.only or FIGURE_MODULES
    unknown = [n for n in names if n not in FIGURE_MODULES]
    if unknown:
        sys.exit(f"unknown figure(s): {', '.join(unknown)}")
    for name in names:
        print(f"[plot] {name}")
        module = importlib.import_module(name)
        module.main()


def main():
    parser = argparse.ArgumentParser(
        description="Build the Bernstein poster figures.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="show runs and cache state")
    p_list.set_defaults(func=cmd_list)

    p_comp = sub.add_parser("compute", help="run simulations into the cache")
    p_comp.add_argument("--only", nargs="+", metavar="RUN")
    p_comp.add_argument("--force", action="store_true",
                        help="recompute even if a cache entry exists")
    p_comp.add_argument("--keep-raw", action="store_true",
                        help="keep the multi-GB solver output for debugging")
    p_comp.set_defaults(func=cmd_compute)

    p_ens = sub.add_parser(
        "ensemble", help="run several seeds per condition for error bands")
    p_ens.add_argument("--only", nargs="+", metavar="RUN")
    p_ens.add_argument("--force", action="store_true")
    p_ens.set_defaults(func=cmd_ensemble)

    p_plot = sub.add_parser("plot", help="draw figures from the cache")
    p_plot.add_argument("--only", nargs="+", metavar="FIGURE")
    p_plot.set_defaults(func=cmd_plot)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
