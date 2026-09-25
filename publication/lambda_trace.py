"""Cache an excerpt of the relaxation rate as it evolves during a run.

The solver already writes mu_eff, D_eff and lambda to data/variables.bin at
every step, so nothing needs patching -- the run just has to be kept long
enough to read them back before the workdir is cleaned up.

    python3 publication/lambda_trace.py [tag]
"""

import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import configs                                              # noqa: E402
import pipeline                                             # noqa: E402

TAG = "sup_fc_supra_low"        # recurrent, so lambda actually moves
T_MAX_MS = 3000.0
EXCERPT_MS = 100.0              # window shown in the figure
EXCERPT_START_MS = 2000.0       # well clear of the startup transient
CACHE = os.path.join(pipeline.CACHE, "lambda_trace.npz")


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else TAG
    params = dict(configs.RUNS[tag])
    params["T_MAX"] = T_MAX_MS

    workdir = pipeline.run_solver("lambda_trace", params,
                                  patches=configs.patches_for(tag))
    try:
        act = np.memmap(os.path.join(workdir, "data", "activity.bin"),
                        dtype=np.float32, mode="r").reshape(-1, 2)
        var = np.memmap(os.path.join(workdir, "data", "variables.bin"),
                        dtype=np.float32, mode="r").reshape(-1, 3)
        steps = min(act.shape[0], var.shape[0])
        dt_ms = params["T_MAX"] / act.shape[0]

        # The static value the unconnected runs would have used throughout.
        cv, r0 = pipeline.derived(params)

        i0 = int(EXCERPT_START_MS / dt_ms)
        i1 = min(steps, i0 + int(EXCERPT_MS / dt_ms))
        sl = slice(i0, i1)

        out = {
            "tag": tag,
            "dt_ms": dt_ms,
            "t_ms": np.arange(i1 - i0) * dt_ms,
            "activity_hz": np.asarray(act[sl, 0], dtype=np.float64) * 1000.0,
            "mass": np.asarray(act[sl, 1], dtype=np.float64),
            "mu_eff": np.asarray(var[sl, 0], dtype=np.float64),
            "d_eff": np.asarray(var[sl, 1], dtype=np.float64),
            "lambda": np.asarray(var[sl, 2], dtype=np.float64),
            "lambda_static": float(r0 / cv),
        }
        np.savez_compressed(CACHE, **out)
        lam = out["lambda"]
        print(f"  excerpt {EXCERPT_MS:.0f} ms, dt {dt_ms * 1e6:.0f} ns, "
              f"{i1 - i0:,} samples")
        print(f"  lambda: mean {lam.mean():.4f}  sd {lam.std():.4f} "
              f"({100 * lam.std() / lam.mean():.1f}%)  "
              f"range {lam.min():.4f}..{lam.max():.4f}")
        print(f"  static lambda = r0/CV = {out['lambda_static']:.4f}")
        print(f"  mu_eff: mean {out['mu_eff'].mean():.4f}  "
              f"sd {out['mu_eff'].std():.4f}")
        print(f"  cached -> {CACHE}")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
