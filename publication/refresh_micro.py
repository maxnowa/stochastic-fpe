"""Recompute the microscopic reference in existing ensemble caches, in place.

The reference network now carries the Brownian-bridge exit correction (see
MICRO_BRIDGE in pipeline). Its spectra are independent of the solver output
already sitting in the cache, so they can be replaced on their own without
re-running a single solver.

    python3 publication/refresh_micro.py <tag> [tag ...]
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import configs                                               # noqa: E402
import pipeline                                              # noqa: E402


def refresh(tag):
    path = pipeline.ensemble_path(tag)
    data = np.load(path, allow_pickle=True)
    if "psd_micro_all" not in data.files:
        print(f"  {tag}: no microscopic reference, skipped")
        return

    params = configs.RUNS[tag]
    out = {k: data[k] for k in data.files}
    old_rate = float(np.atleast_1d(data["rate_micro"]).mean())

    psd, rates, freq = [], [], None
    for seed in configs.seeds_for(tag):
        t0 = time.time()
        micro = pipeline.run_micro(params, params["T_MAX"], seed=seed,
                                   verbose=False,
                                   bridge=configs.micro_bridge_for(tag))
        f, p = pipeline.spectrum(micro.astype(np.float64),
                                 params["DT_NET"] / 1000.0,
                                 skip_ms=configs.SPECTRUM_SKIP_MS)
        freq = f if freq is None else freq
        psd.append(p)
        rates.append(float(micro.mean()))
        print(f"    seed {seed}: {rates[-1]:7.2f} Hz  ({time.time() - t0:.0f} s)")
        del micro

    out["freq_micro"] = freq
    out["psd_micro_all"] = np.array(psd)
    out["rate_micro"] = np.array(rates)
    np.savez_compressed(path, **out)

    new_rate = float(np.mean(rates))
    exact = float(np.atleast_1d(data["rate_siegert_hz"]).mean())
    print(f"  {tag}: {old_rate:.2f} -> {new_rate:.2f} Hz  "
          f"(exact {exact:.2f}: {100 * (old_rate / exact - 1):+.2f}% -> "
          f"{100 * (new_rate / exact - 1):+.2f}%)")


if __name__ == "__main__":
    for tag in sys.argv[1:]:
        print(f"[refresh] {tag}")
        refresh(tag)
