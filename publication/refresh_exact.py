"""Recompute the analytical spectrum in existing cache entries, in place.

The analytical curve used to be truncated at 2500 Hz, which left it dangling
part-way across the panel. Evaluating it is cheap -- a third of a second for
a whole grid -- so the cap is gone and this rewrites the affected caches
without touching the expensive simulation output.

    python3 publication/refresh_exact.py [tag ...]
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import configs                                               # noqa: E402
import pipeline                                              # noqa: E402

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


def params_from(data):
    p = {}
    for key in data.files:
        if key.startswith("param_"):
            p[key[len("param_"):]] = pipeline.val(data, key)
    return p


def refresh(path):
    data = np.load(path, allow_pickle=True)
    if "freq_exact" not in data.files:
        return None
    out = {k: data[k] for k in data.files}

    freq = np.asarray(data["freq_meso"])
    params = params_from(data)
    rate = float(pipeline.val(data, "rate_siegert_hz"))

    f_th = freq[freq > 0.5]
    out["freq_exact"] = f_th
    out["psd_exact"] = pipeline.exact_spectrum(f_th, params, rate)

    np.savez_compressed(path, **out)
    return float(np.asarray(data["freq_exact"]).max()), float(f_th.max())


def main():
    tags = sys.argv[1:]
    if not tags:
        names = sorted(f for f in os.listdir(CACHE) if f.endswith(".npz"))
    else:
        names = [f"{t}.npz" for t in tags] + [f"{t}_ens.npz" for t in tags]

    for name in names:
        path = os.path.join(CACHE, name)
        if not os.path.exists(path):
            continue
        result = refresh(path)
        if result is None:
            continue
        old, new = result
        print(f"  {name:32s} analytic {old:8.0f} Hz -> {new:9.0f} Hz")


if __name__ == "__main__":
    main()
