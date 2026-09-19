"""Sweep the mass-relaxation rate and watch the low-frequency power move.

The mass deficit m = ||p|| - 1 obeys an Ornstein-Uhlenbeck equation

    dm/dt = -k m + sqrt(r/N) xi(t)

where k is whatever rate multiplies the deficit in the rate closure. The
three methods in the solver are three points on that one axis: k = 0 has no
stationary state, k -> infinity annihilates the deficit every step and is
hard normalisation, and k = Lambda/tau with Lambda = r0/CV is the interior
point that is supposed to put the low-frequency plateau in the right place.

This sweeps k over that axis in the unconnected case, where the analytical
spectrum is available as ground truth.

Run:  python publication/lambda_sweep.py
"""

import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import configs                                              # noqa: E402
import pipeline                                             # noqa: E402

BASE_RUN = "psd_relaxation"          # unconnected, exact theory available
LOW_BAND = (3.0, 60.0)
FACTORS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]

CACHE_FILE = os.path.join(pipeline.CACHE, "lambda_sweep.npz")

# lambda is only overwritten from the lookup table when recurrence is on, so
# for an unconnected run scaling it at initialisation is enough.
LAMBDA_LINE = "double lambda = PARAM_R0 / PARAM_CV;"


def patch_for(factor):
    return [(LAMBDA_LINE,
             f"double lambda = (PARAM_R0 / PARAM_CV) * {factor:.6f};")]


def band_mean(freq, psd, band=LOW_BAND):
    m = (freq >= band[0]) & (freq < band[1])
    return float(np.mean(psd[m]))


def run_one(factor, params):
    tag = f"sweep_lambda_{factor:g}".replace(".", "p")
    workdir = pipeline.run_solver(tag, params, patches=patch_for(factor))
    try:
        act = np.memmap(os.path.join(workdir, "data", "activity.bin"),
                        dtype=np.float32, mode="r").reshape(-1, 2)
        n = act.shape[0]
        dt_ms = params["T_MAX"] / n
        activity = np.asarray(act[:, 0], dtype=np.float64) * 1000.0
        freq, psd = pipeline.spectrum(activity, dt_ms / 1000.0)
        mass = np.asarray(act[n // 3::max(1, n // 500_000), 1],
                          dtype=np.float64)
        return {
            "factor": factor,
            "freq": freq,
            "psd": psd,
            "plateau": band_mean(freq, psd),
            "rate": float(activity[n // 3:].mean()),
            "mass_mean": float(mass.mean()),
            "mass_std": float(mass.std()),
        }
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def main():
    params = configs.RUNS[BASE_RUN]
    ref = pipeline.load(BASE_RUN)

    f_ex = np.asarray(ref["freq_exact"])
    p_ex = np.asarray(ref["psd_exact"])
    exact_plateau = band_mean(f_ex, p_ex)

    r0 = pipeline.val(ref, "rate_siegert_hz") / 1000.0
    cv = pipeline.val(ref, "cv")
    n_neurons = pipeline.val(ref, "param_N_NEURONS")
    tau = pipeline.val(ref, "param_TAU")
    lam = r0 / cv

    rows = []
    for factor in FACTORS:
        print(f"\n=== Lambda x {factor:g}  (k = {factor * lam / tau:.3f} /ms) ===")
        rows.append(run_one(factor, params))

    print("\n" + "=" * 86)
    print("MASS RELAXATION RATE SWEEP -- unconnected, against exact theory")
    print("=" * 86)
    print(f"exact low-frequency plateau ({LOW_BAND[0]:g}-{LOW_BAND[1]:g} Hz): "
          f"{exact_plateau:.4f} Hz   [r0 CV^2/N = {r0 * 1000 * cv**2 / n_neurons:.4f}]")
    print(f"correct setting: Lambda = r0/CV = {lam:.3f} /ms  (factor 1.0)\n")
    print(f"{'factor':>8s} {'k (1/ms)':>10s} {'rate':>8s} {'mass std':>10s} "
          f"{'predicted':>10s} {'plateau':>9s} {'/exact':>8s}")
    print("-" * 74)
    for r in rows:
        k = r["factor"] * lam / tau
        pred = np.sqrt((r["rate"] / 1000.0 / n_neurons) / (2.0 * k))
        print(f"{r['factor']:8g} {k:10.3f} {r['rate']:7.0f}H "
              f"{r['mass_std']:10.5f} {pred:10.5f} {r['plateau']:9.4f} "
              f"{r['plateau'] / exact_plateau:8.3f}")

    np.savez_compressed(
        CACHE_FILE,
        factors=np.array([r["factor"] for r in rows]),
        plateau=np.array([r["plateau"] for r in rows]),
        mass_std=np.array([r["mass_std"] for r in rows]),
        rate=np.array([r["rate"] for r in rows]),
        exact_plateau=exact_plateau,
        lam=lam, tau=tau, cv=cv, r0=r0, n_neurons=n_neurons,
        **{f"psd_{i}": r["psd"] for i, r in enumerate(rows)},
        **{f"freq_{i}": r["freq"] for i, r in enumerate(rows)},
    )
    print(f"\ncached -> {CACHE_FILE}")


if __name__ == "__main__":
    main()
