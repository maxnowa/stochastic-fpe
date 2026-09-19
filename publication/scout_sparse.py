"""Find a coupling regime where random connectivity is actually different.

Background. The microscopic weights are renormalised by 1/p, so the mean
recurrent input does not depend on p. Only its variance does, and that
reaches the solver as extra diffusion:

    D_eff = D + (w^2 (1-p)/p / 2N) r     ->     fraction ~ w^2 r / (2 D k)

with k = pN inputs per neuron. The fraction depends on the weight and the
in-degree, not on p and N separately. So separating the two topologies needs
a larger weight or fewer inputs per neuron -- and a larger weight costs
feedback noise, of size w sqrt(r / N dt), which is what produced the
artefact we already removed once. Raising N at fixed in-degree buys low
feedback noise and strong sparsity at the same time, which is why these runs
use a large sparse network rather than the N=500 dense one.

Two numerical hazards, both learned the hard way:

  * The density starts as a delta. Switching full coupling on at t=0 makes
    the whole packet cross threshold together, giving a rate an order of
    magnitude above stationary, which trips the solver's CFL guard and
    aborts the run. The coupling is therefore ramped in.
  * The step size is chosen from a bound on the drive. A bound derived from
    the bare finite-size noise is too tight, because the recurrent loop
    amplifies fluctuations. The bound here is widened until the run actually
    completes, rather than predicted.

A run that aborts early is silently truncated, so every run is checked
against its expected step count and against self-consistent theory.

Run:  python publication/scout_sparse.py
"""

import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import configs                                              # noqa: E402
import pipeline                                             # noqa: E402

sys.path.insert(0, pipeline.PROJECT)
from analysis.lif import rate_whitenoise_benji              # noqa: E402

MU, D, N = 1.2, 0.01, 5000
V_MIN, V_MAX, GRID = -0.8, 1.0, 200
DX = (V_MAX - V_MIN) / GRID
T_MAX_MS = 20000.0
RAMP_MS = 50.0
# Discard before estimating spectra. The Bartlett segments are one second
# long, so the ramp transient lands entirely in the first segment and
# carries orders of magnitude more low-frequency power than the plateau.
SKIP_MS = 2000.0
SIGMAS = 8.0
# The solver sets dt = dx / v_max against the drive bound, with no extra
# safety factor. The bound itself supplies the headroom: with a bound of 2.7
# and a typical drive of 1.5, the working Courant number is about 0.6.
# Applying a further factor here would double-count and mispredict the step.

W = 0.3
# Inputs per neuron for the sparse variants. The fully connected control runs
# at the same weight, so the panels differ only in topology.
K_VALUES = [50, 25, 10]

_RECURRENCE_BLOCK = """        // calculate effective drift
        double mu_eff = mu;
        if (recurrence_mode == 1)
        {
            mu_eff += (w * A_t_delayed);
        }
        // calculate effective diffusion for p<1
        double D_eff = D;
        if (recurrence_mode == 1 && connectivity_p < 1.0)
        {
            // D_w(t) = (sigma_w^2 / 2N) * r(t-d)
            D_eff += (sigma_w_squared / (2.0 * N_neurons)) * r_t_delayed;
        }"""


def ramp_patch(ramp_steps):
    new = ("""        // Recurrence ramped in over the first steps; see scout_sparse.py.
        double ramp = (t < %d) ? ((double)t / (double)%d) : 1.0;
        double mu_eff = mu;
        if (recurrence_mode == 1)
        {
            mu_eff += (ramp * w * A_t_delayed);
        }
        double D_eff = D;
        if (recurrence_mode == 1 && connectivity_p < 1.0)
        {
            D_eff += (ramp * ramp * sigma_w_squared / (2.0 * N_neurons))
                     * r_t_delayed;
        }""" % (ramp_steps, ramp_steps))
    return [(_RECURRENCE_BLOCK, new)]


def self_consistent_rate(w):
    r = rate_whitenoise_benji(MU, np.sqrt(D))
    for _ in range(300):
        r = 0.5 * r + 0.5 * rate_whitenoise_benji(MU + w * r, np.sqrt(D))
    return r


def plan(w, n, inflation):
    """Step size and drive bound, with the noise estimate inflated to cover
    amplification by the recurrent loop."""
    r = self_consistent_rate(w)
    mu_max = MU + w * r
    dt = sd = 0.0
    for _ in range(200):
        v = max(abs(mu_max - V_MIN), abs(mu_max - V_MAX))
        dt = DX / v                      # exactly what the solver will do
        sd = w * np.sqrt(r / (n * dt))
        mu_max = MU + w * r + SIGMAS * inflation * sd
    return (mu_max - MU) / w, dt, sd, r


def run_solver_complete(tag, params, w, n, max_tries=5):
    """Run, widening the drive bound until the run completes."""
    inflation = 3.0        # starts near the Courant number of the validated runs
    for attempt in range(max_tries):
        rate, dt, sd, r = plan(w, n, inflation)
        ramp_steps = max(1, int(RAMP_MS / dt))
        patches = ([("(w > 0) ? 600.0 : 300.0", f"(w > 0) ? {rate:.4f} : 300.0")]
                   + ramp_patch(ramp_steps))
        expected = int(params["T_MAX"] / dt)
        typical_courant = max(abs(MU + w * r - V_MIN),
                              abs(MU + w * r - V_MAX)) * dt / DX
        print(f"    try {attempt + 1}: bound {MU + w * rate:.2f}, "
              f"dt {dt * 1e6:.0f} ns, Courant {typical_courant:.2f}, "
              f"{expected:,} steps expected")
        workdir = pipeline.run_solver(tag, params, verbose=False,
                                      patches=patches)
        try:
            act = np.memmap(os.path.join(workdir, "data", "activity.bin"),
                            dtype=np.float32, mode="r").reshape(-1, 2)
            steps = act.shape[0]
            if steps < 0.99 * expected:
                print(f"      aborted at {steps:,} steps "
                      f"({steps / expected * 100:.0f}%), widening")
                inflation *= 1.7
                continue

            half = steps // 2
            a = np.asarray(act[:, 0], dtype=np.float64) * 1000.0
            rate_hz = float(a[half:].mean())
            mass = float(np.asarray(act[half::max(1, steps // 300_000), 1],
                                    dtype=np.float64).mean())
            freq, psd = pipeline.spectrum(a, dt / 1000.0, skip_ms=SKIP_MS)
            del a

            # What the drive actually did, from the solver's own log.
            var = np.memmap(os.path.join(workdir, "data", "variables.bin"),
                            dtype=np.float32, mode="r").reshape(-1, 3)
            mu_eff = np.asarray(var[half::max(1, steps // 300_000), 0],
                                dtype=np.float64)
            print(f"      complete. rate {rate_hz:.0f} Hz "
                  f"(theory {r * 1000:.0f}), mass {mass:.4f}, "
                  f"drive {mu_eff.mean():.3f} +- {mu_eff.std():.3f}, "
                  f"peak {mu_eff.max():.2f} vs bound {MU + w * rate:.2f}")
            return dict(freq=freq, psd=psd, rate=rate_hz, mass=mass, dt=dt,
                        theory=r * 1000, drive_sd=float(mu_eff.std()))
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
    raise RuntimeError(f"{tag}: could not complete after {max_tries} tries")


def main():
    r = self_consistent_rate(W)
    print(f"weight {W}, N = {N}, self-consistent rate {r * 1000:.0f} Hz\n")

    base = dict(configs.RUNS["net_fc_supra"])
    base.update(T_MAX=T_MAX_MS, W=W, N_NEURONS=N, MU=MU, D=D)

    print("=" * 70 + "\nfully connected control (p = 1)\n" + "=" * 70)
    fc_params = dict(base, CONNECTIVITY=1.0)
    fc = run_solver_complete("scout_fc", fc_params, W, N)
    fc_micro = pipeline.run_micro(fc_params, T_MAX_MS, seed=1, verbose=True)
    fc["f_mi"], fc["p_mi"] = pipeline.spectrum(
        fc_micro.astype(np.float64), fc_params["DT_NET"] / 1000.0,
        skip_ms=SKIP_MS)
    fc["rate_mi"] = float(fc_micro.mean())
    del fc_micro
    fc["lsd"], _ = pipeline.deviation_metrics(fc["f_mi"], fc["p_mi"],
                                              fc["freq"], fc["psd"])
    print(f"  micro {fc['rate_mi']:.0f} Hz, solver tracking LSD {fc['lsd']:.3f}")

    results = {}
    for k in K_VALUES:
        p = k / N
        extra = (W * W * (1 - p) / p / (2 * N)) * r / D
        print("\n" + "=" * 70)
        print(f"random, {k} inputs per neuron (p = {p:g}), "
              f"extra diffusion {extra * 100:.0f}%")
        print("=" * 70)
        params = dict(base, CONNECTIVITY=p)
        rc = run_solver_complete(f"scout_k{k}", params, W, N)
        micro = pipeline.run_micro(params, T_MAX_MS, seed=1, verbose=True)
        rc["f_mi"], rc["p_mi"] = pipeline.spectrum(
            micro.astype(np.float64), params["DT_NET"] / 1000.0,
            skip_ms=SKIP_MS)
        rc["rate_mi"] = float(micro.mean())
        del micro
        rc["lsd"], _ = pipeline.deviation_metrics(rc["f_mi"], rc["p_mi"],
                                                  rc["freq"], rc["psd"])
        rc["sep"], _ = pipeline.deviation_metrics(fc["f_mi"], fc["p_mi"],
                                                  rc["f_mi"], rc["p_mi"])
        rc["extra"] = extra
        rc["k"] = k
        results[k] = rc
        print(f"  micro {rc['rate_mi']:.0f} Hz, solver tracking "
              f"LSD {rc['lsd']:.3f}, topology separation {rc['sep']:.3f}")

    print("\n" + "=" * 74)
    print("SUMMARY -- want separation LARGE and tracking SMALL")
    print("=" * 74)
    print(f"{'inputs':>7s} {'extra D':>8s} {'separation':>11s} "
          f"{'track RC':>9s} {'track FC':>9s}")
    print("-" * 74)
    for k, rc in results.items():
        print(f"{k:7d} {rc['extra'] * 100:7.0f}% {rc['sep']:11.3f} "
              f"{rc['lsd']:9.3f} {fc['lsd']:9.3f}")

    np.savez_compressed(
        os.path.join(pipeline.CACHE, "scout_sparse.npz"),
        w=W, n_neurons=N, t_max=T_MAX_MS,
        fc_freq=fc["freq"], fc_psd=fc["psd"],
        fc_f_mi=fc["f_mi"], fc_p_mi=fc["p_mi"], fc_lsd=fc["lsd"],
        **{f"k{k}_{key}": rc[key] for k, rc in results.items()
           for key in ("freq", "psd", "f_mi", "p_mi", "lsd", "sep", "extra")})
    print(f"\ncached -> {pipeline.CACHE}/scout_sparse.npz")


if __name__ == "__main__":
    main()
