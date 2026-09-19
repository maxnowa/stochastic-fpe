"""Numerical experiments on the low-frequency offset in the recurrent spectra.

The mesoscopic spectrum sits systematically below the microscopic one at low
frequency in both recurrent topologies. Four candidate causes were
identified; these three experiments separate them.

  1. bigdt   -- the CFL heuristic assumes a peak rate of 600 (in 1/ms), which
                forces a time step about eight times smaller than the drive
                actually needs. Everything that scales with the step is
                inflated as a result. Relax the constant and see what moves.
  2. filter  -- low-pass the dynamic Lambda lookup, which isolates the Jensen
                bias from evaluating a convex function at a very noisy
                effective drive.
  3. microdt -- halve, then quarter, the microscopic time step and compare
                against exact theory. Tests whether the reference itself is
                biased rather than the solver.

Run:  python publication/diagnostics.py
"""

import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import configs                                              # noqa: E402
import pipeline                                             # noqa: E402

BASE_RUN = "net_fc_supra"
BASE_RUN_UNCONN = "psd_relaxation"
LOW_BAND = (3.0, 60.0)

# Patch pairs applied to a copy of the solver source.
PATCH_BIGDT = [("(w > 0) ? 600.0 : 300.0", "(w > 0) ? 20.0 : 300.0")]
PATCH_FILTER = [("int filter = 0;", "int filter = 1;")]


def band_ratio(f_ref, p_ref, f_test, p_test, band=LOW_BAND):
    """Mean of reference / test over a frequency band."""
    lo, hi = band
    mask = (f_test >= lo) & (f_test < hi)
    return float(np.mean(np.interp(f_test[mask], f_ref, p_ref) / p_test[mask]))


def solver_spectrum(tag, params, patches, verbose=True):
    """Run the solver once and return its activity spectrum."""
    workdir = pipeline.run_solver(tag, params, verbose=verbose, patches=patches)
    try:
        act = np.memmap(os.path.join(workdir, "data", "activity.bin"),
                        dtype=np.float32, mode="r").reshape(-1, 2)
        n_steps = act.shape[0]
        dt_ms = params["T_MAX"] / n_steps
        activity = np.asarray(act[:, 0], dtype=np.float64) * 1000.0
        freq, psd = pipeline.spectrum(activity, dt_ms / 1000.0)
        stats = {
            "dt_ms": dt_ms,
            "n_steps": n_steps,
            "rate_mean_hz": float(activity[int(0.3 * n_steps):].mean()),
            "mass_mean": float(np.asarray(
                act[int(0.3 * n_steps)::max(1, n_steps // 500_000), 1],
                dtype=np.float64).mean()),
        }
        return freq, psd, stats
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def experiment_solver_side():
    """Experiments 1 and 2: both change the solver, not the reference."""
    base = pipeline.load(BASE_RUN)
    f_mic = np.asarray(base["freq_micro"])
    p_mic = np.asarray(base["psd_micro"])
    params = configs.RUNS[BASE_RUN]

    variants = [
        ("baseline", None),
        ("bigdt", PATCH_BIGDT),
        ("filter", PATCH_FILTER),
        ("bigdt+filter", PATCH_BIGDT + PATCH_FILTER),
    ]

    rows = []
    for name, patches in variants:
        if name == "baseline":
            freq = np.asarray(base["freq_meso"])
            psd = np.asarray(base["psd_meso"])
            stats = {"dt_ms": pipeline.val(base, "dt_ms"),
                     "n_steps": pipeline.val(base, "n_steps"),
                     "rate_mean_hz": pipeline.val(base, "rate_mean_hz"),
                     "mass_mean": pipeline.val(base, "mass_mean")}
            print(f"\n=== {name} (from cache) ===")
        else:
            print(f"\n=== {name} ===")
            freq, psd, stats = solver_spectrum(
                f"diag_{name.replace('+', '_')}", params, patches)

        lsd, _ = pipeline.deviation_metrics(f_mic, p_mic, freq, psd)
        rows.append({
            "name": name,
            "dt_ns": stats["dt_ms"] * 1e6,
            "steps": stats["n_steps"],
            "rate": stats["rate_mean_hz"],
            "mass": stats["mass_mean"],
            "ratio": band_ratio(f_mic, p_mic, freq, psd),
            "lsd": lsd,
        })

    print("\n" + "=" * 78)
    print("EXPERIMENTS 1 and 2 -- solver variants, all against the SAME "
          "microscopic run")
    print("=" * 78)
    print(f"{'variant':14s} {'dt (ns)':>9s} {'steps':>12s} {'rate':>8s} "
          f"{'mass':>8s} {'micro/meso':>11s} {'LSD':>6s}")
    print("-" * 78)
    for r in rows:
        print(f"{r['name']:14s} {r['dt_ns']:9.0f} {r['steps']:12,d} "
              f"{r['rate']:7.0f}H {r['mass']:8.4f} {r['ratio']:11.3f} "
              f"{r['lsd']:6.3f}")
    print("\n(micro/meso is the mean power ratio over "
          f"{LOW_BAND[0]:g}-{LOW_BAND[1]:g} Hz; 1.00 means they agree)")
    return rows


def experiment_micro_timestep():
    """Experiment 3: is the microscopic reference converged?"""
    ref = pipeline.load("psd_relaxation")          # unconnected: exact theory
    f_ex = np.asarray(ref["freq_exact"])
    p_ex = np.asarray(ref["psd_exact"])
    f_me = np.asarray(ref["freq_meso"])
    p_me = np.asarray(ref["psd_meso"])
    n = pipeline.val(ref, "param_N_NEURONS")

    print("\n" + "=" * 78)
    print("EXPERIMENT 3 -- microscopic time step, against EXACT theory")
    print("=" * 78)

    rows = []
    for dt_net in (0.005, 0.0025, 0.00125):
        params = dict(configs.RUNS["psd_relaxation"])
        params["DT_NET"] = dt_net
        print(f"\n--- microscopic dt = {dt_net} ms ---")
        act = pipeline.run_micro(params, params["T_MAX"], seed=7)
        freq, psd = pipeline.spectrum(act.astype(np.float64), dt_net / 1000.0)
        rate = float(act.mean())
        del act
        plateau = float(np.mean(np.interp(
            f_ex[(f_ex > LOW_BAND[0]) & (f_ex < LOW_BAND[1])], freq, psd)))
        rows.append({
            "dt": dt_net,
            "rate": rate,
            "ratio": band_ratio(freq, psd, f_ex, p_ex) ** -1,
            "cv": np.sqrt(plateau * n / rate),
        })

    siegert = pipeline.val(ref, "rate_siegert_hz")
    cv_theory = pipeline.val(ref, "cv")
    plateau_meso = float(np.mean(np.interp(
        f_ex[(f_ex > LOW_BAND[0]) & (f_ex < LOW_BAND[1])], f_me, p_me)))

    print(f"\n{'micro dt (ms)':>14s} {'rate':>9s} {'rate err':>9s} "
          f"{'S/exact':>9s} {'implied CV':>11s} {'CV err':>8s}")
    print("-" * 68)
    for r in rows:
        print(f"{r['dt']:14g} {r['rate']:8.1f}H "
              f"{(r['rate'] / siegert - 1) * 100:8.2f}% {r['ratio']:9.3f} "
              f"{r['cv']:11.4f} {(r['cv'] / cv_theory - 1) * 100:7.2f}%")
    print("-" * 68)
    meso_cv = np.sqrt(plateau_meso * n / pipeline.val(ref, "rate_mean_hz"))
    print(f"{'mesoscopic':>14s} {pipeline.val(ref, 'rate_mean_hz'):8.1f}H "
          f"{(pipeline.val(ref, 'rate_mean_hz') / siegert - 1) * 100:8.2f}% "
          f"{band_ratio(f_me, p_me, f_ex, p_ex) ** -1:9.3f} "
          f"{meso_cv:11.4f} {(meso_cv / cv_theory - 1) * 100:7.2f}%")
    print(f"{'exact theory':>14s} {siegert:8.1f}H {0.0:8.2f}% {1.0:9.3f} "
          f"{cv_theory:11.4f} {0.0:7.2f}%")
    return rows


# --------------------------------------------------------------------------
# Experiment 4: does the LEVEL of Lambda matter, and is dynamic Lambda needed?
# --------------------------------------------------------------------------
# Experiment 2 low-passed Lambda itself, i.e. it averaged the OUTPUT of a
# nonlinear map. That converges to <Lambda(mu_eff)>, which is the inflated
# value, so it removed the fluctuation of Lambda but not its bias. These
# variants change the level instead, all at the corrected time step so the
# step artefact cannot mask the effect.

# Freeze Lambda at r0/CV for the base parameters: never consult the table.
PATCH_STATIC = [("if (recurrence_mode == 1 && t > (int)(burn_in*steps)) {",
                 "if (0) {")]

# Look the table up at a low-passed effective drive rather than the
# instantaneous one. This is what actually removes the Jensen bias.
PATCH_MUFILT = [
    ("double lambda_filtered = lambda;",
     "double lambda_filtered = lambda;\n    double mu_filt = PARAM_MU;"),
    ("double lambda_new = get_lambda(&lut, mu_eff, D_eff);",
     "mu_filt = (1.0 - alpha) * mu_filt + alpha * mu_eff;\n"
     "            double lambda_new = get_lambda(&lut, mu_filt, D_eff);"),
]


def experiment_lambda_level():
    base = pipeline.load(BASE_RUN)
    f_mic = np.asarray(base["freq_micro"])
    p_mic = np.asarray(base["psd_micro"])
    params = configs.RUNS[BASE_RUN]

    variants = [
        ("dynamic (bigdt)", PATCH_BIGDT),
        ("static Lambda", PATCH_BIGDT + PATCH_STATIC),
        ("Lambda at filtered mu", PATCH_BIGDT + PATCH_MUFILT),
    ]

    rows = []
    for name, patches in variants:
        print(f"\n=== {name} ===")
        freq, psd, stats = solver_spectrum(
            "diag_" + name.split()[0].lower(), params, patches)
        lsd, _ = pipeline.deviation_metrics(f_mic, p_mic, freq, psd)
        rows.append({"name": name, "rate": stats["rate_mean_hz"],
                     "mass": stats["mass_mean"],
                     "ratio": band_ratio(f_mic, p_mic, freq, psd),
                     "lsd": lsd})

    print("\n" + "=" * 78)
    print("EXPERIMENT 4 -- level of Lambda, all at the corrected time step")
    print("=" * 78)
    print(f"{'variant':24s} {'rate':>8s} {'mass':>9s} {'micro/meso':>11s} {'LSD':>7s}")
    print("-" * 78)
    for r in rows:
        print(f"{r['name']:24s} {r['rate']:7.0f}H {r['mass']:9.4f} "
              f"{r['ratio']:11.3f} {r['lsd']:7.3f}")
    return rows


# --------------------------------------------------------------------------
# Experiment 5: error bars on the low-frequency plateau
# --------------------------------------------------------------------------
# The solver never calls srand(), so rand() restarts from the default seed on
# every run and every solver run replays the same noise. That makes two runs
# at the same time step exactly paired, which is why the filter comparison in
# experiment 2 was so tight. It also means independent realisations need a
# seed injected, which is what PATCH_SEED does.
#
# The question this answers: is the gap between the microscopic and the
# mesoscopic low-frequency plateau larger than the run-to-run scatter?

N_SEEDS = 5


def patch_seed(seed):
    return [("// seed_rng((uint64_t)time(NULL));", f"srand({seed});")]


def experiment_seeds():
    ref = pipeline.load(BASE_RUN_UNCONN)
    f_ex = np.asarray(ref["freq_exact"])
    p_ex = np.asarray(ref["psd_exact"])
    lo, hi = LOW_BAND
    ex_mask = (f_ex >= lo) & (f_ex < hi)
    exact_plateau = float(np.mean(p_ex[ex_mask]))
    params = configs.RUNS[BASE_RUN_UNCONN]

    def plateau(freq, psd):
        return float(np.mean(np.interp(f_ex[ex_mask], freq, psd)))

    meso, micro = [], []
    for i in range(N_SEEDS):
        seed = 1000 + 17 * i
        print(f"\n=== mesoscopic, seed {seed} ===")
        freq, psd, _ = solver_spectrum(f"diag_seed{seed}", params,
                                       patch_seed(seed))
        meso.append(plateau(freq, psd))

        print(f"=== microscopic, seed {seed} ===")
        act = pipeline.run_micro(params, params["T_MAX"], seed=seed)
        f_u, p_u = pipeline.spectrum(act.astype(np.float64),
                                     params["DT_NET"] / 1000.0)
        del act
        micro.append(plateau(f_u, p_u))

    meso = np.array(meso)
    micro = np.array(micro)

    print("\n" + "=" * 78)
    print(f"EXPERIMENT 5 -- low-frequency plateau over {N_SEEDS} independent "
          "seeds")
    print("=" * 78)
    print(f"exact theory                {exact_plateau:.5f} Hz")
    for name, arr in (("mesoscopic solver", meso),
                      ("microscopic network", micro)):
        sem = arr.std(ddof=1) / np.sqrt(len(arr))
        print(f"{name:27s} {arr.mean():.5f} +- {sem:.5f} Hz (sem)   "
              f"scatter {arr.std(ddof=1) / arr.mean() * 100:.1f}%   "
              f"vs exact {arr.mean() / exact_plateau:.3f}")

    diff = micro.mean() - meso.mean()
    sed = np.sqrt(micro.var(ddof=1) / len(micro) + meso.var(ddof=1) / len(meso))
    print(f"\nmicro minus meso: {diff:+.5f} +- {sed:.5f} Hz  "
          f"-> {abs(diff) / sed:.1f} sigma")
    print("(a gap under about 2 sigma is not resolved by this many seeds)")
    return meso, micro

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+",
                    choices=["solver", "microdt", "lambda", "seeds"])
    a = ap.parse_args()
    which = a.only or ["solver", "microdt", "lambda", "seeds"]
    if "solver" in which:
        experiment_solver_side()
    if "microdt" in which:
        experiment_micro_timestep()
    if "lambda" in which:
        experiment_lambda_level()
    if "seeds" in which:
        experiment_seeds()
