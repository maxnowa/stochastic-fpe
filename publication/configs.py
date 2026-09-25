"""Parameter sets for every poster figure.

Each entry in RUNS is one invocation of the C solver. The pipeline turns a
run into a compact `.npz` in publication/cache/, so the expensive part
happens once and the figure scripts stay instant to re-run.

Keys mirror the PARAM_* names in src/params.h without the prefix.
"""

# Defaults shared by every run. Matches src/params.h except that recurrence
# is off, which is the unconnected reference case.
BASE = {
    "N_NEURONS": 500,
    "T_MAX": 40000.0,
    "TAU": 1,
    "MU": 1.2,
    "D": 0.01,
    "V_TH": 1.0,
    "V_RESET": 0.0,
    "DT_NET": 0.005,
    "GRID_N": 200,
    "METHOD": 1,        # 0 = r(t)=J_out, 1 = relaxation term, 2 = renormalise
    "FLUX_METHOD": 1,
    "DELAY": 0,
    "W": 0.0,
    "RECURRENCE": 0,
    "CONNECTIVITY": 1.0,
}

# Voltage grid bounds hard-coded in src/stochastic_fpe.c. Needed to turn the
# density grid index back into a membrane voltage.
V_MIN = -0.8
V_MAX = 1.0


# --------------------------------------------------------------------------
# Source patches carried by particular runs
# --------------------------------------------------------------------------
# The solver picks its time step from a worst-case drive built on a hard-coded
# "anticipated rate" of 600, in units of 1/ms. For w=0.1 that implies an
# effective drive of 61, where the drive actually seen has mean 1.27 and a
# per-step fluctuation under 0.1. The resulting step is about 15 times smaller
# than needed, and it produces a spurious low-frequency deficit in the
# recurrent spectra: 33% at the original step, 6% once relaxed, with the runs
# 14 times faster. Until the constant is fixed in src/stochastic_fpe.c the
# recurrent poster runs carry this patch, applied to a copy of the source.
CFL_FIX = [("(w > 0) ? 600.0 : 300.0", "(w > 0) ? 20.0 : 300.0")]

RUN_PATCHES = {
    "net_fc_supra": CFL_FIX,
    "net_rc_supra": CFL_FIX,
}

# Seeds used for the ensemble runs. The solver never calls srand(), so a seed
# has to be injected for the realisations to differ at all.
SEEDS = [1000, 1017, 1034, 1051, 1068]

# Runs that use fewer seeds than the default, and runs whose microscopic
# reference skips the Brownian-bridge exit correction. The sparse pair does
# both: its reference is recurrent and sparse, which is the most expensive
# thing in the whole set, and in this suprathreshold regime the uncorrected
# rate deficit is around one percent. Traded deliberately for turnaround.
RUN_SEEDS = {}
RUN_MICRO_BRIDGE = {}

# Discarded from the start of every trace before estimating a spectrum. The
# Bartlett segments are one second long, so a startup transient lands wholly
# inside the first segment and carries far more low-frequency power than the
# finite-size plateau being measured. Without this both simulations sit about
# 9% above exact theory; with it they land on it.
SPECTRUM_SKIP_MS = 2000.0

# --- sparse-connectivity panel -------------------------------------------
# Separating random from full connectivity needs extra diffusion
#     w^2 r / (2 D k),   k = inputs per neuron
# which depends on the weight and the in-degree, not on p and N separately.
# At the poster's w=0.1 and 250 inputs it is 0.07%, so the two topologies are
# the same simulation. Weight 0.3 with 10 inputs per neuron gives 40%.
#
# A large sparse population rather than a small dense one, because the
# feedback noise w sqrt(r / N dt) is what breaks the solver at strong
# coupling: 31% of the mean drive at N=500, 6% at N=5000.
#
# The two patches are required. The drive bound sets the time step (3.58
# gives 2053 ns, a working Courant number of 0.52); the ramp exists because
# the density starts as a delta and full coupling at t=0 sends the whole
# packet across threshold at once, tripping the solver's stability guard.
# N and p chosen from the scout: p must stay biologically defensible, which
# rules out the 0.002 this started at. At N=1000 with ten inputs per neuron
# the two topologies separate by 0.194 in the solver against 0.196 in the
# microscopic network, so the solver reproduces the topology effect to within
# one percent at a realistic connection probability. The feedback noise there
# is 19.4% of the drive, which the scout confirmed the solver handles.
SPARSE_W = 0.3
SPARSE_N = 1000
SPARSE_K = 10

_SPARSE_BOUND_RATE = 23.6043       # -> drive bound 8.28, dt 991 ns
_SPARSE_RAMP_STEPS = 50451         # 50 ms at that step

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

_RECURRENCE_RAMPED = """        // Recurrence ramped in; see the note in configs.py.
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
        }""" % (_SPARSE_RAMP_STEPS, _SPARSE_RAMP_STEPS)

SPARSE_FIX = [
    ("(w > 0) ? 600.0 : 300.0", f"(w > 0) ? {_SPARSE_BOUND_RATE:.4f} : 300.0"),
    (_RECURRENCE_BLOCK, _RECURRENCE_RAMPED),
]


RUN_PATCHES["sparse_fc"] = SPARSE_FIX
RUN_PATCHES["sparse_rc"] = SPARSE_FIX

for _tag in ("sparse_fc", "sparse_rc"):
    RUN_SEEDS[_tag] = SEEDS[:3]
    RUN_MICRO_BRIDGE[_tag] = False


def seeds_for(tag):
    return RUN_SEEDS.get(tag, SEEDS)


def micro_bridge_for(tag):
    return RUN_MICRO_BRIDGE.get(tag, None)


def patches_for(tag):
    return RUN_PATCHES.get(tag, [])


def cfg(**overrides):
    out = dict(BASE)
    out.update(overrides)
    return out


# --------------------------------------------------------------------------
# Runs, grouped by the figure that consumes them
# --------------------------------------------------------------------------
RUNS = {
    # --- Figure 1: solver behaviour in two regimes -----------------------
    # Unconnected population, relaxation method. Low noise / suprathreshold
    # and high noise / subthreshold bracket the operating range.
    "sim_supra_lownoise": cfg(MU=1.2, D=0.01, METHOD=1),
    "sim_sub_highnoise":  cfg(MU=0.8, D=0.10, METHOD=1),

    # --- Figure 2: spectral cost of hard normalisation -------------------
    # Same physics, three ways of closing the mass budget. The renormalised
    # run is the one the poster is about; the relaxation run is the control.
    "psd_renormalise": cfg(MU=1.2, D=0.01, METHOD=2),
    "psd_relaxation":  cfg(MU=1.2, D=0.01, METHOD=1),

    # --- Figure 3: recurrent topologies ----------------------------------
    # Fully connected (p=1) and sparse random (p=0.5) at matched coupling.
    "net_fc_supra": cfg(MU=1.2, D=0.01, METHOD=1, RECURRENCE=1, W=0.1,
                        CONNECTIVITY=1.0),
    "net_rc_supra": cfg(MU=1.2, D=0.01, METHOD=1, RECURRENCE=1, W=0.1,
                        CONNECTIVITY=0.5),

    # --- Figure 4: mass conservation failure ------------------------------
    # Short runs: the naive flux definition collapses within the first
    # second, so 10 s of simulated time is plenty.
    # --- Figure 6: all four operating regimes, unconnected ----------------
    # The solver has to hold up across the sub/suprathreshold and low/high
    # noise corners, not just the one the other figures use. Subthreshold low
    # noise is the hard case: the rate is exponentially sensitive to the
    # drive, so the relaxation rate responds about six times more steeply
    # there than it does above threshold.
    "regime_supra_low":  cfg(MU=1.2, D=0.01, METHOD=1),
    "regime_supra_high": cfg(MU=1.2, D=0.10, METHOD=1),
    "regime_sub_low":    cfg(MU=0.8, D=0.01, METHOD=1),
    "regime_sub_high":   cfg(MU=0.8, D=0.10, METHOD=1),

    # --- Figure 7: a sparse network that actually differs from a dense one
    "sparse_fc": cfg(MU=1.2, D=0.01, METHOD=1, RECURRENCE=1, W=SPARSE_W,
                     CONNECTIVITY=1.0, N_NEURONS=SPARSE_N, T_MAX=20000.0),
    "sparse_rc": cfg(MU=1.2, D=0.01, METHOD=1, RECURRENCE=1, W=SPARSE_W,
                     CONNECTIVITY=SPARSE_K / SPARSE_N, N_NEURONS=SPARSE_N,
                     T_MAX=20000.0),

    "mass_flux":       cfg(MU=1.2, D=0.01, METHOD=0, T_MAX=10000.0),
    "mass_relaxation": cfg(MU=1.2, D=0.01, METHOD=1, T_MAX=10000.0),
}


# Which runs each figure needs, and whether a microscopic Monte-Carlo
# reference and an analytical curve are part of that figure.
#   micro  -- run the N-neuron Langevin network alongside the solver
#   exact  -- evaluate the analytical finite-size PSD (unconnected only)
FIGURES = {
    "fig1_mass_conservation": {
        "runs": ["mass_flux", "mass_relaxation"],
        "micro": False,
        "exact": False,
    },
    "fig2_simulation_results": {
        "runs": ["sim_supra_lownoise", "sim_sub_highnoise"],
        "micro": False,
        "exact": False,
    },
    "fig3_psd_normalization": {
        "runs": ["psd_renormalise", "psd_relaxation"],
        "micro": True,
        "exact": True,
    },
    "fig4_psd_networks": {
        "runs": ["net_fc_supra", "net_rc_supra"],
        "micro": True,
        "exact": False,
    },
    # Built by publication/benchmark.py rather than from a named run, since
    # it sweeps the population size rather than fixing it.
    "fig5_cost_scaling": {
        "runs": [],
        "micro": False,
        "exact": False,
    },
    "fig6_regimes": {
        "runs": ["regime_sub_low", "regime_supra_low",
                 "regime_sub_high", "regime_supra_high"],
        "micro": True,
        "exact": True,
    },
    "fig7_sparse_connectivity": {
        "runs": ["sparse_fc", "sparse_rc"],
        "micro": True,
        "exact": False,
    },
    # Pure arithmetic from the grid size and the two time steps -- no
    # simulation, no machine. Counts work rather than elapsed time, so the
    # cost claim survives the objection that a parallel network implementation
    # would change the wall-clock picture.
    "fig8_work_scaling": {
        "runs": [],
        "micro": False,
        "exact": False,
    },
    # Supplementary. No analytical spectrum: it exists only for the
    # unconnected case, so these are checked against the network.
    "figS1_fc_regimes": {
        "runs": ["sup_fc_supra_low", "sup_fc_supra_high",
                 "sup_fc_sub_low", "sup_fc_sub_high"],
        "micro": True, "exact": False,
    },
    "figS2_rc_regimes": {
        "runs": ["sup_rc_supra_low", "sup_rc_supra_high",
                 "sup_rc_sub_low", "sup_rc_sub_high"],
        "micro": True, "exact": False,
    },
    "figS3_inhibitory": {
        "runs": ["sup_inh_supra_low", "sup_inh_supra_high",
                 "sup_inh_sub_low", "sup_inh_sub_high"],
        "micro": True, "exact": False,
    },
    # Built by publication/lambda_trace.py, which keeps the solver's
    # variables.bin rather than reducing it away.
    "figS5_lambda": {
        "runs": [], "micro": False, "exact": False,
    },
    "figS4_delay": {
        "runs": ["sup_delay_1", "sup_delay_2",
                 "sup_delay_3", "sup_delay_5"],
        "micro": True, "exact": False,
    },
}



# --------------------------------------------------------------------------
# Supplementary runs
# --------------------------------------------------------------------------
# Recurrence across every regime, for both topologies, plus an inhibitory
# weight and a transmission delay. Each needs its own step size: the drive
# bound the solver assumes is hard-coded for one operating point and is simply
# wrong for a negative weight, so recurrent_plan computes what each run needs.
#
# Half the simulated time of the main runs and three seeds, since these are
# supplementary; that still leaves 18 one-second Bartlett segments per seed.
# The microscopic reference is the plain scheme, as in figures 4 and 7.
import recurrent_plan as _rp                                # noqa: E402

_SUP_T = 20000.0
_REGIMES = [("supra_low", 1.2, 0.01), ("supra_high", 1.2, 0.10),
            ("sub_low", 0.8, 0.01), ("sub_high", 0.8, 0.10)]

_SUPPLEMENTARY = {}
for _name, _mu, _d in _REGIMES:
    # S1 fully connected, S2 random, S3 inhibitory -- all four regimes each.
    _SUPPLEMENTARY[f"sup_fc_{_name}"] = cfg(
        MU=_mu, D=_d, W=0.1, CONNECTIVITY=1.0, RECURRENCE=1, T_MAX=_SUP_T)
    _SUPPLEMENTARY[f"sup_rc_{_name}"] = cfg(
        MU=_mu, D=_d, W=0.1, CONNECTIVITY=0.5, RECURRENCE=1, T_MAX=_SUP_T)
    _SUPPLEMENTARY[f"sup_inh_{_name}"] = cfg(
        MU=_mu, D=_d, W=-0.05, CONNECTIVITY=1.0, RECURRENCE=1, T_MAX=_SUP_T)

# S4 delay, fully connected in the suprathreshold low-noise regime. The delay
# is an int in the solver, so it is a whole number of milliseconds.
SUP_DELAYS = [1, 2, 3, 5]
for _d_ms in SUP_DELAYS:
    _SUPPLEMENTARY[f"sup_delay_{_d_ms}"] = cfg(
        MU=1.2, D=0.01, W=0.1, CONNECTIVITY=1.0, RECURRENCE=1,
        DELAY=_d_ms, T_MAX=_SUP_T)

RUNS.update(_SUPPLEMENTARY)
for _tag, _p in _SUPPLEMENTARY.items():
    RUN_PATCHES[_tag] = _rp.patches(_p)
    RUN_SEEDS[_tag] = SEEDS[:3]
    RUN_MICRO_BRIDGE[_tag] = False

def needs_micro(tag):
    return any(tag in f["runs"] and f["micro"] for f in FIGURES.values())


def needs_exact(tag):
    return any(tag in f["runs"] and f["exact"] for f in FIGURES.values())


def describe_params(p):
    """Short human-readable label for a parameter set."""
    bits = [f"mu={p['MU']:g}", f"D={p['D']:g}", f"N={p['N_NEURONS']}"]
    if p["RECURRENCE"]:
        bits.append(f"w={p['W']:g}")
        bits.append(f"p={p['CONNECTIVITY']:g}")
    else:
        bits.append("unconnected")
    bits.append(f"method={p['METHOD']}")
    return ", ".join(bits)


def describe(tag):
    """Label for a named run; falls back to the tag for ad-hoc runs."""
    if tag not in RUNS:
        return tag
    return describe_params(RUNS[tag])
