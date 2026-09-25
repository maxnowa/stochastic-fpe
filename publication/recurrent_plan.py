"""Drive bound and time step for a recurrent run, in any regime, either sign
of weight.

The solver picks its step from a hard-coded "anticipated rate" that is only
sensible at one operating point, and for w < 0 it is not sensible at all: the
expression (w > 0) ? 600 : 300 hands a negative weight the 300 branch, which
drives the assumed extreme drive far negative and shrinks the step by an order
of magnitude for nothing.

This computes the bound a run actually needs -- its self-consistent rate plus a
margin for feedback noise -- and emits the patch. Same fixed-point iteration as
scout_sparse.plan, but parameterised by the regime instead of fixed to one.

Kept free of a configs import so that configs can use it.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.lif import rate_whitenoise_benji              # noqa: E402

# Voltage bounds hard-coded in src/stochastic_fpe.c.
V_MIN, V_MAX = -0.8, 1.0

SIGMAS = 8.0        # drive excursions covered; 5 is exceeded ~5x in 1e7 steps
COURANT = 0.5       # the recurrent branch runs at 1.0 unpatched


def self_consistent_rate(mu, d, w, iters=300):
    r = rate_whitenoise_benji(mu, np.sqrt(d))
    for _ in range(iters):
        r = 0.5 * r + 0.5 * rate_whitenoise_benji(mu + w * r, np.sqrt(d))
    return float(r)


def plan(params, inflation=3.0):
    """(bound_rate, dt_ms, sd_drive, rate_per_ms) for one run."""
    mu, d, w = params["MU"], params["D"], params["W"]
    n = params["N_NEURONS"]
    dx = (V_MAX - V_MIN) / params["GRID_N"]

    r = self_consistent_rate(mu, d, w)
    mu_max = mu + w * r
    dt = sd = 0.0
    for _ in range(200):
        v = max(abs(mu_max - V_MIN), abs(mu_max - V_MAX))
        dt = COURANT * dx / v
        sd = abs(w) * np.sqrt(r / (n * dt))
        # Push the bound in the direction the weight acts, so an inhibitory
        # run is bounded below and an excitatory one above.
        mu_max = mu + w * r + np.sign(w) * SIGMAS * inflation * sd
    return (mu_max - mu) / w, dt, float(sd), r


def patches(params, inflation=3.0):
    """Source patches that give this run a step it can actually run at."""
    bound, _, _, _ = plan(params, inflation)
    return [("(w > 0) ? 600.0 : 300.0", f"{bound:.4f}"),
            ("dt = (1.0 * dx) / (v_max_worst);",
             f"dt = ({COURANT:.4f} * dx) / (v_max_worst);")]
