import numpy as np
import os
import sys
import time
import csv
import warnings
import struct
from config import load_params_from_header


# --- Physics Model Constants ---
# MUST match params.h
TAU = 1.0        # ms
V_TH = 1.0       # mV
V_RESET = 0.0    # mV

# --- Grid Configuration ---
# 200x200 is sufficient for interpolation.
MU_MIN, MU_MAX = -2.0, 6.0   
N_MU = 500       

D_MIN, D_MAX = 0.001, 0.2
N_SIGMA = 500    

OUTPUT_PATH = "src/lambda_table.dat"
BINARY_OUTPUT_PATH = "src/lambda_table.bin"
LOG_DIR = "logs"
HEATMAP_PATH = os.path.join(LOG_DIR, "lambda_table_heatmap.png")
ISSUE_LOG_PATH = os.path.join(LOG_DIR, "lut_numerical_issues.csv")

# LUT computation mode toggle:
# - "exact": always compute lambda = rate/CV (no z-threshold shortcut)
# - "approx": use fast-path clipping far from threshold for speed
LUT_MODE = "exact"

# Approximation settings (used only when LUT_MODE == "approx")
APPROX_Z_LIMIT = 16.0
APPROX_SUPRATHRESH_LAMBDA = 500.0
APPROX_SUBTHRESH_LAMBDA = 0.0

# Numerical safety settings
CV_EPS = 1e-4
LAMBDA_MAX = 1000.0

# Exact-mode stabilizer for numerically extreme regions.
# Uses asymptotic behavior for large |(mu - V_TH)/sqrt(D)| to avoid overflow-prone
# integrals while preserving physically expected trends.
ASYMPTOTIC_Z_THRESHOLD = 14.0
ASYMPTOTIC_CV_FLOOR = 0.05


def save_heatmap(table, mu_vals, D_vals, output_path=HEATMAP_PATH):
    """Save a compact heatmap visualization of the LUT to disk."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not installed. Skipping LUT heatmap.")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Log-compress dynamic range so high lambda values do not wash out structure.
    # +1 keeps zeros finite.
    table_display = np.log10(1.0 + table)

    plt.figure(figsize=(5.0, 4.0), dpi=150)
    image = plt.imshow(
        table_display,
        origin="lower",
        aspect="auto",
        extent=[D_vals[0], D_vals[-1], mu_vals[0], mu_vals[-1]],
        cmap="viridis",
    )
    cbar = plt.colorbar(image)
    cbar.set_label("log10(1 + lambda)")
    plt.xlabel("D")
    plt.ylabel("mu")
    plt.title("Lambda LUT (Log-Scaled)")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"✓ Saved LUT heatmap to {output_path}")


def save_binary_lut(table, output_path=BINARY_OUTPUT_PATH):
    """
    Binary format (little-endian):
      - 4s: magic = b"LUT1"
      - i : N_MU
      - i : N_SIGMA
      - d : MU_MIN
      - d : MU_MAX
      - d : D_MIN
      - d : D_MAX
      - N_MU*N_SIGMA doubles in row-major order (table[i, j])
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    header = struct.pack(
        "<4siidddd",
        b"LUT1",
        int(N_MU),
        int(N_SIGMA),
        float(MU_MIN),
        float(MU_MAX),
        float(D_MIN),
        float(D_MAX),
    )
    with open(output_path, "wb") as f:
        f.write(header)
        np.asarray(table, dtype=np.float64, order="C").tofile(f)
    print(f"✓ Saved binary LUT to {output_path}")

def generate_lut():
    try:
        from analysis.lif import cv_benji, rate_whitenoise_benji
    except ImportError:
        print("Error: Could not import analysis.lif.")
        sys.exit(1)

    if LUT_MODE not in {"approx", "exact"}:
        raise ValueError("LUT_MODE must be 'approx' or 'exact'")

    print(f"Generating Lambda LUT ({LUT_MODE} mode)...")
    print(f"  Grid: {N_MU}x{N_SIGMA} ({N_MU*N_SIGMA} points)")
    
    mu_vals = np.linspace(MU_MIN, MU_MAX, N_MU)
    D_vals = np.linspace(D_MIN, D_MAX, N_SIGMA)
    table = np.zeros((N_MU, N_SIGMA))
    issue_count = 0
    issue_mu_min = None
    issue_mu_max = None
    issue_D_min = None
    issue_D_max = None

    os.makedirs(os.path.dirname(ISSUE_LOG_PATH), exist_ok=True)
    with open(ISSUE_LOG_PATH, "w", newline="") as issue_fp:
        issue_writer = csv.writer(issue_fp)
        issue_writer.writerow(
            [
                "i",
                "j",
                "mu",
                "D",
                "z",
                "mode",
                "path",
                "warning_seen",
                "bad_value_used",
                "issue_tags",
                "warning_messages",
                "detail",
                "val_before_clamp",
                "val_after_clamp",
            ]
        )

        def log_issue(
            i,
            j,
            mu,
            D,
            z_score,
            path,
            warning_seen,
            bad_value_used,
            issue_tags,
            warning_messages,
            detail,
            val_before_clamp,
            val_after_clamp,
        ):
            nonlocal issue_count, issue_mu_min, issue_mu_max, issue_D_min, issue_D_max
            issue_count += 1
            issue_writer.writerow(
                [
                    i,
                    j,
                    mu,
                    D,
                    z_score,
                    LUT_MODE,
                    path,
                    int(bool(warning_seen)),
                    int(bool(bad_value_used)),
                    "; ".join(sorted(issue_tags)),
                    " | ".join(sorted(warning_messages)),
                    detail,
                    val_before_clamp,
                    val_after_clamp,
                ]
            )
            if issue_mu_min is None or mu < issue_mu_min:
                issue_mu_min = mu
            if issue_mu_max is None or mu > issue_mu_max:
                issue_mu_max = mu
            if issue_D_min is None or D < issue_D_min:
                issue_D_min = D
            if issue_D_max is None or D > issue_D_max:
                issue_D_max = D

        start_time = time.time()

        for i, mu in enumerate(mu_vals):
            # Precise Progress Bar
            if i % 10 == 0:
                elapsed = time.time() - start_time
                # Estimate remaining time
                if i > 0:
                    rate = i / elapsed
                    remaining = (N_MU - i) / rate
                    print(f"  Row {i}/{N_MU} | Time: {elapsed:.0f}s | ETA: {remaining:.0f}s", end='\r')
                else:
                    print(f"  Row {i}/{N_MU} | Starting...", end='\r')

            for j, D in enumerate(D_vals):
                sigma = np.sqrt(D)
                z_score = (mu - V_TH) / sigma
                use_fast_path = False
                issue_tags = set()
                warning_messages = set()
                warning_seen = False
                bad_value_used = False
                detail = ""
                path = "exact"

                if LUT_MODE == "approx":
                    if z_score > APPROX_Z_LIMIT:
                        val = APPROX_SUPRATHRESH_LAMBDA
                        use_fast_path = True
                        path = "approx_suprathreshold"
                    elif z_score < -APPROX_Z_LIMIT:
                        val = APPROX_SUBTHRESH_LAMBDA
                        use_fast_path = True
                        path = "approx_subthreshold"

                if not use_fast_path:
                    if abs(z_score) >= ASYMPTOTIC_Z_THRESHOLD:
                        if z_score < 0:
                            # Rare-event limit: exponentially small rate -> lambda ~ 0.
                            val = 0.0
                            path = "asymptotic_subthreshold"
                        else:
                            # Low-noise suprathreshold limit: deterministic ISI.
                            # Dimensionless deterministic rate for tau=1, V_reset=0, V_th=1.
                            # lambda = rate / CV, with a CV floor to avoid divergence as CV -> 0.
                            denom = np.log((mu - V_RESET) / (mu - V_TH))
                            if np.isfinite(denom) and denom > 0.0:
                                rate_det = 1.0 / denom
                                val = rate_det / max(ASYMPTOTIC_CV_FLOOR, CV_EPS)
                                path = "asymptotic_suprathreshold"
                            else:
                                val = APPROX_SUPRATHRESH_LAMBDA
                                path = "asymptotic_suprathreshold_fallback"
                                bad_value_used = True
                                issue_tags.add("asymptotic_invalid_denom")
                                detail = "Invalid deterministic-period denominator in asymptotic branch."
                    else:
                        path = "exact_eval"

                if not use_fast_path and path == "exact_eval":
                    try:
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter("always", RuntimeWarning)
                            r_val = rate_whitenoise_benji(mu, np.sqrt(D))
                            cv_val = cv_benji(mu, D)

                        for warn in caught:
                            msg = str(warn.message).strip()
                            msg_lower = msg.lower()
                            if (
                                "overflow" in msg_lower
                                or "underflow" in msg_lower
                                or "divide by zero" in msg_lower
                                or "invalid value" in msg_lower
                            ):
                                warning_seen = True
                                warning_messages.add(msg)
                                issue_tags.add("runtime_warning")

                        if cv_val > CV_EPS:
                            val = r_val / cv_val
                        else:
                            val = APPROX_SUPRATHRESH_LAMBDA if mu > V_TH else APPROX_SUBTHRESH_LAMBDA
                            bad_value_used = True
                            issue_tags.add("cv_below_eps")
                            detail = f"cv={cv_val} <= CV_EPS={CV_EPS}; used fallback."
                    except (FloatingPointError, OverflowError, ZeroDivisionError, ValueError, RuntimeError) as exc:
                        val = APPROX_SUPRATHRESH_LAMBDA if mu > V_TH else APPROX_SUBTHRESH_LAMBDA
                        bad_value_used = True
                        issue_tags.add(type(exc).__name__)
                        detail = str(exc)

                # 3. Final Clamp
                val_before_clamp = val
                if np.isnan(val) or np.isinf(val):
                    val = 0.0
                    bad_value_used = True
                    issue_tags.add("invalid_result")
                    detail = f"Non-finite val encountered ({val_before_clamp}); set to 0."
                if val > LAMBDA_MAX:
                    val = LAMBDA_MAX
                    issue_tags.add("clamped_lambda_max")
                val_after_clamp = val

                table[i,j] = val

                if warning_seen or bad_value_used:
                    log_issue(
                        i=i,
                        j=j,
                        mu=mu,
                        D=D,
                        z_score=z_score,
                        path=path,
                        warning_seen=warning_seen,
                        bad_value_used=bad_value_used,
                        issue_tags=issue_tags,
                        warning_messages=warning_messages,
                        detail=detail,
                        val_before_clamp=val_before_clamp,
                        val_after_clamp=val_after_clamp,
                    )

    header = f"{MU_MIN} {MU_MAX} {N_MU} {D_MIN} {D_MAX} {N_SIGMA}"
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    np.savetxt(OUTPUT_PATH, table, header=header, comments='')
    print(f"\n✓ Saved LUT to {OUTPUT_PATH}")
    save_binary_lut(table)
    if issue_count > 0:
        print(f"⚠ Logged {issue_count} numerical issues to {ISSUE_LOG_PATH}")
        print(f"  Issue parameter range: mu in [{issue_mu_min}, {issue_mu_max}], D in [{issue_D_min}, {issue_D_max}]")
    else:
        print(f"✓ No numerical issues detected. Log written to {ISSUE_LOG_PATH}")
    return table, mu_vals, D_vals

if __name__ == "__main__":
    PARAMS = load_params_from_header()
    if PARAMS["PARAM_RECURRENCE"] == 0:
        print("Skipping LUT")
    else:
        table, mu_vals, D_vals = generate_lut()
        save_heatmap(table, mu_vals, D_vals)
