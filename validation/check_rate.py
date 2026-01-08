import numpy as np
import matplotlib.pyplot as plt
import os
from analysis.lif import rate_whitenoise_benji

# Try to import config
try:
    from config import PARAMS
except ImportError:
    print("Warning: config.py not found. Using default parameters.")
    PARAMS = None

# --- CONFIG ---
SMOOTHING_WINDOW_MS = 50.0  # Window size for smoothing

def steady_state_rate(mu, D, tau, V_th, V_reset, t_ref=0):
    """
    Lindner Approximation for steady state firing rate, sub- and suprathreshold regime.
    Valid only for low noise.
    """
    # subthreshold
    if mu <= V_th:
        # implements the subthreshold approximation from Lindner (Neural noosie script p. 66)
        term_pre = np.sqrt((2 * np.pi * D) / (mu - V_th)**2)  
        term_exp = np.exp((mu - V_th)**2 / (2 * D))
        T_total = tau * term_pre * term_exp
        return 1000.0 / T_total
        
    # suprathreshold
    else:
        term_det = np.log((mu - V_reset) / (mu - V_th))
        term_corr = (D / 2.0) * ((1.0 / (mu - V_th)**2) - (1.0 / (mu - V_reset)**2))
        T_total = tau * (term_det - term_corr) + t_ref
        return 1000.0 / T_total

def plot_stationary_check(fpe_file="data/activity.bin"):
    
    # --- 1. Load Parameters ---
    if PARAMS:
        mu = PARAMS["PARAM_MU"]
        D = PARAMS["PARAM_D"]
        tau = PARAMS["PARAM_TAU"]
        V_th = PARAMS["PARAM_V_TH"]
        V_reset = PARAMS["PARAM_V_RESET"]
        T_max = PARAMS["PARAM_T_MAX"] # Needed for time reconstruction
        N = PARAMS.get("PARAM_N_NEURONS", 500)
    else:
        mu, D, tau = 1.2, 0.01, 10.0
        V_th, V_reset = 1.0, 0.0
        T_max = 20000.0

    # --- 2. Load Simulation Data (Binary) ---
    if not os.path.exists(fpe_file):
        print(f"Error: {fpe_file} not found.")
        return

    try:
        # Read raw float32 data
        # CHANGED: Use memmap for instant loading
        raw_data = np.memmap(fpe_file, dtype=np.float32, mode='r')
        # Reshape: [Rate, Mass] (2 columns)
        data = raw_data.reshape(-1, 2)
        
        # Reconstruct Time Axis
        # CHANGED: Avoid creating full time array to save RAM
        num_steps = data.shape[0]
        dt_effective = T_max / num_steps
        
        # Extract Rate and Convert kHz -> Hz
        # Col 0 is Rate (in 1/ms), Col 1 is Mass
        # CHANGED: Keep as view, do not multiply yet
        rate_sim_view = data[:, 0]
        
    except ValueError:
        print(f"Error reading binary file {fpe_file}. Format mismatch.")
        return

    # --- 3. Calculate Theories ---
    # Exact (Siegert) from function lif.py 
    mu_dim = (mu - V_reset) / (V_th - V_reset)
    sigma_dim = np.sqrt(D) / (V_th - V_reset)
    rate_dimensionless = rate_whitenoise_benji(mu_dim, sigma_dim)
    rate_siegert = rate_dimensionless * (1000.0 / tau)
    
    # Approximation (Lindner script)
    rate_lindner = steady_state_rate(mu, D, tau, V_th, V_reset)
    
    # Simulation Mean (ignoring first 30% transient)
    start_idx = int(num_steps * 0.3)
    # CHANGED: Strided mean calculation to avoid loading 25GB
    stat_stride = max(1, num_steps // 1_000_000)
    rate_sim_mean = np.mean(rate_sim_view[start_idx::stat_stride]) * 1000.0
    
    # --- 4. Prepare Data for Plotting (Smoothing & Downsampling) ---
    # CHANGED: Downsample FIRST, then create time axis
    target_points = 100_000
    ds_factor = max(1, num_steps // target_points)
    print(f"Plotting downsampled by factor {ds_factor} ({num_steps} -> {num_steps//ds_factor} points)")

    # Prepare arrays for plotting
    rate_plot = rate_sim_view[::ds_factor] * 1000.0
    time_plot = np.linspace(0, T_max, len(rate_plot))

    # A. Calculate Smoothing
    # CHANGED: Visual smoothing on downsampled data
    window_bins = int(SMOOTHING_WINDOW_MS / (dt_effective * ds_factor))
    
    rate_smooth_plot = rate_plot
    time_smooth_plot = time_plot
    is_smoothed = False

    if window_bins > 1:
        kernel = np.ones(window_bins) / window_bins
        # Convolve small array
        rate_smooth_plot = np.convolve(rate_plot, kernel, mode='same')
        
        # Remove edge artifacts
        valid_slice = slice(window_bins, -window_bins)
        rate_smooth_plot = rate_smooth_plot[valid_slice]
        time_smooth_plot = time_plot[valid_slice]
        is_smoothed = True

    # --- 5. Plot ---
    plt.figure(figsize=(10, 7))  
    plt.rcParams['agg.path.chunksize'] = 10000 # Prevent rendering crash
    
    # Plot Raw (Light Orange, Downsampled)
    # Using rate_plot (which is already downsampled)
    plt.plot(time_plot, rate_plot, 
             label='FPE Simulation (Raw)', 
             color='tab:orange', linewidth=0.5, alpha=0.3)
    
    if is_smoothed:
        label_main = f'Smoothed ({SMOOTHING_WINDOW_MS}ms, Mean: {rate_sim_mean:.2f} Hz)'
    else:
        # If no smoothing, the main plot is just the raw data
        label_main = f'FPE Rate (Mean: {rate_sim_mean:.2f} Hz)'
    
    # Plot Smoothed (Red, Downsampled)
    plt.plot(time_smooth_plot, rate_smooth_plot, 
             color='#D62728', linewidth=1.5, label=label_main)
    
    # Plot Exact Theory
    plt.axhline(rate_siegert, color='blue', linestyle='--', linewidth=2, 
                label=f'Exact (Siegert): {rate_siegert:.2f} Hz')

    # Plot Lindner Approx
    plt.axhline(rate_lindner, color='green', linestyle=':', linewidth=2.5, 
                label=f'Approx (Lindner): {rate_lindner:.2f} Hz')
    
    # Annotate
    plt.title(f"Stationary Rate Validation ($N \\to \\infty$)\n$\\mu={mu}, D={D}, \\tau={tau}ms$")
    plt.xlabel("Time (ms)")
    plt.ylabel("Firing Rate (Hz)")
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    # Add info box (Errors only)
    info = (f"Siegert Err: {abs(rate_sim_mean-rate_siegert)/rate_siegert*100:.2f}%\n"
            f"Lindner Err: {abs(rate_sim_mean-rate_lindner)/rate_lindner*100:.2f}%")
    plt.text(time_plot[0], max(rate_sim_mean, rate_siegert) * 1.1, info, bbox=dict(facecolor='white', alpha=0.9))

    # Smart Y-limits based on smoothed data + margin
    y_max = max(rate_sim_mean, rate_siegert) * 2
    plt.ylim(0, y_max)
    
    save_path = "plots/check_stationary_rate.png"
    if not os.path.exists("plots"): os.makedirs("plots")
    plt.savefig(save_path, dpi=150)
    print(f"\n✓ Plot saved to {save_path}")
    print(f"  Exact (Siegert):  {rate_siegert:.2f} Hz")
    print(f"  Approx (Lindner): {rate_lindner:.2f} Hz")
    print(f"  Sim Mean:         {rate_sim_mean:.2f} Hz")
    plt.show()

if __name__ == "__main__":
    plot_stationary_check()