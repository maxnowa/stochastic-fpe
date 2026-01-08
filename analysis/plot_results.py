import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
import sys

try:
    from config import PARAMS
except ImportError:
    print("Error: Could not import config.py. Make sure you are running from project root.")
    sys.exit(1)

# --- CONFIGURATION FROM PARAMS ---
N_GRID = PARAMS["PARAM_GRID_N"]
T_MAX = PARAMS["PARAM_T_MAX"]

# Files
DENSITY_FILE = "data/density.bin"
ACTIVITY_FILE = "data/activity.bin"
OUTPUT_DIR = "plots"
SMOOTHING_WINDOW_MS = 10.0

def plot_drift_diffusion():
    
    # --- 0. Ensure Output Directory Exists ---
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # --- 1. Load Data (Binary) ---
    print(f"Loading binary data (Grid N={N_GRID})...")
    
    try:
        # A. Load Activity [A_t, mass] -> Shape: (Steps, 2)
        # Use float32 because we used 'float' in C
        raw_activity = np.fromfile(ACTIVITY_FILE, dtype=np.float32)
        data_a = raw_activity.reshape(-1, 2)
        # B. Load Density [p0, p1, ... pN] -> Shape: (Snapshots, N)
        raw_density = np.fromfile(DENSITY_FILE, dtype=np.float32)
        data_p = raw_density.reshape(-1, N_GRID)
        
    except FileNotFoundError:
        print("Error: Binary data files not found in 'data/'. Run the solver first.")
        return
    except ValueError as e:
        print(f"Error reshaping data: {e}")
        print(f"Check if N_GRID in config.py ({N_GRID}) matches the C simulation.")
        return

    # --- 2. Reconstruct Time Axes ---
    # We calculate the effective dt based on the file size to be robust
    # (in case you saved every 100 steps or every 1 step)
    
    num_activity_steps = len(data_a)
    dt_activity = T_MAX / num_activity_steps  # Effective dt for activity plot
    time_vals = np.linspace(0, T_MAX, num_activity_steps)
    
    # Extract columns
    # Activity was saved as [Rate_Hz, Mass] (or similar) in your C code
    # Assuming column 0 is A(t) and column 1 is Mass
    activity_hz_raw = data_a[:, 0] * 1000.0  # Convert if C saved kHz, otherwise check units
    mass_vals = data_a[:, 1]

    # --- 3. Setup Figure ---
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), 
                                        gridspec_kw={'height_ratios': [2, 1.5, 1]})
    
    # =========================================================
    # Subplot 1: Probability Density Evolution P(v,t)
    # =========================================================
    total_snapshots = data_p.shape[0]
    
    # Pick 6 evenly spaced snapshots to plot
    indices_to_plot = np.linspace(0, total_snapshots - 1, 6, dtype=int)
    colors = cm.viridis(np.linspace(0, 1, len(indices_to_plot)))
    max_p_height = 0
    
    for i, idx in enumerate(indices_to_plot):
        # Calculate time for this snapshot
        t_snapshot = idx * (T_MAX / total_snapshots)
        
        ax1.plot(data_p[idx], color=colors[i], linewidth=2, label=f"t={t_snapshot:.0f} ms")
        max_p_height = max(max_p_height, np.max(data_p[idx]))

    ax1.set_title(f"1. Evolution of Probability Density (N={N_GRID})")
    ax1.set_ylabel("Density P(v)")
    ax1.set_xlabel("Grid Index (Voltage proxy)")
    
    # Draw reference lines for Rest/Threshold
    # Assuming grid 0 to N. Reset is typically V_reset (mapped to index)
    # We'll plot vertical lines at boundaries
    ax1.axvline(0, color='red', linestyle=':', alpha=0.5, label='Min Voltage')
    ax1.axvline(N_GRID-1, color='red', linestyle='--', alpha=0.5, label='Threshold')
    
    ax1.set_ylim(0, 2.5)
    ax1.legend(loc='upper right', fontsize='small')
    ax1.grid(True, alpha=0.3)

    # =========================================================
    # Subplot 2: Population Activity A(t)
    # =========================================================
    # DOWNSAMPLE for plotting speed (e.g., plot every 10th point)
    # This keeps the visual "noise" but fixes the memory/overflow error
    ds_factor = 10 
    if len(time_vals) > 100000:
        ds_factor = int(len(time_vals) / 50000)
    
    # Plot Raw (Downsampled)
    ax2.plot(time_vals[::ds_factor], activity_hz_raw[::ds_factor], 
             color='tab:orange', alpha=0.3, linewidth=0.5, label="Raw Output")
    # # Plot Raw
    # ax2.plot(time_vals, activity_hz_raw, color='tab:orange', alpha=0.3, 
    #          linewidth=0.5, label="Raw Output")
    
    # Smoothing
    # Calculate window size in bins based on effective dt
    window_bins = int(SMOOTHING_WINDOW_MS / dt_activity)
    
    if window_bins > 1:
        kernel = np.ones(window_bins) / window_bins
        activity_smooth = np.convolve(activity_hz_raw, kernel, mode='same')
        # Trim edges
        valid_slice = slice(window_bins, -window_bins)
        
        ax2.plot(time_vals[valid_slice], activity_smooth[valid_slice], 
                 color='#D62728', linewidth=1.5, 
                 label=f"Smoothed ({SMOOTHING_WINDOW_MS}ms)")
        
        # Robust Y-limits
        y_vals = activity_smooth[valid_slice]
        if len(y_vals) > 0:
            y_min, y_max = np.percentile(y_vals, [1, 99])
            margin = (y_max - y_min) * 0.5
            #ax2.set_ylim(y_min - margin, y_max + margin)
    else:
        ax2.plot(time_vals, activity_hz_raw, color='red', linewidth=1.0)

    # Mean Line
    mean_val = np.mean(activity_hz_raw[int(len(activity_hz_raw)*0.1):]) # Ignore first 10%
    ax2.axhline(mean_val, color='blue', linestyle='--', linewidth=1.5, 
                label=f"Mean: {mean_val:.2f} Hz")

    ax2.set_title("2. Population Activity A(t)")
    ax2.set_ylabel("Firing Rate (Hz)")
    ax2.set_xlim(0, T_MAX)
    ax2.legend(loc='upper right', fontsize='small')
    ax2.grid(True, alpha=0.3)

    # =========================================================
    # Subplot 3: Mass Conservation
    # =========================================================
    ax3.plot(time_vals, mass_vals, color='tab:green', linewidth=1.5)
    ax3.set_title("3. Mass Conservation")
    ax3.set_ylabel("Total Mass")
    ax3.set_xlabel("Time (ms)")
    ax3.axhline(1.0, color='red', linestyle='--', linewidth=1)
    
    ax3.set_xlim(0, T_MAX)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    
    output_path = f"{OUTPUT_DIR}/simulation_results.png"
    plt.savefig(output_path, dpi=200)
    print(f"✓ Plot saved to: {output_path}")

if __name__ == "__main__":
    plot_drift_diffusion()