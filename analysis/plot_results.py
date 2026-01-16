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
        # Use memmap for instant, lazy loading
        raw_activity = np.memmap(ACTIVITY_FILE, dtype=np.float32, mode='r')
        data_a = raw_activity.reshape(-1, 2)
        
        # B. Load Density [p0, p1, ... pN] -> Shape: (Snapshots, N)
        # Use memmap for instant, lazy loading
        raw_density = np.memmap(DENSITY_FILE, dtype=np.float32, mode='r')
        data_p = raw_density.reshape(-1, N_GRID)
        
    except FileNotFoundError:
        print("Error: Binary data files not found in 'data/'. Run the solver first.")
        return
    except ValueError as e:
        print(f"Error reshaping data: {e}")
        print(f"Check if N_GRID in config.py ({N_GRID}) matches the C simulation.")
        return

    # --- 2. Setup Lazy Access ---
    num_activity_steps = data_a.shape[0]
    dt_activity = T_MAX / num_activity_steps
    
    # Extract Views (No RAM usage yet)
    activity_view = data_a[:, 0]
    mass_view = data_a[:, 1]

    # --- 3. Downsampling Setup ---
    target_points = 100_000
    step = max(1, num_activity_steps // target_points)
    
    print(f"Plotting Activity: Downsampling by factor of {step} (Total points: {num_activity_steps} -> {num_activity_steps//step})")

    # Generate Time Axis ONLY for the points we plot (Tiny array)
    # We slice the activity view first to get the exact length
    activity_hz_plot = activity_view[::step] * 1000.0
    time_plot = np.linspace(0, T_MAX, len(activity_hz_plot))

    # --- 4. Setup Figure ---
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), 
                                        gridspec_kw={'height_ratios': [2, 1.5, 1]})
    
    # =========================================================
    # Subplot 1: Probability Density Evolution P(v,t)
    # =========================================================
    total_snapshots = data_p.shape[0]
    indices_to_plot = np.linspace(0, total_snapshots - 1, 6, dtype=int)
    colors = cm.viridis(np.linspace(0, 1, len(indices_to_plot)))
    max_p_height = 0
    
    for i, idx in enumerate(indices_to_plot):
        t_snapshot = idx * (T_MAX / total_snapshots)
        # Reading single rows from memmap is fast
        ax1.plot(data_p[idx], color=colors[i], linewidth=2, label=f"t={t_snapshot:.0f} ms")
        max_p_height = max(max_p_height, np.max(data_p[idx]))

    ax1.set_title(f"1. Evolution of Probability Density (N={N_GRID})")
    ax1.set_ylabel("Density P(v)")
    ax1.set_xlabel("Grid Index (Voltage proxy)")
    
    ax1.axvline(0, color='red', linestyle=':', alpha=0.5, label='Min Voltage')
    ax1.axvline(N_GRID-1, color='red', linestyle='--', alpha=0.5, label='Threshold')
    
    ax1.set_ylim(0, 2.5)
    ax1.legend(loc='upper right', fontsize='small')
    ax1.grid(True, alpha=0.3)

    # =========================================================
    # Subplot 2: Population Activity A(t)
    # =========================================================
    
    # Plot Raw (Downsampled)
    ax2.plot(time_plot, activity_hz_plot, 
             color='tab:orange', alpha=0.3, linewidth=0.5, label="Raw Output")
    
    # Smoothing Logic (FAST VERSION)
    # We smooth the *downsampled* data (25k points), not the raw data (3B points).
    # This is instantaneous.
    
    # 1. Calculate how many bins the window represents in the full data
    raw_window_bins = int(SMOOTHING_WINDOW_MS / dt_activity)
    
    # 2. Scale that window down to our downsampled grid
    # e.g., if we skipped 100 points, the window is 100x smaller in indices
    ds_window = max(1, int(raw_window_bins / step))
    
    if ds_window > 1:
        kernel = np.ones(ds_window) / ds_window
        
        # Convolve only the small plotting array (Fast!)
        smoothed_plot = np.convolve(activity_hz_plot, kernel, mode='same')
        
        # Trim valid region to avoid edge artifacts
        valid_slice = slice(ds_window, -ds_window)
        
        if len(smoothed_plot[valid_slice]) > 0:
            ax2.plot(time_plot[valid_slice], smoothed_plot[valid_slice], 
                     color='#D62728', linewidth=1.5, 
                     label=f"Smoothed ({SMOOTHING_WINDOW_MS}ms)")
    else:
        # If window is smaller than 1 pixel, just plot the line
        ax2.plot(time_plot, activity_hz_plot, 
                 color='#D62728', linewidth=1.5, label="Smoothed (Window < 1px)")

    # Mean Line (Estimated from downsampled data)
    mean_val = np.mean(activity_hz_plot) 
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
    # Apply same downsampling
    mass_plot = mass_view[::step]
    mass_mean = np.mean(mass_plot)
    ax3.plot(time_plot, mass_plot, color='tab:green', linewidth=1.5)
    ax3.axhline(mass_mean, color='blue', linestyle='--', linewidth=1.5, 
                label=f"Mean: {mass_mean:.5f}")
    ax3.set_title("3. Mass Conservation")
    ax3.set_ylabel("Total Mass")
    ax3.set_xlabel("Time (ms)")
    ax3.axhline(1.0, color='red', linestyle='--', linewidth=1)
    ax3.legend(loc='upper right', fontsize='small')
    ax3.set_xlim(0, T_MAX)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    
    output_path = f"{OUTPUT_DIR}/simulation_results.png"
    plt.savefig(output_path, dpi=200)
    print(f"✓ Plot saved to: {output_path}")

if __name__ == "__main__":
    plot_drift_diffusion()