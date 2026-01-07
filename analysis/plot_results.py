import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os

# --- CONFIGURATION ---
DENSITY_FILE = "data/diffusion_drift_data.csv"
ACTIVITY_FILE = "data/activity_data.csv"
OUTPUT_DIR = "plots"
SMOOTHING_WINDOW_MS = 10.0  # Window size for moving average (5ms is standard)

def plot_drift_diffusion():
    
    # --- 0. Ensure Output Directory Exists ---
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}/")

    # --- 1. Load Data ---
    print("Loading data...")
    try:
        # Load Density (Handle potential trailing comma from C)
        data_p = np.genfromtxt(DENSITY_FILE, delimiter=",")
        if np.isnan(data_p[0, -1]): 
            data_p = data_p[:, :-1]
        
        # Load Activity
        data_a = np.genfromtxt(ACTIVITY_FILE, delimiter=",", skip_header=1)
        time_vals = data_a[:, 0]
        # Convert Rate from kHz (events/ms) to Hz (events/s)
        activity_hz_raw = data_a[:, 1] * 1000.0 
        mass_vals = data_a[:, 2]
        
    except Exception as e:
        print(f"Error loading files: {e}")
        print("Ensure 'data/' folder exists and contains CSVs.")
        return

    # --- 2. Setup Figure ---
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), 
                                        gridspec_kw={'height_ratios': [2, 1.5, 1]})
    
    # =========================================================
    # Subplot 1: Probability Density Evolution P(v,t)
    # =========================================================
    total_snapshots, grid_size = data_p.shape
    total_time_steps = len(time_vals)
    stride = total_time_steps / total_snapshots
    
    indices_to_plot = np.linspace(0, total_snapshots - 1, 6, dtype=int)
    colors = cm.viridis(np.linspace(0, 1, len(indices_to_plot)))
    max_p_height = 0
    
    for i, idx in enumerate(indices_to_plot):
        time_idx = int(idx * stride)
        if time_idx >= len(time_vals): time_idx = len(time_vals) - 1   
        current_time = time_vals[time_idx]
        
        ax1.plot(data_p[idx], color=colors[i], linewidth=2, label=f"t={current_time:.0f} ms")
        if idx > 0: max_p_height = max(max_p_height, np.max(data_p[idx]))

    ax1.set_title("1. Evolution of Probability Density P(v,t)")
    ax1.set_ylabel("Density P(v)")
    ax1.set_xlabel("Voltage (normalized)")
    ax1.axhline(0, color='black', linewidth=0.5, linestyle='--')
    if max_p_height > 0: ax1.set_ylim(0, max_p_height * 1.2)
    
    # Mark Rest and Threshold
    ax1.axvline(grid_size//2, color='red', linestyle=':', alpha=0.5, label='Rest (Reset)')
    ax1.axvline(grid_size-1, color='red', linestyle='--', alpha=0.5, label='Threshold')
    ax1.legend(loc='upper right', fontsize='small')
    ax1.grid(True, alpha=0.3)

    # =========================================================
    # Subplot 2: Population Activity A(t) (With Smoothing)
    # =========================================================
    
    # A. Plot Raw Data (Light Orange Background)
    # This shows the high-frequency grid noise
    ax2.plot(time_vals, activity_hz_raw, color='tab:orange', alpha=0.25, 
             linewidth=0.5, label="Raw (Grid Scale Noise)")
    
    # B. Calculate & Plot Smoothed Data (Red Line)
    dt_sim = time_vals[1] - time_vals[0]
    window_bins = int(SMOOTHING_WINDOW_MS / dt_sim)
    
    if window_bins > 1:
        kernel = np.ones(window_bins) / window_bins
        activity_smooth = np.convolve(activity_hz_raw, kernel, mode='same')
        
        # Remove convolution artifacts at edges
        valid_idx = slice(window_bins, -window_bins)
        
        ax2.plot(time_vals, activity_smooth, color='#D62728', linewidth=1.5, 
                 label=f"Smoothed ({SMOOTHING_WINDOW_MS}ms window)")
        
        # Dynamic Y-Limits based on SMOOTHED signal (ignoring raw outliers)
        # We use percentiles to be robust against startup transients
        y_min = np.percentile(activity_smooth[window_bins:], 1)
        y_max = np.percentile(activity_smooth[window_bins:], 99)
        margin = (y_max - y_min) * 0.5
        ax2.set_ylim(y_min - margin, y_max + margin)
        
    else:
        print("Warning: Timestep too large for requested smoothing window.")
        ax2.plot(time_vals, activity_hz_raw, color='red', linewidth=1.0)

    # C. Mean Line
    mean_val = np.mean(activity_hz_raw)
    ax2.axhline(mean_val, color='blue', linestyle='--', linewidth=1.5, 
                label=f"Global Mean: {mean_val:.2f} Hz")

    ax2.set_title("2. Population Activity A(t)")
    ax2.set_ylabel("Firing Rate (Hz)")
    ax2.set_xlim(0, time_vals[-1])
    ax2.legend(loc='upper right', fontsize='small')
    ax2.grid(True, alpha=0.3)

    # =========================================================
    # Subplot 3: Probability Mass Check
    # =========================================================
    ax3.plot(time_vals, mass_vals, color='tab:green', linewidth=1.5)
    ax3.set_title("3. Conservation of Probability (Mass)")
    ax3.set_ylabel("Total Mass")
    ax3.set_xlabel("Time (ms)")
    ax3.axhline(1.0, color='red', linestyle='--', linewidth=1, label="Target (1.0)")

    # Zoom in to show tiny deviations
    ax3.set_xlim(0, time_vals[-1])
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='lower right', fontsize='small')

    plt.tight_layout()
    
    # --- 3. Save ---
    output_path = f"{OUTPUT_DIR}/simulation_results.png"
    plt.savefig(output_path, dpi=200)
    print(f"✓ Plot saved to: {output_path}")

if __name__ == "__main__":
    plot_drift_diffusion()