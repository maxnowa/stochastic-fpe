import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# --- CONFIGURATION ---
ACTIVITY_FILE = "data/activity.bin"
OUTPUT_DIR = "plots"

# Defaults
T_MAX = 10000.0 
METHOD_ID = 0 # Default to 0 if config is missing

# Method Labels
METHOD_NAMES = {
    0: "No Correction",
    1: "Correction Term",
    2: "Renormalization"
}

# 1. Load Configuration
try:
    from config import PARAMS
    T_MAX = PARAMS.get("PARAM_T_MAX", T_MAX)
    METHOD_ID = PARAMS.get("PARAM_METHOD", 0)
except ImportError:
    print("Warning: config.py not found, using defaults.")

# 2. Dynamic Output Filename Generation
# This ensures the filename always matches the method currently being simulated
OUTPUT_FILENAME = f"mass_conservation_METHOD={METHOD_ID}.png"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

def plot_mass():
    # 3. Setup Directories
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"Loading {ACTIVITY_FILE}...", end=" ", flush=True)
    if not os.path.exists(ACTIVITY_FILE):
        print(f"\nError: File {ACTIVITY_FILE} not found.")
        return

    # 4. Load Data (Memmap is instant)
    try:
        raw_data = np.memmap(ACTIVITY_FILE, dtype=np.float32, mode='r')
        data_reshaped = raw_data.reshape(-1, 2)
    except Exception as e:
        print(f"\nError reading file: {e}")
        return
    
    # Extract just the Mass column (index 1)
    mass_full = data_reshaped[:, 1]
    total_steps = mass_full.shape[0]
    print(f"Steps: {total_steps}")

    # 5. Downsample (Target ~100k points)
    target_points = 100000
    step = max(1, total_steps // target_points)
    
    print(f"Downsampling by {step}...", end=" ", flush=True)
    mass_plot = mass_full[::step]
    
    # Create matching time axis
    time_plot = np.linspace(0, T_MAX, len(mass_plot))
    print("Done.")

    # 6. Moving Average (Smoothing)
    window_ms = 100.0  
    dt_plot = time_plot[1] - time_plot[0]
    window_bins = int(window_ms / dt_plot)

    if window_bins > 1:
        kernel = np.ones(window_bins) / window_bins
        mass_smooth = np.convolve(mass_plot, kernel, mode='same')
        valid_slice = slice(window_bins, -window_bins)
    else:
        mass_smooth = mass_plot
        valid_slice = slice(None)

    # 7. Plotting
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Raw Data
    ax.plot(time_plot, mass_plot, color='tab:green', alpha=0.3, linewidth=0.5, label="Raw Mass")
    
    # Smoothed Trend
    if window_bins > 1:
        ax.plot(time_plot[valid_slice], mass_smooth[valid_slice], 
                color='darkgreen', linewidth=2.0, label=f"Trend ({window_ms:.0f}ms window)")

    # Global Mean
    global_mean = np.mean(mass_plot)
    ax.axhline(global_mean, color='blue', linestyle=':', label=f"Mean ({global_mean:.6f})")
    
    # Target Line
    ax.axhline(1.0, color='red', linestyle='--', linewidth=1, label="Target (1.0)")

    # --- TITLE GENERATION ---
    method_name = METHOD_NAMES.get(METHOD_ID, f"Unknown ({METHOD_ID})")
    duration_sec = T_MAX / 1000.0
    
    title_text = (f"Mass Conservation Check\n"
                  f"Method: {method_name} (ID={METHOD_ID}) | Duration: {duration_sec:.1f} s")
    
    ax.set_title(title_text)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Total Probability Mass")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    # Smart Y-Limit
    mn, mx = np.min(mass_plot), np.max(mass_plot)
    if abs(mx - mn) < 0.05:
        mid = (mx + mn) / 2
        ax.set_ylim(mid - 0.05, mid + 0.05)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=150)
    print(f"\n✓ Plot saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    plot_mass()