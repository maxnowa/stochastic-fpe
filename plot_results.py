import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os

# UPDATE DEFAULTS
def plot_drift_diffusion(density_file="data/diffusion_drift_data.csv", activity_file="data/activity_data.csv"):
    
    # --- 0. Ensure Output Directory Exists ---
    if not os.path.exists("plots"):
        os.makedirs("plots")
        print("Created directory: plots/")

    # --- 1. Load Data ---
    try:
        # Load Density
        data_p = np.genfromtxt(density_file, delimiter=",")
        if np.isnan(data_p[0, -1]): data_p = data_p[:, :-1]
        
        # Load Activity
        data_a = np.genfromtxt(activity_file, delimiter=",", skip_header=1)
        time_vals = data_a[:, 0]
        activity_vals = data_a[:, 1]
        mass_vals = data_a[:, 2]
    except Exception as e:
        print(f"Error loading files from 'data/' folder: {e}")
        print("Did you run the C simulation with the updated paths?")
        return

    # --- 2. Calculate Stride ---
    total_snapshots, grid_size = data_p.shape
    total_time_steps = len(time_vals)
    stride = total_time_steps / total_snapshots
    print(f"Detected saving stride: 1 density snapshot = {stride:.1f} time steps")

    # --- 3. Setup Figure ---
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1, 1]})
    
    # --- Subplot 1: Probability Density ---
    indices_to_plot = np.linspace(0, total_snapshots - 1, 6, dtype=int)
    colors = cm.viridis(np.linspace(0, 1, len(indices_to_plot)))
    max_p_height = 0
    
    for i, idx in enumerate(indices_to_plot):
        time_idx = int(idx * stride)
        if time_idx >= len(time_vals): time_idx = len(time_vals) - 1   
        current_time = time_vals[time_idx]
        
        ax1.plot(data_p[idx], color=colors[i], linewidth=2, label=f"t={current_time:.0f} ms")
        
        if idx > 0: max_p_height = max(max_p_height, np.max(data_p[idx]))

    ax1.set_title("1. Evolution of Probability Density p(x,t)")
    ax1.set_ylabel("Density p(x)")
    ax1.set_xlabel("x")
    ax1.axhline(0, color='black', linewidth=0.5, linestyle='--')
    if max_p_height > 0: ax1.set_ylim(0, max_p_height * 1.2)
    ax1.axvline(grid_size//2, color='red', linestyle=':', alpha=0.5, label='Rest')
    ax1.axvline(grid_size-1, color='red', linestyle='--', alpha=0.5, label='Threshold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # --- Subplot 2: Firing Rate ---
    ax2.plot(time_vals, activity_vals, color='tab:orange', linewidth=0.8)
    ax2.set_title("2. Population Activity A(t)")
    ax2.set_ylabel("Rate (Hz)")
    ax2.set_xlim(0, time_vals[-1])
    ax2.grid(True, alpha=0.3)
    
    # --- Subplot 3: Mass ---
    ax3.plot(time_vals, mass_vals, color='tab:green', linewidth=1.5)
    ax3.set_title("3. Total Probability Mass")
    ax3.set_ylabel("Mass")
    ax3.set_xlabel("Time (ms)")
    ax3.axhline(1.0, color='red', linestyle='--', linewidth=1, label="Target")
    ax3.set_ylim(0.9, 1.1)
    ax3.set_xlim(0, time_vals[-1])
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='lower right')

    plt.tight_layout()
    
    # --- 4. Save Logic ---
    output_path = "plots/simulation_results.png"
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to: {output_path}")

if __name__ == "__main__":
    plot_drift_diffusion()