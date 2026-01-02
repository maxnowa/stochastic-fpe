import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def plot_drift_diffusion(filename="diffusion_drift_data.csv"):
    # 1. Load Data
    # delimiter="," handles the CSV format
    # [:, :-1] removes the last column if your C code leaves a trailing comma
    try:
        data = np.genfromtxt(filename, delimiter=",")
        # specific check for trailing comma common in C loops
        if np.isnan(data[0, -1]):
            data = data[:, :-1]
    except Exception as e:
        print(f"Error loading file: {e}")
        print("Make sure you have run the C simulation first!")
        return

    # 2. Setup Plot
    plt.figure(figsize=(12, 7))
    total_snapshots, grid_size = data.shape
    
    # We will pick 6 snapshots evenly spaced in time
    indices_to_plot = np.linspace(0, total_snapshots - 1, 6, dtype=int)
    
    # Create a color map to show time evolution (Blue -> Red)
    colors = cm.viridis(np.linspace(0, 1, len(indices_to_plot)))

    # 3. Plot Lines
    for i, idx in enumerate(indices_to_plot):
        # Calculate roughly what 'time' this is (percentage)
        progress = (idx / total_snapshots) * 100
        
        label_text = f"Start (t=0)" if idx == 0 else f"Step {idx} ({progress:.0f}%)"
        
        plt.plot(data[idx], color=colors[i], linewidth=2, label=label_text)

    # 4. Styling
    plt.title("Drift-Diffusion Simulation Results")
    plt.xlabel("Grid Index (x)")
    plt.ylabel("Probability Density (p)")
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Highlight the movement direction
    plt.arrow(grid_size/2, 0.5, grid_size/4, 0, head_width=0.05, head_length=5, fc='k', ec='k', alpha=0.5)
    plt.text(grid_size/2 + 10, 0.55, "Drift Direction", fontsize=10, alpha=0.7)

    print(f"Loaded {total_snapshots} time steps with {grid_size} grid points.")
    print("Displaying plot...")
    plt.show()

if __name__ == "__main__":
    plot_drift_diffusion()