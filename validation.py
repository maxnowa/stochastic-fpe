# validate.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
import os

# Import parameters directly to ensure matching physics
try:
    from config import PARAMS
except ImportError:
    print("Error: config.py not found. Please create it first.")
    exit()

def run_network_validation(fpe_file="data/activity_data.csv"):
    # --- 1. Load FPE Data (Macroscopic Model) ---
    if not os.path.exists(fpe_file):
        print(f"Error: File {fpe_file} not found. Run the C simulation first.")
        return

    try:
        data_fpe = np.genfromtxt(fpe_file, delimiter=",", skip_header=1)
        # Handle case where file might have extra columns or nans
        if np.isnan(data_fpe[0, -1]): 
            data_fpe = data_fpe[:, :-1]
            
        time_fpe = data_fpe[:, 0]
        activity_fpe = data_fpe[:, 1]
        
        # Calculate effective sampling rate
        dt_fpe = np.mean(np.diff(time_fpe))
        fs_fpe = 1000.0 / dt_fpe
        print(f"FPE Data Loaded: T={time_fpe[-1]}ms, dt={dt_fpe:.4f}ms, fs={fs_fpe:.1f}Hz")
        
    except Exception as e:
        print(f"Error reading FPE data: {e}")
        return

    # --- 2. Run Network Simulation (Microscopic Ground Truth) ---
    print("\nRunning Network Simulation (Monte Carlo)...")
    
    # Load parameters from config
    N = PARAMS["N_NEURONS"]
    mu = PARAMS["MU"]
    D = PARAMS["D"]
    tau = PARAMS["TAU"]
    V_th = PARAMS["V_TH"]
    V_reset = PARAMS["V_RESET"]
    dt_net = PARAMS["DT_NET"]
    
    # Duration must match the FPE file exactly for fair comparison
    T_sim = time_fpe[-1] 
    steps = int(T_sim / dt_net)
    
    # Initialization
    v = np.zeros(N) 
    activity_net = []
    
    # Physics Pre-calculation
    # Eq: dv = (mu-v)/tau * dt + (sqrt(2D)/tau) * sqrt(dt) * Noise
    sigma_noise = np.sqrt(2 * D) 
    sqrt_dt = np.sqrt(dt_net)
    diffusion_scale = (sigma_noise / tau) * sqrt_dt
    drift_factor = dt_net / tau

    # Simulation Loop
    for _ in range(steps):
        noise = np.random.normal(0, 1, N)
        
        # Euler-Maruyama Update
        # v[t+1] = v[t] + drift + diffusion
        v += ((mu - v) * drift_factor) + (noise * diffusion_scale)
        
        # Spiking logic
        spikes = v >= V_th
        num_spikes = np.sum(spikes)
        v[spikes] = V_reset
        
        # Instantaneous Population Rate
        # Rate = (Count) / (N * dt) -> units of kHz if dt is ms
        rate = num_spikes / (N * dt_net)
        activity_net.append(rate)

    activity_net = np.array(activity_net)
    fs_net = 1000.0 / dt_net
    print(f"Network Sim Complete: dt={dt_net}ms, fs={fs_net:.1f}Hz")

    # --- 3. Compute Normalization & PSD ---
    print("\nComputing Power Spectra...")

    # Z-Score Normalization (Mean=0, Std=1)
    # This removes unit mismatch (kHz vs Hz) and focuses on dynamics
    activity_fpe_norm = (activity_fpe - np.mean(activity_fpe)) / np.std(activity_fpe)
    activity_net_norm = (activity_net - np.mean(activity_net)) / np.std(activity_net)

    # Welch's Method
    # nperseg=2048 gives good frequency resolution for T=1000ms
    freq_fpe, psd_fpe = welch(activity_fpe_norm, fs=fs_fpe, nperseg=2048)
    freq_net, psd_net = welch(activity_net_norm, fs=fs_net, nperseg=2048)

    # --- 4. Plot Comparison ---
    if not os.path.exists("plots"):
        os.makedirs("plots")

    plt.figure(figsize=(10, 6))
    
    # Plot Network (Ground Truth)
    plt.loglog(freq_net, psd_net, color='gray', linewidth=3, alpha=0.6, label='Network (Microscopic)')
    
    # Plot FPE (Theory)
    plt.loglog(freq_fpe, psd_fpe, color='red', linestyle='--', linewidth=2, label='Stochastic FPE (Macroscopic)')

    plt.title(f"Validation: Normalized Power Spectrum (N={N}, $\mu$={mu})")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density (Normalized)")
    
    # Limit x-axis to relevant biological frequencies (cutoff high freq noise)
    plt.xlim(1, 2000) 
    
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    
    save_path = "plots/validation_psd.png"
    plt.savefig(save_path, dpi=300)
    print(f"\n✓ Plot saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    run_network_validation()