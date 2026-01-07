import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
import os
from analysis.utils import periodogram # Assuming this is the function you showed earlier

try:
    from config import PARAMS
except ImportError:
    print("Error: config.py not found.")
    exit()

# Load parameters
N = PARAMS["PARAM_N_NEURONS"]
mu = PARAMS["PARAM_MU"]
D = PARAMS["PARAM_D"]
tau = PARAMS["PARAM_TAU"]
V_th = PARAMS["PARAM_V_TH"]
V_reset = PARAMS["PARAM_V_RESET"]
dt_net = PARAMS["PARAM_DT_NET"] # Microscopic dt
fpe_file = "data/activity_data.csv"

# --- 1. Load FPE Data ---
if not os.path.exists(fpe_file):
    print(f"Error: {fpe_file} not found.")
    exit()

data_fpe = np.genfromtxt(fpe_file, delimiter=",", skip_header=1)
if np.isnan(data_fpe[0, -1]): data_fpe = data_fpe[:, :-1]

time_fpe = data_fpe[:, 0]
# Convert kHz -> Hz immediately for physical consistency
activity_fpe = data_fpe[:, 1] * 1000.0 

dt_fpe = np.mean(np.diff(time_fpe)) # Sampling interval (ms)
print(f"FPE Data Loaded: T={time_fpe[-1]}ms, dt_save={dt_fpe:.3f}ms")

# --- 2. Network Simulation ---
def run_network(N, mu, D, tau, V_th, V_reset, dt_net):
    print("\nRunning Network Simulation (Monte Carlo)...")
    
    T_sim = time_fpe[-1]
    steps = int(T_sim / dt_net)
    v = np.zeros(N) # Start at reset
    activity_net = []
    
    diffusion_coeff = np.sqrt(2 * D / tau)
    # Pre-calculate factors for loop speed
    drift_factor = dt_net / tau
    noise_factor = diffusion_coeff * np.sqrt(dt_net)

    for _ in range(steps):
        noise = np.random.normal(0, 1, N)
        
        # Euler-Maruyama
        v += ((mu - v) * drift_factor) + (noise * noise_factor)
        
        # Spiking
        spikes = v >= V_th
        v[spikes] = V_reset
        
        # Rate in Hz (Count / (N * dt_seconds))
        # dt_net is in ms, so divide by (dt_net/1000)
        rate_hz = np.sum(spikes) / (N * (dt_net / 1000.0))
        activity_net.append(rate_hz)
        
    return np.array(activity_net)

# --- 3. PSD Calculation ---
def calculate_psd(activity_net, activity_fpe, method="bartlett"):
    print(f"\nComputing Spectra using {method} method...")
    
    # Sampling rates in seconds
    dt_net_sec = dt_net / 1000.0
    dt_fpe_sec = dt_fpe / 1000.0
    
    if method == "welch":
        # Welch is robust but requires normalization if dt's differ wildly
        # Using raw here to check magnitude
        fs_fpe = 1.0 / dt_fpe_sec
        fs_net = 1.0 / dt_net_sec
        
        freq_fpe, psd_fpe = welch(activity_fpe, fs=fs_fpe, nperseg=2048)
        freq_net, psd_net = welch(activity_net, fs=fs_net, nperseg=2048)
        
    elif method == "bartlett":
        # Bartlett (Periodogram) with df=1.0 Hz gives high resolution
        # Ideal for validating the analytical curve
        freq_fpe, psd_fpe = periodogram(activity_fpe, dt_fpe_sec, df=1.0)
        freq_net, psd_net = periodogram(activity_net, dt_net_sec, df=1.0)

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    
    # Plot Network (Ground Truth) - Light Gray
    plt.loglog(freq_net, psd_net, color='gray', alpha=0.5, label='Network (Microscopic)')
    
    # Plot FPE (Validation) - Red Dashed
    plt.loglog(freq_fpe, psd_fpe, color='red', linestyle='--', linewidth=2, label='FPE Solver (Macroscopic)')
    
    plt.title(f"Validation Check (N={N}, $\mu$={mu}, D={D})")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density ($Hz^2/Hz$)")
    plt.xlim(1, 1000) # Biological range
    
    # If the simulation is working, these lines should overlap in magnitude
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    
    if not os.path.exists("plots"): os.makedirs("plots")
    plt.savefig("plots/validation_psd.png", dpi=150)
    print("\n✓ Validation Plot saved to plots/validation_psd.png")
    plt.show()

if __name__ == "__main__":
    # Run
    activity_net = run_network(N, mu, D, tau, V_th, V_reset, dt_net)
    
    # Use Bartlett (raw) to verify the actual noise magnitude
    calculate_psd(activity_net, activity_fpe, method="bartlett")