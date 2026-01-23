import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
import os
import sys

from analysis.utils import periodogram, log_smooth
from analysis.lif import rate_whitenoise_benji 
from config import PARAMS
from analysis.lif_susceptibility_psd import get_theoretical_curve

# Load parameters
N_neurons = PARAMS["PARAM_N_NEURONS"]
mu = PARAMS["PARAM_MU"]
D = PARAMS["PARAM_D"]
tau = PARAMS["PARAM_TAU"]
V_th = PARAMS["PARAM_V_TH"]
V_reset = PARAMS["PARAM_V_RESET"]
dt_net = PARAMS["PARAM_DT_NET"] 
T_max = PARAMS["PARAM_T_MAX"]
CV = PARAMS["PARAM_CV"]
R0 = PARAMS["PARAM_R0"]   
recurrence = PARAMS["PARAM_RECURRENCE"]
w = PARAMS["PARAM_W"]
delay = PARAMS["PARAM_DELAY"]

fpe_file = "data/activity.bin"

# --- 1. Load FPE Data (Binary) ---
if not os.path.exists(fpe_file):
    print(f"Error: {fpe_file} not found. Run the solver first.")
    sys.exit(1)

raw_data = np.memmap(fpe_file, dtype=np.float32, mode='r')
data_fpe = raw_data.reshape(-1, 2)

# Reconstruct Time Axis
time_fpe = np.linspace(0, T_max, data_fpe.shape[0])
activity_fpe = data_fpe[:, 0] * 1000.0 
dt_fpe = time_fpe[1] - time_fpe[0]
print(f"FPE Data Loaded: T={time_fpe[-1]}ms, dt_save={dt_fpe:.3f}ms")



# --- 2. Network Simulation ---
def run_network(N, mu, D, tau, V_th, V_reset, dt_net, w, recurrence=False):
    print("\nRunning Network Simulation (Monte Carlo)...")
    
    T_sim = time_fpe[-1]
    steps = int(T_sim / dt_net)
    spike_times = [[] for _ in range(int(N))]
    # 1. Pre-allocate Array (float32 saves 50% RAM vs standard float)
    activity_net = np.zeros(steps, dtype=np.float32)
    
    # 2. Report Memory Usage
    mem_mb = activity_net.nbytes / (1024 ** 2)
    print(f"   Steps: {steps:,} | Memory Allocated: {mem_mb:.2f} MB")
    
    v = np.zeros(int(N))
    
    diffusion_coeff = np.sqrt(2 * D / tau)
    drift_factor = dt_net / tau
    noise_factor = diffusion_coeff * np.sqrt(dt_net)

    buffer_size = int(delay/dt_net) + 1
    past_rate = np.zeros(buffer_size)
    write_idx = 0

    # 3. Use index 'i' to fill the pre-allocated array
    for i in range(steps):
        # get old rate from delay timesteps ago
        A_t = past_rate[write_idx]
        mu_eff = mu
        # set effective mu, else leave it just mu
        if recurrence:
            mu_eff = mu + w*A_t

        noise = np.random.normal(0, 1, int(N))
        v += ((mu_eff - v) * drift_factor) + (noise * noise_factor)
        spikes = v >= V_th
        # record spike times
        if np.any(spikes):
            fired_indices = np.where(spikes)[0]
            current_time = i * dt_net
            for idx in fired_indices:
                spike_times[idx].append(current_time)

        v[spikes] = V_reset
        
        rate_hz = np.sum(spikes) / (N * (dt_net / 1000.0))
        activity_net[i] = rate_hz
        # write new rate and increment write index
        past_rate[write_idx] = rate_hz/1000
        write_idx = (write_idx+1) % buffer_size
    return activity_net, spike_times

# --- 3. PSD Calculation ---
def calculate_psd(activity_net, activity_fpe, spike_times, method="bartlett"):
    print(f"\nComputing Spectra using {method} method...")
    
    dt_net_sec = dt_net / 1000.0
    dt_fpe_sec = dt_fpe / 1000.0
    
    if method == "welch":
        fs_fpe = 1.0 / dt_fpe_sec
        fs_net = 1.0 / dt_net_sec
        freq_fpe, psd_fpe = welch(activity_fpe, fs=fs_fpe, nperseg=2048)
        if activity_net is not np.nan:
            freq_net, psd_net = welch(activity_net, fs=fs_net, nperseg=2048)
        
    elif method == "bartlett":
        freq_fpe, psd_fpe = periodogram(activity_fpe, dt_fpe_sec, df=1)
        if activity_net is not np.nan:
            freq_net, psd_net = periodogram(activity_net, dt_net_sec, df=1)
    
    # apply smoothing
    freq_fpe, psd_fpe = log_smooth(freq_fpe, psd_fpe, bins_per_decade=20)
    if activity_net is not np.nan:
        freq_net, psd_net = log_smooth(freq_net, psd_net, bins_per_decade=20)
        
    # Calculate Theoretical White Noise Level (Poisson Limit)
    mu_dim = (mu - V_reset) / (V_th - V_reset)
    sigma_dim = np.sqrt(D) / (V_th - V_reset)
    rate_exact_hz = rate_whitenoise_benji(mu_dim, sigma_dim) * (1000.0 / tau)
    psd_theory_level = rate_exact_hz / N_neurons
    low_freq_limit = rate_exact_hz*(CV**2)/N_neurons

    cv_list = []
    for neuron_spikes in spike_times:
        if len(neuron_spikes) > 2: # Need at least 2 spikes to have an interval
            isi = np.diff(neuron_spikes) # Inter-spike intervals
            mean_isi = np.mean(isi)
            std_isi = np.std(isi)
            if mean_isi > 0:
                cv_list.append(std_isi / mean_isi)

    cv_micro = np.mean(cv_list)
    low_freq_limit_empirical = np.mean(activity_net)*(cv_micro**2)/N_neurons

    # Calculate Full Theoretical Curve
    # Use FPE frequencies as the axis
    mask = (freq_fpe > 0.5) & (freq_fpe < 2500)
    theory_freqs = freq_fpe[mask]
    theory_curve = get_theoretical_curve(theory_freqs, mu, D, tau, V_th, V_reset, rate_exact_hz, N_neurons)

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    
    if activity_net is not np.nan:
        plt.loglog(freq_net, psd_net, color='gray', alpha=0.5, label='Microscopic')
    
    plt.loglog(freq_fpe, psd_fpe, color='red', linestyle='--', linewidth=2, label='FPE Solver (Mesoscopic)')


    if not recurrence:
        plt.loglog(theory_freqs, theory_curve, color='black', linewidth=1.5, label='Exact')
        plt.axhline(psd_theory_level, color='green', linestyle=':', linewidth=2, label=f'Poisson')
        plt.axhline(low_freq_limit, color='blue', linestyle=':', linewidth=2, label=f'Low frequency limit')
    #plt.axhline(low_freq_limit_empirical, color='orange', linestyle=':', linewidth=2, label=f'Low frequency limit (empirical)')
    else:
        pass
        # insert logic for mean-field theory prediction here
        
    plt.title(fr"PSD Validation Check (N={N_neurons}, $\mu$={mu}, D={D}, $\tau$={tau})")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density")
    plt.xlim(1, 10000)
    
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    
    if not os.path.exists("plots"): os.makedirs("plots")
    plt.savefig("plots/validation_psd.png", dpi=150)
    print(f"Theory Level: {psd_theory_level:.5f}")
    print("\n✓ Validation Plot saved to plots/validation_psd.png")
    # plt.show()

if __name__ == "__main__":
    # Run
    activity_net, spike_times = run_network(N_neurons, mu, D, tau, V_th, V_reset, dt_net, w, recurrence=recurrence) if N_neurons <= 20000 else np.nan
    
    calculate_psd(activity_net, activity_fpe, spike_times, method="bartlett")