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

def plot_stationary_check(fpe_file="data/activity_data.csv"):
    
    # --- 1. Load Parameters ---
    if PARAMS:
        mu = PARAMS["PARAM_MU"]
        D = PARAMS["PARAM_D"]
        tau = PARAMS["PARAM_TAU"]
        V_th = PARAMS["PARAM_V_TH"]
        V_reset = PARAMS["PARAM_V_RESET"]
    else:
        mu, D, tau = 1.2, 0.01, 10.0
        V_th, V_reset = 1.0, 0.0

    # --- 2. Load Simulation Data ---
    if not os.path.exists(fpe_file):
        print(f"Error: {fpe_file} not found.")
        return

    data = np.genfromtxt(fpe_file, delimiter=",", skip_header=1)
    if np.isnan(data[0, -1]): data = data[:, :-1]
        
    time = data[:, 0]
    rate_sim = data[:, 1] * 1000.0 # Convert kHz -> Hz

    # --- 3. Calculate Theories ---
    # Exact (Siegert) from function lif.py 
    mu_dim = (mu - V_reset) / (V_th - V_reset)
    sigma_dim = np.sqrt(D) / (V_th - V_reset)
    rate_dimensionless = rate_whitenoise_benji(mu_dim, sigma_dim)
    rate_siegert = rate_dimensionless * (1000.0 / tau)
    
    # Approximation (Lindner script)
    rate_lindner = steady_state_rate(mu, D, tau, V_th, V_reset)
    
    # Simulation Mean (ignoring first 30% transient)
    start_idx = int(len(time) * 0.3)
    rate_sim_mean = np.mean(rate_sim[start_idx:])
    

    plt.figure(figsize=(10, 7))    
    plt.plot(time, rate_sim, 
             label=f'FPE Simulation (Mean: {rate_sim_mean:.2f} Hz)', 
             color='tab:orange', linewidth=2, alpha=0.6)
    
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
    plt.text(time[0], max(rate_sim)*0.9, info, bbox=dict(facecolor='white', alpha=0.9))

    plt.ylim(0, max(rate_sim_mean, rate_siegert) * 1.5)
    
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