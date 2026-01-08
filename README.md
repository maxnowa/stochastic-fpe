# Numerical Solution of the Stochastic Fokker-Planck Equation for Finite-Size LIF Networks

This repository contains a C/Python implementation for solving the **Stochastic Fokker-Planck Equation (SFPE)**. It models the probability density dynamics of finite populations of Leaky Integrate-and-Fire (LIF) neurons, capturing finite-size effects that standard mean-field limits ignore.

The project includes a robust **validation pipeline** that verifies the solver against analytical theory (Siegert relationship) and microscopic Monte Carlo network simulations.

## Methodology

The solver implements a **Time-Splitting Finite Volume Method** to integrate the drift-diffusion equation:

### Macroscopic Solver (C)
* **Drift (Advection):** Solved using a **Flux-Limited Lax-Wendroff scheme** (Second-Order TVD Upwind).
    * Uses a **Van Leer slope limiter** to prevent numerical oscillations (Gibbs phenomenon) near sharp gradients (e.g., at the threshold).
    * Includes the Lax-Wendroff temporal correction for second-order accuracy in time.
* **Diffusion:** Solved using the **Crank-Nicolson method** (semi-implicit) for unconditional numerical stability.
* **Stochasticity:** Implements finite-size fluctuations by injecting white noise scaled by the instantaneous population firing rate ($\sqrt{r(t)/N}$).
* **Boundary Conditions:** Re-injection of probability flux at $V_{reset}$ closes the feedback loop, capturing network resonance.

### Microscopic Validation (Python)
The validation suite performs two distinct checks:
1.  **Stationary Rate Check:** Compares the time-averaged firing rate against the exact analytical solution (**Siegert formula**) to verify drift and diffusion implementations.
2.  **Fluctuation Check (PSD):** Compares the Power Spectral Density of the population activity against microscopic simulations of $N$ independent LIF neurons to verify finite-size noise scaling ($1/\sqrt{N}$).

## Project Structure

```text
├── config.py              # Single source of truth for physics/grid parameters
├── Makefile               # Orchestration for build, simulation, and plotting
├── src/
│   ├── stochastic_fpe.c   # Main C solver (Finite Volume / Operator Splitting)
│   └── params.h           # Auto-generated C header (do not edit manually)
├── validation/
│   ├── check_rate.py      # Validation script 1: Mean Rate vs Siegert
│   └── check_psd.py       # Validation script 2: PSD vs Microscopic sim
├── analysis/
│   └── plot_results.py    # General plotting (timesteps, density evolution)
└── plots/                 # Generated output figures
```

## Usage

The project uses a `Makefile` to handle parameter generation, compilation, and execution.

**Prerequisites:**
* GCC or Clang compiler
* Python 3.x (`numpy`, `scipy`, `matplotlib`)

### Standard Execution
To run the full validation pipeline (Rate Check + PSD Check):

```bash
make all
```

### Workflow Details
The `make validate` command performs the following steps automatically:
1.  **Configure:** Runs `config.py` to generate `src/params.h`.
2.  **Compile:** Builds `bin/sfpe_solver` using standard `O3` optimizations.
3.  **Simulate:** Runs the C solver to generate `data/output.dat`.
4.  **Verify Rate:** Runs `validation.check_rate` to confirm the mean rate matches theory.
5.  **Verify PSD:** Runs `validation.check_psd` (unless skipped) to overlay Macroscopic and Microscopic spectra.

## Configuration

Physics and simulation parameters are defined in **`config.py`**. This ensures consistency between the C solver and Python analysis scripts.

```python
PARAMS = {
    "PARAM_N_NEURONS": 500,     # neuron number
    "PARAM_T_MAX":     10000.0, # simulation time (ms)
    "PARAM_TAU":       10.0,    # membrane time constant (ms)
    "PARAM_MU":        1.2,     # drift
    "PARAM_D":         0.01,    # noise strength
    "PARAM_V_TH":      1.0,     # threshold
    "PARAM_V_RESET":   0.0,     # reset potential
    "PARAM_DT_NET":    0.02,    # network time step size
    "PARAM_GRID_N":    400      # FPE grid
}