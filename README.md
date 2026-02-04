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
├── generate_lut.py        # Lookup table generator for (μ, D) → rate/CV mapping
├── Makefile               # Orchestration for build, simulation, and plotting
├── src/
│   ├── stochastic_fpe.c   # Main C solver (Finite Volume / Operator Splitting)
│   └── params.h           # Auto-generated C header (do not edit manually)
├── validation/
│   ├── check_rate.py      # Validation script 1: Mean Rate vs Siegert
│   ├── check_psd.py       # Validation script 2: PSD vs Microscopic sim
│   └── check_neural_mass.py  # Validation script 3: Mass conservation check
├── analysis/
│   ├── plot_results.py    # General plotting (timesteps, density evolution)
│   ├── lif.py             # Core theoretical library (Siegert formula, CV)
│   ├── utils.py           # Signal processing utilities (periodogram, binning)
│   └── lif_susceptibility_psd.py  # Theoretical PSD via parabolic cylinder functions
├── data/                  # Binary output from solver
└── plots/                 # Generated output figures
```

## Usage

The project uses a `Makefile` to handle parameter generation, compilation, and execution.

**Prerequisites:**
* GCC or Clang compiler
* Python 3.x (`numpy`, `scipy`, `matplotlib`, `mpmath`, `tqdm`)

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

### Additional Targets

```bash
make run_fpe                  # Run solver only (skip validation)
make run_fpe MU=1.5 D=0.05    # Override parameters via CLI
make mass                     # Check neural mass conservation
make clean                    # Remove build artifacts
```

## Configuration

Physics and simulation parameters are defined in **`config.py`**. This ensures consistency between the C solver and Python analysis scripts.

```python
PARAMS = {
    "PARAM_N_NEURONS": 500,     # Population size
    "PARAM_T_MAX":     10000.0, # Simulation time (ms)
    "PARAM_TAU":       10.0,    # Membrane time constant (ms)
    "PARAM_MU":        1.2,     # Input current (drift)
    "PARAM_D":         0.01,    # Noise intensity (diffusion)
    "PARAM_V_TH":      1.0,     # Threshold voltage
    "PARAM_V_RESET":   0.0,     # Reset voltage
    "PARAM_DT_NET":    0.02,    # Network timestep (ms)
    "PARAM_GRID_N":    400,     # Spatial grid resolution
    # Recurrent connectivity
    "PARAM_W":         0.0,     # Recurrent weight strength
    "PARAM_CONNECTIVITY": 1.0,  # Connection probability
    "PARAM_DELAY":     0.0,     # Synaptic delay (ms)
}
```

## Recurrent Connectivity

The solver supports recurrent network dynamics with configurable coupling:

* **All-to-all connectivity** (`PARAM_CONNECTIVITY=1.0`): Standard mean-field coupling where each neuron receives input from all others.
* **Sparse random connectivity** (`PARAM_CONNECTIVITY<1.0`): Random network with connection probability $p$. Weights are renormalized to maintain effective coupling strength.
* **Synaptic delays** (`PARAM_DELAY`): Configurable transmission delay between neurons.

The effective input current becomes time-dependent: $\mu_{eff}(t) = \mu + w \cdot r(t-d)$, where $r$ is the population firing rate and $d$ is the synaptic delay.

## Output Format

The solver produces binary output files in `data/`:

| File | Contents |
|------|----------|
| `activity.bin` | Time series of `[rate(t), mass(t)]` pairs (float64) |
| `density.bin` | Snapshots of probability density $p(v,t)$ on the voltage grid |

Python scripts in `analysis/` and `validation/` read these files directly using `numpy.fromfile()`.