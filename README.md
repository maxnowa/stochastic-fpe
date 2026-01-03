# Numerical Solution of the Stochastic Fokker-Planck Equation for Finite-Size LIF Networks

This repository contains a C/Python implementation for solving the **Stochastic Fokker-Planck Equation (SFPE)**, modeling the probability density dynamics of finite populations of Leaky Integrate-and-Fire (LIF) neurons.

The project includes a **validation pipeline** that compares the macroscopic SPDE solution against microscopic Monte Carlo network simulations to verify the correct scaling of finite-size fluctuations.


## Methodology

### Macroscopic Solver (C)
* **Advection (Drift):** Solved using a second-order Upwind Scheme with slope limiters to minimize numerical diffusion.
* **Diffusion:** Implemented via the Crank-Nicolson method for unconditional stability.
* **Stochasticity:** Explicit injection of finite-size noise scaled by the instantaneous firing rate and population size ($1/\sqrt{N}$).

### Microscopic Validation (Python)
* **Ground Truth:** Euler-Maruyama integration of $N$ independent LIF neurons.
* **Analysis:** Computes the Power Spectral Density (PSD) of the population activity using Welch's method.
* **Comparison:** Performs Z-score normalization to compare the frequency content of the SPDE against the discrete network simulation.

## Usage

The project utilizes a `Makefile` to orchestrate parameter generation, compilation, simulation, and validation.

**Prerequisites:**
* GCC or Clang compiler
* Python 3.x (`numpy`, `scipy`, `matplotlib`)

**Execution:**
To run the full pipeline:
```bash
make validate
```

This command performs the following steps:
1.  Executes `config.py` to generate the `params.h` header.
2.  Compiles the C simulation.
3.  Runs the macroscopic SPDE solver.
4.  Runs the microscopic network simulation.
5.  Generates a comparative PSD plot in the `plots/` directory.

## Configuration

Physics and simulation parameters are defined in **`config.py`**. This ensures consistency between the C and Python implementations.

```python
PARAMS = {
    "N_NEURONS": 500,       # Network size
    "T_MAX":     1000.0,    # Duration (ms)
    "MU":        0.8,       # Mean drift input
    "D":         0.05,      # Noise intensity
    "GRID_N":    400        # Spatial grid resolution
}
```
