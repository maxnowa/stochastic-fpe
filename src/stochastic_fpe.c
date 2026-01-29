#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "params.h"
#include <time.h>
#include <stdint.h>

void initialize_system(double *p, double *v_grid, int N, double dx, double V_min, double mu, double tau);
void swap_pointers(double **p, double **p_new);
void diffusion_crank_nicholson(double *p, int N, double D, double tau, double dx, double dt, double *x, double *a, double *b, double *c, double *scratch);
void thomas(const int X, double *x, const double *a, const double *b, const double *c, double *scratch);
double slope_limiter(double *p, int i);
void drift(double *p, double *p_new, double *flux, int N, double dx, double dt, double *v_grid, double v_exit);
double set_dt(double *v_grid, int N, double sf, double dx, double D);
double get_upwind_flux(double *p, int i, double v, double dt, double dx, int N);
void update_v_grid(double *v_grid, int N, double dx, double V_min, double mu, double tau);

// Generates standard normal random number N(0,1)
double randn()
{
    double u = (double)rand() / RAND_MAX;
    double v = (double)rand() / RAND_MAX;
    return sqrt(-2.0 * log(u)) * cos(2.0 * M_PI * v);
}

int main()
{
    // seed_rng((uint64_t)time(NULL));

    // Sim parameters
    // for recurrent connections
    int recurrence_mode = PARAM_RECURRENCE;
    int delay = PARAM_DELAY;

    int N = PARAM_GRID_N;
    double D = PARAM_D;
    double T = PARAM_T_MAX;
    double V_min = -0.8;
    double V_max = 1.0;
    double dx = (V_max - V_min) / N;
    double stability_factor = 4.0;
    // since we have a constant mu and D, R0 and CV are constant
    // FIX for the w!=0 case, because mu changes at each step -> this requires re-calculation at each time step, which could make the solver unstable
    // what could be done is that instead of using the instantaneous rate, we low pass filter it or use a running average of lambda/rate
    double lambda = PARAM_R0 / PARAM_CV;
    // UNSAFE initial value
    double dt = 1e-5;
    double w = PARAM_W;
    double connectivity_p = PARAM_CONNECTIVITY;
    double sigma_w_squared = 0.0;
    // Calculate Weight Variance 
    if (connectivity_p < 1.0 && connectivity_p > 0.0) {
        sigma_w_squared = (w * w) * (1.0 - connectivity_p) / connectivity_p;
    }

    // allocate memory
    double *p = (double *)malloc(N * sizeof(double));
    double *p_new = (double *)malloc(N * sizeof(double));
    double *workspace = (double *)malloc(N * sizeof(double));
    double *a_diag = (double *)malloc(N * sizeof(double));
    double *b_diag = (double *)malloc(N * sizeof(double));
    double *c_diag = (double *)malloc(N * sizeof(double));
    double *x_rhs = (double *)malloc(N * sizeof(double));
    double *v_grid = (double *)malloc(N * sizeof(double));

    // Neuron parameters
    double V_rest = 0.0;
    double V_th = PARAM_V_TH;
    double mu = PARAM_MU;
    double tau = PARAM_TAU;
    double N_neurons = (double)PARAM_N_NEURONS;

    // initialize the grid
    initialize_system(p, v_grid, N, dx, V_min, mu, tau);

    if (recurrence_mode == 0)
    {
        dt = set_dt(v_grid, N, stability_factor, dx, D);
    }
    else
    {
        double max_anticipated_rate = (w > 0) ? 600.0 : 300.0;
        double mu_extreme = mu + (w * max_anticipated_rate);
        
        double v_speed_base = fmax(fabs((mu - V_min)/tau), fabs((mu - V_max)/tau));
        double v_speed_extreme = fmax(fabs((mu_extreme - V_min)/tau), fabs((mu_extreme - V_max)/tau));

        double v_max_worst = fmax(v_speed_base, v_speed_extreme);

        if (v_max_worst == 0)
        {
            dt = 0.1 * (dx * dx) / (2.0 * D);
        }
        else
        {
            dt = (1.0 * dx) / (v_max_worst);
        }
    }
    int buffer_size = (int)(delay / dt);
    // this stores the past rates
    double *history_A_t = (double *)calloc((buffer_size + 1), sizeof(double));
    double *history_r_t = (double *)calloc((buffer_size + 1), sizeof(double));

    double initial_rate = 0.0;
    for (int i = 0; i < buffer_size; i++)
    {
        history_r_t[i] = initial_rate;
    }
    int write_idx = 0;

    system("mkdir -p data");

    int steps = (int)(T / dt);
    printf("Dynamic step size of %g determined\n", dt);
    printf("Allocating grid of size %d. Steps: %d\n", N, steps);

    FILE *fp_density = fopen("data/density.bin", "wb");
    FILE *fp_activity = fopen("data/activity.bin", "wb");

    // initialize buffer to block write the data
    float p_buffer[N];
    // --- Setup Timer ---
    int total_steps = (int)(PARAM_T_MAX / dt);
    int report_interval = total_steps / 10000;
    if (report_interval == 0)
        report_interval = 1;
    int save_interval = 1000;
    clock_t start_time = clock();

    for (int t = 0; t < steps; t++)
    {
        double A_t_delayed = history_A_t[write_idx];
        double r_t_delayed = history_r_t[write_idx];
        // calculate effective drift
        double mu_eff = mu; 
        if (recurrence_mode == 1) {
            mu_eff += (w * A_t_delayed);
        }
        // calculate effective diffusion for p<1
        double D_eff = D;
        if (recurrence_mode == 1 && connectivity_p < 1.0) {
            // D_w(t) = (sigma_w^2 / 2N) * r(t-d)
            D_eff += (sigma_w_squared / (2.0 * N_neurons)) * r_t_delayed;
        }

        update_v_grid(v_grid, N, dx, V_min, mu_eff, tau);
        double v_exit = (mu_eff - V_th) / tau;

        // --- Runtime CFL Safety Check ---
        double current_max_v = 0.0;
        for (int i = 0; i < N; i++) {
            if (fabs(v_grid[i]) > current_max_v) {
                current_max_v = fabs(v_grid[i]);
            }
        }
        
        // Check also the exit velocity as it might be the highest
        if (fabs(v_exit) > current_max_v) {
             current_max_v = fabs(v_exit);
        }

        // CFL Condition: (v * dt) / dx must be <= 1.0 (ideally <= 0.8 for safety)
        if ((current_max_v * dt) / dx > 1.0) {
            printf("\nCRITICAL ERROR at Step %d (t=%.4fs):\n", t, t*dt);
            printf("  Stability violation detected!\n");
            printf("  Max Velocity: %.2f mV/ms\n", current_max_v);
            printf("  Required dt: < %.2e s\n", dx / current_max_v);
            printf("  Current dt:  %.2e s\n", dt);
            printf("  Simulation stopped to prevent numerical explosion.\n");
            break; // Breaks out of the time loop
        }

        double mass_old = 0.0;
        for (int i = 0; i < N; i++)
            mass_old += p[i];

        // Operator splitting approach
        // -- Step 1 - Diffusion using crank nicholson --
        diffusion_crank_nicholson(p, N, D_eff, tau, dx, dt, x_rhs, a_diag, b_diag, c_diag, workspace);

        // we measure the diffusion rate directly
        double mass_diff = 0.0;
        for (int i = 0; i < N; i++)
            mass_diff += p[i];
        double J_diff = (mass_old - mass_diff) * (dx / dt);

        // -- Step 2 - Drift using 2nd order uwpind differencing --
        drift(p, p_new, workspace, N, dx, dt, v_grid, v_exit);

        // measure all mass that was lost
        double mass_remaining = 0.0;
        for (int i = 0; i < N; i++)
            mass_remaining += p_new[i];
        double J_mass_lost = (mass_old - mass_remaining) * dx / dt;

        // -- Step 3 - Adding probability mass back --
        // summing over p to get firing rate (mass)
        double current_mass = 0.0;
        for (int i = 0; i < N; i++)
        {
            current_mass += p_new[i];
        }
        current_mass *= dx;

        // Calculate outflux
        double J_out = 0.0;
        switch (PARAM_FLUX_METHOD)
        {

        case 0:
        {
            // J_out = v_grid[N-1]*p_new[N-2] + (D / tau) * (p_new[N - 2] / dx);
            // Drift flux at the boundary, use stored in workspace array
            double J_drift = workspace[N - 1];
            // Diffusive Flux
            // double J_diff = (D / tau) * (p[N-1] / dx);
            // Total Outflux - this method is now equivalent to case 1
            J_out = J_drift + J_diff;
            break;
        }
        case 1:
            // after adjusting boundary handling works for sub and supra
            J_out = J_mass_lost;
            break;
        }
        if (J_out < 0)
            J_out = 0.0;

            
        double r_t = 0.0;
        switch (PARAM_METHOD)
        {

        // APPROACH 1 - Unstable
        case 0:
            r_t = J_out;
            break;

        // APPROACH 2 - correction term (Tilo)
        // we need to use lamda/tau to account for the timescale
        case 1:
            // lambda = r_t_delayed;
            r_t = J_out + (lambda / tau) * (1 - current_mass);
            break;

        // APPROACH 3 - assuming poisson spike statistics, r(t) = J_out / mass, because lambda = r(t) + global renorm
        case 2:
            if (current_mass > 1e-9)
            {
                r_t = J_out / current_mass;
            }
            else
            {
                r_t = J_out; // Fallback if mass is empty
            }
            break;

        default:
            printf("Error: Unknown Rate Method %d\n", PARAM_METHOD);
            exit(1);
        }
        // only positive rates
        if (r_t < 0.0)
            r_t = 0.0;

        double xi_t = randn() / sqrt(dt);
        // -- 4. Calculate Stochastic Rate A(t) --
        // take abs if rate is negative briefly
        double A_t = r_t + sqrt(fabs(r_t) / N_neurons) * xi_t;

        // -- resetting --
        int reset_idx = (int)((V_rest - V_min) / dx);
        // dp/dt = A(t) here because the absorbing boundary takes the mass out of the system
        p_new[reset_idx] += (A_t * dt) / dx;

        if (PARAM_METHOD == 2)
        { // this is now global renormalization
            double new_mass = 0.0;
            for (int i = 0; i < N; i++)
            {
                new_mass += p_new[i] * dx;
            }

            // Normalize to exactly 1.0
            if (new_mass > 1e-9)
            {
                double correction_factor = 1.0 / new_mass;
                for (int i = 0; i < N; i++)
                {
                    p_new[i] *= correction_factor;
                }
            }
        }

        // swap pointers
        swap_pointers(&p, &p_new);

        // record density at
        if (t % save_interval == 0)
        {
            for (int i = 0; i < N; i++)
            {
                p_buffer[i] = (float)p[i];
            }
            fwrite(p_buffer, sizeof(float), N, fp_density);
        }

        // save activity data at every timestep
        float act_data[2];
        act_data[0] = (float)A_t;
        act_data[1] = (float)current_mass;

        // Write 2 floats at once
        fwrite(act_data, sizeof(float), 2, fp_activity);
        // --- INSERT 2: PRINT PROGRESS ---
        if (t % report_interval == 0)
        {
            double progress = (double)t / total_steps;

            // Calculate ETA
            clock_t now = clock();
            double elapsed = (double)(now - start_time) / CLOCKS_PER_SEC;
            double eta = 0.0;
            if (progress > 0)
            {
                eta = elapsed * (1.0 / progress - 1.0);
            }

            printf("\r[%.2f%%] Step: %d | ETA: %.1fs",
                   progress * 100.0, t, eta);
            fflush(stdout);
        }
        history_A_t[write_idx] = A_t;
        history_r_t[write_idx] = r_t;

        write_idx++;
        if (write_idx >= buffer_size)
        {
            write_idx = 0;
        }
    }

    fclose(fp_density);
    fclose(fp_activity);
    // Free all memory
    free(p);
    free(p_new);
    free(workspace);
    free(a_diag);
    free(b_diag);
    free(c_diag);
    free(x_rhs);

    free(v_grid);
    free(history_A_t);
    free(history_r_t);

    printf("\r[100%%] Simulation Complete. Total time: %.1fs\n",
           (double)(clock() - start_time) / CLOCKS_PER_SEC);
    return 0;
}

void initialize_system(double *p, double *v_grid, int N, double dx, double V_min, double mu, double tau)
{
    // initialize the state
    for (int i = 0; i < N; i++)
    {
        double v_real = V_min + i * dx;
        v_grid[i] = (mu - v_real) / tau;
        p[i] = 0.0;
    }
    // intialize system as delta spike
    int index_0 = (int)((0.0 - V_min) / dx);
    // Safety check: ensure index is within bounds
    if (index_0 >= 0 && index_0 < N)
    {
        p[index_0] = 1.0 / dx; // Normalize so Area = 1.0
    }
}

void update_v_grid(double *v_grid, int N, double dx, double V_min, double mu, double tau)
{
    for (int i = 0; i < N; i++)
    {
        double v_real = V_min + i * dx;
        v_grid[i] = (mu - v_real) / tau;
    }
}

// diffusion calculated implicitly using crank nicholson
void diffusion_crank_nicholson(double *p, int N, double D, double tau, double dx, double dt, double *x, double *a, double *b, double *c, double *scratch)
{
    int X = N - 1;
    double alpha = ((D / tau) * dt) / (2.0 * dx * dx);

    for (int i = 0; i < X; i++)
    {
        a[i] = -alpha;
        b[i] = 1.0 + 2.0 * alpha;
        c[i] = -alpha;
    }

    for (int k = 0; k < X; k++)
    {
        int i = k + 1;
        double explicit_val = alpha * p[i - 1] + (1.0 - 2.0 * alpha) * p[i] + alpha * p[i + 1];
        x[k] = explicit_val;
    }

    // solve using thomas algorithm
    thomas(X, x, a, b, c, scratch);

    for (int k = 0; k < X; k++)
    {
        p[k + 1] = x[k];
    }
    p[0] = 0.0;
    // p[N - 1] = 0.0;
}

void thomas(const int X, double *x, const double *a, const double *b, const double *c, double *scratch)
{

    scratch[0] = c[0] / b[0];
    x[0] = x[0] / b[0];

    for (int ix = 1; ix < X; ix++)
    {
        if (ix < X - 1)
        {
            scratch[ix] = c[ix] / (b[ix] - a[ix] * scratch[ix - 1]);
        }
        x[ix] = (x[ix] - a[ix] * x[ix - 1]) / (b[ix] - a[ix] * scratch[ix - 1]);
    }

    for (int ix = X - 2; ix >= 0; ix--)
        x[ix] -= scratch[ix] * x[ix + 1];
}


// drift caclulated using second-order TVD upwind differencing scheme - we are using Lax-Wendroff correction and van Leer flux limiter
void drift(double *p, double *p_new, double *flux, int N, double dx, double dt, double *v_grid, double v_exit)
{
    // Interior flux
    for (int i = 0; i < N - 1; i++)
    {
        // we want the avergae velocity at the interface
        double v_interface = 0.5 * (v_grid[i] + v_grid[i + 1]);
        if (i == 0)
            flux[i] = 0.0;
        else
            flux[i] = get_upwind_flux(p, i, v_interface, dt, dx, N);
    }

    // flux at exactly the boundary
    double J_drift_exit = (v_exit > 0) ? (v_exit * p[N - 1]) : 0.0;

    // save for use in main
    flux[N - 1] = J_drift_exit;

    // update grid
    for (int i = 1; i < N; i++)
    {
        double flux_in = flux[i - 1];
        double flux_out = (i == N - 1) ? J_drift_exit : flux[i];
        // change in probability mass
        p_new[i] = p[i] - (dt / dx) * (flux_out - flux_in);
    }

    // Safety
    p_new[0] = 0.0;
}
//
// void drift(double *p, double *p_new, double *flux, int N, double dx, double dt, double *v_grid)
// {

//     // CURRENT: TODO need to safeguard against out of bounds access since we loop to n-1
//     for (int i = 1; i < N - 1; i++)
//     {
//
//         double v_interface = 0.5 * (v_grid[i] + v_grid[i + 1]);
//         flux[i] = get_upwind_flux(p, i, v_interface, dt, dx, N);
//     }

//     flux[0] = 0.0;
//     //flux[N - 2] = 0.0;
//     //flux[N - 1] = 0.0;

//     for (int i = 1; i < N - 1; i++)
//     {
//
//         p_new[i] = p[i] - (dt / dx) * (flux[i] - flux[i - 1]);
//     }
//     p_new[0] = 0.0;
//     p_new[N - 1] = 0.0;
// }

double get_upwind_flux(double *p, int i, double v, double dt, double dx, int N)
{
    // Lax-Wendroff for temporal correction - we correct for the fraction of a cell that the probability moves in one timestep
    // this make it second order in time
    double courant = (v * dt) / dx;

    if (v >= 0)
    {
        // drift is going from left to right (we only change indexing direction)
        // needs p[i] and p[i-1] for slope
        // drop to first order to avoid out of bounds access
        if (i == 0)
        {
            return v * p[i];
        }

        double limited_slope = slope_limiter(p, i);
        return v * (p[i] + 0.5 * (1.0 - courant) * limited_slope);
    }
    else
    {
        // drift is going from right to left
        // needs p[i+1] and p[i+2] for slope
        // drop to first order to avoid out of bounds access
        if (i >= N - 2)
        {
            return v * p[i + 1]; // 1st Order fallback (Safe)
        }

        double limited_slope = slope_limiter(p, i + 1);
        return v * (p[i + 1] - 0.5 * (1.0 + courant) * limited_slope);
    }
}

// van Leer flux limiter
double slope_limiter(double *p, int i)
{
    // we look at the slope of the density at the left and right cell
    double slope_left = p[i] - p[i - 1];
    double slope_right = p[i + 1] - p[i];
    // go back to first order if the slope is not the same (TVD)
    if (slope_left * slope_right <= 0.0)
    {
        return 0.0;
    }
    // return value based on slope
    // this make it second order in space because we take the shape into account
    else
    {   
        // this implicitly calculates slope*phi(r)
        double limited_slope = (2 * slope_left * slope_right) / (slope_left + slope_right);
        return limited_slope;
    }
}

void swap_pointers(double **p, double **p_new)
{
    double *temp = *p;
    *p = *p_new;
    *p_new = temp;
}

double set_dt(double *v_grid, int N, double sf, double dx, double D)
{
    double max_v = 0.0;
    double dt = 0.0;
    for (int i = 0; i < N; i++)
    {
        double cells_per_second = fabs(v_grid[i]) / dx;
        if (cells_per_second > max_v)
        {
            max_v = cells_per_second;
        }
    }
    // Calculate dt
    if (max_v == 0.0)
    {
        dt = 0.1 * (dx * dx) / (2.0 * D);
    }
    else
    {
        // This ensures a particle takes at least 'sf' steps to cross one cell
        dt = 1.0 / (max_v * sf);
    }
    return dt;
}
