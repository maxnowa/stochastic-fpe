#include <stdio.h>
#include <math.h>
#include <stdlib.h>


void initialize_system(double *p, double *v_grid, int N, double dx, double V_min, double mu, double tau);
void swap_pointers(double **p, double **p_new);
void diffusion_crank_nicholson(double *p, int N, double D, double tau, double dx, double dt, 
                               double *x, double *a, double *b, double *c, double *scratch);
void thomas(const int X, double *x, const double *a, const double *b, const double *c, double *scratch);
double slope_limiter(double *p, int i);
void drift(double *p, double *p_new, double *flux, int N, double dx, double dt, double *v_grid);
double set_dt(double *v_grid, int N, double sf, double dx, double D);
double get_upwind_flux(double *p, int i, double v, double dt, double dx);

// Generates standard normal random number N(0,1)
double randn() {
    double u = (double)rand() / RAND_MAX;
    double v = (double)rand() / RAND_MAX;
    return sqrt(-2.0 * log(u)) * cos(2.0 * M_PI * v);
}

int main()
{   
    // Sim parameters
    int N = 200;
    double D = 0.01;
    double T = 1000.0;
    double V_min = -1.0;
    double V_max = 1.0;
    double dx = (V_max - V_min) / N;  
    double stability_factor = 10.0;  
    int correction = 1;

    // allocate memory
    double *p = (double*)malloc(N * sizeof(double));
    double *p_new = (double*)malloc(N * sizeof(double));
    double *workspace = (double*)malloc(N * sizeof(double));
    double *a_diag = (double*)malloc(N * sizeof(double));
    double *b_diag = (double*)malloc(N * sizeof(double));
    double *c_diag = (double*)malloc(N * sizeof(double));
    double *x_rhs = (double*)malloc(N * sizeof(double));
    double *v_grid = (double*)malloc(N * sizeof(double));

    // Neuron parameters
    double mu = 1;
    double V_rest = 0.0;
    double tau = 10.0;
    int N_neurons = 500;

    // set dt dynamically
    double dt = set_dt(v_grid, N, stability_factor, dx, D);
    int steps = (int)(T / dt);

    // initialize the grid
    initialize_system(p, v_grid, N, dx, V_min, mu, tau);
    system("mkdir -p data");
    printf("Allocating grid of size %d. Steps: %d\n", N, steps);
    FILE *f = fopen("data/diffusion_drift_data.csv", "w");
    FILE *f_activity = fopen("data/activity_data.csv", "w");
    // Write a header so you know which column is which
    fprintf(f_activity, "time,A_t,mass\n");

    for (int t = 0; t < steps; t++)
    {
        if (t % 100 == 0)
        {
            for (int i = 0; i < N; i++)
            {
                fprintf(f, "%f,", p[i]);
            }
            fprintf(f, "\n"); // New line for new time step
        }

        // Operator splitting approach
        // Step 1 - Diffusion using crank nicholson
        diffusion_crank_nicholson(p, N, D, tau, dx, dt, x_rhs, a_diag, b_diag, c_diag, workspace);
        
        // Step 2 - Drift using 2nd order uwpind differencing
        drift(p, p_new, workspace, N, dx, dt, v_grid);
        
        // Step 3 - Adding probability mass back
        // summing over p to get firing rate (mass)
        double current_mass = 0.0;
        for (int i = 0; i < N; i++) {
            current_mass += p_new[i];
        }
        current_mass *= dx;
        
        // Normalization approach
        //double J_out = 1.0 - current_mass;

        // J_out is flux at threshold
        double J_out = (D/tau) * (p_new[N-2] / dx);

        // correction term - naive correction as 1/tau
        //double lambda = correction ? (1.0 / tau) : 0.0;
        //double r_t = J_out + lambda * (1 - current_mass);
        // assuming poisson spike statistics, r(t) = J_out / mass, because lambda = r(t)
        double r_t = 0.0;
        if ( current_mass > 0.001) {
            r_t = J_out / current_mass;
        }
        
        //if (r_t < 0) r_t = 0.0;

        // noise
        double noise = randn();
        double A_t = r_t + sqrt(r_t / N_neurons) * noise;
        // save to file
        fprintf(f_activity, "%g,%g,%g\n", t * dt, A_t, current_mass);
        // resetting
        int reset_idx = (int)((V_rest - V_min) / dx);
        p_new[reset_idx] += (A_t * dt) / dx;

        // swap pointers 
        swap_pointers(&p, &p_new);
    }

    fclose(f);
    fclose(f_activity);
    printf("Simulation Complete. Data saved to diffusion_drift_data.csv\n");
    // Free all memory
    free(p);
    free(p_new);
    free(workspace);
    free(a_diag);
    free(b_diag);
    free(c_diag);
    free(x_rhs);

    return 0;
}



void initialize_system(double *p, double *v_grid, int N, double dx, double V_min, double mu, double tau) {
    // initialize the state
    for (int i=0; i < N; i++) {
        double v_real = V_min + i *dx;
        v_grid[i] = (mu - v_real) /tau;
        p[i] = 0.0;
    }
    // intialize system as delta spike
    int index_0 = (int)((0.0-V_min) / dx);
    // Safety check: ensure index is within bounds
    if (index_0 >= 0 && index_0 < N) {
        p[index_0] = 1.0 / dx; // Normalize so Area = 1.0
    }
}

// diffusion calculated implicitly using crank nicholson
void diffusion_crank_nicholson(double *p, int N, double D, double tau, double dx, double dt, double *x, double *a, double *b, double *c, double *scratch) {
    int X = N -2;
    double alpha = ((D/tau) * dt) / (2.0 * dx * dx);

    for (int i = 0; i < X; i++) {
        a[i] = -alpha;
        b[i] = 1.0 + 2.0 * alpha;
        c[i] = -alpha;
    }

    for (int k = 0; k < X; k++) {
        int i = k + 1;
        double explicit_val = alpha * p[i-1] + (1.0 - 2.0 * alpha) * p[i] + alpha * p[i+1];
        x[k] = explicit_val;
    }

    // solve using thomas algorithm
    thomas(X, x, a, b, c, scratch);

    for (int k = 0; k < X; k++) {
        p[k+1] = x[k];
    }
    p[0] = 0.0;
    p[N-1] = 0.0;
}

void thomas(const int X, double *x, const double *a, const double *b, const double *c, double *scratch) {
    // Note: removed 'restrict' from args here to match standard C, 
    // but kept logic. Ideally compile with -O2.
    
    scratch[0] = c[0] / b[0];
    x[0] = x[0] / b[0];

    for (int ix = 1; ix < X; ix++) {
        if (ix < X-1){
            scratch[ix] = c[ix] / (b[ix] - a[ix] * scratch[ix - 1]);
        }
        x[ix] = (x[ix] - a[ix] * x[ix - 1]) / (b[ix] - a[ix] * scratch[ix - 1]);
    }

    for (int ix = X - 2; ix >= 0; ix--)
        x[ix] -= scratch[ix] * x[ix + 1];
}

double slope_limiter(double *p, int i) {
    double slope_left = p[i] - p[i-1];
    double slope_right = p[i+1] - p[i];
    if (slope_left * slope_right <= 0.0) {
        return 0.0;
    } else {
        double limit = (2 * slope_left * slope_right) / (slope_left + slope_right);
        return limit;
    }
}

// drift caclulated using 2nd order upwind differencing scheme
void drift(double *p, double *p_new, double *flux, int N, double dx, double dt, double *v_grid) {
    for (int i = 1; i < N-2; i++) {
        double v_interface = 0.5 * (v_grid[i] + v_grid[i + 1]);
        flux[i] = get_upwind_flux(p, i, v_interface, dt, dx);
    }
    flux[0] = 0.0;
    flux[N-2] = 0.0;
    flux[N-1] = 0.0;

    for (int i = 1; i < N-1; i++) {
        p_new[i] = p[i] - (dt/dx) * (flux[i] - flux[i-1]);
    }
    p_new[0] = 0.0;
    p_new[N-1] = 0.0;
}

double get_upwind_flux(double *p, int i, double v, double dt, double dx) {
    double courant = (v * dt) / dx;
    if (v >= 0) {
        double slope = slope_limiter(p, i);
        return v * (p[i] + 0.5 * (1.0 - courant) * slope);
    } else {
        double slope = slope_limiter(p, i+1);
        return v * (p[i+1] - 0.5 * (1.0 + courant) * slope);
    }
}

// void calculate_drift_diffusion(double *p, double *p_new, double c, double dx, int N, double D, double dt) {
//     for (int i = 1; i < N-1; i++) {
//         double flux_drift = -c * (p[i]-p[i-1]) / dx;
//         double flux_diff = D * (p[i+1] - 2*p[i] + p[i-1]) / (dx * dx);
//         p_new[i] = p[i] + dt * (flux_drift + flux_diff);
//     }
// }

void swap_pointers(double **p, double **p_new) {
    double *temp = *p;
    *p = *p_new;
    *p_new = temp;
}

double set_dt(double *v_grid, int N, double sf, double dx, double D) {
    double max_v = 0.0;
    double dt = 0.0;
    for (int i = 0; i < N; i++) {
        double cells_per_second = fabs(v_grid[i]) / dx;
        if (cells_per_second > max_v) {
            max_v = cells_per_second;
        }
    }
    // Calculate dt
    if (max_v == 0.0) {
        dt = 0.1 * (dx * dx) / (2.0 * D); 
    } else {
        // This ensures a particle takes at least 'sf' steps to cross one cell
        dt = 1.0 / (max_v * sf);
    }
    return dt;
}

