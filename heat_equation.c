#include <stdio.h>
#include <math.h>
#include <stdlib.h>

void initialize_grid(double *p, int N);
void calculate_drift_diffusion(double *p, double *p_new, double c, double dx, int N, double D, double dt);
void swap_pointers(double **p, double **p_new);
void diffusion_crank_nicholson(double *p, int N, double D, double dx, double dt, 
                               double *x, double *a, double *b, double *c, double *scratch);
// Note: "restrict" keyword requires C99 standard (gcc -std=c99)
void thomas(const int X, double *x, const double *a, const double *b, const double *c, double *scratch);
// Prototype
double slope_limiter(double *p, int i);
void drift(double *p, double *p_new, double *flux, int N, double dx, double dt, double velocity);

int main()
{
    double velocity = 0.1;
    int N = 200;
    double L = 2.0;
    double dx = L/N;
    double D = 0.05;
    double T = 3.0;

    double dt = (0.8 * dx * dx) / (2*D + velocity * dx);
    int steps = (int)(T/dt);
    
    printf("dt = %g\n", dt);
    printf("Allocating grid of size %d. Steps: %d\n", N, steps);

    double *p = (double*)malloc(N * sizeof(double));
    double *p_new = (double*)malloc(N * sizeof(double));
    double *workspace = (double*)malloc(N * sizeof(double));
    double *a_diag = (double*)malloc(N * sizeof(double));
    double *b_diag = (double*)malloc(N * sizeof(double));
    double *c_diag = (double*)malloc(N * sizeof(double));
    double *x_rhs = (double*)malloc(N * sizeof(double));

    // if (p==NULL) {
    //     printf("Memory allocation failed.");
    //     return 1;
    // }

    // initialize the grid
    initialize_grid(p, N);

    FILE *f = fopen("diffusion_drift_data.csv", "w");


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
        diffusion_crank_nicholson(p, N, D, dx, dt, x_rhs, a_diag, b_diag, c_diag, workspace);
        
        // Step 2 - Drift using 2nd order uwpind differencing
        drift(p, p_new, workspace, N, dx, dt, velocity);

        // swap pointers 
        swap_pointers(&p, &p_new);
        
    }

    fclose(f);
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



void initialize_grid(double *p, int N) {
    for (int i=0; i < N; i++) {
        if (i > 0.45 * N && i < 0.55*N) {
            p[i] = 1.0;
        } else {
            p[i] = 0.0;
        }       
    }
}

// diffusion calculated implicitly using crank nicholson
void diffusion_crank_nicholson(double *p, int N, double D, double dx, double dt, double *x, double *a, double *b, double *c, double *scratch) {
    int X = N -2;
    double alpha = (D * dt) / (2.0 * dx * dx);

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
void drift(double *p, double *p_new, double *flux, int N, double dx, double dt, double velocity) {
    double courant = (velocity * dt) / dx;
    for (int i = 1; i < N-1; i++) {
        flux[i] = velocity * (p[i] + 0.5 * (1-courant) * slope_limiter(p, i));
    }
    flux[0] = 0.0;
    flux[N-1] = 0.0;

    for (int i = 1; i < N-1; i++) {
        p_new[i] = p[i] - (dt/dx) * (flux[i] - flux[i-1]);
    }
    p_new[0] = 0.0;
    p_new[N-1] = 0.0;
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