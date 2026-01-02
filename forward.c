#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "nrutil.h"
#include "nrutil.c"

// --- PHYSICS CONSTANTS ---
#define D 0.05
#define TAU 2.0
#define MU 1.2

// --- FINITE SIZE CORRECTION ---
#define POP_N 500.0
#define LAMBDA 10.0

// --- GRID BOUNDARIES ---
#define N_GRID 200
#define V_MIN -1.0
#define V_TH 1.0
#define V_RESET 0.0

// --- SIMULATION ---
#define T_MAX 100.0
#define SF 10.0      // Stability factor

