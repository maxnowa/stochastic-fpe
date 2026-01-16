# Compiler settings
CC = gcc
# -I$(SRC_DIR) tells the compiler to look for .h files in src/
CFLAGS = -O3 -Wall -lm -march=native -flto -I$(SRC_DIR)

# Folders
DATA_DIR = data
PLOT_DIR = plots
SRC_DIR = src
BIN_DIR = bin
ANALYSIS_DIR = analysis
VALIDATION_DIR = validation

# Output Binary
TARGET_BIN = $(BIN_DIR)/sfpe_solver

# --- Argument Parsing for config.py ---
# Define a variable to hold python flags
# Usage: make run_fpe METHOD=0 MU=1.5
PY_FLAGS = 

ifdef MU
    PY_FLAGS += --PARAM_MU $(MU)
endif
ifdef D
    PY_FLAGS += --PARAM_D $(D)
endif
ifdef METHOD
    PY_FLAGS += --PARAM_METHOD $(METHOD)
endif
ifdef T_MAX
    PY_FLAGS += --PARAM_T_MAX $(T_MAX)
endif

# Default Target
all: clean plot validate

# 1. Generate Parameters
# Logic: Run config.py. If it generates params.h in root, move to src/
$(SRC_DIR)/params.h: config.py
	python3 config.py $(PY_FLAGS)
	@[ -f params.h ] && mv params.h $(SRC_DIR)/params.h || true

# 2. Compile C Code
$(TARGET_BIN): $(SRC_DIR)/stochastic_fpe.c $(SRC_DIR)/params.h
	mkdir -p $(BIN_DIR)
	$(CC) $(SRC_DIR)/stochastic_fpe.c -o $(TARGET_BIN) $(CFLAGS)

# 3. Run FPE Simulation
run_fpe: $(TARGET_BIN)
	mkdir -p $(DATA_DIR)
	caffeinate -i ./$(TARGET_BIN)

# 4. Plot Results
# Using '-m' allows python to see the root directory (for config imports)
plot: run_fpe
	mkdir -p $(PLOT_DIR)
	python3 -m analysis.plot_results

# 5. Run Validation
# Runs both the Stationary (Mean) check and the PSD (Noise) check
validate: run_fpe
	mkdir -p $(PLOT_DIR)
	@echo "--- Checking Stationary Rate (N -> inf) ---"
	python3 -m validation.check_rate
	@echo "--- Checking Power Spectrum (N < inf) ---"
	python3 -m validation.check_psd
    
# run just rate check (for N -> inf)
rate: run_fpe
	mkdir -p $(PLOT_DIR)
	@echo "--- Checking Stationary Rate (N -> inf) ---"
	python3 -m validation.check_rate

# run just psd comparison (N < inf)
psd: run_fpe
	mkdir -p $(PLOT_DIR)
	@echo "--- Checking Power Spectrum (N < inf) ---"
	python3 -m validation.check_psd

# Check neural mass evolution
mass: clean run_fpe
	mkdir -p $(PLOT_DIR)
	@echo "--- Checking Neural Mass (T -> inf) ---"
	python3 -m validation.check_neural_mass


# Cleanup
clean:
	rm -f $(TARGET_BIN) $(SRC_DIR)/params.h
	rm -rf $(DATA_DIR) $(PLOT_DIR) $(BIN_DIR)