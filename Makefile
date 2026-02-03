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

# Files
TARGET_BIN = $(BIN_DIR)/sfpe_solver
PARAMS_H = $(SRC_DIR)/params.h
LUT_FILE = $(SRC_DIR)/lambda_table.dat

# --- Argument Parsing for config.py ---
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

# 1. Generate Lookup Table (Physics Landscape)
# Only runs if the file doesn't exist or generate_lut.py changes
$(LUT_FILE): generate_lut.py
	python3 generate_lut.py

# 2. Generate Parameters (Simulation Config)
# Always runs config.py to update params.h with CLI args
$(PARAMS_H): config.py
	python3 config.py $(PY_FLAGS)

# 3. Compile C Code
# Depends on BOTH params.h and the LUT file existing
$(TARGET_BIN): $(SRC_DIR)/stochastic_fpe.c $(PARAMS_H) $(LUT_FILE)
	mkdir -p $(BIN_DIR)
	$(CC) $(SRC_DIR)/stochastic_fpe.c -o $(TARGET_BIN) $(CFLAGS)

# 4. Run FPE Simulation
run_fpe: $(TARGET_BIN)
	mkdir -p $(DATA_DIR)
	caffeinate -i ./$(TARGET_BIN)

# 5. Plot Results
plot: run_fpe
	mkdir -p $(PLOT_DIR)
	python3 -m analysis.plot_results

# 6. Run Validation
validate: run_fpe
	mkdir -p $(PLOT_DIR)
	@echo "--- Checking Stationary Rate (N -> inf) ---"
	python3 -m validation.check_rate
	@echo "--- Checking Power Spectrum (N < inf) ---"
	python3 -m validation.check_psd
    
# Shortcuts
rate: run_fpe
	mkdir -p $(PLOT_DIR)
	@echo "--- Checking Stationary Rate (N -> inf) ---"
	python3 -m validation.check_rate

psd: run_fpe
	mkdir -p $(PLOT_DIR)
	@echo "--- Checking Power Spectrum (N < inf) ---"
	python3 -m validation.check_psd

mass: clean run_fpe
	mkdir -p $(PLOT_DIR)
	@echo "--- Checking Neural Mass (T -> inf) ---"
	python3 -m validation.check_neural_mass

# Explicit target to force table regeneration
lut:
	python3 generate_lut.py

# Cleanup
clean:
	rm -f $(TARGET_BIN) $(PARAMS_H)
	rm -rf $(DATA_DIR) $(PLOT_DIR) $(BIN_DIR)


clean_all: clean
	rm -f $(LUT_FILE)