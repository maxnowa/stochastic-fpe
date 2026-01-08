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

# Default Target
all: clean plot validate

# 1. Generate Parameters
# Logic: Run config.py. If it generates params.h in root, move it to src/
$(SRC_DIR)/params.h: config.py
	python3 config.py
	@[ -f params.h ] && mv params.h $(SRC_DIR)/params.h || true

# 2. Compile C Code
# Links both stochastic_fpe.c and nrutil.c
$(TARGET_BIN): $(SRC_DIR)/stochastic_fpe.c $(SRC_DIR)/params.h
	mkdir -p $(BIN_DIR)
	$(CC) $(SRC_DIR)/stochastic_fpe.c -o $(TARGET_BIN) $(CFLAGS)

# 3. Run FPE Simulation
run_fpe: $(TARGET_BIN)
	mkdir -p $(DATA_DIR)
	./$(TARGET_BIN)

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

# Cleanup
clean:
	rm -f $(TARGET_BIN) $(SRC_DIR)/params.h
	rm -rf $(DATA_DIR) $(PLOT_DIR) $(BIN_DIR)