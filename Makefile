# Compiler settings
CC = gcc
CFLAGS = -O3 -Wall -lm

# Folders
DATA_DIR = data
PLOT_DIR = plots

# Targets
all: validate

# 1. Generate Parameters (Python -> C Header)
params.h: config.py
	python3 config.py

# 2. Compile C Code (Depends on params.h existing)
simulation: main.c params.h
	$(CC) main.c -o sfpe_solver $(CFLAGS)

# 3. Run FPE Simulation (Depends on executable)
run_fpe: sfpe_solver
	mkdir -p $(DATA_DIR)
	./sfpe_solver

# 4. Run Validation (Depends on FPE data)
validate: run_fpe
	mkdir -p $(PLOT_DIR)
	python3 validation.py

# Cleanup
clean:
	rm -f sfpe_solver params.h
	rm -rf $(DATA_DIR) $(PLOT_DIR)