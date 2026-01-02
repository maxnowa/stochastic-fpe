# Compiler Settings
CC = gcc
CFLAGS = -O2 -Wall

# File Names
TARGET = heat_solver
SRC = heat_equation.c
DATA = diffusion_drift_data.csv
PLOT_SCRIPT = plot_results.py  # Change this to your actual Python script name

# --- Rules ---

# Default target: compile, run, and plot
all: $(TARGET) run plot

# 1. Compile the C code
$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)

# 2. Run the simulation
run: $(TARGET)
	@echo "Running simulation..."
	./$(TARGET)

# 3. Plot the results
plot:
	@echo "Plotting results..."
	python3 $(PLOT_SCRIPT)

# Helper to clean up files
clean:
	rm -f $(TARGET) $(DATA)