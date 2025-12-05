
SHELL := /bin/bash

# OpenMP Makefile for vector_add_omp

# Compiler
CC = gcc

# Compiler flags
CFLAGS = -O3 -Wall -fopenmp

# Target executable
TARGET = vector_add_omp

# Source file
SRC = vector_add_omp.c

# Default target
all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)
	@echo "Compilation successful!"
	@echo "Run with: ./$(TARGET) <num_threads>"

# Clean target
clean:
	@rm -f $(TARGET)
	@echo "Cleaned up!"

# Run with THREADS (default 4)
run: $(TARGET)
	@# If user passed THREADS on the make command line, use it; otherwise prompt
	@if [ -n "$(THREADS)" ]; then \
	  threads="$(THREADS)"; \
	else \
	  read -p "Enter number of threads [1,2,4,8]: " threads; \
	  threads=$${threads:-4}; \
	fi; \
	echo "Running with $$threads threads..."; \
	./$(TARGET) $$threads

# Test with different thread counts

test: $(TARGET)
	@echo "Testing with 1 thread..."
	./$(TARGET) 1 || true
	@echo ""
	@echo "Testing with 2 threads..."
	./$(TARGET) 2 || true
	@echo ""
	@echo "Testing with 4 threads..."
	./$(TARGET) 4 || true
	@echo ""
	@echo "Testing with 8 threads..."
	./$(TARGET) 8 || true

	@echo "Testing with 16 threads..."
	./$(TARGET) 16 || true


eval: $(TARGET)
	@echo "=== Performance Evaluation ==="
	@echo "1 thread:"
	./$(TARGET) 1 || true
	@echo ""
	@echo "2 threads:"
	./$(TARGET) 2 || true
	@echo ""
	@echo "4 threads:"
	./$(TARGET) 4 || true
	@echo ""
	@echo "8 threads:"
	./$(TARGET) 8 || true
	@echo ""
	@echo "16 threads:"
	./$(TARGET) 16 || true

# Help
help:
	@echo "Available targets:"
	@echo "  make          - Compile the program"
	@echo "  make clean    - Remove executable"
	@echo "  make run      - Run with chosen number of threads (prompt or THREADS=)"
	@echo "  make test     - Test with 1,2,4,8 threads"
	@echo "  make help     - Show this help"
	@echo "Note: On macOS the system clang may require libomp; see README for clang+libomp flags."
