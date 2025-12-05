
SHELL := /bin/bash

# Makefile for MPI vector_add_mpi

# MPI compiler
CC := mpicc

# Compiler flags
CFLAGS := -O3 -Wall

# Target executable
TARGET := vector_add_mpi

# Source file
SRC := vector_add_mpi.c

# Default mpirun (override with MPIRUN=... on make cmdline)
MPIRUN ?= mpirun

.PHONY: all clean run test help

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)
	@echo "Compilation successful!"
	@echo "Run with: $(MPIRUN) -np <procs> ./$(TARGET)"

clean:
	@rm -f $(TARGET)
	@echo "Cleaned up!"

# Interactive run: prompts for PROCS (or use PROCS= on command line)
run:
	@if [ ! -x ./$(TARGET) ]; then \
	  echo "Executable not found. Building..."; \
	  $(MAKE) $(TARGET); \
	fi; \
	if [ -n "$(PROCS)" ]; then \
	  procs="$(PROCS)"; \
	else \
	  read -p "Enter number of processes [1, 2, 4, 8]: " procs; \
	  procs=$${procs:-4}; \
	fi; \
	echo "Running: $(MPIRUN) -np $$procs ./$(TARGET)"; \
	$(MPIRUN) -np $$procs ./$(TARGET)

# Quick non-interactive tests (1,2,4,8 procs)
test:
	@echo "Testing with 1 process..."
	$(MPIRUN) -np 1 ./$(TARGET) || true
	@echo ""
	@echo "Testing with 2 processes..."
	$(MPIRUN) -np 2 ./$(TARGET) || true
	@echo ""
	@echo "Testing with 4 processes..."
	$(MPIRUN) -np 4 ./$(TARGET) || true
	@echo ""
	@echo "Testing with 8 processes..."
	$(MPIRUN) -np 8 ./$(TARGET) || true

	@echo "Testing with 16 processes..."
	$(MPIRUN) -np 16 ./$(TARGET) || true


eval:
	@echo "=== Performance Evaluation ==="
	@echo "1 process:"
	$(MPIRUN) -np 1 ./$(TARGET) || true
	@echo ""
	@echo "2 processes:"
	$(MPIRUN) -np 2 ./$(TARGET) || true
	@echo ""
	@echo "4 processes:"
	$(MPIRUN) -np 4 ./$(TARGET) || true
	@echo ""
	@echo "8 processes:"
	$(MPIRUN) -np 8 ./$(TARGET) || true
	@echo ""
	@echo "16 processes:"
	$(MPIRUN) -np 16 ./$(TARGET) || true

help:
	@echo "Available targets:"
	@echo "  make          - Compile the program"
	@echo "  make clean    - Remove executable"
	@echo "  make run      - Interactive run (prompts for PROCS) or: make run PROCS=4"
	@echo "  make test     - Quick tests with 1,2,4,8 processes"
