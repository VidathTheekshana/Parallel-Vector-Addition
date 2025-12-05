
#!/bin/bash

echo "=============================================="
echo "RUNNING MPI TESTS FOR PROCESSES: 1, 2, 4, 8, 16"
echo "=============================================="
echo ""

# Check if MPI file exists
if [ ! -f "vector_add_mpi.c" ]; then
    echo "ERROR: vector_add_mpi.c not found!"
    exit 1
fi

# Compile MPI program
echo "Compiling vector_add_mpi.c..."
echo "------------------------------"
mpicc vector_add_mpi.c -o mpi_test

if [ $? -ne 0 ]; then
    echo "MPI compilation failed!"
    exit 1
fi

echo "MPI compilation successful!"
echo ""

# Create output directory
mkdir -p mpi_outputs

echo "Starting MPI tests..."
echo "====================="
echo ""

# Test 1: 1 process
echo "TEST 1: 1 process"
echo "Command: mpirun -np 1 ./mpi_test"
echo "---------------------------------"
mpirun -np 1 ./mpi_test | tee mpi_outputs/1_process.txt
echo ""
echo "Output saved to: mpi_outputs/1_process.txt"
echo ""

# Test 2: 2 processes
echo "TEST 2: 2 processes"
echo "Command: mpirun -np 2 ./mpi_test"
echo "----------------------------------"
mpirun -np 2 ./mpi_test | tee mpi_outputs/2_processes.txt
echo ""
echo "Output saved to: mpi_outputs/2_processes.txt"
echo ""

# Test 3: 4 processes
echo "TEST 3: 4 processes"
echo "Command: mpirun -np 4 ./mpi_test"
echo "----------------------------------"
mpirun -np 4 ./mpi_test | tee mpi_outputs/4_processes.txt
echo ""
echo "Output saved to: mpi_outputs/4_processes.txt"
echo ""

# Test 4: 8 processes
echo "TEST 4: 8 processes"
echo "Command: mpirun -np 8 ./mpi_test"
echo "----------------------------------"
mpirun -np 8 ./mpi_test | tee mpi_outputs/8_processes.txt
echo ""
echo "Output saved to: mpi_outputs/8_processes.txt"
echo ""

# Test 5: 16 processes
echo "TEST 5: 16 processes"
echo "Command: mpirun -np 16 ./mpi_test"
echo "-----------------------------------"
mpirun -np 16 ./mpi_test | tee mpi_outputs/16_processes.txt
echo ""
echo "Output saved to: mpi_outputs/16_processes.txt"
echo ""

echo "=============================================="
echo "ALL MPI TESTS COMPLETED!"
echo "=============================================="
echo ""
echo "Output files created in mpi_outputs/:"
ls -la mpi_outputs/
