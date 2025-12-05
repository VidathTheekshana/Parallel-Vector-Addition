
#!/bin/bash

echo "=============================================="
echo "RUNNING OPENMP TESTS FOR THREADS: 1, 2, 4, 8, 16"
echo "=============================================="
echo ""

# Check if file exists
if [ ! -f "vector_add_omp.c" ]; then
    echo "ERROR: vector_add_omp.c not found!"
    exit 1
fi

# Compile
echo "Compiling vector_add_omp.c..."
echo "------------------------------"
clang -Xpreprocessor -fopenmp \
      -I/opt/homebrew/opt/libomp/include \
      -L/opt/homebrew/opt/libomp/lib \
      -lomp vector_add_omp.c -o vector_add_omp -lm

if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

echo "Compilation successful!"
echo ""

# Create output directory
mkdir -p openmp_outputs

echo "Starting tests..."
echo "================="
echo ""

# Test 1: 1 thread
echo "TEST 1: 1 thread"
echo "Command: OMP_NUM_THREADS=1 ./vector_add_omp 1"
echo "----------------------------------------------"
OMP_NUM_THREADS=1 ./vector_add_omp 1 | tee openmp_outputs/1_thread.txt
echo ""
echo "Output saved to: openmp_outputs/1_thread.txt"
echo ""

# Test 2: 2 threads  
echo "TEST 2: 2 threads"
echo "Command: OMP_NUM_THREADS=2 ./vector_add_omp 2"
echo "----------------------------------------------"
OMP_NUM_THREADS=2 ./vector_add_omp 2 | tee openmp_outputs/2_threads.txt
echo ""
echo "Output saved to: openmp_outputs/2_threads.txt"
echo ""

# Test 3: 4 threads
echo "TEST 3: 4 threads"
echo "Command: OMP_NUM_THREADS=4 ./vector_add_omp 4"
echo "----------------------------------------------"
OMP_NUM_THREADS=4 ./vector_add_omp 4 | tee openmp_outputs/4_threads.txt
echo ""
echo "Output saved to: openmp_outputs/4_threads.txt"
echo ""

# Test 4: 8 threads
echo "TEST 4: 8 threads"
echo "Command: OMP_NUM_THREADS=8 ./vector_add_omp 8"
echo "----------------------------------------------"
OMP_NUM_THREADS=8 ./vector_add_omp 8 | tee openmp_outputs/8_threads.txt
echo ""
echo "Output saved to: openmp_outputs/8_threads.txt"
echo ""

# Test 5: 16 threads
echo "TEST 5: 16 threads"
echo "Command: OMP_NUM_THREADS=16 ./vector_add_omp 16"
echo "------------------------------------------------"
OMP_NUM_THREADS=16 ./vector_add_omp 16 | tee openmp_outputs/16_threads.txt
echo ""
echo "Output saved to: openmp_outputs/16_threads.txt"
echo ""

echo "=============================================="
echo "ALL TESTS COMPLETED!"
echo "=============================================="
echo ""
echo "Output files created in openmp_outputs/:"
ls -la openmp_outputs/
echo ""
echo "To view any output:"
echo "  cat openmp_outputs/1_thread.txt"
echo "  cat openmp_outputs/2_threads.txt"
echo "  cat openmp_outputs/4_threads.txt"
echo "  cat openmp_outputs/8_threads.txt"
echo "  cat openmp_outputs/16_threads.txt"
