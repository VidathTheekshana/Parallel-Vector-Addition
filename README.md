# 🚀 Parallel Vector Addition Suite

A comparative analysis of vector addition (C[i] = A[i] + B[i]) across multiple execution models: Serial, OpenMP, MPI and CUDA. This repository contains implementations, Makefiles, test scripts, and analysis utilities to help you reproduce and compare results locally (macOS) and on GPU (Google Colab).


## Repository layout (important files)

```
source_codes/
├─ Serial/
│  ├─ vector_add_serial.c
│  └─ Makefile                # build/run/test for serial
│
├─ OpenMp/
│  ├─ vector_add_omp.c
│  └─ Makefile                # build/run/test/eval for OpenMP (THREADS variable)
│
├─ Mpi/
│  ├─ vector_add_mpi.c
│  └─ Makefile                # build/run/test/eval for MPI (PROCS variable)
│
├─ Cuda/
│  ├─ vector_add_cuda.cu
│  └─ (Colab notebook + scripts)
│
├─ comparative_analysis/
│  └─ complete_comparative_analysis.py
│
└─ README.md
```

## Quick start — Makefile targets

Each of the three implementations under `source_codes` contains a Makefile with common targets:

- `make` or `make all` — compile the binary
- `make run` — run the program (prompts for thread/process count unless provided)
- `make test` — quick tests over common thread/process counts (1,2,4,8,16)
- `make eval` — performance evaluation runs (heavier input)
- `make clean` — remove binaries
- `make help` — show usage/help

Below are copy‑paste commands you can run in zsh on macOS.

### Serial
```bash
cd /Users/vidaththeekshana/Desktop/vector_addition/source_codes/Serial
# build
make

# run (binary will be built if missing)
make run

# quick test
make test

# clean
make clean
```

### OpenMP
The OpenMP Makefile accepts a `THREADS` variable. The program reads the thread count from argv and/or `omp_set_num_threads`.

```bash
cd /Users/vidaththeekshana/Desktop/vector_addition/source_codes/OpenMp
# build
make

# run with 4 threads
make run THREADS=4

# run quick tests (1,2,4,8,16)
make test

# run performance evaluation
make eval

# clean
make clean
```

If you prefer to invoke the binary directly (it accepts the thread count as a single argument):
```bash
./vector_add_omp 4
```

macOS notes: the system clang may not support `-fopenmp` by default. If compilation fails with `-fopenmp` or `omp.h` errors:

```bash
# Install libomp or GCC via Homebrew
brew install libomp
brew install gcc

# Then build with Homebrew gcc (example)
make CC=gcc-13

# Or compile manually with clang+libomp flags
clang -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include \
      -L/opt/homebrew/opt/libomp/lib -lomp -o vector_add_omp vector_add_omp.c
```

### MPI
The MPI Makefile uses `mpicc` and `mpirun` by default. Set `PROCS` to control process count or pass `MPIRUN=mpiexec` if your environment uses mpiexec.

```bash
cd /Users/vidaththeekshana/Desktop/vector_addition/source_codes/Mpi
# build
make

# run with 4 processes
make run PROCS=4

# or use mpirun directly
mpirun -np 4 ./vector_add_mpi

# quick test (1,2,4,8,16)
make test

# eval (heavier runs)
make eval

# clean
make clean
```

If you get "not enough slots" for high process counts (e.g., 16), your local machine likely doesn't have that many slots/cores. To oversubscribe on a single machine:

```bash
mpirun -np 16 --map-by :OVERSUBSCRIBE ./vector_add_mpi
# or
mpirun --use-hwthread-cpus -np 16 ./vector_add_mpi
```


