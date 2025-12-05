%%writefile compute_intensive.cu

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

#define SIZE 10000000  // 10 million elements
#define ITERATIONS 100  // Number of compute iterations

__global__ void computeIntensive(int *A, int *B, int *C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // Simple computation
        float temp = A[i];
        
        // Multiple iterations to make it compute-intensive
        for (int iter = 0; iter < ITERATIONS; iter++) {
            temp = sin(temp) * cos(temp) + sqrt(fabs(temp));
        }
        
        C[i] = (int)temp + B[i];
    }
}

__global__ void vectorAddition(int *A, int *B, int *C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <threads_per_block>\n", argv[0]);
        return 1;
    }
    
    int threads_per_block = atoi(argv[1]);

    printf("=== COLAB CUDA COMPUTE INTENSIVE ===\n");
    
    // Get GPU info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Threads/Block: %d, Size: %d, Iterations: %d\n", 
           threads_per_block, SIZE, ITERATIONS);

    int *A, *B, *C;
    int *d_A, *d_B, *d_C;
    size_t bytes = SIZE * sizeof(int);

    // Allocate host memory
    A = (int*)malloc(bytes);
    B = (int*)malloc(bytes);
    C = (int*)malloc(bytes);

    if (A == NULL || B == NULL || C == NULL) {
        printf("Host memory allocation failed!\n");
        return 1;
    }

    // Initialize with values
    printf("Initializing vectors...\n");
    for (int i = 0; i < SIZE; i++) {
        A[i] = i % 1000;
        B[i] = 50;
    }

    // Allocate device memory
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // Copy data to device
    cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice);

    // Calculate grid size
    int blocks = (SIZE + threads_per_block - 1) / threads_per_block;
    printf("Grid: %d blocks, %d threads/block\n", blocks, threads_per_block);

    // Warm-up kernel
    printf("Warm-up kernel...\n");
    vectorAddition<<<blocks, threads_per_block>>>(d_A, d_B, d_C, SIZE);
    cudaDeviceSynchronize();

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Time compute-intensive kernel
    printf("Launching compute-intensive kernel...\n");
    
    cudaEventRecord(start, 0);
    computeIntensive<<<blocks, threads_per_block>>>(d_A, d_B, d_C, SIZE);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    double time_taken = milliseconds / 1000.0;

    // Check for kernel errors
    cudaError_t kernelErr = cudaGetLastError();
    if (kernelErr != cudaSuccess) {
        printf("Kernel failed: %s\n", cudaGetErrorString(kernelErr));
        return 1;
    }

    // Copy result back to host
    cudaMemcpy(C, d_C, bytes, cudaMemcpyDeviceToHost);

    printf("Time: %.6f seconds (%.2f ms)\n", time_taken, milliseconds);

    // Verification (first few elements)
    printf("First 5 results:\n");
    int correct = 1;
    for (int i = 0; i < 5; i++) {
        printf("C[%d] = computed(A[%d]=%d) + %d = %d\n", 
               i, i, A[i], B[i], C[i]);
    }

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(A);
    free(B);
    free(C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    printf("Done.\n");
    return 0;
}