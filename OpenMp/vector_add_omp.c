#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define SIZE 10000000

int main(int argc, char *argv[]) {
    int num_threads = 4;
    if (argc > 1) num_threads = atoi(argv[1]);
    
    printf("=== OPENMP VECTOR ADDITION ===\n");
    printf("Threads: %d, Size: %d\n", num_threads, SIZE);
    
    int *A = (int*)malloc(SIZE * sizeof(int));
    int *B = (int*)malloc(SIZE * sizeof(int));
    int *C = (int*)malloc(SIZE * sizeof(int));
    
    // Initialize vectors
    srand(12345);
    for (int i = 0; i < SIZE; i++) {
        A[i] = rand() % 100;
        B[i] = rand() % 100;
    }
    
    omp_set_num_threads(num_threads);
    
    double start = omp_get_wtime();
    
    #pragma omp parallel for
    for (int i = 0; i < SIZE; i++) {
        C[i] = A[i] + B[i];
    }
    
    double end = omp_get_wtime();
    printf("Time: %.3f seconds\n", end - start);
    
    // Verification
    printf("First 5 results: ");
    for (int i = 0; i < 5; i++) {
        printf("%d ", C[i]);
    }
    printf("\n");
    
    free(A);
    free(B);
    free(C);
    return 0;
}