#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SIZE 10000000

void vectorAddition(int *A, int *B, int *C, int size) {
    for (int i = 0; i < size; i++) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    printf("=== SERIAL VECTOR ADDITION ===\n");
    printf("Vector Size: %d elements\n", SIZE);
    
    int *A = (int*)malloc(SIZE * sizeof(int));
    int *B = (int*)malloc(SIZE * sizeof(int));
    int *C = (int*)malloc(SIZE * sizeof(int));
    
    // Initialize vectors (use fixed seed for reproducible results)
    srand(12345);
    for (int i = 0; i < SIZE; i++) {
        A[i] = rand() % 100;
        B[i] = rand() % 100;
    }
    
    clock_t start = clock();
    vectorAddition(A, B, C, SIZE);
    clock_t end = clock();
    
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %.3f seconds\n", time_taken);
    
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