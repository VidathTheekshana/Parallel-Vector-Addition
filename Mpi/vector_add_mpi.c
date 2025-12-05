#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define SIZE 10000000

int main(int argc, char *argv[]) {
    int *A = NULL, *B = NULL, *C = NULL;
    int *local_A, *local_B, *local_C;
    int rank, size, local_size;
    double start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    local_size = SIZE / size;

    if (rank == 0) {
        printf("=== MPI VECTOR ADDITION ===\n");
        printf("Processes: %d, Size: %d\n", size, SIZE);
        
        A = (int*)malloc(SIZE * sizeof(int));
        B = (int*)malloc(SIZE * sizeof(int));
        C = (int*)malloc(SIZE * sizeof(int));
        
        srand(12345);
        for (int i = 0; i < SIZE; i++) {
            A[i] = rand() % 100;
            B[i] = rand() % 100;
        }
    }

    local_A = (int*)malloc(local_size * sizeof(int));
    local_B = (int*)malloc(local_size * sizeof(int));
    local_C = (int*)malloc(local_size * sizeof(int));

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    MPI_Scatter(A, local_size, MPI_INT, local_A, local_size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(B, local_size, MPI_INT, local_B, local_size, MPI_INT, 0, MPI_COMM_WORLD);

    // SIMPLE computation - just addition (same as others)
    for (int i = 0; i < local_size; i++) {
        local_C[i] = local_A[i] + local_B[i];
    }

    MPI_Gather(local_C, local_size, MPI_INT, C, local_size, MPI_INT, 0, MPI_COMM_WORLD);

    end_time = MPI_Wtime();

    if (rank == 0) {
        printf("Time: %.3f seconds\n", end_time - start_time);
        
        printf("First 5 results: ");
        for (int i = 0; i < 5; i++) {
            printf("%d ", C[i]);
        }
        printf("\n");
        
        free(A);
        free(B);
        free(C);
    }

    free(local_A);
    free(local_B);
    free(local_C);
    MPI_Finalize();
    return 0;
}