#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <time.h>

#define N 100
#define TERMS 10000

double** create_matrix(int n) {
    double** matrix = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) {
        matrix[i] = (double*)malloc(n * sizeof(double));
    }
    return matrix;
}

void free_matrix(double** matrix, int n) {
    for (int i = 0; i < n; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

void init_matrix(double** matrix, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i][j] = (double)rand() / RAND_MAX * 2.0 - 1.0;
        }
    }
}

void init_identity(double** matrix, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }
}

void matrix_multiply(double** A, double** B, double** C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = 0.0;
            for (int k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void matrix_add(double** A, double** B, double** C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
}

void matrix_copy(double** A, double** B, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            B[i][j] = A[i][j];
        }
    }
}

double factorial(int n) {
    double result = 1.0;
    for (int i = 2; i <= n; i++) {
        result *= i;
    }
    return result;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double** A = create_matrix(N);
    double** term = create_matrix(N);
    double** temp = create_matrix(N);
    double** result = create_matrix(N);
    double** partial_sum = create_matrix(N);

    if (rank == 0) {
        srand(time(NULL));
        init_matrix(A, N);
    }

    for (int i = 0; i < N; i++) {
        MPI_Bcast(A[i], N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    init_identity(result, N);

    if (rank == 0) {
        matrix_copy(A, term, N);
    }

    for (int i = 0; i < N; i++) {
        MPI_Bcast(term[i], N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    double start_time = MPI_Wtime();

    for (int k = 1; k <= TERMS; k++) {
        if (k % size == rank) {
            matrix_multiply(term, A, temp, N);
            matrix_copy(temp, term, N);

            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    term[i][j] /= k;
                }
            }

            matrix_add(partial_sum, term, temp, N);
            matrix_copy(temp, partial_sum, N);
        }
    }

    for (int i = 0; i < N; i++) {
        MPI_Reduce(partial_sum[i], result[i], N, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        for (int i = 0; i < N; i++) {
            result[i][i] += 1.0;
        }
    }

    double end_time = MPI_Wtime();

    if (rank == 0) {
        printf("Matrix size: %dx%d\n", N, N);
        printf("Number of terms: %d\n", TERMS);
        printf("Number of processes: %d\n", size);
        printf("Execution time: %.6f seconds\n", end_time - start_time);

        FILE* fp = fopen("timings.txt", "a");
        fprintf(fp, "%d %.6f\n", size, end_time - start_time);
        fclose(fp);
    }

    free_matrix(A, N);
    free_matrix(term, N);
    free_matrix(temp, N);
    free_matrix(result, N);
    free_matrix(partial_sum, N);

    MPI_Finalize();
    return 0;
}