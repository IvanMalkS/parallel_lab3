#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <time.h>

#define N 3
#define TERMS 50000000

double** create_matrix(int n) {
    double** matrix = (double**)malloc(n * sizeof(double*));
    if (matrix == NULL) {
        perror("Failed to allocate memory for matrix rows");
        return NULL;
    }
    matrix[0] = (double*)malloc((long long)n * n * sizeof(double));
    if (matrix[0] == NULL) {
        perror("Failed to allocate memory for matrix data");
        free(matrix);
        return NULL;
    }
    for (int i = 1; i < n; i++) {
        matrix[i] = matrix[0] + (long long)i * n;
    }
    return matrix;
}

void free_matrix(double** matrix, int n) {
    if (matrix != NULL) {
        if (matrix[0] != NULL) {
            free(matrix[0]);
        }
        free(matrix);
    }
}


void print_matrix(const char* name, double** matrix, int n) {
    printf("Matrix: %s (%dx%d)\n", name, n, n);
    for (int i = 0; i < n; i++) {
        printf("  [");
        for (int j = 0; j < n; j++) {
            printf("%8.4f%s", matrix[i][j], (j == n - 1) ? "" : " ");
        }
        printf(" ]\n");
    }
    printf("\n");
}


void init_identity(double** matrix, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }
}

void zero_matrix(double** matrix, int n) {
     for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i][j] = 0.0;
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

void matrix_scale(double** A, double scalar, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i][j] *= scalar;
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double** A           = create_matrix(N);
    double** term        = create_matrix(N);
    double** next_term   = create_matrix(N);
    double** temp_mult   = create_matrix(N);
    double** result      = create_matrix(N);
    double** partial_sum = create_matrix(N);

    if (!A || !term || !next_term || !temp_mult || !result || !partial_sum) {
         fprintf(stderr, "[Rank %d] Ошибка выделения памяти для матриц!\n", rank);
         MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        printf("Matrix size: %dx%d\n", N, N);
        printf("Number of terms: %d (+ Identity)\n", TERMS);
        printf("Number of processes: %d\n", size);

        // Фиксированная инициализация матрицы A
        A[0][0] = 0.1; A[0][1] = 0.4; A[0][2] = 0.2;
        A[1][0] = 0.3; A[1][1] = 0.0; A[1][2] = 0.5;
        A[2][0] = 0.6; A[2][1] = 0.2; A[2][2] = 0.1;

        print_matrix("A (Initial)", A, N);
        printf("Starting calculation...\n");
    }

    MPI_Bcast(A[0], N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    zero_matrix(partial_sum, N);
    init_identity(term, N);

    if (0 % size == rank) {
        matrix_add(partial_sum, term, partial_sum, N);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();


    for (int k = 1; k <= TERMS; k++) {
        matrix_multiply(term, A, temp_mult, N);
        matrix_copy(temp_mult, next_term, N);
        matrix_scale(next_term, 1.0 / (double)k, N);
        matrix_copy(next_term, term, N);

        if (k % size == rank) {
             matrix_add(partial_sum, term, partial_sum, N);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();

    if (rank == 0) {
        zero_matrix(result, N);
    }

    MPI_Reduce(partial_sum[0], result[0], N * N, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double total_time = end_time - start_time;
        printf("Calculation finished.\n");
        printf("Execution time: %.6f seconds\n", total_time);

        print_matrix("e^A (Result)", result, N);

        FILE* fp = fopen("timings.txt", "a");
        if (fp != NULL) {
            fprintf(fp, "%d %.6f\n", size, total_time);
            fclose(fp);
        } else {
            perror("Failed to open timings.txt for appending");
        }
    }

    free_matrix(A, N);
    free_matrix(term, N);
    free_matrix(next_term, N);
    free_matrix(temp_mult, N);
    free_matrix(result, N);
    free_matrix(partial_sum, N);

    MPI_Finalize();
    return 0;
}